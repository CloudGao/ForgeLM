from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .command import Emit, GetDesiredState

DATA_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".gz",
    ".json",
    ".jsonl",
    ".markdown",
    ".md",
    ".ndjson",
    ".parquet",
    ".text",
    ".tsv",
    ".txt",
    ".zst",
}
DEFAULT_DATA_PATTERNS = [f"*{suffix}" for suffix in sorted(DATA_SUFFIXES)]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
HFD_PROGRESS_RE = re.compile(
    r"\[\s*(?P<percent>\d+)%\]\s+(?P<completed>\d+)/(?P<files>\d+) files\s+\|\s+"
    r"(?P<downloaded>[^/|]+)/(?P<total>[^|]+)\s+\|\s+"
    r"(?P<speed>[^|]+)\s+\|\s+ETA\s+(?P<eta>\S+)"
)


class ProcessInterrupted(RuntimeError):
    def __init__(self, state: str) -> None:
        super().__init__(f"Dataset import interrupted: {state}")
        self.state = state


def huggingface_endpoint(source_config: dict[str, Any]) -> str:
    endpoint = (
        str(
            source_config.get("hf_endpoint") or os.getenv("HF_ENDPOINT") or "https://huggingface.co"
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Hugging Face mirror endpoint must be a complete HTTP(S) URL")
    return endpoint


class ScriptTransferRunner:
    """Translate dataset source settings into mature CLI invocations.

    This class intentionally owns no HTTP protocol, retry, range, or repository traversal logic.
    Those behaviors remain the responsibility of curl, hf/hf-xet, hfd/aria2c, and git.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()

    @staticmethod
    def _patterns(value: Any, *, defaults: list[str] | None = None) -> list[str]:
        values = re.split(r"[,，]", value) if isinstance(value, str) else value or []
        result = [str(item).strip() for item in values if str(item).strip()]
        return result or list(defaults or [])

    def _tool_root(self) -> Path:
        from llm_platform_core.storage_layout import unified_storage_root
        root = unified_storage_root()
        if root:
            return root / "platform" / "tools"
        try:
            return self.run_dir.parents[2] / "tools"
        except IndexError:
            return Path(".llm-platform/tools").resolve()

    @staticmethod
    def _resolve_tool(name: str, env_name: str, candidates: list[Path] | None = None) -> str:
        configured = os.getenv(env_name, "").strip()
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return str(path)
            raise FileNotFoundError(f"{env_name} points to a missing executable: {path}")
        for candidate in candidates or []:
            if candidate.is_file():
                return str(candidate.resolve())
        if resolved := shutil.which(name):
            return resolved
        raise FileNotFoundError(
            f"Required data import tool '{name}' is unavailable. "
            f"Run scripts/setup.ps1 or configure {env_name}."
        )

    def _resolve_hf(self) -> str:
        suffix = ".exe" if os.name == "nt" else ""
        return self._resolve_tool(
            "hf", "LLM_PLATFORM_HF_CLI", [Path(sys.executable).parent / f"hf{suffix}"]
        )

    def _resolve_hfd_tools(self) -> tuple[str, str, str]:
        root = self._tool_root()
        bash_candidates = [Path(r"C:\Program Files\Git\bin\bash.exe")] if os.name == "nt" else []
        bash = self._resolve_tool("bash", "LLM_PLATFORM_BASH", bash_candidates)
        hfd = self._resolve_tool("hfd.sh", "LLM_PLATFORM_HFD_SCRIPT", [root / "hfd.sh"])
        aria_candidates = list(root.glob("aria2-*/aria2c.exe")) if os.name == "nt" else []
        aria2c = self._resolve_tool("aria2c", "LLM_PLATFORM_ARIA2C", aria_candidates)
        return bash, hfd, aria2c

    @staticmethod
    def _bash_path(path: Path) -> str:
        resolved = path.resolve()
        if os.name != "nt":
            return resolved.as_posix()
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix().split(":", 1)[-1].lstrip("/")
        return f"/{drive}/{tail}"

    @staticmethod
    def _human_bytes(value: str) -> int | None:
        match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGTPE]?B)\s*", value, re.IGNORECASE)
        if not match:
            return None
        units = {
            "B": 1,
            "KB": 1000,
            "MB": 1000**2,
            "GB": 1000**3,
            "TB": 1000**4,
            "PB": 1000**5,
            "EB": 1000**6,
        }
        return int(float(match.group(1)) * units[match.group(2).upper()])

    @staticmethod
    async def _tool_version(executable: str, *args: str) -> str | None:
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *(args or ("--version",)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, TimeoutError):
            return None
        line = output.decode("utf-8", errors="replace").splitlines()
        return line[0].strip()[:200] if process.returncode == 0 and line else None

    @staticmethod
    def _sha256(path: str) -> str | None:
        digest = hashlib.sha256()
        try:
            with Path(path).open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return None
        return digest.hexdigest()

    @classmethod
    def _progress_from_line(cls, line: str, tool: str) -> dict[str, Any] | None:
        clean = ANSI_ESCAPE_RE.sub("", line).strip()
        match = HFD_PROGRESS_RE.search(clean)
        if match:
            percent = min(int(match.group("percent")), 100)
            return {
                "progress": 0.05 + 0.73 * percent / 100,
                "phase": "downloading",
                "tool": tool,
                "file_count_completed": int(match.group("completed")),
                "file_count_total": int(match.group("files")),
                "downloaded_bytes": cls._human_bytes(match.group("downloaded")),
                "total_bytes": cls._human_bytes(match.group("total")),
                "speed_bps": cls._human_bytes(match.group("speed").removesuffix("/s")),
                "eta": match.group("eta"),
            }
        generic = re.search(r"(?<!\d)(\d{1,3})%(?!\d)", clean)
        if generic:
            percent = min(int(generic.group(1)), 100)
            return {
                "progress": 0.05 + 0.73 * percent / 100,
                "phase": "downloading",
                "tool": tool,
            }
        return None

    async def _run_process(
        self,
        args: list[str],
        *,
        tool: str,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
        env: dict[str, str] | None = None,
        parse_progress: bool = True,
    ) -> None:
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            raise ProcessInterrupted(desired)
        await emit("progress", {"progress": 0.05, "phase": "starting_tool", "tool": tool})
        process = await asyncio.create_subprocess_exec(
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        recent: list[str] = []

        async def pump(stream: asyncio.StreamReader, stream_name: str) -> None:
            buffer = ""
            while chunk := await stream.read(4096):
                buffer += chunk.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]+", buffer)
                buffer = parts.pop()
                for raw_line in parts:
                    await publish(raw_line, stream_name)
            await publish(buffer, stream_name)

        async def publish(raw_line: str, stream_name: str) -> None:
            line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
            if not line:
                return
            recent.append(line)
            del recent[:-20]
            progress = self._progress_from_line(line, tool) if parse_progress else None
            await emit(
                "progress" if progress else "log",
                progress or {"tool": tool, "stream": stream_name, "message": line},
            )

        assert process.stdout is not None and process.stderr is not None
        readers = [
            asyncio.create_task(pump(process.stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, "stderr")),
        ]
        interrupted: str | None = None
        while process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.5)
            except TimeoutError:
                desired = get_desired_state() if get_desired_state else "running"
                if desired in {"paused", "cancelled"}:
                    interrupted = desired
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
        await asyncio.gather(*readers)
        if interrupted:
            raise ProcessInterrupted(interrupted)
        if process.returncode:
            detail = "\n".join(recent[-8:]) or "No diagnostic output was produced."
            raise RuntimeError(f"{tool} exited with code {process.returncode}: {detail}")
        await emit("progress", {"progress": 0.78, "phase": "downloaded", "tool": tool})

    async def download_http(
        self,
        source_uri: str,
        destination: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        parsed = urlparse(source_uri)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("HTTP dataset source must use http:// or https://")
        curl = self._resolve_tool("curl", "LLM_PLATFORM_CURL")
        target = destination / (unquote(Path(parsed.path).name) or "download.bin")
        partial = target.with_name(f"{target.name}.part")
        if target.is_file():
            await emit(
                "progress",
                {
                    "progress": 0.78,
                    "phase": "downloaded",
                    "tool": "curl",
                    "downloaded_bytes": target.stat().st_size,
                    "reused": True,
                },
            )
        else:
            await self._run_process(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--retry-delay",
                    "2",
                    "--connect-timeout",
                    "30",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    "--url",
                    source_uri,
                ],
                tool="curl",
                emit=emit,
                get_desired_state=get_desired_state,
            )
            partial.replace(target)
        return {
            "tool": "curl",
            "executable": Path(curl).name,
            "version": await self._tool_version(curl),
            "resume": True,
            "retries": 5,
        }

    async def download_huggingface(
        self,
        repo_id: str,
        source_config: dict[str, Any],
        destination: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        if not repo_id or "/" not in repo_id:
            raise ValueError("Hugging Face source must be a dataset repository ID such as org/name")
        endpoint = huggingface_endpoint(source_config)
        revision = str(source_config.get("revision") or "main")
        include = self._patterns(
            source_config.get("include_patterns"), defaults=DEFAULT_DATA_PATTERNS
        )
        exclude = self._patterns(source_config.get("exclude_patterns"))
        concurrency = max(1, min(int(source_config.get("max_workers") or 8), 10))
        host = (urlparse(endpoint).hostname or "").lower()
        child_env = os.environ.copy()
        child_env.update({"HF_ENDPOINT": endpoint, "HF_HUB_DISABLE_UPDATE_CHECK": "1"})
        if child_env.get("HUGGING_FACE_HUB_TOKEN") and not child_env.get("HF_TOKEN"):
            child_env["HF_TOKEN"] = child_env["HUGGING_FACE_HUB_TOKEN"]

        if host == "hf-mirror.com" or host.endswith(".hf-mirror.com"):
            bash, hfd, aria2c = self._resolve_hfd_tools()
            child_env["PATH"] = str(Path(aria2c).parent) + os.pathsep + child_env.get("PATH", "")
            args = [
                bash,
                self._bash_path(Path(hfd)),
                repo_id,
                "--dataset",
                "--tool",
                "aria2c",
                "-x",
                str(concurrency),
                "-j",
                str(concurrency),
                "--revision",
                revision,
                "--local-dir",
                self._bash_path(destination),
            ]
            if include:
                args.extend(["--include", *include])
            if exclude:
                args.extend(["--exclude", *exclude])
            tool = "hfd + aria2c"
            version = await self._tool_version(aria2c, "--version")
            tool_metadata = {
                "executable": Path(aria2c).name,
                "version": version,
                "script_sha256": self._sha256(hfd),
            }
        else:
            hf = self._resolve_hf()
            args = [
                hf,
                "download",
                repo_id,
                "--repo-type",
                "dataset",
                "--revision",
                revision,
                "--local-dir",
                str(destination),
                "--max-workers",
                str(concurrency),
            ]
            if include:
                args.extend(["--include", *include])
            if exclude:
                args.extend(["--exclude", *exclude])
            tool = "hf download"
            tool_metadata = {
                "executable": Path(hf).name,
                "version": await self._tool_version(hf, "--version"),
            }
        await self._run_process(
            args, tool=tool, emit=emit, get_desired_state=get_desired_state, env=child_env
        )
        return {
            "tool": tool,
            "endpoint": endpoint,
            "revision": revision,
            "include_patterns": include,
            "exclude_patterns": exclude,
            "concurrency": concurrency,
            **tool_metadata,
        }

    async def clone_git(
        self,
        source_uri: str,
        source_config: dict[str, Any],
        destination: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        if not source_uri:
            raise ValueError("Git repository address is required")
        revision = str(source_config.get("revision") or "").strip()
        frozen_commit = str(source_config.get("commit_hash") or "").strip().lower()
        if frozen_commit and not re.fullmatch(r"[0-9a-f]{40,64}", frozen_commit):
            raise ValueError("Frozen Git commit hash is invalid")
        git = self._resolve_tool("git", "LLM_PLATFORM_GIT")
        shutil.rmtree(destination)
        if frozen_commit:
            await self._run_process(
                [git, "init", str(destination)],
                tool="git init",
                emit=emit,
                get_desired_state=get_desired_state,
            )
            await self._run_process(
                [git, "-C", str(destination), "remote", "add", "origin", source_uri],
                tool="git remote add",
                emit=emit,
                get_desired_state=get_desired_state,
            )
            await self._run_process(
                [
                    git,
                    "-C",
                    str(destination),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    frozen_commit,
                ],
                tool="git fetch",
                emit=emit,
                get_desired_state=get_desired_state,
            )
            await self._run_process(
                [git, "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
                tool="git checkout",
                emit=emit,
                get_desired_state=get_desired_state,
            )
        else:
            args = [git, "clone", "--depth", "1"]
            if revision:
                args.extend(["--branch", revision])
            args.extend([source_uri, str(destination)])
            await self._run_process(
                args, tool="git clone", emit=emit, get_desired_state=get_desired_state
            )
        process = await asyncio.create_subprocess_exec(
            git,
            "-C",
            str(destination),
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git rev-parse failed: {detail}")
        resolved_commit = stdout.decode("ascii", errors="strict").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40,64}", resolved_commit):
            raise RuntimeError("git rev-parse returned an invalid commit hash")
        if frozen_commit and resolved_commit != frozen_commit:
            raise RuntimeError("Git checkout does not match the frozen commit")
        shutil.rmtree(destination / ".git", ignore_errors=True)
        return {
            "tool": "git clone",
            "executable": Path(git).name,
            "version": await self._tool_version(git),
            "revision": revision or "default",
            "commit_hash": resolved_commit,
        }
