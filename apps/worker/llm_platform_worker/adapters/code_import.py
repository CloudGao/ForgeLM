from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

Emit = Callable[[str, dict[str, Any]], Awaitable[str]]
GetDesiredState = Callable[[], str]

DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".llm-platform",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "venv",
}


class CodeImportAdapter:
    """Create an immutable source snapshot without importing code into the Worker process."""

    @staticmethod
    def _local_path(uri: str) -> Path:
        if len(uri) >= 3 and uri[1] == ":" and uri[2] in {"/", "\\"}:
            path = Path(uri).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Code source does not exist: {path}")
            return path
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError("Local code source must use a path or file:// URI")
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Code source does not exist: {path}")
        return path

    @staticmethod
    def _safe_target(root: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("../"):
            raise ValueError(f"Unsafe archive member: {member_name}")
        target = (root / normalized).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member: {member_name}") from exc
        return target

    @classmethod
    def _extract_archive(cls, archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise ValueError(
                            f"Code archive contains a symbolic link: {member.filename}"
                        )
                    target = cls._safe_target(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
            return
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Code archive contains a link: {member.name}")
                    target = cls._safe_target(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is not None:
                            with source, target.open("wb") as output:
                                shutil.copyfileobj(source, output)
            return
        raise ValueError("Code archive must be zip or tar")

    @staticmethod
    def _git_commit(path: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        commit = result.stdout.strip().lower()
        return (
            commit
            if len(commit) == 40 and all(char in "0123456789abcdef" for char in commit)
            else None
        )

    @staticmethod
    def _copy_tree(source: Path, destination: Path, excludes: set[str]) -> None:
        def ignored(_directory: str, names: list[str]) -> set[str]:
            return {name for name in names if name in excludes}

        shutil.copytree(source, destination, ignore=ignored, dirs_exist_ok=True)

    @staticmethod
    def _select_root(root: Path) -> Path:
        children = [child for child in root.iterdir() if child.name not in {"__MACOSX"}]
        return children[0] if len(children) == 1 and children[0].is_dir() else root

    @classmethod
    def _select_subdirectory(cls, root: Path, raw_subdirectory: str | None) -> Path:
        root = cls._select_root(root)
        if not raw_subdirectory:
            return root
        subdirectory = (root / raw_subdirectory).resolve()
        try:
            subdirectory.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("Code source subdirectory escapes the imported root") from exc
        if not subdirectory.is_dir():
            raise FileNotFoundError(f"Code source subdirectory does not exist: {raw_subdirectory}")
        return subdirectory

    async def _download(self, uri: str, target: Path) -> None:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            async with client.stream("GET", uri) as response:
                response.raise_for_status()
                with target.open("wb") as output:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        output.write(chunk)

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        source_type = str(payload.get("source_type") or "")
        source_uri = str(payload.get("source_uri") or "").strip()
        source_config = payload.get("source_config") or {}
        entrypoints = payload.get("entrypoints") or {}
        if source_type not in {"git", "local", "archive"}:
            raise ValueError(f"Unsupported code source type: {source_type}")
        if not source_uri:
            raise ValueError("Code source URI is required")

        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        staging = run_dir / "code-import"
        imported = staging / "imported"
        snapshot = staging / "snapshot"
        if staging.exists():
            shutil.rmtree(staging)
        imported.mkdir(parents=True)
        await emit("progress", {"progress": 0.05, "phase": "fetching_code"})

        git_commit: str | None = None
        if source_type == "git":
            repository = imported / "repository"
            clone = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--no-tags",
                source_uri,
                str(repository),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await clone.communicate()
            if clone.returncode != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"git clone failed: {message}")
            revision = str(source_config.get("revision") or "").strip()
            if revision:
                checkout = await asyncio.create_subprocess_exec(
                    "git",
                    "-C",
                    str(repository),
                    "checkout",
                    "--detach",
                    revision,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _checkout_stdout, checkout_stderr = await checkout.communicate()
                if checkout.returncode != 0:
                    message = checkout_stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"git checkout failed: {message}")
            git_commit = self._git_commit(repository)
            source_root = self._select_subdirectory(
                repository, str(source_config.get("subdirectory") or "") or None
            )
        elif source_type == "local":
            local_source = self._local_path(source_uri)
            git_commit = self._git_commit(
                local_source if local_source.is_dir() else local_source.parent
            )
            if local_source.is_dir():
                source_root = self._select_subdirectory(
                    local_source, str(source_config.get("subdirectory") or "") or None
                )
            else:
                self._extract_archive(local_source, imported)
                source_root = self._select_subdirectory(
                    imported, str(source_config.get("subdirectory") or "") or None
                )
        else:
            parsed = urlparse(source_uri)
            if parsed.scheme in {"http", "https"}:
                archive_path = staging / "downloaded-archive"
                await self._download(source_uri, archive_path)
            else:
                archive_path = self._local_path(source_uri)
            self._extract_archive(archive_path, imported)
            source_root = self._select_subdirectory(
                imported, str(source_config.get("subdirectory") or "") or None
            )

        if get_desired_state and get_desired_state() in {"paused", "cancelled"}:
            return {"controlled_state": get_desired_state()}
        await emit("progress", {"progress": 0.55, "phase": "creating_snapshot"})
        excludes = DEFAULT_EXCLUDES | {
            str(item).strip() for item in (source_config.get("exclude") or []) if str(item).strip()
        }
        self._copy_tree(source_root, snapshot, excludes)
        for kind, relative_path in entrypoints.items():
            entrypoint = (snapshot / str(relative_path)).resolve()
            try:
                entrypoint.relative_to(snapshot.resolve())
            except ValueError as exc:
                raise ValueError(f"Code entrypoint escapes snapshot: {relative_path}") from exc
            if not entrypoint.is_file():
                raise FileNotFoundError(f"Code entrypoint does not exist: {kind}={relative_path}")

        files = [path for path in snapshot.rglob("*") if path.is_file()]
        size_bytes = sum(path.stat().st_size for path in files)
        if not files:
            raise ValueError("Code source produced an empty snapshot")
        source_snapshot = {
            "source_type": source_type,
            "source_uri": source_uri,
            "requested_revision": source_config.get("revision"),
            "git_commit": git_commit,
            "subdirectory": source_config.get("subdirectory"),
            "file_count": len(files),
            "size_bytes": size_bytes,
            "excluded_names": sorted(excludes),
        }
        await emit("progress", {"progress": 0.9, "phase": "uploading_snapshot"})
        return {
            "git_commit": git_commit,
            "source_snapshot": source_snapshot,
            "outputs": [
                {
                    "ref": "source-code",
                    "kind": "source_code",
                    "path": str(snapshot),
                    "media_type": "application/gzip",
                }
            ],
        }
