from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.command import CommandAdapter, Emit, GetDesiredState

_SAFE_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,179}$")
_SAFE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
_CATALOG_RUNTIMES = {
    ("plugin.llm-compressor.quantization", "0.1.0"): {
        "entrypoint": "plugin.llm-compressor.quantization",
        "runner": "llm_compressor",
        "isolation": "venv",
        "requirements": ["llmcompressor==0.10.0"],
        "probe": (
            "import json,llmcompressor;"
            "print(json.dumps({'event':'result','package_version':"
            "getattr(llmcompressor,'__version__','unknown')}))"
        ),
    },
    ("plugin.forgelm.native-exporter", "0.1.0"): {
        "entrypoint": "plugin.forgelm.native-exporter",
        "runner": "forgelm_native_export",
        "isolation": "worker-process",
        "requirements": [],
        "probe": (
            "import json;"
            "print(json.dumps({'event':'result','package_version':'0.1.0'}))"
        ),
    },
    ("plugin.local-directory.publisher", "0.1.0"): {
        "entrypoint": "plugin.local-directory.publisher",
        "runner": "local_directory_publish",
        "isolation": "worker-process",
        "requirements": [],
        "probe": (
            "import json;"
            "print(json.dumps({'event':'result','package_version':'0.1.0'}))"
        ),
    },
}


def _worker_runtime_environment() -> dict[str, str]:
    """Freeze the trusted Worker's import roots for bundled child processes."""
    roots: list[str] = []
    for value in sys.path:
        if not str(value or "").strip():
            continue
        try:
            resolved = Path(value).resolve()
        except OSError:
            continue
        if resolved.is_dir():
            roots.append(str(resolved))
    return {"PYTHONPATH": os.pathsep.join(dict.fromkeys(roots))}


def _validate_identity(plugin_id: str, version: str) -> None:
    if not _SAFE_SEGMENT.fullmatch(plugin_id) or not _SAFE_VERSION.fullmatch(version):
        raise ValueError("Plugin identity contains an unsafe path segment")


def plugin_root_path(configured: str | None, workspace_root: str | None) -> Path:
    if str(configured or "").strip():
        root = Path(str(configured)).expanduser().resolve()
    elif str(workspace_root or "").strip():
        root = (Path(str(workspace_root)).expanduser().resolve() / ".plugins").resolve()
    else:
        root = (Path.cwd() / ".llm-platform" / "worker-plugins").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def installation_directory(root: Path, plugin_id: str, version: str) -> Path:
    _validate_identity(plugin_id, version)
    target = (root.resolve() / plugin_id / version).resolve()
    target.relative_to(root.resolve())
    return target


def venv_python(installation_dir: Path) -> Path:
    if os.name == "nt":
        return installation_dir / "venv" / "Scripts" / "python.exe"
    return installation_dir / "venv" / "bin" / "python"


def read_installation_manifest(root: Path, plugin_id: str, version: str) -> dict[str, Any]:
    manifest_path = installation_directory(root, plugin_id, version) / "installation.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Plugin {plugin_id}@{version} is not installed")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Plugin installation manifest must be an object")
    if value.get("plugin_id") != plugin_id or value.get("version") != version:
        raise ValueError("Plugin installation manifest identity mismatch")
    python = Path(str(value.get("python") or "")).resolve()
    catalog = _CATALOG_RUNTIMES.get((plugin_id, version))
    expected = (
        Path(sys.executable).resolve()
        if catalog and catalog["isolation"] == "worker-process"
        else venv_python(manifest_path.parent).resolve()
    )
    if python != expected or not python.is_file():
        raise ValueError("Plugin installation manifest references an invalid Python runtime")
    if (
        catalog is None
        or value.get("entrypoint") != catalog["entrypoint"]
        or value.get("runner") != catalog["runner"]
        or value.get("isolation") != catalog["isolation"]
        or value.get("requirements") != catalog["requirements"]
    ):
        raise ValueError("Plugin installation manifest is not in the Worker catalog allowlist")
    return value


def discover_plugin_capabilities(root: Path) -> list[str]:
    capabilities: list[str] = []
    if not root.is_dir():
        return capabilities
    for manifest_path in root.glob("*/*/installation.json"):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            plugin_id = str(value.get("plugin_id") or "")
            version = str(value.get("version") or "")
            entrypoint = str(value.get("entrypoint") or "")
            expected = installation_directory(root, plugin_id, version) / "installation.json"
            if manifest_path.resolve() != expected.resolve():
                continue
            read_installation_manifest(root, plugin_id, version)
            catalog = _CATALOG_RUNTIMES.get((plugin_id, version))
            if catalog is None or catalog["entrypoint"] != entrypoint:
                continue
            capabilities.extend([entrypoint, f"{entrypoint}@{version}"])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return list(dict.fromkeys(capabilities))


class PluginManagerAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.command = CommandAdapter()

    async def _command(
        self,
        original_task: dict[str, Any],
        executable: str,
        args: list[str],
        emit: Emit,
        get_desired_state: GetDesiredState | None,
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        task = {
            **original_task,
            "payload": {
                **(original_task.get("payload") or {}),
                "command": {
                    "executable": executable,
                    "args": args,
                    "env": environment or {},
                },
            },
        }
        return await self.command.run(task, emit, get_desired_state)

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        plugin = (task.get("payload") or {}).get("plugin") or {}
        plugin_id = str(plugin.get("id") or "")
        version = str(plugin.get("version") or "")
        entrypoint = str(plugin.get("entrypoint") or "")
        runtime = plugin.get("runtime") or {}
        _validate_identity(plugin_id, version)
        catalog = _CATALOG_RUNTIMES.get((plugin_id, version))
        if (
            catalog is None
            or plugin.get("source") != {"type": "builtin-catalog"}
            or entrypoint != catalog["entrypoint"]
            or runtime.get("runner") != catalog["runner"]
            or runtime.get("requirements") != catalog["requirements"]
            or runtime.get("isolation") != catalog["isolation"]
        ):
            raise ValueError("Plugin installation payload is not in the Worker catalog allowlist")

        target = installation_directory(self.root, plugin_id, version)
        if target.is_dir():
            manifest = read_installation_manifest(self.root, plugin_id, version)
            await emit(
                "log",
                {
                    "stream": "platform",
                    "level": "info",
                    "message": f"插件 {plugin_id}@{version} 已安装，完成一致性校验",
                },
            )
            return {"installation": manifest, "already_installed": True}

        plugin_dir = target.parent
        plugin_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=plugin_dir)).resolve()
        staging.relative_to(plugin_dir.resolve())
        try:
            if catalog["isolation"] == "worker-process":
                await emit(
                    "progress",
                    {"phase": "validating_worker_runtime", "progress": 0.1},
                )
                # Keep the venv launcher path intact.  On POSIX a venv's Python is
                # commonly a symlink to the system interpreter; resolving it would
                # drop the venv prefix and make installed packages unavailable.
                python = Path(sys.executable)
            else:
                await emit(
                    "progress",
                    {"phase": "creating_plugin_environment", "progress": 0.05},
                )
                control = await self._command(
                    task,
                    sys.executable,
                    ["-m", "venv", str(staging / "venv")],
                    emit,
                    get_desired_state,
                )
                if control.get("controlled_state"):
                    return control
                python = venv_python(staging)
            if catalog["requirements"]:
                await emit(
                    "progress",
                    {"phase": "installing_plugin_dependencies", "progress": 0.2},
                )
                control = await self._command(
                    task,
                    str(python),
                    [
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        *catalog["requirements"],
                    ],
                    emit,
                    get_desired_state,
                )
                if control.get("controlled_state"):
                    return control
            probe = await self._command(
                task,
                str(python),
                [
                    "-c",
                    catalog["probe"],
                ],
                emit,
                get_desired_state,
                (
                    _worker_runtime_environment()
                    if catalog["isolation"] == "worker-process"
                    else None
                ),
            )
            manifest = {
                "schema_version": 1,
                "plugin_id": plugin_id,
                "version": version,
                "entrypoint": entrypoint,
                "runner": catalog["runner"],
                "isolation": catalog["isolation"],
                "requirements": list(catalog["requirements"]),
                "package_version": probe.get("package_version"),
                "runtime_versions": probe.get("runtime_versions") or {},
                "python": str(
                    Path(sys.executable)
                    if catalog["isolation"] == "worker-process"
                    else venv_python(target)
                ),
                "installed_at": datetime.now(UTC).isoformat(),
            }
            (staging / "installation.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            staging.replace(target)
            manifest["manifest_path"] = str(target / "installation.json")
            persisted = {**manifest, "manifest_path": manifest["manifest_path"]}
            (target / "installation.json").write_text(
                json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await emit(
                "progress", {"phase": "plugin_installation_ready", "progress": 1.0}
            )
            return {"installation": persisted}
        finally:
            if staging.exists():
                staging.relative_to(plugin_dir.resolve())
                shutil.rmtree(staging, ignore_errors=True)


class InstalledPluginAdapter:
    def __init__(self, root: Path, release_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.release_root = release_root.resolve() if release_root is not None else None
        self.command = CommandAdapter()

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        plugin = payload.get("plugin") or {}
        plugin_id = str(plugin.get("id") or "")
        version = str(plugin.get("version") or "")
        entrypoint = str(plugin.get("entrypoint") or "")
        catalog = _CATALOG_RUNTIMES.get((plugin_id, version))
        if catalog is None or entrypoint != catalog["entrypoint"]:
            raise ValueError("Task requests an unsupported installed plugin")
        manifest = read_installation_manifest(self.root, plugin_id, version)
        if manifest.get("entrypoint") != entrypoint:
            raise ValueError("Installed plugin manifest does not match the task entrypoint")
        runner = Path(__file__).resolve().parent / "plugin_runners" / f"{catalog['runner']}.py"
        if not runner.is_file():
            raise FileNotFoundError("Bundled plugin runner is missing")
        command = dict(payload.get("command") or {})
        environment = dict(command.get("env") or {})
        if catalog["isolation"] == "worker-process":
            environment.update(_worker_runtime_environment())
        if catalog["runner"] == "local_directory_publish":
            if self.release_root is None:
                raise ValueError("Local publisher requires an administrator-configured release root")
            environment["LLM_PLATFORM_RELEASE_ROOT"] = str(self.release_root)
        command.update(
            {
                "executable": str(manifest["python"]),
                "args": [str(runner)],
                "cwd": str((task.get("runtime_context") or {}).get("attempt_dir") or ""),
                "env": environment,
            }
        )
        payload["command"] = command
        return await self.command.run(task, emit, get_desired_state)
