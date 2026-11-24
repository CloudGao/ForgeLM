from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_platform_core.storage_layout import platform_state_root

from ..config import get_settings


class LocalEnvironmentError(RuntimeError):
    pass


def _decode_windows_output(value: bytes) -> str:
    if not value:
        return ""
    if b"\x00" in value:
        return value.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return value.decode(errors="replace")


def _wsl_names(*arguments: str) -> list[str]:
    try:
        result = subprocess.run(
            ["wsl.exe", *arguments],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise LocalEnvironmentError("WSL is not available on this control-plane host") from exc
    return [line.strip().lstrip("*").strip() for line in _decode_windows_output(result.stdout).splitlines() if line.strip()]


def discover_local_environments() -> list[dict[str, Any]]:
    environments = [
        {
            "id": "host:native",
            "kind": "windows" if os.name == "nt" else "linux",
            "name": f"{platform.system()} Native",
            "state": "running",
            "injection_supported": False,
            "start_supported": os.name == "nt",
            "description": "控制面所在的原生环境；默认启动脚本可同时启动本地 Worker。",
        }
    ]
    if os.name != "nt":
        return environments
    try:
        distros = _wsl_names("--list", "--quiet")
        running = set(_wsl_names("--list", "--running", "--quiet"))
    except LocalEnvironmentError:
        return environments
    for distro in distros:
        if distro.lower() in {"docker-desktop", "docker-desktop-data"}:
            continue
        environments.append(
            {
                "id": f"wsl:{distro}",
                "kind": "wsl",
                "name": distro,
                "state": "running" if distro in running else "stopped",
                "injection_supported": True,
                "start_supported": False,
                "description": "本机 WSL2 发行版；ForgeLM 可注入并注册 systemd Worker。",
            }
        )
    return environments


def managed_local_wsl_distros(workers: Iterable[Any]) -> set[str]:
    distros: set[str] = set()
    for worker in workers:
        labels = getattr(worker, "labels", None) or {}
        distro = str(labels.get("distro") or "").strip()
        if (
            labels.get("location") == "local"
            and labels.get("environment") == "wsl"
            and distro
        ):
            distros.add(distro)
    return distros


class LocalWslLifecycleSupervisor:
    def __init__(self, *, restart_delay_seconds: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._explicitly_managed: set[str] = set()
        self._restart_delay_seconds = restart_delay_seconds

    @staticmethod
    def _available_distros() -> set[str]:
        return {
            name
            for name in _wsl_names("--list", "--quiet")
            if name.lower() not in {"docker-desktop", "docker-desktop-data"}
        }

    @staticmethod
    def _command(distro: str) -> list[str]:
        return [
            "wsl.exe",
            "-d",
            distro,
            "--exec",
            "/usr/bin/sleep",
            "infinity",
        ]

    def _refresh_locked(self, distro: str) -> None:
        process = self._processes.get(distro)
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        self._processes.pop(distro, None)
        state = self._states.setdefault(distro, {})
        state.update(
            {
                "state": "error",
                "pid": None,
                "last_exit_code": exit_code,
                "last_error": f"WSL lifecycle host exited with code {exit_code}",
                "next_start_at": time.monotonic() + self._restart_delay_seconds,
            }
        )

    def _start_locked(self, distro: str) -> None:
        state = self._states.setdefault(
            distro,
            {
                "managed": True,
                "state": "stopped",
                "pid": None,
                "restart_count": 0,
                "last_exit_code": None,
                "last_error": None,
                "next_start_at": 0.0,
            },
        )
        if time.monotonic() < float(state.get("next_start_at") or 0):
            return
        had_started = bool(state.get("started_once"))
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                self._command(distro),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(
                {
                    "managed": True,
                    "state": "error",
                    "pid": None,
                    "last_error": str(exc),
                    "next_start_at": time.monotonic() + self._restart_delay_seconds,
                }
            )
            return
        self._processes[distro] = process
        state.update(
            {
                "managed": True,
                "state": "running",
                "pid": process.pid,
                "restart_count": int(state.get("restart_count") or 0)
                + (1 if had_started else 0),
                "last_exit_code": None,
                "last_error": None,
                "next_start_at": 0.0,
                "started_once": True,
            }
        )

    def manage(self, distro: str) -> None:
        if os.name != "nt":
            raise LocalEnvironmentError("Local WSL lifecycle supervision requires Windows")
        distro = distro.strip()
        if not distro or distro not in self._available_distros():
            raise LocalEnvironmentError("The selected WSL distribution is not available")
        with self._lock:
            self._explicitly_managed.add(distro)
            self._refresh_locked(distro)
            if distro not in self._processes:
                self._start_locked(distro)

    def reconcile(self, distros: Iterable[str]) -> None:
        if os.name != "nt":
            return
        try:
            available = self._available_distros()
        except LocalEnvironmentError as exc:
            with self._lock:
                for distro in set(distros) | self._explicitly_managed:
                    state = self._states.setdefault(distro, {"managed": True})
                    state.update({"state": "error", "pid": None, "last_error": str(exc)})
            return
        requested = {str(distro).strip() for distro in distros if str(distro).strip()}
        with self._lock:
            targets = (requested | self._explicitly_managed) & available
            for distro in targets:
                self._refresh_locked(distro)
                if distro not in self._processes:
                    self._start_locked(distro)
            for distro in requested | self._explicitly_managed:
                if distro not in available:
                    state = self._states.setdefault(distro, {"managed": True})
                    state.update(
                        {
                            "state": "error",
                            "pid": None,
                            "last_error": "The managed WSL distribution is not available",
                        }
                    )

    def view(self, distro: str) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked(distro)
            state = self._states.get(distro)
            if state is None:
                return {
                    "managed": False,
                    "state": "stopped",
                    "pid": None,
                    "restart_count": 0,
                    "last_exit_code": None,
                    "last_error": None,
                }
            return {
                key: value
                for key, value in state.items()
                if key not in {"next_start_at", "started_once"}
            }

    def shutdown(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
            self._processes.clear()
            for state in self._states.values():
                state.update({"state": "stopped", "pid": None})
        for process in processes:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


local_wsl_lifecycle_supervisor = LocalWslLifecycleSupervisor()


class LocalNativeWorkerSupervisor:
    def __init__(self, *, repo_root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._state: dict[str, Any] | None = None
        self._repo_root = repo_root or Path(__file__).resolve().parents[4]
        self._log_path = (
            (platform_state_root(get_settings().storage_root) / "logs" / "local-environment-operations" if get_settings().storage_root else self._repo_root / ".llm-platform" / "local-environment-operations") / "native-worker.log"
        )

    @staticmethod
    def _stopped_view() -> dict[str, Any]:
        return {
            "managed": False,
            "state": "stopped",
            "pid": None,
            "last_exit_code": None,
            "last_error": None,
        }

    def _refresh_locked(self) -> None:
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._process = None
        state = self._state or self._stopped_view()
        state.update(
            {
                "managed": True,
                "state": "stopped" if exit_code == 0 else "error",
                "pid": None,
                "last_exit_code": exit_code,
                "last_error": None
                if exit_code == 0
                else f"Windows Native Worker exited with code {exit_code}",
            }
        )
        self._state = state

    @staticmethod
    def validate(environment_id: str) -> None:
        if os.name != "nt":
            raise LocalEnvironmentError("Windows Native Worker startup requires Windows")
        if environment_id != "host:native":
            raise LocalEnvironmentError("Only the discovered Native environment can be started")

    def start(self, environment_id: str) -> dict[str, Any]:
        self.validate(environment_id)
        with self._lock:
            self._refresh_locked()
            if self._process is not None:
                return self._view_locked()

            settings = get_settings()
            environment = os.environ.copy()
            environment.update(
                {
                    "LLM_PLATFORM_API_URL": f"http://127.0.0.1:{settings.api_port}",
                    "LLM_PLATFORM_AGENT_TOKEN": settings.agent_token,
                    "LLM_PLATFORM_WORKER_NAME": f"local-{platform.node()}",
                    "LLM_PLATFORM_WORKER_RELEASE_ROOT": str(
                        ((platform_state_root(get_settings().storage_root) / "releases") if get_settings().storage_root else (self._repo_root / ".llm-platform" / "releases")).resolve()
                    ),
                    "LLM_PLATFORM_WORKER_LABELS": json.dumps(
                        {
                            "location": "local",
                            "environment": "native",
                        },
                        separators=(",", ":"),
                    ),
                }
            )
            environment.pop("LLM_PLATFORM_WORKER_WORKSPACE_ROOT", None)
            environment.pop("LLM_PLATFORM_WORKER_CAPABILITIES", None)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            log_file = None
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = self._log_path.open("ab")
                process = subprocess.Popen(
                    [sys.executable, "-m", "llm_platform_worker.main"],
                    cwd=self._repo_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self._state = {
                    "managed": True,
                    "state": "error",
                    "pid": None,
                    "last_exit_code": None,
                    "last_error": str(exc),
                }
                raise LocalEnvironmentError(
                    f"Could not start the Windows Native Worker: {exc}"
                ) from exc
            finally:
                if log_file is not None:
                    log_file.close()
            self._process = process
            self._state = {
                "managed": True,
                "state": "running",
                "pid": process.pid,
                "last_exit_code": None,
                "last_error": None,
            }
            return self._view_locked()

    def _view_locked(self) -> dict[str, Any]:
        state = self._state or self._stopped_view()
        return dict(state)

    def view(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            return self._view_locked()

    def shutdown(self) -> None:
        with self._lock:
            self._refresh_locked()
            process = self._process
            self._process = None
            if self._state is not None:
                self._state.update({"state": "stopped", "pid": None})
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


local_native_worker_supervisor = LocalNativeWorkerSupervisor()


class LocalInjectionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._operations: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_root = (platform_state_root(get_settings().storage_root) / ("logs/local-environment-operations" if get_settings().storage_root else "local-environment-operations")).resolve()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._view(operation) for operation in self._operations.values()]

    def start(
        self,
        *,
        environment_id: str,
        profile: str,
        workspace_root: str,
    ) -> dict[str, Any]:
        if os.name != "nt":
            raise LocalEnvironmentError("Local WSL injection is only available on Windows")
        if not environment_id.startswith("wsl:"):
            raise LocalEnvironmentError("Only discovered WSL environments can be injected")
        distro = environment_id.removeprefix("wsl:").strip()
        discovered = {item["name"] for item in discover_local_environments() if item["kind"] == "wsl"}
        if distro not in discovered:
            raise LocalEnvironmentError("The selected WSL distribution is not available")
        if profile != "base":
            raise LocalEnvironmentError("Model runtimes must be installed through external suites; Worker profile must be base")
        if not workspace_root.startswith("/") or "'" in workspace_root or "\n" in workspace_root:
            raise LocalEnvironmentError("WSL workspace root must be an absolute POSIX path")
        with self._lock:
            for operation in self._operations.values():
                if operation["environment_id"] == environment_id and operation["status"] == "running":
                    raise LocalEnvironmentError("This environment already has an active injection")

        default_user = self._wsl_text(distro, "id", "-un").strip()
        if not default_user or default_user == "root":
            raise LocalEnvironmentError("WSL must have a non-root default user")
        repo_root = Path(__file__).resolve().parents[4]
        wsl_repo_root = self._wsl_text(distro, "wslpath", "-a", str(repo_root)).strip()
        if not wsl_repo_root.startswith("/"):
            raise LocalEnvironmentError("Could not resolve the ForgeLM repository inside WSL")

        settings = get_settings()
        operation_id = str(uuid.uuid4())
        self._log_root.mkdir(parents=True, exist_ok=True)
        log_path = self._log_root / f"{operation_id}.log"
        labels = json.dumps(
            {
                "location": "local",
                "environment": "wsl",
                "distro": distro,
                "project": "general",
            },
            separators=(",", ":"),
        )
        command = [
            "wsl.exe",
            "-d",
            distro,
            "-u",
            "root",
            "--exec",
            "/usr/bin/env",
            f"FORGELM_WORKER_USER={default_user}",
            "FORGELM_WORKER_ROOT=/data/forgelm",
            f"LLM_PLATFORM_API_URL=http://127.0.0.1:{settings.api_port}",
            f"LLM_PLATFORM_AGENT_TOKEN={settings.agent_token}",
            f"LLM_PLATFORM_WORKER_NAME=forgelm-wsl-{platform.node()}",
            f"LLM_PLATFORM_WORKER_WORKSPACE_ROOT={workspace_root}",
            f"LLM_PLATFORM_WORKER_LABELS={labels}",
            "/bin/bash",
            f"{wsl_repo_root}/scripts/install-wsl-worker.sh",
            "--start",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log_file = log_path.open("wb")
        try:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            log_file.close()
            raise
        now = datetime.now(UTC).isoformat()
        operation = {
            "id": operation_id,
            "environment_id": environment_id,
            "profile": profile,
            "workspace_root": workspace_root,
            "status": "running",
            "pid": process.pid,
            "created_at": now,
            "updated_at": now,
            "exit_code": None,
            "log_path": str(log_path),
        }
        with self._lock:
            self._operations[operation_id] = operation
            self._processes[operation_id] = process
        threading.Thread(
            target=self._wait,
            args=(operation_id, process, log_file),
            daemon=True,
        ).start()
        return self._view(operation)

    def _wsl_text(self, distro: str, *arguments: str) -> str:
        try:
            result = subprocess.run(
                ["wsl.exe", "-d", distro, "--exec", *arguments],
                capture_output=True,
                check=True,
                timeout=15,
            )
        except subprocess.SubprocessError as exc:
            raise LocalEnvironmentError(f"Could not inspect WSL distribution {distro}") from exc
        return _decode_windows_output(result.stdout)

    def _wait(
        self,
        operation_id: str,
        process: subprocess.Popen[bytes],
        log_file,
    ) -> None:
        exit_code = process.wait()
        log_file.close()
        lifecycle_error: str | None = None
        with self._lock:
            distro = self._operations[operation_id]["environment_id"].removeprefix("wsl:")
        if exit_code == 0:
            try:
                local_wsl_lifecycle_supervisor.manage(distro)
            except LocalEnvironmentError as exc:
                lifecycle_error = str(exc)
        with self._lock:
            operation = self._operations[operation_id]
            operation["exit_code"] = exit_code
            operation["status"] = (
                "succeeded" if exit_code == 0 and lifecycle_error is None else "failed"
            )
            if lifecycle_error is not None:
                operation["error"] = lifecycle_error
            operation["updated_at"] = datetime.now(UTC).isoformat()
            self._processes.pop(operation_id, None)

    @staticmethod
    def _tail(path: str, limit: int = 12_000) -> str:
        log_path = Path(path)
        if not log_path.is_file():
            return ""
        with log_path.open("rb") as source:
            source.seek(0, 2)
            size = source.tell()
            source.seek(max(0, size - limit))
            return source.read().decode(errors="replace")

    def _view(self, operation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in operation.items()
            if key not in {"log_path"}
        } | {"log_tail": self._tail(operation["log_path"])}


local_injection_manager = LocalInjectionManager()
