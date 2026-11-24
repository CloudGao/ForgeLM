from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any
from uuid import uuid4

import psutil

from .services.control_plane import (
    INSTANCE_ID_ENV,
    SUPERVISOR_DIR_ENV,
    SUPERVISOR_SCHEMA,
    SUPERVISOR_TOKEN_ENV,
    _atomic_write_json,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


class SupervisorAlreadyRunning(RuntimeError):
    pass


class _SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _SingleInstanceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise SupervisorAlreadyRunning("已有一个控制面 Supervisor 正在运行") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class ControlPlaneSupervisor:
    def __init__(
        self,
        *,
        state_directory: Path | None = None,
        command: list[str] | None = None,
        working_directory: Path | None = None,
        environment: dict[str, str] | None = None,
        poll_interval: float = 0.1,
        restart_grace_period: float = 10.0,
        shutdown_timeout: float = 15.0,
        restart_delay: float = 0.4,
    ) -> None:
        root = _repository_root()
        from .config import get_settings
        from llm_platform_core.storage_layout import platform_state_root
        settings = get_settings()
        configured_directory = os.getenv(SUPERVISOR_DIR_ENV, "").strip()
        self.state_directory = (
            state_directory
            or (Path(configured_directory) if configured_directory else None)
            or platform_state_root(settings.storage_root) / "control-plane-supervisor"
        ).resolve()
        self.command = command or [sys.executable, "-m", "llm_platform.main"]
        self.working_directory = (working_directory or root).resolve()
        self.environment = dict(environment or os.environ)
        if settings.storage_root:
            self.environment["LLM_PLATFORM_STORAGE_ROOT"] = settings.storage_root
        self.poll_interval = poll_interval
        self.restart_grace_period = restart_grace_period
        self.shutdown_timeout = shutdown_timeout
        self.restart_delay = restart_delay
        self.token = secrets.token_urlsafe(32)
        self.stop_event = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.instance_id: str | None = None
        self.started_at = _utc_now()
        self.child_started_at: str | None = None
        self.restart_count = 0
        self.last_restart_at: str | None = None
        self.last_restart_reason: str | None = None
        self.last_exit_code: int | None = None
        self.last_error: str | None = None

    @property
    def status_path(self) -> Path:
        return self.state_directory / "status.json"

    def _status_payload(self, state: str) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_SCHEMA,
            "state": state,
            "supervisor_pid": os.getpid(),
            "child_pid": self.process.pid if self.process and self.process.poll() is None else None,
            "instance_id": self.instance_id,
            "started_at": self.started_at,
            "child_started_at": self.child_started_at,
            "restart_count": self.restart_count,
            "last_restart_at": self.last_restart_at,
            "last_restart_reason": self.last_restart_reason,
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
        }

    def _write_status(self, state: str) -> None:
        _atomic_write_json(self.status_path, self._status_payload(state))

    def _remove_stale_requests(self) -> None:
        for path in self.state_directory.glob("request-*.json"):
            try:
                path.unlink()
            except OSError:
                pass

    def _consume_restart_request(self) -> dict[str, Any] | None:
        accepted: dict[str, Any] | None = None
        for path in sorted(self.state_directory.glob("request-*.json")):
            try:
                if path.stat().st_size > 64 * 1024:
                    continue
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(candidate, dict)
                    and candidate.get("schema") == SUPERVISOR_SCHEMA
                    and candidate.get("action") == "restart"
                    and secrets.compare_digest(str(candidate.get("token", "")), self.token)
                    and candidate.get("instance_id") == self.instance_id
                ):
                    accepted = candidate
            except (OSError, ValueError, TypeError):
                pass
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return accepted

    def _start_child(self) -> None:
        self.instance_id = str(uuid4())
        child_environment = dict(self.environment)
        child_environment.update(
            {
                SUPERVISOR_DIR_ENV: str(self.state_directory),
                SUPERVISOR_TOKEN_ENV: self.token,
                INSTANCE_ID_ENV: self.instance_id,
            }
        )
        self.child_started_at = _utc_now()
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.working_directory,
                env=child_environment,
            )
        except OSError as exc:
            self.last_error = str(exc)
            self._write_status("failed")
            raise
        self.last_error = None
        self._write_status("running")

    def _stop_child(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            root = psutil.Process(process.pid)
            owned_processes = root.children(recursive=True) + [root]
        except psutil.Error:
            owned_processes = []
        for owned in reversed(owned_processes):
            try:
                owned.terminate()
            except psutil.Error:
                pass
        if not owned_processes:
            process.terminate()
        _, alive = psutil.wait_procs(owned_processes, timeout=self.shutdown_timeout)
        for owned in alive:
            try:
                owned.kill()
            except psutil.Error:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=self.shutdown_timeout)
        try:
            process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.shutdown_timeout)

    def request_stop(self, _signum: int | None = None, _frame: FrameType | None = None) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.state_directory.mkdir(parents=True, exist_ok=True)
        with _SingleInstanceLock(self.state_directory / "supervisor.lock"):
            self._remove_stale_requests()
            previous_handlers: dict[int, Any] = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self.request_stop)
            pending_restart: dict[str, Any] | None = None
            restart_deadline: float | None = None
            try:
                self._start_child()
                while not self.stop_event.is_set():
                    request = self._consume_restart_request()
                    if request is not None and pending_restart is None:
                        pending_restart = request
                        restart_deadline = time.monotonic() + self.restart_grace_period
                        self._write_status("restarting")

                    process = self.process
                    exit_code = process.poll() if process is not None else -1
                    if exit_code is not None:
                        self.last_exit_code = exit_code
                        if self.stop_event.is_set():
                            break
                        self.restart_count += 1
                        self.last_restart_at = _utc_now()
                        self.last_restart_reason = (
                            str(pending_restart.get("reason") or "requested")
                            if pending_restart
                            else "unexpected_exit"
                        )
                        self._write_status("backing_off")
                        if self.stop_event.wait(self.restart_delay):
                            break
                        pending_restart = None
                        restart_deadline = None
                        self._start_child()
                        continue

                    if (
                        pending_restart is not None
                        and restart_deadline is not None
                        and time.monotonic() >= restart_deadline
                    ):
                        self._stop_child()
                    self.stop_event.wait(self.poll_interval)
            finally:
                self.stop_event.set()
                try:
                    self._stop_child()
                finally:
                    self.process = None
                    self.instance_id = None
                    self._write_status("stopped")
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)


def run() -> None:
    try:
        ControlPlaneSupervisor().run()
    except SupervisorAlreadyRunning as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    run()
