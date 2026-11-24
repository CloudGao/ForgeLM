from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

SUPERVISOR_DIR_ENV = "LLM_PLATFORM_CONTROL_PLANE_SUPERVISOR_DIR"
SUPERVISOR_TOKEN_ENV = "LLM_PLATFORM_CONTROL_PLANE_SUPERVISOR_TOKEN"
INSTANCE_ID_ENV = "LLM_PLATFORM_CONTROL_PLANE_INSTANCE_ID"
SUPERVISOR_SCHEMA = "forgelm.control-plane-supervisor.v1"

_INSTANCE_ID = os.getenv(INSTANCE_ID_ENV) or str(uuid4())


def control_plane_instance_id() -> str:
    return _INSTANCE_ID


def _supervisor_configuration() -> tuple[Path, str] | None:
    raw_directory = os.getenv(SUPERVISOR_DIR_ENV, "").strip()
    token = os.getenv(SUPERVISOR_TOKEN_ENV, "").strip()
    if not raw_directory or not token:
        return None
    return Path(raw_directory).resolve(), token


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > 64 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _is_current_process_or_ancestor(process_id: object) -> bool:
    try:
        expected = int(process_id)
        current = psutil.Process(os.getpid())
        return expected == current.pid or any(parent.pid == expected for parent in current.parents())
    except (TypeError, ValueError, psutil.Error):
        return False


def control_plane_status(*, server_attached: bool) -> dict[str, Any]:
    configuration = _supervisor_configuration()
    status: dict[str, Any] | None = None
    if configuration is not None:
        directory, _ = configuration
        status = _read_json(directory / "status.json")

    managed = bool(
        configuration
        and status
        and status.get("schema") == SUPERVISOR_SCHEMA
        and status.get("instance_id") == _INSTANCE_ID
        and _is_current_process_or_ancestor(status.get("child_pid"))
    )
    response: dict[str, Any] = {
        "managed": managed,
        "restart_supported": managed and server_attached,
        "state": status.get("state", "unmanaged") if managed and status else "unmanaged",
        "instance_id": _INSTANCE_ID,
        "pid": os.getpid(),
    }
    if managed and status:
        response.update(
            {
                "supervisor_pid": status.get("supervisor_pid"),
                "started_at": status.get("child_started_at"),
                "restart_count": status.get("restart_count", 0),
                "last_restart_at": status.get("last_restart_at"),
                "last_restart_reason": status.get("last_restart_reason"),
            }
        )
    return response


def submit_restart_request(reason: str) -> dict[str, Any]:
    configuration = _supervisor_configuration()
    if configuration is None:
        raise RuntimeError("控制面不是由 Supervisor 管理，当前进程不能安全重启")

    directory, token = configuration
    request_id = str(uuid4())
    payload = {
        "schema": SUPERVISOR_SCHEMA,
        "action": "restart",
        "request_id": request_id,
        "instance_id": _INSTANCE_ID,
        "reason": reason,
        "token": token,
    }
    _atomic_write_json(directory / f"request-{request_id}.json", payload)
    return {"request_id": request_id, "instance_id": _INSTANCE_ID}
