from __future__ import annotations

import json
import os
from pathlib import Path

from llm_platform.services import control_plane
from llm_platform.supervisor import ControlPlaneSupervisor


def test_managed_status_matches_the_exact_supervised_child(
    tmp_path: Path, monkeypatch
) -> None:
    token = "private-supervisor-token"
    instance_id = "instance-a"
    monkeypatch.setenv(control_plane.SUPERVISOR_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(control_plane.SUPERVISOR_TOKEN_ENV, token)
    monkeypatch.setattr(control_plane, "_INSTANCE_ID", instance_id)
    control_plane._atomic_write_json(
        tmp_path / "status.json",
        {
            "schema": control_plane.SUPERVISOR_SCHEMA,
            "state": "running",
            "supervisor_pid": 99,
            "child_pid": os.getpid(),
            "instance_id": instance_id,
            "child_started_at": "2026-08-19T00:00:00+00:00",
            "restart_count": 2,
        },
    )

    status = control_plane.control_plane_status(server_attached=True)

    assert status["managed"] is True
    assert status["restart_supported"] is True
    assert status["supervisor_pid"] == 99
    assert "token" not in status


def test_restart_request_is_signed_for_the_current_instance(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(control_plane.SUPERVISOR_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(control_plane.SUPERVISOR_TOKEN_ENV, "private-supervisor-token")
    monkeypatch.setattr(control_plane, "_INSTANCE_ID", "instance-a")

    result = control_plane.submit_restart_request("test restart")
    request_path = next(tmp_path.glob("request-*.json"))
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert result == {"request_id": payload["request_id"], "instance_id": "instance-a"}
    assert payload["action"] == "restart"
    assert payload["reason"] == "test restart"
    assert payload["token"] == "private-supervisor-token"


def test_supervisor_ignores_unsigned_requests_and_collapses_valid_requests(
    tmp_path: Path,
) -> None:
    supervisor = ControlPlaneSupervisor(state_directory=tmp_path)
    supervisor.instance_id = "instance-a"
    valid = {
        "schema": control_plane.SUPERVISOR_SCHEMA,
        "action": "restart",
        "instance_id": "instance-a",
        "token": supervisor.token,
        "reason": "latest request",
    }
    (tmp_path / "request-1.json").write_text(
        json.dumps({**valid, "token": "wrong"}), encoding="utf-8"
    )
    (tmp_path / "request-2.json").write_text(json.dumps(valid), encoding="utf-8")

    accepted = supervisor._consume_restart_request()

    assert accepted is not None
    assert accepted["reason"] == "latest request"
    assert list(tmp_path.glob("request-*.json")) == []
