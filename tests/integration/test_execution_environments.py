from __future__ import annotations

from fastapi.testclient import TestClient
from llm_platform.routers import system


def test_control_plane_reports_unmanaged_test_server(client: TestClient) -> None:
    response = client.get("/api/control-plane")

    assert response.status_code == 200
    assert response.json()["managed"] is False
    assert response.json()["restart_supported"] is False

    restart = client.post("/api/control-plane/restart", json={})
    assert restart.status_code == 409
    assert restart.json()["detail"]["code"] == "control_plane_not_supervised"


def test_control_plane_restart_submits_before_stopping_server(
    client: TestClient, monkeypatch
) -> None:
    class FakeServer:
        should_exit = False

    fake_server = FakeServer()
    monkeypatch.setattr(
        system,
        "control_plane_status",
        lambda *, server_attached: {
            "managed": True,
            "restart_supported": server_attached,
            "state": "running",
            "instance_id": "instance-a",
        },
    )
    monkeypatch.setattr(
        system,
        "submit_restart_request",
        lambda reason: {"request_id": "request-a", "instance_id": "instance-a"},
    )

    async def stop_immediately(server) -> None:
        server.should_exit = True

    monkeypatch.setattr(system, "_stop_server_after_response", stop_immediately)
    client.app.state.control_plane_server = fake_server
    try:
        response = client.post(
            "/api/control-plane/restart",
            json={"reason": "integration test"},
        )
    finally:
        del client.app.state.control_plane_server

    assert response.status_code == 202
    assert response.json() == {
        "accepted": True,
        "request_id": "request-a",
        "instance_id": "instance-a",
    }
    assert fake_server.should_exit is True


def test_local_execution_environment_api_uses_discovered_hosts(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        system,
        "discover_local_environments",
        lambda: [
            {
                "id": "host:native",
                "kind": "windows",
                "name": "Windows Native",
                "state": "running",
                "injection_supported": False,
                "description": "native",
            },
            {
                "id": "wsl:MicroLM-Ubuntu",
                "kind": "wsl",
                "name": "MicroLM-Ubuntu",
                "state": "stopped",
                "injection_supported": True,
                "description": "wsl",
            },
        ],
    )
    monkeypatch.setattr(
        system.local_wsl_lifecycle_supervisor,
        "view",
        lambda distro: {
            "managed": distro == "MicroLM-Ubuntu",
            "state": "running",
            "pid": 123,
            "restart_count": 0,
            "last_exit_code": None,
            "last_error": None,
        },
    )
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "view",
        lambda: {
            "managed": False,
            "state": "stopped",
            "pid": None,
            "last_exit_code": None,
            "last_error": None,
        },
    )

    response = client.get("/api/execution-environments/local")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [
        "host:native",
        "wsl:MicroLM-Ubuntu",
    ]
    assert response.json()[0]["supervision"] == {
        "managed": False,
        "state": "stopped",
        "pid": None,
        "last_exit_code": None,
        "last_error": None,
    }
    assert response.json()[1]["supervision"] == {
        "managed": True,
        "state": "running",
        "pid": 123,
        "restart_count": 0,
        "last_exit_code": None,
        "last_error": None,
    }


def test_local_injection_rejects_native_environment(client: TestClient) -> None:
    response = client.post(
        "/api/execution-environments/local/inject",
        json={
            "environment_id": "host:native",
            "profile": "base",
            "workspace_root": "/data/forgelm/workspaces",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "local_environment_injection_failed"


def test_local_native_worker_start_uses_fixed_supervisor(
    client: TestClient, monkeypatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(system, "_has_online_native_worker", lambda _session: False)
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "validate",
        lambda environment_id: calls.append(f"validate:{environment_id}"),
    )
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "start",
        lambda environment_id: calls.append(f"start:{environment_id}")
        or {"managed": True, "state": "running", "pid": 321},
    )

    response = client.post(
        "/api/execution-environments/local/start",
        json={"environment_id": "host:native"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "managed": True,
        "state": "running",
        "pid": 321,
        "already_online": False,
        "worker_status": "offline",
    }
    assert calls == ["validate:host:native", "start:host:native"]


def test_local_native_worker_start_is_idempotent_when_worker_is_online(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(system, "_has_online_native_worker", lambda _session: True)
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "validate",
        lambda _environment_id: None,
    )
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "view",
        lambda: {"managed": False, "state": "stopped", "pid": None},
    )
    monkeypatch.setattr(
        system.local_native_worker_supervisor,
        "start",
        lambda _environment_id: (_ for _ in ()).throw(AssertionError("duplicate start")),
    )

    response = client.post(
        "/api/execution-environments/local/start",
        json={"environment_id": "host:native"},
    )

    assert response.status_code == 202
    assert response.json()["already_online"] is True
    assert response.json()["worker_status"] == "online"
