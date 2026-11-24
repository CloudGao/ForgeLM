from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from llm_platform.services import local_environments


def test_decode_windows_output_handles_wsl_utf16() -> None:
    encoded = "MicroLM-Ubuntu\r\n".encode("utf-16-le")

    assert local_environments._decode_windows_output(encoded) == "MicroLM-Ubuntu\r\n"


def test_discover_local_environments_lists_wsl_state(monkeypatch) -> None:
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    calls: list[tuple[str, ...]] = []

    def names(*arguments: str) -> list[str]:
        calls.append(arguments)
        return ["docker-desktop", "MicroLM-Ubuntu", "Ubuntu"] if "--running" not in arguments else ["Ubuntu"]

    monkeypatch.setattr(local_environments, "_wsl_names", names)

    environments = local_environments.discover_local_environments()

    assert environments[0]["id"] == "host:native"
    assert [(item["name"], item["state"]) for item in environments[1:]] == [
        ("MicroLM-Ubuntu", "stopped"),
        ("Ubuntu", "running"),
    ]
    assert calls == [("--list", "--quiet"), ("--list", "--running", "--quiet")]


def test_injection_rejects_non_wsl_environment(monkeypatch) -> None:
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    manager = local_environments.LocalInjectionManager()

    with pytest.raises(local_environments.LocalEnvironmentError, match="Only discovered WSL"):
        manager.start(
            environment_id="host:native",
            profile="base",
            workspace_root="/data/forgelm/workspaces",
        )


def test_managed_local_wsl_distros_only_uses_local_wsl_workers() -> None:
    workers = [
        SimpleNamespace(
            labels={
                "location": "local",
                "environment": "wsl",
                "distro": "MicroLM-Ubuntu",
            }
        ),
        SimpleNamespace(
            labels={"location": "remote", "environment": "wsl", "distro": "Remote-Ubuntu"}
        ),
        SimpleNamespace(labels={"location": "local", "environment": "native"}),
    ]

    assert local_environments.managed_local_wsl_distros(workers) == {"MicroLM-Ubuntu"}


def test_wsl_lifecycle_supervisor_starts_restarts_and_stops(monkeypatch) -> None:
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    monkeypatch.setattr(
        local_environments,
        "_wsl_names",
        lambda *_arguments: ["docker-desktop", "MicroLM-Ubuntu"],
    )
    created = []

    class FakeProcess:
        def __init__(self, command, **_kwargs) -> None:
            self.command = command
            self.pid = 1000 + len(created)
            self.exit_code = None
            self.terminated = False
            created.append(self)

        def poll(self):
            return self.exit_code

        def terminate(self) -> None:
            self.terminated = True
            self.exit_code = 0

        def wait(self, timeout=None):
            return self.exit_code

        def kill(self) -> None:
            self.exit_code = -9

    monkeypatch.setattr(local_environments.subprocess, "Popen", FakeProcess)
    supervisor = local_environments.LocalWslLifecycleSupervisor(restart_delay_seconds=0)

    supervisor.reconcile({"MicroLM-Ubuntu"})

    assert len(created) == 1
    assert created[0].command == [
        "wsl.exe",
        "-d",
        "MicroLM-Ubuntu",
        "--exec",
        "/usr/bin/sleep",
        "infinity",
    ]
    assert supervisor.view("MicroLM-Ubuntu") == {
        "managed": True,
        "state": "running",
        "pid": 1000,
        "restart_count": 0,
        "last_exit_code": None,
        "last_error": None,
    }

    supervisor.reconcile({"MicroLM-Ubuntu"})
    assert len(created) == 1

    created[0].exit_code = 9
    supervisor.reconcile({"MicroLM-Ubuntu"})

    assert len(created) == 2
    assert supervisor.view("MicroLM-Ubuntu")["restart_count"] == 1
    supervisor.shutdown()
    assert created[1].terminated is True
    assert supervisor.view("MicroLM-Ubuntu")["state"] == "stopped"


def test_wsl_lifecycle_supervisor_rejects_unknown_distro(monkeypatch) -> None:
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    monkeypatch.setattr(local_environments, "_wsl_names", lambda *_arguments: ["Ubuntu"])
    supervisor = local_environments.LocalWslLifecycleSupervisor()

    with pytest.raises(local_environments.LocalEnvironmentError, match="not available"):
        supervisor.manage("MicroLM-Ubuntu")


def test_native_worker_supervisor_starts_fixed_worker_and_stops(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    monkeypatch.setattr(
        local_environments,
        "get_settings",
        lambda: SimpleNamespace(api_port=8123, agent_token="agent-secret", storage_root=""),
    )
    monkeypatch.setattr(local_environments.platform, "node", lambda: "WORKSTATION")
    monkeypatch.setenv("LLM_PLATFORM_WORKER_WORKSPACE_ROOT", "C:/stale-worker-root")
    monkeypatch.setenv("LLM_PLATFORM_WORKER_CAPABILITIES", "microlm.workflow.v1")
    created = []

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            self.command = command
            self.kwargs = kwargs
            self.pid = 4321
            self.exit_code = None
            self.terminated = False
            created.append(self)

        def poll(self):
            return self.exit_code

        def terminate(self) -> None:
            self.terminated = True
            self.exit_code = 0

        def wait(self, timeout=None):
            return self.exit_code

        def kill(self) -> None:
            self.exit_code = -9

    monkeypatch.setattr(local_environments.subprocess, "Popen", FakeProcess)
    supervisor = local_environments.LocalNativeWorkerSupervisor(repo_root=tmp_path)

    state = supervisor.start("host:native")

    assert state == {
        "managed": True,
        "state": "running",
        "pid": 4321,
        "last_exit_code": None,
        "last_error": None,
    }
    assert created[0].command == [
        local_environments.sys.executable,
        "-m",
        "llm_platform_worker.main",
    ]
    assert created[0].kwargs["cwd"] == tmp_path
    assert created[0].kwargs["env"]["LLM_PLATFORM_API_URL"] == "http://127.0.0.1:8123"
    assert created[0].kwargs["env"]["LLM_PLATFORM_AGENT_TOKEN"] == "agent-secret"
    assert created[0].kwargs["env"]["LLM_PLATFORM_WORKER_NAME"] == "local-WORKSTATION"
    assert json.loads(created[0].kwargs["env"]["LLM_PLATFORM_WORKER_LABELS"]) == {
        "location": "local",
        "environment": "native",
    }
    assert "LLM_PLATFORM_WORKER_WORKSPACE_ROOT" not in created[0].kwargs["env"]
    assert "LLM_PLATFORM_WORKER_CAPABILITIES" not in created[0].kwargs["env"]

    assert supervisor.start("host:native")["pid"] == 4321
    assert len(created) == 1
    supervisor.shutdown()
    assert created[0].terminated is True
    assert supervisor.view()["state"] == "stopped"


def test_native_worker_supervisor_rejects_unsupported_hosts(monkeypatch, tmp_path) -> None:
    supervisor = local_environments.LocalNativeWorkerSupervisor(repo_root=tmp_path)
    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="posix", environ=local_environments.os.environ))

    with pytest.raises(local_environments.LocalEnvironmentError, match="requires Windows"):
        supervisor.start("host:native")

    monkeypatch.setattr(local_environments, "os", SimpleNamespace(name="nt", environ=local_environments.os.environ))
    with pytest.raises(local_environments.LocalEnvironmentError, match="Native environment"):
        supervisor.start("wsl:Ubuntu")
