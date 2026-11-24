from __future__ import annotations

import hashlib
import shlex
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import RemoteWorkerEnvironment, WorkerNode
from llm_platform.schemas import RemoteWorkerEnvironmentCreate
from llm_platform.services.remote_workers import _environment_file, deployment_options
from llm_platform.services.ssh_tunnels import AgentGateway, TunnelManager
from pydantic import ValidationError


def payload(**values):
    return {
        "name": "autodl",
        "ssh_command": "ssh -p 12345 root@gpu.example.test",
        "ssh_password": "test-password",
        "transport": "ssh_tunnel",
        **values,
    }


def test_parse_autodl_command_and_root_runtime():
    result = RemoteWorkerEnvironmentCreate(**payload(runtime="inference"))
    assert (result.ssh_host, result.ssh_port, result.ssh_user) == (
        "gpu.example.test",
        12345,
        "root",
    )
    assert result.runtime == "inference"
    assert "test-password" not in repr(result)


@pytest.mark.parametrize(
    "command",
    [
        "ssh -o ProxyCommand=evil root@host",
        "ssh root@host touch /tmp/evil",
        "ssh -p 0 root@host",
        "ssh root@host;evil",
        "ssh -p 22 -bad@host",
        "ssh root@host\nwhoami",
        "ssh root@$(whoami)",
    ],
)
def test_reject_executable_ssh_input(command):
    with pytest.raises(ValidationError):
        RemoteWorkerEnvironmentCreate(**payload(ssh_command=command))


def test_credentials_and_data_runtime_are_unambiguous():
    with pytest.raises(ValidationError):
        RemoteWorkerEnvironmentCreate(**payload(credential_ref="KEY"))
    with pytest.raises(ValidationError):
        RemoteWorkerEnvironmentCreate(**payload(profile="data", runtime="inference"))


def test_environment_is_parsed_as_data_not_shell():
    env = SimpleNamespace(
        id="test",
        name='$(touch /tmp/evil) `evil` "name"',
        control_plane_url="http://example.test",
        install_root="/opt/worker",
        workspace_root="/srv/worker",
        profile="general",
        labels={},
        metadata_json={"deployment": {"transport": "ssh_tunnel", "runtime": "inference"}},
    )
    values = {
        line.split("=", 1)[0]: shlex.split(line.split("=", 1)[1])[0]
        for line in _environment_file(env, "secret").splitlines()
    }
    assert values["LLM_PLATFORM_WORKER_NAME"] == env.name
    assert values["LLM_PLATFORM_API_URL"] == "http://127.0.0.1:18765"
    assert "LLM_PLATFORM_WORKER_CAPABILITIES" not in values
    assert deployment_options(SimpleNamespace(metadata_json={})) == {
        "transport": "direct",
        "runtime": "base",
        "service_mode": "systemd",
        "download_policy": "official",
    }


def test_gateway_is_scoped_to_environment_and_cannot_access_local_management(client):
    from starlette.responses import JSONResponse

    called = []

    async def app(scope, receive, send):
        called.append(scope)
        await JSONResponse({"ok": True})(scope, receive, send)

    with SessionLocal() as session:
        worker = WorkerNode(
            name=uuid4().hex, hostname="remote", token_hash=hashlib.sha256(b"node").hexdigest()
        )
        session.add(worker)
        session.flush()
        env = RemoteWorkerEnvironment(
            name=uuid4().hex,
            ssh_host="example.test",
            ssh_user="root",
            credential_ref="KEY",
            host_key="test",
            host_key_fingerprint="test",
            control_plane_url="http://localhost",
            worker_id=worker.id,
            enrollment_token_hash=hashlib.sha256(b"enroll").hexdigest(),
        )
        session.add(env)
        session.commit()
        env_id, worker_id = env.id, worker.id
    gateway = AgentGateway(app, env_id)
    test = TestClient(gateway)
    headers = {"X-Agent-Token": "node", "X-Worker-ID": worker_id}
    assert test.get("/api/projects", headers=headers).status_code == 403
    assert test.post("/api/execution-environments/remote", headers=headers).status_code == 403
    assert (
        test.get(
            "/api/agent/artifacts/test", headers={"X-Agent-Token": "local-development-token"}
        ).status_code
        == 401
    )
    assert (
        test.get(
            "/api/agent/artifacts/test", headers={**headers, "X-Worker-ID": "another-worker"}
        ).status_code
        == 401
    )
    assert test.get("/api/agent/artifacts/test", headers=headers).status_code == 200
    assert called[-1]["client"][0] == "remote-worker"
    assert test.post("/api/agent/enroll", headers={"X-Agent-Token": "enroll"}).status_code == 200
    assert (
        test.post("/api/agent/tasks/claim", headers={"X-Agent-Token": "enroll"}).status_code == 401
    )
    with SessionLocal() as session:
        session.delete(session.get(RemoteWorkerEnvironment, env_id))
        session.commit()
    assert test.get("/api/agent/artifacts/test", headers=headers).status_code == 401


def test_manager_reconnects_with_backoff_and_closes_only_owned_tunnels(monkeypatch):
    import threading

    from llm_platform.services import ssh_tunnels

    now = [100.0]
    monkeypatch.setattr(ssh_tunnels.time, "monotonic", lambda: now[0])

    class FakeTunnel:
        def __init__(self, app, environment):
            self.process = None
            self.ready = threading.Event()
            self.failures = 0
            self.next_retry = 0
            self.errors = []
            self.starts = 0
            self.closed = False

        def start(self, env):
            self.starts += 1
            self.process = SimpleNamespace(poll=lambda: None)
            self.ready.set()

        def close(self):
            self.closed = True

    monkeypatch.setattr(ssh_tunnels, "Tunnel", FakeTunnel)
    manager = TunnelManager()
    manager.app = object()
    env = SimpleNamespace(id="node")
    manager.ensure(env)
    tunnel = manager.tunnels["node"]
    manager.ensure(env)
    assert tunnel.starts == 1
    assert manager.view("node")["state"] == "connected"
    tunnel.process.poll = lambda: 255
    manager.ensure(env)
    assert tunnel.starts == 1
    assert manager.view("node")["state"] == "reconnecting"
    now[0] += 4
    manager.ensure(env)
    assert tunnel.starts == 2
    assert manager.view("node")["reconnect_count"] == 1
    manager.shutdown()
    assert tunnel.closed
    assert not manager.tunnels


def test_password_bootstrap_persists_only_dedicated_key(monkeypatch, tmp_path):
    import paramiko
    from llm_platform.services import ssh_credentials

    connections, commands = [], []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_host_keys(self):
            return paramiko.HostKeys()

        def connect(self, host, **kwargs):
            connections.append(kwargs)

        def exec_command(self, command, **kwargs):
            commands.append(command)
            return None, SimpleNamespace(channel=SimpleNamespace(recv_exit_status=lambda: 0)), None

    monkeypatch.setattr(paramiko, "SSHClient", Client)
    directory = tmp_path / "keys"
    directory.mkdir()
    monkeypatch.setattr(ssh_credentials, "credential_directory", lambda: directory)
    host = paramiko.RSAKey.generate(2048)
    reference = ssh_credentials.prepare_credential(
        RemoteWorkerEnvironmentCreate(**payload()),
        f"[gpu.example.test]:12345 {host.get_name()} {host.get_base64()}",
    )
    assert connections[0]["password"] == "test-password"
    assert "password" not in connections[1]
    assert connections[1]["key_filename"] == str(directory / reference)
    key = paramiko.RSAKey.from_private_key_file(str(directory / reference))
    assert key.get_base64() in commands[0]
    assert "test-password" not in (directory / reference).read_text()
    assert len(list(directory.iterdir())) == 1
