from __future__ import annotations

import base64
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from llm_platform.database import SessionLocal
from llm_platform.models import RemoteWorkerEnvironment, RemoteWorkerOperation
from llm_platform.services import remote_workers


def test_capture_host_key_falls_back_after_keyscan_kex_failure(monkeypatch) -> None:
    import paramiko

    key = paramiko.RSAKey.generate(2048)
    monkeypatch.setattr(
        remote_workers, "_run_command",
        MagicMock(side_effect=remote_workers.RemoteWorkerError("unsupported KEX method")),
    )
    connection = MagicMock()
    monkeypatch.setattr(remote_workers.socket, "create_connection", connection)
    transport = MagicMock()
    transport.__enter__.return_value = transport
    transport.get_remote_server_key.return_value = key
    constructor = MagicMock(return_value=transport)
    monkeypatch.setattr(paramiko, "Transport", constructor)
    expected = remote_workers._fingerprint(key.get_base64())

    host_key, fingerprint = remote_workers.capture_host_key(
        "host.test", 44731, expected_fingerprint=expected
    )

    assert host_key == f"[host.test]:44731 ssh-rsa {key.get_base64()}"
    assert fingerprint == expected
    connection.assert_called_once_with(("host.test", 44731), timeout=15)
    transport.start_client.assert_called_once_with(timeout=15)
    transport.auth_password.assert_not_called()
    with pytest.raises(remote_workers.RemoteWorkerError, match="host key mismatch"):
        remote_workers.capture_host_key("host.test", 44731, expected_fingerprint="SHA256:wrong")


def test_capture_host_key_pins_expected_fingerprint(monkeypatch) -> None:
    key_blob = base64.b64encode(b"test-host-key").decode("ascii")
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(b"test-host-key").digest()).decode(
        "ascii"
    ).rstrip("=")
    monkeypatch.setattr(
        remote_workers,
        "_run_command",
        lambda _command, timeout=90: subprocess.CompletedProcess(
            [], 0, stdout=f"host.test ssh-ed25519 {key_blob}\n", stderr=""
        ),
    )

    host_key, actual = remote_workers.capture_host_key(
        "host.test", 22, expected_fingerprint=fingerprint
    )

    assert host_key == f"host.test ssh-ed25519 {key_blob}"
    assert actual == fingerprint
    with pytest.raises(remote_workers.RemoteWorkerError, match="host key mismatch"):
        remote_workers.capture_host_key("host.test", 22, expected_fingerprint="SHA256:not-the-host")


def test_remote_provision_builds_release_and_uses_fixed_bootstrap(
    monkeypatch, tmp_path: Path, client
) -> None:
    del tmp_path, client
    environment_name = f"remote-operation-{uuid4().hex[:8]}"
    with SessionLocal() as session:
        environment = RemoteWorkerEnvironment(
            name=environment_name,
            ssh_host="compute.example.test",
            ssh_port=22,
            ssh_user="forgelm",
            credential_ref="FORGELM_TEST_SSH_KEY",
            host_key="compute.example.test ssh-ed25519 AAAATEST",
            host_key_fingerprint="SHA256:test",
            control_plane_url="https://control.example.test",
            workspace_root="/srv/forgelm/workspaces",
            install_root="/opt/forgelm-worker",
            profile="data",
            labels={"zone": "test"},
        )
        session.add(environment)
        session.flush()
        operation = RemoteWorkerOperation(
            environment_id=environment.id,
            kind="provision",
            status="queued",
            phase="queued",
        )
        session.add(operation)
        session.commit()
        environment_id = environment.id
        operation_id = operation.id

    commands: list[str] = []
    uploaded_environment = ""

    def fake_ssh(_environment, _known_hosts, command, *, timeout=90, on_line=None):
        del timeout
        commands.append(command)
        if "--phase prepare" in command:
            with SessionLocal() as session:
                assert session.get(RemoteWorkerEnvironment, environment_id).enrollment_token_hash is None
            on_line("Installing runtime dependencies")
        return subprocess.CompletedProcess([], 0, stdout="ok", stderr="")

    def fake_build(destination: Path) -> Path:
        wheel = destination / "local_llm_platform-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"fixed-wheel")
        return wheel

    def fake_scp(_environment, _known_hosts, sources, _destination) -> None:
        nonlocal uploaded_environment
        environment_file = next(path for path in sources if path.name == "worker.env")
        uploaded_environment = environment_file.read_text(encoding="utf-8")

    monkeypatch.setattr(remote_workers, "_ssh", fake_ssh)
    monkeypatch.setattr(remote_workers, "_scp", fake_scp)
    monkeypatch.setattr(remote_workers, "_build_wheel", fake_build)

    remote_workers.run_remote_worker_operation(operation_id)

    with SessionLocal() as session:
        operation = session.get(RemoteWorkerOperation, operation_id)
        environment = session.get(RemoteWorkerEnvironment, environment_id)
        assert operation is not None
        assert environment is not None
        assert operation.status == "succeeded"
        assert operation.phase == "waiting_for_agent"
        assert environment.deployed_release
        assert environment.enrollment_token_hash
        assert all("LLM_PLATFORM_AGENT_TOKEN" not in line for line in operation.logs)
    probe_command = commands[0]
    assert "test -d /run/systemd/system" in probe_command
    assert "Python 3.12" in probe_command
    assert "venv/pip" in probe_command
    install_command = next(command for command in commands if "--wheel" in command)
    assert "sudo -n bash" in install_command
    assert "bootstrap-remote-worker.sh" in install_command
    assert "--service-user forgelm" in install_command
    assert commands.index(next(command for command in commands if "--phase prepare" in command)) < commands.index(next(command for command in commands if "--phase activate" in command))
    assert f'LLM_PLATFORM_ENROLLMENT_ENVIRONMENT_ID="{environment_id}"' in uploaded_environment
    assert "LLM_PLATFORM_AGENT_TOKEN=" in uploaded_environment
    assert "FORGELM_TEST_SSH_KEY" not in uploaded_environment
