from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import RemoteWorkerEnvironment, RemoteWorkerOperation, ResourceLease, utcnow

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOOTSTRAP_SCRIPT = _REPO_ROOT / "scripts" / "bootstrap-remote-worker.sh"
_SENSITIVE_LOG = re.compile(
    r"(?i)(authorization|password|passwd|secret|token|api[_-]?key)(\s*[:=]\s*)\S+"
)
_RELEASE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class RemoteWorkerError(RuntimeError):
    pass


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _redact(value: str, *, limit: int = 12_000) -> str:
    return _SENSITIVE_LOG.sub(r"\1\2[REDACTED]", value).strip()[-limit:]


def _run_command(
    command: list[str], *, timeout: int = 90, on_line=None
) -> subprocess.CompletedProcess[str]:
    try:
        if on_line is not None:
            return _stream_command(command, timeout=timeout, on_line=on_line)
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise RemoteWorkerError(f"required executable is unavailable: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteWorkerError(f"command timed out: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = _redact((exc.stderr or exc.stdout or "").strip())
        raise RemoteWorkerError(detail or f"command failed: {command[0]}") from exc


def _stream_command(command, *, timeout, on_line):
    output = queue.Queue(maxsize=500)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    done = threading.Event()

    def read():
        try:
            for line in process.stdout:
                try:
                    output.put_nowait(line[-2000:])
                except queue.Full:
                    pass
        finally:
            process.stdout.close()
            done.set()

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + timeout
    tail = []
    try:
        while not done.is_set() or not output.empty():
            if time.monotonic() > deadline:
                raise RemoteWorkerError("远端安装超时；可查看日志后重试修复")
            try:
                line = output.get(timeout=0.2)
            except queue.Empty:
                continue
            tail = [*tail, _redact(line)][-80:]
            on_line(_redact(line))
        code = process.wait(timeout=5)
        if code:
            raise RemoteWorkerError("\n".join(tail[-15:]))
        return subprocess.CompletedProcess(command, code, stdout="\n".join(tail), stderr="")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _fingerprint(key_blob: str) -> str:
    try:
        decoded = base64.b64decode(key_blob.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RemoteWorkerError("ssh-keyscan returned an invalid host key") from exc
    digest = base64.b64encode(hashlib.sha256(decoded).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def capture_host_key(
    host: str, port: int, *, expected_fingerprint: str | None = None
) -> tuple[str, str]:
    try:
        result = _run_command(["ssh-keyscan", "-T", "10", "-p", str(port), host], timeout=20)
        scanned = result.stdout
    except RemoteWorkerError:
        scanned = ""
    candidates: list[tuple[int, str, str]] = []
    priority = {"ssh-ed25519": 0, "ecdsa-sha2-nistp256": 1, "ssh-rsa": 2}
    for raw_line in scanned.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or fields[1] not in priority:
            continue
        candidates.append((priority[fields[1]], line, _fingerprint(fields[2])))
    if not candidates:
        # Windows OpenSSH 9.5 keyscan may select an unsupported sntrup KEX.
        # Fetch the public key via an unauthenticated handshake, then pin it as usual.
        import paramiko
        try:
            with socket.create_connection((host, port), timeout=15) as connection:
                with paramiko.Transport(connection) as transport:
                    transport.start_client(timeout=15)
                    key = transport.get_remote_server_key()
                    endpoint = host if port == 22 else f"[{host}]:{port}"
                    candidates.append((0, f"{endpoint} {key.get_name()} {key.get_base64()}", _fingerprint(key.get_base64())))
        except Exception as exc:
            raise RemoteWorkerError("无法获取 SSH 主机公钥，请检查地址、端口及实例状态") from exc
    _, host_key, fingerprint = sorted(candidates, key=lambda item: item[0])[0]
    if expected_fingerprint and not secrets.compare_digest(
        expected_fingerprint.strip(), fingerprint
    ):
        raise RemoteWorkerError(
            f"SSH host key mismatch: expected {expected_fingerprint.strip()}, received {fingerprint}"
        )
    return host_key, fingerprint


def resolve_private_key(credential_ref: str) -> Path:
    from .ssh_credentials import managed_key_path

    managed = managed_key_path(credential_ref)
    if managed is not None:
        if not managed.is_file():
            raise RemoteWorkerError("托管 SSH 私钥不存在")
        return managed
    configured = os.getenv(credential_ref, "").strip()
    if not configured:
        raise RemoteWorkerError(
            f"control-plane environment variable {credential_ref} is not configured"
        )
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise RemoteWorkerError(f"SSH private key referenced by {credential_ref} does not exist")
    return path


def remote_environment_view(environment: RemoteWorkerEnvironment) -> dict[str, Any]:
    view = environment.to_dict()
    view.pop("enrollment_token_hash", None)
    from .ssh_credentials import managed_key_path
    from .ssh_tunnels import tunnel_manager

    managed = managed_key_path(environment.credential_ref)
    view["credential_configured"] = (
        managed.is_file() if managed else bool(os.getenv(environment.credential_ref, "").strip())
    )
    options = deployment_options(environment)
    view.update(options)
    view["storage_layout"] = (environment.metadata_json or {}).get("storage_layout")
    view["tunnel"] = (
        tunnel_manager.view(environment.id) if options["transport"] == "ssh_tunnel" else None
    )
    if view["tunnel"] and view["tunnel"]["state"] != "connected" and view["status"] == "online":
        view["status"] = "offline"
    return view


def deployment_options(environment: RemoteWorkerEnvironment) -> dict:
    return {
        "transport": "direct",
        "runtime": "base",
        "service_mode": "systemd",
        "download_policy": "official",
        **(environment.metadata_json or {}).get("deployment", {}),
    }


def remote_operation_view(operation: RemoteWorkerOperation) -> dict[str, Any]:
    return operation.to_dict()


def issue_enrollment_token(
    environment: RemoteWorkerEnvironment, *, ttl: timedelta = timedelta(minutes=15)
) -> str:
    token = secrets.token_urlsafe(48)
    environment.enrollment_token_hash = hash_token(token)
    environment.enrollment_expires_at = utcnow() + ttl
    environment.enrollment_used_at = None
    return token


def _ssh_options(environment: RemoteWorkerEnvironment, known_hosts: Path) -> list[str]:
    private_key = resolve_private_key(environment.credential_ref)
    return [
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


def _remote_target(environment: RemoteWorkerEnvironment, *, scp: bool = False) -> str:
    host = environment.ssh_host
    if scp and ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{environment.ssh_user}@{host}"


def _ssh(
    environment: RemoteWorkerEnvironment,
    known_hosts: Path,
    remote_command: str,
    *,
    timeout: int = 90,
    on_line=None,
) -> subprocess.CompletedProcess[str]:
    return _run_command(
        [
            "ssh",
            *_ssh_options(environment, known_hosts),
            "-p",
            str(environment.ssh_port),
            _remote_target(environment),
            remote_command,
        ],
        timeout=timeout,
        **({"on_line": on_line} if on_line else {}),
    )


def _scp(
    environment: RemoteWorkerEnvironment,
    known_hosts: Path,
    sources: list[Path],
    destination: str,
) -> None:
    _run_command(
        [
            "scp",
            *_ssh_options(environment, known_hosts),
            "-P",
            str(environment.ssh_port),
            *[str(source) for source in sources],
            f"{_remote_target(environment, scp=True)}:{destination}",
        ],
        timeout=600,
    )


def _environment_file(environment: RemoteWorkerEnvironment, enrollment_token: str) -> str:
    values = {
        "LLM_PLATFORM_API_URL": "http://127.0.0.1:18765"
        if deployment_options(environment)["transport"] == "ssh_tunnel"
        else environment.control_plane_url,
        "LLM_PLATFORM_AGENT_TOKEN": enrollment_token,
        "LLM_PLATFORM_ENROLLMENT_ENVIRONMENT_ID": environment.id,
        "LLM_PLATFORM_AGENT_TOKEN_FILE": f"{environment.install_root}/state/agent-token",
        "LLM_PLATFORM_WORKER_NAME": environment.name,
        "LLM_PLATFORM_WORKER_PROFILE": environment.profile,
        "LLM_PLATFORM_WORKER_WORKSPACE_ROOT": environment.workspace_root,
        "LLM_PLATFORM_WORKER_PLUGIN_ROOT": f"{environment.workspace_root}/plugins",
        "LLM_PLATFORM_WORKER_RELEASE_ROOT": f"{environment.workspace_root}/project-releases",
        "HF_HOME": f"{environment.workspace_root}/.cache/huggingface",
        "HF_ENDPOINT": "https://huggingface.co",
        "PIP_INDEX_URL": "https://pypi.org/simple",
        "XDG_CACHE_HOME": f"{environment.workspace_root}/.cache",
        "PIP_CACHE_DIR": f"{environment.workspace_root}/.cache/pip",
        "UV_CACHE_DIR": f"{environment.workspace_root}/.cache/uv",
        "TORCH_HOME": f"{environment.workspace_root}/.cache/torch",
        "TRITON_CACHE_DIR": f"{environment.workspace_root}/.cache/triton",
        "TMPDIR": f"{environment.workspace_root}/.tmp",
        "LLM_PLATFORM_WORKER_LABELS": json.dumps(
            {**(environment.labels or {}), "environment": "remote-ssh"},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    if deployment_options(environment)["download_policy"] == "domestic":
        values.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1",
                      PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple")
    lines: list[str] = []
    for key, value in values.items():
        if "\n" in value or "\r" in value or "\x00" in value:
            raise RemoteWorkerError(f"invalid newline in Worker environment value: {key}")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{escaped}"')
    return "\n".join(lines) + "\n"


def _release_id(environment: RemoteWorkerEnvironment, wheel: Path) -> str:
    digest = hashlib.sha256(
        wheel.read_bytes()
        + _BOOTSTRAP_SCRIPT.read_bytes()
        + (_REPO_ROOT / "scripts/remote-worker-supervisor.py").read_bytes()
        + json.dumps(deployment_options(environment), sort_keys=True).encode()
    ).hexdigest()[:16]
    label = _RELEASE_COMPONENT.sub("-", environment.desired_release).strip("-._") or "source"
    return f"{label[:72]}-{digest}"


def _build_wheel(destination: Path) -> Path:
    result = _run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--wheel-dir",
            str(destination),
            str(_REPO_ROOT),
        ],
        timeout=600,
    )
    wheels = sorted(destination.glob("*.whl"), key=lambda path: path.stat().st_mtime_ns)
    if not wheels:
        raise RemoteWorkerError(_redact(result.stdout) or "Worker wheel build produced no output")
    return wheels[-1]


def _operation_phase(
    session: Session,
    operation: RemoteWorkerOperation,
    phase: str,
    message: str,
) -> None:
    operation.phase = phase
    operation.logs = [*(operation.logs or []), _redact(message)][-200:]
    session.commit()


def run_remote_worker_operation(operation_id: str) -> None:
    with SessionLocal() as session:
        operation = session.get(RemoteWorkerOperation, operation_id)
        if operation is None:
            return
        environment = session.get(RemoteWorkerEnvironment, operation.environment_id)
        if environment is None:
            return
        operation.status = "running"
        operation.started_at = utcnow()
        if operation.kind != "probe" and environment.worker_id:
            operation.phase = "waiting_for_idle"
            operation.logs = [*(operation.logs or []), "等待当前任务完成；期间不领取新任务，保留当前计算。"][-200:]
            session.commit()
            while session.scalar(select(ResourceLease.id).where(
                ResourceLease.worker_id == environment.worker_id,
                ResourceLease.released_at.is_(None),
            ).limit(1)):
                session.rollback()
                time.sleep(3)
        environment.status = "provisioning" if operation.kind != "probe" else environment.status
        session.commit()
        remote_staging = f"/tmp/forgelm-bootstrap-{uuid4().hex}"
        try:
            with tempfile.TemporaryDirectory(prefix="forgelm-remote-") as raw_temp:
                temp_root = Path(raw_temp)
                known_hosts = temp_root / "known_hosts"
                known_hosts.write_text(environment.host_key + "\n", encoding="utf-8")
                _operation_phase(session, operation, "probing", "正在验证 SSH 与远程前置条件")
                probe = _ssh(
                    environment,
                    known_hosts,
                    shlex.join(
                        [
                            "bash",
                            "-lc",
                            'set -e; test "$(uname -s)" = Linux; command -v bash; command -v flock; '
                            "command -v python3 || test -x /root/miniconda3/bin/python; "
                            "id; if test -d /run/systemd/system; then echo systemd=available; else echo systemd=unavailable; fi; "
                            "if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; fi; "
                            'df -Pk /; echo "Python 3.12 and venv/pip will be prepared by the installer"; '
                            'echo FORGELM_MOUNTS_BEGIN; cat /proc/self/mountinfo; echo FORGELM_MOUNTS_END',
                        ]
                    ),
                    timeout=45,
                )
                from llm_platform_core.storage_layout import storage_layout

                probe_output = probe.stdout or ""
                mountinfo = probe_output.partition("FORGELM_MOUNTS_BEGIN\n")[2].partition("FORGELM_MOUNTS_END")[0]
                layout = storage_layout(environment.workspace_root, mountinfo)
                if operation.kind != "probe":
                    environment.workspace_root = layout["workspace_root"]
                environment.metadata_json = {**(environment.metadata_json or {}), "storage_layout": layout}
                remote_staging = f'{layout["temporary_root"]}/forgelm-bootstrap-{uuid4().hex}'
                operation.logs = [
                    *(operation.logs or []),
                    _redact(probe_output.partition("FORGELM_MOUNTS_BEGIN")[0] or "远程主机前置条件通过"),
                    "存储识别: " + json.dumps(layout, ensure_ascii=False),
                ][-200:]
                if operation.kind == "probe":
                    operation.status = "succeeded"
                    operation.phase = "completed"
                    operation.result = {
                        "host_key_fingerprint": environment.host_key_fingerprint,
                        "probe": "passed",
                        "storage_layout": layout,
                    }
                    operation.ended_at = utcnow()
                    session.commit()
                    return

                _operation_phase(session, operation, "building_release", "正在构建 Worker release")
                wheel_root = temp_root / "wheel"
                wheel_root.mkdir()
                wheel = _build_wheel(wheel_root)
                release_id = _release_id(environment, wheel)
                options = deployment_options(environment)
                enrollment_token = ""
                environment_path = temp_root / "worker.env"
                from .ssh_credentials import protect_path

                environment_path.touch(mode=0o600)
                protect_path(environment_path)
                environment_path.write_text(
                    _environment_file(environment, enrollment_token), encoding="utf-8"
                )
                session.commit()

                _operation_phase(session, operation, "uploading", "正在上传固定引导脚本与 release")
                _ssh(
                    environment,
                    known_hosts,
                    shlex.join(["install", "-d", "-m", "700", remote_staging]),
                )
                _scp(
                    environment,
                    known_hosts,
                    [
                        wheel,
                        _BOOTSTRAP_SCRIPT,
                        environment_path,
                        _REPO_ROOT / "scripts/remote-worker-supervisor.py",
                    ],
                    f"{remote_staging}/",
                )
                remote_wheel = f"{remote_staging}/{wheel.name}"
                remote_script = f"{remote_staging}/{_BOOTSTRAP_SCRIPT.name}"
                remote_environment = f"{remote_staging}/{environment_path.name}"
                _operation_phase(session, operation, "installing", "正在原子安装并启动 Worker 服务")
                install_command = shlex.join(
                    [
                        "bash",
                        remote_script,
                        "--wheel",
                        remote_wheel,
                        "--environment-file",
                        remote_environment,
                        "--install-root",
                        environment.install_root,
                        "--workspace-root",
                        environment.workspace_root,
                        "--release-id",
                        release_id,
                        "--service-user",
                        environment.ssh_user,
                        "--service-name",
                        "forgelm-worker",
                        "--runtime",
                        options["runtime"],
                        "--download-policy",
                        options["download_policy"],
                        "--service-mode",
                        options["service_mode"],
                        "--supervisor",
                        f"{remote_staging}/remote-worker-supervisor.py",
                        "--requirements-dir",
                        remote_staging,
                    ]
                )
                if environment.ssh_user != "root" and options["service_mode"] != "process":
                    if options["service_mode"] == "systemd":
                        install_command = "sudo -n " + install_command
                    else:
                        install_command = shlex.join(
                            [
                                "bash",
                                "-lc",
                                "if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then exec sudo -n "
                                + install_command
                                + ' "$@"; else exec '
                                + install_command
                                + ' "$@"; fi',
                                "forgelm-install",
                            ]
                        )
                if options["transport"] == "ssh_tunnel":
                    from .ssh_tunnels import tunnel_manager

                    _operation_phase(
                        session, operation, "connecting_tunnel", "正在建立 SSH 反向隧道"
                    )
                    tunnel_manager.ensure(environment, wait=25)
                _operation_phase(session, operation, "installing", "正在准备 Python 和运行依赖")

                def log_install(line):
                    operation.logs = [*(operation.logs or []), line][-200:]
                    session.commit()

                _ssh(
                    environment,
                    known_hosts,
                    install_command + " --phase prepare",
                    timeout=7200,
                    on_line=log_install,
                )
                # Mint the short-lived token AFTER potentially lengthy dependency installation.
                session.refresh(environment)
                enrollment_token = issue_enrollment_token(environment)
                environment_path.write_text(
                    _environment_file(environment, enrollment_token), encoding="utf-8"
                )
                session.commit()
                _scp(environment, known_hosts, [environment_path], f"{remote_staging}/")
                _operation_phase(session, operation, "activating", "环境已校验，正在启动 Worker")
                if environment.worker_id and session.scalar(
                    select(ResourceLease.id)
                    .where(
                        ResourceLease.worker_id == environment.worker_id,
                        ResourceLease.released_at.is_(None),
                    )
                    .limit(1)
                ):
                    raise RemoteWorkerError(
                        "Worker 仍有任务租约，已保留构建结果；请结束任务后再修复"
                    )
                installed = _ssh(
                    environment,
                    known_hosts,
                    install_command + " --phase activate",
                    timeout=120,
                    on_line=log_install,
                )
                _ssh(
                    environment,
                    known_hosts,
                    shlex.join(["rm", "-rf", "--", remote_staging]),
                )
                session.refresh(environment)
                environment.deployed_release = release_id
                operation.status = "succeeded"
                operation.phase = "waiting_for_agent"
                operation.logs = [
                    *(operation.logs or []),
                    _redact(installed.stdout or "Worker 服务已启动，等待 Agent 注册"),
                ][-200:]
                operation.result = {
                    "release_id": release_id,
                    "waiting_for_agent": True,
                }
                operation.ended_at = utcnow()
                session.commit()
        except Exception as exc:
            try:
                with tempfile.TemporaryDirectory(prefix="forgelm-remote-cleanup-") as cleanup:
                    cleanup_hosts = Path(cleanup) / "known_hosts"
                    cleanup_hosts.write_text(environment.host_key + "\n", encoding="utf-8")
                    _ssh(
                        environment,
                        cleanup_hosts,
                        shlex.join(["rm", "-rf", "--", remote_staging]),
                    )
            except Exception:
                # Preserve the original provisioning failure. The random staging
                # directory is mode 0700 and can be removed by a later repair.
                pass
            session.rollback()
            operation = session.get(RemoteWorkerOperation, operation_id)
            environment = (
                session.get(RemoteWorkerEnvironment, operation.environment_id)
                if operation is not None
                else None
            )
            error = {"code": "remote_worker_operation_failed", "message": _redact(str(exc))}
            if operation is not None:
                operation.status = "failed"
                operation.phase = "failed"
                operation.error = error
                operation.ended_at = utcnow()
            if environment is not None:
                environment.status = "failed"
                environment.last_error = error
            session.commit()


def start_remote_worker_operation(operation_id: str) -> None:
    threading.Thread(
        target=run_remote_worker_operation,
        args=(operation_id,),
        name=f"remote-worker-{operation_id[:8]}",
        daemon=True,
    ).start()


def recover_interrupted_remote_operations() -> None:
    """SSH subprocesses are not a durable job queue. Make interrupted work retryable."""
    with SessionLocal() as session:
        for operation in session.scalars(
            select(RemoteWorkerOperation).where(
                RemoteWorkerOperation.status.in_({"queued", "running"})
            )
        ).all():
            operation.status = "failed"
            operation.phase = "interrupted"
            operation.error = {
                "code": "control_plane_restarted",
                "message": "控制面重启中断了部署，请重新初始化或修复；已完成的环境构建可复用",
            }
            operation.ended_at = utcnow()
            environment = session.get(RemoteWorkerEnvironment, operation.environment_id)
            if environment:
                environment.status = "failed"
                environment.last_error = operation.error
        session.commit()
