"""Control-plane owned OpenSSH reverse tunnels with an Agent-only ASGI gateway."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import uvicorn
from sqlalchemy import select
from starlette.responses import JSONResponse

from ..database import SessionLocal
from ..models import RemoteWorkerEnvironment, WorkerNode


class AgentGateway:
    def __init__(self, app, environment_id: str):
        self.app = app
        self.environment_id = environment_id

    def authorized(self, path: str, headers: dict) -> bool:
        token = headers.get(b"x-agent-token", b"").decode("utf-8", errors="replace")
        if not token:
            return False
        digest = hashlib.sha256(token.encode()).hexdigest()
        with SessionLocal() as session:
            environment = session.get(RemoteWorkerEnvironment, self.environment_id)
            if environment is None:
                return False
            if path == "/api/agent/enroll":
                return bool(
                    environment.enrollment_token_hash
                    and hmac.compare_digest(digest, environment.enrollment_token_hash)
                )
            worker_id = headers.get(b"x-worker-id", b"").decode("utf-8", errors="replace")
            if worker_id != environment.worker_id:
                return False
            worker = session.get(WorkerNode, worker_id)
            return bool(
                worker and worker.token_hash and hmac.compare_digest(digest, worker.token_hash)
            )

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or not path.startswith("/api/agent/")
            or any(part in {".", ".."} for part in path.split("/"))
        ):
            await JSONResponse({"detail": "Agent routes only"}, status_code=403)(
                scope, receive, send
            )
            return
        if not await asyncio.to_thread(self.authorized, path, dict(scope.get("headers", []))):
            await JSONResponse({"detail": "Invalid environment credential"}, status_code=401)(
                scope, receive, send
            )
            return
        # No loopback identity or proxy-header trust leaks through this gateway.
        scope = {
            **scope,
            "client": ("remote-worker", 0),
            "headers": [
                (key, value)
                for key, value in scope.get("headers", [])
                if not key.lower().startswith(b"x-forwarded-") and key.lower() != b"forwarded"
            ],
        }
        await self.app(scope, receive, send)


class Tunnel:
    def __init__(self, app, environment):
        self.environment_id = environment.id
        self.temp = tempfile.TemporaryDirectory(prefix="forgelm-tunnel-")
        self.known_hosts = Path(self.temp.name) / "known_hosts"
        self.known_hosts.write_text(environment.host_key + "\n", encoding="utf-8")
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(
                AgentGateway(app, environment.id),
                lifespan="off",
                access_log=False,
                log_level="error",
                proxy_headers=False,
                timeout_graceful_shutdown=3,
            )
        )
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True
        )
        self.thread.start()
        self.process = None
        self.ready = threading.Event()
        self.errors = deque(maxlen=12)
        self.failures = 0
        self.next_retry = 0.0

    def start(self, environment):
        from .remote_workers import _remote_target, _ssh_options

        self.ready.clear()
        self.errors.clear()
        from .owned_process import start_owned_process

        self.process = start_owned_process(
            [
                "ssh",
                *_ssh_options(environment, self.known_hosts),
                "-T",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "GatewayPorts=no",
                "-R",
                f"127.0.0.1:18765:127.0.0.1:{self.port}",
                "-p",
                str(environment.ssh_port),
                _remote_target(environment),
                f"set -e; if test -x {shlex.quote(environment.install_root + '/start-worker')}; then "
                f"{shlex.quote(environment.install_root + '/start-worker')}; fi; "
                "printf 'FORGELM_TUNNEL_READY\\n'; exec sleep infinity",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process = self.process

        def read_output():
            from .remote_workers import _redact

            for line in process.stdout:
                if line.strip() == "FORGELM_TUNNEL_READY":
                    self.ready.set()
                else:
                    self.errors.append(_redact(line, limit=1500))
            process.stdout.close()

        threading.Thread(target=read_output, daemon=True).start()

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.socket.close()
        self.temp.cleanup()


class TunnelManager:
    def __init__(self):
        self.app = None
        self.lock = threading.RLock()
        self.tunnels: dict[str, Tunnel] = {}

    def view(self, environment_id: str) -> dict:
        with self.lock:
            tunnel = self.tunnels.get(environment_id)
            if not tunnel:
                return {"state": "stopped", "reconnect_count": 0}
            alive = tunnel.process is not None and tunnel.process.poll() is None
            return {
                "state": "connected"
                if alive and tunnel.ready.is_set()
                else "connecting"
                if alive
                else "reconnecting",
                "reconnect_count": tunnel.failures,
                "last_error": "\n".join(tunnel.errors) or None,
            }

    def ensure(self, environment, *, wait: float = 0):
        from .remote_workers import RemoteWorkerError

        with self.lock:
            if self.app is None:
                raise RemoteWorkerError("SSH 隧道管理器尚未启动")
            tunnel = self.tunnels.get(environment.id)
            if tunnel is None:
                tunnel = self.tunnels[environment.id] = Tunnel(self.app, environment)
            if tunnel.process is None or tunnel.process.poll() is not None:
                now = time.monotonic()
                if now >= tunnel.next_retry:
                    tunnel.failures += int(tunnel.process is not None)
                    tunnel.next_retry = now + min(60, 3 * 2 ** min(tunnel.failures, 5))
                    try:
                        tunnel.start(environment)
                    except Exception as exc:
                        tunnel.errors.append(
                            type(exc).__name__ + ": SSH 连接无法启动，请检查凭据和 OpenSSH"
                        )
                        if wait:
                            raise RemoteWorkerError(tunnel.errors[-1]) from exc
        if wait:
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if tunnel.process and tunnel.process.poll() is not None:
                    break
                if tunnel.ready.wait(timeout=0.1):
                    return
            raise RemoteWorkerError(
                "SSH 反向转发未建立；请检查远端 AllowTcpForwarding 和端口 18765。"
                + "\n".join(tunnel.errors)
            )

    def reconcile(self):
        from .remote_workers import deployment_options

        with SessionLocal() as session:
            environments = session.scalars(select(RemoteWorkerEnvironment)).all()
            ids = {item.id for item in environments}
            for item in environments:
                if deployment_options(item)["transport"] == "ssh_tunnel" and (
                    item.deployed_release or item.status == "provisioning"
                ):
                    try:
                        self.ensure(item)
                    except Exception:
                        # A broken endpoint must not disable maintenance of other Workers.
                        continue
        with self.lock:
            for environment_id in set(self.tunnels) - ids:
                self.tunnels.pop(environment_id).close()

    def remove(self, environment_id: str):
        with self.lock:
            tunnel = self.tunnels.pop(environment_id, None)
            if tunnel:
                tunnel.close()

    def shutdown(self):
        with self.lock:
            for tunnel in self.tunnels.values():
                tunnel.close()
            self.tunnels.clear()
            self.app = None


tunnel_manager = TunnelManager()
