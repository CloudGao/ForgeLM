"""Exercise the real OpenSSH client against an isolated local SSH test server."""

from __future__ import annotations

import os
import select
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import httpx
import pytest


def test_real_reverse_forward_and_shutdown(monkeypatch, tmp_path):
    paramiko = pytest.importorskip("paramiko")
    if not shutil.which("ssh"):
        pytest.skip("OpenSSH client unavailable")
    from llm_platform.services.ssh_credentials import protect_path
    from llm_platform.services.ssh_tunnels import AgentGateway, TunnelManager
    from starlette.responses import JSONResponse

    host_key = paramiko.RSAKey.generate(2048)
    client_key = paramiko.RSAKey.generate(2048)
    path = tmp_path / "client-key"
    path.touch()
    if os.name == "nt":
        # Explicit ACEs survive /inheritance:r; protection must remove them too.
        subprocess.run(
            ["icacls", str(path), "/grant", "*S-1-1-0:(R)"],
            check=True, capture_output=True,
        )
    protect_path(path)
    client_key.write_private_key_file(str(path))
    monkeypatch.setenv("TUNNEL_TEST_KEY", str(path))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stopped = threading.Event()
    transports = []
    forwarded_listeners = []
    commands = []

    def bridge(connection, channel):
        try:
            while not stopped.is_set():
                ready, _, _ = select.select([connection, channel], [], [], 0.2)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    (channel if source is connection else connection).sendall(data)
        except (OSError, EOFError):
            pass
        finally:
            connection.close()
            channel.close()

    class Server(paramiko.ServerInterface):
        def __init__(self, transport):
            self.transport = transport
            self.channels = []

        def check_auth_publickey(self, username, key):
            return paramiko.AUTH_SUCCESSFUL if key == client_key else paramiko.AUTH_FAILED

        def get_allowed_auths(self, username):
            return "publickey"

        def check_channel_request(self, kind, chanid):
            return (
                paramiko.OPEN_SUCCEEDED
                if kind == "session"
                else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
            )

        def check_channel_exec_request(self, channel, command):
            commands.append(command.decode())
            self.channels.append(channel)
            channel.send(b"FORGELM_TUNNEL_READY\n")
            return True

        def check_port_forward_request(self, address, remote_port):
            assert address == "127.0.0.1"
            remote = socket.socket()
            remote.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                remote.bind((address, remote_port))
            except OSError:
                remote.close()
                return False
            remote.listen()
            remote.settimeout(0.2)
            forwarded_listeners.append(remote)

            def accept_forwarded():
                while self.transport.is_active() and not stopped.is_set():
                    try:
                        connection, origin = remote.accept()
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    channel = self.transport.open_forwarded_tcpip_channel(
                        origin, (address, remote_port)
                    )
                    threading.Thread(target=bridge, args=(connection, channel), daemon=True).start()
                remote.close()

            threading.Thread(target=accept_forwarded, daemon=True).start()
            return remote_port

    def serve():
        while not stopped.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            transport = paramiko.Transport(connection)
            transports.append(transport)
            transport.add_server_key(host_key)
            transport.start_server(server=Server(transport))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    manager = TunnelManager()

    async def app(scope, receive, send):
        await JSONResponse({"through_ssh": True})(scope, receive, send)

    manager.app = app
    monkeypatch.setattr(AgentGateway, "authorized", lambda *_: True)
    env = SimpleNamespace(
        id="integration",
        ssh_host="127.0.0.1",
        ssh_port=port,
        ssh_user="worker",
        credential_ref="TUNNEL_TEST_KEY",
        install_root="/tmp/test-forgelm",
        host_key=f"[127.0.0.1]:{port} {host_key.get_name()} {host_key.get_base64()}",
    )
    try:
        manager.ensure(env, wait=12)
        response = httpx.get("http://127.0.0.1:18765/api/agent/check", trust_env=False)
        assert response.json() == {"through_ssh": True}
        assert httpx.get("http://127.0.0.1:18765/api/projects", trust_env=False).status_code == 403
        assert commands and "start-worker" in commands[0]
        tunnel = manager.tunnels[env.id]
        tunnel.process.terminate()
        tunnel.process.wait(timeout=5)
        time.sleep(0.5)
        tunnel.next_retry = 0
        manager.ensure(env, wait=12)
        assert manager.view(env.id)["reconnect_count"] == 1
        assert (
            httpx.get("http://127.0.0.1:18765/api/agent/check", trust_env=False).status_code == 200
        )
    finally:
        manager.shutdown()
        stopped.set()
        for transport in transports:
            transport.close()
        for remote in forwarded_listeners:
            remote.close()
        listener.close()
        thread.join(timeout=2)
