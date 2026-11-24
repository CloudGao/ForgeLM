from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import psutil


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_supervisor(process: subprocess.Popen, status_path: Path) -> None:
    child_pid = None
    try:
        child_pid = json.loads(status_path.read_text(encoding="utf-8")).get("child_pid")
    except (OSError, ValueError, TypeError):
        pass

    if process.poll() is None:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if child_pid and psutil.pid_exists(child_pid):
        child = psutil.Process(child_pid)
        child.terminate()
        try:
            child.wait(timeout=5)
        except psutil.TimeoutExpired:
            child.kill()


def test_supervisor_restarts_the_control_plane_with_a_new_instance() -> None:
    root = Path(__file__).resolve().parents[2]
    state_root = Path(tempfile.mkdtemp(prefix="forge-lm-supervisor-e2e-"))
    supervisor_directory = state_root / "supervisor"
    status_path = supervisor_directory / "status.json"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_PLATFORM_API_HOST": "127.0.0.1",
            "LLM_PLATFORM_API_PORT": str(port),
            "LLM_PLATFORM_DATABASE_URL": (
                f"sqlite:///{(state_root / 'platform.db').as_posix()}"
            ),
            "LLM_PLATFORM_CONTROL_PLANE_SUPERVISOR_DIR": str(supervisor_directory),
        }
    )
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "llm_platform.supervisor"],
        cwd=root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        with httpx.Client(base_url=base_url, timeout=2) as client:
            initial = None
            for _ in range(100):
                if supervisor.poll() is not None:
                    raise AssertionError(
                        f"Supervisor exited before readiness with code {supervisor.returncode}"
                    )
                try:
                    response = client.get("/api/control-plane")
                    if response.status_code == 200 and response.json()["restart_supported"]:
                        initial = response.json()
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            assert initial is not None, "supervised API did not become ready"

            restart = client.post(
                "/api/control-plane/restart",
                json={"reason": "process-level restart test"},
            )
            assert restart.status_code == 202
            assert restart.json()["instance_id"] == initial["instance_id"]

            recovered = None
            for _ in range(200):
                try:
                    response = client.get("/api/control-plane")
                    if response.status_code == 200:
                        candidate = response.json()
                        if (
                            candidate["restart_supported"]
                            and candidate["state"] == "running"
                            and candidate["instance_id"] != initial["instance_id"]
                        ):
                            recovered = candidate
                            break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)

            assert recovered is not None, "control plane did not recover with a new instance"
            assert recovered["pid"] != initial["pid"]
            assert recovered["restart_count"] == 1
    finally:
        _stop_supervisor(supervisor, status_path)
        shutil.rmtree(state_root, ignore_errors=True)
