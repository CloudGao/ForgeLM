"""Standalone Linux smoke test: real venv/pip, a minimal local Worker wheel, no network/GPU.

Run: python3 tests/integration/remote_bootstrap_smoke.py
"""

from __future__ import annotations

import json
import os
import pwd
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


def wait_for(check, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError("smoke check timed out")


def main():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="forgelm-bootstrap-smoke-") as temporary:
        root = Path(temporary)
        install = root / "runtime"
        workspace = root / "workspace"
        wheel = root / "smoke_worker-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            files = {
                "llm_platform_worker/__init__.py": "",
                "llm_platform_worker/main.py": """import json, os, time
from pathlib import Path
if __name__ == "__main__":
    with Path("starts.jsonl").open("a") as f:
        f.write(json.dumps({"pid": os.getpid(), "name": os.environ["LLM_PLATFORM_WORKER_NAME"]}) + "\\n")
    while True: time.sleep(1)
""",
                "smoke_worker-0.1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: smoke-worker\nVersion: 0.1.0\n",
                "smoke_worker-0.1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: smoke\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            }
            files["smoke_worker-0.1.0.dist-info/RECORD"] = (
                "".join(f"{name},,\n" for name in files) + "smoke_worker-0.1.0.dist-info/RECORD,,\n"
            )
            for name, content in files.items():
                archive.writestr(name, content)
        environment = root / "worker.env"
        literal_name = "literal $(echo should-not-execute) `echo no`"
        environment.write_text(
            f'LLM_PLATFORM_WORKER_NAME="{literal_name}"\nLLM_PLATFORM_WORKER_WORKSPACE_ROOT="{workspace}"\n'
        )
        command = [
            "bash",
            str(repo / "scripts/bootstrap-remote-worker.sh"),
            "--wheel",
            str(wheel),
            "--environment-file",
            str(environment),
            "--install-root",
            str(install),
            "--workspace-root",
            str(workspace),
            "--release-id",
            "test-release",
            "--service-user",
            pwd.getpwuid(os.getuid()).pw_name,
            "--service-mode",
            "process",
            "--runtime",
            "base",
            "--supervisor",
            str(repo / "scripts/remote-worker-supervisor.py"),
            "--requirements-dir",
            str(root),
        ]
        try:
            prepared = subprocess.run(
                [*command, "--phase", "prepare"],
                env={**os.environ, "PIP_NO_INDEX": "1"},
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert prepared.returncode == 0, prepared.stdout + prepared.stderr
            assert not (install / "current").exists()
            assert (install / "releases/test-release/ready").exists()
            activated = subprocess.run(
                [*command, "--phase", "activate"], capture_output=True, text=True, timeout=20
            )
            assert activated.returncode == 0, activated.stdout + activated.stderr
            starts = workspace / "starts.jsonl"
            wait_for(lambda: starts.exists() and len(starts.read_text().splitlines()) >= 1)
            first = json.loads(starts.read_text().splitlines()[0])
            assert first["name"] == literal_name
            # Reconnection invokes start-worker again: flock prevents duplicate Workers.
            subprocess.run([str(install / "start-worker")], check=True)
            time.sleep(1.5)
            assert len(starts.read_text().splitlines()) == 1
            os.kill(first["pid"], signal.SIGKILL)
            wait_for(lambda: len(starts.read_text().splitlines()) == 2)
            # A repair reuses the same venv and reloads the supervised Worker.
            repaired = subprocess.run(
                [*command, "--phase", "activate"], capture_output=True, text=True, timeout=20
            )
            assert repaired.returncode == 0, repaired.stderr
            wait_for(lambda: len(starts.read_text().splitlines()) == 3)
            print(
                json.dumps(
                    {
                        "prepare_activate": "passed",
                        "environment_literal": "passed",
                        "duplicate_prevention": "passed",
                        "crash_restart": "passed",
                        "repair": "passed",
                    }
                )
            )
        finally:
            pid_path = install / "state/supervisor.pid"
            if pid_path.exists():
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
                wait_for(lambda: not pid_path.exists(), timeout=40)


if __name__ == "__main__":
    main()
