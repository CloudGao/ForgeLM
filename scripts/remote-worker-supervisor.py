"""Linux standalone Worker supervisor. One process per installation, independent of SSH."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def worker_environment(root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for line in (root / "worker.env").read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        environment[key] = shlex.split(value)[0]
    environment["PATH"] = (
        str(root / "current/venv/bin")
        + ":/usr/local/cuda/bin:"
        + environment.get("PATH", "/usr/bin:/bin")
    )
    return environment


def main():
    root = Path(sys.argv[1]).resolve()
    state = root / "state"
    state.mkdir(exist_ok=True)
    with (state / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        stop = False

        def stopping(_signal, _frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, stopping)
        signal.signal(signal.SIGINT, stopping)
        (state / "supervisor.pid").write_text(str(os.getpid()))
        (state / "reload").unlink(missing_ok=True)
        try:
            while not stop:
                env = worker_environment(root)
                with (state / "worker.log").open("ab") as output:
                    child = subprocess.Popen(
                        [str(root / "current/venv/bin/python"), "-m", "llm_platform_worker.main"],
                        env=env,
                        cwd=env["LLM_PLATFORM_WORKER_WORKSPACE_ROOT"],
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                    )
                    (state / "worker.pid").write_text(str(child.pid))
                    while child.poll() is None and not stop:
                        if (state / "reload").exists():
                            (state / "reload").unlink(missing_ok=True)
                            break
                        time.sleep(1)
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    (state / "last-exit.json").write_text(
                        json.dumps({"exit_code": child.returncode, "time": time.time()})
                    )
                if not stop:
                    time.sleep(3)
        finally:
            (state / "supervisor.pid").unlink(missing_ok=True)
            (state / "worker.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
