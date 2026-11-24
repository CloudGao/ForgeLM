from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from llm_platform.services import owned_process


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_child_is_terminated_when_owner_is_forcibly_killed():
    api_root = str(Path(owned_process.__file__).resolve().parents[2])
    script = (
        f"import sys,time;sys.path.insert(0,{api_root!r});"
        "from llm_platform.services.owned_process import start_owned_process;"
        "child=start_owned_process([sys.executable,'-c','import time;time.sleep(60)'],"
        "creationflags=0x08000000);print(child.pid,flush=True);time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    child = None
    try:
        child = psutil.Process(int(parent.stdout.readline()))
        assert child.is_running()
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while child.is_running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not child.is_running()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child is not None and child.is_running():
            child.kill()
        parent.stdout.close()
