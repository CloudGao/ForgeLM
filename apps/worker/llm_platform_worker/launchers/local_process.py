from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from llm_platform_core.plugins import LaunchSpec, RuntimeHandle


class LocalProcessLauncher:
    def validate(self, launch_spec: LaunchSpec, allocation: dict[str, Any]) -> list:
        issues = []
        if not launch_spec.executable:
            issues.append({"code": "missing_executable", "severity": "blocker"})
        if int(allocation.get("gpus", 0)) > 1:
            issues.append({"code": "single_process_gpu_limit", "severity": "warning"})
        return issues

    def launch(self, launch_spec: LaunchSpec, allocation: dict[str, Any]) -> RuntimeHandle:
        env = os.environ.copy()
        env.update(launch_spec.env)
        gpus = allocation.get("gpu_ids")
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            [launch_spec.executable, *launch_spec.args],
            cwd=launch_spec.cwd,
            env=env,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
        return RuntimeHandle(runtime_id=str(process.pid), pid=process.pid)

    def inspect(self, handle: RuntimeHandle) -> str:
        if handle.pid is None:
            return "unknown"
        try:
            os.kill(handle.pid, 0)
        except OSError:
            return "exited"
        return "running"

    def control(self, handle: RuntimeHandle, action: str) -> dict[str, Any]:
        if handle.pid is None:
            return {"status": "not_found"}
        if action in {"safe_stop", "terminate"}:
            if os.name == "nt":
                os.kill(handle.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(handle.pid, signal.SIGTERM)
            return {"status": "requested"}
        return {"status": "unsupported"}

    def cleanup(self, handle: RuntimeHandle) -> None:
        return None
