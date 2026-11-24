from __future__ import annotations

import asyncio
import gzip
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from llm_platform_worker.adapters.command import CommandAdapter


def test_command_journal_survives_full_queue_cancel_and_redacts_secrets(tmp_path, monkeypatch):
    original_queue = asyncio.Queue
    queues = []

    class ObservedQueue(original_queue):
        high_water = 0

        async def put(self, value):
            await super().put(value)
            self.high_water = max(self.high_water, self.qsize())

    def queue(*args, **kwargs):
        result = ObservedQueue(*args, **kwargs)
        queues.append(result)
        return result

    monkeypatch.setattr("llm_platform_worker.adapters.command.asyncio.Queue", queue)

    async def emit(kind, payload):
        await asyncio.sleep(0.2)
        return "cancelled"

    script = "import json\nfor i in range(10000): print(json.dumps({'event':'log','password':'hide-me','message':str(i)}), flush=True)"
    task = {"runtime_context": {"attempt_dir": str(tmp_path)},
            "payload": {"command": {"executable": sys.executable, "args": ["-c", script]}}}
    result = asyncio.run(asyncio.wait_for(CommandAdapter().run(task, emit), 15))
    assert result["controlled_state"] == "cancelled"
    assert queues[0].maxsize == 256
    assert queues[0].high_water == 256
    with gzip.open(tmp_path / "command-output.jsonl.gz", "rt", encoding="utf-8") as journal:
        rows = [json.loads(line) for line in journal]
    assert len(rows) >= 256
    assert "hide-me" not in json.dumps(rows)
    assert json.loads(rows[0]["text"])["password"] == "[REDACTED]"


def test_command_journal_retains_all_delivered_lines_in_order(tmp_path):
    events = []

    async def emit(kind, payload):
        events.append(payload["message"])
        return "running"

    task = {"runtime_context": {"attempt_dir": str(tmp_path)},
            "payload": {"command": {"executable": sys.executable,
                                    "args": ["-c", "for i in range(600): print(i)"]}}}
    asyncio.run(CommandAdapter().run(task, emit))
    with gzip.open(tmp_path / "command-output.jsonl.gz", "rt", encoding="utf-8") as journal:
        rows = [json.loads(line) for line in journal]
    assert [row["text"] for row in rows] == events == [str(i) for i in range(600)]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups")
def test_command_cancel_reaps_spawned_engine_process() -> None:
    child_pid = None

    async def emit(event_type: str, payload: dict) -> str:
        nonlocal child_pid
        if event_type == "child.started":
            child_pid = payload["pid"]
            return "cancelled"
        return "running"

    script = (
        "import json, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(json.dumps({'event': 'child.started', 'pid': child.pid}), flush=True); "
        "time.sleep(60)"
    )
    task = {"payload": {"command": {"executable": sys.executable, "args": ["-c", script]}}}
    try:
        result = asyncio.run(asyncio.wait_for(CommandAdapter().run(task, emit), 10))
        assert result["controlled_state"] == "cancelled"
        assert child_pid is not None
        for _ in range(40):
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("cancelled command left its child process running")
    finally:
        if child_pid is not None:
            try:
                command_line = Path(f"/proc/{child_pid}/cmdline")
                if command_line.exists() and b"time.sleep(60)" in command_line.read_bytes():
                    os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_training_runtime_wraps_frozen_script_preserving_arguments_and_imports(tmp_path):
    (tmp_path / "local_config.py").write_text("VALUE = 370\n", encoding="utf-8")
    script = tmp_path / "train.py"
    script.write_text(
        "import json, sys\nfrom local_config import VALUE\n"
        "print(json.dumps({'event':'result','step':VALUE,'arguments':sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    events = []

    async def emit(kind, payload):
        events.append((kind, payload))
        return "running"

    task = {"kind": "train", "payload": {"command": {
        "executable": "__worker_python__", "args": [str(script), "--checkpoint", "checkpoint-370"],
        "cwd": str(tmp_path),
    }}}
    result = asyncio.run(CommandAdapter().run(task, emit))
    assert result["step"] == 370
    assert result["arguments"] == ["--checkpoint", "checkpoint-370"]
    report = next(payload for kind, payload in events if kind == "runtime.optimization.resolved")
    assert report["schema_version"] == "forgelm.training-runtime.v1"
    assert len(report["implementation_sha256"]) == 64


def test_command_adapter_runs_without_shell_and_parses_jsonl() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    event = json.dumps({"event": "progress", "progress": 0.5, "phase": "train"})
    result_event = json.dumps({"event": "result", "step": 12, "metrics": {"loss": 1.2}})
    script = (
        f"import sys; print({event!r}); print({result_event!r}); "
        "print('diagnostic', file=sys.stderr)"
    )
    task = {
        "payload": {
            "command": {
                "executable": sys.executable,
                "args": ["-c", script],
            }
        }
    }

    result = asyncio.run(CommandAdapter().run(task, emit))

    assert result == {
        "exit_code": 0,
        "command_result": "completed",
        "step": 12,
        "metrics": {"loss": 1.2},
    }
    assert ("progress", {"progress": 0.5, "phase": "train"}) in events
    assert ("log", {"stream": "stderr", "message": "diagnostic"}) in events


def test_command_adapter_requests_checkpointed_pause_without_terminating_process(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "paused"

    script = (
        "import json, os, pathlib, time; "
        "path = pathlib.Path(os.environ['LLM_PLATFORM_CONTROL_FILE']); "
        "deadline = time.time() + 4; "
        "\nwhile not path.is_file() and time.time() < deadline: time.sleep(0.02); "
        "\nrequest = json.loads(path.read_text(encoding='utf-8')); "
        "print(json.dumps({'event':'control.acknowledged','action':request['action']}), flush=True); "
        "print(json.dumps({'event':'result','controlled_state':'paused','step':7}), flush=True)"
    )
    task = {
        "id": "training-task",
        "kind": "train",
        "payload": {
            "training_run_id": "training-run",
            "command": {
                "executable": sys.executable,
                "args": ["-c", script],
                "env": {"LLM_PLATFORM_RUN_DIR": str(tmp_path)},
                "control": {
                    "safe_pause": {
                        "protocol": "forgelm-command-control-file-v1",
                    }
                },
            }
        },
    }

    result = asyncio.run(
        asyncio.wait_for(CommandAdapter().run(task, emit, lambda: "paused"), timeout=5)
    )

    assert result["controlled_state"] == "paused"
    assert result["step"] == 7
    request = json.loads(
        (tmp_path / "control" / "safe-pause.json").read_text(encoding="utf-8")
    )
    assert request == {
        "action": "safe_pause",
        "protocol": "forgelm-command-control-file-v1",
        "task_id": "training-task",
    }
    assert any(event_type == "control.requested" for event_type, _ in events)
    assert ("control.acknowledged", {"action": "safe_pause"}) in events


def test_command_adapter_coalesces_carriage_return_progress() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result_event = json.dumps({"event": "result", "step": 12})
    script = (
        "import sys; "
        "sys.stderr.write(('progress 0\\r' * 10000) + 'progress done\\n'); "
        "sys.stderr.flush(); "
        f"print({result_event!r})"
    )
    task = {
        "payload": {
            "command": {
                "executable": sys.executable,
                "args": ["-c", script],
            }
        }
    }

    result = asyncio.run(
        asyncio.wait_for(CommandAdapter().run(task, emit), timeout=5)
    )

    assert result["step"] == 12
    stderr_logs = [
        payload["message"]
        for event_type, payload in events
        if event_type == "log" and payload["stream"] == "stderr"
    ]
    assert stderr_logs == ["progress done"]


def test_command_adapter_fails_fast_on_oversized_frame() -> None:
    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    script = (
        "import sys, time; "
        "sys.stderr.write('x' * 70000); "
        "sys.stderr.flush(); "
        "time.sleep(30)"
    )
    task = {
        "payload": {
            "command": {
                "executable": sys.executable,
                "args": ["-c", script],
            }
        }
    }

    with pytest.raises(RuntimeError, match="stderr output frame exceeds 65,536 bytes"):
        asyncio.run(
            asyncio.wait_for(CommandAdapter().run(task, emit), timeout=5)
        )


def test_command_adapter_allows_bounded_large_result_frame() -> None:
    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    script = (
        "import json; "
        "print(json.dumps({'event': 'result', 'blob': 'x' * 70000}))"
    )
    task = {
        "payload": {
            "command": {
                "executable": sys.executable,
                "args": ["-c", script],
                "max_output_frame_bytes": 2 * 1024 * 1024,
            }
        }
    }

    result = asyncio.run(CommandAdapter().run(task, emit))
    assert len(result["blob"]) == 70_000


def test_v2_model_contract_requires_matching_loaded_signature() -> None:
    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    task = {
        "payload": {
            "model_specification": {
                "schema_version": "forgelm.model-spec.v2",
                "config_signature": "expected-signature",
            },
            "command": {
                "executable": sys.executable,
                "args": ["-c", "print('training finished')"],
            },
        }
    }

    with pytest.raises(RuntimeError, match="did not load the bound model"):
        asyncio.run(CommandAdapter().run(task, emit))


def test_v2_model_contract_accepts_matching_loaded_signature() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    loaded = json.dumps(
        {"event": "model.loaded", "config_signature": "expected-signature"}
    )
    task = {
        "payload": {
            "model_specification": {
                "schema_version": "forgelm.model-spec.v2",
                "config_signature": "expected-signature",
            },
            "command": {
                "executable": sys.executable,
                "args": ["-c", f"print({loaded!r})"],
            },
        }
    }

    result = asyncio.run(CommandAdapter().run(task, emit))

    assert result["exit_code"] == 0
    assert ("model.loaded", {"config_signature": "expected-signature"}) in events
