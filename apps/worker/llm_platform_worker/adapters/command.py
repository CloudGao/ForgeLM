from __future__ import annotations

import asyncio
import gzip
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..log_redaction import redact_log_value
from ..training_runtime import runtime_command_args

Emit = Callable[[str, dict[str, Any]], Awaitable[str]]
GetDesiredState = Callable[[], str]

_DEFAULT_MAX_STREAM_FRAME_BYTES = 65_536
_MAX_CONFIGURED_STREAM_FRAME_BYTES = 8 * 1024 * 1024
_STREAM_READ_BYTES = 16_384
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 10
_SAFE_PAUSE_PROTOCOL = "forgelm-command-control-file-v1"


def _safe_pause_protocol(command: dict[str, Any]) -> str:
    control = command.get("control") or {}
    if not isinstance(control, dict):
        return ""
    safe_pause = control.get("safe_pause") or {}
    if not isinstance(safe_pause, dict):
        return ""
    return str(safe_pause.get("protocol") or "")


def _write_safe_pause_request(path: Path, task_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            {
                "protocol": _SAFE_PAUSE_PROTOCOL,
                "action": "safe_pause",
                "task_id": task_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None and os.name == "nt":
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            # Commands own a new session. vLLM and similar runtimes spawn
            # engine children that must not outlive a cancelled task.
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(
            process.wait(), timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        if process.returncode is None:
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        await process.wait()
    finally:
        if os.name != "nt":
            # The group can remain after its leader exits. Do not leave GPU
            # children orphaned even when the main process stopped promptly.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class CommandAdapter:
    """Level-1 command adapter. It never invokes a shell and accepts JSONL events."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        command = (task.get("payload") or {}).get("command") or {}
        executable = command.get("executable")
        if not executable or not isinstance(executable, str):
            raise ValueError("command.executable is required")
        if executable == "__worker_python__":
            executable = sys.executable
        args = command.get("args") or []
        if not all(isinstance(arg, str) for arg in args):
            raise ValueError("command.args must contain strings")
        args = runtime_command_args(task, args)
        max_stream_frame_bytes = command.get(
            "max_output_frame_bytes", _DEFAULT_MAX_STREAM_FRAME_BYTES
        )
        if (
            isinstance(max_stream_frame_bytes, bool)
            or not isinstance(max_stream_frame_bytes, int)
            or not _DEFAULT_MAX_STREAM_FRAME_BYTES
            <= max_stream_frame_bytes
            <= _MAX_CONFIGURED_STREAM_FRAME_BYTES
        ):
            raise ValueError(
                "command.max_output_frame_bytes must be an integer between "
                f"{_DEFAULT_MAX_STREAM_FRAME_BYTES:,} and "
                f"{_MAX_CONFIGURED_STREAM_FRAME_BYTES:,}"
            )
        cwd = command.get("cwd")
        if cwd:
            cwd = str(Path(cwd).resolve())
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (command.get("env") or {}).items()})
        execution_ack = None
        if env.get("LLM_PLATFORM_RUN_DIR"):
            execution_ack = Path(env["LLM_PLATFORM_RUN_DIR"]) / "control" / "execution-ack.txt"
            execution_ack.parent.mkdir(parents=True, exist_ok=True)
            execution_ack.unlink(missing_ok=True)
            env["FORGELM_EXECUTION_ACK"] = str(execution_ack)
        safe_pause_path: Path | None = None
        if _safe_pause_protocol(command) == _SAFE_PAUSE_PROTOCOL:
            if task.get("kind") != "train" or not (task.get("payload") or {}).get(
                "training_run_id"
            ):
                raise ValueError(
                    "safe pause requires a training task with a stable training_run_id"
                )
            run_dir = Path(str(env.get("LLM_PLATFORM_RUN_DIR") or "")).resolve()
            if not env.get("LLM_PLATFORM_RUN_DIR"):
                raise ValueError("safe pause requires LLM_PLATFORM_RUN_DIR")
            safe_pause_path = run_dir / "control" / "safe-pause.json"
            safe_pause_path.parent.mkdir(parents=True, exist_ok=True)
            safe_pause_path.unlink(missing_ok=True)
            safe_pause_path.with_suffix(safe_pause_path.suffix + ".partial").unlink(
                missing_ok=True
            )
            env["LLM_PLATFORM_CONTROL_FILE"] = str(safe_pause_path)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(__import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        assert process.stdout is not None and process.stderr is not None
        lines: asyncio.Queue[tuple[str, bytes | None, BaseException | None]] = (
            asyncio.Queue(maxsize=max(2, min(256, 16 * 1024**2 // max_stream_frame_bytes)))
        )
        attempt_dir = (task.get("runtime_context") or {}).get("attempt_dir")
        journal = None
        deliver_output = True

        def stop_delivery() -> None:
            nonlocal deliver_output
            deliver_output = False
            # Let pipe readers finish journaling while the child terminates.
            # A full queue must not block process.wait() on unread pipe data.
            while not lines.empty():
                lines.get_nowait()
        if attempt_dir:
            try:
                journal_path = Path(attempt_dir) / "command-output.jsonl.gz"
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                journal = gzip.open(journal_path, "at", encoding="utf-8")
            except BaseException:
                await _terminate_process(process)
                raise

        async def enqueue(stream_name: str, frame: bytes) -> None:
            if journal is not None:
                decoded = frame.decode("utf-8", errors="replace")
                try:
                    decoded = json.dumps(redact_log_value(json.loads(decoded)), ensure_ascii=False)
                except json.JSONDecodeError:
                    decoded = redact_log_value(decoded)
                journal.write(json.dumps({
                    "received_at": datetime.now(UTC).isoformat(),
                    "stream": stream_name, "text": decoded,
                }, ensure_ascii=False) + "\n")
                journal.flush()
            if deliver_output:
                await lines.put((stream_name, frame, None))

        async def pump(stream: asyncio.StreamReader, stream_name: str) -> None:
            frame = bytearray()
            pending_carriage_return = False
            try:
                while chunk := await stream.read(_STREAM_READ_BYTES):
                    for value in chunk:
                        if pending_carriage_return:
                            pending_carriage_return = False
                            if value == 0x0A:
                                await enqueue(stream_name, bytes(frame))
                                frame.clear()
                                continue
                            frame.clear()
                        if value == 0x0D:
                            pending_carriage_return = True
                            continue
                        if value == 0x0A:
                            await enqueue(stream_name, bytes(frame))
                            frame.clear()
                            continue
                        frame.append(value)
                        if len(frame) > max_stream_frame_bytes:
                            raise ValueError(
                                f"{stream_name} output frame exceeds "
                                f"{max_stream_frame_bytes:,} bytes"
                            )
                if frame or pending_carriage_return:
                    await enqueue(stream_name, bytes(frame))
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await lines.put((stream_name, None, exc))
            finally:
                if deliver_output and not asyncio.current_task().cancelling():
                    await lines.put((stream_name, None, None))

        pumps = [
            asyncio.create_task(pump(process.stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, "stderr")),
        ]
        finished_streams = 0
        result_payload: dict[str, Any] = {}
        expected_model_signature = str(
            (((task.get("payload") or {}).get("model_specification") or {}).get(
                "config_signature"
            ))
            or ""
        )
        require_model_handshake = (
            ((task.get("payload") or {}).get("model_specification") or {}).get(
                "schema_version"
            )
            == "forgelm.model-spec.v2"
        )
        loaded_model_signature = ""
        safe_pause_requested = False

        async def handle_control(desired: str) -> str | None:
            nonlocal safe_pause_requested
            if desired == "paused" and safe_pause_path is not None:
                if not safe_pause_requested:
                    _write_safe_pause_request(safe_pause_path, str(task.get("id") or ""))
                    safe_pause_requested = True
                    await emit(
                        "control.requested",
                        {
                            "action": "safe_pause",
                            "protocol": _SAFE_PAUSE_PROTOCOL,
                            "message": "等待当前 optimizer step 完成并保存完整 Checkpoint",
                        },
                    )
                return None
            if desired in {"paused", "cancelled"}:
                stop_delivery()
                await _terminate_process(process)
                return desired
            return None

        try:
            try:
                while finished_streams < 2:
                    try:
                        stream_name, line, reader_error = await asyncio.wait_for(
                            lines.get(), timeout=1
                        )
                    except TimeoutError:
                        desired = get_desired_state() if get_desired_state else "running"
                        controlled = await handle_control(desired)
                        if controlled is not None:
                            return {"controlled_state": controlled}
                        continue
                    if reader_error is not None:
                        raise RuntimeError(
                            f"Command {stream_name} reader failed: {reader_error}"
                        ) from reader_error
                    if line is None:
                        finished_streams += 1
                        continue
                    text = line.decode("utf-8", errors="replace").rstrip()
                    desired = get_desired_state() if get_desired_state else "running"
                    if stream_name == "stdout" and text.startswith("{"):
                        try:
                            payload = json.loads(text)
                            event_type = str(payload.pop("event", "log"))
                            if event_type == "result":
                                result_payload.update(payload)
                            elif event_type == "model.loaded":
                                loaded_model_signature = str(
                                    payload.get("config_signature") or ""
                                )
                            desired = await emit(event_type, payload)
                            if event_type == "execution.resolved" and execution_ack is not None:
                                from forgelm_plugin_sdk.training import digest
                                temporary = execution_ack.with_suffix(".partial")
                                temporary.write_text(digest(payload), encoding="utf-8")
                                temporary.replace(execution_ack)
                        except json.JSONDecodeError:
                            desired = await emit(
                                "log", {"stream": stream_name, "message": text}
                            )
                    else:
                        desired = await emit(
                            "log", {"stream": stream_name, "message": text}
                        )
                    controlled = await handle_control(desired)
                    if controlled is not None:
                        return {"controlled_state": controlled}
                exit_code = await process.wait()
            except BaseException:
                stop_delivery()
                await _terminate_process(process)
                raise
        finally:
            for pump_task in pumps:
                if not pump_task.done():
                    pump_task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            if journal is not None:
                journal.close()
        if exit_code != 0:
            raise RuntimeError(f"Command exited with {exit_code}")
        if require_model_handshake and loaded_model_signature != expected_model_signature:
            raise RuntimeError(
                "Training command did not load the bound model specification "
                f"(expected signature {expected_model_signature or '<missing>'})"
            )
        if safe_pause_requested and result_payload.get("controlled_state") not in {
            None,
            "paused",
        }:
            raise RuntimeError("training command returned an invalid safe-pause state")
        if (
            safe_pause_requested
            and result_payload.get("controlled_state") is None
            and not result_payload.get("outputs")
        ):
            raise RuntimeError(
                "training command exited without acknowledging the safe-pause request"
            )
        return {
            "exit_code": exit_code,
            "command_result": "completed",
            **result_payload,
        }
