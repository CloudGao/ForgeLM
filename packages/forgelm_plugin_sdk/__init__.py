"""Public process-local API for ForgeLM suite components."""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__version__ = "1.1.0"


class Cancelled(Exception):
    pass


class Paused(Exception):
    pass


@dataclass(frozen=True)
class Output:
    port: str
    path: Path
    format: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    outputs: list[Output] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    config: dict
    inputs: dict
    resources: dict
    node_id: str
    resume: dict = field(default_factory=dict)


class Context:
    def __init__(self, request: dict, stream=None):
        self.output_dir = Path(request["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resources = request.get("resources", {})
        self._inputs = request.get("inputs", {})
        self._run_id = request.get("run_id", "probe")
        self._stream = stream or sys.stdout
        self._seq = 0
        self._lock = threading.Lock()
        self.cancelled = threading.Event()
        self.pause_requested = threading.Event()
        self.training_contract = request.get("node", {}).get("training")
        self.previous_resolution = request.get("previous_resolution") or {}
        self._checkpoint_paths = set()
        self._execution = {}
        self._require_ack = bool(request.get("require_execution_ack"))
        self._ack = threading.Condition()
        self._acknowledged = set()

    def acknowledge(self, sequence: int):
        with self._ack:
            self._acknowledged.add(sequence)
            self._ack.notify_all()

    def _wait_for_ack(self, sequence: int):
        if not self._require_ack:
            return
        deadline = time.monotonic() + 300
        with self._ack:
            while sequence not in self._acknowledged:
                self.raise_if_cancelled()
                if time.monotonic() > deadline:
                    raise TimeoutError("Host did not persist the required training record")
                self._ack.wait(0.25)
            self._acknowledged.remove(sequence)

    def execution(self, config: dict, backends: list[dict], **metadata) -> None:
        if self._execution:
            raise ValueError("Execution resolution is immutable")
        self._execution = {"resolved_config": config, "backends": backends, **metadata}
        self._wait_for_ack(self.emit("training.resolved", self._execution))

    def input_path(self, name: str) -> Path:
        return Path(self._inputs[name]["path"]).resolve()

    def materialize_input(self, name: str) -> Path:
        from .artifacts import unpack
        from .package import identity
        destination = self.output_dir.parent / "suite-materialized" / identity(name)
        if not destination.exists():
            unpack(self.input_path(name), destination)
        return destination

    def emit(self, event: str, data: dict) -> int:
        with self._lock:
            self._seq += 1
            value = json.dumps({"protocol": "forgelm.plugin.v1", "run_id": self._run_id,
                                "seq": self._seq, "event": event, "data": data}, ensure_ascii=False, allow_nan=False)
            if len(value.encode()) > 1024 * 1024:
                raise ValueError("Plugin event exceeds 1 MiB")
            self._stream.write(value + "\n")
            self._stream.flush()
            return self._seq

    def log(self, message: str, level: str = "info") -> None:
        self.emit("log", {"message": message, "level": level})

    def metric(self, name: str, value: Any, step: int = 0) -> None:
        self.emit("metric", {"name": name, "value": value, "step": step})

    def progress(self, phase: str, completed: int, total: int | None = None) -> None:
        self.emit("progress", {"phase": phase, "completed": completed, "total": total})

    def raise_if_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled("Execution cancelled")

    def output(self, output: Output) -> dict:
        path = Path(output.path).resolve()
        relative = path.relative_to(self.output_dir)
        if path.is_symlink() or not path.exists():
            raise ValueError("Output does not exist or is a link")
        return {"port": output.port, "path": relative.as_posix(), "format": output.format,
                "metadata": output.metadata}

    def checkpoint(self, path: Path, metadata: dict) -> None:
        if not self.training_contract:
            raise ValueError("Node has no training checkpoint contract")
        output = self.output(Output("checkpoint", path, self.training_contract["checkpoint_format"], metadata))
        if output["path"] in self._checkpoint_paths:
            raise ValueError("Checkpoint paths must be unique and never reused")
        self._checkpoint_paths.add(output["path"])
        self._wait_for_ack(self.emit("training.checkpoint", output))
