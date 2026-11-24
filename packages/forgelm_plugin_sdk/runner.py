"""Trusted host shim; components load only inside their installed environment."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import signal
import sys
import threading
from pathlib import Path

from . import Cancelled, Context, Paused, Request, Result
from .package import json_object
from .validation import defaults, validate_schema_instance


def load_component(component: dict):
    dist = importlib.metadata.distribution(component["distribution"])
    matches = [e for e in dist.entry_points if e.group == "forgelm.components.v1" and e.name == component["entryPoint"]["name"]]
    if len(matches) != 1:
        raise ValueError("Component entry point must exist exactly once in its distribution")
    instance = matches[0].load()()
    for name in ("probe", "validate"):
        if not callable(getattr(instance, name, None)):
            raise ValueError(f"Component lacks {name}()")
    return instance


def control(context: Context):
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if command.get("command") in {"cancel", "shutdown"}:
                context.cancelled.set()
            elif command.get("command") == "pause":
                context.pause_requested.set()
            elif command.get("command") == "ack" and type(command.get("seq")) is int:
                context.acknowledge(command["seq"])
        except (ValueError, TypeError):
            continue


def execute(raw: dict, stream=None) -> int:
    context = Context(raw, stream)
    if raw.get("control_stdin"):
        threading.Thread(target=control, args=(context,), daemon=True).start()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            plugin = load_component(raw["component"])
            kinds = raw.get("execution_kinds", [raw.get("node", {}).get("execution", {}).get("kind", "task")])
            required = {"run"} if "task" in kinds else set()
            if "service" in kinds:
                required.update({"start", "health", "generate", "drain", "shutdown"})
            for method in required:
                if not callable(getattr(plugin, method, None)):
                    raise ValueError(f"Component lacks {method}()")
            probe = plugin.probe(context)
            if not isinstance(probe, dict) or probe.get("ready") is not True:
                raise ValueError(f"Component probe failed: {probe}")
            if raw.get("mode") == "probe":
                context.emit("result", {"probe": probe})
                return 0
            config = defaults(raw.get("config", {}), raw["node"]["config_schema"])
            validate_schema_instance(config, raw["node"]["config_schema"])
            request = Request(config, raw.get("inputs", {}), raw.get("resources", {}), raw["node"]["id"], raw.get("resume", {}))
            issues = plugin.validate(request, context)
            if not isinstance(issues, list) or any(i.get("severity") == "error" for i in issues):
                raise ValueError(f"Component validation failed: {issues}")
            context.raise_if_cancelled()
            if raw.get("mode") == "service":
                from .serve import serve
                serve(plugin, {**raw, "config": config}, context)
                context.emit("result", {"outputs": [], "service": "stopped"})
                return 0
            if not raw["node"].get("training"):
                context.execution(config, [], probe=probe)
            result = plugin.run(request, context)
            context.raise_if_cancelled()
            if not isinstance(result, Result):
                raise ValueError("run() must return Result")
            outputs = [context.output(item) for item in result.outputs]
            declared = {p["name"]: p for p in raw["node"]["outputs"]}
            ports = [o["port"] for o in outputs]
            if len(ports) != len(set(ports)) or set(ports) - declared.keys():
                raise ValueError("Unexpected/duplicate output port")
            for name, port in declared.items():
                if port["required"] and name not in ports:
                    raise ValueError(f"Missing output {name}")
            if any(o["format"] != declared[o["port"]]["format"] for o in outputs):
                raise ValueError("Output format mismatch")
            context.emit("result", {"outputs": outputs, "metadata": result.metadata})
            return 0
    except Paused:
        context.emit("result", {"controlled_state": "paused", "outputs": []})
        return 0
    except Cancelled as exc:
        context.emit("error", {"code": "cancelled", "message": str(exc), "retryable": False})
        return 2
    except Exception as exc:
        context.emit("error", {"code": "component_failed", "message": str(exc), "retryable": False})
        return 1


def main():
    parent = int(os.getenv("FORGELM_PARENT_PID", "0"))
    if parent and os.name != "nt":
        import time
        def watch_parent():
            while True:
                if os.getppid() != parent:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                time.sleep(0.25)
        threading.Thread(target=watch_parent, daemon=True).start()
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    raw = json_object(Path(args.request).read_bytes())
    raw["control_stdin"] = True
    raise SystemExit(execute(raw))


if __name__ == "__main__":
    main()
