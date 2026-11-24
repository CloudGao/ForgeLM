"""Process-scoped training runtime optimizations, independent of training stage.

Keep backend dispatch and its input/device verifiers intact. Only dependency
availability is stable for the lifetime of a training process.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

RUNTIME_PROTOCOL = "forgelm.training-runtime.v1"


def cache_backend_availability(backend: type) -> bool:
    """Cache positive AND negative dependency probes, never execution verifiers."""
    original = backend.is_available
    if getattr(original, "_forgelm_availability_cache", False):
        return False

    @functools.lru_cache(maxsize=None)
    def available(cls):
        # Use the original descriptor so inherited class methods bind correctly.
        return original.__func__(cls)

    available._forgelm_availability_cache = True
    backend.is_available = classmethod(available)
    return True


def prepare_training_runtime() -> dict:
    report = {"schema_version": RUNTIME_PROTOCOL, "optimizations": [],
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if importlib.util.find_spec("fla") is None:
        report["status"] = "not_applicable"
        return report
    # These two FLA backends import the optional tilelang package on every
    # dispatch. Do not bypass dispatch: other installed backends must still work.
    for name, class_name in (("fla.ops.common.backends.tilelang", "TileLangBackend"),
                             ("fla.ops.kda.backends.tilelang", "KDATileLangBackend")):
        try:
            module = importlib.import_module(name)
            backend = getattr(module, class_name)
            cache_backend_availability(backend)
            available = bool(backend.is_available())
        except (ImportError, AttributeError) as exc:
            report["optimizations"].append({"module": name, "status": "unsupported", "reason": str(exc)})
        else:
            report["optimizations"].append({"module": name, "status": "applied", "available": available,
                                            "policy": "process_dependency_availability_cache"})
    report["status"] = "applied" if any(x["status"] == "applied" for x in report["optimizations"]) else "unsupported"
    return report


def runtime_command_args(task: dict, args: list[str]) -> list[str]:
    payload = task.get("payload") or {}
    command = payload.get("command") or {}
    # Standard Python script entrypoints only. Arbitrary executables, -m/-c and
    # shell commands keep their own runtime contract.
    if (task.get("kind") not in {"train", "training_performance"}
            or command.get("executable") != "__worker_python__"
            or not args or not args[0].endswith(".py")
            or (command.get("env") or {}).get("FORGELM_TRAINING_RUNTIME", "auto") == "off"):
        return args
    return ["-m", "llm_platform_worker.training_runtime", *args]


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    # Match `python script.py`: local frozen source precedes installed packages.
    sys.path[0] = str(script.parent)
    report = prepare_training_runtime()
    print(json.dumps({"event": "runtime.optimization.resolved", **report}), flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
