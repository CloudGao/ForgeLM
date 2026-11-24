"""Minimal ForgeLM command-adapter contract example.

Replace the example loop with a training, evaluation, quantization, or inference
engine. The process communicates with ForgeLM through JSON Lines on stdout and
writes declared output artifacts under its run directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def main() -> None:
    task_spec_path = Path(os.environ["LLM_PLATFORM_TASK_SPEC"])
    run_dir = Path(os.environ["LLM_PLATFORM_RUN_DIR"])
    task_spec = json.loads(task_spec_path.read_text(encoding="utf-8"))
    task = task_spec["task"]
    emit("log", message=f"accepted {task['kind']} task {task['id']}")
    for step in range(1, 6):
        emit(
            "progress",
            progress=step / 5,
            phase="external-engine",
            step=step,
            metrics={"loss": round(2.5 / step, 4)},
        )

    output_dir = run_dir / "checkpoint-final"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "engine-result.json").write_text(
        json.dumps({"task_id": task["id"], "engine": "example"}, indent=2),
        encoding="utf-8",
    )
    artifact_kind = {
        "data_import": "dataset_snapshot",
        "tokenizer_train": "tokenizer",
        "train": "checkpoint",
        "evaluate": "evaluation_report",
        "quantize": "quantized_model",
    }.get(task["kind"], "task_output")
    emit(
        "result",
        step=5,
        tokens=1024,
        metrics={"loss": 0.5},
        outputs=[{"kind": artifact_kind, "path": str(output_dir)}],
    )


if __name__ == "__main__":
    main()
