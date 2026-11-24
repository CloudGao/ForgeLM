"""Build a portable, immutable formal-training input artifact from local inputs."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def pack_training_bundle(task_spec: Path, model_config: Path, output: Path, validation_file: Path | None = None):
    spec = json.loads(task_spec.read_text(encoding="utf-8"))
    source = (spec.get("task") or spec).get("payload", spec)
    allowed = {"stages", "tokenizer", "tokenized_data_inputs", "training_data_view_inputs", "tokenized_corpora",
               "supervision_training_view_inputs", "validation_data_input", "validation_set", "reference_checkpoint",
               "objective_kind", "selected_stage_indexes", "learning_rate_schedules"}
    with tempfile.TemporaryDirectory(prefix="forgelm-training-bundle-") as temporary:
        root = Path(temporary)
        paths = {}
        def freeze(value):
            if isinstance(value, list):
                return [freeze(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = {}
            for key, item in value.items():
                if key == "runtime_path":
                    path = Path(item).resolve(strict=True)
                    if path not in paths:
                        relative = f"inputs/{len(paths)}"
                        paths[path] = relative
                        destination = root / relative
                        destination.parent.mkdir(exist_ok=True)
                        if path.is_dir():
                            if any(p.is_symlink() for p in path.rglob("*")):
                                raise ValueError("Training input directories cannot contain links")
                            shutil.copytree(path, destination)
                        else:
                            shutil.copy2(path, destination)
                    result[key] = paths[path]
                else:
                    result[key] = freeze(item)
            return result
        payload = freeze({key: value for key, value in source.items() if key in allowed})
        manifest = {"schema": "forgelm.training-input-bundle.v1", "payload": payload}
        if validation_file:
            shutil.copy2(validation_file, root / "validation.txt")
            manifest["validation_file"] = "validation.txt"
        initial = source.get("resume_checkpoint") or {}
        if initial.get("runtime_path"):
            manifest["initial_model"] = freeze({"runtime_path": initial["runtime_path"]})["runtime_path"]
        if not validation_file and (payload.get("validation_data_input") or {}).get("format") != "forgelm-validation-token-holdout-v1" and not payload.get("supervision_training_view_inputs"):
            raise ValueError("Provide a frozen validation file or tokenized validation holdout")
        shutil.copy2(model_config, root / "model-config.json")
        (root / "training-inputs.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return output
