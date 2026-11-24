from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _load_spec() -> tuple[dict[str, Any], Path]:
    spec_path = Path(os.environ["LLM_PLATFORM_TASK_SPEC"]).resolve()
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    task = value.get("task") or {}
    attempt_dir = Path(os.environ["LLM_PLATFORM_ATTEMPT_DIR"]).resolve()
    if not attempt_dir.is_dir():
        raise ValueError("Worker attempt directory does not exist")
    return task, attempt_dir


def _files(root: Path) -> list[Path]:
    values: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Release source contains a symbolic link: {path.relative_to(root)}")
        if path.is_file():
            values.append(path)
    return sorted(values, key=lambda item: item.relative_to(root).as_posix())


def _tree_manifest(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total = 0
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                file_digest.update(chunk)
                size += len(chunk)
        file_hash = file_digest.hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        records.append({"path": relative, "size_bytes": size, "sha256": file_hash})
        total += size
    return records, digest.hexdigest(), total


def _microlm_smoke(output_dir: Path, inference_format: str) -> dict[str, Any]:
    import torch
    from microlm_306m_v3 import MicroLMForCausalLM

    model = MicroLMForCausalLM.from_pretrained(output_dir, map_location="cpu").eval()
    token_id = int(getattr(model.config, "eos_token_id", 2) or 2)
    with torch.inference_mode():
        output = model(torch.tensor([[token_id]], dtype=torch.long), use_cache=True)
    if output.logits is None or output.logits.shape[-1] != model.config.vocab_size:
        raise RuntimeError("Native MicroLM smoke forward returned an invalid logits shape")
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Native MicroLM smoke forward returned non-finite logits")
    return {
        "status": "passed",
        "inference_format": inference_format,
        "generated_token_id": int(output.logits[:, -1, :].argmax(dim=-1).item()),
    }


def main() -> None:
    task, attempt_dir = _load_spec()
    payload = task.get("payload") or {}
    source_snapshot = payload.get("model") or {}
    source = Path(str(source_snapshot.get("runtime_source") or "")).resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Native exporter requires a materialized model directory with config.json")
    if not any(source.rglob("*.safetensors")):
        raise ValueError("Native exporter requires SafeTensors model weights")
    config = payload.get("config") or {}
    output_dir = (attempt_dir / "release-runtime").resolve()
    output_dir.relative_to(attempt_dir)
    if output_dir.exists():
        raise FileExistsError("Release runtime output directory already exists")

    started = time.monotonic()
    _emit("progress", phase="validating_source", progress=0.05)
    _files(source)
    _emit("progress", phase="exporting", progress=0.2)
    shutil.copytree(source, output_dir, symlinks=False)
    records, payload_tree_hash, total_bytes = _tree_manifest(output_dir)
    metadata = source_snapshot.get("metadata") or {}
    inference = metadata.get("inference") if isinstance(metadata.get("inference"), dict) else {}
    source_format = str(source_snapshot.get("format") or inference.get("format") or "")
    inference_format = str(inference.get("format") or source_format)
    smoke: dict[str, Any]
    if bool(config.get("verify_smoke", True)) and inference_format.startswith("microlm-"):
        _emit("progress", phase="smoke_testing", progress=0.8)
        smoke = _microlm_smoke(output_dir, inference_format)
    else:
        smoke = {
            "status": "not_supported",
            "reason": "当前 Native Exporter 只对 MicroLM 格式执行运行时烟雾检查",
        }
    runtime_manifest = {
        "schema": "forgelm-release-runtime-v1",
        "runtime_format": "forgelm-native-model-directory-v1",
        "source_format": source_format,
        "inference_format": inference_format,
        "source_artifact_id": source_snapshot.get("artifact_id"),
        "source_artifact_content_hash": source_snapshot.get("artifact_content_hash"),
        "payload_tree_hash": payload_tree_hash,
        "payload_size_bytes": total_bytes,
        "files": records,
        "smoke_test": smoke,
    }
    (output_dir / "forgelm-runtime.json").write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _emit("progress", phase="registering_artifact", progress=0.95)
    _emit(
        "result",
        outputs=[
            {
                "path": str(output_dir),
                "kind": "release_runtime",
                "ref": "release_runtime",
                "media_type": "application/gzip",
                "metadata": {
                    "format": "forgelm-native-model-directory-v1",
                    "source_format": source_format,
                    "inference": {"format": inference_format},
                    "payload_tree_hash": payload_tree_hash,
                    "payload_size_bytes": total_bytes,
                },
            }
        ],
        runtime={
            "format": "forgelm-native-model-directory-v1",
            "source_format": source_format,
            "inference_format": inference_format,
            "payload_tree_hash": payload_tree_hash,
            "payload_size_bytes": total_bytes,
            "file_count": len(records) + 1,
            "smoke_test": smoke,
            "duration_seconds": round(time.monotonic() - started, 3),
        },
        validation_report={"status": "passed", "smoke_test": smoke},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
