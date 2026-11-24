from __future__ import annotations

import asyncio
import hashlib
import json

from llm_platform_worker.adapters.model_weight import ModelWeightAdapter


def test_model_weight_manifest_is_reproducible_and_transformers_ready(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    source = {
        "repo_id": "Qwen/example",
        "requested_revision": "main",
        "commit_hash": "a" * 40,
        "hf_endpoint": "https://huggingface.co",
        "selected_paths": ["model.safetensors", "tokenizer.json", "config.json"],
    }

    manifest, encoded = ModelWeightAdapter._build_manifest(tmp_path, source, "b" * 64)

    assert manifest["format"] == "forgelm-model-weight-v1"
    assert manifest["format_summary"]["transformers"] is True
    assert manifest["format_summary"]["tokenizer"] is True
    assert "lm_eval_hf" in manifest["runtime_capabilities"]
    assert hashlib.sha256(encoded).hexdigest()
    assert json.loads(encoded) == manifest


def test_model_weight_evict_requires_matching_registered_manifest(tmp_path) -> None:
    identity_hash = "c" * 64
    destination = ModelWeightAdapter._destination(identity_hash, str(tmp_path))
    destination.mkdir(parents=True)
    (destination / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
    (destination / "model.safetensors").write_bytes(b"weights")
    source = {
        "repo_id": "Qwen/example",
        "requested_revision": "main",
        "commit_hash": "a" * 40,
        "hf_endpoint": "https://huggingface.co",
        "selected_paths": ["config.json", "tokenizer.json", "model.safetensors"],
    }
    _, encoded = ModelWeightAdapter._build_manifest(destination, source, identity_hash)
    (destination / "manifest.json").write_bytes(encoded)

    async def emit(_: str, __: dict) -> None:
        return None

    result = asyncio.run(
        ModelWeightAdapter().run(
            {
                "id": "evict-task",
                "payload": {
                    "operation": "evict",
                    "identity_hash": identity_hash,
                    "location_id": "location-1",
                    "location_uri": destination.as_uri(),
                    "content_hash": hashlib.sha256(encoded).hexdigest(),
                },
                "runtime_context": {
                    "project_workspace": str(tmp_path),
                    "run_dir": str(tmp_path / "run"),
                },
            },
            emit,
        )
    )

    assert result["operation"] == "evict"
    assert not destination.exists()
