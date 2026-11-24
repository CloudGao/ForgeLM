from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
pytest.importorskip("transformers", reason="Optional tokenizer runtime")
from llm_platform_worker.adapters.pretokenize import PretokenizeAdapter


@pytest.mark.asyncio
async def test_pretokenize_adapter_builds_immutable_document_aware_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tokenizer:
        eos_token_id = 2

        def __len__(self) -> int:
            return 32_000

        def __call__(self, texts: list[str], **kwargs) -> dict[str, list[list[int]]]:
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [[3 + ord(character) % 20 for character in text] for text in texts]}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.pretokenize.HfTokenizerAdapter._corpus",
        lambda *_args, **_kwargs: (iter(["ab"] * 20), [], {}),
    )
    project_workspace = tmp_path / "project"
    tokenizer_path = tmp_path / "tokenizer"
    run_dir = tmp_path / "run"
    project_workspace.mkdir()
    tokenizer_path.mkdir()
    run_dir.mkdir()
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    identity_hash = "b" * 64
    task = {
        "id": "task-1",
        "payload": {
            "identity_hash": identity_hash,
            "sequence_length": 4,
            "target_tokens": 16,
            "shard_sequences": 2,
            "packing_version": "document-pack-v1",
            "data_input": {"manifest_hash": "c" * 64},
            "tokenizer": {
                "runtime_path": str(tokenizer_path),
                "content_hash": "d" * 64,
            },
        },
        "runtime_context": {
            "project_workspace": str(project_workspace),
            "run_dir": str(run_dir),
        },
    }

    first = await PretokenizeAdapter().run(task, emit)
    repeated = await PretokenizeAdapter().run(task, emit)

    output = first["outputs"][0]
    destination = (
        project_workspace
        / ".llm-platform"
        / "data-assets"
        / "tokenized"
        / identity_hash
    )
    manifest_bytes = (destination / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["sequence_count"] == 4
    assert manifest["token_count"] == 16
    assert len(manifest["shards"]) == 2
    assert manifest["boundary_bitorder"] == "little"
    assert output["storage_type"] == "external"
    assert output["content_hash"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert repeated["outputs"][0]["content_hash"] == output["content_hash"]
    assert events[-1][1]["progress"] == 1.0

    failing_task = {
        **task,
        "id": "task-2",
        "payload": {
            **task["payload"],
            "identity_hash": "e" * 64,
            "target_tokens": 400,
        },
    }
    with pytest.raises(ValueError, match="packed sequences"):
        await PretokenizeAdapter().run(failing_task, emit)
    partial = (
        project_workspace
        / ".llm-platform"
        / "data-assets"
        / "tokenized"
        / f".{failing_task['payload']['identity_hash']}.task-2.partial"
    )
    assert not partial.exists()


@pytest.mark.asyncio
async def test_pretokenize_adapter_recovers_historical_eos_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tokenizer:
        eos_token_id = None
        eos_token = None

        def __len__(self) -> int:
            return 32_000

        def get_vocab(self) -> dict[str, int]:
            return {"<unk>": 0, "<s>": 1, "</s>": 2}

        def __call__(self, texts: list[str], **_kwargs) -> dict[str, list[list[int]]]:
            return {"input_ids": [[3] for _text in texts]}

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)
            if name == "eos_token" and value == "</s>":
                object.__setattr__(self, "eos_token_id", 2)

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.pretokenize.HfTokenizerAdapter._corpus",
        lambda *_args, **_kwargs: (iter(["a"] * 8), [], {}),
    )
    project_workspace = tmp_path / "project"
    tokenizer_path = tmp_path / "tokenizer"
    run_dir = tmp_path / "run"
    project_workspace.mkdir()
    tokenizer_path.mkdir()
    run_dir.mkdir()

    async def emit(_event: str, _payload: dict) -> None:
        return None

    result = await PretokenizeAdapter().run(
        {
            "id": "task-historical-tokenizer",
            "payload": {
                "identity_hash": "f" * 64,
                "sequence_length": 4,
                "target_tokens": 8,
                "shard_sequences": 2,
                "packing_version": "document-pack-v1",
                "data_input": {"manifest_hash": "c" * 64},
                "tokenizer": {
                    "runtime_path": str(tokenizer_path),
                    "content_hash": "d" * 64,
                    "special_tokens": ["<unk>", "<s>", "</s>"],
                },
            },
            "runtime_context": {
                "project_workspace": str(project_workspace),
                "run_dir": str(run_dir),
            },
        },
        emit,
    )

    manifest = json.loads(
        (
            project_workspace
            / ".llm-platform"
            / "data-assets"
            / "tokenized"
            / ("f" * 64)
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["eos_token_id"] == 2
    assert result["outputs"][0]["storage_type"] == "external"
