from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from llm_platform_worker.adapters.supervision import (
    SupervisionScanAdapter,
    SupervisionTrainingIndexAdapter,
    _conversation,
    _indexed_canonical_records,
    _preference,
    _require_sft_span_eos,
    _restore_declared_eos_token,
    _spans,
    _truncate_sft_prompt,
    supervision_available,
)


def test_supervision_capability_requires_jinja2(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_jinja(name, *args, **kwargs):
        if name == "jinja2":
            raise ImportError("jinja2 intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_jinja)
    assert supervision_available() is False


def test_conversation_and_preference_mapping_are_canonical() -> None:
    conversation = _conversation(
        {"messages": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        {
            "messages_path": "messages",
            "role_field": "from",
            "content_field": "value",
            "role_map": {"human": "user", "gpt": "assistant"},
        },
    )
    assert conversation == {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    preference = _preference(
        {"question": "2+2?", "good": "4", "bad": "5"},
        {"prompt_path": "question", "chosen_path": "good", "rejected_path": "bad"},
    )
    assert preference["prompt"][0]["role"] == "user"
    assert preference["chosen"][0]["content"] == "4"
    assert preference["rejected"][0]["content"] == "5"


def test_assistant_mask_spans_and_prompt_truncation_preserve_completion() -> None:
    assert _spans([0, 0, 1, 1, 0, 1]) == [(2, 2), (5, 1)]
    ids, mask = _truncate_sft_prompt(list(range(10)), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 8)
    assert ids == [0, 3, 4, 5, 6, 7, 8, 9]
    assert mask == [0, 0, 0, 1, 1, 1, 1, 1]


def test_historical_tokenizer_eos_is_restored_and_required_per_assistant_span() -> None:
    class Tokenizer:
        eos_token = None
        eos_token_id = None
        pad_token = None
        pad_token_id = None

        def get_vocab(self) -> dict[str, int]:
            return {"<unk>": 0, "<s>": 1, "</s>": 2}

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)
            if name == "eos_token" and value == "</s>":
                object.__setattr__(self, "eos_token_id", 2)
            if name == "pad_token" and value == "</s>":
                object.__setattr__(self, "pad_token_id", 2)

    tokenizer = Tokenizer()
    eos_token_id = _restore_declared_eos_token(
        tokenizer, {"special_tokens": ["<unk>", "<s>", "</s>"]}
    )
    assert eos_token_id == 2
    assert tokenizer.pad_token_id == 2
    _require_sft_span_eos([10, 11, 2, 20, 21, 2], [(1, 2), (4, 2)], eos_token_id)
    with pytest.raises(ValueError, match="assistant generation span"):
        _require_sft_span_eos([10, 11, 2, 20, 21, 22], [(1, 2), (4, 2)], eos_token_id)


@pytest.mark.asyncio
async def test_scan_adapter_writes_immutable_record_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"prompt": "A", "completion": "B"}),
                json.dumps({"prompt": "A", "completion": "B"}),
                json.dumps({"prompt": "missing completion"}),
            ]
        ),
        encoding="utf-8",
    )
    identity = hashlib.sha256(b"scan-test").hexdigest()
    events: list[tuple[str, dict]] = []

    async def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    result = await SupervisionScanAdapter().run(
        {
            "id": "task-1",
            "payload": {
                "identity_hash": identity,
                "kind": "conversation",
                "mapping": {"prompt_field": "prompt", "completion_field": "completion"},
                "filters": {"exact_deduplicate": True},
                "data_input": {"runtime_sources": [{"runtime_path": str(source)}]},
            },
            "runtime_context": {"project_workspace": str(tmp_path)},
        },
        emit,
    )
    assert result["accepted_record_count"] == 1
    assert result["rejected_record_count"] == 2
    output = result["outputs"][0]
    destination = Path(output["uri"].removeprefix("file:///"))
    if not destination.is_dir():
        destination = tmp_path / ".llm-platform" / "data-assets" / "supervision-records" / identity
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "forgelm-supervision-record-index-v1"
    assert manifest["validation_report"]["rejection_reasons"] == {
        "conversation_fields_missing": 1,
        "exact_duplicate": 1,
    }
    pointers = [
        json.loads(line)
        for line in (destination / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert pointers == [
        {
            "logical_path": "train.jsonl",
            "record_index": 0,
            "semantic_hash": pointers[0]["semantic_hash"],
        }
    ]
    assert "messages" not in pointers[0]
    rejections = [
        json.loads(line)
        for line in (destination / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["reason"] for row in rejections} == {
        "conversation_fields_missing",
        "exact_duplicate",
    }
    reconstructed = list(
        _indexed_canonical_records(
            source,
            destination / "records.jsonl",
            kind="conversation",
            mapping={"prompt_field": "prompt", "completion_field": "completion"},
            get_desired_state=None,
        )
    )
    assert reconstructed == [
        {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}
    ]
    assert events[-1][1]["progress"] == 1.0


@pytest.mark.asyncio
async def test_scan_adapter_joins_frozen_lookup_and_embeds_canonical_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "questions"
    lookup = tmp_path / "mother"
    source.mkdir()
    lookup.mkdir()
    (source / "questions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"question": "在哪里？", "answer": "在北方。", "node_id": "n1"}),
                json.dumps({"question": "缺少什么？", "answer": "未知。", "node_id": "missing"}),
            ]
        ),
        encoding="utf-8",
    )
    (lookup / "nodes.jsonl").write_text(
        json.dumps({"id": "n1", "dense_note": "这是必须逐字召回的母知识。"}),
        encoding="utf-8",
    )
    identity = hashlib.sha256(b"lookup-scan-test").hexdigest()

    async def emit(_event: str, _payload: dict) -> None:
        return None

    mapping = {
        "prompt_parts": [{"field": {"path": "question"}}],
        "completion_parts": [
            {"literal": "Recall："},
            {"field": {"path": "_lookup.mother.dense_note"}},
            {"literal": "\nA："},
            {"field": {"path": "answer"}},
        ],
        "_lookups": [
            {
                "name": "mother",
                "source_key": "node_id",
                "lookup_key": "id",
                "fields": {"dense_note": "dense_note"},
                "required": True,
            }
        ],
    }
    result = await SupervisionScanAdapter().run(
        {
            "id": "lookup-task",
            "payload": {
                "identity_hash": identity,
                "kind": "conversation",
                "mapping": mapping,
                "filters": {},
                "data_inputs": [
                    {"runtime_sources": [{"runtime_path": str(source)}]},
                    {"runtime_sources": [{"runtime_path": str(lookup)}]},
                ],
            },
            "runtime_context": {"project_workspace": str(tmp_path)},
        },
        emit,
    )
    assert result["accepted_record_count"] == 1
    assert result["rejected_record_count"] == 1
    destination = (
        tmp_path / ".llm-platform" / "data-assets" / "supervision-records" / identity
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["materialization_mode"] == "canonical_rows_v1"
    row = json.loads((destination / "records.jsonl").read_text(encoding="utf-8"))
    assert row["messages"][1]["content"] == "Recall：这是必须逐字召回的母知识。\nA：在北方。"
    assert list(
        _indexed_canonical_records(
            source,
            destination / "records.jsonl",
            kind="conversation",
            mapping=mapping,
            get_desired_state=None,
        )
    ) == [{"messages": row["messages"]}]
    rejection = json.loads((destination / "rejections.jsonl").read_text(encoding="utf-8"))
    assert rejection["reason"] == "lookup_match_missing:mother"


@pytest.mark.asyncio
async def test_training_view_converts_step_budget_to_deterministic_rows(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "format": "forgelm-sft-corpus-v1",
                "identity_hash": "a" * 64,
                "example_count": 5,
                "pair_count": 0,
                "raw_token_count": 50,
                "supervised_token_count": 25,
                "sources": [{"source_index": 0, "row_start": 0, "row_count": 5, "weight": 1}],
            }
        ),
        encoding="utf-8",
    )

    async def emit(_event: str, _payload: dict) -> None:
        return None

    identity = hashlib.sha256(b"view-test").hexdigest()
    result = await SupervisionTrainingIndexAdapter().run(
        {
            "id": "task-view",
            "payload": {
                "identity_hash": identity,
                "format": "forgelm-sft-training-index-v1",
                "objective_kind": "sft",
                "configuration": {
                    "budget_unit": "steps",
                    "budget_value": 3,
                    "consumption_policy": "repeat_epochs",
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "seed": 7,
                },
                "supervision_corpus": {"runtime_path": str(corpus)},
            },
            "runtime_context": {"project_workspace": str(tmp_path)},
        },
        emit,
    )
    assert result["example_count"] == 24
    destination = (
        tmp_path
        / ".llm-platform"
        / "data-assets"
        / "supervision-training-views"
        / identity
    )
    order = np.fromfile(destination / "sample-order.idx", dtype="<u8")
    assert len(order) == 24
    assert set(order.tolist()) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_training_view_global_shuffle_is_reproducible_and_supports_disjoint_slices(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-global"
    corpus.mkdir()
    raw_lengths = np.asarray([110, 220, 70, 330, 95, 145, 180, 260], dtype="<u8")
    supervised_lengths = np.asarray([100, 200, 50, 300, 75, 125, 150, 225], dtype="<u8")
    indexes = np.zeros((len(raw_lengths), 4), dtype="<u8")
    indexes[:, 1] = raw_lengths
    indexes[:, 2] = np.arange(len(raw_lengths), dtype="<u8")
    indexes[:, 3] = 1
    indexes.tofile(corpus / "examples.idx")
    spans = np.column_stack(
        (np.zeros(len(raw_lengths), dtype="<u8"), supervised_lengths)
    )
    spans.tofile(corpus / "loss-spans.idx")
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "format": "forgelm-sft-corpus-v1",
                "identity_hash": "b" * 64,
                "example_count": len(raw_lengths),
                "pair_count": 0,
                "raw_token_count": int(raw_lengths.sum()),
                "supervised_token_count": int(supervised_lengths.sum()),
                "sources": [
                    {
                        "source_index": 0,
                        "row_start": 0,
                        "row_count": len(raw_lengths),
                        "weight": 1,
                    }
                ],
                "files": {
                    "index": {"path": "examples.idx"},
                    "loss_spans": {"path": "loss-spans.idx"},
                },
            }
        ),
        encoding="utf-8",
    )

    async def emit(_event: str, _payload: dict) -> None:
        return None

    async def build(identity_label: str, *, budget: int, offset: int = 0, seed: int = 42):
        identity = hashlib.sha256(identity_label.encode()).hexdigest()
        result = await SupervisionTrainingIndexAdapter().run(
            {
                "id": f"task-{identity_label}",
                "payload": {
                    "identity_hash": identity,
                    "format": "forgelm-sft-training-index-v1",
                    "objective_kind": "sft",
                    "configuration": {
                        "budget_unit": "supervised_tokens",
                        "budget_value": budget,
                        "consumption_policy": "unique_only",
                        "sample_order_policy": "global_shuffle_v1",
                        "sample_offset": offset,
                        "seed": seed,
                    },
                    "supervision_corpus": {
                        "runtime_path": str(corpus),
                        "identity_hash": "b" * 64,
                    },
                },
                "runtime_context": {"project_workspace": str(tmp_path)},
            },
            emit,
        )
        root = (
            tmp_path
            / ".llm-platform"
            / "data-assets"
            / "supervision-training-views"
            / identity
        )
        return result, np.fromfile(root / "sample-order.idx", dtype="<u8"), json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )

    first, first_order, first_manifest = await build("first", budget=500)
    duplicate, duplicate_order, duplicate_manifest = await build("duplicate", budget=500)
    assert first_order.tolist() == duplicate_order.tolist()
    assert first_manifest["order_identity_hash"] == duplicate_manifest["order_identity_hash"]
    assert len(set(first_order.tolist())) == len(first_order)
    selected_supervised = int(supervised_lengths[first_order].sum())
    assert selected_supervised >= 500
    assert len(first_order) == 1 or int(supervised_lengths[first_order[:-1]].sum()) < 500
    assert first["supervised_token_count"] == selected_supervised
    assert first["raw_token_count"] == int(raw_lengths[first_order].sum())

    _suffix, suffix_order, suffix_manifest = await build(
        "suffix", budget=10_000, offset=len(first_order)
    )
    assert not set(first_order.tolist()).intersection(suffix_order.tolist())
    assert sorted([*first_order.tolist(), *suffix_order.tolist()]) == list(
        range(len(raw_lengths))
    )
    assert first_manifest["sample_range"]["end"] == suffix_manifest["sample_range"]["start"]
    assert first_manifest["order_identity_hash"] == suffix_manifest["order_identity_hash"]

    _different, different_order, different_manifest = await build(
        "different", budget=500, seed=7
    )
    assert different_manifest["order_identity_hash"] != first_manifest["order_identity_hash"]
    assert different_order.tolist() != first_order.tolist()
