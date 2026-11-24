from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from llm_platform_core.mixtures import (
    MIXTURE_INDEX_FORMAT,
    RECORD_MANIFEST_FORMAT,
    REFERENCE_SAMPLE_BLOCK_SIZE,
    REFERENCE_SAMPLER_VERSION,
    normalize_weight_units,
    open_text_manifest,
    sample_reference,
    select_records,
)
from llm_platform_worker.adapters.data_mixture import DataMixtureAdapter
from llm_platform_worker.adapters.mixture_resolve import resolve_manifest_source


def record(record_id: str, path: str, length: int = 100, tokens: int = 25) -> dict:
    return {
        "type": "record",
        "record_id": record_id,
        "logical_path": path,
        "record_index": int(record_id.rsplit("-", 1)[-1]),
        "content_hash": record_id.rjust(64, "0"),
        "character_length": length,
        "byte_length": length,
        "token_estimate": tokens,
        "language": "en",
    }


def test_integer_weights_and_selection_are_stable_across_input_order() -> None:
    assert normalize_weight_units([1, 2, 7]) == [100_000, 200_000, 700_000]
    records = [record(f"record-{index}", f"train/{index}.jsonl") for index in range(20)]
    selection = {
        "sample_ratio": 0.5,
        "max_tokens": 200,
        "min_length": 50,
        "max_length": 200,
        "include_paths": ["train/**"],
        "exclude_paths": ["**/19.jsonl"],
        "seed": 42,
    }
    first = select_records(records, selection, dataset_version_id="dataset-v1", default_seed=0)
    second = select_records(
        list(reversed(records)), selection, dataset_version_id="dataset-v1", default_seed=0
    )
    assert [item["record_id"] for item in first] == [item["record_id"] for item in second]
    assert sum(item["token_estimate"] for item in first) <= 200


def test_weighted_sampler_returns_the_same_reference_for_the_same_index() -> None:
    header = {
        "seed": 42,
        "stream_key": "a" * 64,
        "sources": [{"weight_units": 700_000}, {"weight_units": 300_000}],
    }
    groups = [[record("record-0", "a")], [record("record-1", "b")]]
    first = [sample_reference(header, groups, index) for index in range(100)]
    second = [sample_reference(header, groups, index) for index in range(100)]
    assert first == second
    assert {source for source, _ in first} == {0, 1}


def test_reference_block_sampler_keeps_each_block_local_and_weighted() -> None:
    header = {
        "seed": 42,
        "stream_key": "b" * 64,
        "sampler": REFERENCE_SAMPLER_VERSION,
        "sample_block_size": 8,
        "sources": [{"weight_units": 700_000}, {"weight_units": 300_000}],
    }
    groups = [
        [record(f"a-{index}", "a") for index in range(101)],
        [record(f"b-{index}", "b") for index in range(101)],
    ]

    references = [sample_reference(header, groups, index) for index in range(800)]
    assert references == [sample_reference(header, groups, index) for index in range(800)]
    for block_start in range(0, len(references), 8):
        block = references[block_start : block_start + 8]
        assert len({source_index for source_index, _ in block}) == 1
        assert all(
            current[1] == (previous[1] + 1) % len(groups[current[0]])
            for previous, current in zip(block, block[1:], strict=False)
        )
    assert {source_index for source_index, _ in references} == {0, 1}


def test_reference_sampler_rejects_unknown_version() -> None:
    header = {
        "seed": 42,
        "stream_key": "c" * 64,
        "sampler": "unknown_sampler_v99",
        "sources": [{"weight_units": 1_000_000}],
    }
    with pytest.raises(ValueError, match="unsupported mixture reference sampler"):
        sample_reference(header, [[record("record-0", "a")]], 0)


def record_manifest(dataset_version_id: str, prefix: str) -> bytes:
    lines = [
        json.dumps(
            {
                "type": "header",
                "format": RECORD_MANIFEST_FORMAT,
                "dataset_version_id": dataset_version_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    lines.extend(
        json.dumps(
            record(f"{prefix}-{index}", f"train/{index}.jsonl"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(10)
    )
    return ("\n".join(lines) + "\n").encode()


def test_duckdb_resolver_matches_in_memory_selection_with_token_limit(tmp_path: Path) -> None:
    records = [
        record(f"record-{index}", f"train/{index}.jsonl", length=50 + index, tokens=7 + index)
        for index in range(40)
    ]
    input_path = tmp_path / "records.jsonl"
    input_path.write_bytes(record_manifest("dataset-v1", "unused"))
    input_path.write_text(
        json.dumps(
            {
                "type": "header",
                "format": RECORD_MANIFEST_FORMAT,
                "dataset_version_id": "dataset-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        + "\n".join(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in records)
        + "\n",
        encoding="utf-8",
    )
    selection = {
        "sample_ratio": 0.75,
        "max_tokens": 240,
        "min_length": 55,
        "max_length": 85,
        "include_paths": ["train/**"],
        "exclude_paths": ["**/19.jsonl"],
        "seed": 17,
    }
    output_path = tmp_path / "body.jsonl"
    progress: dict = {}
    result = resolve_manifest_source(
        input_path=input_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        dataset_version_id="dataset-v1",
        selection=selection,
        default_seed=0,
        source_index=3,
        cancel_path=tmp_path / "cancel",
        progress=progress,
        progress_lock=threading.Lock(),
    )
    expected = select_records(records, selection, dataset_version_id="dataset-v1", default_seed=0)
    actual = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [item["record_id"] for item in actual] == [item["record_id"] for item in expected]
    assert [item["source_position"] for item in actual] == list(range(len(expected)))
    assert all(item["source_index"] == 3 for item in actual)
    assert result["selected_document_count"] == len(expected)
    assert result["selected_tokens"] == sum(item["token_estimate"] for item in expected)


@pytest.mark.parametrize("filename", ["records.jsonl", "records.jsonl.gz"])
def test_duckdb_resolver_reads_concatenated_gzip_members(tmp_path: Path, filename: str) -> None:
    header, *record_lines = record_manifest("dataset-v1", "gzip").splitlines(keepends=True)
    input_path = tmp_path / filename
    input_path.write_bytes(gzip.compress(header) + gzip.compress(b"".join(record_lines)))
    output_path = tmp_path / "body.jsonl"
    result = resolve_manifest_source(
        input_path=input_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        dataset_version_id="dataset-v1",
        selection={"sample_ratio": 1},
        default_seed=0,
        source_index=0,
        cancel_path=tmp_path / "cancel",
        progress={},
        progress_lock=threading.Lock(),
    )
    assert result["selected_document_count"] == 10
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 10


def test_duckdb_resolver_reads_plain_manifest_with_gzip_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl.gz"
    input_path.write_bytes(record_manifest("dataset-v1", "plain"))
    result = resolve_manifest_source(
        input_path=input_path,
        output_path=tmp_path / "body.jsonl",
        work_dir=tmp_path / "work",
        dataset_version_id="dataset-v1",
        selection={"include_paths": ["train/**"]},
        default_seed=0,
        source_index=0,
        cancel_path=tmp_path / "cancel",
        progress={},
        progress_lock=threading.Lock(),
    )
    assert result["selected_document_count"] == 10


@pytest.mark.asyncio
async def test_mixture_resolver_stop_interrupts_background_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocking_resolver(**kwargs: object) -> dict[str, int]:
        cancel_path = kwargs["cancel_path"]
        progress = kwargs["progress"]
        progress_lock = kwargs["progress_lock"]
        assert isinstance(cancel_path, Path)
        assert isinstance(progress, dict)
        assert isinstance(progress_lock, type(threading.Lock()))
        with progress_lock:
            progress.update(phase="external_sort", processed_bytes=1, total_bytes=100)
        while not cancel_path.exists():
            time.sleep(0.01)
        raise RuntimeError("mixture resolution interrupted")

    monkeypatch.setattr(
        "llm_platform_worker.adapters.data_mixture.resolve_manifest_source",
        blocking_resolver,
    )
    adapter = DataMixtureAdapter(httpx.AsyncClient(base_url="http://test"))

    async def emit(_event_type: str, _payload: dict) -> str:
        return "cancelled"

    started = time.monotonic()
    resolved, state = await adapter._resolve_source(
        input_path=tmp_path / "input.jsonl",
        body_path=tmp_path / "body.jsonl",
        work_dir=tmp_path / "work",
        dataset_version_id="dataset-v1",
        selection={},
        seed=0,
        source_index=0,
        source_count=1,
        emit=emit,
        get_desired_state=lambda: "cancelled",
    )
    await adapter.client.aclose()
    assert resolved is None
    assert state == "cancelled"
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_mixture_adapter_emits_identical_manifest_for_identical_inputs(
    tmp_path: Path, compressed: bool,
) -> None:
    payloads = {
        "artifact-a": record_manifest("dataset-a", "a"),
        "artifact-b": record_manifest("dataset-b", "b"),
    }
    if compressed:
        for key, payload in payloads.items():
            header, body = payload.split(b"\n", 1)
            payloads[key] = gzip.compress(header + b"\n") + gzip.compress(body)

    def handler(request: httpx.Request) -> httpx.Response:
        artifact_id = request.url.path.split("/")[-2]
        return httpx.Response(200, content=payloads[artifact_id])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        adapter = DataMixtureAdapter(client)

        async def emit(_event_type: str, _payload: dict) -> str:
            return "running"

        task_payload = {
            "mixture_version_id": "mixture-v1",
            "seed": 42,
            "target_tokens": 1_000,
            "entries": [
                {
                    "dataset_version_id": "dataset-a",
                    "record_manifest_artifact_id": "artifact-a",
                    "record_manifest_hash": hashlib.sha256(payloads["artifact-a"]).hexdigest(),
                    "weight": 7,
                    "selection": {"sample_ratio": 0.8, "seed": 11},
                },
                {
                    "dataset_version_id": "dataset-b",
                    "record_manifest_artifact_id": "artifact-b",
                    "record_manifest_hash": hashlib.sha256(payloads["artifact-b"]).hexdigest(),
                    "weight": 3,
                    "selection": {"sample_ratio": 1, "seed": 12},
                },
            ],
        }
        first = await adapter.run(
            {"payload": task_payload, "runtime_context": {"run_dir": str(tmp_path / "one")}},
            emit,
        )
        second = await adapter.run(
            {"payload": task_payload, "runtime_context": {"run_dir": str(tmp_path / "two")}},
            emit,
        )
    first_path = Path(first["outputs"][0]["path"])
    second_path = Path(second["outputs"][0]["path"])
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["resolved_document_count"] == second["resolved_document_count"]
    with open_text_manifest(first_path) as stream:
        header = json.loads(stream.readline())
    assert first_path.read_bytes()[:2] == b"\x1f\x8b"
    assert header["format"] == "llm-mixture-manifest-v1"
    assert header["sources"][0]["weight_units"] == 700_000
    suffix = ".jsonl.gz" if compressed else ".jsonl"
    for run in ("one", "two"):
        inputs = tmp_path / run / "inputs"
        assert sorted(path.name for path in inputs.iterdir()) == [
            f"source-{index:05d}{suffix}" for index in range(2)
        ]
        for index, payload in enumerate(payloads.values()):
            assert (inputs / f"source-{index:05d}{suffix}").read_bytes() == payload


@pytest.mark.asyncio
async def test_identity_mixture_is_a_header_only_reference_index(tmp_path: Path) -> None:
    def unexpected_download(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("identity mixtures must not copy the Record Manifest")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected_download), base_url="http://test"
    ) as client:
        adapter = DataMixtureAdapter(client)

        async def emit(_event_type: str, _payload: dict) -> str:
            return "running"

        result = await adapter.run(
            {
                "payload": {
                    "mixture_version_id": "mixture-v2",
                    "seed": 42,
                    "target_tokens": 10_000_000,
                    "entries": [
                        {
                            "dataset_version_id": "dataset-v1",
                            "dataset_artifact_id": "snapshot-a",
                            "dataset_artifact_hash": "a" * 64,
                            "record_manifest_artifact_id": "records-a",
                            "record_manifest_hash": "b" * 64,
                            "document_count": 100,
                            "token_estimate": 2500,
                            "weight": 1,
                            "selection": {"sample_ratio": 1, "seed": 42},
                        }
                    ],
                },
                "runtime_context": {"run_dir": str(tmp_path)},
            },
            emit,
        )

    output = Path(result["outputs"][0]["path"])
    lines = output.read_text(encoding="utf-8").splitlines()
    assert result["format"] == MIXTURE_INDEX_FORMAT
    assert result["outputs"][0]["metadata"]["format"] == MIXTURE_INDEX_FORMAT
    assert len(lines) == 1
    assert output.stat().st_size < 4096
    header = json.loads(lines[0])
    assert header["format"] == MIXTURE_INDEX_FORMAT
    assert header["record_order"] == "record_manifest"
    assert header["sampler"] == REFERENCE_SAMPLER_VERSION
    assert header["sample_block_size"] == REFERENCE_SAMPLE_BLOCK_SIZE
    assert header["sources"][0]["selected_document_count"] == 100
