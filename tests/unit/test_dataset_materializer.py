from __future__ import annotations

import os
from urllib.parse import urlsplit
from urllib.request import url2pathname
from pathlib import Path

import pytest
from llm_platform_worker.adapters.dataset_materializer import (
    DatasetMaterializerAdapter,
    _directory_hash,
)
from llm_platform_worker.main import WorkerAgent, _archive_output_directory


@pytest.mark.asyncio
async def test_dataset_materializer_copies_verifies_and_reuses_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.jsonl").write_text('{"text":"a"}\n', encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("second", encoding="utf-8")
    expected_hash, expected_size = _directory_hash(source)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict):
        events.append((event_type, payload))
        return "running"

    task = {
        "payload": {
            "materialization_id": "11111111-1111-1111-1111-111111111111",
            "dataset_version_id": "22222222-2222-2222-2222-222222222222",
            "expected_content_hash": expected_hash,
            "materialization_input": {"runtime_path": str(source)},
        },
        "runtime_context": {
            "project_workspace": str(tmp_path),
            "worker_id": "33333333-3333-3333-3333-333333333333",
        },
    }
    adapter = DatasetMaterializerAdapter()
    result = await adapter.run(task, emit)
    target = Path(url2pathname(urlsplit(result["materialized_uri"]).path))
    assert target.is_dir()
    assert _directory_hash(target) == (expected_hash, expected_size)
    assert result["outputs"][0]["content_hash"] == expected_hash

    events.clear()
    reused = await adapter.run(task, emit)
    assert reused["materialized_uri"] == result["materialized_uri"]
    assert any(payload["phase"] == "reusing_verified_replica" for _, payload in events)


@pytest.mark.asyncio
async def test_dataset_materializer_rejects_wrong_frozen_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"actual")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def emit(_event_type: str, _payload: dict):
        return "running"

    with pytest.raises(ValueError, match="failed content verification"):
        await DatasetMaterializerAdapter().run(
            {
                "payload": {
                    "materialization_id": "11111111-1111-1111-1111-111111111111",
                    "dataset_version_id": "22222222-2222-2222-2222-222222222222",
                    "expected_content_hash": "0" * 64,
                    "materialization_input": {"runtime_path": str(source)},
                },
                "runtime_context": {
                    "project_workspace": str(workspace),
                    "worker_id": "33333333-3333-3333-3333-333333333333",
                },
            },
            emit,
        )


def test_dataset_snapshot_directory_archive_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    payload = source / "data.jsonl"
    payload.write_text('{"text":"stable"}\n', encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    _archive_output_directory(source, first, deterministic=True)
    os.utime(payload, (2_000_000_000, 2_000_000_000))
    _archive_output_directory(source, second, deterministic=True)

    assert first.read_bytes() == second.read_bytes()


def test_rebuilt_managed_snapshot_persists_the_exact_uploaded_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "data.jsonl").write_text('{"text":"stable"}\n', encoding="utf-8")
    archive = tmp_path / "payload.tar.gz"
    _archive_output_directory(source, archive, deterministic=True)
    expected_hash = WorkerAgent._sha256_file(archive)
    task = {
        "payload": {
            "materialization_id": "11111111-1111-1111-1111-111111111111",
            "dataset_version_id": "22222222-2222-2222-2222-222222222222",
            "expected_content_hash": expected_hash,
        },
        "runtime_context": {"project_workspace": str(workspace)},
    }
    agent = object.__new__(WorkerAgent)

    persisted = agent._persist_materialized_payload(task, source, archive)

    assert persisted.read_bytes() == archive.read_bytes()
    assert persisted.name == f"{expected_hash}.tar.gz"
