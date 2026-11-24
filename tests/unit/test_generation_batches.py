import json
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from llm_platform_core.data_generation import DataGenerationCreate, FactTranslationCreate
from llm_platform_worker.data_generation_runtime import (
    generation_job_key,
    jobs,
    prepare_generation_checkpoint,
    records,
)
from llm_platform_worker.generation_continuation import (
    import_checkpoint_range,
    remaining_verified_jobs,
    sealed_checkpoint_index,
)


@pytest.mark.parametrize("extension", ["jsonl", "parquet"])
def test_range_preserves_source_ordinals_across_row_groups_and_styles(tmp_path, extension):
    path = tmp_path / f"source.{extension}"
    rows = [{"text": f"第{i}条中文材料"} for i in range(11)]
    if extension == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    all_keys = []
    for start, stop in [(0, 4), (4, 8), (8, 11)]:
        cfg = DataGenerationCreate(name="range", input_dataset_version_id="source", model_weight_version_id="model",
            styles=["textbook", "plain"], record_range={"logical_file": path.name, "start": start, "stop": stop, "total": 11})
        selected = list(jobs({"runtime_path": str(path)}, cfg, "knowledge"))
        assert {row["source_ordinal"] for row in selected} == set(range(start, stop))
        all_keys.extend(generation_job_key(row) for row in selected)
    assert len(all_keys) == len(set(all_keys)) == 22
    with pytest.raises(ValueError, match="record count"):
        list(records(path, "text", start=0, stop=10, expected_total=10))


def checkpoint(tmp_path):
    task = {"id": str(uuid4()), "payload": {"output_dataset_version_id": str(uuid4()),
        "generation": {"mode": "fact_translation"}, "generation_input": {"artifact_hash": "input"},
        "generation_model": {"identity_hash": "model"}}}
    attempt = str(uuid4())
    provenance = {"input_artifact_hash": "input", "model_identity_hash": "model"}
    directory = tmp_path / f"{task['payload']['output_dataset_version_id']}-{attempt}"
    prepare_generation_checkpoint(task, tmp_path, directory, provenance, {})
    def row(i):
        return {"text": f"中文{i}", "metadata": {**provenance, "source_file": "source.jsonl",
            "source_ordinal": i, "source_sha256": f"hash-{i}", "character_start": 0,
            "character_end": 10, "style": "fact_translation", "generation_tokens": 8}}
    (directory / "data.partial").write_bytes(
        ("".join(json.dumps(row(i)) + "\n" for i in [4, 0, 2])).encode() + b'{"unfinished":')
    (directory / "skipped.records").write_text(json.dumps({**row(1), "reason": "bad"}) + "\n", encoding="utf-8")
    return {"task": task, "attempt_id": attempt, "worker_id": "worker"}, provenance, directory, row


def test_unordered_checkpoint_import_is_bounded_idempotent_and_preserves_rejections(tmp_path):
    binding, provenance, source, row = checkpoint(tmp_path)
    original = (source / "data.partial").read_bytes()
    target = tmp_path / "new"
    target.mkdir()
    task = {"payload": {"generation_continuation": binding, "generation": {
        "record_range": {"logical_file": "source.jsonl", "start": 0, "stop": 4}}},
        "runtime_context": {"worker_id": "worker"}}
    report, completed = {}, set()
    imported = import_checkpoint_range(task, tmp_path, target, provenance, report, completed)
    assert report["resumed_records"] == 2 and report["resumed_skipped_records"] == 1
    assert report["resumed_generated_tokens"] == 16
    rest = list(remaining_verified_jobs((row(i)["metadata"] for i in range(4)), completed, imported))
    assert [r["source_ordinal"] for r in rest] == [3]
    before = (target / "data.partial").read_bytes()
    import_checkpoint_range(task, tmp_path, target, provenance, report, completed)
    assert (target / "data.partial").read_bytes() == before
    assert (source / "data.partial").read_bytes() == original
    with pytest.raises(ValueError, match="immutable source"):
        list(remaining_verified_jobs([row(0)["metadata"]], completed, imported))
    (source / "data.partial").write_bytes(original + b"extra")
    with pytest.raises(ValueError, match="changed after publication"):
        sealed_checkpoint_index(tmp_path, binding, provenance)


def test_checkpoint_sealing_rejects_corrupt_provenance_and_duplicate_rows(tmp_path):
    binding, provenance, source, row = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        sealed_checkpoint_index(tmp_path, binding, {**provenance, "model_identity_hash": "other"})
    (source / "data.partial").write_text(json.dumps(row(0)) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        sealed_checkpoint_index(tmp_path, binding, provenance)
    (source / "data.partial").write_text((json.dumps(row(0)) + "\n") * 2, encoding="utf-8")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        sealed_checkpoint_index(tmp_path, binding, provenance)


@pytest.mark.parametrize("patch", [
    {"record_range": {"logical_file": "a", "start": 4, "stop": 4}},
    {"record_range": {"logical_file": "a", "start": 0, "stop": 4, "total": 3}},
    {"checkpoint_source_task_id": str(uuid4())},
    {"checkpoint_source_task_id": str(uuid4()), "checkpoint_source_attempt_id": str(uuid4())},
])
def test_invalid_continuation_contract(patch):
    with pytest.raises(ValueError):
        FactTranslationCreate(name="test", input_dataset_version_id="input", model_weight_version_id="model", **patch)


def test_compact_index_has_no_body_and_detects_same_size_source_tampering(tmp_path):
    import os
    import sqlite3
    from contextlib import closing

    from llm_platform_worker.generation_continuation import checkpoint_rows

    binding, provenance, source, row = checkpoint(tmp_path)
    body = row(0)
    body["text"] = "large body " * 100000
    original = (json.dumps(body) + "\n").encode()
    target = source / "data.partial"
    target.write_bytes(original)
    index, metadata = sealed_checkpoint_index(tmp_path, binding, provenance)
    assert metadata["storage"] == "offsets-v2"
    assert index.stat().st_size < len(original) // 10
    selection = {"logical_file": "source.jsonl", "start": 0, "stop": 1}
    with closing(sqlite3.connect(index)) as db:
        assert "row" not in {r[1] for r in db.execute("pragma table_info(rows)")}
        assert list(checkpoint_rows(db, metadata, source, selection))[0][0] == original
        stamp = target.stat()
        target.write_bytes(original.replace(b"large body", b"other body", 1))
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with pytest.raises(ValueError, match="source row changed"):
            list(checkpoint_rows(db, metadata, source, selection))


def test_legacy_full_row_index_still_imports(tmp_path):
    import sqlite3
    from contextlib import closing

    binding, provenance, source, row = checkpoint(tmp_path)
    index, _ = sealed_checkpoint_index(tmp_path, binding, provenance)
    with closing(sqlite3.connect(index)) as db:
        db.execute("DROP TABLE rows")
        db.execute("CREATE TABLE rows (file, ordinal, start, style, row, skipped)")
        db.execute("INSERT INTO rows VALUES (?,?,?,?,?,?)", ("source.jsonl", 0, 0, "fact_translation", json.dumps(row(0)), 0))
        db.execute("DELETE FROM metadata WHERE key='storage'")
        db.commit()
    output = tmp_path / "legacy-target"
    output.mkdir()
    task = {"payload": {"generation_continuation": binding, "generation": {
        "record_range": {"logical_file": "source.jsonl", "start": 0, "stop": 1}}},
        "runtime_context": {"worker_id": "worker"}}
    report = {}
    import_checkpoint_range(task, tmp_path, output, provenance, report, set())
    assert report["resumed_records"] == 1
    assert json.loads((output / "data.partial").read_text(encoding="utf-8"))["text"] == "中文0"
