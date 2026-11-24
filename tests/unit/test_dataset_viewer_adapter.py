from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from llm_platform_worker.adapters.dataset_labeler import execute_label_operation
from llm_platform_worker.adapters.dataset_viewer import execute_dataset_query


def _write_conversation_dataset(root: Path) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
                "source": "alpha",
                "score": 0.9,
            },
            {
                "messages": [{"role": "assistant", "content": "Hi"}],
                "source": "beta",
                "score": 0.7,
            },
        ]
    )
    pq.write_table(table, data_dir / "train-00000-of-00001.parquet")
    pq.write_table(table.slice(0, 1), data_dir / "test-00000-of-00001.parquet")


def _query(root: Path, tmp_path: Path, sql: str = "SELECT * FROM dataset", **kwargs):
    return execute_dataset_query(
        input_path=root,
        sql=sql,
        split=str(kwargs.get("split") or ""),
        offset=int(kwargs.get("offset") or 0),
        limit=int(kwargs.get("limit") or 100),
        timeout_seconds=10,
        temp_dir=tmp_path / "duckdb-temp",
        search=kwargs.get("search", ""),
        logical_files=kwargs.get("logical_files"),
        record_manifest_path=kwargs.get("record_manifest_path"),
        label_snapshot_path=kwargs.get("label_snapshot_path"),
    )


def _write_manifest(path: Path, logical_path: str, count: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"type": "record_manifest", "format": "llm.dataset-records.v1"}) + "\n")
        for index in range(count):
            output.write(json.dumps({"type": "record", "record_id": f"record-{index}", "logical_path": logical_path, "record_index": index, "content_hash": f"{index + 1:064x}"}) + "\n")


def test_dataset_viewer_preserves_original_columns_nested_values_and_splits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)

    result = _query(root, tmp_path, "SELECT messages, source, score FROM dataset ORDER BY source")

    assert result["split"] == "train"
    assert result["splits"] == ["train", "test"]
    assert result["total_rows"] == 2
    assert [column["name"] for column in result["columns"]] == ["messages", "source", "score"]
    assert result["rows"][0][0][1] == {"role": "user", "content": "Hello"}
    assert result["rows"][1][1] == "beta"


def test_dataset_viewer_runs_read_only_sql_with_pagination(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)

    result = _query(
        root,
        tmp_path,
        "WITH filtered AS (SELECT source FROM dataset WHERE score >= 0.7) "
        "SELECT * FROM filtered ORDER BY source",
        offset=1,
        limit=1,
    )

    assert result["total_rows"] == 2
    assert result["rows"] == [["beta"]]
    assert result["has_more"] is False


@pytest.mark.parametrize(
    ("filename", "content", "expected_format"),
    [
        ("train.jsonl", '{"name":"alpha","score":1}\n{"name":"beta","score":2}\n', "json"),
        ("train.csv", "name,score\nalpha,1\nbeta,2\n", "csv"),
        ("train.tsv", "name\tscore\nalpha\t1\nbeta\t2\n", "csv"),
    ],
)
def test_dataset_viewer_reads_supported_row_formats(
    tmp_path: Path, filename: str, content: str, expected_format: str
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / filename).write_text(content, encoding="utf-8")

    result = _query(root, tmp_path, "SELECT name, score FROM dataset ORDER BY score")

    assert result["format"] == expected_format
    assert [column["name"] for column in result["columns"]] == ["name", "score"]
    assert result["rows"] == [["alpha", 1], ["beta", 2]]


@pytest.mark.parametrize("logical_files", [None, ["data/train.jsonl"]])
def test_dataset_viewer_excludes_huggingface_tool_metadata(
    tmp_path: Path, logical_files: list[str] | None
) -> None:
    root = tmp_path / "snapshot"
    data_dir = root / "data"
    cache_dir = root / ".cache" / "huggingface" / "trees"
    data_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        '{"name":"alpha","score":1}\n{"name":"beta","score":2}\n',
        encoding="utf-8",
    )
    (cache_dir / "revision.json").write_text(
        '{"format_version":1,"files":{"README.md":{"size":10}}}\n',
        encoding="utf-8",
    )

    result = _query(root, tmp_path, logical_files=logical_files)

    assert [column["name"] for column in result["columns"]] == ["name", "score"]
    assert result["total_rows"] == 2


@pytest.mark.parametrize(
    ("logical_files", "message"),
    [
        (["../outside.parquet"], "escapes snapshot root"),
        (["missing.parquet"], "is missing"),
    ],
)
def test_dataset_viewer_rejects_invalid_frozen_file_entries(
    tmp_path: Path, logical_files: list[str], message: str
) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _query(root, tmp_path, logical_files=logical_files)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM dataset",
        "SELECT 1; SELECT 2",
        "PRAGMA version",
        "COPY (SELECT * FROM dataset) TO 'leak.parquet'",
    ],
)
def test_dataset_viewer_rejects_non_select_or_multiple_statements(
    tmp_path: Path, sql: str
) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)

    with pytest.raises(ValueError, match="only accepts"):
        _query(root, tmp_path, sql)


def test_dataset_viewer_cannot_read_outside_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)
    outside = tmp_path / "outside.parquet"
    pq.write_table(pa.Table.from_pylist([{"secret": "no"}]), outside)

    with pytest.raises(Exception, match="disabled by configuration|not allowed"):
        _query(root, tmp_path, f"SELECT * FROM read_parquet('{outside.as_posix()}')")


def test_dataset_viewer_returns_stable_hidden_row_identities(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)
    manifest = tmp_path / "records.jsonl.gz"
    _write_manifest(manifest, "data/train-00000-of-00001.parquet", 2)

    result = _query(root, tmp_path, record_manifest_path=manifest)

    assert result["labelable"] is True
    assert result["row_identities"] == [
        {"record_id": "record-0", "record_hash": f"{1:064x}"},
        {"record_id": "record-1", "record_hash": f"{2:064x}"},
    ]
    assert "_forgelm_record_id" not in [column["name"] for column in result["columns"]]
    assert "_forgelm_labels" in [column["name"] for column in result["columns"]]

    projected = _query(
        root,
        tmp_path,
        "SELECT source FROM dataset",
        record_manifest_path=manifest,
    )
    assert projected["labelable"] is False
    assert projected["row_identities"] == []


def test_dataset_labeler_freezes_query_and_replaces_single_group(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    _write_conversation_dataset(root)
    manifest = tmp_path / "records.jsonl.gz"
    _write_manifest(manifest, "data/train-00000-of-00001.parquet", 2)
    label_base = {
        "scheme_id": "scheme",
        "scheme_code": "quality",
        "scheme_version_id": "scheme-v1",
        "scheme_version": 1,
        "group_id": "sentiment",
        "group_code": "sentiment",
        "selection_mode": "single",
    }
    first = execute_label_operation(
        dataset_path=root,
        manifest_path=manifest,
        base_snapshot_path=None,
        logical_files=["data/train-00000-of-00001.parquet"],
        source_query={"sql": "SELECT * FROM dataset WHERE score >= 0.7", "split": "train"},
        scope="query",
        explicit_records=[],
        action="apply",
        labels=[{**label_base, "label_id": "positive", "label_code": "positive"}],
        annotator="alice",
        operation_id="operation-1",
        output_dir=tmp_path / "first",
    )
    assert first["selection_count"] == 2
    assert first["assignment_count"] == 2

    second = execute_label_operation(
        dataset_path=root,
        manifest_path=manifest,
        base_snapshot_path=tmp_path / "first" / "record-labels.parquet",
        logical_files=["data/train-00000-of-00001.parquet"],
        source_query={"sql": "SELECT * FROM dataset", "split": "train"},
        scope="selected",
        explicit_records=[{"record_id": "record-0", "record_hash": f"{1:064x}"}],
        action="apply",
        labels=[{**label_base, "label_id": "negative", "label_code": "negative"}],
        annotator="bob",
        operation_id="operation-2",
        output_dir=tmp_path / "second",
    )
    rows = pq.read_table(tmp_path / "second" / "record-labels.parquet").to_pylist()
    assert second["assignment_count"] == 2
    assert {(row["record_id"], row["label_code"], row["annotated_by"]) for row in rows} == {
        ("record-0", "negative", "bob"),
        ("record-1", "positive", "alice"),
    }

    third = execute_label_operation(
        dataset_path=root,
        manifest_path=manifest,
        base_snapshot_path=tmp_path / "second" / "record-labels.parquet",
        logical_files=["data/train-00000-of-00001.parquet"],
        source_query={"sql": "SELECT * FROM dataset WHERE list_contains(_forgelm_labels, 'quality:sentiment:positive')", "split": "train"},
        scope="query",
        explicit_records=[],
        action="remove",
        labels=[{**label_base, "label_id": "positive", "label_code": "positive"}],
        annotator="carol",
        operation_id="operation-3",
        output_dir=tmp_path / "third",
    )
    assert third["selection_count"] == 1
    assert pq.read_table(tmp_path / "third" / "record-labels.parquet").to_pylist()[0][
        "label_code"
    ] == "negative"
