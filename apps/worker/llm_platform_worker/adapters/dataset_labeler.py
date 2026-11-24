from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .dataset_viewer import (
    _select_files,
    _source_expression,
    _sql_literal,
    _validate_sql,
    discover_dataset_files,
)

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
GetDesiredState = Callable[[], str]
LABELER_ENGINE = "duckdb_data_labeler_v1"


def _selected_logical_files(files: list[Path], allowed_root: Path) -> str:
    values = [path.relative_to(allowed_root).as_posix() for path in files]
    return "(" + ", ".join(_sql_literal(value) for value in values) + ")"


def _prepare_identity_dataset(
    connection: Any,
    *,
    dataset_path: Path,
    manifest_path: Path,
    label_snapshot_path: Path | None,
    logical_files: list[str],
    split: str,
) -> tuple[str, list[Path], Path]:
    root = dataset_path.resolve()
    grouped = discover_dataset_files(root, logical_files)
    selected_split, family, files, _splits = _select_files(grouped, split)
    allowed_root = root if root.is_dir() else root.parent
    connection.execute(
        "CREATE TEMP VIEW manifest_rows AS "
        "SELECT record_id, content_hash AS record_hash, logical_path, record_index, "
        "row_number() OVER (ORDER BY logical_path, record_index) AS ordinal "
        f"FROM read_json_auto({_sql_literal(manifest_path.as_posix())}, "
        "format = 'newline_delimited', ignore_errors = true) "
        "WHERE record_id IS NOT NULL AND logical_path IN "
        f"{_selected_logical_files(files, allowed_root)}"
    )
    label_join = ""
    label_column = "[] AS _forgelm_labels"
    if label_snapshot_path is not None:
        connection.execute(
            "CREATE TEMP VIEW label_lists AS "
            "SELECT record_id, list(scheme_code || ':' || group_code || ':' || label_code "
            "ORDER BY scheme_code, group_code, label_code) AS labels "
            f"FROM read_parquet({_sql_literal(label_snapshot_path.as_posix())}) "
            "GROUP BY record_id"
        )
        label_join = " LEFT JOIN label_lists ON label_lists.record_id = manifest_rows.record_id"
        label_column = (
            "CASE WHEN label_lists.labels IS NULL THEN [] ELSE label_lists.labels END "
            "AS _forgelm_labels"
        )
    connection.execute(
        "CREATE VIEW dataset AS WITH source AS ("
        f"SELECT *, row_number() OVER () AS ordinal FROM {_source_expression(family, files)}"
        ") SELECT source.* EXCLUDE (ordinal), "
        "manifest_rows.record_id AS _forgelm_record_id, "
        "manifest_rows.record_hash AS _forgelm_record_hash, "
        f"{label_column} "
        "FROM source LEFT JOIN manifest_rows USING (ordinal)"
        f"{label_join}"
    )
    source_count = int(connection.execute("SELECT count(*) FROM dataset").fetchone()[0])
    manifest_count = int(connection.execute("SELECT count(*) FROM manifest_rows").fetchone()[0])
    if source_count != manifest_count:
        raise ValueError(
            "Structured source rows do not align one-to-one with the frozen Record Manifest"
        )
    return selected_split, files, allowed_root


def execute_label_operation(
    *,
    dataset_path: Path,
    manifest_path: Path,
    base_snapshot_path: Path | None,
    logical_files: list[str],
    source_query: dict[str, Any],
    scope: str,
    explicit_records: list[dict[str, str]],
    action: str,
    labels: list[dict[str, Any]],
    annotator: str,
    operation_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Data labeling requires DuckDB") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "record-selection.parquet"
    delta_path = output_dir / "label-delta.parquet"
    snapshot_path = output_dir / "record-labels.parquet"
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '1 GiB'")
        temp_dir = output_dir / "duckdb-temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        allowed_directories = [
            dataset_path.resolve() if dataset_path.is_dir() else dataset_path.resolve().parent,
            manifest_path.resolve().parent,
            output_dir.resolve(),
            temp_dir.resolve(),
        ]
        if base_snapshot_path is not None:
            allowed_directories.append(base_snapshot_path.resolve().parent)
        connection.execute(f"SET temp_directory = {_sql_literal(temp_dir.as_posix())}")
        connection.execute(
            "SET allowed_directories = ["
            + ", ".join(_sql_literal(path.as_posix()) for path in allowed_directories)
            + "]"
        )
        connection.execute("SET allowed_paths = []")
        connection.execute("SET enable_external_access = false")
        connection.execute("SET allow_community_extensions = false")
        connection.execute("SET autoinstall_known_extensions = false")
        connection.execute("SET autoload_known_extensions = false")
        connection.execute("SET allow_persistent_secrets = false")
        selected_split, _files, _allowed_root = _prepare_identity_dataset(
            connection,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            label_snapshot_path=base_snapshot_path,
            logical_files=logical_files,
            split=str(source_query.get("split") or ""),
        )
        connection.execute(
            "CREATE TEMP TABLE selected_records(record_id VARCHAR, record_hash VARCHAR)"
        )
        if scope == "query":
            query = _validate_sql(connection, str(source_query.get("sql") or ""))
            counts = connection.execute(
                "SELECT count(*), count(_forgelm_record_id), count(_forgelm_record_hash) "
                f"FROM ({query}) AS frozen_query"
            ).fetchone()
            if not counts or counts[0] == 0:
                raise ValueError("The frozen query did not select any records")
            if counts[0] != counts[1] or counts[0] != counts[2]:
                raise ValueError("The frozen query contains rows without stable record identities")
            connection.execute(
                "INSERT INTO selected_records "
                "SELECT DISTINCT _forgelm_record_id, _forgelm_record_hash "
                f"FROM ({query}) AS frozen_query"
            )
        else:
            connection.executemany(
                "INSERT INTO selected_records VALUES (?, ?)",
                [
                    (str(item.get("record_id") or ""), str(item.get("record_hash") or ""))
                    for item in explicit_records
                ],
            )
        invalid_count = int(
            connection.execute(
                "SELECT count(*) FROM selected_records AS selected "
                "LEFT JOIN manifest_rows AS manifest "
                "ON manifest.record_id = selected.record_id "
                "AND manifest.record_hash = selected.record_hash "
                "WHERE manifest.record_id IS NULL"
            ).fetchone()[0]
        )
        if invalid_count:
            raise ValueError("Selected record identities do not match the frozen Record Manifest")
        selection_count = int(
            connection.execute("SELECT count(*) FROM selected_records").fetchone()[0]
        )
        if selection_count == 0:
            raise ValueError("No records were selected for labeling")
        connection.execute(
            "CREATE TEMP TABLE effective_labels("
            "record_id VARCHAR, record_hash VARCHAR, scheme_id VARCHAR, scheme_code VARCHAR, "
            "scheme_version_id VARCHAR, scheme_version INTEGER, group_id VARCHAR, group_code VARCHAR, "
            "label_id VARCHAR, label_code VARCHAR, annotated_by VARCHAR, annotated_at TIMESTAMP, "
            "operation_id VARCHAR)"
        )
        if base_snapshot_path is not None:
            connection.execute(
                "INSERT INTO effective_labels SELECT * FROM read_parquet(?)",
                [base_snapshot_path.as_posix()],
            )
        connection.execute(
            "CREATE TEMP TABLE requested_labels("
            "scheme_id VARCHAR, scheme_code VARCHAR, scheme_version_id VARCHAR, scheme_version INTEGER, "
            "group_id VARCHAR, group_code VARCHAR, selection_mode VARCHAR, label_id VARCHAR, label_code VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO requested_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    label["scheme_id"], label["scheme_code"], label["scheme_version_id"],
                    int(label["scheme_version"]), label["group_id"], label["group_code"],
                    label["selection_mode"], label["label_id"], label["label_code"],
                )
                for label in labels
            ],
        )
        annotated_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        connection.execute(
            "COPY (SELECT selected.record_id, selected.record_hash, requested.*, ? AS action, "
            "? AS annotated_by, ?::TIMESTAMP AS annotated_at, ? AS operation_id "
            "FROM selected_records AS selected CROSS JOIN requested_labels AS requested "
            "ORDER BY selected.record_id, requested.group_id, requested.label_id) "
            f"TO {_sql_literal(delta_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)",
            [action, annotator, annotated_at, operation_id],
        )
        if action == "apply":
            connection.execute(
                "DELETE FROM effective_labels AS effective USING selected_records AS selected, "
                "requested_labels AS requested WHERE effective.record_id = selected.record_id "
                "AND effective.scheme_version_id = requested.scheme_version_id "
                "AND effective.group_id = requested.group_id "
                "AND (requested.selection_mode = 'single' OR effective.label_id = requested.label_id)"
            )
            connection.execute(
                "INSERT INTO effective_labels "
                "SELECT selected.record_id, selected.record_hash, requested.scheme_id, "
                "requested.scheme_code, requested.scheme_version_id, requested.scheme_version, "
                "requested.group_id, requested.group_code, requested.label_id, requested.label_code, "
                "?, ?::TIMESTAMP, ? FROM selected_records AS selected "
                "CROSS JOIN requested_labels AS requested",
                [annotator, annotated_at, operation_id],
            )
        else:
            connection.execute(
                "DELETE FROM effective_labels AS effective USING selected_records AS selected, "
                "requested_labels AS requested WHERE effective.record_id = selected.record_id "
                "AND effective.scheme_version_id = requested.scheme_version_id "
                "AND effective.group_id = requested.group_id "
                "AND effective.label_id = requested.label_id"
            )
        connection.execute(
            "COPY (SELECT * FROM selected_records ORDER BY record_id) "
            f"TO {_sql_literal(selection_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            "COPY (SELECT * FROM effective_labels ORDER BY record_id, scheme_version_id, group_id, label_id) "
            f"TO {_sql_literal(snapshot_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        assignment_count = int(
            connection.execute("SELECT count(*) FROM effective_labels").fetchone()[0]
        )
        labeled_record_count = int(
            connection.execute("SELECT count(DISTINCT record_id) FROM effective_labels").fetchone()[0]
        )
        return {
            "engine": LABELER_ENGINE,
            "scope": scope,
            "action": action,
            "split": selected_split,
            "selection_count": selection_count,
            "assignment_count": assignment_count,
            "labeled_record_count": labeled_record_count,
            "outputs": [
                {"ref": "label_selection", "path": str(selection_path), "kind": "data_label_selection", "media_type": "application/vnd.apache.parquet"},
                {"ref": "label_delta", "path": str(delta_path), "kind": "data_label_delta", "media_type": "application/vnd.apache.parquet"},
                {"ref": "label_snapshot", "path": str(snapshot_path), "kind": "data_label_snapshot", "media_type": "application/vnd.apache.parquet"},
            ],
        }
    finally:
        connection.close()


class DatasetLabelerAdapter:
    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime_context = task.get("runtime_context") or {}
        if get_desired_state and get_desired_state() in {"cancelled", "paused"}:
            return {"controlled_state": get_desired_state()}
        await emit("progress", {"progress": 0.1, "phase": "freezing_label_selection"})
        result = await asyncio.to_thread(
            execute_label_operation,
            dataset_path=Path(str((payload.get("dataset_snapshot_input") or {}).get("runtime_path") or "")).resolve(),
            manifest_path=Path(str((payload.get("record_manifest_input") or {}).get("runtime_path") or "")).resolve(),
            base_snapshot_path=(
                Path(str((payload.get("label_snapshot_input") or {}).get("runtime_path"))).resolve()
                if (payload.get("label_snapshot_input") or {}).get("runtime_path")
                else None
            ),
            logical_files=[str(item) for item in (payload.get("dataset_snapshot_input") or {}).get("logical_files") or []],
            source_query=payload.get("source_query") or {},
            scope=str(payload.get("scope") or "selected"),
            explicit_records=list(payload.get("records") or []),
            action=str(payload.get("action") or "apply"),
            labels=list(payload.get("labels") or []),
            annotator=str(payload.get("annotator") or "local-user"),
            operation_id=str(payload.get("operation_id") or task.get("id") or ""),
            output_dir=Path(str(runtime_context.get("attempt_dir") or ".")).resolve() / "outputs" / "data-labeling",
        )
        await emit(
            "progress",
            {
                "progress": 1.0,
                "phase": "data_labeling_complete",
                "selection_count": result["selection_count"],
                "assignment_count": result["assignment_count"],
            },
        )
        return result
