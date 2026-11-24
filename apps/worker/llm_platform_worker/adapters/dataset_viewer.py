from __future__ import annotations

import asyncio
import base64
import datetime as dt
import decimal
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
GetDesiredState = Callable[[], str]

VIEWER_ENGINE = "duckdb_dataset_viewer_v2"
SUPPORTED_SUFFIXES = {".parquet", ".pq", ".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
FORMAT_PRIORITY = ("parquet", "json", "csv")
MAX_SQL_CHARS = 20_000
MAX_CELL_CHARS = 64_000
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_FILES = 10_000
IGNORED_METADATA_DIRECTORIES = {".cache", ".git", ".hg", ".svn", "__pycache__"}
_SPLIT_PATTERN = re.compile(
    r"(?:^|[-_./])(train|training|test|validation|valid|dev)(?:[-_./]|$)", re.IGNORECASE
)
_LEADING_COMMENT = re.compile(r"\A\s*(?:(?:--[^\r\n]*(?:\r?\n|\Z))|(?:/\*.*?\*/\s*))*", re.DOTALL)


class DatasetQueryInterrupted(RuntimeError):
    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__(f"dataset query {state}")


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_string_list(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_literal(path.as_posix()) for path in paths) + "]"


def _format_family(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    return None


def _split_name(path: Path, root: Path) -> str:
    try:
        logical = path.relative_to(root).as_posix()
    except ValueError:
        logical = path.name
    match = _SPLIT_PATTERN.search(logical)
    if not match:
        return "default"
    value = match.group(1).casefold()
    if value == "training":
        return "train"
    if value in {"valid", "dev"}:
        return "validation"
    return value


def _frozen_snapshot_files(root: Path, logical_files: list[str]) -> list[Path]:
    resolved_root = root.resolve()
    if root.is_file():
        return [resolved_root]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in logical_files:
        logical_path = PurePosixPath(str(raw_path).replace("\\", "/"))
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise ValueError(f"Dataset viewer file escapes snapshot root: {raw_path}")
        candidate = (resolved_root / Path(*logical_path.parts)).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Dataset viewer file escapes snapshot root: {raw_path}") from exc
        if candidate.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        if not candidate.is_file():
            raise FileNotFoundError(f"Frozen dataset viewer file is missing: {raw_path}")
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return candidates


def _discovered_snapshot_files(root: Path) -> list[Path]:
    resolved_root = root.resolve()
    if root.is_file():
        return [resolved_root]
    return [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_SUFFIXES
        and not any(
            part.casefold() in IGNORED_METADATA_DIRECTORIES
            for part in path.relative_to(resolved_root).parts[:-1]
        )
    ]


def discover_dataset_files(
    root: Path, logical_files: list[str] | None = None
) -> dict[str, dict[str, list[Path]]]:
    """Return supported snapshot files grouped by split and format."""
    resolved_root = root.resolve()
    supported = (
        _frozen_snapshot_files(root, logical_files)
        if logical_files
        else _discovered_snapshot_files(root)
    )
    if len(supported) > MAX_FILES:
        raise ValueError(f"Dataset viewer supports at most {MAX_FILES:,} source files per snapshot")
    grouped: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(supported, key=lambda item: item.as_posix().casefold()):
        if path != resolved_root and resolved_root not in path.parents:
            raise ValueError("Dataset snapshot contains a file outside its materialized root")
        family = _format_family(path)
        if family is None:
            continue
        split = _split_name(path, resolved_root if root.is_dir() else resolved_root.parent)
        grouped.setdefault(split, {}).setdefault(family, []).append(path)
    return grouped


def _select_files(
    grouped: dict[str, dict[str, list[Path]]], requested_split: str
) -> tuple[str, str, list[Path], list[str]]:
    splits = sorted(grouped, key=lambda value: (value != "train", value != "default", value))
    if not splits:
        raise ValueError("Dataset snapshot does not contain Parquet, JSON, CSV, or TSV data files")
    split = requested_split.strip()
    if not split:
        split = "train" if "train" in grouped else "default" if "default" in grouped else splits[0]
    if split not in grouped:
        raise ValueError(f"Unknown dataset split: {split}")
    by_format = grouped[split]
    family = next((value for value in FORMAT_PRIORITY if by_format.get(value)), "")
    if not family:
        raise ValueError(f"Dataset split {split} has no supported structured data files")
    return split, family, by_format[family], splits


def _source_expression(family: str, files: list[Path]) -> str:
    values = _sql_string_list(files)
    if family == "parquet":
        return f"read_parquet({values}, union_by_name = true)"
    if family == "json":
        return f"read_json_auto({values}, union_by_name = true, format = 'auto')"
    delimiter = "\t" if all(path.suffix.casefold() == ".tsv" for path in files) else ","
    return f"read_csv_auto({values}, union_by_name = true, delim = {_sql_literal(delimiter)})"


def _validate_sql(connection: Any, value: str) -> str:
    sql = value.strip() or "SELECT * FROM dataset"
    if len(sql) > MAX_SQL_CHARS:
        raise ValueError(f"SQL cannot exceed {MAX_SQL_CHARS:,} characters")
    leading = _LEADING_COMMENT.sub("", sql).lstrip()
    first_keyword = re.match(r"[A-Za-z]+", leading)
    if first_keyword is None or first_keyword.group(0).casefold() not in {"select", "with"}:
        raise ValueError("Dataset viewer only accepts one SELECT or WITH query")
    statements = connection.extract_statements(sql)
    if len(statements) != 1 or str(statements[0].type) != "StatementType.SELECT":
        raise ValueError("Dataset viewer only accepts one read-only SELECT statement")
    return sql.rstrip().rstrip(";").rstrip()


def _json_safe(value: Any, *, max_chars: int = MAX_CELL_CHARS) -> tuple[Any, bool]:
    if value is None or isinstance(value, (bool, int, str)):
        normalized: Any = value
    elif isinstance(value, float):
        normalized = value if math.isfinite(value) else str(value)
    elif isinstance(value, (dt.date, dt.time, dt.datetime, uuid.UUID, decimal.Decimal)):
        normalized = str(value)
    elif isinstance(value, bytes):
        normalized = {"$binary": base64.b64encode(value).decode("ascii")}
    elif isinstance(value, dict):
        normalized = {str(key): _json_safe(child, max_chars=max_chars)[0] for key, child in value.items()}
    elif isinstance(value, (list, tuple)):
        normalized = [_json_safe(child, max_chars=max_chars)[0] for child in value]
    else:
        normalized = str(value)
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(encoded) <= max_chars:
        return normalized, False
    return encoded[:max_chars] + "…", True


def execute_dataset_query(
    *,
    input_path: Path,
    sql: str,
    split: str,
    offset: int,
    limit: int,
    timeout_seconds: int,
    temp_dir: Path,
    search: str = "",
    logical_files: list[str] | None = None,
    record_manifest_path: Path | None = None,
    label_snapshot_path: Path | None = None,
    get_desired_state: GetDesiredState | None = None,
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Dataset viewer requires DuckDB") from exc

    root = input_path.resolve()
    if not root.exists():
        raise FileNotFoundError("Frozen dataset snapshot was not materialized")
    grouped = discover_dataset_files(root, logical_files)
    selected_split, family, files, splits = _select_files(grouped, split)
    allowed_root = root if root.is_dir() else root.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    finished = threading.Event()
    interrupted: dict[str, str] = {}
    deadline = time.monotonic() + timeout_seconds

    def monitor() -> None:
        while not finished.wait(0.1):
            desired = get_desired_state() if get_desired_state is not None else "running"
            if desired in {"cancelled", "paused"}:
                interrupted["state"] = desired
                connection.interrupt()
                return
            if time.monotonic() >= deadline:
                interrupted["state"] = "timed_out"
                connection.interrupt()
                return

    monitor_thread = threading.Thread(target=monitor, name="dataset-viewer-timeout", daemon=True)
    started = time.perf_counter()
    identity_aligned = False
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '512 MiB'")
        connection.execute(f"SET temp_directory = {_sql_literal(temp_dir.as_posix())}")
        allowed_directories = [allowed_root, temp_dir.resolve()]
        if record_manifest_path is not None:
            allowed_directories.append(record_manifest_path.resolve().parent)
        if label_snapshot_path is not None:
            allowed_directories.append(label_snapshot_path.resolve().parent)
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
        source_expression = _source_expression(family, files)
        if record_manifest_path is not None:
            selected_logical_files = [
                path.relative_to(allowed_root).as_posix() for path in files
            ]
            manifest_filter = "(" + ", ".join(
                _sql_literal(value) for value in selected_logical_files
            ) + ")"
            connection.execute(
                "CREATE TEMP VIEW _forgelm_manifest AS "
                "SELECT record_id, content_hash AS record_hash, logical_path, record_index, "
                "row_number() OVER (ORDER BY logical_path, record_index) AS _forgelm_ordinal "
                f"FROM read_json_auto({_sql_literal(record_manifest_path.as_posix())}, "
                "format = 'newline_delimited', ignore_errors = true) "
                "WHERE record_id IS NOT NULL "
                f"AND logical_path IN {manifest_filter}"
            )
            label_join = ""
            label_column = "[] AS _forgelm_labels"
            if label_snapshot_path is not None:
                connection.execute(
                    "CREATE TEMP VIEW _forgelm_label_lists AS "
                    "SELECT record_id, list(scheme_code || ':' || group_code || ':' || label_code "
                    "ORDER BY scheme_code, group_code, label_code) AS labels "
                    f"FROM read_parquet({_sql_literal(label_snapshot_path.as_posix())}) "
                    "GROUP BY record_id"
                )
                label_join = (
                    " LEFT JOIN _forgelm_label_lists AS label_lists "
                    "ON label_lists.record_id = manifest.record_id"
                )
                label_column = (
                    "CASE WHEN label_lists.labels IS NULL THEN [] ELSE label_lists.labels END "
                    "AS _forgelm_labels"
                )
            connection.execute(
                "CREATE VIEW dataset AS "
                "WITH source AS ("
                f"SELECT *, row_number() OVER () AS _forgelm_ordinal FROM {source_expression}"
                ") SELECT source.* EXCLUDE (_forgelm_ordinal), "
                "manifest.record_id AS _forgelm_record_id, "
                "manifest.record_hash AS _forgelm_record_hash, "
                f"{label_column} "
                "FROM source LEFT JOIN _forgelm_manifest AS manifest "
                "USING (_forgelm_ordinal)"
                f"{label_join}"
            )
            source_count = int(connection.execute("SELECT count(*) FROM dataset").fetchone()[0])
            manifest_count = int(
                connection.execute("SELECT count(*) FROM _forgelm_manifest").fetchone()[0]
            )
            identity_aligned = source_count == manifest_count
        else:
            connection.execute(f"CREATE VIEW dataset AS SELECT * FROM {source_expression}")
        base_query = _validate_sql(connection, sql)
        query = base_query
        if search:
            query = (
                f"SELECT * FROM ({base_query}) AS viewer_search WHERE "
                "contains(lower(CAST(to_json(viewer_search) AS VARCHAR)), "
                f"lower({_sql_literal(search)}))"
            )
        connection.execute("SET lock_configuration = true")
        monitor_thread.start()
        total_rows = int(
            connection.execute(f"SELECT count(*)::BIGINT FROM ({query}) AS viewer_count").fetchone()[0]
        )
        cursor = connection.execute(
            f"SELECT * FROM ({query}) AS viewer_page LIMIT ? OFFSET ?", [limit, offset]
        )
        description = cursor.description or []
        raw_columns = [
            {"name": str(item[0]), "type": str(item[1]) if len(item) > 1 else "UNKNOWN"}
            for item in description
        ]
        identity_names = {"_forgelm_record_id", "_forgelm_record_hash"}
        visible_indexes = [
            index for index, column in enumerate(raw_columns) if column["name"] not in identity_names
        ]
        identity_indexes = {
            column["name"]: index
            for index, column in enumerate(raw_columns)
            if column["name"] in identity_names
        }
        columns = [raw_columns[index] for index in visible_indexes]
        rows: list[list[Any]] = []
        row_identities: list[dict[str, str]] = []
        truncated_cells = 0
        result_bytes = 0
        result_limited = False
        identity_complete = identity_aligned and len(identity_indexes) == 2
        for raw_row in cursor.fetchall():
            row: list[Any] = []
            if identity_complete:
                record_id = raw_row[identity_indexes["_forgelm_record_id"]]
                record_hash = raw_row[identity_indexes["_forgelm_record_hash"]]
                identity_complete = bool(record_id and record_hash)
                if identity_complete:
                    row_identities.append(
                        {"record_id": str(record_id), "record_hash": str(record_hash)}
                    )
            for index in visible_indexes:
                value = raw_row[index]
                normalized, truncated = _json_safe(value)
                encoded_size = len(
                    json.dumps(normalized, ensure_ascii=False, default=str).encode("utf-8")
                )
                if result_bytes + encoded_size > MAX_RESULT_BYTES:
                    result_limited = True
                    break
                result_bytes += encoded_size
                truncated_cells += int(truncated)
                row.append(normalized)
            if result_limited:
                break
            rows.append(row)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "engine": VIEWER_ENGINE,
            "sql": query,
            "base_sql": base_query,
            "search": search,
            "configuration": "default",
            "configurations": ["default"],
            "split": selected_split,
            "splits": splits,
            "format": family,
            "files": [path.relative_to(allowed_root).as_posix() for path in files],
            "columns": columns,
            "rows": rows,
            "row_identities": row_identities if identity_complete else [],
            "labelable": bool(identity_complete and len(row_identities) == len(rows)),
            "labelability_reason": (
                ""
                if identity_complete and len(row_identities) == len(rows)
                else "查询必须保留 dataset 的 _forgelm_record_id 与 _forgelm_record_hash 列，且每行需映射到稳定记录"
            ),
            "offset": offset,
            "limit": limit,
            "total_rows": total_rows,
            "has_more": offset + len(rows) < total_rows,
            "truncated_cells": truncated_cells,
            "result_limited": result_limited,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        state = interrupted.get("state")
        if state in {"cancelled", "paused"}:
            raise DatasetQueryInterrupted(state) from exc
        if state == "timed_out":
            raise TimeoutError(f"Dataset query exceeded {timeout_seconds} seconds") from exc
        raise
    finally:
        finished.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=1)
        connection.close()


class DatasetViewerAdapter:
    """Run bounded read-only SQL against a frozen structured dataset snapshot."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        viewer_input = payload.get("dataset_viewer_input") or {}
        runtime_context = task.get("runtime_context") or {}
        input_path = Path(str(viewer_input.get("runtime_path") or "")).resolve()
        attempt_dir = Path(str(runtime_context.get("attempt_dir") or ".")).resolve()
        offset = min(max(int(payload.get("offset") or 0), 0), 10_000_000)
        limit = min(max(int(payload.get("limit") or 100), 1), 500)
        timeout_seconds = min(max(int(payload.get("timeout_seconds") or 30), 1), 60)
        frozen_files = viewer_input.get("logical_files")
        if frozen_files is not None and not isinstance(frozen_files, list):
            raise ValueError("Frozen dataset viewer file list must be an array")
        logical_files = [str(value) for value in frozen_files or [] if str(value).strip()]
        manifest_path_value = str((payload.get("record_manifest_input") or {}).get("runtime_path") or "")
        label_snapshot_value = str((payload.get("label_snapshot_input") or {}).get("runtime_path") or "")
        await emit("progress", {"progress": 0.1, "phase": "preparing_dataset_viewer"})
        try:
            result = await asyncio.to_thread(
                execute_dataset_query,
                input_path=input_path,
                sql=str(payload.get("sql") or "SELECT * FROM dataset"),
                split=str(payload.get("split") or ""),
                offset=offset,
                limit=limit,
                timeout_seconds=timeout_seconds,
                temp_dir=attempt_dir / "duckdb-temp",
                search=str(payload.get("search") or ""),
                logical_files=logical_files,
                record_manifest_path=Path(manifest_path_value).resolve() if manifest_path_value else None,
                label_snapshot_path=Path(label_snapshot_value).resolve() if label_snapshot_value else None,
                get_desired_state=get_desired_state,
            )
        except DatasetQueryInterrupted as exc:
            await emit("progress", {"progress": 0.1, "phase": exc.state})
            return {"controlled_state": exc.state}
        await emit(
            "progress",
            {
                "progress": 1.0,
                "phase": "dataset_query_complete",
                "row_count": len(result["rows"]),
                "total_rows": result["total_rows"],
            },
        )
        return result
