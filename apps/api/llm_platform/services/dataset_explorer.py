from __future__ import annotations

import fnmatch
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import Dataset, DatasetVersion
from ..schemas import DatasetExplorerQuery
from .artifacts import artifact_storage
from .dataset_analyses import effective_record_manifest, effective_sample_preview


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _fnmatch_regex(pattern: str) -> str:
    translated = fnmatch.translate(pattern.replace("\\", "/").lstrip("/"))
    if translated.startswith("(?s:") and translated.endswith(")\\Z"):
        return translated[4:-3]
    return translated.removesuffix("\\Z")


def _path_predicate(patterns: list[str]) -> str:
    normalized_path = "ltrim(replace(logical_path, chr(92), '/'), '/')"
    return " OR ".join(
        f"regexp_full_match({normalized_path}, {_sql_literal(_fnmatch_regex(pattern))})"
        for pattern in patterns
    )


def _manifest_source(path: Path) -> str:
    return f"""
        read_json(
            {_sql_literal(path.as_posix())},
            format = 'newline_delimited',
            compression = 'auto_detect',
            columns = {{
                type: 'VARCHAR', record_id: 'VARCHAR', logical_path: 'VARCHAR',
                record_index: 'BIGINT', content_hash: 'VARCHAR',
                character_length: 'BIGINT', byte_length: 'BIGINT',
                token_estimate: 'BIGINT', language: 'VARCHAR'
            }}
        )
    """


def _source_query(
    *,
    dataset: Dataset,
    version: DatasetVersion,
    manifest_path: Path,
    payload: DatasetExplorerQuery,
) -> str:
    predicates = [
        "type = 'record'",
        "record_id IS NOT NULL",
        f"coalesce(character_length, 0) >= {payload.min_length}",
    ]
    if payload.max_length:
        predicates.append(f"coalesce(character_length, 0) <= {payload.max_length}")
    if payload.include_paths:
        predicates.append(f"({_path_predicate(payload.include_paths)})")
    if payload.exclude_paths:
        predicates.append(f"NOT ({_path_predicate(payload.exclude_paths)})")
    if payload.languages:
        values = ", ".join(_sql_literal(language) for language in payload.languages)
        predicates.append(f"lower(coalesce(language, 'other')) IN ({values})")
    order_key = (
        f"sha256({_sql_literal(payload.seed)} || chr(0) || "
        f"{_sql_literal(version.id)} || chr(0) || record_id || chr(0))"
    )
    if payload.sample_ratio <= 0:
        predicates.append("false")
    elif payload.sample_ratio < 1:
        ratio_limit = int(payload.sample_ratio * (1 << 256))
        predicates.append(f"{order_key} < {_sql_literal(f'{ratio_limit:064x}')}")
    return f"""
        SELECT
            {_sql_literal(dataset.id)} AS dataset_id,
            {_sql_literal(dataset.name)} AS dataset_name,
            {_sql_literal(version.id)} AS dataset_version_id,
            {version.version}::BIGINT AS dataset_version,
            record_id,
            logical_path,
            record_index,
            content_hash,
            coalesce(character_length, 0)::BIGINT AS character_length,
            coalesce(byte_length, 0)::BIGINT AS byte_length,
            coalesce(token_estimate, 0)::BIGINT AS token_estimate,
            coalesce(language, 'other') AS language,
            {order_key} AS order_key
        FROM {_manifest_source(manifest_path)}
        WHERE {' AND '.join(predicates)}
    """


def query_dataset_records(
    session: Session,
    *,
    access_project_id: str,
    payload: DatasetExplorerQuery,
) -> dict[str, Any]:
    sources: list[str] = []
    previews: dict[tuple[str, str, int], dict[str, Any]] = {}
    missing: list[str] = []
    for version_id in payload.dataset_version_ids:
        version = session.get(DatasetVersion, version_id)
        dataset = session.get(Dataset, version.dataset_id) if version else None
        manifest = effective_record_manifest(session, version) if version else None
        path = (
            artifact_storage.managed_payload_path(
                session, manifest, access_project_id
            )
            if manifest
            else None
        )
        if version is None or dataset is None or manifest is None or path is None:
            missing.append(version_id)
            continue
        sources.append(
            _source_query(
                dataset=dataset,
                version=version,
                manifest_path=path,
                payload=payload,
            )
        )
        for sample in effective_sample_preview(session, version) or []:
            logical_path = str(sample.get("source_file") or "")
            sample_key = str(sample.get("sample_key") or "")
            try:
                record_index = int(sample_key.rsplit(":", 1)[1]) - 1
            except (IndexError, TypeError, ValueError):
                continue
            previews[(version.id, logical_path, record_index)] = sample
    if missing:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_manifest_unavailable",
                "message": "部分数据版本没有控制面可读的稳定行索引",
                "dataset_version_ids": missing,
            },
        )
    if not sources:
        return {
            "rows": [],
            "total_rows": 0,
            "total_tokens": 0,
            "sources": [],
            "selection": payload.mixture_selection(),
        }
    union = " UNION ALL ".join(f"({source})" for source in sources)
    with TemporaryDirectory(prefix="forgelm-duckdb-explorer-") as temp_dir:
        connection = duckdb.connect()
        try:
            connection.execute("SET threads = 4")
            connection.execute("SET memory_limit = '512 MiB'")
            connection.execute(
                f"SET temp_directory = {_sql_literal(Path(temp_dir).as_posix())}"
            )
            connection.execute(f"CREATE TEMP VIEW candidates AS {union}")
            summary = connection.execute(
                """
                SELECT count(*)::BIGINT, coalesce(sum(token_estimate), 0)::BIGINT
                FROM candidates
                """
            ).fetchone()
            source_rows = connection.execute(
                """
                SELECT dataset_version_id, dataset_name, dataset_version,
                       count(*)::BIGINT, coalesce(sum(token_estimate), 0)::BIGINT
                FROM candidates
                GROUP BY dataset_version_id, dataset_name, dataset_version
                ORDER BY dataset_name, dataset_version
                """
            ).fetchall()
            rows = connection.execute(
                f"""
                SELECT dataset_id, dataset_name, dataset_version_id, dataset_version,
                       record_id, logical_path, record_index, content_hash,
                       character_length, byte_length, token_estimate, language
                FROM candidates
                ORDER BY order_key, dataset_version_id, record_id
                LIMIT {payload.limit} OFFSET {payload.offset}
                """
            ).fetchall()
        finally:
            connection.close()
    columns = [
        "dataset_id",
        "dataset_name",
        "dataset_version_id",
        "dataset_version",
        "record_id",
        "logical_path",
        "record_index",
        "content_hash",
        "character_length",
        "byte_length",
        "token_estimate",
        "language",
    ]
    result_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        sample = previews.get(
            (
                str(item["dataset_version_id"]),
                str(item["logical_path"]),
                int(item["record_index"]),
            )
        )
        if sample:
            item["preview"] = {
                "raw_text": sample.get("raw_text"),
                "cleaned_text": sample.get("cleaned_text"),
            }
        result_rows.append(item)
    return {
        "rows": result_rows,
        "total_rows": int(summary[0] if summary else 0),
        "total_tokens": int(summary[1] if summary else 0),
        "offset": payload.offset,
        "limit": payload.limit,
        "sources": [
            {
                "dataset_version_id": str(row[0]),
                "dataset_name": str(row[1]),
                "dataset_version": int(row[2]),
                "document_count": int(row[3]),
                "token_estimate": int(row[4]),
            }
            for row in source_rows
        ],
        "selection": payload.mixture_selection(),
        "engine": "duckdb_record_manifest_v1",
        "materializable_as_mixture": not payload.languages,
    }
