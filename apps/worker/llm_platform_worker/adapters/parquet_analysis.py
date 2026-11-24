from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from llm_platform_worker.file_cache import (
    advise_sequential,
    release_file_cache_descriptor,
)
from llm_platform_worker.memory_budget import (
    PARQUET_PROCESS_MEMORY_BYTES,
    duckdb_memory_limit,
    parquet_memory_worker_limit,
)

TOKEN_ESTIMATE_METHOD = "byte_delta_cjk_v2"
PARQUET_ENGINE = "duckdb_native_v1"


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _text_expression(fields: list[str]) -> str:
    columns = [_sql_identifier(field) for field in fields]
    if len(columns) == 1:
        return f"cast({columns[0]} AS varchar)"
    return (
        "concat_ws(chr(10), " + ", ".join(f"cast({column} AS varchar)" for column in columns) + ")"
    )


def _sha256_file(path: Path, cancel_path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        advise_sequential(stream.fileno())
        released_offset = 0
        while chunk := stream.read(8 * 1024 * 1024):
            if cancel_path.exists():
                raise RuntimeError("analysis interrupted")
            digest.update(chunk)
            current_offset = stream.tell()
            if current_offset - released_offset >= 64 * 1024 * 1024:
                release_file_cache_descriptor(
                    stream.fileno(), released_offset, current_offset - released_offset
                )
                released_offset = current_offset
        release_file_cache_descriptor(stream.fileno(), released_offset, 0)
    return digest.hexdigest()


def _start_interrupt_monitor(
    connection: Any, cancel_path: Path
) -> tuple[threading.Event, threading.Thread]:
    finished = threading.Event()

    def monitor() -> None:
        while not finished.wait(0.2):
            if cancel_path.exists():
                connection.interrupt()
                return

    thread = threading.Thread(target=monitor, name="duckdb-analysis-cancel", daemon=True)
    thread.start()
    return finished, thread


def analyze_parquet_file(job: dict[str, Any]) -> dict[str, Any]:
    """Build one deterministic manifest shard without materializing document rows in Python."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("High-throughput Parquet analysis requires DuckDB") from exc

    path = Path(str(job["path"]))
    shard_path = Path(str(job["shard_path"]))
    cancel_path = Path(str(job["cancel_path"]))
    temp_dir = Path(str(job["temp_dir"]))
    logical_path = str(job["logical_path"])
    dataset_version_id = str(job["dataset_version_id"])
    text_fields = [str(value) for value in job["text_fields"]]
    quality_budget = max(int(job.get("quality_budget") or 0), 0)
    expected_size = int(job.get("known_size") or 0)
    known_hash = str(job.get("known_hash") or "")
    actual_size = path.stat().st_size
    reused_hash = bool(known_hash and expected_size == actual_size)
    file_hash = known_hash if reused_hash else _sha256_file(path, cancel_path)
    if cancel_path.exists():
        raise RuntimeError("analysis interrupted")

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.unlink(missing_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    finished, monitor = _start_interrupt_monitor(connection, cancel_path)
    started = time.perf_counter()
    try:
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = true")
        connection.execute(
            f"SET memory_limit = '{duckdb_memory_limit(PARQUET_PROCESS_MEMORY_BYTES)}'"
        )
        connection.execute(f"SET temp_directory = {_sql_literal(temp_dir.as_posix())}")
        text_expression = _text_expression(text_fields)
        source = f"read_parquet({_sql_literal(path.as_posix())})"
        connection.execute(
            f"""
            CREATE TEMP TABLE manifest AS
            WITH filtered AS (
                SELECT {text_expression} AS text
                FROM {source}
                WHERE {text_expression} IS NOT NULL AND {text_expression} <> ''
            ), indexed AS (
                SELECT row_number() OVER () - 1 AS record_index, text
                FROM filtered
            ), measured AS (
                SELECT
                    record_index,
                    text,
                    sha256(text) AS content_hash,
                    length(text) AS character_length,
                    strlen(text) AS byte_length
                FROM indexed
            ), estimated AS (
                SELECT *, least(
                    character_length,
                    greatest(0, (byte_length - character_length) // 2)
                ) AS cjk_estimate
                FROM measured
            )
            SELECT
                'record' AS type,
                sha256(
                    {_sql_literal(dataset_version_id)} || chr(0) ||
                    {_sql_literal(logical_path)} || chr(0) ||
                    cast(record_index AS varchar) || chr(0) ||
                    content_hash || chr(0)
                ) AS record_id,
                {_sql_literal(logical_path)} AS logical_path,
                record_index,
                content_hash,
                character_length,
                byte_length,
                cjk_estimate + ((character_length - cjk_estimate + 3) // 4) AS token_estimate,
                CASE
                    WHEN contains(text, 'def ') OR contains(text, 'class ')
                        OR contains(text, 'function ') OR contains(text, '=>') THEN 'code'
                    WHEN cjk_estimate > (character_length - cjk_estimate) * 0.25 THEN 'zh'
                    WHEN regexp_matches(text, '[A-Za-z]') THEN 'en'
                    ELSE 'other'
                END AS language
            FROM estimated
            """
        )
        if cancel_path.exists():
            raise RuntimeError("analysis interrupted")
        summary_row = connection.execute(
            """
            SELECT
                count(*)::BIGINT,
                coalesce(sum(token_estimate), 0)::BIGINT,
                coalesce(sum(character_length), 0)::BIGINT,
                count(*) FILTER (WHERE language = 'zh')::BIGINT,
                count(*) FILTER (WHERE language = 'en')::BIGINT,
                count(*) FILTER (WHERE language = 'code')::BIGINT,
                count(*) FILTER (WHERE language = 'other')::BIGINT,
                count(*) FILTER (WHERE character_length <= 128)::BIGINT,
                count(*) FILTER (WHERE character_length BETWEEN 129 AND 512)::BIGINT,
                count(*) FILTER (WHERE character_length BETWEEN 513 AND 2048)::BIGINT,
                count(*) FILTER (WHERE character_length > 2048)::BIGINT
            FROM manifest
            """
        ).fetchone()
        assert summary_row is not None
        (
            document_count,
            token_estimate,
            total_chars,
            language_zh,
            language_en,
            language_code,
            language_other,
            length_short,
            length_medium,
            length_long,
            length_extra,
        ) = [int(value) for value in summary_row]

        quality = {
            "evaluated_documents": 0,
            "duplicates": 0,
            "pii_documents": 0,
            "ad_documents": 0,
            "replacement_chars": 0,
            "total_chars": 0,
        }
        samples: list[dict[str, Any]] = []
        sample_limit = min(quality_budget, document_count)
        if sample_limit:
            duplicate_row = connection.execute(
                f"""
                SELECT count(*) - count(DISTINCT content_hash)
                FROM (SELECT content_hash FROM manifest LIMIT {sample_limit})
                """
            ).fetchone()
            quality_sql = f"""
                WITH filtered AS (
                    SELECT {text_expression} AS text
                    FROM {source}
                    WHERE {text_expression} IS NOT NULL AND {text_expression} <> ''
                    LIMIT {sample_limit}
                )
                SELECT
                    count(*)::BIGINT,
                    count(*) FILTER (WHERE
                        regexp_matches(text, '[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{{2,}}', 'i')
                        OR regexp_matches(text, '(\\+?86[- ]?)?1[3-9][0-9]{{9}}')
                    )::BIGINT,
                    count(*) FILTER (WHERE
                        regexp_matches(text, 'https?://|www[.]', 'i')
                        AND regexp_matches(text, 'buy|sale|discount|点击|优惠|购买', 'i')
                    )::BIGINT,
                    coalesce(sum(length(text) - length(replace(text, '�', ''))), 0)::BIGINT,
                    coalesce(sum(length(text)), 0)::BIGINT
                FROM filtered
            """
            quality_row = connection.execute(quality_sql).fetchone()
            assert quality_row is not None
            quality = {
                "evaluated_documents": int(quality_row[0]),
                "duplicates": int(duplicate_row[0] if duplicate_row else 0),
                "pii_documents": int(quality_row[1]),
                "ad_documents": int(quality_row[2]),
                "replacement_chars": int(quality_row[3]),
                "total_chars": int(quality_row[4]),
            }
            sample_rows = connection.execute(
                f"""
                SELECT {text_expression} AS text
                FROM {source}
                WHERE {text_expression} IS NOT NULL AND {text_expression} <> ''
                LIMIT {min(sample_limit, 50)}
                """
            ).fetchall()
            samples = [
                {
                    "sample_key": f"{logical_path}:{index + 1}",
                    "source_file": logical_path,
                    "raw_text": str(row[0])[:4000],
                    "cleaned_text": str(row[0])[:4000],
                }
                for index, row in enumerate(sample_rows)
            ]

        connection.execute(
            f"""
            COPY (
                SELECT * FROM manifest ORDER BY record_index
            ) TO {_sql_literal(shard_path.as_posix())}
            (FORMAT JSON, ARRAY false, COMPRESSION gzip)
            """
        )
        if cancel_path.exists():
            raise RuntimeError("analysis interrupted")
        return {
            "logical_path": logical_path,
            "shard_path": str(shard_path),
            "file_record": {
                "path": logical_path,
                "size_bytes": actual_size,
                "sha256": file_hash,
                "hash_reused": reused_hash,
            },
            "file_detail": {
                "path": logical_path,
                "format": "parquet",
                "document_count": document_count,
                "text_fields": text_fields,
                "engine": PARQUET_ENGINE,
            },
            "document_count": document_count,
            "token_estimate": token_estimate,
            "total_chars": total_chars,
            "language_counts": {
                "zh": language_zh,
                "en": language_en,
                "code": language_code,
                "other": language_other,
            },
            "length_counts": {
                "0-128": length_short,
                "129-512": length_medium,
                "513-2048": length_long,
                "2049+": length_extra,
            },
            "quality": quality,
            "samples": samples,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception:
        shard_path.unlink(missing_ok=True)
        raise
    finally:
        finished.set()
        monitor.join(timeout=1)
        connection.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def recommended_processes(
    file_count: int,
    configured: object = None,
    available_memory: int | None = None,
) -> int:
    logical_cpus = os.cpu_count() or 1
    default = max(logical_cpus - 2, 1)
    requested = int(configured) if configured not in {None, ""} else default
    return max(
        1,
        min(
            requested,
            logical_cpus,
            18,
            max(file_count, 1),
            parquet_memory_worker_limit(available_memory),
        ),
    )
