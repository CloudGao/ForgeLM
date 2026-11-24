from __future__ import annotations

import fnmatch
import gzip
import io
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from llm_platform_core.mixtures import canonical_json

from llm_platform_worker.memory_budget import (
    duckdb_memory_budget_bytes,
    duckdb_memory_limit,
)

MIXTURE_BATCH_SIZE = 65_536


@contextmanager
def _open_text_output(path: Path):
    if path.suffix.lower() != ".gz":
        with path.open("w", encoding="utf-8", newline="\n") as output:
            yield output
        return
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                yield output


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


def _start_monitor(
    connection: Any,
    cancel_path: Path,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
    input_bytes: int,
    scan_base_bytes: int,
) -> tuple[threading.Event, threading.Thread]:
    finished = threading.Event()

    def monitor() -> None:
        while not finished.wait(0.25):
            if cancel_path.exists():
                connection.interrupt()
                return
            raw_progress = float(connection.query_progress())
            if raw_progress < 0:
                continue
            fraction = raw_progress / 100 if raw_progress > 1 else raw_progress
            with progress_lock:
                if progress.get("phase") == "writing_manifest" and progress.get("total_rows"):
                    progress["processed_rows"] = min(
                        int(progress["total_rows"]),
                        max(int(int(progress["total_rows"]) * fraction), 0),
                    )
                else:
                    progress["processed_bytes"] = min(
                        scan_base_bytes + input_bytes,
                        scan_base_bytes + max(int(input_bytes * fraction), 0),
                    )

    thread = threading.Thread(target=monitor, name="duckdb-mixture-monitor", daemon=True)
    thread.start()
    return finished, thread


def _materialize_duckdb_input(
    input_path: Path,
    work_dir: Path,
    cancel_path: Path,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
) -> tuple[Path, int, int]:
    input_bytes = input_path.stat().st_size
    # DuckDB reads gzip JSONL directly. Expanding it first made scratch usage
    # proportional to the full uncompressed manifest for no semantic benefit.
    return input_path, input_bytes, 0


def _write_budgeted_body(
    connection: Any,
    output_path: Path,
    source_index: int,
    max_tokens: int,
    cancel_path: Path,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
) -> tuple[int, int]:
    cursor = connection.execute(
        """
        SELECT
            byte_length, character_length, content_hash, language, logical_path,
            record_id, record_index, token_estimate, type
        FROM candidates
        ORDER BY order_key, record_id
        """
    )
    selected_count = 0
    selected_tokens = 0
    scanned_rows = 0
    with _open_text_output(output_path) as output:
        while rows := cursor.fetchmany(MIXTURE_BATCH_SIZE):
            if cancel_path.exists():
                raise RuntimeError("mixture resolution interrupted")
            for row in rows:
                scanned_rows += 1
                tokens = max(int(row[7] or 0), 0)
                if selected_tokens + tokens > max_tokens:
                    continue
                output.write(
                    canonical_json(
                        {
                            "byte_length": row[0],
                            "character_length": row[1],
                            "content_hash": row[2],
                            "language": row[3],
                            "logical_path": row[4],
                            "record_id": row[5],
                            "record_index": row[6],
                            "source_index": source_index,
                            "source_position": selected_count,
                            "token_estimate": row[7],
                            "type": "record",
                        }
                    )
                    + "\n"
                )
                selected_count += 1
                selected_tokens += tokens
            with progress_lock:
                progress["processed_rows"] = scanned_rows
                progress["selected_rows"] = selected_count
    return selected_count, selected_tokens


def resolve_manifest_source(
    *,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    dataset_version_id: str,
    selection: dict[str, Any] | None,
    default_seed: int,
    source_index: int,
    cancel_path: Path,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
) -> dict[str, int]:
    """Resolve one source through a bounded-memory DuckDB external sort."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Large data-mixture resolution requires DuckDB") from exc

    rule = selection or {}
    include_paths = [str(value) for value in rule.get("include_paths") or [] if str(value)]
    exclude_paths = [str(value) for value in rule.get("exclude_paths") or [] if str(value)]
    minimum = max(int(rule.get("min_length") or 0), 0)
    maximum = max(int(rule.get("max_length") or 0), 0)
    ratio = min(max(float(rule.get("sample_ratio", 1.0)), 0.0), 1.0)
    max_tokens = max(int(rule.get("max_tokens") or 0), 0)
    seed = int(rule.get("seed", default_seed))
    input_bytes = input_path.stat().st_size

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    try:
        duckdb_input, total_scan_bytes, scan_base_bytes = _materialize_duckdb_input(
            input_path, work_dir, cancel_path, progress, progress_lock
        )
        # Historical downloads may have a .jsonl suffix despite containing gzip.
        # DuckDB auto_detect uses the name, unlike open_text_manifest's magic check.
        with duckdb_input.open("rb") as stream:
            compression = "gzip" if stream.read(2) == b"\x1f\x8b" else "uncompressed"
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    database_path = work_dir / "resolver.duckdb"
    connection = duckdb.connect(str(database_path))
    finished, monitor = _start_monitor(
        connection,
        cancel_path,
        progress,
        progress_lock,
        input_bytes,
        scan_base_bytes,
    )
    try:
        threads = max(1, min((os.cpu_count() or 1) - 2, 8))
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"SET memory_limit = '{duckdb_memory_limit(duckdb_memory_budget_bytes())}'"
        )
        connection.execute("SET enable_progress_bar = true")
        connection.execute("SET enable_progress_bar_print = false")
        connection.execute("SET progress_bar_time = 0")
        connection.execute(f"SET temp_directory = {_sql_literal((work_dir / 'spill').as_posix())}")

        predicates = [
            "type = 'record'",
            "record_id IS NOT NULL",
            f"coalesce(character_length, 0) >= {minimum}",
        ]
        if maximum:
            predicates.append(f"coalesce(character_length, 0) <= {maximum}")
        if include_paths:
            predicates.append(f"({_path_predicate(include_paths)})")
        if exclude_paths:
            predicates.append(f"NOT ({_path_predicate(exclude_paths)})")

        order_key = (
            f"sha256({_sql_literal(seed)} || chr(0) || "
            f"{_sql_literal(dataset_version_id)} || chr(0) || record_id || chr(0))"
        )
        if ratio <= 0:
            predicates.append("false")
        elif ratio < 1:
            ratio_limit = int(ratio * (1 << 256))
            predicates.append(f"{order_key} < {_sql_literal(f'{ratio_limit:064x}')}")

        source = f"""
            read_json(
                {_sql_literal(duckdb_input.as_posix())},
                format = 'newline_delimited',
                compression = {_sql_literal(compression)},
                columns = {{
                    type: 'VARCHAR', record_id: 'VARCHAR', logical_path: 'VARCHAR',
                    record_index: 'BIGINT', content_hash: 'VARCHAR',
                    character_length: 'BIGINT', byte_length: 'BIGINT',
                    token_estimate: 'BIGINT', language: 'VARCHAR'
                }}
            )
        """
        with progress_lock:
            progress.update(
                phase="external_sort",
                processed_bytes=scan_base_bytes,
                total_bytes=total_scan_bytes,
                processed_rows=0,
            )
        connection.execute(
            f"""
            CREATE TABLE candidates AS
            SELECT
                byte_length, character_length, content_hash, language, logical_path,
                record_id, record_index, token_estimate, 'record'::VARCHAR AS type,
                {order_key} AS order_key
            FROM {source}
            WHERE {" AND ".join(predicates)}
            """
        )
        if cancel_path.exists():
            raise RuntimeError("mixture resolution interrupted")
        summary = connection.execute(
            "SELECT count(*)::BIGINT, coalesce(sum(greatest(coalesce(token_estimate, 0), 0)), 0)::BIGINT FROM candidates"
        ).fetchone()
        assert summary is not None
        candidate_count, candidate_tokens = (int(summary[0]), int(summary[1]))
        with progress_lock:
            progress.update(
                phase="writing_manifest",
                processed_bytes=total_scan_bytes,
                processed_rows=0,
                total_rows=candidate_count,
            )

        if max_tokens:
            selected_count, selected_tokens = _write_budgeted_body(
                connection,
                output_path,
                source_index,
                max_tokens,
                cancel_path,
                progress,
                progress_lock,
            )
        else:
            copy_options = (
                "FORMAT JSON, ARRAY false, COMPRESSION gzip"
                if output_path.suffix.lower() == ".gz"
                else "FORMAT JSON, ARRAY false"
            )
            connection.execute(
                f"""
                COPY (
                    SELECT
                        byte_length, character_length, content_hash, language, logical_path,
                        record_id, record_index,
                        {source_index}::BIGINT AS source_index,
                        row_number() OVER (ORDER BY order_key, record_id) - 1 AS source_position,
                        token_estimate, type
                    FROM candidates
                    ORDER BY order_key, record_id
                ) TO {_sql_literal(output_path.as_posix())}
                ({copy_options})
                """
            )
            selected_count, selected_tokens = candidate_count, candidate_tokens
            with progress_lock:
                progress.update(processed_rows=selected_count, selected_rows=selected_count)
        if cancel_path.exists():
            raise RuntimeError("mixture resolution interrupted")
        return {
            "selected_document_count": selected_count,
            "selected_tokens": selected_tokens,
            "candidate_document_count": candidate_count,
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        finished.set()
        monitor.join(timeout=1)
        connection.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def progress_snapshot(progress: dict[str, Any], lock: threading.Lock) -> dict[str, Any]:
    with lock:
        return dict(progress)
