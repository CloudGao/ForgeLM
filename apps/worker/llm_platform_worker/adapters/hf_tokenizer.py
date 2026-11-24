from __future__ import annotations

import asyncio
import bisect
import hashlib
import itertools
import json
import math
import shutil
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
from llm_platform_core.mixtures import (
    MIXTURE_INDEX_FORMAT,
    MIXTURE_MANIFEST_FORMAT,
    RECORD_MANIFEST_FORMAT,
    open_text_manifest,
    sample_reference,
)
from llm_platform_core.tokenizers import validate_builtin_tokenizer_sample_tokens

from llm_platform_worker.file_cache import release_file_cache

from .command import Emit, GetDesiredState
from .data_import import PARQUET_SUFFIXES, TEXT_SUFFIXES, AnalysisInterrupted, DataImportAdapter


def hf_tokenizer_available() -> bool:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def _raise_if_controlled(get_desired_state: GetDesiredState | None) -> None:
    desired = get_desired_state() if get_desired_state else "running"
    if desired in {"paused", "cancelled"}:
        raise AnalysisInterrupted(desired)


class HfTokenizerAdapter:
    """Train a tokenizer with the official Hugging Face Tokenizers library."""

    EVALUATION_DOCUMENT_LIMIT = 500
    EVALUATION_CHARACTER_LIMIT = 16_384
    PROGRESS_INTERVAL_SECONDS = 1.0
    TRAINING_BATCH_DOCUMENTS = 256
    TRAINING_BATCH_CHARACTERS = 4_000_000
    FILE_CACHE_RELEASE_INTERVAL = 4096
    MANIFEST_CACHE_RELEASE_BYTES = 64 * 1024 * 1024
    MIXTURE_LOOKAHEAD_DOCUMENTS = 256
    MIXTURE_LOOKAHEAD_BYTES = 64 * 1024 * 1024
    MIXTURE_SAMPLE_PADDING = 256

    @staticmethod
    def _read_manifest_header(
        path: Path,
        expected_format: str,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        _raise_if_controlled(get_desired_state)
        with open_text_manifest(path) as stream:
            first = stream.readline()
            if not first:
                raise ValueError("Tokenizer data manifest is empty")
            header = json.loads(first)
            if header.get("type") != "header" or header.get("format") != expected_format:
                raise ValueError("Tokenizer data manifest has an unsupported format")
        return header

    @classmethod
    def _iter_manifest_records(
        cls,
        path: Path,
        expected_format: str,
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[dict[str, Any]]:
        bytes_since_cache_release = 0
        with open_text_manifest(path) as stream:
            first = stream.readline()
            if not first:
                raise ValueError("Tokenizer data manifest is empty")
            header = json.loads(first)
            if header.get("type") != "header" or header.get("format") != expected_format:
                raise ValueError("Tokenizer data manifest has an unsupported format")
            for line_number, line in enumerate(stream, start=2):
                bytes_since_cache_release += len(line)
                if line_number % 4096 == 0:
                    _raise_if_controlled(get_desired_state)
                if bytes_since_cache_release >= cls.MANIFEST_CACHE_RELEASE_BYTES:
                    release_file_cache(path)
                    bytes_since_cache_release = 0
                if line.strip():
                    record = json.loads(line)
                    if record.get("type") != "record":
                        raise ValueError("Tokenizer data manifest contains an invalid record")
                    yield record
        release_file_cache(path)
        _raise_if_controlled(get_desired_state)

    @staticmethod
    def _source_files(root: Path) -> dict[str, Path]:
        if root.is_file():
            return {root.name: root}
        return {
            path.relative_to(root).as_posix(): path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES | PARQUET_SUFFIXES
            and not ({".hfd", ".cache", ".git"} & set(path.relative_to(root).parts))
        }

    @classmethod
    def _iter_selected_texts(
        cls,
        root: Path,
        records: Iterable[dict[str, Any]],
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        files = cls._source_files(root)
        single_file = next(iter(files.values())) if len(files) == 1 else None
        completed_paths: set[str] = set()
        for logical_path, grouped_records in itertools.groupby(
            records, key=lambda record: str(record.get("logical_path") or "")
        ):
            _raise_if_controlled(get_desired_state)
            if logical_path in completed_paths:
                raise ValueError(
                    f"Tokenizer manifest records for {logical_path} are not contiguous"
                )
            completed_paths.add(logical_path)
            match = files.get(logical_path) or single_file
            if match is None:
                raise FileNotFoundError(f"Tokenizer source file is missing: {logical_path}")
            requested_records = iter(grouped_records)
            if match.suffix.lower() in PARQUET_SUFFIXES:
                try:
                    with closing(cls._iter_parquet_records_direct(
                        match, requested_records, get_desired_state
                    )) as direct:
                        for text, record in direct:
                            if text is None:
                                requested_records = itertools.chain([record], requested_records)
                                break
                            yield text, record
                        else:
                            continue
                finally:
                    release_file_cache(match)
            requested = iter(requested_records)
            current = next(requested, None)
            try:
                for record_index, text in enumerate(
                    DataImportAdapter._iter_documents(
                        match,
                        get_desired_state=get_desired_state,
                    )
                ):
                    if record_index % 4096 == 0:
                        _raise_if_controlled(get_desired_state)
                    while (
                        current is not None
                        and int(current.get("record_index") or 0) == record_index
                    ):
                        yield text, current
                        current = next(requested, None)
                    if current is None:
                        break
                    if int(current.get("record_index") or 0) < record_index:
                        raise ValueError(
                            f"Tokenizer manifest records for {logical_path} are not ordered"
                        )
            finally:
                release_file_cache(match)
            if current is not None:
                raise ValueError(
                    f"Tokenizer source record is missing from {logical_path}: "
                    f"{int(current.get('record_index') or 0)}"
                )

    @staticmethod
    def _iter_parquet_records_direct(
        path: Path,
        records: Iterable[dict[str, Any]],
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[tuple[str | None, dict[str, Any]]]:
        """Read only requested rows in bounded batches; None requests legacy fallback."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet data access requires pyarrow on this Worker") from exc
        fields = DataImportAdapter._parquet_text_fields(path)
        with pq.ParquetFile(path) as parquet:
            offsets = [0]
            for row_group in range(parquet.num_row_groups):
                offsets.append(offsets[-1] + parquet.metadata.row_group(row_group).num_rows)
            active_group = -1
            previous_index = -1
            batch = None
            batch_start = 0
            batches = iter(())
            for record in records:
                _raise_if_controlled(get_desired_state)
                physical_index = int(record.get("record_index") or 0)
                if physical_index < previous_index:
                    raise ValueError(f"Tokenizer manifest records for {path.name} are not ordered")
                previous_index = physical_index
                row_group = bisect.bisect_right(offsets, physical_index) - 1
                if row_group < 0 or row_group >= parquet.num_row_groups:
                    yield None, record
                    return
                if row_group != active_group:
                    active_group = row_group
                    batch_start = offsets[row_group]
                    batch = None
                    batches = parquet.iter_batches(
                        batch_size=512, row_groups=[row_group], columns=fields, use_threads=True
                    )
                while batch is None or physical_index >= batch_start + batch.num_rows:
                    _raise_if_controlled(get_desired_state)
                    if batch is not None:
                        batch_start += batch.num_rows
                    batch = next(batches)
                row_index = physical_index - batch_start
                values = [batch.column(index)[row_index].as_py() for index in range(len(fields))]
                text = "\n".join(
                    str(value) for value in values if value not in {None, ""}
                )
                expected_hash = str(record.get("content_hash") or "")
                if not text or (
                    expected_hash
                    and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_hash
                ):
                    # Older manifests number only non-empty rows. A mismatch means
                    # physical row ordinals cannot be used safely for this file.
                    yield None, record
                    return
                yield text, record

    @classmethod
    def _append_evaluation(cls, evaluation: list[str], text: str) -> None:
        if len(evaluation) < cls.EVALUATION_DOCUMENT_LIMIT:
            evaluation.append(text[: cls.EVALUATION_CHARACTER_LIMIT])

    @classmethod
    def _dataset_corpus(
        cls,
        manifest_path: Path,
        source_root: Path,
        sample_tokens: int,
        report_progress: Callable[[dict[str, Any]], None] | None,
        get_desired_state: GetDesiredState | None,
        record_callback=None,
    ) -> tuple[Iterator[str], list[str], dict[str, int]]:
        cls._read_manifest_header(manifest_path, RECORD_MANIFEST_FORMAT, get_desired_state)
        evaluation: list[str] = []
        stats = {"documents": 0, "token_estimate": 0}

        def dataset_stream() -> Iterator[str]:
            last_report = 0.0
            records = cls._iter_manifest_records(
                manifest_path, RECORD_MANIFEST_FORMAT, get_desired_state
            )
            for text, record in cls._iter_selected_texts(source_root, records, get_desired_state):
                _raise_if_controlled(get_desired_state)
                stats["documents"] += 1
                stats["token_estimate"] += max(int(record.get("token_estimate") or 0), 1)
                cls._append_evaluation(evaluation, text)
                now = time.monotonic()
                if report_progress and (
                    now - last_report >= cls.PROGRESS_INTERVAL_SECONDS
                    or (sample_tokens and stats["token_estimate"] >= sample_tokens)
                ):
                    report_progress(
                        {
                            "phase": "streaming_training",
                            "sampled_documents": stats["documents"],
                            "sampled_tokens": stats["token_estimate"],
                            "target_tokens": sample_tokens or None,
                            "progress_basis": "manifest_token_estimate",
                        }
                    )
                    last_report = now
                if record_callback is not None:
                    record_callback(record)
                yield text
                if sample_tokens and stats["token_estimate"] >= sample_tokens:
                    return

        return dataset_stream(), evaluation, stats

    @classmethod
    def _build_mixture_index(
        cls,
        manifest_path: Path,
        sources: list[dict[str, Any]],
        index_path: Path,
        sample_tokens: int,
        sample_offset: int,
        manifest_format: str,
        report_progress: Callable[[dict[str, Any]], None] | None,
        get_desired_state: GetDesiredState | None,
    ) -> tuple[dict[str, Any], list[int], int, int]:
        header = cls._read_manifest_header(manifest_path, manifest_format, get_desired_state)
        raw_sources = header.get("sources") or []
        if len(raw_sources) != len(sources) or not raw_sources:
            raise ValueError("Mixture manifest sources do not match materialized sources")
        counts = [max(int(source.get("selected_document_count") or 0), 0) for source in raw_sources]
        token_totals = [max(int(source.get("selected_token_estimate") or 0), 0) for source in raw_sources]
        if not all(counts) or not all(token_totals):
            if manifest_format == MIXTURE_INDEX_FORMAT:
                raise ValueError(
                    "Mixture reference index requires exact per-source document and token counts"
                )
            counts = [0 for _ in raw_sources]
            token_totals = [0 for _ in raw_sources]
            for record in cls._iter_manifest_records(
                manifest_path, MIXTURE_MANIFEST_FORMAT, get_desired_state
            ):
                source_index = int(record.get("source_index", -1))
                if source_index < 0 or source_index >= len(counts):
                    raise ValueError("Mixture manifest record has an invalid source index")
                counts[source_index] += 1
                token_totals[source_index] += max(int(record.get("token_estimate") or 0), 1)
        if not all(counts):
            raise ValueError("Mixture manifest source has no selected records")

        class SizedRecords:
            def __init__(self, size: int):
                self.size = size

            def __len__(self) -> int:
                return self.size

        record_groups = [SizedRecords(size) for size in counts]
        total_records = sum(counts)
        total_token_estimate = sum(token_totals)
        target_tokens = sample_tokens or total_token_estimate
        weight_total = sum(max(int(source.get("weight_units") or 0), 0) for source in raw_sources)
        weighted_average = sum(
            max(int(source.get("weight_units") or 0), 0)
            * (token_totals[index] / counts[index])
            for index, source in enumerate(raw_sources)
        ) / max(weight_total, 1)
        planned_draws = max(
            1,
            math.ceil(target_tokens / max(weighted_average, 1) * 1.10)
            + cls.MIXTURE_SAMPLE_PADDING,
        )
        max_draws = max(total_records, 1) * 1000
        planned_draws = min(planned_draws, max_draws)
        # A sample row is only integer/location metadata. Budget deliberately does not
        # scale with source text bytes: raw text must remain unique in the dataset snapshot.
        required_free = planned_draws * 256 + 256 * 1024 * 1024
        free = shutil.disk_usage(index_path.parent).free
        if free < required_free:
            raise OSError(
                "Mixture sample index needs at least "
                f"{required_free / (1024**3):.1f} GiB free, but only "
                f"{free / (1024**3):.1f} GiB is available"
            )

        request_path = index_path.with_suffix(".requests.sqlite3")
        index_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        requests = sqlite3.connect(request_path)
        output = sqlite3.connect(index_path)
        for connection in (requests, output):
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
        requests.execute(
            "CREATE TABLE requests (sample_index INTEGER PRIMARY KEY, source_index INTEGER NOT NULL, ordinal INTEGER NOT NULL)"
        )
        requests.execute(
            "CREATE INDEX requests_source_ordinal ON requests(source_index, ordinal, sample_index)"
        )
        output.execute(
            "CREATE TABLE samples (sample_index INTEGER PRIMARY KEY, source_index INTEGER NOT NULL, "
            "ordinal INTEGER NOT NULL, logical_path TEXT NOT NULL, record_index INTEGER NOT NULL, "
            "content_hash TEXT NOT NULL, token_estimate INTEGER NOT NULL, byte_length INTEGER NOT NULL)"
        )

        populated = 0
        try:
            while True:
                request_batch = []
                for sample_index in range(
                    sample_offset + populated, sample_offset + planned_draws
                ):
                    source_index, ordinal = sample_reference(
                        header, record_groups, sample_index  # type: ignore[arg-type]
                    )
                    request_batch.append((sample_index, source_index, ordinal))
                    if len(request_batch) >= 8192:
                        requests.executemany("INSERT INTO requests VALUES (?, ?, ?)", request_batch)
                        request_batch.clear()
                if request_batch:
                    requests.executemany("INSERT INTO requests VALUES (?, ?, ?)", request_batch)
                requests.commit()

                cursors = [
                    requests.execute(
                        "SELECT ordinal, sample_index FROM requests "
                        "WHERE source_index = ? AND sample_index >= ? ORDER BY ordinal, sample_index",
                        (source_index, sample_offset + populated),
                    )
                    for source_index in range(len(counts))
                ]
                pending = [cursor.fetchone() for cursor in cursors]
                ordinals = [0 for _ in counts]
                insert_batch: list[tuple[int, int, int, str, int, str, int, int]] = []
                if manifest_format == MIXTURE_INDEX_FORMAT:
                    record_stream = (
                        (source_index, record)
                        for source_index, source in enumerate(sources)
                        for record in cls._iter_manifest_records(
                            Path(str(source.get("runtime_record_manifest_path") or "")),
                            RECORD_MANIFEST_FORMAT,
                            get_desired_state,
                        )
                    )
                else:
                    record_stream = (
                        (int(record.get("source_index", -1)), record)
                        for record in cls._iter_manifest_records(
                            manifest_path, MIXTURE_MANIFEST_FORMAT, get_desired_state
                        )
                    )
                for position, (source_index, record) in enumerate(record_stream, start=1):
                    if source_index < 0 or source_index >= len(counts):
                        raise ValueError("Mixture manifest record has an invalid source index")
                    ordinal = ordinals[source_index]
                    ordinals[source_index] += 1
                    while pending[source_index] is not None and int(pending[source_index][0]) == ordinal:
                        _, sample_index = pending[source_index]
                        insert_batch.append(
                            (
                                int(sample_index),
                                source_index,
                                ordinal,
                                str(record.get("logical_path") or ""),
                                int(record.get("record_index") or 0),
                                str(record.get("content_hash") or ""),
                                max(int(record.get("token_estimate") or 0), 1),
                                max(int(record.get("byte_length") or 0), 0),
                            )
                        )
                        pending[source_index] = cursors[source_index].fetchone()
                    if len(insert_batch) >= 8192:
                        output.executemany(
                            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            insert_batch,
                        )
                        output.commit()
                        insert_batch.clear()
                    if report_progress and position % 1_000_000 == 0:
                        report_progress(
                            {
                                "phase": "preparing_corpus",
                                "scanned_manifest_records": position,
                                "planned_samples": planned_draws,
                                "progress_basis": "sample_reference_index",
                            }
                        )
                if insert_batch:
                    output.executemany(
                        "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        insert_batch,
                    )
                    output.commit()
                if any(item is not None for item in pending):
                    raise ValueError("Mixture manifest is missing a sampled record reference")

                running_tokens = 0
                cutoff: int | None = None
                for sample_index, token_estimate in output.execute(
                    "SELECT sample_index, token_estimate FROM samples ORDER BY sample_index"
                ):
                    running_tokens += max(int(token_estimate), 1)
                    if running_tokens >= target_tokens:
                        cutoff = int(sample_index)
                        break
                if cutoff is not None:
                    output.execute("DELETE FROM samples WHERE sample_index > ?", (cutoff,))
                    output.commit()
                    break
                if planned_draws >= max_draws:
                    raise ValueError("Tokenizer sample target could not be reached from the mixture")
                populated = planned_draws
                planned_draws = min(max_draws, planned_draws * 2)
            output.execute(
                "CREATE INDEX samples_source_path ON samples(source_index, logical_path, record_index)"
            )
            output.commit()
            sampled = int(output.execute("SELECT count(*) FROM samples").fetchone()[0])
            sampled_tokens = int(
                output.execute("SELECT coalesce(sum(token_estimate), 0) FROM samples").fetchone()[0]
            )
        except BaseException:
            output.close()
            requests.close()
            index_path.unlink(missing_ok=True)
            request_path.unlink(missing_ok=True)
            raise
        output.close()
        requests.close()
        request_path.unlink(missing_ok=True)
        release_file_cache(manifest_path)
        release_file_cache(index_path)
        return header, counts, sampled, sampled_tokens

    @classmethod
    def _mixture_corpus(
        cls,
        manifest_path: Path,
        sources: list[dict[str, Any]],
        sample_tokens: int,
        work_dir: Path,
        report_progress: Callable[[dict[str, Any]], None] | None,
        get_desired_state: GetDesiredState | None,
        sample_offset: int = 0,
        manifest_format: str = MIXTURE_MANIFEST_FORMAT,
        spool_path: Path | None = None,
    ) -> tuple[Iterator[str], list[str], dict[str, int]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        index_path = work_dir / "mixture-corpus.sqlite3"
        _header, _counts, total_records, total_token_estimate = cls._build_mixture_index(
            manifest_path,
            sources,
            index_path,
            sample_tokens,
            sample_offset,
            manifest_format,
            report_progress,
            get_desired_state,
        )
        evaluation: list[str] = []
        stats = {"documents": 0, "token_estimate": 0}

        target_tokens = sample_tokens or total_token_estimate

        if spool_path is not None:
            spool_path.parent.mkdir(parents=True, exist_ok=True)
            spool_path.unlink(missing_ok=True)
            sample_index = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
            sample_index.execute("PRAGMA query_only=ON")
            spool = sqlite3.connect(spool_path)
            spool.execute("PRAGMA journal_mode=OFF")
            spool.execute("PRAGMA synchronous=OFF")
            spool.execute("PRAGMA temp_store=FILE")
            spool.execute("PRAGMA page_size=65536")
            spool.execute(
                "CREATE TABLE documents (sample_index INTEGER PRIMARY KEY, "
                "token_estimate INTEGER NOT NULL, text TEXT NOT NULL) WITHOUT ROWID"
            )
            inserted = 0
            insert_batch: list[tuple[int, int, str]] = []
            last_report = 0.0
            try:
                for source_index, source in enumerate(sources):
                    rows = sample_index.execute(
                        "SELECT sample_index, ordinal, logical_path, record_index, "
                        "content_hash, token_estimate, byte_length FROM samples "
                        "WHERE source_index = ? "
                        "ORDER BY logical_path, record_index, sample_index",
                        (source_index,),
                    )

                    def ordered_records(_rows=rows) -> Iterator[dict[str, Any]]:
                        for row in _rows:
                            yield {
                                "sample_index": int(row[0]),
                                "ordinal": int(row[1]),
                                "logical_path": str(row[2]),
                                "record_index": int(row[3]),
                                "content_hash": str(row[4]),
                                "token_estimate": int(row[5]),
                                "byte_length": int(row[6]),
                            }

                    source_root = Path(str(source.get("runtime_path") or ""))
                    for text, record in cls._iter_selected_texts(
                        source_root, ordered_records(), get_desired_state
                    ):
                        insert_batch.append(
                            (
                                int(record["sample_index"]),
                                max(int(record.get("token_estimate") or 0), 1),
                                text,
                            )
                        )
                        inserted += 1
                        if len(insert_batch) >= 4096:
                            spool.executemany("INSERT INTO documents VALUES (?, ?, ?)", insert_batch)
                            spool.commit()
                            insert_batch.clear()
                        now = time.monotonic()
                        if report_progress and (
                            now - last_report >= cls.PROGRESS_INTERVAL_SECONDS
                            or inserted >= total_records
                        ):
                            report_progress(
                                {
                                    "phase": "spooling_corpus",
                                    "spooled_documents": inserted,
                                    "total_documents": total_records,
                                    "progress_basis": "physical_source_order",
                                }
                            )
                            last_report = now
                if insert_batch:
                    spool.executemany("INSERT INTO documents VALUES (?, ?, ?)", insert_batch)
                    spool.commit()
                if inserted != total_records:
                    raise ValueError(
                        f"Mixture spool contains {inserted:,} documents; "
                        f"{total_records:,} are required"
                    )
            except BaseException:
                spool.close()
                sample_index.close()
                spool_path.unlink(missing_ok=True)
                raise
            spool.close()
            sample_index.close()
            release_file_cache(spool_path)

            def spooled_stream() -> Iterator[str]:
                connection = sqlite3.connect(
                    f"file:{spool_path.as_posix()}?mode=ro", uri=True
                )
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA mmap_size=0")
                connection.execute("PRAGMA cache_size=-65536")
                last_stream_report = 0.0
                try:
                    for text, token_estimate in connection.execute(
                        "SELECT text, token_estimate FROM documents ORDER BY sample_index"
                    ):
                        _raise_if_controlled(get_desired_state)
                        stats["documents"] += 1
                        stats["token_estimate"] += max(int(token_estimate), 1)
                        cls._append_evaluation(evaluation, str(text))
                        now = time.monotonic()
                        if report_progress and (
                            now - last_stream_report >= cls.PROGRESS_INTERVAL_SECONDS
                            or stats["token_estimate"] >= target_tokens
                        ):
                            report_progress(
                                {
                                    "phase": "streaming_training",
                                    "sampled_documents": stats["documents"],
                                    "sampled_tokens": stats["token_estimate"],
                                    "target_tokens": target_tokens,
                                    "progress_basis": "manifest_token_estimate",
                                }
                            )
                            last_stream_report = now
                        yield str(text)
                finally:
                    connection.close()
                    release_file_cache(spool_path)
                    spool_path.unlink(missing_ok=True)

            return spooled_stream(), evaluation, stats

        def mixture_stream() -> Iterator[str]:
            connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA cache_size=-65536")
            last_report = 0.0
            try:
                cursor = connection.execute(
                    "SELECT sample_index, source_index, ordinal, logical_path, record_index, "
                    "content_hash, token_estimate, byte_length FROM samples ORDER BY sample_index"
                )
                while True:
                    rows = []
                    planned_bytes = 0
                    while len(rows) < cls.MIXTURE_LOOKAHEAD_DOCUMENTS:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)
                        planned_bytes += max(int(row[7]), 1)
                        if planned_bytes >= cls.MIXTURE_LOOKAHEAD_BYTES:
                            break
                    if not rows:
                        return
                    _raise_if_controlled(get_desired_state)
                    texts: dict[int, str] = {}
                    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for row in rows:
                        grouped[int(row[1])].append(
                            {
                                "sample_index": int(row[0]),
                                "ordinal": int(row[2]),
                                "logical_path": str(row[3]),
                                "record_index": int(row[4]),
                                "content_hash": str(row[5]),
                                "token_estimate": int(row[6]),
                            }
                        )
                    for source_index, records in grouped.items():
                        ordered = sorted(
                            records,
                            key=lambda record: (
                                str(record["logical_path"]),
                                int(record["record_index"]),
                                int(record["sample_index"]),
                            ),
                        )
                        for text, record in cls._iter_selected_texts(
                            Path(str(sources[source_index]["runtime_path"])),
                            ordered,
                            get_desired_state,
                        ):
                            texts[int(record["sample_index"])] = text
                    if len(texts) != len(rows):
                        raise ValueError("Mixture sample index is incomplete")
                    for row in rows:
                        sample_index = int(row[0])
                        text = texts.pop(sample_index)
                        stats["documents"] += 1
                        stats["token_estimate"] += max(int(row[6]), 1)
                        cls._append_evaluation(evaluation, text)
                        now = time.monotonic()
                        if report_progress and (
                            now - last_report >= cls.PROGRESS_INTERVAL_SECONDS
                            or stats["token_estimate"] >= target_tokens
                        ):
                            report_progress(
                                {
                                    "phase": "streaming_training",
                                    "sampled_documents": stats["documents"],
                                    "sampled_tokens": stats["token_estimate"],
                                    "target_tokens": target_tokens,
                                    "progress_basis": "manifest_token_estimate",
                                }
                            )
                            last_report = now
                        yield text
                if stats["documents"] != total_records:
                    raise ValueError("Mixture sample index ended unexpectedly")
            finally:
                connection.close()
                release_file_cache(index_path)

        return mixture_stream(), evaluation, stats

    @classmethod
    def validation_corpus_prefix(
        cls,
        data_input: dict[str, Any],
        sample_tokens: int,
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[str]:
        """Read a deterministic, bounded validation prefix without a global sample index.

        Validation examples are resident and small. Resolving random ordinals from a
        compressed record manifest would require a full scan for every training run,
        so this path assigns each source a stable weight-proportional token quota and
        stops each manifest as soon as its prefix covers that quota.
        """
        if data_input.get("format") != MIXTURE_INDEX_FORMAT:
            raise ValueError("Bounded validation sampling requires a mixture reference index")
        manifest_path = Path(str(data_input.get("runtime_path") or ""))
        sources = sorted(
            data_input.get("runtime_sources") or [],
            key=lambda item: int(item.get("source_index", -1)),
        )
        if not manifest_path.is_file() or not sources:
            raise ValueError("Validation input was not materialized on this Worker")
        header = cls._read_manifest_header(
            manifest_path, MIXTURE_INDEX_FORMAT, get_desired_state
        )
        raw_sources = sorted(
            header.get("sources") or [],
            key=lambda item: int(item.get("source_index", -1)),
        )
        if len(raw_sources) != len(sources) or not raw_sources:
            raise ValueError("Validation mixture sources do not match materialized sources")
        weights = [max(int(source.get("weight_units") or 0), 0) for source in raw_sources]
        if not any(weights):
            weights = [1 for _source in raw_sources]
        total_weight = sum(weights)

        def validation_stream() -> Iterator[str]:
            for source_index, (source, weight) in enumerate(
                zip(sources, weights, strict=True)
            ):
                if int(source.get("source_index", -1)) != source_index:
                    raise ValueError("Validation mixture source indexes are not contiguous")
                record_manifest = Path(
                    str(source.get("runtime_record_manifest_path") or "")
                )
                source_root = Path(str(source.get("runtime_path") or ""))
                if not record_manifest.is_file() or not source_root.exists():
                    raise ValueError("Validation source artifacts were not materialized")
                quota = max(1, math.ceil(sample_tokens * weight / total_weight))

                def prefix_records(
                    _record_manifest: Path = record_manifest,
                    _quota: int = quota,
                ) -> Iterator[dict[str, Any]]:
                    estimated_tokens = 0
                    for record in cls._iter_manifest_records(
                        _record_manifest,
                        RECORD_MANIFEST_FORMAT,
                        get_desired_state,
                    ):
                        yield record
                        estimated_tokens += max(int(record.get("token_estimate") or 0), 1)
                        if estimated_tokens >= _quota:
                            break

                for text, _record in cls._iter_selected_texts(
                    source_root, prefix_records(), get_desired_state
                ):
                    yield text

        return validation_stream()

    @classmethod
    def validation_corpus_random(
        cls,
        data_input: dict[str, Any],
        sample_tokens: int,
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[str]:
        """Select a small weighted validation sample without a persistent index.

        Each source Record Manifest is scanned once to capture deterministic random
        ordinals. Only the bounded selected records and texts are held in memory, then
        their stable random keys define one cross-source output order.
        """
        if data_input.get("format") != MIXTURE_INDEX_FORMAT:
            raise ValueError("Random validation sampling requires a mixture reference index")
        if sample_tokens <= 0:
            raise ValueError("Random validation sampling requires a positive token target")
        manifest_path = Path(str(data_input.get("runtime_path") or ""))
        sources = sorted(
            data_input.get("runtime_sources") or [],
            key=lambda item: int(item.get("source_index", -1)),
        )
        if not manifest_path.is_file() or not sources:
            raise ValueError("Validation input was not materialized on this Worker")
        header = cls._read_manifest_header(
            manifest_path, MIXTURE_INDEX_FORMAT, get_desired_state
        )
        raw_sources = sorted(
            header.get("sources") or [],
            key=lambda item: int(item.get("source_index", -1)),
        )
        if len(raw_sources) != len(sources) or not raw_sources:
            raise ValueError("Validation mixture sources do not match materialized sources")
        weights = [max(int(source.get("weight_units") or 0), 0) for source in raw_sources]
        if not any(weights):
            weights = [1 for _source in raw_sources]
        total_weight = sum(weights)
        sampling = data_input.get("validation_sampling") or {}
        seed = int(sampling.get("seed", data_input.get("seed", header.get("seed", 0))))
        manifest_identity = str(
            data_input.get("manifest_hash")
            or header.get("stream_key")
            or header.get("identity_hash")
            or ""
        )

        def derived_seed(namespace: str) -> int:
            payload = (
                f"weighted_random_materialized_v1:{manifest_identity}:{seed}:{namespace}"
            )
            return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "little")

        def random_ordinals(count: int, size: int, random_seed: int) -> list[int]:
            if not 0 <= size <= count or count > 2**64:
                raise ValueError("Validation source document count is out of range")
            if size == count:
                return list(range(count))
            select_complement = size > count // 2
            target = count - size if select_complement else size
            selected: set[int] = set()
            bit_generator = np.random.PCG64(random_seed)
            uint64_range = 2**64
            unbiased_limit = uint64_range - (uint64_range % count)
            while len(selected) < target:
                value = int(bit_generator.random_raw())
                if value < unbiased_limit:
                    selected.add(value % count)
            if select_complement:
                return [ordinal for ordinal in range(count) if ordinal not in selected]
            return sorted(selected)

        selected_texts: list[tuple[bytes, int, int, str]] = []
        for source_index, (source, raw_source, weight) in enumerate(
            zip(sources, raw_sources, weights, strict=True)
        ):
            if (
                int(source.get("source_index", -1)) != source_index
                or int(raw_source.get("source_index", -1)) != source_index
            ):
                raise ValueError("Validation mixture source indexes are not contiguous")
            if weight <= 0:
                continue
            record_count = max(int(raw_source.get("selected_document_count") or 0), 0)
            token_estimate = max(int(raw_source.get("selected_token_estimate") or 0), 0)
            if record_count <= 0 or token_estimate <= 0:
                raise ValueError("Random validation sampling requires exact source statistics")
            quota = max(1, math.ceil(sample_tokens * weight / total_weight))
            average_tokens = token_estimate / record_count
            selected_count = min(
                record_count,
                max(1, math.ceil(quota / max(average_tokens, 1) * 1.25)),
            )
            selected_ordinals = random_ordinals(
                record_count,
                selected_count,
                derived_seed(f"source:{source_index}"),
            )
            ordinal_iterator = iter(selected_ordinals)
            wanted = next(ordinal_iterator, None)
            selected_records: list[dict[str, Any]] = []
            record_manifest = Path(
                str(source.get("runtime_record_manifest_path") or "")
            )
            source_root = Path(str(source.get("runtime_path") or ""))
            if not record_manifest.is_file() or not source_root.exists():
                raise ValueError("Validation source artifacts were not materialized")
            for ordinal, record in enumerate(
                cls._iter_manifest_records(
                    record_manifest,
                    RECORD_MANIFEST_FORMAT,
                    get_desired_state,
                )
            ):
                if wanted is None:
                    break
                if ordinal == wanted:
                    selected_records.append(
                        {**record, "_validation_source_ordinal": ordinal}
                    )
                    wanted = next(ordinal_iterator, None)
            if wanted is not None or len(selected_records) != selected_count:
                raise ValueError("Validation source statistics do not match its Record Manifest")
            for text, record in cls._iter_selected_texts(
                source_root, selected_records, get_desired_state
            ):
                ordinal = int(record["_validation_source_ordinal"])
                order_key = hashlib.sha256(
                    (
                        "weighted_random_materialized_v1:order:"
                        f"{manifest_identity}:{seed}:{source_index}:{ordinal}"
                    ).encode()
                ).digest()
                selected_texts.append((order_key, source_index, ordinal, text))

        if not selected_texts:
            raise ValueError("Random validation sampling selected no documents")
        selected_texts.sort(key=lambda item: (item[0], item[1], item[2]))
        return (text for _key, _source, _ordinal, text in selected_texts)

    @classmethod
    def _corpus(
        cls,
        data_input: dict[str, Any],
        sample_tokens: int,
        work_dir: Path,
        report_progress: Callable[[dict[str, Any]], None] | None = None,
        get_desired_state: GetDesiredState | None = None,
        sample_offset: int = 0,
        spool_path: Path | None = None,
        record_callback=None,
    ) -> tuple[Iterator[str], list[str], dict[str, int]]:
        manifest_path = Path(str(data_input.get("runtime_path") or ""))
        sources = data_input.get("runtime_sources") or []
        if not manifest_path.is_file() or not sources:
            raise ValueError("Tokenizer input was not materialized on this Worker")

        if data_input.get("format") in {MIXTURE_MANIFEST_FORMAT, MIXTURE_INDEX_FORMAT}:
            manifest_format = str(data_input["format"])
            ordered_sources = sorted(sources, key=lambda item: int(item.get("source_index", -1)))
            if manifest_format == MIXTURE_INDEX_FORMAT and any(
                not source.get("runtime_record_manifest_path") for source in ordered_sources
            ):
                raise ValueError("Mixture index source record manifests were not materialized")
            return cls._mixture_corpus(
                manifest_path,
                ordered_sources,
                sample_tokens,
                work_dir,
                report_progress,
                get_desired_state,
                sample_offset,
                manifest_format,
                spool_path,
            )
        if data_input.get("format") != "llm-dataset-input-v1":
            raise ValueError("Tokenizer data input has an unsupported format")
        return cls._dataset_corpus(
            manifest_path,
            Path(str(sources[0]["runtime_path"])),
            sample_tokens,
            report_progress,
            get_desired_state,
            record_callback,
        )

    @staticmethod
    def _build_tokenizer(algorithm: str, config: dict[str, Any], vocab_size: int):
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        special_tokens = [
            str(value) for value in config.get("special_tokens") or ["<unk>", "<s>", "</s>"]
        ]
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Tokenizer special tokens must be unique")
        unk_token = special_tokens[0] if special_tokens else "<unk>"
        if unk_token not in special_tokens:
            special_tokens.insert(0, unk_token)
        byte_fallback = bool(config.get("byte_fallback", False))
        common = {
            "vocab_size": vocab_size,
            "special_tokens": special_tokens,
            "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
            "show_progress": False,
        }
        normalized_algorithm = algorithm.lower()
        if normalized_algorithm == "bpe":
            model = models.BPE(unk_token=unk_token, byte_fallback=byte_fallback)
            trainer = trainers.BpeTrainer(**common)
        elif normalized_algorithm == "unigram":
            model = models.Unigram(byte_fallback=byte_fallback)
            trainer = trainers.UnigramTrainer(**common, unk_token=unk_token)
        elif normalized_algorithm == "wordpiece":
            if byte_fallback:
                raise ValueError("Hugging Face WordPiece does not support byte fallback")
            model = models.WordPiece(unk_token=unk_token, continuing_subword_prefix="")
            trainer = trainers.WordPieceTrainer(**common, continuing_subword_prefix="")
        else:
            raise ValueError(f"Unsupported Hugging Face tokenizer algorithm: {algorithm}")

        tokenizer = Tokenizer(model)
        normalization = str(config.get("normalization") or "NFKC")
        if normalization == "NFKC":
            tokenizer.normalizer = normalizers.NFKC()
        elif normalization == "NFC":
            tokenizer.normalizer = normalizers.NFC()
        elif normalization != "None":
            raise ValueError(f"Unsupported tokenizer normalization: {normalization}")
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer, trainer, special_tokens

    @classmethod
    def _batched_corpus(cls, corpus: Iterator[str]) -> Iterator[list[str]]:
        """Cross the Python/Rust boundary in bounded batches without materializing the corpus."""
        batch: list[str] = []
        characters = 0
        for text in corpus:
            if batch and (
                len(batch) >= cls.TRAINING_BATCH_DOCUMENTS
                or characters + len(text) > cls.TRAINING_BATCH_CHARACTERS
            ):
                yield batch
                batch = []
                characters = 0
            batch.append(text)
            characters += len(text)
        if batch:
            yield batch

    @staticmethod
    def _metrics(tokenizer, corpus: list[str], special_tokens: list[str]) -> dict[str, Any]:
        totals = defaultdict(lambda: [0, 0])
        exact = 0
        unknown = 0
        byte_fallback = 0
        token_count = 0
        newline_exact = 0
        newline_documents = 0
        indentation_exact = 0
        indentation_documents = 0
        unk_token = special_tokens[0]
        for text in corpus[:500]:
            language = DataImportAdapter._language(text)
            encoding = tokenizer.encode(text)
            count = len(encoding.ids)
            totals[language][0] += len(text)
            totals[language][1] += count
            math_characters = sum(
                char.isdigit() or char in "+-*/=<>^%∑∫√≈≠≤≥()[]{}" for char in text
            )
            if math_characters >= 3:
                totals["math"][0] += len(text)
                totals["math"][1] += count
            token_count += count
            unknown += sum(token == unk_token for token in encoding.tokens)
            byte_fallback += sum(
                len(token) == 6
                and token.startswith("<0x")
                and token.endswith(">")
                and all(char in "0123456789abcdefABCDEF" for char in token[3:5])
                for token in encoding.tokens
            )
            expected = tokenizer.normalizer.normalize_str(text) if tokenizer.normalizer else text
            decoded_exact = tokenizer.decode(encoding.ids) == expected
            exact += int(decoded_exact)
            if "\n" in text or "\r" in text:
                newline_documents += 1
                newline_exact += int(decoded_exact)
            if any(line.startswith((" ", "\t")) for line in text.splitlines() if line):
                indentation_documents += 1
                indentation_exact += int(decoded_exact)

        def ratio(language: str) -> float | None:
            characters, tokens = totals[language]
            return round(characters / tokens, 4) if tokens else None

        evaluated = min(len(corpus), 500)
        vocab = tokenizer.get_vocab()
        garbage_tokens = sum(
            not token
            or "\ufffd" in token
            or "\x00" in token
            or len(token) > 128
            or any(ord(char) < 32 and char not in "\t\r\n" for char in token)
            for token in vocab
        )
        intact_special_tokens = sum(
            tokenizer.token_to_id(token) is not None
            and tokenizer.encode(token, add_special_tokens=False).ids
            == [tokenizer.token_to_id(token)]
            for token in special_tokens
        )
        return {
            "zh_chars_per_token": ratio("zh"),
            "en_chars_per_token": ratio("en"),
            "code_chars_per_token": ratio("code"),
            "math_chars_per_token": ratio("math"),
            "round_trip": round(exact / evaluated, 6) if evaluated else 0,
            "unk_rate": round(unknown / token_count, 8) if token_count else 0,
            "byte_fallback_rate": round(byte_fallback / token_count, 8) if token_count else 0,
            "newline_round_trip": (
                round(newline_exact / newline_documents, 6) if newline_documents else None
            ),
            "indentation_round_trip": (
                round(indentation_exact / indentation_documents, 6)
                if indentation_documents
                else None
            ),
            "special_token_integrity": (
                round(intact_special_tokens / len(special_tokens), 6)
                if special_tokens
                else 0
            ),
            "vocab_garbage_rate": round(garbage_tokens / len(vocab), 8) if vocab else 0,
            "evaluated_documents": evaluated,
            "newline_evaluated_documents": newline_documents,
            "indentation_evaluated_documents": indentation_documents,
        }

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        if not hf_tokenizer_available():
            raise RuntimeError("Hugging Face tokenizers is not installed on this Worker")
        payload = task.get("payload") or {}
        config = payload.get("config") or {}
        data_input = payload.get("data_input") or {}
        sample_tokens = validate_builtin_tokenizer_sample_tokens(config.get("sample_tokens"))
        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        loop = asyncio.get_running_loop()

        def report_progress(progress: dict[str, Any]) -> None:
            sampled_tokens = max(int(progress.get("sampled_tokens") or 0), 0)
            target_tokens = max(int(progress.get("target_tokens") or sample_tokens), 0)
            if progress.get("phase") == "streaming_training":
                ratio = min(sampled_tokens / target_tokens, 1.0) if target_tokens else 0.0
                task_progress = 0.2 + 0.65 * ratio
            else:
                source_index = max(int(progress.get("source_index") or 0), 0)
                task_progress = min(0.18, 0.06 + source_index * 0.02)
            future = asyncio.run_coroutine_threadsafe(
                emit("progress", {**progress, "progress": task_progress}), loop
            )
            future.result()

        await emit("progress", {"progress": 0.05, "phase": "preparing_corpus"})
        try:
            corpus, evaluation_corpus, corpus_stats = await asyncio.to_thread(
                self._corpus,
                data_input,
                sample_tokens,
                run_dir / "corpus-index",
                report_progress,
                get_desired_state,
            )
        except AnalysisInterrupted as exc:
            await emit("progress", {"progress": 0.05, "phase": exc.state})
            return {"controlled_state": exc.state}
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            return {"controlled_state": desired}

        tokenizer, trainer, special_tokens = self._build_tokenizer(
            str(payload.get("algorithm") or "BPE"),
            config,
            int(payload.get("vocab_size") or 32000),
        )
        await emit(
            "progress",
            {
                "progress": 0.2,
                "phase": "streaming_training",
                "sampled_documents": 0,
                "sampled_tokens": 0,
                "target_tokens": sample_tokens or None,
                "progress_basis": "manifest_token_estimate",
            },
        )
        try:
            await asyncio.to_thread(
                tokenizer.train_from_iterator,
                self._batched_corpus(corpus),
                trainer=trainer,
            )
        except AnalysisInterrupted as exc:
            await emit("progress", {"progress": 0.2, "phase": exc.state})
            return {"controlled_state": exc.state}
        if corpus_stats["documents"] <= 0:
            raise ValueError("Tokenizer input selected no readable documents")
        await emit(
            "progress",
            {
                "progress": 0.9,
                "phase": "finalizing",
                "sampled_documents": corpus_stats["documents"],
                "sampled_tokens": corpus_stats["token_estimate"],
                "target_tokens": sample_tokens or None,
            },
        )
        metrics = self._metrics(tokenizer, evaluation_corpus, special_tokens)
        output_dir = run_dir / "tokenizer"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_dir / "tokenizer.json"))
        metadata = {
            "framework": "huggingface-tokenizers",
            "algorithm": payload.get("algorithm"),
            "vocab_size_requested": int(payload.get("vocab_size") or 32000),
            "vocab_size_actual": tokenizer.get_vocab_size(),
            "normalization": config.get("normalization", "NFKC"),
            "byte_fallback": bool(config.get("byte_fallback", False)),
            "special_tokens": special_tokens,
            "sampled_documents": corpus_stats["documents"],
            "sample_token_estimate": corpus_stats["token_estimate"],
            "metrics": metrics,
        }
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "unk_token": special_tokens[0],
                    **({"bos_token": special_tokens[1]} if len(special_tokens) >= 2 else {}),
                    **(
                        {"eos_token": special_tokens[2], "pad_token": special_tokens[2]}
                        if len(special_tokens) >= 3
                        else {}
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        await emit("progress", {"progress": 1.0, "phase": "completed", "metrics": metrics})
        return {
            "framework": "huggingface-tokenizers",
            "metrics": metrics,
            "outputs": [
                {
                    "ref": "tokenizer",
                    "path": str(output_dir),
                    "kind": "tokenizer",
                    "media_type": "application/gzip",
                }
            ],
        }
