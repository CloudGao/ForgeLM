from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from llm_platform_core.mixtures import (
    MIXTURE_MANIFEST_FORMAT,
    RECORD_MANIFEST_FORMAT,
    open_text_manifest,
)

from .data_import import AnalysisInterrupted


def check_control(get_desired_state) -> None:
    state = get_desired_state() if get_desired_state else "running"
    if state in {"paused", "cancelled"}:
        raise AnalysisInterrupted(state)


@contextmanager
def selected_records(binding: dict, work_dir: Path, get_desired_state=None):
    """Read the frozen resolver result, never reinterpret selection predicates."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with Path(binding["runtime_path"]).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != binding["manifest_hash"]:
            raise ValueError("Filtered training manifest hash does not match")
    database = work_dir / "selected-records.sqlite3"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute(
            "CREATE TABLE selected (record_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, logical_path TEXT NOT NULL, record_index INTEGER NOT NULL, payload TEXT NOT NULL, ordinal INTEGER)"
        )
        source_index = int(binding["source_index"])
        with open_text_manifest(binding["runtime_path"]) as stream:
            header = json.loads(stream.readline())
            sources = header.get("sources") or []
            if header.get("format") != MIXTURE_MANIFEST_FORMAT or not 0 <= source_index < len(
                sources
            ):
                raise ValueError("Filtered training requires a frozen record selection manifest")
            source = sources[source_index]
            for key in ("dataset_version_id", "dataset_artifact_hash", "record_manifest_hash"):
                if not binding.get(key) or source.get(key) != binding[key]:
                    raise ValueError(
                        f"Filtered training source {key} does not match its frozen manifest"
                    )
            batch = []
            count = 0
            for line_number, line in enumerate(stream):
                if line_number % 4096 == 0:
                    check_control(get_desired_state)
                record = json.loads(line)
                if record.get("type") != "record":
                    raise ValueError("Filtered training manifest contains a non-record row")
                if int(record.get("source_index", -1)) != source_index:
                    continue
                if not record.get("record_id") or not record.get("content_hash"):
                    raise ValueError("Filtered training record lacks a stable content identity")
                batch.append(
                    (
                        record["record_id"],
                        record["content_hash"],
                        record["logical_path"],
                        int(record["record_index"]),
                        json.dumps(record),
                    )
                )
                count += 1
                if len(batch) >= 4096:
                    connection.executemany("INSERT INTO selected VALUES (?,?,?,?,?,NULL)", batch)
                    batch.clear()
            connection.executemany("INSERT INTO selected VALUES (?,?,?,?,?,NULL)", batch)
            if count != int(binding["selected_document_count"]) or count != int(
                source["selected_document_count"]
            ):
                raise ValueError(
                    "Filtered training selection count does not match its frozen manifest"
                )
        connection.commit()
        yield connection
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def ordered_records(connection):
    for (payload,) in connection.execute(
        "SELECT payload FROM selected ORDER BY logical_path, record_index, record_id"
    ):
        yield json.loads(payload)


def corpus_identities(root: Path, manifest: dict, binding: dict):
    spec = (manifest.get("files") or {}).get("document_identities")
    if spec:
        path = (root / spec["path"]).resolve()
        if (
            root.resolve() not in path.parents
            or not path.is_file()
            or path.stat().st_size != int(spec["size_bytes"])
        ):
            raise ValueError("Corpus document identity file is missing or unsafe")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise ValueError("Corpus document identity file hash does not match")
        with open_text_manifest(path) as stream:
            for line in stream:
                yield json.loads(line)
        return
    # The legacy encoder consumes the exact manifest prefix, but skipping even
    # one empty document destroys the ordinal correspondence.
    count = int(manifest.get("document_count") or 0)
    if (
        manifest.get("source_document_count") != count
        or any(w.get("code") == "empty_documents_skipped" for w in manifest.get("warnings") or [])
        or (manifest.get("selection") or {}).get("mode") not in {"all", "estimated_token_limit"}
        or (manifest.get("identity_snapshot") or {}).get("record_manifest_hash")
        != binding["record_manifest_hash"]
    ):
        raise ValueError("Legacy Corpus document identities cannot be proven; prepare a new Corpus")
    with open_text_manifest(binding["runtime_record_manifest_path"]) as stream:
        header = json.loads(stream.readline())
        if (
            header.get("dataset_version_id") != binding["dataset_version_id"]
            or header.get("format") != RECORD_MANIFEST_FORMAT
        ):
            raise ValueError("Legacy Corpus record manifest belongs to another dataset")
        for line in itertools.islice(stream, count):
            yield json.loads(line)


def selected_ordinals(
    root: Path, manifest: dict, binding: dict, work_dir: Path, get_desired_state=None
):
    snapshot = manifest.get("identity_snapshot") or {}
    for key in ("dataset_version_id", "dataset_artifact_hash", "record_manifest_hash"):
        if snapshot.get(key) != binding[key]:
            raise ValueError(f"Corpus {key} differs from the frozen selection")
    with selected_records(binding, work_dir, get_desired_state) as connection:
        count = 0
        batch = []
        for record in corpus_identities(root, manifest, binding):
            skipped = bool(record.get("skipped"))
            batch.append((-1 if skipped else count, record["record_id"], record["content_hash"]))
            count += not skipped
            if len(batch) >= 4096:
                check_control(get_desired_state)
                connection.executemany(
                    "UPDATE selected SET ordinal=CASE WHEN ordinal IS NULL THEN ? ELSE -2 END WHERE record_id=? AND content_hash=?",
                    batch,
                )
                batch.clear()
        connection.executemany(
            "UPDATE selected SET ordinal=CASE WHEN ordinal IS NULL THEN ? ELSE -2 END WHERE record_id=? AND content_hash=?",
            batch,
        )
        if connection.execute("SELECT count(*) FROM selected WHERE ordinal=-2").fetchone()[0]:
            raise ValueError("Selected identity occurs more than once in Corpus")
        if count != int(manifest["document_count"]):
            raise ValueError("Corpus document identity count does not match token documents")
        missing = connection.execute(
            "SELECT count(*) FROM selected WHERE ordinal IS NULL"
        ).fetchone()[0]
        if missing:
            raise ValueError(
                f"Corpus does not cover {missing} selected document identities; prepare a matching Corpus"
            )
        result = np.fromiter(
            (
                row[0]
                for row in connection.execute(
                    "SELECT ordinal FROM selected WHERE ordinal>=0 ORDER BY ordinal"
                )
            ),
            dtype=np.int64,
        )
        if len(np.unique(result)) != len(result):
            raise ValueError("Selected records map to duplicate Corpus documents")
        return result
