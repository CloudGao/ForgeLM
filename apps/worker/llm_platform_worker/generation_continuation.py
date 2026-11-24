"""Read-only, bounded checkpoint sealing for native generation batch continuation."""

import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack, closing
from uuid import UUID

from .file_cache import release_file_cache, release_file_cache_descriptor


def sealed_checkpoint_index(root, binding, provenance):
    from .data_generation_runtime import generation_checkpoint_identity, generation_job_key

    original = binding["task"]
    attempt_id = str(UUID(binding["attempt_id"]))
    version_id = str(UUID(original["payload"]["output_dataset_version_id"]))
    signature = generation_checkpoint_identity(original)
    source = root / f"{version_id}-{attempt_id}"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Generation continuation checkpoint directory is unavailable")
    manifest = json.loads((source / "resume-state.json").read_text(encoding="utf-8"))
    if manifest != {"schema": "forgelm-generation-resume-v1", "task_id": original["id"],
                    "output_dataset_version_id": version_id, "identity": signature}:
        raise ValueError("Generation continuation checkpoint identity mismatch")
    files = [source / "data.partial", source / "skipped.records"]
    if any(path.is_symlink() for path in files) or not files[0].is_file():
        raise ValueError("Generation continuation checkpoint files are unavailable")
    fingerprint = json.dumps([
        [path.name, path.stat().st_size, path.stat().st_mtime_ns] if path.exists() else [path.name, None]
        for path in files
    ], sort_keys=True)
    destination = root / f"checkpoint-{signature}-{attempt_id}.sqlite"
    if destination.exists():
        with closing(sqlite3.connect(f"{destination.as_uri()}?mode=ro", uri=True)) as db:
            metadata = dict(db.execute("SELECT key, value FROM metadata"))
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("Sealed generation checkpoint changed after publication")
        return destination, metadata
    temporary = destination.with_suffix(".partial.sqlite")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA cache_size=-8192")
        db.execute("CREATE TABLE rows (file TEXT, ordinal INTEGER, start INTEGER, end INTEGER, style TEXT, offset INTEGER, length INTEGER, sha256 TEXT, skipped INTEGER, PRIMARY KEY(file, ordinal, start, end, style)) WITHOUT ROWID")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        count = 0
        for skipped, path in enumerate(files):
            if not path.exists():
                continue
            with path.open("rb") as stream:
                offset = 0
                for line in stream:
                    if not line.endswith(b"\n"):
                        break  # Only an incomplete terminal JSONL frame may be ignored.
                    row = json.loads(line)
                    meta = row["metadata"]
                    if any(meta.get(key) != value for key, value in provenance.items()):
                        raise ValueError("Generation continuation row provenance mismatch")
                    if (skipped and not row.get("reason")) or (not skipped and not isinstance(row.get("text"), str)):
                        raise ValueError("Invalid generation continuation record")
                    file, ordinal, _, start, end, style = generation_job_key(meta)
                    db.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (file, ordinal, start, end, style, offset, len(line),
                                hashlib.sha256(line).hexdigest(), skipped))
                    offset += len(line)
                    digest.update(bytes([skipped]) + line)
                    count += 1
                    if count % 10000 == 0:
                        db.commit()
                        release_file_cache_descriptor(stream.fileno())
                        release_file_cache(temporary)
            release_file_cache(path)
        after = json.dumps([
            [path.name, path.stat().st_size, path.stat().st_mtime_ns] if path.exists() else [path.name, None]
            for path in files
        ], sort_keys=True)
        if after != fingerprint:
            raise ValueError("Generation checkpoint changed while sealing")
        metadata = {"fingerprint": fingerprint, "sha256": digest.hexdigest(),
                    "records": str(count), "identity": signature, "storage": "offsets-v2"}
        db.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        db.commit()
    finally:
        db.close()
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    temporary.replace(destination)
    release_file_cache(destination)
    return destination, metadata


def checkpoint_rows(db, metadata, source, selection):
    """Legacy full rows remain readable; new indexes reference immutable source bytes."""
    params = (selection["logical_file"], selection["start"], selection["stop"])
    clause = " FROM rows WHERE file=? AND ordinal>=? AND ordinal<? ORDER BY ordinal, start, style"
    if metadata.get("storage") != "offsets-v2":
        yield from db.execute("SELECT row, skipped" + clause, params)
        return
    with ExitStack() as stack:
        streams = {}
        for offset, length, digest, skipped in db.execute("SELECT offset, length, sha256, skipped" + clause, params):
            if skipped not in (0, 1) or offset < 0 or length <= 0:
                raise ValueError("Invalid generation checkpoint offset")
            if skipped not in streams:
                name = "skipped.records" if skipped else "data.partial"
                streams[skipped] = stack.enter_context((source / name).open("rb"))
            stream = streams[skipped]
            stream.seek(offset)
            line = stream.read(length)
            if len(line) != length or hashlib.sha256(line).hexdigest() != digest:
                raise ValueError("Generation checkpoint source row changed")
            yield line, skipped


def import_checkpoint_range(task, root, output, provenance, report, completed):
    from .data_generation_runtime import generation_job_key

    binding = task["payload"].get("generation_continuation")
    if not binding:
        return set()
    if task["runtime_context"]["worker_id"] != binding["worker_id"]:
        raise ValueError("Generation continuation must use the source Worker")
    selection = task["payload"]["generation"]["record_range"]
    from pathlib import Path
    source_name = f"{binding['task']['payload']['output_dataset_version_id']}-{binding['attempt_id']}"
    source_root = next((candidate for candidate in [root, *map(Path, task.get("storage_resume_roots", []))]
                        if (candidate / source_name).is_dir()), root)
    path, metadata = sealed_checkpoint_index(source_root, binding, provenance)
    report["checkpoint_source"] = {
        "task_id": binding["task"]["id"], "attempt_id": binding["attempt_id"], **metadata,
    }
    imported = set()
    with (
        closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db,
        (output / "data.partial").open("a", encoding="utf-8", newline="\n") as accepted,
        (output / "skipped.records").open("a", encoding="utf-8", newline="\n") as skipped_stream,
    ):
        source = source_root / source_name
        for serialized, skipped in checkpoint_rows(db, metadata, source, selection):
            row = json.loads(serialized)
            key = generation_job_key(row["metadata"])
            imported.add(key)
            if key in completed:
                continue
            row["metadata"]["checkpoint_source_task_id"] = binding["task"]["id"]
            row["metadata"]["checkpoint_source_attempt_id"] = binding["attempt_id"]
            (skipped_stream if skipped else accepted).write(json.dumps(row, ensure_ascii=False) + "\n")
            completed.add(key)
            counter = "resumed_skipped_records" if skipped else "resumed_records"
            report[counter] = int(report.get(counter) or 0) + 1
            if not skipped:
                report["resumed_generated_tokens"] = int(report.get("resumed_generated_tokens") or 0) + int(row["metadata"].get("generation_tokens") or 0)
        for stream in (accepted, skipped_stream):
            stream.flush()
            os.fsync(stream.fileno())
    report["checkpoint_imported_records"] = len(imported)
    return imported


def remaining_verified_jobs(source, completed, imported):
    from .data_generation_runtime import generation_job_key

    missing = set(imported)
    for item in source:
        key = generation_job_key(item)
        missing.discard(key)
        if key not in completed:
            yield item
    if missing:
        raise ValueError("Continuation rows do not match the immutable source range")
