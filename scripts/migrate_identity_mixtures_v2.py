from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from llm_platform_core.mixtures import (
    MIXTURE_INDEX_FORMAT,
    MIXTURE_MANIFEST_FORMAT,
    REFERENCE_RESOLVER_VERSION,
    REFERENCE_SAMPLE_BLOCK_SIZE,
    REFERENCE_SAMPLER_VERSION,
    WEIGHT_NORMALIZER_VERSION,
    canonical_json,
    is_identity_selection,
    stream_key,
)

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "lost"}


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"managed artifact is not a file URI: {uri}")
    value = unquote(parsed.path)
    if parsed.netloc:
        value = f"//{parsed.netloc}{value}"
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value).resolve()


def workspace_path(uri: str) -> Path:
    return uri_to_path(uri) if uri.startswith("file:") else Path(uri).resolve()


def manifest_header(path: Path) -> dict:
    with path.open("rb") as raw:
        opener = gzip.GzipFile(fileobj=raw) if raw.read(2) == b"\x1f\x8b" else raw
        raw.seek(0)
        line = opener.readline()
    return json.loads(line)


def source_header(entry: dict) -> dict:
    count = int(entry.get("selected_document_count") or entry.get("document_count") or 0)
    tokens = int(
        entry.get("selected_tokens")
        or entry.get("selected_token_estimate")
        or entry.get("token_estimate")
        or 0
    )
    if count <= 0 or tokens <= 0:
        raise ValueError("identity source is missing exact document/token counts")
    return {
        "dataset_version_id": str(entry["dataset_version_id"]),
        "dataset_name": entry.get("name"),
        "dataset_artifact_id": entry.get("dataset_artifact_id"),
        "dataset_artifact_hash": entry.get("dataset_artifact_hash"),
        "record_manifest_artifact_id": entry["record_manifest_artifact_id"],
        "record_manifest_hash": entry["record_manifest_hash"],
        "weight": entry.get("weight"),
        "weight_units": int(entry["weight_units"]),
        "selection": entry.get("selection") or {},
        "selected_document_count": count,
        "selected_token_estimate": tokens,
    }


def make_payload(row: sqlite3.Row) -> bytes:
    entries = json.loads(row["entries"])
    if not entries or not all(is_identity_selection(entry.get("selection")) for entry in entries):
        raise ValueError("mixture is not an identity selection")
    header = {
        "type": "header",
        "format": MIXTURE_INDEX_FORMAT,
        "mixture_version_id": row["version_id"],
        "seed": int(row["seed"]),
        "target_tokens": int(row["target_tokens"]),
        "resolver": REFERENCE_RESOLVER_VERSION,
        "weight_normalizer": WEIGHT_NORMALIZER_VERSION,
        "sampler": REFERENCE_SAMPLER_VERSION,
        "sample_block_size": REFERENCE_SAMPLE_BLOCK_SIZE,
        "sampling_mode": "deterministic_with_replacement",
        "record_order": "record_manifest",
        "sources": [source_header(entry) for entry in entries],
    }
    header["stream_key"] = stream_key(header)
    return (canonical_json(header) + "\n").encode("utf-8")


def backup_database(database: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = database.parent / "backups" / f"platform.before-mixture-index-v2-{stamp}.db"
    backup.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace full-selection V1 mixture manifests with header-only V2 references"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    database = args.database.resolve()
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    project = connection.execute(
        "SELECT workspace_uri FROM projects WHERE id = ?", (args.project_id,)
    ).fetchone()
    if project is None:
        raise ValueError("project does not exist")
    workspace = workspace_path(str(project["workspace_uri"]))
    artifact_root = (workspace / ".llm-platform" / "artifacts").resolve()
    rows = connection.execute(
        """
        SELECT mv.id AS version_id, mv.entries, mv.seed, mv.target_tokens,
               mv.manifest_artifact_id AS old_artifact_id,
               a.content_hash AS old_hash, a.size_bytes AS old_size,
               al.uri AS old_uri
        FROM data_mixture_versions mv
        JOIN data_mixtures m ON m.id = mv.mixture_id
        JOIN artifacts a ON a.id = mv.manifest_artifact_id
        JOIN artifact_locations al ON al.artifact_id = a.id
        WHERE m.project_id = ? AND mv.status = 'ready' AND mv.locked = 1
          AND a.kind = 'data_mixture_manifest' AND al.storage_type = 'managed'
          AND al.project_id = ? AND al.availability_state = 'available'
        ORDER BY mv.created_at
        """,
        (args.project_id, args.project_id),
    ).fetchall()
    candidates: list[tuple[sqlite3.Row, Path, bytes, str]] = []
    for row in rows:
        old_path = uri_to_path(str(row["old_uri"]))
        if not old_path.is_file():
            continue
        old_header = manifest_header(old_path)
        if old_header.get("format") not in {MIXTURE_MANIFEST_FORMAT, MIXTURE_INDEX_FORMAT}:
            continue
        if old_header.get("sampler") == REFERENCE_SAMPLER_VERSION:
            continue
        try:
            payload = make_payload(row)
        except ValueError:
            continue
        candidates.append((row, old_path, payload, hashlib.sha256(payload).hexdigest()))
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "candidate_count": len(candidates),
                "old_size_bytes": sum(int(item[0]["old_size"]) for item in candidates),
                "new_size_bytes": sum(len(item[2]) for item in candidates),
                "versions": [item[0]["version_id"] for item in candidates],
            },
            indent=2,
        )
    )
    if not args.execute or not candidates:
        return 0

    old_ids = [str(item[0]["old_artifact_id"]) for item in candidates]
    active = connection.execute(
        f"SELECT id, observed_state FROM tasks WHERE observed_state NOT IN (?, ?, ?, ?) "
        f"AND ({' OR '.join('payload LIKE ?' for _ in old_ids)})",
        (*TERMINAL_STATES, *(f"%{artifact_id}%" for artifact_id in old_ids)),
    ).fetchall()
    if active:
        raise RuntimeError(f"non-terminal tasks still reference old artifacts: {list(active)}")

    backup = backup_database(database)
    now = datetime.now(UTC).isoformat()
    deletions: list[Path] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        for row, old_path, payload, digest in candidates:
            artifact_dir = artifact_root / "data_mixture_manifest" / digest[:2] / digest
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload_path = artifact_dir / "data-mixture-index.jsonl"
            temporary = artifact_dir / ".data-mixture-index.jsonl.tmp"
            temporary.write_bytes(payload)
            temporary.replace(payload_path)
            metadata = {
                "format": MIXTURE_INDEX_FORMAT,
                "migrated_from_artifact_id": row["old_artifact_id"],
            }
            (artifact_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "data_mixture_manifest",
                        "content_hash": digest,
                        "size_bytes": len(payload),
                        "media_type": "application/x-ndjson",
                        "filename": payload_path.name,
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            existing = connection.execute(
                "SELECT id FROM artifacts WHERE kind = ? AND content_hash = ?",
                ("data_mixture_manifest", digest),
            ).fetchone()
            artifact_id = str(existing["id"]) if existing else str(uuid4())
            if existing is None:
                connection.execute(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "data_mixture_manifest",
                        payload_path.as_uri(),
                        digest,
                        len(payload),
                        "application/x-ndjson",
                        1,
                        None,
                        "available",
                        json.dumps(metadata),
                        artifact_id,
                        now,
                        now,
                    ),
                )
            location = connection.execute(
                "SELECT id FROM artifact_locations WHERE artifact_id = ? AND uri = ?",
                (artifact_id, payload_path.as_uri()),
            ).fetchone()
            if location is None:
                connection.execute(
                    "INSERT INTO artifact_locations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        artifact_id,
                        "managed",
                        payload_path.as_uri(),
                        None,
                        "available",
                        "{}",
                        now,
                        str(uuid4()),
                        now,
                        now,
                        args.project_id,
                    ),
                )
            connection.execute(
                """
                UPDATE data_mixture_versions
                SET manifest_artifact_id = ?, manifest_hash = ?, resolver_version = ?,
                    sampler_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    artifact_id,
                    digest,
                    REFERENCE_RESOLVER_VERSION,
                    REFERENCE_SAMPLER_VERSION,
                    now,
                    row["version_id"],
                ),
            )
            remaining = connection.execute(
                "SELECT count(*) FROM data_mixture_versions WHERE manifest_artifact_id = ?",
                (row["old_artifact_id"],),
            ).fetchone()[0]
            if not remaining:
                connection.execute(
                    "UPDATE artifacts SET availability_state = 'missing', updated_at = ? WHERE id = ?",
                    (now, row["old_artifact_id"]),
                )
                connection.execute(
                    "UPDATE artifact_locations SET availability_state = 'missing', "
                    "updated_at = ? WHERE artifact_id = ?",
                    (now, row["old_artifact_id"]),
                )
                deletions.append(old_path.parent.resolve())
            connection.execute(
                """
                INSERT INTO audit_events
                (project_id, actor, action, subject_type, subject_id, reason, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    args.project_id,
                    "codex-storage-migration",
                    "mixture.reference_index_migrated",
                    "data_mixture_version",
                    row["version_id"],
                    "Replace duplicated per-record manifest with immutable references",
                    json.dumps(
                        {
                            "old_artifact_id": row["old_artifact_id"],
                            "old_hash": row["old_hash"],
                            "new_artifact_id": artifact_id,
                            "new_hash": digest,
                        }
                    ),
                    now,
                ),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    reclaimed = 0
    kind_root = (artifact_root / "data_mixture_manifest").resolve()
    for directory in deletions:
        if not directory.is_relative_to(kind_root) or directory == kind_root:
            raise RuntimeError(f"refusing to delete unexpected path: {directory}")
        if directory.is_dir():
            reclaimed += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory)
    print(
        json.dumps(
            {
                "backup": str(backup),
                "deleted_directories": [str(path) for path in deletions],
                "reclaimed_bytes": reclaimed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
