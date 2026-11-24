from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Inventory:
    files: int
    bytes: int


def inventory(root: Path) -> Inventory:
    files = 0
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                files += 1
                total += path.stat().st_size
    return Inventory(files=files, bytes=total)


def inventory_many(roots: list[Path]) -> Inventory:
    values = [inventory(root) for root in roots]
    return Inventory(
        files=sum(value.files for value in values),
        bytes=sum(value.bytes for value in values),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def uri_to_path(uri: str) -> Path:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"managed artifact is not a file URI: {uri}")
    value = unquote(parsed.path)
    if parsed.netloc:
        value = f"//{parsed.netloc}{value}"
    if len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    return Path(value)


def managed_payloads(
    connection: sqlite3.Connection, project_id: str, uri_prefix: str
) -> list[tuple[str, Path]]:
    rows = connection.execute(
        """
        SELECT artifacts.content_hash, artifact_locations.uri
        FROM artifact_locations
        JOIN artifacts ON artifacts.id = artifact_locations.artifact_id
        WHERE artifact_locations.storage_type = 'managed'
          AND artifact_locations.project_id = ?
          AND artifact_locations.uri LIKE ?
        ORDER BY artifact_locations.uri
        """,
        (project_id, f"{uri_prefix}%"),
    ).fetchall()
    return [(str(content_hash), uri_to_path(str(uri))) for content_hash, uri in rows]


def verify_payloads(payloads: list[tuple[str, Path]]) -> None:
    for expected_hash, path in payloads:
        if not path.is_file():
            raise FileNotFoundError(f"managed artifact payload is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"managed artifact hash mismatch: {path} expected={expected_hash} actual={actual_hash}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move global runs/artifacts into one project's workspace"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-state-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    database = args.database.resolve()
    source_state = args.source_state_root.resolve()
    workspace = args.workspace.resolve()
    source_artifacts = source_state / "artifacts"
    source_runs = source_state / "runs"
    target_state = workspace / ".llm-platform"
    target_artifacts = target_state / "artifacts"
    target_runs = target_state / "runs"

    if not database.is_file():
        raise FileNotFoundError(database)
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)
    if not source_artifacts.is_dir():
        raise NotADirectoryError(source_artifacts)
    if not source_runs.is_dir():
        raise NotADirectoryError(source_runs)
    if target_artifacts.exists() or target_runs.exists():
        raise FileExistsError(f"migration target already exists: {target_state}")

    source_artifact_uri = source_artifacts.as_uri().rstrip("/") + "/"
    target_artifact_uri = target_artifacts.as_uri().rstrip("/") + "/"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(artifact_locations)")
        }
        if "project_id" not in columns:
            raise RuntimeError("database schema has not been upgraded for project-scoped storage")
        project = connection.execute(
            "SELECT workspace_uri FROM projects WHERE id = ?", (args.project_id,)
        ).fetchone()
        if project is None:
            raise ValueError("project does not exist")
        configured = Path(str(project[0])).resolve()
        if configured != workspace:
            raise ValueError(
                f"project workspace mismatch: database={configured} requested={workspace}"
            )
        source_location_count = connection.execute(
            """
            SELECT COUNT(*) FROM artifact_locations
            WHERE storage_type = 'managed'
              AND project_id = ?
              AND uri LIKE ?
            """,
            (args.project_id, f"{source_artifact_uri}%"),
        ).fetchone()[0]
        payloads_before = managed_payloads(
            connection, args.project_id, source_artifact_uri
        )
        task_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM tasks WHERE project_id = ? ORDER BY id",
                (args.project_id,),
            )
        ]

    artifact_directories = sorted(
        {path.parent.resolve() for _, path in payloads_before}, key=str
    )
    for directory in artifact_directories:
        directory.relative_to(source_artifacts)
    run_directories = [source_runs / task_id for task_id in task_ids]
    run_directories = [path for path in run_directories if path.is_dir()]
    before_artifacts = inventory_many(artifact_directories)
    before_runs = inventory_many(run_directories)

    print(f"artifacts: {before_artifacts.files} files, {before_artifacts.bytes} bytes")
    print(f"runs: {before_runs.files} files, {before_runs.bytes} bytes")
    print(f"managed locations to migrate: {source_location_count}")
    print(f"target: {target_state}")
    if not args.execute:
        print("preview only; pass --execute to migrate")
        return 0

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = database.with_name(f"{database.stem}.before-project-storage-{timestamp}.db")
    with sqlite3.connect(database) as source_db, sqlite3.connect(backup) as backup_db:
        source_db.execute("PRAGMA wal_checkpoint(FULL)")
        source_db.backup(backup_db)

    moved: list[tuple[Path, Path]] = []
    try:
        target_artifacts.mkdir(parents=True, exist_ok=False)
        target_runs.mkdir(parents=True, exist_ok=False)
        for source in artifact_directories:
            target = target_artifacts / source.relative_to(source_artifacts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((target, source))
        for source in run_directories:
            target = target_runs / source.name
            shutil.move(str(source), str(target))
            moved.append((target, source))

        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE artifact_locations
                SET uri = ? || substr(uri, length(?) + 1)
                WHERE storage_type = 'managed'
                  AND project_id = ?
                  AND uri LIKE ?
                """,
                (
                    target_artifact_uri,
                    source_artifact_uri,
                    args.project_id,
                    f"{source_artifact_uri}%",
                ),
            )
            changed_locations = connection.execute("SELECT changes()").fetchone()[0]
            connection.execute(
                """
                UPDATE artifacts
                SET uri = ? || substr(uri, length(?) + 1)
                WHERE uri LIKE ?
                  AND id IN (
                    SELECT artifact_id FROM artifact_locations
                    WHERE project_id = ? AND storage_type = 'managed'
                  )
                """,
                (
                    target_artifact_uri,
                    source_artifact_uri,
                    f"{source_artifact_uri}%",
                    args.project_id,
                ),
            )
            connection.commit()

        if changed_locations != source_location_count:
            raise RuntimeError(
                f"location update count mismatch: expected={source_location_count} actual={changed_locations}"
            )
        if inventory(target_artifacts) != before_artifacts:
            raise RuntimeError("artifact directory inventory changed during migration")
        if inventory(target_runs) != before_runs:
            raise RuntimeError("run directory inventory changed during migration")
        with sqlite3.connect(database) as connection:
            payloads = managed_payloads(
                connection, args.project_id, target_artifact_uri
            )
        verify_payloads(payloads)
    except Exception:
        for target, source in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        for root in (target_artifacts, target_runs):
            if root.exists() and not any(root.rglob("*")):
                root.rmdir()
        shutil.copy2(backup, database)
        raise

    print(f"migration complete; database backup: {backup}")
    print(f"verified managed payloads: {len(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
