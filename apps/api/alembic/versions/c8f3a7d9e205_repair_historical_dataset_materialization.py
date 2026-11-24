"""repair historical dataset materialization metadata

Revision ID: c8f3a7d9e205
Revises: b7e2d5f8a104
Create Date: 2026-08-30
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "c8f3a7d9e205"
down_revision: str | Sequence[str] | None = "b7e2d5f8a104"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DOWNLOAD_SOURCE_TYPES = {"http", "https", "huggingface", "git"}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {}
        return copy.deepcopy(parsed) if isinstance(parsed, dict) else {}
    return {}


def _database_json(value: dict[str, Any], original: Any) -> Any:
    if isinstance(original, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _materialization_kind(row: dict[str, Any]) -> str:
    if (
        row.get("parent_version_id")
        or row.get("conversion_recipe_version_id")
        or row.get("processing_recipe_version_id")
    ):
        return "derived"
    if str(row.get("source_type") or "").lower() in _DOWNLOAD_SOURCE_TYPES:
        return "download"
    return "imported"


def _repair_spec(row: dict[str, Any]) -> dict[str, Any]:
    kind = _materialization_kind(row)
    spec = _json_object(row.get("materialization_spec"))
    spec["schema_version"] = 1
    spec["kind"] = kind

    task_payload = _json_object(row.get("task_payload"))
    task_succeeded = str(row.get("task_status") or "") == "succeeded"
    replay_matches_version = not task_payload.get("dataset_version_id") or str(
        task_payload.get("dataset_version_id")
    ) == str(row["version_id"])
    existing_replay = _json_object(spec.get("replay_payload"))
    existing_adapter = str(spec.get("replay_adapter_id") or "")

    if kind == "download":
        task_replayable = bool(
            task_succeeded
            and row.get("task_id")
            and row.get("task_adapter_id") == "builtin.data-import"
            and task_payload
            and replay_matches_version
        )
        frozen_payload = (
            existing_replay
            if existing_adapter == "builtin.data-import" and existing_replay
            else task_payload
            if task_replayable
            else {}
        )
        replayable = bool(frozen_payload)
        spec["source_type"] = str(frozen_payload.get("source_type") or row.get("source_type") or "")
        spec["source_uri"] = frozen_payload.get("source_uri") or row.get("source_uri")
        spec["source_config"] = copy.deepcopy(
            frozen_payload.get("source_config")
            or spec.get("source_config")
            or _json_object(row.get("dataset_source_config"))
        )
        if replayable:
            spec["replay_adapter_id"] = "builtin.data-import"
            spec["replay_payload"] = frozen_payload
        else:
            spec.pop("replay_adapter_id", None)
            spec.pop("replay_payload", None)
    elif kind == "derived":
        task_replayable = bool(
            task_succeeded
            and row.get("task_id")
            and row.get("task_adapter_id")
            and task_payload
            and replay_matches_version
        )
        spec["source_task_id"] = row.get("task_id")
        spec["parent_dataset_version_id"] = row.get("parent_version_id")
        if existing_adapter and existing_replay:
            spec["replay_adapter_id"] = existing_adapter
            spec["replay_payload"] = existing_replay
        elif task_replayable:
            spec["replay_adapter_id"] = str(row["task_adapter_id"])
            spec["replay_payload"] = task_payload
        else:
            spec.pop("replay_adapter_id", None)
            spec.pop("replay_payload", None)
    else:
        spec["source_type"] = str(row.get("source_type") or "")
        spec["source_uri"] = row.get("source_uri")
        spec["source_config"] = copy.deepcopy(
            spec.get("source_config") or _json_object(row.get("dataset_source_config"))
        )
        spec.pop("replay_adapter_id", None)
        spec.pop("replay_payload", None)

    if row.get("content_hash"):
        spec["expected_content_hash"] = str(row["content_hash"])
        spec["artifact_id"] = str(row["artifact_id"])
    snapshot = _json_object(row.get("source_snapshot"))
    if isinstance(snapshot.get("files"), list) and snapshot["files"]:
        spec["expected_snapshot_files"] = copy.deepcopy(snapshot["files"])
    return spec


def upgrade() -> None:
    bind = op.get_bind()
    datasets = sa.table(
        "datasets",
        sa.column("id", sa.String()),
        sa.column("project_id", sa.String()),
        sa.column("source_type", sa.String()),
        sa.column("source_uri", sa.Text()),
        sa.column("source_config", sa.JSON()),
    )
    versions = sa.table(
        "dataset_versions",
        sa.column("id", sa.String()),
        sa.column("dataset_id", sa.String()),
        sa.column("artifact_id", sa.String()),
        sa.column("task_id", sa.String()),
        sa.column("parent_version_id", sa.String()),
        sa.column("conversion_recipe_version_id", sa.String()),
        sa.column("processing_recipe_version_id", sa.String()),
        sa.column("source_snapshot", sa.JSON()),
        sa.column("materialization_kind", sa.String()),
        sa.column("materialization_spec", sa.JSON()),
        sa.column("materialization_state", sa.String()),
        sa.column("primary_location_id", sa.String()),
    )
    tasks = sa.table(
        "tasks",
        sa.column("id", sa.String()),
        sa.column("adapter_id", sa.String()),
        sa.column("observed_state", sa.String()),
        sa.column("payload", sa.JSON()),
    )
    artifacts = sa.table(
        "artifacts",
        sa.column("id", sa.String()),
        sa.column("content_hash", sa.String()),
        sa.column("created_by_task_attempt_id", sa.String()),
    )
    locations = sa.table(
        "artifact_locations",
        sa.column("id", sa.String()),
        sa.column("artifact_id", sa.String()),
        sa.column("project_id", sa.String()),
        sa.column("worker_id", sa.String()),
        sa.column("storage_type", sa.String()),
        sa.column("availability_state", sa.String()),
        sa.column("metadata", sa.JSON()),
        sa.column("last_verified_at", sa.DateTime(timezone=True)),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    attempts = sa.table(
        "task_attempts",
        sa.column("id", sa.String()),
        sa.column("task_id", sa.String()),
        sa.column("status", sa.String()),
        sa.column("ended_at", sa.DateTime(timezone=True)),
    )

    rows = (
        bind.execute(
            sa.select(
                versions.c.id.label("version_id"),
                versions.c.artifact_id,
                versions.c.task_id,
                versions.c.parent_version_id,
                versions.c.conversion_recipe_version_id,
                versions.c.processing_recipe_version_id,
                versions.c.source_snapshot,
                versions.c.materialization_spec,
                versions.c.primary_location_id,
                datasets.c.project_id,
                datasets.c.source_type,
                datasets.c.source_uri,
                datasets.c.source_config.label("dataset_source_config"),
                tasks.c.adapter_id.label("task_adapter_id"),
                tasks.c.observed_state.label("task_status"),
                tasks.c.payload.label("task_payload"),
                artifacts.c.content_hash,
                artifacts.c.created_by_task_attempt_id,
            )
            .select_from(
                versions.join(datasets, datasets.c.id == versions.c.dataset_id)
                .outerjoin(tasks, tasks.c.id == versions.c.task_id)
                .outerjoin(artifacts, artifacts.c.id == versions.c.artifact_id)
            )
            .order_by(versions.c.id)
        )
        .mappings()
        .all()
    )

    for raw_row in rows:
        row = dict(raw_row)
        kind = _materialization_kind(row)
        primary_id = row.get("primary_location_id")
        if not primary_id and row.get("artifact_id"):
            primary = (
                bind.execute(
                    sa.select(
                        locations.c.id,
                        locations.c.storage_type,
                        locations.c.availability_state,
                        locations.c.metadata,
                        locations.c.last_verified_at,
                        locations.c.created_at,
                        locations.c.updated_at,
                    )
                    .where(
                        locations.c.artifact_id == row["artifact_id"],
                        locations.c.availability_state == "available",
                        sa.or_(
                            locations.c.project_id == row["project_id"],
                            sa.and_(
                                locations.c.project_id.is_(None),
                                locations.c.storage_type == "external",
                            ),
                        ),
                    )
                    .order_by(
                        sa.case((locations.c.storage_type == "managed", 0), else_=1),
                        locations.c.created_at,
                        locations.c.id,
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if primary is not None:
                primary_id = str(primary["id"])

        primary = (
            bind.execute(
                sa.select(
                    locations.c.id,
                    locations.c.storage_type,
                    locations.c.availability_state,
                    locations.c.metadata,
                    locations.c.last_verified_at,
                    locations.c.created_at,
                    locations.c.updated_at,
                ).where(locations.c.id == primary_id)
            )
            .mappings()
            .first()
            if primary_id
            else None
        )
        state = (
            "ready"
            if primary is not None and primary["availability_state"] == "available"
            else "degraded"
            if row.get("artifact_id")
            else "pending"
        )
        bind.execute(
            versions.update()
            .where(versions.c.id == row["version_id"])
            .values(
                materialization_kind=kind,
                materialization_spec=_database_json(
                    _repair_spec(row), row.get("materialization_spec")
                ),
                materialization_state=state,
                primary_location_id=primary_id,
            )
        )

        if primary is None or primary["last_verified_at"] is not None:
            continue
        verified_at = None
        if primary["storage_type"] == "managed":
            verified_at = primary["updated_at"] or primary["created_at"]
        elif row.get("created_by_task_attempt_id"):
            attempt = (
                bind.execute(
                    sa.select(
                        attempts.c.task_id,
                        attempts.c.status,
                        attempts.c.ended_at,
                    ).where(attempts.c.id == row["created_by_task_attempt_id"])
                )
                .mappings()
                .first()
            )
            location_metadata = _json_object(primary["metadata"])
            if (
                attempt is not None
                and attempt["status"] == "succeeded"
                and str(location_metadata.get("task_id") or "") == str(attempt["task_id"])
            ):
                verified_at = attempt["ended_at"] or primary["updated_at"]
        if verified_at is not None:
            bind.execute(
                locations.update()
                .where(locations.c.id == primary_id)
                .values(last_verified_at=verified_at)
            )


def downgrade() -> None:
    # Data-only repair: keep reconstructed immutable recovery metadata when
    # downgrading to the schema-creating revision.
    pass
