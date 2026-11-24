"""repair imported lineage from projected domain outputs

Revision ID: c6a1e7d4b290
Revises: b3d6f9a12c4e
Create Date: 2026-08-22
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "c6a1e7d4b290"
down_revision: str | Sequence[str] | None = "b3d6f9a12c4e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIVE_DOMAIN_STATUSES = {
    "queued",
    "assigned",
    "starting",
    "running",
    "pausing",
    "stopping",
    "resolving",
}
_READY_DOMAIN_STATUSES = {
    "succeeded",
    "passed",
    "ready",
    "available",
    "published",
    "official",
}


def _value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_ref(
    asset_type: str,
    item: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    version = item.get("version")
    return {
        "asset_type": asset_type,
        "asset_id": str(item["id"]),
        "role": "output",
        "label": label,
        "version": str(version) if version not in (None, "") else None,
        "status": str(item.get("status") or ""),
        "metadata": {},
    }


def _domain_outputs(
    bind: Any,
    revision: dict[str, Any],
    training_run_id: str,
    node_id: str,
) -> list[dict[str, Any]]:
    if node_id == "model":
        if revision.get("model_version_id"):
            rows = bind.execute(
                sa.text(
                    "SELECT id, name, version, status FROM model_versions WHERE id = :id"
                ),
                {"id": revision["model_version_id"]},
            ).mappings()
        elif training_run_id:
            rows = bind.execute(
                sa.text(
                    "SELECT model.id, model.name, model.version, model.status "
                    "FROM model_versions AS model "
                    "JOIN checkpoints AS checkpoint "
                    "ON checkpoint.id = model.source_checkpoint_id "
                    "WHERE checkpoint.training_run_id = :training_run_id "
                    "ORDER BY model.created_at DESC"
                ),
                {"training_run_id": training_run_id},
            ).mappings()
        else:
            return []
        return [
            _row_ref(
                "model_version",
                dict(item),
                f"{item['name']} · V{item['version']}",
            )
            for item in rows
        ]

    field, table_name, asset_type, label = {
        "evaluation": (
            "evaluation_run_id",
            "evaluation_runs",
            "evaluation_run",
            "评测运行",
        ),
        "quantization": (
            "quantization_run_id",
            "quantization_runs",
            "quantization_run",
            "量化运行",
        ),
        "release": ("release_id", "releases", "release", "发布版本"),
    }[node_id]
    record_id = revision.get(field)
    if not record_id:
        return []
    columns = "id, status" if node_id != "release" else "id, version, status"
    item = bind.execute(
        sa.text(f"SELECT {columns} FROM {table_name} WHERE id = :id"),
        {"id": record_id},
    ).mappings().first()
    if item is None:
        return []
    suffix = f" · V{item['version']}" if item.get("version") not in (None, "") else ""
    return [_row_ref(asset_type, dict(item), f"{label}{suffix}")]


def _terminal_status(
    node_id: str,
    outputs: list[dict[str, Any]],
    persisted_status: str,
) -> str | None:
    if not outputs:
        return persisted_status
    if node_id == "model":
        return "succeeded"
    raw_status = str(outputs[0].get("status") or "").lower()
    if raw_status in _ACTIVE_DOMAIN_STATUSES:
        return None
    if raw_status in {"failed", "lost"}:
        return "failed"
    if raw_status == "cancelled":
        return "cancelled"
    if raw_status in _READY_DOMAIN_STATUSES:
        return "succeeded"
    return persisted_status


def _relation_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_type": item["asset_type"],
        "asset_id": item["asset_id"],
        "role": item["role"],
        "label": item["label"],
        "version": item["version"],
        "status": item["status"],
        "metadata": _value(item["metadata"], {}),
    }


def _refresh_hash(bind: Any, execution_id: str) -> None:
    execution = bind.execute(
        sa.text("SELECT * FROM lineage_executions WHERE id = :id"),
        {"id": execution_id},
    ).mappings().one()
    relations = list(
        bind.execute(
            sa.text(
                "SELECT * FROM lineage_asset_relations "
                "WHERE execution_id = :execution_id ORDER BY created_at, id"
            ),
            {"execution_id": execution_id},
        ).mappings()
    )
    snapshot = {
        "project_id": execution["project_id"],
        "pipeline_run_id": execution["pipeline_run_id"],
        "node_run_id": execution["node_run_id"],
        "node_id": execution["node_id"],
        "node_type": execution["node_type"],
        "status": execution["status"],
        "execution_owner": execution["execution_owner"],
        "execution_identity": _value(execution["execution_identity"], {}),
        "consumed_assets": [
            _relation_snapshot(dict(item))
            for item in relations
            if item["relation_kind"] == "consumed"
        ],
        "produced_assets": [
            _relation_snapshot(dict(item))
            for item in relations
            if item["relation_kind"] == "produced"
        ],
        "started_at": str(execution["started_at"] or ""),
        "ended_at": str(execution["ended_at"] or ""),
    }
    bind.execute(
        sa.text(
            "UPDATE lineage_executions "
            "SET content_hash = :content_hash, updated_at = :updated_at WHERE id = :id"
        ),
        {
            "id": execution_id,
            "content_hash": _hash(snapshot),
            "updated_at": datetime.now(UTC),
        },
    )


def upgrade() -> None:
    bind = op.get_bind()
    executions = list(
        bind.execute(
            sa.text(
                "SELECT execution.*, run.context_snapshot, run.input_snapshot AS run_input, "
                "run.revision_id FROM lineage_executions AS execution "
                "JOIN project_pipeline_runs AS run ON run.id = execution.pipeline_run_id "
                "WHERE execution.node_id IN ('model', 'evaluation', 'quantization', 'release')"
            )
        ).mappings()
    )
    revision_cache: dict[str, dict[str, Any]] = {}
    now = datetime.now(UTC)
    for execution in executions:
        context = _value(execution["context_snapshot"], {})
        metadata = _value(context.get("__forgelm"), {})
        if metadata.get("run_kind") != "imported":
            continue
        revision_id = str(execution["revision_id"])
        if revision_id not in revision_cache:
            revision = bind.execute(
                sa.text("SELECT * FROM project_pipeline_revisions WHERE id = :id"),
                {"id": revision_id},
            ).mappings().first()
            revision_cache[revision_id] = dict(revision) if revision else {}
        mapping = _value(_value(execution["run_input"], {}).get("legacy_mapping"), {})
        identity = _value(execution["execution_identity"], {})
        training_run_id = str(
            metadata.get("source_training_run_id")
            or mapping.get("training_run_id")
            or identity.get("training_run_id")
            or ""
        )
        outputs = _domain_outputs(
            bind,
            revision_cache[revision_id],
            training_run_id,
            str(execution["node_id"]),
        )
        status = _terminal_status(
            str(execution["node_id"]),
            outputs,
            str(execution["status"]),
        )
        if status is None:
            bind.execute(
                sa.text(
                    "DELETE FROM lineage_asset_relations WHERE execution_id = :execution_id"
                ),
                {"execution_id": execution["id"]},
            )
            bind.execute(
                sa.text("DELETE FROM lineage_executions WHERE id = :execution_id"),
                {"execution_id": execution["id"]},
            )
            continue
        bind.execute(
            sa.text(
                "UPDATE lineage_executions SET status = :status, updated_at = :updated_at "
                "WHERE id = :execution_id"
            ),
            {
                "status": status,
                "updated_at": now,
                "execution_id": execution["id"],
            },
        )
        for item in outputs:
            exists = bind.execute(
                sa.text(
                    "SELECT id FROM lineage_asset_relations WHERE execution_id = :execution_id "
                    "AND relation_kind = 'produced' AND asset_type = :asset_type "
                    "AND asset_id = :asset_id AND role = :role"
                ),
                {"execution_id": execution["id"], **item},
            ).first()
            if exists:
                continue
            bind.execute(
                sa.text(
                    "INSERT INTO lineage_asset_relations "
                    "(id, created_at, updated_at, project_id, execution_id, relation_kind, "
                    "asset_type, asset_id, role, label, version, status, metadata) VALUES "
                    "(:id, :created_at, :updated_at, :project_id, :execution_id, 'produced', "
                    ":asset_type, :asset_id, :role, :label, :version, :status, :metadata)"
                ),
                {
                    "id": str(uuid4()),
                    "created_at": now,
                    "updated_at": now,
                    "project_id": execution["project_id"],
                    "execution_id": execution["id"],
                    **item,
                    "metadata": json.dumps(
                        item["metadata"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            )
        _refresh_hash(bind, str(execution["id"]))


def downgrade() -> None:
    # This data-only repair makes existing b3 ledgers match b3's corrected backfill.
    pass
