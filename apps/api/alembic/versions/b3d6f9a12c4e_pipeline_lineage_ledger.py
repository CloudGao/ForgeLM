"""add immutable pipeline lineage ledger

Revision ID: b3d6f9a12c4e
Revises: fa9c4d2e7b10
Create Date: 2026-08-21
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "b3d6f9a12c4e"
down_revision: str | Sequence[str] | None = "fa9c4d2e7b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TERMINAL = ("succeeded", "failed", "cancelled", "skipped")
_EXECUTION_ONLY = {"training_run"}
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


def _is_asset_type(value: str) -> bool:
    return bool(
        value
        and (
            value.endswith("_version")
            or value.endswith("_revision")
            or value
            in {
                "artifact",
                "checkpoint",
                "evaluation_run",
                "quantization_run",
                "release",
                "recipe_version",
                "training_run",
            }
        )
    )


def _ref(value: Any, role: str = "") -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    asset_type = str(value.get("asset_type") or value.get("kind") or "")
    asset_id = str(
        value.get("asset_version_id")
        or value.get("asset_id")
        or value.get("id")
        or ""
    )
    if (
        not asset_id
        or not _is_asset_type(asset_type)
        or asset_type in _EXECUTION_ONLY
    ):
        return None
    version = value.get("version")
    if version is None and asset_type == "plan_revision":
        version = value.get("revision")
    label = str(value.get("label") or value.get("name") or "")
    if not label:
        label = f"{asset_type} · {asset_id[:8]}"
    return {
        "asset_type": asset_type,
        "asset_id": asset_id,
        "role": str(value.get("role") or role),
        "label": label,
        "version": str(version) if version not in (None, "") else None,
        "status": str(value.get("status") or ""),
        "metadata": _value(value.get("metadata"), {}),
    }


def _collect(value: Any, role: str = "") -> list[dict[str, Any]]:
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_collect(item, role))
        return result
    if not isinstance(value, dict):
        return []
    result = []
    direct = _ref(value, role)
    if direct:
        result.append(direct)
    for key, item in value.items():
        if key in {"metadata", "asset"}:
            continue
        result.extend(_collect(item, key if key == "data_refs" else role))
    return result


def _dedupe(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in refs:
        key = (item["asset_type"], item["asset_id"], item["role"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _mapping_refs(mapping: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if node_id in {"data", "training"}:
        for item in mapping.get("data") or []:
            if isinstance(item, dict):
                mapped = _ref(
                    {**item, "id": item.get("asset_version_id")},
                    "training_data",
                )
                if mapped:
                    refs.append(mapped)
    for key, asset_type, nodes in (
        ("tokenizer", "tokenizer_version", {"tokenizer", "training"}),
        ("code", "code_version", {"code", "training"}),
        ("plan", "plan_revision", {"training"}),
    ):
        item = mapping.get(key)
        if node_id in nodes and isinstance(item, dict) and item.get("id"):
            mapped = _ref({**item, "kind": asset_type}, key)
            if mapped:
                refs.append(mapped)
    return refs


def _row_ref(
    asset_type: str,
    asset_id: str,
    label: str,
    *,
    version: Any = None,
    status: str = "",
    role: str = "output",
) -> dict[str, Any]:
    suffix = ""
    if version not in (None, ""):
        suffix = f" · {'R' if asset_type == 'plan_revision' else 'V'}{version}"
    return {
        "asset_type": asset_type,
        "asset_id": asset_id,
        "role": role,
        "label": f"{label}{suffix}",
        "version": str(version) if version not in (None, "") else None,
        "status": status,
        "metadata": {},
    }


def _scalar_row(bind: Any, statement: str, record_id: str) -> dict[str, Any] | None:
    row = bind.execute(sa.text(statement), {"id": record_id}).mappings().first()
    return dict(row) if row else None


def _compatibility_outputs(
    bind: Any,
    revision: dict[str, Any],
    mapping: dict[str, Any],
    node_id: str,
) -> list[dict[str, Any]]:
    if node_id == "training" and mapping.get("training_run_id"):
        rows = bind.execute(
            sa.text(
                "SELECT id, step, status FROM checkpoints "
                "WHERE training_run_id = :training_run_id ORDER BY step"
            ),
            {"training_run_id": mapping["training_run_id"]},
        ).mappings()
        return [
            _row_ref(
                "checkpoint",
                str(item["id"]),
                "Checkpoint · step " + f"{int(item['step']):,}",
                status=str(item["status"] or ""),
            )
            for item in rows
        ]
    if node_id == "model":
        if revision.get("model_version_id"):
            items = [
                item
                for item in [
                    _scalar_row(
                        bind,
                        "SELECT id, name, version, status FROM model_versions WHERE id = :id",
                        str(revision["model_version_id"]),
                    )
                ]
                if item is not None
            ]
        elif mapping.get("training_run_id"):
            items = [
                dict(item)
                for item in bind.execute(
                    sa.text(
                        "SELECT model.id, model.name, model.version, model.status "
                        "FROM model_versions AS model "
                        "JOIN checkpoints AS checkpoint "
                        "ON checkpoint.id = model.source_checkpoint_id "
                        "WHERE checkpoint.training_run_id = :training_run_id "
                        "ORDER BY model.created_at DESC"
                    ),
                    {"training_run_id": mapping["training_run_id"]},
                ).mappings()
            ]
        else:
            items = []
        return [
            _row_ref(
                "model_version",
                str(item["id"]),
                str(item["name"]),
                version=item["version"],
                status=str(item["status"] or ""),
            )
            for item in items
        ]
    for expected_node, field, table_name, asset_type, label in (
        ("evaluation", "evaluation_run_id", "evaluation_runs", "evaluation_run", "评测运行"),
        ("quantization", "quantization_run_id", "quantization_runs", "quantization_run", "量化运行"),
        ("release", "release_id", "releases", "release", "发布版本"),
    ):
        record_id = revision.get(field)
        if node_id != expected_node or not record_id:
            continue
        columns = "id, status" if asset_type != "release" else "id, version, status"
        item = _scalar_row(
            bind,
            f"SELECT {columns} FROM {table_name} WHERE id = :id",
            str(record_id),
        )
        if item:
            return [
                _row_ref(
                    asset_type,
                    str(item["id"]),
                    label,
                    version=item.get("version"),
                    status=str(item.get("status") or ""),
                )
            ]
    return []


def _compatibility_terminal_status(
    node_id: str,
    produced: list[dict[str, Any]],
    persisted_status: str,
) -> str | None:
    if node_id not in {"model", "evaluation", "quantization", "release"} or not produced:
        return persisted_status
    if node_id == "model":
        return "succeeded"
    raw_status = str(produced[0].get("status") or "").lower()
    if raw_status in _ACTIVE_DOMAIN_STATUSES:
        return None
    if raw_status in {"failed", "lost"}:
        return "failed"
    if raw_status == "cancelled":
        return "cancelled"
    if raw_status in _READY_DOMAIN_STATUSES:
        return "succeeded"
    return persisted_status


def _backfill() -> None:
    bind = op.get_bind()
    metadata = sa.MetaData()
    node_runs = sa.Table("project_pipeline_node_runs", metadata, autoload_with=bind)
    runs = sa.Table("project_pipeline_runs", metadata, autoload_with=bind)
    revisions = sa.Table("project_pipeline_revisions", metadata, autoload_with=bind)
    tasks = sa.Table("tasks", metadata, autoload_with=bind)
    executions = sa.Table("lineage_executions", metadata, autoload_with=bind)
    relations = sa.Table("lineage_asset_relations", metadata, autoload_with=bind)

    rows = bind.execute(
        sa.select(
            node_runs.c.id.label("node_run_id"),
            node_runs.c.pipeline_run_id,
            node_runs.c.node_id,
            node_runs.c.node_type,
            node_runs.c.status,
            node_runs.c.config_snapshot,
            node_runs.c.input_snapshot,
            node_runs.c.output_snapshot,
            node_runs.c.task_id,
            node_runs.c.started_at,
            node_runs.c.ended_at,
            runs.c.project_id,
            runs.c.input_snapshot.label("run_input_snapshot"),
            runs.c.context_snapshot,
            runs.c.revision_id,
            tasks.c.adapter_id,
            tasks.c.launcher_id,
        )
        .select_from(
            node_runs.join(runs, runs.c.id == node_runs.c.pipeline_run_id).outerjoin(
                tasks, tasks.c.id == node_runs.c.task_id
            )
        )
        .where(node_runs.c.status.in_(_TERMINAL))
        .order_by(node_runs.c.created_at.asc())
    ).mappings()
    now = datetime.now(UTC)
    revision_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        context = _value(row["context_snapshot"], {})
        metadata_value = _value(context.get("__forgelm"), {})
        imported = metadata_value.get("run_kind") == "imported"
        owner = "domain_task" if imported else "pipeline_scheduler"
        config = _value(row["config_snapshot"], {})
        frozen_input = _value(row["input_snapshot"], {})
        frozen_output = _value(row["output_snapshot"], {})
        run_input = _value(row["run_input_snapshot"], {})
        mapping = _value(run_input.get("legacy_mapping"), {})
        consumed = ([] if imported else _collect(config, "binding")) + _collect(
            frozen_input, "input"
        )
        consumed.extend(_collect(frozen_output.get("consumed_assets"), "declared_input"))
        if imported:
            consumed.extend(_mapping_refs(mapping, str(row["node_id"])))

        raw_outputs = (
            frozen_output.get("produced_assets")
            if frozen_output.get("produced_assets") is not None
            else frozen_output.get("versions") or []
        )
        output_refs = _collect(raw_outputs, "output")
        produced: list[dict[str, Any]] = []
        legacy_source = imported and row["node_id"] in {"data", "tokenizer", "code"}
        for item in output_refs:
            if legacy_source or item["asset_type"] == "plan_revision":
                consumed.append(item)
            else:
                produced.append(item)

        revision_id = str(row["revision_id"])
        if imported:
            if revision_id not in revision_cache:
                revision_row = bind.execute(
                    sa.select(revisions).where(revisions.c.id == revision_id)
                ).mappings().first()
                revision_cache[revision_id] = dict(revision_row) if revision_row else {}
            produced.extend(
                _compatibility_outputs(
                    bind,
                    revision_cache[revision_id],
                    mapping,
                    str(row["node_id"]),
                )
            )

        effective_status = str(row["status"])
        if imported:
            projected_status = _compatibility_terminal_status(
                str(row["node_id"]),
                produced,
                effective_status,
            )
            if projected_status is None:
                continue
            effective_status = projected_status

        consumed = _dedupe(consumed)
        produced = _dedupe(produced)
        identity = {
            "execution_owner": owner,
            "config_hash": _hash(config),
        }
        if row["task_id"]:
            identity["task_id"] = row["task_id"]
        if row["adapter_id"]:
            identity["adapter_id"] = row["adapter_id"]
        if row["launcher_id"]:
            identity["launcher_id"] = row["launcher_id"]
        for field in (
            "code_version_id",
            "recipe_version_id",
            "plan_revision_id",
            "training_run_id",
        ):
            if config.get(field):
                identity[field] = config[field]
        if metadata_value.get("source_training_run_id"):
            identity["training_run_id"] = metadata_value["source_training_run_id"]
        command = config.get("command") if isinstance(config.get("command"), dict) else {}
        if command.get("executable"):
            identity["executable"] = command["executable"]
            identity["command_digest"] = _hash(command)

        execution_id = str(uuid4())
        snapshot = {
            "project_id": row["project_id"],
            "pipeline_run_id": row["pipeline_run_id"],
            "node_run_id": row["node_run_id"],
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "status": effective_status,
            "execution_owner": owner,
            "execution_identity": identity,
            "consumed_assets": consumed,
            "produced_assets": produced,
            "started_at": str(row["started_at"] or ""),
            "ended_at": str(row["ended_at"] or ""),
        }
        bind.execute(
            executions.insert().values(
                id=execution_id,
                created_at=now,
                updated_at=now,
                project_id=row["project_id"],
                pipeline_run_id=row["pipeline_run_id"],
                node_run_id=row["node_run_id"],
                task_id=row["task_id"],
                node_id=row["node_id"],
                node_type=row["node_type"],
                status=effective_status,
                execution_owner=owner,
                execution_identity=identity,
                content_hash=_hash(snapshot),
                started_at=row["started_at"],
                ended_at=row["ended_at"],
            )
        )
        for relation_kind, refs in (("consumed", consumed), ("produced", produced)):
            for item in refs:
                bind.execute(
                    relations.insert().values(
                        id=str(uuid4()),
                        created_at=now,
                        updated_at=now,
                        project_id=row["project_id"],
                        execution_id=execution_id,
                        relation_kind=relation_kind,
                        asset_type=item["asset_type"],
                        asset_id=item["asset_id"],
                        role=item["role"],
                        label=item["label"],
                        version=item["version"],
                        status=item["status"],
                        metadata=item["metadata"],
                    )
                )


def upgrade() -> None:
    op.create_table(
        "lineage_executions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_run_id", sa.String(length=36), nullable=False),
        sa.Column("node_run_id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("node_id", sa.String(length=120), nullable=False),
        sa.Column("node_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "execution_owner",
            sa.String(length=40),
            nullable=False,
            server_default="pipeline_scheduler",
        ),
        sa.Column("execution_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('succeeded', 'failed', 'cancelled', 'skipped')",
            name="ck_lineage_execution_terminal_status",
        ),
        sa.ForeignKeyConstraint(["node_run_id"], ["project_pipeline_node_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["project_pipeline_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("node_run_id", name="uq_lineage_execution_node_run"),
    )
    for column in (
        "project_id",
        "pipeline_run_id",
        "node_run_id",
        "task_id",
        "node_id",
        "node_type",
        "status",
        "content_hash",
    ):
        op.create_index(f"ix_lineage_executions_{column}", "lineage_executions", [column])

    op.create_table(
        "lineage_asset_relations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("execution_id", sa.String(length=36), nullable=False),
        sa.Column("relation_kind", sa.String(length=24), nullable=False),
        sa.Column("asset_type", sa.String(length=80), nullable=False),
        sa.Column("asset_id", sa.String(length=120), nullable=False),
        sa.Column("role", sa.String(length=80), nullable=False, server_default=""),
        sa.Column("label", sa.String(length=220), nullable=False),
        sa.Column("version", sa.String(length=80), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False, server_default=""),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.CheckConstraint(
            "relation_kind IN ('consumed', 'produced')",
            name="ck_lineage_asset_relation_kind",
        ),
        sa.ForeignKeyConstraint(["execution_id"], ["lineage_executions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "execution_id",
            "relation_kind",
            "asset_type",
            "asset_id",
            "role",
            name="uq_lineage_asset_relation",
        ),
    )
    for column in ("project_id", "execution_id", "relation_kind", "asset_type", "asset_id"):
        op.create_index(
            f"ix_lineage_asset_relations_{column}",
            "lineage_asset_relations",
            [column],
        )
    op.create_index(
        "ix_lineage_asset_relations_project_asset",
        "lineage_asset_relations",
        ["project_id", "asset_type", "asset_id"],
    )
    _backfill()


def downgrade() -> None:
    op.drop_table("lineage_asset_relations")
    op.drop_table("lineage_executions")
