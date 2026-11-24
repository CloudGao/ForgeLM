"""import existing named pipeline runs into the instance hierarchy

Revision ID: fa9c4d2e7b10
Revises: f8a3b5c7d9e1
Create Date: 2026-08-16
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "fa9c4d2e7b10"
down_revision: str | Sequence[str] | None = "f8a3b5c7d9e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_META_KEY = "__forgelm"


def _graph(row: sa.RowMapping) -> dict:
    node_specs = [
        ("data", "数据", "data", {"data_refs": row["data_refs"] or []}),
        (
            "tokenizer",
            "Tokenizer",
            "tokenizer",
            {"tokenizer_version_id": row["tokenizer_version_id"]},
        ),
        ("code", "代码", "custom", {"code_version_id": row["code_version_id"]}),
        (
            "training",
            "训练",
            "training",
            {
                "plan_revision_id": row["plan_revision_id"],
                "training_run_id": row["training_run_id"],
                "execution_owner": "domain_task",
            },
        ),
        ("model", "模型", "custom", {"model_version_id": row["model_version_id"]}),
        (
            "evaluation",
            "评测",
            "evaluation",
            {"evaluation_run_id": row["evaluation_run_id"]},
        ),
        (
            "quantization",
            "量化",
            "quantization",
            {"quantization_run_id": row["quantization_run_id"]},
        ),
        ("release", "发布", "release", {"release_id": row["release_id"]}),
    ]
    nodes = []
    for index, (node_id, label, node_type, references) in enumerate(node_specs):
        config = {"legacy_mapping": True}
        config.update(
            {
                key: value
                for key, value in references.items()
                if value not in (None, "", [])
            }
        )
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "position": {"x": index * 240.0, "y": 180.0},
                "label": label,
                "description": "兼容既有固定产线映射",
                "config": config,
            }
        )
    edges = [
        {
            "id": f"legacy-{node_specs[index][0]}-{node_specs[index + 1][0]}",
            "source": node_specs[index][0],
            "target": node_specs[index + 1][0],
            "source_handle": "out",
            "target_handle": "in",
            "label": "",
        }
        for index in range(len(node_specs) - 1)
    ]
    return {"nodes": nodes, "edges": edges, "variables": {}, "viewport": {}}


def _content_hash(graph: dict, variables: dict) -> str:
    content = {
        "nodes": sorted(graph["nodes"], key=lambda item: item["id"]),
        "edges": sorted(graph["edges"], key=lambda item: item["id"]),
        "variables": variables,
    }
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_status(observed_state: str | None) -> str:
    if observed_state in {"running", "pausing", "paused"}:
        return "running"
    if observed_state == "stopping":
        return "cancelling"
    if observed_state == "succeeded":
        return "succeeded"
    if observed_state in {"failed", "lost"}:
        return "failed"
    if observed_state == "cancelled":
        return "cancelled"
    return "queued"


def _training_node_status(observed_state: str | None) -> str:
    if observed_state in {"queued", "assigned", "starting"}:
        return "queued"
    if observed_state in {"running", "pausing", "stopping"}:
        return "running"
    if observed_state == "paused":
        return "paused"
    if observed_state == "succeeded":
        return "succeeded"
    if observed_state in {"failed", "lost"}:
        return "failed"
    if observed_state == "cancelled":
        return "cancelled"
    return "waiting"


def _unique_name(bind, instances, project_id: str, pipeline_name: str, run_id: str) -> str:
    names = set(
        bind.execute(
            sa.select(instances.c.name).where(instances.c.project_id == project_id)
        ).scalars()
    )
    base = f"{pipeline_name} · 现有运行"
    if base not in names:
        return base
    candidate = f"{base} [{run_id[:8]}]"
    if candidate not in names:
        return candidate
    suffix = 2
    while f"{candidate} #{suffix}" in names:
        suffix += 1
    return f"{candidate} #{suffix}"


def upgrade() -> None:
    bind = op.get_bind()
    metadata = sa.MetaData()
    pipelines = sa.Table("project_pipelines", metadata, autoload_with=bind)
    revisions = sa.Table("project_pipeline_revisions", metadata, autoload_with=bind)
    instances = sa.Table("project_pipeline_instances", metadata, autoload_with=bind)
    pipeline_runs = sa.Table("project_pipeline_runs", metadata, autoload_with=bind)
    node_runs = sa.Table("project_pipeline_node_runs", metadata, autoload_with=bind)
    training_runs = sa.Table("training_runs", metadata, autoload_with=bind)
    tasks = sa.Table("tasks", metadata, autoload_with=bind)

    rows = bind.execute(
        sa.select(
            pipelines.c.id.label("pipeline_id"),
            pipelines.c.project_id,
            pipelines.c.name.label("pipeline_name"),
            revisions.c.id.label("revision_id"),
            revisions.c.revision,
            revisions.c.nodes,
            revisions.c.data_refs,
            revisions.c.tokenizer_version_id,
            revisions.c.code_version_id,
            revisions.c.plan_revision_id,
            revisions.c.training_run_id,
            revisions.c.model_version_id,
            revisions.c.evaluation_run_id,
            revisions.c.quantization_run_id,
            revisions.c.release_id,
            revisions.c.mapping_snapshot,
            training_runs.c.task_id,
            training_runs.c.started_at.label("training_started_at"),
            tasks.c.observed_state,
            tasks.c.progress,
            tasks.c.error.label("task_error"),
            tasks.c.started_at.label("task_started_at"),
            tasks.c.ended_at.label("task_ended_at"),
            tasks.c.created_at.label("task_created_at"),
        )
        .join(revisions, revisions.c.id == pipelines.c.current_revision_id)
        .join(training_runs, training_runs.c.id == revisions.c.training_run_id)
        .join(tasks, tasks.c.id == training_runs.c.task_id)
        .where(
            pipelines.c.archived.is_(False),
            revisions.c.training_run_id.is_not(None),
        )
    ).mappings()
    now = datetime.now(UTC)
    for row in rows:
        if row.get("nodes"):
            continue
        already_imported = False
        existing = bind.execute(
            sa.select(instances.c.variables).where(
                instances.c.pipeline_id == row["pipeline_id"],
                instances.c.definition_revision_id == row["revision_id"],
            )
        ).scalars()
        for variables in existing:
            marker = (variables or {}).get(_META_KEY, {})
            if marker.get("instance_kind") == "imported":
                already_imported = True
                break
        if already_imported:
            continue

        instance_id = str(uuid4())
        pipeline_run_id = str(uuid4())
        graph = _graph(row)
        marker = {
            "instance_kind": "imported",
            "run_kind": "imported",
            "execution_owner": "domain_task",
            "source_pipeline_revision_id": row["revision_id"],
            "source_training_run_id": row["training_run_id"],
            "source_task_id": row["task_id"],
        }
        variables = {_META_KEY: marker}
        instance_name = _unique_name(
            bind,
            instances,
            row["project_id"],
            row["pipeline_name"],
            row["training_run_id"],
        )
        bind.execute(
            instances.insert().values(
                id=instance_id,
                created_at=now,
                updated_at=now,
                project_id=row["project_id"],
                pipeline_id=row["pipeline_id"],
                definition_revision_id=row["revision_id"],
                name=instance_name,
                description="由现有命名产线自动接入；配置和运行事实由原 TrainingRun/Task 管理。",
                status="ready",
                nodes=graph["nodes"],
                edges=graph["edges"],
                variables=variables,
                viewport=graph["viewport"],
                validation_report={
                    "valid": True,
                    "errors": [],
                    "warnings": [
                        {
                            "severity": "warning",
                            "code": "imported_pipeline_read_only",
                            "message": "兼容实例只读展示原任务事实，不能由 DAG 调度器重新执行。",
                        }
                    ],
                    "node_count": len(graph["nodes"]),
                    "edge_count": len(graph["edges"]),
                },
                content_hash=_content_hash(graph, variables),
                activated_at=now,
                locked=True,
            )
        )
        run_status = _run_status(row["observed_state"])
        terminal = run_status in {"succeeded", "failed", "cancelled"}
        bind.execute(
            pipeline_runs.insert().values(
                id=pipeline_run_id,
                created_at=now,
                updated_at=now,
                project_id=row["project_id"],
                pipeline_id=row["pipeline_id"],
                revision_id=row["revision_id"],
                instance_id=instance_id,
                status=run_status,
                trigger_kind="imported",
                input_snapshot={"legacy_mapping": row["mapping_snapshot"] or {}},
                context_snapshot={
                    "pipeline_name": row["pipeline_name"],
                    "pipeline_instance_name": instance_name,
                    "revision": row["revision"],
                    _META_KEY: marker,
                },
                graph_snapshot={**graph, "variables": variables},
                node_summary={},
                error=row["task_error"] or {} if run_status == "failed" else {},
                started_at=row["task_started_at"]
                or row["task_created_at"]
                or row["training_started_at"],
                ended_at=row["task_ended_at"] if terminal else None,
            )
        )

        task_linked = bind.execute(
            sa.select(node_runs.c.id).where(node_runs.c.task_id == row["task_id"])
        ).first()
        downstream_refs = {
            "model": row["model_version_id"],
            "evaluation": row["evaluation_run_id"],
            "quantization": row["quantization_run_id"],
            "release": row["release_id"],
        }
        for node in graph["nodes"]:
            node_id = node["id"]
            if node_id == "training":
                node_status = _training_node_status(row["observed_state"])
            elif node_id == "data":
                node_status = "succeeded" if row["data_refs"] else "waiting"
            elif node_id == "tokenizer":
                node_status = "succeeded" if row["tokenizer_version_id"] else "waiting"
            elif node_id == "code":
                node_status = "succeeded" if row["code_version_id"] else "waiting"
            elif downstream_refs.get(node_id):
                node_status = "succeeded"
            else:
                node_status = "skipped" if terminal else "waiting"
            is_training = node_id == "training"
            bind.execute(
                node_runs.insert().values(
                    id=str(uuid4()),
                    created_at=now,
                    updated_at=now,
                    pipeline_run_id=pipeline_run_id,
                    node_id=node_id,
                    node_type=node["type"],
                    label=node["label"],
                    status=node_status,
                    config_snapshot=node["config"],
                    input_snapshot={"source": "existing_project_pipeline"},
                    output_snapshot={},
                    task_id=row["task_id"]
                    if is_training and task_linked is None
                    else None,
                    branch_decision=None,
                    error=row["task_error"] or {}
                    if is_training and node_status == "failed"
                    else {},
                    started_at=(row["task_started_at"] or row["task_created_at"])
                    if is_training
                    else None,
                    ended_at=row["task_ended_at"]
                    if is_training
                    and node_status in {"succeeded", "failed", "cancelled", "skipped"}
                    else None,
                )
            )


def downgrade() -> None:
    bind = op.get_bind()
    metadata = sa.MetaData()
    instances = sa.Table("project_pipeline_instances", metadata, autoload_with=bind)
    pipeline_runs = sa.Table("project_pipeline_runs", metadata, autoload_with=bind)
    node_runs = sa.Table("project_pipeline_node_runs", metadata, autoload_with=bind)
    imported_ids = []
    for instance_id, variables in bind.execute(
        sa.select(instances.c.id, instances.c.variables)
    ):
        marker = (variables or {}).get(_META_KEY, {})
        if marker.get("instance_kind") == "imported":
            imported_ids.append(instance_id)
    if not imported_ids:
        return
    run_ids = list(
        bind.execute(
            sa.select(pipeline_runs.c.id).where(
                pipeline_runs.c.instance_id.in_(imported_ids)
            )
        ).scalars()
    )
    if run_ids:
        bind.execute(node_runs.delete().where(node_runs.c.pipeline_run_id.in_(run_ids)))
        bind.execute(pipeline_runs.delete().where(pipeline_runs.c.id.in_(run_ids)))
    bind.execute(instances.delete().where(instances.c.id.in_(imported_ids)))
