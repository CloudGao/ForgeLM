"""Follow explicit consumers of a center's outputs, never siblings of its inputs."""

from sqlalchemy import String, cast, or_, select

from .. import models as m
from .global_data import ASSET_TYPES, project_has_asset_access
from .training_history import frozen_training_payload
from .training_upstream_lineage import ASSETS, _domain_source


def downstream_facts(session, project_id, kind, identity):
    from .training_execution_graph import refs

    # The terminal ledger is authoritative for native DAG executions.
    consumers = session.scalars(
        select(m.LineageExecution)
        .join(m.LineageAssetRelation)
        .where(
            m.LineageExecution.project_id == project_id,
            m.LineageAssetRelation.project_id == project_id,
            m.LineageAssetRelation.asset_type == kind,
            m.LineageAssetRelation.asset_id == identity,
            m.LineageAssetRelation.relation_kind == "consumed",
        )
        .order_by(m.LineageExecution.created_at, m.LineageExecution.id)
    ).unique()
    ledger_tasks = set()
    for execution in consumers:
        row = session.get(m.ProjectPipelineNodeRun, execution.node_run_id)
        if row and (row.input_snapshot or {}).get("reused_from_node_run_id"):
            continue
        if execution.task_id:
            ledger_tasks.add(execution.task_id)
        outputs = session.scalars(
            select(m.LineageAssetRelation).where(
                m.LineageAssetRelation.execution_id == execution.id,
                m.LineageAssetRelation.project_id == project_id,
                m.LineageAssetRelation.relation_kind == "produced",
            )
        ).all()
        yield {
            "id": f"execution:{execution.id}",
            "label": row.label if row else execution.node_id,
            "status": execution.status,
            "source": {
                "attempt_id": execution.pipeline_run_id,
                "node_run_id": execution.node_run_id,
                "task_id": execution.task_id,
                "instance_id": (execution.execution_identity or {}).get("training_instance_id"),
            },
            "outputs": [(item.asset_type, item.asset_id) for item in outputs],
        }

    # JSON matching only narrows the query; exact typed references decide edges.
    for run in session.scalars(
        select(m.TrainingRun)
        .where(
            m.TrainingRun.project_id == project_id,
            cast(m.TrainingRun.config_snapshot, String).contains(identity),
        )
        .order_by(m.TrainingRun.created_at, m.TrainingRun.id)
    ):
        task = session.get(m.Task, run.task_id)
        if (
            run.task_id in ledger_tasks
            or not task
            or (kind, identity) not in refs(frozen_training_payload(run, task))
        ):
            continue
        checkpoints = session.scalars(
            select(m.Checkpoint).where(
                m.Checkpoint.training_run_id == run.id,
                m.Checkpoint.project_id == project_id,
            )
        ).all()
        yield {
            "id": f"training:{run.id}",
            "label": task.title,
            "status": run.status,
            "source": {"training_run_id": run.id, "task_id": run.task_id},
            "outputs": [("checkpoint", item.id) for item in checkpoints],
        }

    if kind == "model_version":
        for run in session.scalars(
            select(m.EvaluationRun)
            .where(
                m.EvaluationRun.project_id == project_id,
                cast(m.EvaluationRun.model_version_ids, String).contains(identity),
            )
            .order_by(m.EvaluationRun.created_at, m.EvaluationRun.id)
        ):
            if identity not in run.model_version_ids or run.task_id in ledger_tasks:
                continue
            yield {
                "id": f"evaluation:{run.id}",
                "label": "模型评测",
                "status": run.status,
                "source": {"task_id": run.task_id},
                "outputs": [("evaluation_run", run.id)],
            }

    # Exact version derivations also cover domain records predating the ledger.
    for output_kind, model in ASSETS.items():
        conditions = [
            getattr(model, field) == identity
            for field in (
                "parent_version_id",
                "dataset_version_id",
                "source_checkpoint_id",
                "parent_model_version_id",
                "model_weight_version_id",
                "mixture_version_id",
                "corpus_version_id",
            )
            if hasattr(model, field)
        ]
        conditions += [
            cast(getattr(model, field), String).contains(identity)
            for field in ("entries", "config_snapshot")
            if hasattr(model, field)
        ]
        if not conditions or output_kind == "checkpoint":
            continue
        query = select(model).where(or_(*conditions))
        if output_kind not in ASSET_TYPES and hasattr(model, "project_id"):
            query = query.where(model.project_id == project_id)
        for record in session.scalars(query.order_by(model.created_at, model.id)):
            if output_kind in ASSET_TYPES and not project_has_asset_access(
                session, project_id, output_kind, record.id
            ):
                continue
            fact = _domain_source(session, output_kind, record)
            if (
                fact
                and (kind, identity) in fact["inputs"]
                and fact["source"].get("task_id") not in ledger_tasks
            ):
                yield {**fact, "outputs": [(output_kind, record.id)]}
