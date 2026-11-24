"""Stable Training Instances for legacy training; execution identity stays TaskAttempt."""

from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from .. import models as m
from .pipeline_dags import graph_content_hash

HISTORY = "__training_history"


def source_run_id(instance):
    variables = instance.variables or {}
    return variables.get(HISTORY, {}).get("training_run_id") or variables.get("__forgelm", {}).get(
        "source_training_run_id"
    )


def spec_from_run(run):
    snapshot = run.config_snapshot or {}
    revision = deepcopy(snapshot.get("revision") or {})
    revision["stages"] = deepcopy(snapshot.get("selected_stages") or revision.get("stages") or [])
    revision["engine_config"] = deepcopy(
        revision.get("effective_config", {}).get("engine_config") or {}
    )
    checkpoint = snapshot.get("resume_checkpoint") or {}
    if checkpoint.get("checkpoint_id"):
        revision["input_model_ref"] = {
            **revision.get("input_model_ref", {}),
            "type": "checkpoint_finetune",
            "checkpoint_id": checkpoint["checkpoint_id"],
            "data_cursor_policy": "reset",
        }
    return revision


def ensure_history_instances(session, project_id=None):
    query = select(m.ProjectPipelineInstance)
    if project_id:
        query = query.where(m.ProjectPipelineInstance.project_id == project_id)
    instances = session.scalars(query).all()
    represented = {source_run_id(item) for item in instances}
    query = select(m.TrainingRun)
    if project_id:
        query = query.where(m.TrainingRun.project_id == project_id)
    created = []
    for run in session.scalars(query).all():
        owner = (
            (run.config_snapshot or {})
            .get("revision", {})
            .get("effective_config", {})
            .get("training_instance_id")
        )
        if run.id in represented or owner:
            continue
        instance_id = str(uuid5(NAMESPACE_URL, f"forgelm:training-history:{run.id}"))
        if session.get(m.ProjectPipelineInstance, instance_id):
            continue
        revision = session.get(m.PlanRevision, run.plan_revision_id)
        plan = session.get(m.TrainingPlan, revision.plan_id) if revision else None
        name = (plan.name if plan else run.current_stage) or "模型训练"
        nodes = [
            {
                "id": "training",
                "type": "training",
                "label": run.current_stage or name,
                "description": "",
                "position": {"x": 360, "y": 100},
                "config": {
                    "execution_mode": "native",
                    "training_spec": spec_from_run(run),
                    "training_run_id": run.id,
                },
            }
        ]
        variables = {
            HISTORY: {"training_run_id": run.id, "task_id": run.task_id, "schema_version": 1}
        }
        instance = m.ProjectPipelineInstance(
            id=instance_id,
            project_id=run.project_id,
            name=f"{name[:130]} · {run.id[:12]}",
            description="",
            nodes=nodes,
            edges=[],
            variables=variables,
            viewport={},
            locked=True,
            status="ready",
            content_hash=graph_content_hash(nodes, [], variables),
            created_at=run.created_at,
            updated_at=run.created_at,
        )
        session.add(instance)
        created.append(instance)
        represented.add(run.id)
    session.flush()
    return created


def history_run(session, instance):
    run_id = source_run_id(instance)
    run = session.get(m.TrainingRun, run_id) if run_id else None
    return run if run and run.project_id == instance.project_id else None


def status(value):
    return {
        "lost": "failed",
        "assigned": "queued",
        "starting": "queued",
        "stopping": "cancelling",
        "pausing": "running",
    }.get(value, value)


def frozen_training_payload(run, task):
    """Logical inputs come from the run snapshot, not a subsequently resumed Task."""
    return {
        **deepcopy(run.config_snapshot or {}),
        "training_run_id": run.id,
        "command": deepcopy((task.payload or {}).get("command") or {}),
    }


def history_attempts(session, instance):
    run = history_run(session, instance)
    task = session.get(m.Task, run.task_id) if run else None
    if not task:
        return []
    attempts = session.scalars(
        select(m.TaskAttempt)
        .where(m.TaskAttempt.task_id == task.id)
        .order_by(m.TaskAttempt.attempt_number.desc())
    ).all()
    graph = {
        "nodes": deepcopy(instance.nodes),
        "edges": deepcopy(instance.edges),
        "variables": deepcopy(instance.variables),
        "viewport": deepcopy(instance.viewport),
    }
    # Old named lifecycle mappings are replaced only in this read model, never in storage.
    graph["nodes"] = [
        {
            "id": "training",
            "type": "training",
            "label": run.current_stage or instance.name,
            "description": "",
            "position": {"x": 360, "y": 100},
            "config": {
                "execution_mode": "native",
                "training_spec": spec_from_run(run),
                "training_run_id": run.id,
            },
        }
    ]
    graph["edges"] = []
    result = []
    for attempt in attempts:
        row = {
            "id": f"{attempt.id}:training",
            "pipeline_run_id": attempt.id,
            "node_id": "training",
            "node_type": "training",
            "label": graph["nodes"][0]["label"],
            "status": status(attempt.status),
            "task_id": task.id,
            "task": task.to_dict(),
            "config_snapshot": deepcopy(graph["nodes"][0]["config"]),
            "input_snapshot": {"training_run_id": run.id, "task_attempt_id": attempt.id},
            "output_snapshot": {},
            "error": attempt.error or {},
            "started_at": attempt.started_at.isoformat() if attempt.started_at else None,
            "ended_at": attempt.ended_at.isoformat() if attempt.ended_at else None,
        }
        row["task"]["observed_state"] = status(attempt.status)
        row["task"]["started_at"] = row["started_at"]
        row["task"]["ended_at"] = row["ended_at"]
        row["task"]["result"] = {}
        row["task"]["payload"] = frozen_training_payload(run, task)
        result.append(
            {
                "id": attempt.id,
                "instance_id": instance.id,
                "instance_name": instance.name,
                "project_id": instance.project_id,
                "pipeline_id": None,
                "pipeline_name": "",
                "revision_id": None,
                "revision": None,
                "status": status(attempt.status),
                "run_kind": "imported",
                "execution_owner": "domain_task",
                "trigger_kind": "historical",
                "source_refs": {"training_run_id": run.id, "task_id": task.id},
                "input_snapshot": {},
                "context_snapshot": {
                    "__training_instance": {"attempt_number": attempt.attempt_number}
                },
                "graph_snapshot": deepcopy(graph),
                "node_runs": [row],
                "node_summary": {
                    "total": 1,
                    "completed": int(
                        attempt.status in {"succeeded", "failed", "cancelled", "lost"}
                    ),
                    "progress": 1 if attempt.status == "succeeded" else 0,
                },
                "error": attempt.error or {},
                "created_at": attempt.created_at.isoformat(),
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
            }
        )
    return result


def history_instance_view(session, instance):
    run = history_run(session, instance)
    task = session.get(m.Task, run.task_id) if run else None
    attempts = history_attempts(session, instance)
    graph = (
        attempts[0]["graph_snapshot"]
        if attempts
        else {"nodes": instance.nodes, "edges": instance.edges}
    )
    return {
        **instance.to_dict(),
        **graph,
        "instance_kind": "imported",
        "training_instance": True,
        "execution_owner": "domain_task",
        "source_refs": {"training_run_id": run.id, "task_id": run.task_id} if run else {},
        "can_run": False,
        "can_restore_execution": bool(
            task and task.observed_state in {"failed", "lost", "cancelled", "paused"}
        ),
        "run_count": len(attempts),
        "attempt_count": len(attempts),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "latest_run": {
            k: v for k, v in attempts[0].items() if k not in {"node_runs", "graph_snapshot"}
        }
        if attempts
        else None,
        "metrics_summary": {
            "step": run.current_step,
            "tokens": run.current_tokens,
            "loss": (run.latest_metrics or {}).get(
                "loss", (run.latest_metrics or {}).get("train_loss")
            ),
        }
        if run
        else {},
        "output_summary": {
            "checkpoints": len(
                session.scalars(
                    select(m.Checkpoint.id).where(m.Checkpoint.training_run_id == run.id)
                ).all()
            )
        }
        if run
        else {},
    }
