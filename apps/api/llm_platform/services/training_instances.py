"""Immutable training definitions and recoverable DAG execution attempts.

The existing pipeline tables are storage, not a requirement for a definition/template.
Domain Task/TrainingRun records remain the execution adapters for individual nodes.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import models as m
from ..schemas import TrainingPlanCreate, TrainingStart
from .global_data import project_has_asset_access
from .lineage import _database_asset_ref
from .transactions import domain_transaction

META = "__training_instance"
ACTIVE = {"queued", "running", "cancelling"}
TERMINAL = {"succeeded", "failed", "cancelled"}
logger = logging.getLogger(__name__)


def fail(code: str, message: str, status: int = 409):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def managed(instance: m.ProjectPipelineInstance) -> bool:
    return bool((instance.variables or {}).get(META))


def native(node: dict) -> bool:
    return (
        node.get("type") == "training"
        and (node.get("config") or {}).get("execution_mode") == "native"
    )


def spec_payload(node: dict) -> TrainingPlanCreate:
    spec = deepcopy((node.get("config") or {}).get("training_spec") or {})
    spec["name"] = node["label"]
    return TrainingPlanCreate.model_validate(spec)


def validate_graph(session: Session, project_id: str, nodes: list, edges: list) -> dict:
    from .pipeline_dags import validate_pipeline_graph

    validation_nodes = deepcopy(nodes)
    for node in validation_nodes:
        if native(node):
            node["config"] = {"command": {"executable": "__worker_python__", "args": []}}
    report = validate_pipeline_graph(
        session, project_id, validation_nodes, edges, execution_bindings=True
    )
    for node in nodes:
        config = node.get("config") or {}
        if config.get("code_version_id"):
            try:
                from ..routers.training import _get_project_code_version

                _get_project_code_version(session, project_id, config["code_version_id"])
            except HTTPException as exc:
                report["errors"].append(
                    {
                        "severity": "error",
                        "code": "invalid_code_version",
                        "node_id": node["id"],
                        "message": str(exc.detail),
                    }
                )
        if not native(node):
            continue
        try:
            if config.get("input_bindings"):
                from .native_pipeline_nodes import validate_node
                report["errors"].extend(validate_node(node, edges))
                if not (config.get("training_spec") or {}).get("stages"):
                    raise ValueError("至少配置一个训练阶段")
                continue
            payload = spec_payload(node)
            if not payload.stages:
                raise ValueError("至少配置一个训练阶段")
            from ..routers.training import _plan_model_binding, _validate_validation_reference

            _plan_model_binding(
                session,
                project_id,
                payload.model_specification_revision_id,
                payload.code_version_id,
                payload.input_model_ref,
            )
            _validate_validation_reference(
                session,
                project_id,
                payload.validation_set_id,
                payload.validation_mixture_version_id,
            )
            if not payload.code_version_id and not (
                payload.engine_config.get("executable") and payload.engine_config.get("args")
            ):
                raise ValueError("请选择训练代码版本或填写完整训练命令")
            checkpoint_id = payload.input_model_ref.get("checkpoint_id")
            if checkpoint_id:
                checkpoint = session.get(m.Checkpoint, checkpoint_id)
                if (
                    checkpoint is None
                    or checkpoint.project_id != project_id
                    or checkpoint.status != "available"
                    or not checkpoint.artifact_id
                ):
                    raise ValueError("初始 Checkpoint 不存在、不可用或不属于当前项目")
            if payload.tokenizer_version_id:
                tokenizer = session.get(m.TokenizerVersion, payload.tokenizer_version_id)
                if tokenizer is None or tokenizer.project_id != project_id:
                    raise ValueError("Tokenizer 版本不存在或不属于当前项目")
            for stage in payload.stages:
                template_id = stage.get("chat_template_version_id")
                if template_id:
                    template = session.get(m.ChatTemplateVersion, template_id)
                    owner = session.get(m.ChatTemplate, template.template_id) if template else None
                    if owner is None or owner.project_id != project_id:
                        raise ValueError("Chat Template 版本不存在或不属于当前项目")
                preparation = stage.get("data_preparation") or {}
                post_training = stage.get("type") in {"sft", "dpo"}
                mixture_id = (
                    preparation.get("mixture_version_id")
                    or stage.get("mixture_version_id")
                    or (stage.get("data_ref_id") if not post_training else None)
                )
                objective = stage.get("objective") or {}
                supervision_id = (
                    objective.get("mixture_version_id")
                    or stage.get("supervision_mixture_version_id")
                    or (stage.get("data_ref_id") if post_training else None)
                )
                for kind, record_id in (
                    ("mixture_version", mixture_id),
                    ("supervision_mixture_version", supervision_id),
                ):
                    if record_id and not project_has_asset_access(
                        session, project_id, kind, record_id
                    ):
                        raise ValueError("训练数据版本不存在或未绑定到当前项目")
        except (ValidationError, ValueError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            report["errors"].append(
                {
                    "severity": "error",
                    "code": "invalid_training_node",
                    "node_id": node["id"],
                    "message": detail.get("message", str(detail))
                    if isinstance(detail, dict)
                    else str(detail),
                }
            )
    report["valid"] = not report["errors"]
    return report


def create_instance(session: Session, project_id: str, payload) -> m.ProjectPipelineInstance:
    from .pipeline_dags import graph_content_hash

    nodes = [node.model_dump() for node in payload.nodes]
    edges = [edge.model_dump() for edge in payload.edges]
    for node in nodes:
        node["config"].pop("plan_revision_id", None)
        node["config"].pop("_native_plan_id", None)
        node["config"].pop("code_version", None)
    report = validate_graph(session, project_id, nodes, edges)
    if not report["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "training_instance_invalid",
                "message": "请修正节点配置后创建实例",
                **report,
            },
        )
    if session.scalar(
        select(m.ProjectPipelineInstance.id).where(
            m.ProjectPipelineInstance.project_id == project_id,
            m.ProjectPipelineInstance.name == payload.name.strip(),
        )
    ):
        fail("instance_name_exists", "该训练实例名称已存在，请使用新名称")
    if payload.cloned_from_instance_id:
        source = session.get(m.ProjectPipelineInstance, payload.cloned_from_instance_id)
        if source is None or source.project_id != project_id:
            fail("clone_source_missing", "来源实例不存在", 404)
    variables = deepcopy(payload.variables)
    variables.pop("__forgelm", None)
    variables[META] = {
        "schema_version": 1,
        "cloned_from_instance_id": payload.cloned_from_instance_id,
    }
    instance = m.ProjectPipelineInstance(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description,
        nodes=deepcopy(nodes),
        edges=edges,
        variables=variables,
        viewport=payload.viewport,
        status="ready",
        locked=True,
        validation_report=report,
        activated_at=m.utcnow(),
    )
    session.add(instance)
    session.flush()
    from ..routers.training import create_training_plan

    with domain_transaction(session):
        for node in nodes:
            code_id = node["config"].get("code_version_id")
            if code_id:
                version = session.get(m.CodeVersion, code_id)
                artifact = (
                    session.get(m.Artifact, version.artifact_id) if version.artifact_id else None
                )
                if version.state != "ready" or artifact is None:
                    fail("code_version_unavailable", "代码版本尚未准备完成", 422)
                node["config"]["code_version"] = {
                    "version_id": version.id,
                    "version": version.version,
                    "artifact_id": artifact.id,
                    "content_hash": artifact.content_hash,
                    "git_commit": version.git_commit,
                    "entrypoints": deepcopy(version.entrypoints or {}),
                    "dependency_spec": deepcopy(version.dependency_spec),
                }
            if not native(node) or node["config"].get("input_bindings"):
                continue
            plan = create_training_plan(project_id, spec_payload(node), session)
            revision = session.get(m.PlanRevision, plan["current_revision_id"])
            revision.locked = True
            raw_input = spec_payload(node).input_model_ref
            if raw_input.get("type") == "checkpoint_finetune":
                revision.input_model_ref = {**revision.input_model_ref, **raw_input}
            if revision.model_specification_revision_id:
                model_revision = session.get(
                    m.ModelSpecificationRevision, revision.model_specification_revision_id
                )
                if model_revision:
                    model_revision.locked = True
            revision.effective_config = {
                **revision.effective_config,
                "training_instance_id": instance.id,
            }
            node["config"] = {
                **node["config"],
                "plan_revision_id": revision.id,
                "_native_plan_id": plan["id"],
            }
    instance.nodes = deepcopy(nodes)
    instance.content_hash = graph_content_hash(nodes, edges, variables)
    return instance


def _acquire_instance(session: Session, instance: m.ProjectPipelineInstance) -> None:
    # A write before checking active runs serializes concurrent requests on SQLite.
    session.execute(
        update(m.ProjectPipelineInstance)
        .where(m.ProjectPipelineInstance.id == instance.id)
        .values(updated_at=m.ProjectPipelineInstance.updated_at)
    )
    session.refresh(instance)
    if instance.status != "ready" or not instance.locked:
        fail("instance_not_ready", "该实例已归档或尚未冻结，请复制后创建新实例")
    if session.scalar(
        select(m.ProjectPipelineRun.id)
        .where(
            m.ProjectPipelineRun.instance_id == instance.id, m.ProjectPipelineRun.status.in_(ACTIVE)
        )
        .limit(1)
    ):
        fail("attempt_active", "该实例已有活动执行，请先等待结束或停止")


def start_attempt(
    session: Session,
    instance: m.ProjectPipelineInstance,
    retry_from: str | None = None,
    *,
    selection_mode: str = "all",
    node_ids: list[str] | None = None,
    source_attempt_id: str | None = None,
) -> m.ProjectPipelineRun:
    from .pipeline_dags import advance_pipeline_run, is_imported_pipeline_instance
    from .training_execution_scope import execution_scope

    from .training_history import source_run_id

    if is_imported_pipeline_instance(instance) or source_run_id(instance):
        fail("historical_instance", "该历史记录由原训练任务管理，请从原训练详情继续")
    _acquire_instance(session, instance)
    if retry_from and (selection_mode != "all" or node_ids or source_attempt_id):
        fail("ambiguous_attempt_request", "重试继承原执行范围，不能同时覆盖范围或输入来源", 422)
    source_id = retry_from or source_attempt_id
    source = session.get(m.ProjectPipelineRun, source_id) if source_id else None
    if retry_from and (
        source is None
        or source.instance_id != instance.id
        or source.status not in {"failed", "cancelled"}
    ):
        fail("attempt_not_retryable", "只能重试同一实例已失败或已停止的执行")
    latest = session.scalar(
        select(m.ProjectPipelineRun)
        .where(m.ProjectPipelineRun.instance_id == instance.id)
        .order_by(m.ProjectPipelineRun.created_at.desc())
        .limit(1)
    )
    if retry_from and source and latest and source.id != latest.id:
        fail("stale_retry", "请从最近一次执行重试，避免恢复旧进度")
    if retry_from:
        inherited = (source.context_snapshot or {}).get(META, {}).get("execution_scope") or {}
        selection_mode = inherited.get("selection_mode", "all")
        node_ids = inherited.get("node_ids", [])
        source_attempt_id = source.id if selection_mode != "all" else None
    scope = execution_scope(
        session,
        instance,
        selection_mode=selection_mode,
        node_ids=node_ids,
        source_attempt_id=source_attempt_id,
    )
    if not scope["valid"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "execution_scope_invalid",
                "message": "执行范围或上游依赖不可用",
                **scope,
            },
        )
    selected_ids = set(scope["selected_node_ids"])
    reused_ids = set(scope["reused_node_ids"])
    excluded_ids = set(scope["excluded_node_ids"])
    report = validate_graph(session, instance.project_id, instance.nodes, instance.edges)
    report["errors"] = [
        error
        for error in report["errors"]
        if not error.get("node_id") or error["node_id"] in selected_ids
    ]
    report["valid"] = not report["errors"]
    if not report["valid"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "instance_inputs_unavailable",
                "message": "范围内的冻结输入已不可用，请恢复资产或复制配置",
                **report,
            },
        )
    old_nodes = (
        {
            n.node_id: n
            for n in session.scalars(
                select(m.ProjectPipelineNodeRun).where(
                    m.ProjectPipelineNodeRun.pipeline_run_id == source.id
                )
            ).all()
        }
        if source
        else {}
    )
    number = (
        len(
            session.scalars(
                select(m.ProjectPipelineRun.id).where(
                    m.ProjectPipelineRun.instance_id == instance.id
                )
            ).all()
        )
        + 1
    )
    if retry_from:
        from .lineage import record_terminal_run_lineage

        record_terminal_run_lineage(session, source, list(old_nodes.values()), finalize_lost=True)
    attempt = m.ProjectPipelineRun(
        project_id=instance.project_id,
        instance_id=instance.id,
        pipeline_id=instance.pipeline_id,
        revision_id=instance.definition_revision_id,
        status="running",
        trigger_kind="retry" if retry_from else "partial" if selection_mode != "all" else "manual",
        input_snapshot=deepcopy(instance.variables or {}),
        context_snapshot={
            META: {
                "attempt_number": number,
                "retry_from_attempt_id": retry_from,
                "execution_scope": {
                    key: scope[key]
                    for key in (
                        "selection_mode",
                        "node_ids",
                        "source_attempt_id",
                        "selected_node_ids",
                        "reused_node_ids",
                        "excluded_node_ids",
                    )
                },
            },
            "pipeline_instance_content_hash": instance.content_hash,
        },
        graph_snapshot={
            "nodes": deepcopy(instance.nodes),
            "edges": deepcopy(instance.edges),
            "variables": deepcopy(instance.variables),
            "viewport": deepcopy(instance.viewport),
        },
        started_at=m.utcnow(),
    )
    session.add(attempt)
    session.flush()
    pending_retries = []
    for node in instance.nodes:
        previous = old_nodes.get(node["id"]) if retry_from or node["id"] in reused_ids else None
        row = m.ProjectPipelineNodeRun(
            pipeline_run_id=attempt.id,
            node_id=node["id"],
            node_type=node["type"],
            label=node["label"],
            config_snapshot=deepcopy(node.get("config") or {}),
            status="waiting",
        )
        session.add(row)
        if node["id"] in excluded_ids:
            row.status = "skipped"
            row.input_snapshot = {"excluded_from_execution": True}
            row.ended_at = m.utcnow()
        elif previous and previous.status in {"succeeded", "skipped"}:
            row.status = previous.status
            row.output_snapshot = deepcopy(previous.output_snapshot)
            row.branch_decision = previous.branch_decision
            row.input_snapshot = {
                "reused_from_node_run_id": previous.id,
                "reused_from_attempt_id": previous.pipeline_run_id,
                "reused_outputs": deepcopy(previous.output_snapshot),
            }
            row.ended_at = m.utcnow()
        elif previous and previous.task_id:
            task = session.get(m.Task, previous.task_id)
            if task and task.observed_state in {"failed", "lost", "paused", "cancelled"}:
                if task.observed_state == "cancelled" and task.kind != "train" and not (node.get("config") or {}).get("native_operation"):
                    continue  # A new command task starts from its frozen definition.
                row.task_id = task.id
                if (node.get("config") or {}).get("native_operation"):
                    row.output_snapshot = deepcopy(previous.output_snapshot)
                    row.output_snapshot["produced_assets"] = [ref for ref in row.output_snapshot.get("produced_assets", []) if ref.get("asset_type") != "artifact"]
                row.status = "queued"
                row.input_snapshot = {
                    **deepcopy(previous.input_snapshot),
                    "resumed_from_node_run_id": previous.id,
                    "resume_task_attempt_id": task.active_attempt_id,
                }
                row.input_snapshot.pop("task_snapshot", None)
                row.input_snapshot.pop("task_attempt_id", None)
                pending_retries.append(task)
        elif previous and (previous.input_snapshot or {}).get("preparation"):
            row.input_snapshot = {
                **deepcopy(previous.input_snapshot),
                "preparation": deepcopy(previous.input_snapshot["preparation"]),
                "resumed_from_node_run_id": previous.id,
            }
            for task_id in preparation_task_ids(previous.input_snapshot["preparation"]):
                task = session.get(m.Task, task_id)
                if (
                    task
                    and task.observed_state in {"failed", "lost", "paused"}
                    and task not in pending_retries
                ):
                    pending_retries.append(task)
    session.flush()
    from ..routers.tasks import control_task
    from ..schemas import TaskControl

    with domain_transaction(session):
        session.info["training_attempt_control"] = True
        try:
            for task in pending_retries:
                control_task(
                    task.id,
                    TaskControl(
                        action="resume"
                        if task.observed_state in {"paused", "cancelled"}
                        else "retry",
                        reason="从训练实例恢复执行",
                    ),
                    session,
                )
        finally:
            session.info.pop("training_attempt_control", None)
        advance_pipeline_run(session, attempt)
    return attempt


def advance_native_node(
    session: Session,
    run: m.ProjectPipelineRun,
    node: dict,
    row: m.ProjectPipelineNodeRun,
    context: dict,
) -> bool:
    """Return false while preparation is pending, without re-entering the DAG loop."""
    from ..routers.training import prepare_training_data, start_training_run

    config = node["config"]
    if config.get("input_bindings"):
        from .native_pipeline_nodes import resolve_parameters
        from ..routers.training import create_training_plan
        saved = (row.input_snapshot or {}).get("bound_training")
        if saved is None:
            resolved = resolve_parameters(config, context, key="training_spec")
            resolved_node = {**node, "config": {"execution_mode": "native", "training_spec": resolved}}
            report = validate_graph(session, run.project_id, [resolved_node], [])
            if not report["valid"]:
                raise ValueError(str(report["errors"]))
            with domain_transaction(session):
                plan = create_training_plan(run.project_id, spec_payload(resolved_node), session)
                revision = session.get(m.PlanRevision, plan["current_revision_id"])
                revision.locked = True
                revision.effective_config = {**revision.effective_config, "training_instance_id": run.instance_id}
                raw_input = resolved.get("input_model_ref") or {}
                if raw_input.get("type") == "checkpoint_finetune":
                    revision.input_model_ref = {**revision.input_model_ref, **raw_input}
                if revision.model_specification_revision_id:
                    session.get(m.ModelSpecificationRevision, revision.model_specification_revision_id).locked = True
                saved = {"plan_revision_id": revision.id, "_native_plan_id": plan["id"], "training_spec": resolved}
                row.input_snapshot = {**row.input_snapshot, "bound_training": saved}
        config = {**config, **saved}
    request = TrainingStart(
        revision_id=config["plan_revision_id"],
        selected_stage_indexes=config.get("selected_stage_indexes"),
        resume_checkpoint_id=(config.get("training_spec", {}).get("input_model_ref") or {}).get(
            "checkpoint_id"
        ),
    )
    old_preparation = (row.input_snapshot or {}).get("preparation") or {}
    task_ids = preparation_task_ids(old_preparation)
    failed = [
        task
        for key in task_ids
        if (task := session.get(m.Task, key))
        and task.observed_state in {"failed", "lost", "cancelled"}
    ]
    if failed:
        row.status = "failed"
        row.error = {
            "code": "training_data_preparation_failed",
            "message": "训练数据准备失败；可在任务中心修复准备任务后重试",
            "task_ids": [t.id for t in failed],
        }
        row.ended_at = m.utcnow()
        return True
    with domain_transaction(session):
        preparation = prepare_training_data(config["_native_plan_id"], request, session)
        row.input_snapshot = {**row.input_snapshot, **context, "preparation": preparation}
        if not preparation.get("ready"):
            return False
        result = start_training_run(config["_native_plan_id"], request, session)
        task = session.get(m.Task, result["task_id"])
        task.payload = {
            **task.payload,
            "pipeline_instance_id": run.instance_id,
            "pipeline_run_id": run.id,
            "pipeline_node_run_id": row.id,
            "pipeline_node_id": row.node_id,
        }
        row.task_id = task.id
        row.status = "queued"
        row.input_snapshot = {
            **row.input_snapshot,
            "training_run_id": result["id"],
            "training_inputs": deepcopy(result["config_snapshot"]),
            "consumed_assets": training_input_refs(session, task.payload),
        }
    return True


def preparation_task_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        return ({str(value["task_id"])} if value.get("task_id") else set()).union(
            *(preparation_task_ids(v) for v in value.values())
        )
    if isinstance(value, list):
        return set().union(*(preparation_task_ids(v) for v in value))
    return set()


def training_input_refs(session: Session, payload: dict) -> list[dict]:
    result = []
    for key, kind, id_key in (
        ("code", "code_version", "version_id"),
        ("code_version", "code_version", "version_id"),
        ("tokenizer", "tokenizer_version", "version_id"),
        ("model_specification", "model_specification_revision", "revision_id"),
        ("resume_checkpoint", "checkpoint", "id"),
        ("reference_checkpoint", "checkpoint", "id"),
    ):
        value = payload.get(key) or {}
        record_id = value.get(id_key) or value.get("checkpoint_id")
        if record_id:
            result.append(_database_asset_ref(session, kind, record_id, role=key))
    model_code = (payload.get("model_specification") or {}).get("code_version_id")
    if model_code:
        result.append(
            _database_asset_ref(session, "code_version", model_code, role="model_implementation")
        )
    for key, kind in (
        ("data_inputs", "mixture_version"),
        ("training_data_view_inputs", "training_data_view_version"),
        ("supervision_training_view_inputs", "supervision_training_view_version"),
        ("tokenized_data_inputs", "tokenized_dataset_version"),
    ):
        for value in payload.get(key) or []:
            record_id = (
                value.get("mixture_version_id") if key == "data_inputs" else value.get("version_id")
            )
            if record_id:
                ref = _database_asset_ref(session, kind, record_id, role=key)
                ref["metadata"] = {
                    **ref.get("metadata", {}),
                    **{
                        k: value[k]
                        for k in ("artifact_id", "content_hash", "identity_hash", "manifest_hash")
                        if value.get(k)
                    },
                }
                result.append(ref)
            if value.get("mixture_version_id") and key != "data_inputs":
                result.append(
                    _database_asset_ref(
                        session,
                        "mixture_version",
                        value["mixture_version_id"],
                        role="source_mixture",
                    )
                )
    validation = payload.get("validation_data_input") or {}
    if validation.get("mixture_version_id"):
        result.append(
            _database_asset_ref(
                session, "mixture_version", validation["mixture_version_id"], role="validation"
            )
        )
    for key in ("validation_input", "validation_data_input", "validation_set"):
        value = payload.get(key) or {}
        if value.get("artifact_id"):
            result.append(
                {
                    "asset_type": "artifact",
                    "asset_id": value["artifact_id"],
                    "label": "训练验证集",
                    "version": value.get("content_hash"),
                    "role": "validation",
                }
            )
    return result


def node_output(session: Session, task: m.Task, node_run: m.ProjectPipelineNodeRun) -> dict:
    base = deepcopy(node_run.output_snapshot or {}) if (node_run.config_snapshot or {}).get("native_operation") else {}
    result = {**base, **deepcopy(task.result or {})}
    # A node may safely pause/resume across TaskAttempts. Attribute only artifacts
    # generated after this graph attempt began, including partial failed outputs.
    artifacts = session.scalars(
        select(m.Artifact)
        .join(m.TaskAttempt, m.Artifact.created_by_task_attempt_id == m.TaskAttempt.id)
        .where(m.TaskAttempt.task_id == task.id, m.Artifact.created_at >= node_run.created_at)
    ).all()
    artifact_refs = [
        {
            "asset_type": "artifact",
            "asset_id": a.id,
            "label": a.kind,
            "version": a.content_hash,
            "role": "output",
            "metadata": {"content_hash": a.content_hash, "size_bytes": a.size_bytes},
        }
        for a in artifacts
    ]
    if artifact_refs:
        result["produced_assets"] = [
            *(result.get("produced_assets") or result.get("versions") or []),
            *artifact_refs,
        ]
    run_id = (task.payload or {}).get("training_run_id")
    if task.kind == "train" and run_id:
        checkpoints = session.scalars(
            select(m.Checkpoint).where(
                m.Checkpoint.training_run_id == run_id,
                m.Checkpoint.artifact_id.in_([a.id for a in artifacts]),
            )
        ).all()
        ids = [c.id for c in checkpoints]
        models = (
            session.scalars(
                select(m.ModelVersion).where(m.ModelVersion.source_checkpoint_id.in_(ids))
            ).all()
            if ids
            else []
        )
        refs = [
            _database_asset_ref(session, "checkpoint", c.id, role="checkpoint") for c in checkpoints
        ]
        refs += [_database_asset_ref(session, "model_version", v.id, role="model") for v in models]
        result["produced_assets"] = [*refs, *artifact_refs]
        result["versions"] = [{"kind": r["asset_type"], "id": r["asset_id"], **r} for r in refs]
        result["training_run_id"] = run_id
        result["model_version_ids"] = [v.id for v in models]
        result["checkpoint_ids"] = ids
        result["execution_identity"] = {
            "training_run_id": run_id,
            "task_attempt_id": task.active_attempt_id,
        }
    return result


def reconcile_training_attempts(session: Session) -> None:
    """Durable progress: also called after API restart by the control-plane reaper."""
    from .pipeline_dags import advance_pipeline_run

    runs = session.scalars(
        select(m.ProjectPipelineRun).where(m.ProjectPipelineRun.status.in_({"queued", "running"}))
    ).all()
    for run in runs:
        if not (run.context_snapshot or {}).get(META):
            continue
        run_id = run.id
        try:
            with session.begin_nested(), domain_transaction(session):
                advance_pipeline_run(session, run)
        except Exception:
            # One invalid or unavailable dependency must not stop lease recovery
            # or prevent unrelated instances from making progress.
            logger.exception("Training attempt reconciliation failed: %s", run_id)


def instance_view(session: Session, instance: m.ProjectPipelineInstance) -> dict:
    from .pipeline_dags import pipeline_instance_view
    from .training_history import history_instance_view, source_run_id

    if source_run_id(instance):
        return history_instance_view(session, instance)

    view = pipeline_instance_view(session, instance)
    view["cloned_from_instance_id"] = (
        (instance.variables or {}).get(META, {}).get("cloned_from_instance_id")
    )
    view["attempt_count"] = view["run_count"]
    view["can_run"] = view["can_run"] and (view.get("latest_run") or {}).get("status") not in ACTIVE
    view["training_instance"] = managed(instance)
    return view
