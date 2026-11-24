from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import ProjectPipelineInstance, ProjectPipelineRun, Task, TrainingRun
from ..schemas import ProjectPipelineInstanceDraftUpdate
from ..services import training_history as history
from ..services import training_instances as service
from ..services.generation_batches import InterleavedGenerationCreate, compile_interleaved
from ..services.pipeline_dags import (
    PipelineDagError,
    cancel_pipeline_run,
    list_pipeline_runs,
    pipeline_run_view,
)
from ..services.transactions import domain_transaction
from .common import add_audit
from .pipelines import _instance_pipeline_run, _user_project

router = APIRouter(tags=["training-instances"])


class InstanceCreate(ProjectPipelineInstanceDraftUpdate):
    model_config = ConfigDict(extra="forbid")
    cloned_from_instance_id: str | None = None

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("实例名称不能为空")
        return value.strip()


class AttemptCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry_from_attempt_id: str | None = None
    selection_mode: Literal["all", "from_node", "selected"] = "all"
    node_ids: list[str] = Field(default_factory=list, max_length=500)
    source_attempt_id: str | None = None


class InstanceArchive(BaseModel):
    archived: bool


def get_instance(session: Session, project_id: str, instance_id: str):
    _user_project(session, project_id)
    instance = session.get(ProjectPipelineInstance, instance_id)
    if instance is None or instance.project_id != project_id:
        service.fail("instance_not_found", "训练实例不存在", 404)
    return instance


@router.get("/projects/{project_id}/training-instances")
def list_instances(
    project_id: str, include_archived: bool = False, session: Session = Depends(get_session)
):
    _user_project(session, project_id)
    query = select(ProjectPipelineInstance).where(ProjectPipelineInstance.project_id == project_id)
    if not include_archived:
        query = query.where(ProjectPipelineInstance.status != "archived")
    return [
        service.instance_view(session, item)
        for item in session.scalars(query.order_by(ProjectPipelineInstance.created_at.desc())).all()
    ]


@router.post("/projects/{project_id}/training-instances/validate")
def validate_instance(
    project_id: str, payload: InstanceCreate, session: Session = Depends(get_session)
):
    _user_project(session, project_id)
    return service.validate_graph(
        session,
        project_id,
        [n.model_dump() for n in payload.nodes],
        [e.model_dump() for e in payload.edges],
    )


@router.post("/projects/{project_id}/training-instances", status_code=201)
def create_instance(
    project_id: str, payload: InstanceCreate, session: Session = Depends(get_session)
):
    _user_project(session, project_id)
    with domain_transaction(session):
        instance = service.create_instance(session, project_id, payload)
        add_audit(
            session,
            project_id=project_id,
            action="training_instance.create",
            subject_type="training_instance",
            subject_id=instance.id,
            payload={
                "content_hash": instance.content_hash,
                "cloned_from_instance_id": payload.cloned_from_instance_id,
            },
        )
    session.commit()
    return service.instance_view(session, instance)


@router.get("/projects/{project_id}/training-instances/{instance_id}")
def instance_detail(project_id: str, instance_id: str, session: Session = Depends(get_session)):
    return service.instance_view(session, get_instance(session, project_id, instance_id))


@router.post("/projects/{project_id}/training-instances/{instance_id}/interleaved-preview")
def preview_interleaved_generation(project_id: str, instance_id: str, payload: InterleavedGenerationCreate,
                                  session: Session = Depends(get_session)):
    graph = compile_interleaved(session, get_instance(session, project_id, instance_id), payload, preview=True)
    return {"graph": graph, "summary": graph["variables"]["generation_interleave"]}


@router.post("/projects/{project_id}/training-instances/{instance_id}/interleaved", status_code=201)
def create_interleaved_generation(project_id: str, instance_id: str, payload: InterleavedGenerationCreate,
                                 session: Session = Depends(get_session)):
    graph = compile_interleaved(session, get_instance(session, project_id, instance_id), payload)
    return create_instance(project_id, InstanceCreate.model_validate(graph), session)


@router.get("/projects/{project_id}/training-instances/resolve-run/{training_run_id}")
def resolve_training_run(
    project_id: str, training_run_id: str, session: Session = Depends(get_session)
):
    _user_project(session, project_id)
    run = session.get(TrainingRun, training_run_id)
    if not run or run.project_id != project_id:
        service.fail("training_run_not_found", "训练记录不存在", 404)
    owner = (
        (run.config_snapshot or {})
        .get("revision", {})
        .get("effective_config", {})
        .get("training_instance_id")
    )
    instance = session.get(ProjectPipelineInstance, owner) if owner else None
    if not instance:
        instance = next(
            (
                item
                for item in session.scalars(
                    select(ProjectPipelineInstance).where(
                        ProjectPipelineInstance.project_id == project_id
                    )
                ).all()
                if history.source_run_id(item) == run.id
            ),
            None,
        )
    if not instance or instance.project_id != project_id:
        service.fail("instance_not_found", "训练实例尚未登记", 404)
    return {"instance_id": instance.id}


@router.get("/projects/{project_id}/training-instances/{instance_id}/execution-graph")
def instance_execution_graph(
    project_id: str,
    instance_id: str,
    attempt_id: str | None = None,
    include_upstream: bool = True,
    max_depth: int | None = Query(default=None, ge=1, le=100),
    center_node_id: str | None = None,
    session: Session = Depends(get_session),
):
    from ..services.training_execution_graph import execution_graph

    instance = get_instance(session, project_id, instance_id)
    view = attempt_detail(project_id, instance_id, attempt_id, session) if attempt_id else None
    if not view:
        attempts = list_attempts(project_id, instance_id, session)
        view = attempts[0] if attempts else None
    return execution_graph(
        session,
        instance,
        view,
        include_upstream=include_upstream,
        max_depth=max_depth,
        center_node_id=center_node_id,
    )


@router.patch("/projects/{project_id}/training-instances/{instance_id}/archive")
def archive_instance(
    project_id: str,
    instance_id: str,
    payload: InstanceArchive,
    session: Session = Depends(get_session),
):
    instance = get_instance(session, project_id, instance_id)
    source = history.history_run(session, instance)
    task = session.get(Task, source.task_id) if source else None
    if task and task.observed_state in {
        "queued",
        "assigned",
        "starting",
        "running",
        "pausing",
        "paused",
        "stopping",
    }:
        service.fail("attempt_active", "实例有活动执行，结束后才能归档")
    session.execute(
        update(ProjectPipelineInstance)
        .where(ProjectPipelineInstance.id == instance.id)
        .values(updated_at=ProjectPipelineInstance.updated_at)
    )
    if session.scalar(
        select(ProjectPipelineRun.id)
        .where(
            ProjectPipelineRun.instance_id == instance.id,
            ProjectPipelineRun.status.in_(service.ACTIVE),
        )
        .limit(1)
    ):
        service.fail("attempt_active", "实例有活动执行，结束后才能归档")
    instance.status = "archived" if payload.archived else "ready" if instance.locked else "draft"
    add_audit(
        session,
        project_id=project_id,
        action="training_instance.archive" if payload.archived else "training_instance.restore",
        subject_type="training_instance",
        subject_id=instance.id,
    )
    session.commit()
    return service.instance_view(session, instance)


@router.get("/projects/{project_id}/training-instances/{instance_id}/attempts")
def list_attempts(project_id: str, instance_id: str, session: Session = Depends(get_session)):
    instance = get_instance(session, project_id, instance_id)
    if history.source_run_id(instance):
        return history.history_attempts(session, instance)
    return list_pipeline_runs(session, project_id, instance_id=instance_id)


@router.post("/projects/{project_id}/training-instances/{instance_id}/attempts", status_code=201)
def start_attempt(
    project_id: str,
    instance_id: str,
    payload: AttemptCreate,
    session: Session = Depends(get_session),
):
    instance = get_instance(session, project_id, instance_id)
    if history.source_run_id(instance):
        service.fail(
            "historical_execution_owner",
            "该历史实例由原训练任务管理；恢复使用原任务协议，修改配置请复制实例。",
        )
    with domain_transaction(session):
        run = service.start_attempt(
            session,
            instance,
            payload.retry_from_attempt_id,
            selection_mode=payload.selection_mode,
            node_ids=payload.node_ids,
            source_attempt_id=payload.source_attempt_id,
        )
        add_audit(
            session,
            project_id=project_id,
            action="training_attempt.retry"
            if payload.retry_from_attempt_id
            else "training_attempt.start",
            subject_type="execution_attempt",
            subject_id=run.id,
            payload={
                "training_instance_id": instance.id,
                "retry_from_attempt_id": payload.retry_from_attempt_id,
                "execution_scope": run.context_snapshot[service.META]["execution_scope"],
            },
        )
    session.commit()
    return pipeline_run_view(session, run)


@router.post("/projects/{project_id}/training-instances/{instance_id}/attempts/preview")
def preview_attempt(
    project_id: str,
    instance_id: str,
    payload: AttemptCreate,
    session: Session = Depends(get_session),
):
    from ..services.training_execution_scope import execution_scope

    instance = get_instance(session, project_id, instance_id)
    if payload.retry_from_attempt_id:
        service.fail(
            "retry_preview_unsupported", "重试自动继承原执行范围；选择范围时请使用历史输出来源", 422
        )
    return execution_scope(
        session,
        instance,
        selection_mode=payload.selection_mode,
        node_ids=payload.node_ids,
        source_attempt_id=payload.source_attempt_id,
    )


@router.get("/projects/{project_id}/training-instances/{instance_id}/attempts/{attempt_id}")
def attempt_detail(
    project_id: str, instance_id: str, attempt_id: str, session: Session = Depends(get_session)
):
    instance = get_instance(session, project_id, instance_id)
    if history.source_run_id(instance):
        result = next(
            (
                item
                for item in history.history_attempts(session, instance)
                if item["id"] == attempt_id
            ),
            None,
        )
        if result is None:
            service.fail("attempt_not_found", "执行记录不存在", 404)
        return result
    return pipeline_run_view(
        session, _instance_pipeline_run(session, project_id, instance_id, attempt_id)
    )


@router.get(
    "/projects/{project_id}/training-instances/{instance_id}/attempts/{attempt_id}/upstream-lineage"
)
def attempt_upstream_lineage(
    project_id: str,
    instance_id: str,
    attempt_id: str,
    max_depth: int = Query(default=4, ge=1, le=12),
    max_nodes: int = Query(default=120, ge=10, le=300),
    session: Session = Depends(get_session),
):
    from ..services.training_upstream_lineage import upstream_lineage

    instance = get_instance(session, project_id, instance_id)
    if history.source_run_id(instance):
        from types import SimpleNamespace

        view = attempt_detail(project_id, instance_id, attempt_id, session)
        run = SimpleNamespace(
            id=view["id"], project_id=project_id, graph_snapshot=view["graph_snapshot"]
        )
        return upstream_lineage(
            session,
            run,
            max_depth=max_depth,
            max_nodes=max_nodes,
            projected_rows=[SimpleNamespace(**row) for row in view["node_runs"]],
        )
    run = _instance_pipeline_run(session, project_id, instance_id, attempt_id)
    return upstream_lineage(session, run, max_depth=max_depth, max_nodes=max_nodes)


@router.post("/projects/{project_id}/training-instances/{instance_id}/attempts/{attempt_id}/cancel")
def cancel_attempt(
    project_id: str, instance_id: str, attempt_id: str,
    finish_running: bool = Query(default=False), session: Session = Depends(get_session)
):
    get_instance(session, project_id, instance_id)
    run = _instance_pipeline_run(session, project_id, instance_id, attempt_id)
    try:
        cancel_pipeline_run(session, run, finish_running=finish_running)
    except PipelineDagError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}
        ) from exc
    add_audit(
        session,
        project_id=project_id,
        action="training_attempt.cancel",
        subject_type="execution_attempt",
        subject_id=run.id,
        payload={"finish_running": finish_running},
    )
    session.commit()
    return pipeline_run_view(session, run)
