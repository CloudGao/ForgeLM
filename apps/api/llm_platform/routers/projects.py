from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    ArtifactLocation,
    Dataset,
    DatasetVersion,
    EvaluationRun,
    ModelVersion,
    Project,
    ProjectDataBinding,
    Task,
    TrainingRun,
)
from ..schemas import ProjectCreate, ProjectUpdate
from ..services.dataset_analyses import effective_train_ready
from ..services.project_lifecycle import LIFECYCLE_STAGES, project_lifecycle_snapshot
from ..services.project_storage import ProjectStorageError, local_workspace_path
from .common import add_audit, get_or_404

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectDeletionExecute(BaseModel):
    confirmation: str = Field(min_length=1, max_length=160)
    fingerprint: str = Field(min_length=64, max_length=64)


@router.get("/{project_id}/deletion-preview")
def preview_project_deletion(project_id: str, session: Session = Depends(get_session)) -> dict:
    from ..services.project_deletion import preview
    return preview(session, _user_project(session, project_id))


@router.post("/{project_id}/delete")
def delete_project(project_id: str, payload: ProjectDeletionExecute, session: Session = Depends(get_session)) -> dict:
    from ..services.project_deletion import execute
    return execute(session, _user_project(session, project_id), payload.confirmation, payload.fingerprint)


def _user_project(session: Session, project_id: str) -> Project:
    project = get_or_404(session, Project, project_id, "项目")
    if project.system_kind:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _validated_workspace(
    session: Session, workspace_uri: str, *, exclude_project_id: str | None = None
) -> str:
    try:
        requested = local_workspace_path(workspace_uri)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_project_workspace", "message": str(exc)},
        ) from exc
    for other in session.scalars(select(Project).where(Project.workspace_uri != "")).all():
        if other.id == exclude_project_id:
            continue
        try:
            other_path = local_workspace_path(other.workspace_uri)
        except ProjectStorageError:
            continue
        if other_path == requested:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "project_workspace_in_use",
                    "message": f"项目工作区已由项目 {other.name} 使用",
                },
            )
    return str(workspace_uri).strip()


def project_summary(session: Session, project: Project) -> dict:
    latest_run = session.scalar(
        select(TrainingRun)
        .where(TrainingRun.project_id == project.id)
        .order_by(TrainingRun.created_at.desc())
        .limit(1)
    )
    latest_eval = session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.project_id == project.id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(1)
    )
    dataset_count = int(
        session.scalar(select(func.count(Dataset.id)).where(Dataset.project_id == project.id)) or 0
    )
    project_versions = session.scalars(
        select(DatasetVersion)
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(Dataset.project_id == project.id)
    ).all()
    ready_count = sum(
        1 for version in project_versions if effective_train_ready(session, version)
    )
    official = (
        session.get(ModelVersion, project.current_model_version_id)
        if project.current_model_version_id
        else None
    )
    return {
        **project.to_dict(),
        "current_model": official.to_dict() if official else None,
        "recent_training_status": latest_run.status if latest_run else "not_started",
        "latest_evaluation": latest_eval.summary if latest_eval else None,
        "data_progress": {
            "dataset_count": dataset_count,
            "ready_versions": ready_count,
            "percent": 0
            if dataset_count == 0
            else min(100, round(ready_count / dataset_count * 100)),
        },
    }


@router.get("")
def list_projects(
    include_archived: bool = Query(False), session: Session = Depends(get_session)
) -> list[dict]:
    query = select(Project).where(Project.system_kind == "").order_by(Project.updated_at.desc())
    if not include_archived:
        query = query.where(Project.archived.is_(False))
    return [project_summary(session, project) for project in session.scalars(query).all()]


@router.post("", status_code=201)
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> dict:
    values = payload.model_dump()
    from uuid import uuid4
    from llm_platform_core.storage_layout import project_directory, unified_storage_root
    from ..config import get_settings
    root = unified_storage_root(get_settings().storage_root)
    if root:
        identity = str(uuid4())
        workspace = root / "projects" / project_directory(identity, values["name"])
        workspace.mkdir(parents=True, exist_ok=True)
        values.update(id=identity, workspace_uri=str(workspace))
    values["workspace_uri"] = _validated_workspace(session, values.get("workspace_uri", ""))
    project = Project(**values)
    session.add(project)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409, detail={"code": "duplicate_project", "message": "项目名称已存在"}
        ) from exc
    add_audit(
        session,
        project_id=project.id,
        action="project.create",
        subject_type="project",
        subject_id=project.id,
    )
    session.commit()
    return project_summary(session, project)


@router.get("/{project_id}")
def get_project(project_id: str, session: Session = Depends(get_session)) -> dict:
    return project_summary(session, _user_project(session, project_id))


@router.patch("/{project_id}")
def update_project(
    project_id: str, payload: ProjectUpdate, session: Session = Depends(get_session)
) -> dict:
    project = _user_project(session, project_id)
    updates = payload.model_dump(exclude_none=True)
    if "workspace_uri" in updates:
        requested_uri = _validated_workspace(
            session, str(updates["workspace_uri"]), exclude_project_id=project.id
        )
        current_path = None
        try:
            current_path = local_workspace_path(project.workspace_uri)
        except ProjectStorageError:
            pass
        requested_path = local_workspace_path(requested_uri)
        if current_path is not None and current_path != requested_path:
            has_tasks = session.scalar(
                select(func.count(Task.id)).where(Task.project_id == project.id)
            )
            has_locations = session.scalar(
                select(func.count(ArtifactLocation.id)).where(
                    ArtifactLocation.project_id == project.id
                )
            )
            if has_tasks or has_locations:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "project_workspace_locked",
                        "message": "项目已有运行或制品，必须通过停服迁移流程更换工作区",
                    },
                )
        updates["workspace_uri"] = requested_uri
    for key, value in updates.items():
        setattr(project, key, value)
    add_audit(
        session,
        project_id=project.id,
        action="project.update",
        subject_type="project",
        subject_id=project.id,
        payload=updates,
    )
    session.commit()
    return project_summary(session, project)


@router.post("/{project_id}/copy", status_code=201)
def copy_project(project_id: str, session: Session = Depends(get_session)) -> dict:
    source = _user_project(session, project_id)
    suffix = datetime.now(UTC).strftime("%H%M%S")
    copied = Project(
        name=f"{source.name}-copy-{suffix}",
        target_model_name=source.target_model_name,
        workspace_uri="",
        project_type=source.project_type,
        target_parameters=source.target_parameters,
        primary_languages=source.primary_languages,
        target_capabilities=source.target_capabilities,
        default_hardware=source.default_hardware,
        training_budget=source.training_budget,
        acceptance_goal=source.acceptance_goal,
    )
    session.add(copied)
    session.flush()
    for binding in session.scalars(
        select(ProjectDataBinding).where(ProjectDataBinding.project_id == source.id)
    ).all():
        session.add(
            ProjectDataBinding(
                project_id=copied.id,
                asset_type=binding.asset_type,
                asset_version_id=binding.asset_version_id,
                purpose=binding.purpose,
                alias=binding.alias,
                metadata_json=dict(binding.metadata_json or {}),
            )
        )
    add_audit(
        session,
        project_id=copied.id,
        action="project.copy",
        subject_type="project",
        subject_id=copied.id,
        payload={"source_project_id": source.id},
    )
    session.commit()
    return project_summary(session, copied)


@router.post("/{project_id}/archive")
def archive_project(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _user_project(session, project_id)
    project.archived = True
    add_audit(
        session,
        project_id=project.id,
        action="project.archive",
        subject_type="project",
        subject_id=project.id,
    )
    session.commit()
    return project.to_dict()


@router.get("/{project_id}/dashboard")
def project_dashboard(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _user_project(session, project_id)
    lifecycle = project_lifecycle_snapshot(session, project_id)
    from ..services.training_metrics import training_run_metrics_view

    if lifecycle["current"].get("training_run"):
        lifecycle["current"]["training_run"] = training_run_metrics_view(
            session, lifecycle["current"]["training_run"]
        )
    active_tasks = session.scalars(
        select(Task)
        .where(
            Task.project_id == project_id,
            Task.observed_state.not_in(["succeeded", "failed", "cancelled", "lost"]),
        )
        .order_by(Task.created_at.desc())
        .limit(5)
    ).all()
    alerts: list[dict] = []
    nodes = {node["key"]: node for node in lifecycle["lifecycle_nodes"]}
    tokenizer_node = nodes["tokenizer"]
    if tokenizer_node["status"] == "waiting" and tokenizer_node.get("version"):
        alerts.append(
            {
                "severity": "warning",
                "title": "当前 Tokenizer 尚未达到锁定可用状态",
                "target": "tokenizers",
            }
        )
    elif tokenizer_node["status"] == "blocked":
        alerts.append(
            {
                "severity": "blocker",
                "title": "当前 Tokenizer 版本未通过质量门禁",
                "target": "tokenizers",
            }
        )
    data_node = nodes["data_preparation"]
    if data_node["status"] == "blocked":
        alerts.append(
            {
                "severity": "blocker",
                "title": "当前绑定数据版本尚不可用于下游",
                "target": "data",
            }
        )
    for key in ("pretraining", "post_training", "evaluation", "quantization"):
        node = nodes[key]
        if node["status"] == "failed":
            alerts.append(
                {
                    "severity": "blocker",
                    "title": f"{node['label']}最近一次真实运行失败",
                    "target": node["target"],
                }
            )
        elif node["status"] == "stopped":
            alerts.append(
                {
                    "severity": "warning",
                    "title": f"{node['label']}最近一次运行已取消",
                    "target": node["target"],
                }
            )
    candidate = lifecycle["current"]["candidate_model"]
    if not active_tasks and not candidate and nodes["pretraining"]["status"] == "not_started":
        alerts.append(
            {"severity": "info", "title": "可以从训练计划启动首次运行", "target": "training"}
        )
    project_view = project_summary(session, project)
    project_view["lifecycle_stage"] = lifecycle["derived_lifecycle_stage"]
    return {
        "project": project_view,
        "lifecycle": [stage for stage, _label, _target in LIFECYCLE_STAGES],
        "derived_lifecycle_stage": lifecycle["derived_lifecycle_stage"],
        "lifecycle_nodes": lifecycle["lifecycle_nodes"],
        "current": lifecycle["current"],
        "active_tasks": [task.to_dict() for task in active_tasks],
        "alerts": alerts,
    }
