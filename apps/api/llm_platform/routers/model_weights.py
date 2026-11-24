from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ArtifactLocation,
    ModelVersion,
    ModelWeightAsset,
    ModelWeightVersion,
    Project,
    ProjectPipelineInstance,
    Task,
    TaskAttempt,
)
from ..services.global_data import ensure_global_data_project
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(prefix="/model-weights", tags=["model-weights"])
TERMINAL_TASK_STATES = {"succeeded", "failed", "cancelled", "lost"}


class WeightScanRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=500)
    revision: str = Field(default="main", min_length=1, max_length=180)
    hf_endpoint: str | None = Field(default=None, max_length=2048)
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})

    @field_validator("repo_id", "revision", "hf_endpoint", mode="before")
    @classmethod
    def strip_strings(cls, value):
        return value.strip() if isinstance(value, str) else value


class WeightPlanRequest(BaseModel):
    scan_task_id: str = Field(min_length=36, max_length=36)
    selected_paths: list[str] = Field(min_length=1)


class WeightCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    plan_task_id: str = Field(min_length=36, max_length=36)
    description: str = Field(default="", max_length=4000)
    license: str = Field(default="unknown", max_length=120)
    tags: list[str] = Field(default_factory=list)
    parameter_count: int = Field(default=0, ge=0)
    quantization: str | None = Field(default=None, max_length=80)


class WeightUseRequest(BaseModel):
    project_id: str = Field(min_length=36, max_length=36)
    name: str | None = Field(default=None, max_length=180)
    version: int | None = Field(default=None, ge=1)
    kind: str = Field(default="base", min_length=1, max_length=40)
    is_candidate: bool = True


def _with_capability(resource_request: dict[str, Any]) -> dict[str, Any]:
    request = dict(resource_request)
    selector = dict(request.get("worker_selector") or {})
    capabilities = [str(value) for value in selector.get("capabilities") or []]
    selector["capabilities"] = list(
        dict.fromkeys([*capabilities, "builtin.model-weight", "builtin.hf-asset"])
    )
    request["worker_selector"] = selector
    request.setdefault("gpus", 0)
    return request


def _latest_attempt(session: Session, task_id: str) -> TaskAttempt | None:
    return session.scalar(
        select(TaskAttempt)
        .where(TaskAttempt.task_id == task_id)
        .order_by(TaskAttempt.attempt_number.desc())
    )


def _successful_task(session: Session, task_id: str, kind: str) -> Task:
    task = get_or_404(session, Task, task_id, "权重任务")
    if task.kind != kind or task.observed_state != "succeeded" or not isinstance(task.result, dict):
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_task_not_ready", "message": "权重任务阶段尚未成功完成"},
        )
    return task


def _identity(source: dict[str, Any]) -> str:
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _location_view(location: ArtifactLocation) -> dict[str, Any]:
    return location.to_dict()


def weight_version_view(session: Session, version: ModelWeightVersion) -> dict[str, Any]:
    asset = session.get(ModelWeightAsset, version.asset_id)
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    locations = (
        session.scalars(
            select(ArtifactLocation)
            .where(ArtifactLocation.artifact_id == version.artifact_id)
            .order_by(ArtifactLocation.created_at)
        ).all()
        if version.artifact_id
        else []
    )
    return {
        **version.to_dict(),
        "asset": asset.to_dict() if asset else None,
        "artifact": artifact.to_dict() if artifact else None,
        "locations": [_location_view(item) for item in locations],
        "project_model_count": int(
            session.scalar(
                select(func.count(ModelVersion.id)).where(
                    ModelVersion.model_weight_version_id == version.id
                )
            )
            or 0
        ),
    }


@router.get("")
def list_model_weights(session: Session = Depends(get_session)) -> dict[str, Any]:
    versions = session.scalars(
        select(ModelWeightVersion).order_by(ModelWeightVersion.created_at.desc())
    ).all()
    return {"items": [weight_version_view(session, item) for item in versions]}


@router.get("/{version_id}")
def get_model_weight(version_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    return weight_version_view(
        session, get_or_404(session, ModelWeightVersion, version_id, "权重版本")
    )


@router.post("/huggingface/scan", status_code=202)
def scan_model_repository(
    payload: WeightScanRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    if "/" not in payload.repo_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_repo_id", "message": "Repo ID 必须使用 org/model 格式"},
        )
    project = ensure_global_data_project(session)
    config: dict[str, Any] = {
        "revision": payload.revision,
        "repo_type": "model",
    }
    if payload.hf_endpoint:
        config["hf_endpoint"] = payload.hf_endpoint
    task = create_task(
        session,
        project_id=project.id,
        kind="model_weight_scan",
        title=f"解析模型仓库：{payload.repo_id}",
        adapter_id="builtin.model-weight",
        payload={
            "operation": "hf_scan",
            "source_uri": payload.repo_id,
            "source_config": config,
            "resource_request": _with_capability(payload.resource_request),
        },
    )
    session.commit()
    return task.to_dict()


@router.post("/huggingface/plans", status_code=202)
def plan_model_download(
    payload: WeightPlanRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    project = ensure_global_data_project(session)
    scan_task = _successful_task(session, payload.scan_task_id, "model_weight_scan")
    if scan_task.project_id != project.id:
        raise HTTPException(status_code=404, detail={"code": "weight_scan_not_found"})
    scan = scan_task.result or {}
    repository_paths = {
        str(item.get("path"))
        for item in scan.get("files") or []
        if isinstance(item, dict) and item.get("path")
    }
    selected_paths = list(dict.fromkeys(str(value).strip() for value in payload.selected_paths))
    if not selected_paths or any(path not in repository_paths for path in selected_paths):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_weight_selection", "message": "所选文件不属于已冻结仓库版本"},
        )
    frozen_identity = {
        "provider": "huggingface",
        "repository_id": scan.get("repo_id"),
        "resolved_commit": scan.get("commit_hash"),
        "endpoint": scan.get("endpoint"),
        "selected_paths": sorted(selected_paths),
    }
    identity_hash = _identity(frozen_identity)
    scan_attempt = _latest_attempt(session, scan_task.id)
    if scan_attempt is None or not scan_attempt.worker_id:
        raise HTTPException(status_code=409, detail={"code": "weight_scan_worker_missing"})
    source_config = {
        "repo_type": "model",
        "repo_id": scan.get("repo_id"),
        "requested_revision": scan.get("requested_revision"),
        "commit_hash": scan.get("commit_hash"),
        "selected_paths": selected_paths,
        "hf_endpoint": scan.get("endpoint"),
    }
    task = create_task(
        session,
        project_id=project.id,
        kind="model_weight_plan",
        title=f"预检模型权重：{scan.get('repo_id')}",
        adapter_id="builtin.model-weight",
        payload={
            "operation": "hf_plan",
            "identity_hash": identity_hash,
            "source_uri": scan.get("repo_id"),
            "source_config": source_config,
            "project_workspace_uri": project.workspace_uri,
            "scan_task_id": scan_task.id,
            "resource_request": _with_capability(
                {"gpus": 0, "worker_selector": {"worker_id": scan_attempt.worker_id}}
            ),
        },
    )
    session.commit()
    return task.to_dict()


@router.post("", status_code=202)
def create_model_weight(
    payload: WeightCreateRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    project = ensure_global_data_project(session)
    plan_task = _successful_task(session, payload.plan_task_id, "model_weight_plan")
    if plan_task.project_id != project.id:
        raise HTTPException(status_code=404, detail={"code": "weight_plan_not_found"})
    plan = plan_task.result or {}
    if not plan.get("disk_space_sufficient"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "weight_disk_space_insufficient",
                "message": "目标 Worker 磁盘空间不足",
            },
        )
    identity_hash = str(plan.get("identity_hash") or "")
    existing = session.scalar(
        select(ModelWeightVersion).where(ModelWeightVersion.identity_hash == identity_hash)
    )
    if existing is not None:
        existing_task = session.get(Task, existing.task_id) if existing.task_id else None
        return {
            "weight": weight_version_view(session, existing),
            "task": existing_task.to_dict() if existing_task else None,
            "reused": True,
        }
    repo_id = str(plan.get("repo_id") or "")
    asset = session.scalar(
        select(ModelWeightAsset).where(
            ModelWeightAsset.source_provider == "huggingface",
            ModelWeightAsset.repository_id == repo_id,
        )
    )
    if asset is None:
        asset = ModelWeightAsset(
            name=payload.name,
            description=payload.description,
            source_provider="huggingface",
            repository_id=repo_id,
            license=payload.license,
            tags=payload.tags,
        )
        session.add(asset)
        session.flush()
    version_number = (
        int(
            session.scalar(
                select(func.max(ModelWeightVersion.version)).where(
                    ModelWeightVersion.asset_id == asset.id
                )
            )
            or 0
        )
        + 1
    )
    version = ModelWeightVersion(
        asset_id=asset.id,
        version=version_number,
        identity_hash=identity_hash,
        requested_revision=str(plan.get("requested_revision") or "main"),
        resolved_commit=str(plan.get("commit_hash") or ""),
        source_endpoint=str(plan.get("endpoint") or ""),
        selected_paths=list(plan.get("selected_paths") or []),
        source_manifest={
            "provider": "huggingface",
            "repository_id": repo_id,
            "plan_task_id": plan_task.id,
            "worker_id": plan.get("worker_id"),
            "planned_file_count": plan.get("file_count"),
            "planned_selected_bytes": plan.get("selected_bytes"),
            "planned_cached_bytes": plan.get("cached_bytes"),
            "planned_download_bytes": plan.get("download_bytes"),
        },
        parameter_count=payload.parameter_count,
        quantization=payload.quantization,
        state="queued",
    )
    session.add(version)
    session.flush()
    plan_attempt = _latest_attempt(session, plan_task.id)
    if plan_attempt is None or not plan_attempt.worker_id:
        raise HTTPException(status_code=409, detail={"code": "weight_plan_worker_missing"})
    source_config = {
        "repo_type": "model",
        "repo_id": repo_id,
        "requested_revision": version.requested_revision,
        "revision": version.resolved_commit,
        "commit_hash": version.resolved_commit,
        "selected_paths": version.selected_paths,
        "allow_patterns": version.selected_paths,
        "hf_endpoint": version.source_endpoint,
        "planned_file_count": plan.get("file_count"),
        "planned_selected_bytes": plan.get("selected_bytes"),
        "planned_cached_bytes": plan.get("cached_bytes"),
        "planned_download_bytes": plan.get("download_bytes"),
    }
    task = create_task(
        session,
        project_id=project.id,
        kind="model_weight_download",
        title=f"下载模型权重：{asset.name} V{version.version}",
        adapter_id="builtin.model-weight",
        payload={
            "operation": "hf_download",
            "model_weight_version_id": version.id,
            "identity_hash": version.identity_hash,
            "source_uri": repo_id,
            "source_config": source_config,
            "project_workspace_uri": project.workspace_uri,
            "resource_request": _with_capability(
                {"gpus": 0, "worker_selector": {"worker_id": plan_attempt.worker_id}}
            ),
        },
    )
    version.task_id = task.id
    add_audit(
        session,
        project_id=None,
        action="model_weight.create",
        subject_type="model_weight_version",
        subject_id=version.id,
        payload={"repository_id": repo_id, "identity_hash": identity_hash},
    )
    session.commit()
    return {"weight": weight_version_view(session, version), "task": task.to_dict()}


@router.post("/{version_id}/project-models", status_code=201)
def use_model_weight(
    version_id: str, payload: WeightUseRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    version = get_or_404(session, ModelWeightVersion, version_id, "权重版本")
    asset = get_or_404(session, ModelWeightAsset, version.asset_id, "权重")
    project = get_or_404(session, Project, payload.project_id, "项目")
    if project.system_kind:
        raise HTTPException(status_code=422, detail={"code": "system_project_not_allowed"})
    if version.state != "ready" or not version.artifact_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_version_not_ready", "message": "只有可用权重才能加入项目"},
        )
    name = (payload.name or asset.name).strip()
    model_version = (
        payload.version
        or int(
            session.scalar(
                select(func.max(ModelVersion.version)).where(
                    ModelVersion.project_id == project.id, ModelVersion.name == name
                )
            )
            or 0
        )
        + 1
    )
    if session.scalar(
        select(ModelVersion.id).where(
            ModelVersion.project_id == project.id,
            ModelVersion.name == name,
            ModelVersion.version == model_version,
        )
    ):
        raise HTTPException(status_code=409, detail={"code": "model_version_exists"})
    model = ModelVersion(
        project_id=project.id,
        name=name,
        version=model_version,
        kind=payload.kind,
        artifact_id=version.artifact_id,
        model_weight_version_id=version.id,
        parameter_count=version.parameter_count,
        status="candidate" if payload.is_candidate else "registered",
        is_candidate=payload.is_candidate,
        metadata_json={
            "registry_source": {"type": "model_weight", "model_weight_version_id": version.id},
            "inference": {
                "adapter_id": "builtin.transformers-inference",
                "format": "transformers",
                "display_name": "全局权重库",
            },
            "lm_eval": {
                "backend": "hf",
                "model_args": "pretrained=__FORGELM_MODEL_WEIGHT__",
                "capabilities": version.runtime_capabilities,
            },
        },
    )
    session.add(model)
    session.flush()
    add_audit(
        session,
        project_id=project.id,
        action="model.register_weight",
        subject_type="model_version",
        subject_id=model.id,
        payload={"model_weight_version_id": version.id},
    )
    session.commit()
    return model.to_dict()


@router.post("/{version_id}/download", status_code=202)
def redownload_model_weight(
    version_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    version = get_or_404(session, ModelWeightVersion, version_id, "权重版本")
    asset = get_or_404(session, ModelWeightAsset, version.asset_id, "权重")
    if version.artifact_id and session.scalar(
        select(ArtifactLocation.id).where(
            ArtifactLocation.artifact_id == version.artifact_id,
            ArtifactLocation.availability_state.in_({"available", "deleting"}),
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_location_exists", "message": "该权重已有可用或待清理副本"},
        )
    source = version.source_manifest or {}
    worker_id = str(source.get("worker_id") or "")
    if not worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_download_worker_missing", "message": "原下载 Worker 信息缺失"},
        )
    project = ensure_global_data_project(session)
    source_config = {
        "repo_type": "model",
        "repo_id": asset.repository_id,
        "requested_revision": version.requested_revision,
        "revision": version.resolved_commit,
        "commit_hash": version.resolved_commit,
        "selected_paths": version.selected_paths,
        "allow_patterns": version.selected_paths,
        "hf_endpoint": version.source_endpoint,
        "planned_file_count": source.get("planned_file_count"),
        "planned_selected_bytes": source.get("planned_selected_bytes"),
        "planned_cached_bytes": source.get("planned_cached_bytes"),
        "planned_download_bytes": source.get("planned_download_bytes"),
    }
    task = create_task(
        session,
        project_id=project.id,
        kind="model_weight_download",
        title=f"重新下载模型权重：{asset.name} V{version.version}",
        adapter_id="builtin.model-weight",
        payload={
            "operation": "hf_download",
            "model_weight_version_id": version.id,
            "identity_hash": version.identity_hash,
            "source_uri": asset.repository_id,
            "source_config": source_config,
            "project_workspace_uri": project.workspace_uri,
            "resource_request": _with_capability(
                {"gpus": 0, "worker_selector": {"worker_id": worker_id}}
            ),
        },
    )
    version.task_id = task.id
    version.state = "queued"
    version.failure_reason = None
    session.commit()
    return task.to_dict()


@router.delete("/{version_id}/locations/{location_id}", status_code=202)
def evict_model_weight_location(
    version_id: str, location_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    version = get_or_404(session, ModelWeightVersion, version_id, "权重版本")
    location = get_or_404(session, ArtifactLocation, location_id, "权重位置")
    if not version.artifact_id or location.artifact_id != version.artifact_id:
        raise HTTPException(status_code=404, detail={"code": "weight_location_not_found"})
    if not location.worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "managed_location_not_evictable", "message": "该位置不属于具体 Worker"},
        )
    active = next(
        (
            task.id
            for task in session.scalars(
                select(Task).where(Task.observed_state.not_in(TERMINAL_TASK_STATES))
            ).all()
            if version.id in json.dumps(task.payload or {}, ensure_ascii=False)
        ),
        None,
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_location_in_use", "message": "仍有任务正在使用该权重版本"},
        )
    project = ensure_global_data_project(session)
    task = create_task(
        session,
        project_id=project.id,
        kind="model_weight_evict",
        title=f"清理权重副本：{version.identity_hash[:12]}",
        adapter_id="builtin.model-weight",
        payload={
            "operation": "evict",
            "model_weight_version_id": version.id,
            "location_id": location.id,
            "location_uri": location.uri,
            "content_hash": session.get(Artifact, version.artifact_id).content_hash,
            "identity_hash": version.identity_hash,
            "project_workspace_uri": project.workspace_uri,
            "resource_request": _with_capability(
                {"gpus": 0, "worker_selector": {"worker_id": location.worker_id}}
            ),
        },
    )
    location.availability_state = "deleting"
    session.commit()
    return task.to_dict()


@router.delete("/{version_id}", status_code=204)
def delete_model_weight_version(version_id: str, session: Session = Depends(get_session)) -> None:
    version = get_or_404(session, ModelWeightVersion, version_id, "权重版本")
    blockers: list[str] = []
    if session.scalar(
        select(ModelVersion.id).where(ModelVersion.model_weight_version_id == version.id)
    ):
        blockers.append("仍有项目模型引用该版本")
    if version.artifact_id and session.scalar(
        select(ArtifactLocation.id).where(ArtifactLocation.artifact_id == version.artifact_id)
    ):
        blockers.append("仍有 Worker 保存该版本的物理副本")
    for task in session.scalars(
        select(Task).where((Task.observed_state.not_in(TERMINAL_TASK_STATES)) | (Task.kind == "data_generate"))
    ).all():
        if version.id in json.dumps(task.payload or {}, ensure_ascii=False):
            blockers.append("仍有任务或生成血缘引用该版本")
            break
    for instance in session.scalars(select(ProjectPipelineInstance)):
        if any((node.get("config") or {}).get("parameters", {}).get("model_weight_version_id") == version.id for node in instance.nodes or []):
            blockers.append("仍有不可变产线实例引用该版本")
            break
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={"code": "weight_delete_blocked", "message": "；".join(blockers)},
        )
    asset = session.get(ModelWeightAsset, version.asset_id)
    session.delete(version)
    session.flush()
    if asset and not session.scalar(
        select(ModelWeightVersion.id).where(ModelWeightVersion.asset_id == asset.id)
    ):
        session.delete(asset)
    session.commit()
