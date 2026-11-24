from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Query
from llm_platform_core.inference import (
    MICROLM_INFERENCE_ADAPTER,
    resolve_inference_profile,
)
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    Checkpoint,
    ModelVersion,
    ModelWeightAsset,
    ModelWeightVersion,
    Project,
    TokenizerVersion,
    TrainingRun,
    WorkerNode,
)
from ..schemas import ModelLifecycleAction, ModelVersionCreate, ModelVersionRename
from ..services.artifacts import artifact_storage
from .common import add_audit, get_or_404

router = APIRouter(tags=["models"])


def _openai_endpoint_runtime(payload: ModelVersionCreate) -> tuple[dict, dict]:
    assert payload.endpoint_url and payload.endpoint_model and payload.endpoint_mode
    parsed = urlsplit(payload.endpoint_url)
    path = parsed.path.rstrip("/")
    suffix = "/completions" if payload.endpoint_mode == "completions" else "/chat/completions"
    mode_mismatch = (
        path.endswith("/chat/completions")
        if payload.endpoint_mode == "completions"
        else path.endswith("/completions") and not path.endswith("/chat/completions")
    )
    if mode_mismatch:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "endpoint_mode_mismatch",
                "message": "接口地址与所选 OpenAI 协议不一致",
            },
        )
    if not path.endswith(suffix):
        path = f"{path}{suffix}" if path.endswith("/v1") else f"{path}/v1{suffix}"
    endpoint_url = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
    backend = (
        "local-completions"
        if payload.endpoint_mode == "completions"
        else "local-chat-completions"
    )
    model_args = [
        f"model={payload.endpoint_model}",
        f"base_url={endpoint_url}",
        f"num_concurrent={payload.endpoint_num_concurrent}",
        "max_retries=3",
        f"timeout={payload.endpoint_timeout_seconds}",
        f"verify_certificate={str(payload.endpoint_verify_tls).lower()}",
        "tokenized_requests=False",
    ]
    capabilities = ["generate"]
    if payload.endpoint_mode == "completions":
        assert payload.endpoint_tokenizer
        model_args.extend(
            [
                f"tokenizer={payload.endpoint_tokenizer}",
                "tokenizer_backend=huggingface",
            ]
        )
        capabilities.extend(["loglikelihood", "loglikelihood_rolling"])
    endpoint = {
        "protocol": "openai_compatible",
        "url": endpoint_url,
        "model": payload.endpoint_model,
        "mode": payload.endpoint_mode,
        "tokenizer": payload.endpoint_tokenizer,
        "auth_env_var": payload.endpoint_auth_env_var,
        "num_concurrent": payload.endpoint_num_concurrent,
        "timeout_seconds": payload.endpoint_timeout_seconds,
        "verify_tls": payload.endpoint_verify_tls,
        "capabilities": capabilities,
        "reproducibility": "provider_managed",
    }
    runtime = {
        "backend": backend,
        "model_args": ",".join(model_args),
        "auth_env_var": payload.endpoint_auth_env_var,
        "capabilities": capabilities,
        "endpoint_mode": payload.endpoint_mode,
    }
    return endpoint, runtime


def _artifact_summary(artifact: Artifact | None) -> dict | None:
    if artifact is None:
        return None
    return {
        "id": artifact.id,
        "kind": artifact.kind,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
        "media_type": artifact.media_type,
        "availability_state": artifact.availability_state,
        "created_at": artifact.created_at.isoformat(),
    }


def _training_run_summary(run: TrainingRun | None) -> dict | None:
    if run is None:
        return None
    return {
        "id": run.id,
        "status": run.status,
        "current_stage": run.current_stage,
        "current_step": run.current_step,
        "current_tokens": run.current_tokens,
        "latest_metrics": run.latest_metrics,
        "created_at": run.created_at.isoformat(),
    }


def checkpoint_view(session: Session, checkpoint: Checkpoint) -> dict:
    result = checkpoint.to_dict()
    result["artifact"] = _artifact_summary(
        session.get(Artifact, checkpoint.artifact_id) if checkpoint.artifact_id else None
    )
    result["training_run"] = _training_run_summary(
        session.get(TrainingRun, checkpoint.training_run_id)
    )
    return result


def model_view(session: Session, model: ModelVersion) -> dict:
    result = model.to_dict()
    metadata = model.metadata_json or {}
    inference = resolve_inference_profile(metadata)
    pretrained = (
        inference.get("pretrained") or metadata.get("pretrained") or metadata.get("huggingface_id")
    )
    profile_adapter = str(inference.get("adapter_id") or "").strip()
    profile_command = inference.get("command")
    profile_is_command = (
        profile_adapter
        and isinstance(profile_command, dict)
        and profile_command.get("executable")
    )
    profile_is_builtin = profile_adapter == MICROLM_INFERENCE_ADAPTER
    endpoint = metadata.get("openai_endpoint")
    if isinstance(endpoint, dict):
        result["inference_source"] = {
            "type": "api",
            "display_name": "OpenAI 兼容评测端点",
            "endpoint_url": endpoint.get("url"),
            "endpoint_model": endpoint.get("model"),
            "endpoint_mode": endpoint.get("mode"),
            "capabilities": endpoint.get("capabilities") or [],
            "reproducibility": endpoint.get("reproducibility") or "provider_managed",
        }
    elif profile_is_command or profile_is_builtin:
        if model.artifact_id:
            artifact = session.get(Artifact, model.artifact_id)
            access = (
                artifact_storage.inference_access(session, artifact, model.project_id)
                if artifact is not None
                else {"type": "unavailable", "artifact_id": model.artifact_id}
            )
        else:
            access = {"worker_ids": [], "globally_accessible": True}
        result["inference_source"] = {
            **access,
            "type": (
                "unavailable"
                if access.get("type") == "unavailable"
                else "command"
                if profile_is_command
                else "native"
            ),
            "adapter_id": profile_adapter,
            "display_name": str(inference.get("display_name") or "项目原生推理"),
            "format": inference.get("format"),
            "resource_request": inference.get("resource_request") or {},
        }
    elif pretrained:
        result["inference_source"] = {
            "type": "pretrained",
            "value": str(pretrained),
            "worker_ids": [],
            "globally_accessible": True,
        }
    elif model.artifact_id:
        artifact = session.get(Artifact, model.artifact_id)
        if artifact is None:
            result["inference_source"] = {
                "type": "unavailable",
                "artifact_id": model.artifact_id,
            }
        elif artifact.kind == "checkpoint":
            result["inference_source"] = {
                "type": "unsupported",
                "artifact_id": artifact.id,
                "artifact_kind": artifact.kind,
                "reason": "训练 Checkpoint 尚未封装为 Transformers 模型制品",
            }
        else:
            result["inference_source"] = artifact_storage.inference_access(
                session, artifact, model.project_id
            )
    else:
        result["inference_source"] = {"type": "missing"}
    checkpoint = (
        session.get(Checkpoint, model.source_checkpoint_id)
        if model.source_checkpoint_id
        else None
    )
    tokenizer = (
        session.get(TokenizerVersion, model.tokenizer_version_id)
        if model.tokenizer_version_id
        else None
    )
    parent = (
        session.get(ModelVersion, model.parent_model_version_id)
        if model.parent_model_version_id
        else None
    )
    result["artifact"] = _artifact_summary(
        session.get(Artifact, model.artifact_id) if model.artifact_id else None
    )
    weight = (
        session.get(ModelWeightVersion, model.model_weight_version_id)
        if model.model_weight_version_id
        else None
    )
    weight_asset = session.get(ModelWeightAsset, weight.asset_id) if weight else None
    result["model_weight"] = (
        {
            "id": weight.id,
            "version": weight.version,
            "state": weight.state,
            "identity_hash": weight.identity_hash,
            "resolved_commit": weight.resolved_commit,
            "repository_id": weight_asset.repository_id if weight_asset else None,
            "name": weight_asset.name if weight_asset else None,
        }
        if weight
        else None
    )
    result["source_checkpoint"] = checkpoint_view(session, checkpoint) if checkpoint else None
    result["source_training_run"] = _training_run_summary(
        session.get(TrainingRun, checkpoint.training_run_id) if checkpoint else None
    )
    result["tokenizer"] = (
        {
            "id": tokenizer.id,
            "name": tokenizer.name,
            "version": tokenizer.version,
            "algorithm": tokenizer.algorithm,
            "vocab_size": tokenizer.vocab_size,
            "status": tokenizer.status,
            "locked": tokenizer.locked,
        }
        if tokenizer
        else None
    )
    result["parent_model"] = (
        {
            "id": parent.id,
            "name": parent.name,
            "version": parent.version,
            "kind": parent.kind,
            "status": parent.status,
        }
        if parent
        else None
    )
    result["child_model_count"] = int(
        session.scalar(
            select(func.count(ModelVersion.id)).where(
                ModelVersion.parent_model_version_id == model.id
            )
        )
        or 0
    )
    if model.status == "deleted":
        result["inference_source"] = {"type": "unavailable", "reason": "模型版本已删除"}
    return result


@router.get("/projects/{project_id}/models")
def list_models(
    project_id: str,
    include_archived: bool = Query(False),
    include_deleted: bool = Query(False),
    session: Session = Depends(get_session),
) -> dict:
    model_query = select(ModelVersion).where(ModelVersion.project_id == project_id)
    if not include_deleted:
        model_query = model_query.where(ModelVersion.status != "deleted")
    if not include_archived:
        model_query = model_query.where(ModelVersion.status != "archived")
    models = session.scalars(model_query.order_by(ModelVersion.created_at.desc())).all()
    checkpoints = session.scalars(
        select(Checkpoint)
        .where(Checkpoint.project_id == project_id)
        .order_by(Checkpoint.created_at.desc())
    ).all()
    nodes = [model_view(session, model) for model in models]
    edges = [
        {"source": model.parent_model_version_id, "target": model.id, "kind": "derived_from"}
        for model in models
        if model.parent_model_version_id
    ]
    return {
        "models": nodes,
        "checkpoints": [checkpoint_view(session, checkpoint) for checkpoint in checkpoints],
        "lineage": {"nodes": nodes, "edges": edges},
    }


@router.post("/projects/{project_id}/models", status_code=201)
def create_model(
    project_id: str,
    payload: ModelVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    version = payload.version
    if version is None:
        version = int(
            session.scalar(
                select(func.max(ModelVersion.version)).where(
                    ModelVersion.project_id == project_id,
                    ModelVersion.name == payload.name,
                )
            )
            or 0
        ) + 1
    existing = session.scalar(
        select(ModelVersion).where(
            ModelVersion.project_id == project_id,
            ModelVersion.name == payload.name,
            ModelVersion.version == version,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_version_exists",
                "message": f"模型 {payload.name} V{version} 已存在",
            },
        )

    parent = None
    if payload.parent_model_version_id:
        parent = get_or_404(
            session, ModelVersion, payload.parent_model_version_id, "父模型版本"
        )
        if parent.project_id != project_id:
            raise HTTPException(status_code=404, detail={"code": "parent_model_not_found"})

    tokenizer = None
    if payload.tokenizer_version_id:
        tokenizer = get_or_404(
            session, TokenizerVersion, payload.tokenizer_version_id, "Tokenizer 版本"
        )
        if tokenizer.project_id != project_id:
            raise HTTPException(status_code=404, detail={"code": "tokenizer_not_found"})

    checkpoint = None
    artifact = None
    source_metadata: dict = {"type": payload.source_type}
    if payload.source_type == "checkpoint":
        checkpoint = get_or_404(
            session, Checkpoint, str(payload.source_checkpoint_id), "Checkpoint"
        )
        if checkpoint.project_id != project_id:
            raise HTTPException(status_code=404, detail={"code": "checkpoint_not_found"})
        if checkpoint.status != "available":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_unavailable",
                    "message": "只有可用 Checkpoint 才能登记为模型版本",
                },
            )
        if not checkpoint.artifact_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_artifact_missing",
                    "message": "该 Checkpoint 尚未登记权重制品",
                },
            )
        artifact = session.get(Artifact, checkpoint.artifact_id)
        source_metadata["checkpoint_id"] = checkpoint.id
    elif payload.source_type == "huggingface":
        source_metadata.update(
            {
                "repository_id": payload.huggingface_id,
                "revision": payload.huggingface_revision,
            }
        )
    elif payload.source_type == "openai_compatible":
        endpoint, runtime = _openai_endpoint_runtime(payload)
        source_metadata["endpoint"] = endpoint
    else:
        artifact_payload = payload.artifact
        assert artifact_payload is not None
        if artifact_payload.worker_id and session.get(WorkerNode, artifact_payload.worker_id) is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "worker_not_found", "message": "制品绑定的 Worker 不存在"},
            )
        try:
            artifact, location = artifact_storage.register_external(
                session,
                project_id=project_id,
                kind=artifact_payload.kind,
                uri=artifact_payload.uri,
                content_hash=artifact_payload.content_hash,
                size_bytes=artifact_payload.size_bytes,
                media_type=artifact_payload.media_type,
                availability_state=artifact_payload.availability_state,
                worker_id=artifact_payload.worker_id,
                metadata={**artifact_payload.metadata, "project_id": project_id},
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_external_artifact", "message": str(exc)},
            ) from exc
        source_metadata.update({"artifact_id": artifact.id, "location_id": location.id})

    metadata = dict(payload.metadata)
    metadata["registry_source"] = source_metadata
    if payload.source_type == "huggingface":
        inference = metadata.get("inference")
        metadata["inference"] = {
            **(inference if isinstance(inference, dict) else {}),
            "pretrained": payload.huggingface_id,
            "revision": payload.huggingface_revision,
        }
    elif payload.source_type == "openai_compatible":
        metadata["openai_endpoint"] = endpoint
        metadata["lm_eval"] = runtime
    model = ModelVersion(
        project_id=project_id,
        name=payload.name,
        version=version,
        kind=payload.kind,
        parent_model_version_id=parent.id if parent else None,
        source_checkpoint_id=checkpoint.id if checkpoint else None,
        artifact_id=artifact.id if artifact else None,
        tokenizer_version_id=tokenizer.id if tokenizer else None,
        parameter_count=payload.parameter_count,
        status="candidate" if payload.is_candidate else "registered",
        is_candidate=payload.is_candidate,
        metadata_json=metadata,
    )
    session.add(model)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="model.register",
        subject_type="model_version",
        subject_id=model.id,
        payload={
            "name": model.name,
            "version": model.version,
            "source_type": payload.source_type,
            "artifact_id": model.artifact_id,
        },
    )
    session.commit()
    return model_view(session, model)


@router.get("/models/{model_id}")
def get_model(model_id: str, session: Session = Depends(get_session)) -> dict:
    from ..services.model_deletion import model_record
    return model_view(session, model_record(session, model_id))


@router.patch("/models/{model_id}/name")
def rename_model(
    model_id: str,
    payload: ModelVersionRename,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.status == "deleted":
        raise HTTPException(
            status_code=410,
            detail={"code": "model_deleted", "message": "已删除模型不能重命名"},
        )
    duplicate = session.scalar(
        select(ModelVersion.id).where(
            ModelVersion.project_id == model.project_id,
            ModelVersion.name == payload.name,
            ModelVersion.version == model.version,
            ModelVersion.id != model.id,
        )
    )
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_version_exists",
                "message": f"模型 {payload.name} V{model.version} 已存在",
            },
        )
    previous_name = model.name
    model.name = payload.name
    add_audit(
        session,
        project_id=model.project_id,
        action="model.rename",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason,
        payload={"previous_name": previous_name, "name": model.name, "version": model.version},
    )
    session.commit()
    return model_view(session, model)


class ModelDeletionPreview(BaseModel):
    mode: Literal["normal", "cascade"] = "normal"


class ModelDeletionExecute(BaseModel):
    confirmation: str


@router.post("/models/{model_id}/deletion-plans", status_code=201)
def preview_model_deletion(model_id: str, payload: ModelDeletionPreview,
                           session: Session = Depends(get_session)):
    from ..services.model_deletion import model_record, preview
    return preview(session, model_record(session, model_id), payload.mode)


@router.get("/models/{model_id}/deletion-plans/{plan_id}")
def model_deletion_plan(model_id: str, plan_id: str, session: Session = Depends(get_session)):
    from ..services.model_deletion import get_plan, plan_view
    return plan_view(session, get_plan(session, model_id, plan_id))


@router.post("/models/{model_id}/deletion-plans/{plan_id}/execute", status_code=202)
def execute_model_deletion(model_id: str, plan_id: str, payload: ModelDeletionExecute,
                           session: Session = Depends(get_session)):
    from ..services.model_deletion import execute, get_plan, model_record
    return execute(session, model_record(session, model_id), get_plan(session, model_id, plan_id),
                   payload.confirmation)


@router.post("/models/{model_id}/mark-candidate")
def mark_candidate(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.status == "archived":
        raise HTTPException(
            status_code=409,
            detail={"code": "archived_model", "message": "请先恢复已归档模型"},
        )
    if model.is_official:
        raise HTTPException(
            status_code=409,
            detail={"code": "official_model", "message": "当前正式模型不能同时标记为候选"},
        )
    model.is_candidate = True
    model.status = "candidate"
    add_audit(
        session,
        project_id=model.project_id,
        action="model.mark_candidate",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)


@router.post("/models/{model_id}/mark-best")
def mark_best(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.status == "archived":
        raise HTTPException(
            status_code=409,
            detail={"code": "archived_model", "message": "已归档模型不能标记为最佳"},
        )
    for previous in session.scalars(
        select(ModelVersion).where(
            ModelVersion.project_id == model.project_id,
            ModelVersion.is_best.is_(True),
        )
    ).all():
        previous.is_best = False
    model.is_best = True
    if not model.is_official:
        model.is_candidate = True
        model.status = "candidate"
    add_audit(
        session,
        project_id=model.project_id,
        action="model.mark_best",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)


@router.post("/models/{model_id}/unmark-best")
def unmark_best(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if not model.is_best:
        raise HTTPException(
            status_code=409,
            detail={"code": "model_not_best", "message": "该模型当前未标记为最佳"},
        )
    model.is_best = False
    add_audit(
        session,
        project_id=model.project_id,
        action="model.unmark_best",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)


@router.post("/models/{model_id}/promote")
def promote_model(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.status == "archived":
        raise HTTPException(
            status_code=409, detail={"code": "archived_model", "message": "已归档模型不能晋升"}
        )
    project = get_or_404(session, Project, model.project_id, "项目")
    for previous in session.scalars(
        select(ModelVersion).where(
            ModelVersion.project_id == model.project_id,
            ModelVersion.is_official.is_(True),
            ModelVersion.id != model.id,
        )
    ).all():
        previous.is_official = False
        previous.status = "registered"
    model.is_official = True
    model.is_candidate = False
    model.status = "official"
    project.current_model_version_id = model.id
    add_audit(
        session,
        project_id=model.project_id,
        action="model.promote",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)


@router.post("/models/{model_id}/archive")
def archive_model(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.is_official:
        raise HTTPException(
            status_code=409, detail={"code": "official_model", "message": "当前正式模型不能归档"}
        )
    model.status = "archived"
    model.is_candidate = False
    model.is_best = False
    add_audit(
        session,
        project_id=model.project_id,
        action="model.archive",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)


@router.post("/models/{model_id}/restore")
def restore_model(
    model_id: str,
    payload: ModelLifecycleAction | None = None,
    session: Session = Depends(get_session),
) -> dict:
    model = get_or_404(session, ModelVersion, model_id, "模型版本")
    if model.status != "archived":
        raise HTTPException(
            status_code=409,
            detail={"code": "model_not_archived", "message": "只有已归档模型可以恢复"},
        )
    model.status = "registered"
    add_audit(
        session,
        project_id=model.project_id,
        action="model.restore",
        subject_type="model_version",
        subject_id=model.id,
        reason=payload.reason if payload else None,
    )
    session.commit()
    return model_view(session, model)
