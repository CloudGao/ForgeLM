from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.inference import (
    MICROLM_INFERENCE_ADAPTER,
    MICROLM_INFERENCE_FORMATS,
    resolve_inference_profile,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    FeedbackCandidate,
    InferenceRecord,
    ModelVersion,
    PlaygroundSession,
    Project,
    WorkerNode,
    Task,
)
from ..schemas import FeedbackCreate, PlaygroundGenerate
from ..services.artifacts import artifact_storage
from ..services.tasks import create_task, worker_matches_selector
from .common import add_audit, require_executable

router = APIRouter(tags=["playground"])

TRANSFORMERS_ADAPTER = "builtin.transformers-inference"
BUILTIN_MODEL_ADAPTERS = {TRANSFORMERS_ADAPTER, MICROLM_INFERENCE_ADAPTER}


def _validated_inference_config(config: dict) -> dict:
    try:
        result = {
            "temperature": float(config.get("temperature", 0.7)),
            "top_p": float(config.get("top_p", 0.9)),
            "top_k": int(config.get("top_k", 40)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.05)),
            "max_tokens": int(config.get("max_tokens", config.get("max_new_tokens", 512))),
            "device": str(config.get("device", "auto")),
            "dtype": str(config.get("dtype", "auto")),
            "prompt_mode": str(config.get("prompt_mode", "auto")),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_inference_config", "message": "推理参数格式无效"},
        ) from exc
    valid = (
        0 <= result["temperature"] <= 2
        and 0 < result["top_p"] <= 1
        and result["top_k"] >= 0
        and 1 <= result["repetition_penalty"] <= 2
        and 1 <= result["max_tokens"] <= 8192
        and result["device"] in {"auto", "cpu", "cuda"}
        and result["dtype"] in {"auto", "float16", "bfloat16", "float32"}
        and result["prompt_mode"] in {"auto", "completion", "chat"}
    )
    if not valid:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_inference_config", "message": "推理参数超出允许范围"},
        )
    return result


def _model_snapshot(model: ModelVersion) -> dict:
    metadata = model.metadata_json or {}
    inference = metadata.get("inference") if isinstance(metadata.get("inference"), dict) else {}
    pretrained = (
        inference.get("pretrained") or metadata.get("pretrained") or metadata.get("huggingface_id")
    )
    return {
        "id": model.id,
        "project_id": model.project_id,
        "name": model.name,
        "version": model.version,
        "kind": model.kind,
        "artifact_id": model.artifact_id,
        "tokenizer_version_id": model.tokenizer_version_id,
        "metadata": metadata,
        "runtime_source": str(pretrained).strip() if pretrained else None,
        "revision": inference.get("revision") or metadata.get("revision"),
    }


def _inference_profile(model_snapshot: dict) -> dict:
    metadata = model_snapshot.get("metadata") or {}
    return resolve_inference_profile(metadata)


def _merge_resource_request(profile_request: dict, requested: dict) -> dict:
    merged = {**profile_request, **requested}
    for key in ("gpus", "cpus"):
        if key in profile_request:
            merged[key] = max(
                int(profile_request.get(key, 0) or 0),
                int(requested.get(key, 0) or 0),
            )
    profile_selector = profile_request.get("worker_selector") or {}
    requested_selector = requested.get("worker_selector") or {}
    if profile_selector or requested_selector:
        selector = {**profile_selector, **requested_selector}
        labels = {
            **(profile_selector.get("labels") or {}),
            **(requested_selector.get("labels") or {}),
        }
        if labels:
            selector["labels"] = labels
        capabilities = list(
            dict.fromkeys(
                [
                    *(profile_selector.get("capabilities") or []),
                    *(requested_selector.get("capabilities") or []),
                ]
            )
        )
        if capabilities:
            selector["capabilities"] = capabilities
        merged["worker_selector"] = selector
    return merged


def _inference_resource_request(
    session: Session,
    model_snapshot: dict,
    resource_request: dict,
    *,
    adapter_id: str,
    launcher_id: str,
    allow_checkpoint: bool = False,
) -> dict:
    request = {**resource_request}
    artifact_id = model_snapshot.get("artifact_id")
    if model_snapshot.get("runtime_source") or not artifact_id:
        return request
    artifact = session.get(Artifact, str(artifact_id))
    if artifact is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "model_artifact_missing", "message": "模型绑定的制品记录不存在"},
        )
    if artifact.kind == "checkpoint" and not allow_checkpoint:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_artifact_not_inference_ready",
                "message": (
                    "模型版本绑定的是训练 Checkpoint，不是可由 Transformers 直接加载的模型制品；"
                    "请先封装包含 config.json 与 Tokenizer 文件的模型目录，或改用用户命令推理引擎"
                ),
            },
        )
    access = artifact_storage.inference_access(
        session, artifact, str(model_snapshot.get("project_id") or "")
    )
    if access["type"] == "unavailable":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_artifact_unavailable",
                "message": "模型制品没有可下载的托管副本或可用的外部位置",
            },
        )
    if access["type"] != "external" or access["globally_accessible"]:
        return request

    artifact_worker_ids = set(access["worker_ids"])
    selector = dict(request.get("worker_selector") or {})
    selected_worker_id = selector.get("worker_id")
    if selected_worker_id and str(selected_worker_id) not in artifact_worker_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "worker_location_mismatch",
                "message": "所选 Worker 无法访问该模型制品位置",
            },
        )
    if selected_worker_id:
        artifact_worker_ids = {str(selected_worker_id)}
    selected_worker_ids = {str(value) for value in (selector.get("worker_ids") or [])}
    if selected_worker_ids:
        artifact_worker_ids &= selected_worker_ids
    if not artifact_worker_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "worker_location_mismatch",
                "message": "资源选择条件与模型制品所在 Worker 不兼容",
            },
        )
    if len(artifact_worker_ids) == 1:
        selector.pop("worker_ids", None)
        selector["worker_id"] = next(iter(artifact_worker_ids))
    else:
        selector.pop("worker_id", None)
        selector["worker_ids"] = sorted(artifact_worker_ids)
    request["worker_selector"] = selector

    workers = session.scalars(
        select(WorkerNode).where(
            WorkerNode.id.in_(artifact_worker_ids),
            WorkerNode.status == "online",
        )
    ).all()
    compatible = any(
        worker_matches_selector(worker, request)
        and (
            "*" in set(worker.capabilities or [])
            or {adapter_id, launcher_id}.issubset(set(worker.capabilities or []))
        )
        for worker in workers
    )
    if not compatible:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "artifact_worker_unavailable",
                "message": (
                    "模型制品仅能由指定 Worker 访问，但这些 Worker 未在线或未提供 "
                    f"{adapter_id} 推理能力；请启动匹配的项目 Worker，或先创建托管模型副本"
                ),
            },
        )
    return request


@router.get("/projects/{project_id}/playground/sessions")
def list_sessions(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    sessions = session.scalars(
        select(PlaygroundSession)
        .where(PlaygroundSession.project_id == project_id)
        .order_by(PlaygroundSession.updated_at.desc())
    ).all()
    records = session.scalars(select(InferenceRecord).where(
        InferenceRecord.session_id.in_([item.id for item in sessions])).order_by(InferenceRecord.created_at)).all()
    task_rows = session.execute(select(Task.id, Task.active_attempt_id, Task.observed_state,
        Task.current_phase, Task.error).where(Task.id.in_([r.task_id for r in records if r.task_id]))).mappings().all()
    task_by_id = {row["id"]: dict(row) for row in task_rows}
    records_by_session = {}
    for record in records:
        records_by_session.setdefault(record.session_id, []).append({**record.to_dict(), "task": task_by_id.get(record.task_id)})
    result = [{**item.to_dict(), "records": records_by_session.get(item.id, [])} for item in sessions]
    return result


@router.post("/projects/{project_id}/playground/generate", status_code=202)
def generate(
    project_id: str, payload: PlaygroundGenerate, session: Session = Depends(get_session)
) -> dict:
    request_received_at = datetime.now(UTC).isoformat()
    if session.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail={"code": "project_not_found"})
    models = [session.get(ModelVersion, model_id) for model_id in payload.model_version_ids]
    missing = [
        model_id
        for model_id, model in zip(payload.model_version_ids, models, strict=True)
        if model is None
    ]
    if missing:
        raise HTTPException(status_code=404, detail={"code": "models_not_found", "ids": missing})
    if any(model.project_id != project_id for model in models if model is not None):
        raise HTTPException(status_code=404, detail={"code": "models_not_found"})
    model_snapshots = [_model_snapshot(model) for model in models if model is not None]
    profile = _inference_profile(model_snapshots[0])
    profile_adapter = str(profile.get("adapter_id") or "").strip()
    profile_command = profile.get("command") if isinstance(profile.get("command"), dict) else {}
    adapter_id = payload.adapter_id or profile_adapter or TRANSFORMERS_ADAPTER
    uses_profile = bool(profile_adapter and adapter_id == profile_adapter)
    uses_profile_command = bool(uses_profile and profile_command)
    model_format = str(profile.get("format") or "").strip()
    if adapter_id in BUILTIN_MODEL_ADAPTERS and len(models) != 1:
        raise HTTPException(
            status_code=422,
            detail={"code": "single_model_only", "message": "内置推理当前仅支持单模型会话"},
        )
    if adapter_id == TRANSFORMERS_ADAPTER and model_format in MICROLM_INFERENCE_FORMATS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "incompatible_inference_adapter",
                "message": (
                    "该模型是 MicroLM 自研架构，不能由 Transformers 自动加载；"
                    "请使用 MicroLM 原生推理适配器"
                ),
            },
        )
    if adapter_id == MICROLM_INFERENCE_ADAPTER and model_format not in MICROLM_INFERENCE_FORMATS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "incompatible_inference_adapter",
                "message": "MicroLM 原生推理仅接受已登记的 MicroLM 原生或 W8A16 模型制品",
            },
        )
    if adapter_id in BUILTIN_MODEL_ADAPTERS and not (
        model_snapshots[0]["runtime_source"] or model_snapshots[0]["artifact_id"]
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_source_missing",
                "message": "模型版本必须绑定模型制品，或在 metadata.inference.pretrained 中填写模型仓库 ID",
            },
        )
    config = (
        _validated_inference_config(payload.config)
        if adapter_id in BUILTIN_MODEL_ADAPTERS or uses_profile
        else payload.config
    )
    if adapter_id == MICROLM_INFERENCE_ADAPTER:
        config["device"] = "cuda"
    command = {**profile_command} if uses_profile_command else payload.command
    if uses_profile and adapter_id not in BUILTIN_MODEL_ADAPTERS and not command.get(
        "executable"
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": "profile_command_missing", "message": "项目推理配置缺少可执行文件"},
        )
    base_request = _merge_resource_request(
        (profile.get("resource_request") or {}) if uses_profile else {},
        payload.resource_request,
    )
    resource_request = (
        _inference_resource_request(
            session,
            model_snapshots[0],
            base_request,
            adapter_id=adapter_id,
            launcher_id=payload.launcher_id,
            allow_checkpoint=uses_profile_command,
        )
        if adapter_id in BUILTIN_MODEL_ADAPTERS or uses_profile
        else base_request
    )
    selected_worker_id = (resource_request.get("worker_selector") or {}).get("worker_id")
    if selected_worker_id:
        worker = session.get(WorkerNode, str(selected_worker_id))
        heartbeat = worker.last_heartbeat_at if worker else None
        if heartbeat and heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=UTC)
        if (worker is None or worker.status != "online" or not heartbeat
                or datetime.now(UTC) - heartbeat > timedelta(seconds=35)):
            raise HTTPException(409, detail={"code": "worker_unavailable", "message": "指定 Worker 不在线，请检查连接后重试"})
        capabilities = set(worker.capabilities or [])
        if not worker_matches_selector(worker, resource_request) or (
            "*" not in capabilities and not {adapter_id, payload.launcher_id}.issubset(capabilities)
        ):
            raise HTTPException(422, detail={"code": "worker_incompatible", "message": "指定 Worker 不支持当前推理配置"})
    playground = session.get(PlaygroundSession, payload.session_id) if payload.session_id else None
    if playground is not None and playground.project_id != project_id:
        raise HTTPException(status_code=404, detail={"code": "session_not_found"})
    if playground is not None and session.scalar(select(Task.id).join(
            InferenceRecord, InferenceRecord.task_id == Task.id).where(
            InferenceRecord.session_id == playground.id,
            Task.observed_state.notin_(["succeeded", "failed", "cancelled", "lost"])).limit(1)):
        raise HTTPException(409, detail={"code": "session_busy", "message": "请等待本轮完成或停止生成后再发送"})
    if playground is not None and playground.model_version_ids != payload.model_version_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "session_model_mismatch",
                "message": "同一会话不能在中途切换模型，请新建会话",
            },
        )
    if playground is None:
        playground = PlaygroundSession(
            project_id=project_id,
            mode=payload.mode,
            model_version_ids=payload.model_version_ids,
            config={"system_prompt": payload.system_prompt, **config},
        )
        session.add(playground)
        session.flush()
    require_executable(adapter_id, command)
    prior_records = session.scalars(
        select(InferenceRecord)
        .where(InferenceRecord.session_id == playground.id)
        .order_by(InferenceRecord.created_at)
    ).all()
    messages = []
    if payload.system_prompt.strip():
        messages.append({"role": "system", "content": payload.system_prompt})
    for prior in prior_records:
        if prior.output:
            messages.extend(
                [
                    {"role": "user", "content": prior.prompt},
                    {"role": "assistant", "content": prior.output},
                ]
            )
    messages.append({"role": "user", "content": payload.prompt})
    task = create_task(
        session,
        project_id=project_id,
        kind="inference",
        title="Playground 推理",
        adapter_id=adapter_id,
        launcher_id=payload.launcher_id,
        payload={
            "request_received_at": request_received_at,
            "session_id": playground.id,
            "model_version_ids": payload.model_version_ids,
            "model": model_snapshots[0] if len(model_snapshots) == 1 else None,
            "models": model_snapshots,
            "messages": messages,
            "prompt": payload.prompt,
            "system_prompt": payload.system_prompt,
            "config": config,
            "resource_request": resource_request,
            "command": command,
        },
    )
    if selected_worker_id:
        task.current_phase = "waiting_for_worker_capacity"
    record = InferenceRecord(
        session_id=playground.id,
        task_id=task.id,
        prompt=payload.prompt,
        rendered_prompt="",
        stop_reason="pending",
    )
    session.add(record)
    session.flush()
    session.commit()
    return {"session": playground.to_dict(), "record": record.to_dict(), "task": task.to_dict()}


@router.post("/projects/{project_id}/playground/feedback", status_code=201)
def create_feedback(
    project_id: str, payload: FeedbackCreate, session: Session = Depends(get_session)
) -> dict:
    record = session.get(InferenceRecord, payload.inference_record_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail={"code": "inference_not_found", "message": "推理记录不存在"}
        )
    candidate = FeedbackCandidate(project_id=project_id, **payload.model_dump())
    session.add(candidate)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="playground.feedback.create",
        subject_type="feedback_candidate",
        subject_id=candidate.id,
    )
    session.commit()
    return candidate.to_dict()


@router.get("/projects/{project_id}/feedback")
def list_feedback(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        row.to_dict()
        for row in session.scalars(
            select(FeedbackCandidate)
            .where(FeedbackCandidate.project_id == project_id)
            .order_by(FeedbackCandidate.created_at.desc())
        ).all()
    ]
