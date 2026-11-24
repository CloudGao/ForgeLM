from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    GateResult,
    ModelVersion,
    PluginVersion,
    Project,
    ProjectPluginBinding,
    QuantizationRun,
    Release,
    ReleaseBuildRun,
    ReleaseDeliveryJob,
    ReleaseDeliveryRun,
    ReleaseJob,
    WorkerNode,
    WorkerPluginInstallation,
)
from ..schemas import (
    QuantizationStart,
    ReleaseBuildRunCreate,
    ReleaseCreate,
    ReleaseDeliveryJobCreate,
    ReleaseDeliveryJobPatch,
    ReleaseJobCreate,
    ReleaseJobPatch,
)
from ..services.artifacts import artifact_storage
from ..services.schema_validation import SchemaValidationError, validate_schema_instance
from ..services.tasks import create_task
from .common import add_audit, get_or_404, require_executable

router = APIRouter(tags=["releases"])
_SAFE_TARGET_SUBDIRECTORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise _error(404, "project_not_found", "项目不存在")
    return project


def _plugin(session: Session, plugin_version_id: str, expected_kind: str) -> PluginVersion:
    plugin = session.get(PluginVersion, plugin_version_id)
    if plugin is None:
        raise _error(404, "plugin_version_not_found", "插件版本不存在")
    if plugin.kind != expected_kind or (plugin.source or {}).get("type") != "builtin-catalog":
        raise _error(422, "plugin_kind_mismatch", "插件版本类型与当前发布阶段不匹配")
    if not plugin.enabled:
        raise _error(409, "plugin_disabled", "插件版本已停用")
    return plugin


def _validate_config(plugin: PluginVersion, config: dict[str, Any]) -> None:
    try:
        validate_schema_instance(config, plugin.config_schema or {})
    except SchemaValidationError as exc:
        location = ".".join(str(part) for part in exc.path)
        suffix = f"（{location}）" if location else ""
        raise _error(422, "invalid_plugin_config", f"插件参数不符合 Schema{suffix}：{exc}") from exc


def _require_binding(
    session: Session, project_id: str, plugin: PluginVersion
) -> ProjectPluginBinding:
    binding = session.scalar(
        select(ProjectPluginBinding).where(
            ProjectPluginBinding.project_id == project_id,
            ProjectPluginBinding.plugin_id == plugin.plugin_id,
            ProjectPluginBinding.plugin_version_id == plugin.id,
            ProjectPluginBinding.enabled.is_(True),
        )
    )
    if binding is None:
        raise _error(409, "plugin_not_bound", "项目尚未启用该精确插件版本")
    return binding


def _model_format(model: ModelVersion, artifact: Artifact) -> str:
    metadata = model.metadata_json or {}
    inference = metadata.get("inference") if isinstance(metadata.get("inference"), dict) else {}
    artifact_metadata = artifact.metadata_json or {}
    artifact_inference = (
        artifact_metadata.get("inference")
        if isinstance(artifact_metadata.get("inference"), dict)
        else {}
    )
    value = artifact_metadata.get("format") or artifact_inference.get("format") or inference.get("format")
    if value:
        return str(value)
    if inference.get("pretrained") or metadata.get("source_type") == "huggingface":
        return "transformers-pretrained"
    return ""


def _resolve_source(
    session: Session, project_id: str, source_kind: str, source_id: str
) -> tuple[ModelVersion, QuantizationRun | None, Artifact, dict[str, Any]]:
    quantization: QuantizationRun | None = None
    if source_kind == "model_version":
        model = session.get(ModelVersion, source_id)
        if model is None or model.project_id != project_id:
            raise _error(404, "model_not_found", "源模型版本不存在")
        artifact_id = model.artifact_id
    elif source_kind == "quantization_run":
        quantization = session.get(QuantizationRun, source_id)
        if quantization is None or quantization.project_id != project_id:
            raise _error(404, "quantization_not_found", "量化运行不存在")
        if quantization.status != "succeeded" or not quantization.artifact_id:
            raise _error(409, "quantization_not_ready", "量化运行尚未成功产生制品")
        model = session.get(ModelVersion, quantization.model_version_id)
        if model is None or model.project_id != project_id:
            raise _error(409, "quantization_model_mismatch", "量化运行的源模型不属于当前项目")
        artifact_id = quantization.artifact_id
    else:
        raise _error(422, "invalid_release_source_kind", "不支持的发布源类型")
    if not artifact_id:
        raise _error(422, "source_artifact_required", "发布源没有已登记制品")
    artifact = session.get(Artifact, artifact_id)
    if artifact is None or artifact.availability_state != "available":
        raise _error(422, "source_artifact_unavailable", "发布源制品不存在或不可用")
    source_format = str(quantization.format or "") if quantization else _model_format(model, artifact)
    metadata = {**(model.metadata_json or {}), **(artifact.metadata_json or {})}
    snapshot = {
        "source_kind": source_kind,
        "source_id": source_id,
        "model_version_id": model.id,
        "quantization_run_id": quantization.id if quantization else None,
        "name": model.name,
        "version": model.version,
        "kind": model.kind,
        "artifact_id": artifact.id,
        "artifact_kind": artifact.kind,
        "artifact_content_hash": artifact.content_hash,
        "format": source_format,
        "metadata": metadata,
    }
    return model, quantization, artifact, snapshot


def _gate_snapshot(session: Session, project_id: str, policy: dict[str, Any]) -> dict[str, Any]:
    mode = str(policy.get("mode") or "")
    if mode == "bypassed":
        reason = str(policy.get("reason") or "").strip()
        if not reason:
            raise _error(422, "gate_bypass_reason_required", "跳过发布 Gate 必须填写原因")
        return {"status": "bypassed", "reason": reason}
    if mode == "required":
        gate_id = str(policy.get("gate_result_id") or "")
        gate = session.get(GateResult, gate_id) if gate_id else None
        if gate is None or gate.project_id != project_id:
            raise _error(404, "gate_result_not_found", "发布 Gate 结果不存在")
        if gate.status != "passed":
            raise _error(409, "release_gate_not_passed", "所选发布 Gate 尚未通过")
        return {**gate.to_dict(), "status": "passed"}
    raise _error(422, "invalid_gate_policy", "Gate 策略必须明确选择 required 或 bypassed")


def _eligible_workers(
    session: Session,
    *,
    project_id: str,
    plugin: PluginVersion,
    artifact: Artifact,
    resource_request: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    version_capability = f"{plugin.entrypoint}@{plugin.version}"
    installations = session.scalars(
        select(WorkerPluginInstallation).where(
            WorkerPluginInstallation.plugin_id == plugin.plugin_id,
            WorkerPluginInstallation.state == "ready",
        )
    ).all()
    installed_worker_ids = {
        row.worker_id
        for row in installations
        if any(
            isinstance(item, dict) and item.get("plugin_version_id") == plugin.id
            for item in (row.installed_versions or [])
        )
    }
    workers = (
        session.scalars(
            select(WorkerNode).where(
                WorkerNode.id.in_(installed_worker_ids), WorkerNode.status == "online"
            )
        ).all()
        if installed_worker_ids
        else []
    )
    eligible = {
        worker.id
        for worker in workers
        if {plugin.entrypoint, version_capability}.issubset(set(worker.capabilities or []))
        and (
            plugin.plugin_id != "plugin.local-directory.publisher"
            or bool(((worker.resources or {}).get("release_delivery") or {}).get("local_directory"))
        )
    }
    if not installed_worker_ids:
        raise _error(409, "plugin_version_not_installed", "该插件版本尚未在任何 Worker 上安装")
    access = artifact_storage.inference_access(session, artifact, project_id)
    if access["type"] == "unavailable":
        raise _error(422, "artifact_unavailable", "输入制品没有可供 Worker 使用的位置")
    if access["type"] == "external" and not access["globally_accessible"]:
        eligible &= set(access["worker_ids"])
    request = dict(resource_request or {})
    selector = dict(request.get("worker_selector") or {})
    if selector.get("worker_id"):
        eligible &= {str(selector["worker_id"])}
    requested = {str(value) for value in selector.get("worker_ids") or []}
    if requested:
        eligible &= requested
    if not eligible:
        raise _error(
            422,
            "plugin_artifact_worker_mismatch",
            "没有同时安装精确插件版本、具备目标能力并能访问输入制品的在线 Worker",
        )
    selector["worker_ids"] = sorted(eligible)
    selector["capabilities"] = list(
        dict.fromkeys(
            [
                *(str(value) for value in selector.get("capabilities") or []),
                plugin.entrypoint,
                version_capability,
            ]
        )
    )
    request["worker_selector"] = selector
    return sorted(eligible), request


def _validate_release_job_values(
    session: Session, project_id: str, values: dict[str, Any]
) -> tuple[ModelVersion, QuantizationRun | None, Artifact, PluginVersion, dict, dict]:
    model, quantization, artifact, source = _resolve_source(
        session, project_id, str(values["source_kind"]), str(values["source_id"])
    )
    plugin = _plugin(session, str(values["exporter_plugin_version_id"]), "release_exporter")
    binding = _require_binding(session, project_id, plugin)
    values["config"] = {**(binding.default_config or {}), **(values.get("config") or {})}
    _validate_config(plugin, values["config"])
    input_formats = set(str(value) for value in (plugin.capabilities or {}).get("input_formats") or [])
    if source["format"] not in input_formats:
        raise _error(
            422,
            "exporter_source_format_mismatch",
            f"格式适配插件不接受源格式 {source['format'] or 'unknown'}",
        )
    gate = _gate_snapshot(session, project_id, values.get("gate_policy") or {})
    return model, quantization, artifact, plugin, source, gate


def _job_view(session: Session, job: ReleaseJob) -> dict[str, Any]:
    plugin = session.get(PluginVersion, job.exporter_plugin_version_id)
    runs = session.scalars(
        select(ReleaseBuildRun)
        .where(ReleaseBuildRun.release_job_id == job.id)
        .order_by(ReleaseBuildRun.run_number.desc())
        .limit(20)
    ).all()
    return {
        **job.to_dict(),
        "exporter_plugin_version": plugin.to_dict() if plugin else None,
        "runs": [run.to_dict() for run in runs],
    }


def _delivery_job_view(session: Session, job: ReleaseDeliveryJob) -> dict[str, Any]:
    plugin = session.get(PluginVersion, job.publisher_plugin_version_id)
    runs = session.scalars(
        select(ReleaseDeliveryRun)
        .where(ReleaseDeliveryRun.delivery_job_id == job.id)
        .order_by(ReleaseDeliveryRun.run_number.desc())
        .limit(20)
    ).all()
    return {
        **job.to_dict(),
        "publisher_plugin_version": plugin.to_dict() if plugin else None,
        "runs": [run.to_dict() for run in runs],
    }


def _release_view(session: Session, release: Release) -> dict[str, Any]:
    jobs = session.scalars(
        select(ReleaseDeliveryJob)
        .where(ReleaseDeliveryJob.release_id == release.id)
        .order_by(ReleaseDeliveryJob.created_at.desc())
    ).all()
    return {**release.to_dict(), "delivery_jobs": [_delivery_job_view(session, job) for job in jobs]}


@router.get("/projects/{project_id}/quantizations")
def list_quantizations(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        row.to_dict()
        for row in session.scalars(
            select(QuantizationRun)
            .where(QuantizationRun.project_id == project_id)
            .order_by(QuantizationRun.created_at.desc())
        ).all()
    ]


@router.post("/projects/{project_id}/quantizations", status_code=201)
def start_quantization(
    project_id: str, payload: QuantizationStart, session: Session = Depends(get_session)
) -> dict:
    model = session.get(ModelVersion, payload.model_version_id)
    if model is None or model.project_id != project_id:
        raise _error(404, "model_not_found", "模型版本不存在")
    adapter_id = payload.adapter_id or "builtin.command.quantization"
    require_executable(adapter_id, payload.command)
    task = create_task(
        session,
        project_id=project_id,
        kind="quantize",
        title=f"量化 {model.name} → {payload.format}/{payload.method}",
        adapter_id=adapter_id,
        launcher_id=payload.launcher_id,
        payload={
            "model_version_id": payload.model_version_id,
            "method": payload.method,
            "format": payload.format,
            "calibration_data_ref": payload.calibration_data_ref,
            "resource_request": payload.resource_request,
            "command": payload.command,
        },
    )
    run = QuantizationRun(
        project_id=project_id,
        task_id=task.id,
        status="queued",
        model_version_id=payload.model_version_id,
        method=payload.method,
        format=payload.format,
        calibration_data_ref=payload.calibration_data_ref,
    )
    session.add(run)
    session.flush()
    session.commit()
    return {**run.to_dict(), "task": task.to_dict()}


@router.get("/projects/{project_id}/release-jobs")
def list_release_jobs(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    _project(session, project_id)
    jobs = session.scalars(
        select(ReleaseJob)
        .where(ReleaseJob.project_id == project_id)
        .order_by(ReleaseJob.created_at.desc())
    ).all()
    return [_job_view(session, job) for job in jobs]


@router.post("/projects/{project_id}/release-jobs/preflight")
def preflight_release_job(
    project_id: str, payload: ReleaseJobCreate, session: Session = Depends(get_session)
) -> dict:
    _project(session, project_id)
    values = payload.model_dump()
    _model, _quant, artifact, plugin, source, gate = _validate_release_job_values(
        session, project_id, values
    )
    worker_ids, effective_request = _eligible_workers(
        session,
        project_id=project_id,
        plugin=plugin,
        artifact=artifact,
        resource_request=values.get("resource_request") or {},
    )
    return {
        "status": "passed",
        "source": source,
        "plugin": plugin.to_dict(),
        "gate": gate,
        "eligible_worker_ids": worker_ids,
        "effective_resource_request": effective_request,
    }


@router.post("/projects/{project_id}/release-jobs", status_code=201)
def create_release_job(
    project_id: str, payload: ReleaseJobCreate, session: Session = Depends(get_session)
) -> dict:
    _project(session, project_id)
    values = payload.model_dump()
    _validate_release_job_values(session, project_id, values)
    job = ReleaseJob(project_id=project_id, **values)
    session.add(job)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "release_job_name_exists", "项目中已存在同名发布构建任务") from exc
    add_audit(session, project_id=project_id, action="release_job.create", subject_type="release_job", subject_id=job.id)
    session.commit()
    return _job_view(session, job)


@router.patch("/projects/{project_id}/release-jobs/{job_id}")
def update_release_job(
    project_id: str,
    job_id: str,
    payload: ReleaseJobPatch,
    session: Session = Depends(get_session),
) -> dict:
    _project(session, project_id)
    job = get_or_404(session, ReleaseJob, job_id, "发布构建任务")
    if job.project_id != project_id:
        raise _error(404, "release_job_not_found", "发布构建任务不存在")
    changes = payload.model_dump(exclude_unset=True)
    values = {**job.to_dict(), **changes}
    _validate_release_job_values(session, project_id, values)
    if "config" in changes:
        changes["config"] = values["config"]
    for key, value in changes.items():
        setattr(job, key, value)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "release_job_name_exists", "项目中已存在同名发布构建任务") from exc
    add_audit(
        session,
        project_id=project_id,
        action="release_job.update",
        subject_type="release_job",
        subject_id=job.id,
        payload={"fields": sorted(changes)},
    )
    session.commit()
    return _job_view(session, job)


@router.get("/projects/{project_id}/release-jobs/{job_id}/runs")
def list_release_build_runs(
    project_id: str, job_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    job = get_or_404(session, ReleaseJob, job_id, "发布构建任务")
    if job.project_id != project_id:
        raise _error(404, "release_job_not_found", "发布构建任务不存在")
    return [
        run.to_dict()
        for run in session.scalars(
            select(ReleaseBuildRun)
            .where(ReleaseBuildRun.release_job_id == job.id)
            .order_by(ReleaseBuildRun.run_number.desc())
        ).all()
    ]


@router.get("/release-build-runs/{run_id}")
def get_release_build_run(run_id: str, session: Session = Depends(get_session)) -> dict:
    return get_or_404(session, ReleaseBuildRun, run_id, "发布构建运行").to_dict()


@router.post("/projects/{project_id}/release-jobs/{job_id}/runs", status_code=201)
def run_release_job(
    project_id: str,
    job_id: str,
    payload: ReleaseBuildRunCreate,
    session: Session = Depends(get_session),
) -> dict:
    _project(session, project_id)
    job = get_or_404(session, ReleaseJob, job_id, "发布构建任务")
    if job.project_id != project_id:
        raise _error(404, "release_job_not_found", "发布构建任务不存在")
    if not job.enabled:
        raise _error(409, "release_job_disabled", "发布构建任务已停用")
    if session.scalar(
        select(Release.id).where(Release.project_id == project_id, Release.version == payload.release_version)
    ) or session.scalar(
        select(ReleaseBuildRun.id).where(
            ReleaseBuildRun.project_id == project_id,
            ReleaseBuildRun.release_version == payload.release_version,
        )
    ):
        raise _error(409, "release_version_exists", "项目中已存在该发布版本或正在构建")
    values = job.to_dict()
    _model, _quantization, artifact, plugin, source, gate = _validate_release_job_values(
        session, project_id, values
    )
    worker_ids, effective_request = _eligible_workers(
        session,
        project_id=project_id,
        plugin=plugin,
        artifact=artifact,
        resource_request=job.resource_request or {},
    )
    run_number = int(
        session.scalar(
            select(func.max(ReleaseBuildRun.run_number)).where(ReleaseBuildRun.release_job_id == job.id)
        )
        or 0
    ) + 1
    plugin_snapshot = {
        "plugin_id": plugin.plugin_id,
        "plugin_version_id": plugin.id,
        "version": plugin.version,
        "entrypoint": plugin.entrypoint,
        "protocol_version": plugin.protocol_version,
    }
    config_snapshot = {"config": job.config or {}, "resource_request": effective_request, "launcher_id": job.launcher_id}
    task = create_task(
        session,
        project_id=project_id,
        kind="release_export",
        title=f"构建发布 {payload.release_name} {payload.release_version}",
        adapter_id=plugin.entrypoint,
        launcher_id=job.launcher_id,
        payload={
            "release_job_id": job.id,
            "run_number": run_number,
            "release_name": payload.release_name,
            "release_version": payload.release_version,
            "plugin": {
                "id": plugin.plugin_id,
                "version": plugin.version,
                "plugin_version_id": plugin.id,
                "entrypoint": plugin.entrypoint,
                "runtime": plugin.runtime or {},
            },
            "model": source,
            "config": job.config or {},
            "resource_request": effective_request,
            "command": {},
        },
    )
    run = ReleaseBuildRun(
        project_id=project_id,
        release_job_id=job.id,
        run_number=run_number,
        task_id=task.id,
        status="queued",
        release_name=payload.release_name,
        release_version=payload.release_version,
        source_snapshot=source,
        plugin_snapshot=plugin_snapshot,
        config_snapshot=config_snapshot,
        gate_snapshot=gate,
        model_card_snapshot=job.model_card_template,
        license_snapshot=job.license_policy or {},
    )
    session.add(run)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "release_version_exists", "项目中已存在该发布版本或正在构建") from exc
    add_audit(
        session,
        project_id=project_id,
        action="release_job.run",
        subject_type="release_build_run",
        subject_id=run.id,
        payload={"job_id": job.id, "run_number": run_number, "task_id": task.id, "eligible_worker_ids": worker_ids},
    )
    session.commit()
    return {**run.to_dict(), "task": task.to_dict()}


@router.get("/projects/{project_id}/releases")
def list_releases(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    _project(session, project_id)
    rows = session.scalars(
        select(Release).where(Release.project_id == project_id).order_by(Release.created_at.desc())
    ).all()
    return [_release_view(session, row) for row in rows]


@router.get("/projects/{project_id}/releases/{release_id}")
def get_release(project_id: str, release_id: str, session: Session = Depends(get_session)) -> dict:
    release = get_or_404(session, Release, release_id, "发布版本")
    if release.project_id != project_id:
        raise _error(404, "release_not_found", "发布版本不存在")
    return _release_view(session, release)


def _validate_delivery_values(
    session: Session, project_id: str, release: Release, values: dict[str, Any]
) -> tuple[PluginVersion, Artifact]:
    if release.project_id != project_id:
        raise _error(404, "release_not_found", "发布版本不存在")
    if release.release_kind != "runtime" or not release.runtime_artifact_id:
        raise _error(409, "release_runtime_required", "仅 Manifest 的旧版发布不能执行目标交付")
    artifact = session.get(Artifact, release.runtime_artifact_id)
    if artifact is None or artifact.availability_state != "available":
        raise _error(422, "release_artifact_unavailable", "发布运行时制品不可用")
    plugin = _plugin(session, str(values["publisher_plugin_version_id"]), "release_publisher")
    binding = _require_binding(session, project_id, plugin)
    values["config"] = {**(binding.default_config or {}), **(values.get("config") or {})}
    _validate_config(plugin, values["config"])
    if plugin.plugin_id == "plugin.local-directory.publisher" and not _SAFE_TARGET_SUBDIRECTORY.fullmatch(
        str((values.get("config") or {}).get("target_subdirectory") or "")
    ):
        raise _error(
            422,
            "invalid_target_subdirectory",
            "本地交付子目录只能使用一个安全路径片段",
        )
    runtime_format = str((release.runtime_snapshot or {}).get("format") or "")
    accepted = set(str(value) for value in (plugin.capabilities or {}).get("input_formats") or [])
    if runtime_format not in accepted:
        raise _error(
            422,
            "publisher_runtime_format_mismatch",
            f"目标交付插件不接受运行时格式 {runtime_format or 'unknown'}",
        )
    return plugin, artifact


@router.get("/projects/{project_id}/releases/{release_id}/delivery-jobs")
def list_delivery_jobs(project_id: str, release_id: str, session: Session = Depends(get_session)) -> list[dict]:
    release = get_or_404(session, Release, release_id, "发布版本")
    if release.project_id != project_id:
        raise _error(404, "release_not_found", "发布版本不存在")
    jobs = session.scalars(
        select(ReleaseDeliveryJob)
        .where(ReleaseDeliveryJob.release_id == release_id)
        .order_by(ReleaseDeliveryJob.created_at.desc())
    ).all()
    return [_delivery_job_view(session, job) for job in jobs]


@router.post("/projects/{project_id}/releases/{release_id}/delivery-preflight")
def preflight_delivery(
    project_id: str,
    release_id: str,
    payload: ReleaseDeliveryJobCreate,
    session: Session = Depends(get_session),
) -> dict:
    release = get_or_404(session, Release, release_id, "发布版本")
    values = payload.model_dump()
    plugin, artifact = _validate_delivery_values(session, project_id, release, values)
    worker_ids, effective_request = _eligible_workers(
        session,
        project_id=project_id,
        plugin=plugin,
        artifact=artifact,
        resource_request=values.get("resource_request") or {},
    )
    return {
        "status": "passed",
        "plugin": plugin.to_dict(),
        "runtime_format": (release.runtime_snapshot or {}).get("format"),
        "eligible_worker_ids": worker_ids,
        "effective_resource_request": effective_request,
    }


@router.post("/projects/{project_id}/releases/{release_id}/delivery-jobs", status_code=201)
def create_delivery_job(
    project_id: str,
    release_id: str,
    payload: ReleaseDeliveryJobCreate,
    session: Session = Depends(get_session),
) -> dict:
    release = get_or_404(session, Release, release_id, "发布版本")
    values = payload.model_dump()
    _validate_delivery_values(session, project_id, release, values)
    job = ReleaseDeliveryJob(project_id=project_id, release_id=release_id, **values)
    session.add(job)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "delivery_job_name_exists", "该发布版本中已存在同名交付任务") from exc
    add_audit(
        session,
        project_id=project_id,
        action="release_delivery_job.create",
        subject_type="release_delivery_job",
        subject_id=job.id,
        payload={"release_id": release_id},
    )
    session.commit()
    return _delivery_job_view(session, job)


@router.patch("/projects/{project_id}/releases/{release_id}/delivery-jobs/{job_id}")
def update_delivery_job(
    project_id: str,
    release_id: str,
    job_id: str,
    payload: ReleaseDeliveryJobPatch,
    session: Session = Depends(get_session),
) -> dict:
    release = get_or_404(session, Release, release_id, "发布版本")
    job = get_or_404(session, ReleaseDeliveryJob, job_id, "交付任务")
    if release.project_id != project_id or job.release_id != release_id:
        raise _error(404, "delivery_job_not_found", "交付任务不存在")
    changes = payload.model_dump(exclude_unset=True)
    values = {**job.to_dict(), **changes}
    _validate_delivery_values(session, project_id, release, values)
    if "config" in changes:
        changes["config"] = values["config"]
    for key, value in changes.items():
        setattr(job, key, value)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "delivery_job_name_exists", "该发布版本中已存在同名交付任务") from exc
    add_audit(
        session,
        project_id=project_id,
        action="release_delivery_job.update",
        subject_type="release_delivery_job",
        subject_id=job.id,
        payload={"fields": sorted(changes)},
    )
    session.commit()
    return _delivery_job_view(session, job)


@router.get("/projects/{project_id}/releases/{release_id}/delivery-jobs/{job_id}/runs")
def list_delivery_runs(
    project_id: str,
    release_id: str,
    job_id: str,
    session: Session = Depends(get_session),
) -> list[dict]:
    job = get_or_404(session, ReleaseDeliveryJob, job_id, "交付任务")
    if job.project_id != project_id or job.release_id != release_id:
        raise _error(404, "delivery_job_not_found", "交付任务不存在")
    return [
        row.to_dict()
        for row in session.scalars(
            select(ReleaseDeliveryRun)
            .where(ReleaseDeliveryRun.delivery_job_id == job.id)
            .order_by(ReleaseDeliveryRun.run_number.desc())
        ).all()
    ]


@router.get("/release-delivery-runs/{run_id}")
def get_delivery_run(run_id: str, session: Session = Depends(get_session)) -> dict:
    return get_or_404(session, ReleaseDeliveryRun, run_id, "交付运行").to_dict()


@router.post(
    "/projects/{project_id}/releases/{release_id}/delivery-jobs/{job_id}/runs",
    status_code=201,
)
def run_delivery_job(
    project_id: str,
    release_id: str,
    job_id: str,
    session: Session = Depends(get_session),
) -> dict:
    release = get_or_404(session, Release, release_id, "发布版本")
    job = get_or_404(session, ReleaseDeliveryJob, job_id, "交付任务")
    if release.project_id != project_id or job.project_id != project_id or job.release_id != release_id:
        raise _error(404, "delivery_job_not_found", "交付任务不存在")
    if not job.enabled:
        raise _error(409, "delivery_job_disabled", "交付任务已停用")
    values = job.to_dict()
    plugin, artifact = _validate_delivery_values(session, project_id, release, values)
    worker_ids, effective_request = _eligible_workers(
        session,
        project_id=project_id,
        plugin=plugin,
        artifact=artifact,
        resource_request=job.resource_request or {},
    )
    run_number = int(
        session.scalar(
            select(func.max(ReleaseDeliveryRun.run_number)).where(ReleaseDeliveryRun.delivery_job_id == job.id)
        )
        or 0
    ) + 1
    release_snapshot = {
        "id": release.id,
        "name": release.name,
        "version": release.version,
        "manifest_artifact_id": release.manifest_artifact_id,
        "gate_result": release.gate_result,
    }
    artifact_snapshot = {
        "id": artifact.id,
        "kind": artifact.kind,
        "content_hash": artifact.content_hash,
        "format": (release.runtime_snapshot or {}).get("format"),
        "metadata": artifact.metadata_json or {},
    }
    plugin_snapshot = {
        "plugin_id": plugin.plugin_id,
        "plugin_version_id": plugin.id,
        "version": plugin.version,
        "entrypoint": plugin.entrypoint,
        "protocol_version": plugin.protocol_version,
    }
    config_snapshot = {"config": job.config or {}, "resource_request": effective_request, "launcher_id": job.launcher_id}
    task = create_task(
        session,
        project_id=project_id,
        kind="release_publish",
        title=f"交付发布 {release.name} {release.version} → {job.name}",
        adapter_id=plugin.entrypoint,
        launcher_id=job.launcher_id,
        payload={
            "delivery_job_id": job.id,
            "run_number": run_number,
            "plugin": {
                "id": plugin.plugin_id,
                "version": plugin.version,
                "plugin_version_id": plugin.id,
                "entrypoint": plugin.entrypoint,
                "runtime": plugin.runtime or {},
            },
            "release": release_snapshot,
            "artifact": artifact_snapshot,
            "model": {
                "artifact_id": artifact.id,
                "artifact_content_hash": artifact.content_hash,
                "format": artifact_snapshot["format"],
                "metadata": artifact.metadata_json or {},
            },
            "config": job.config or {},
            "resource_request": effective_request,
            "command": {},
        },
    )
    run = ReleaseDeliveryRun(
        project_id=project_id,
        delivery_job_id=job.id,
        run_number=run_number,
        task_id=task.id,
        status="queued",
        release_snapshot=release_snapshot,
        artifact_snapshot=artifact_snapshot,
        plugin_snapshot=plugin_snapshot,
        config_snapshot=config_snapshot,
    )
    session.add(run)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="release_delivery_job.run",
        subject_type="release_delivery_run",
        subject_id=run.id,
        payload={"job_id": job.id, "run_number": run_number, "task_id": task.id, "eligible_worker_ids": worker_ids},
    )
    session.commit()
    return {**run.to_dict(), "task": task.to_dict()}


@router.post("/projects/{project_id}/releases", status_code=201)
def create_legacy_release(
    project_id: str, payload: ReleaseCreate, session: Session = Depends(get_session)
) -> dict:
    """Compatibility-only manifest release. New UI uses ReleaseJob/BuildRun."""
    model = session.get(ModelVersion, payload.model_version_id)
    if model is None or model.project_id != project_id:
        raise _error(404, "model_not_found", "模型版本不存在")
    quantization: QuantizationRun | None = None
    if payload.quantization_run_id:
        quantization = session.get(QuantizationRun, payload.quantization_run_id)
        if (
            quantization is None
            or quantization.project_id != project_id
            or quantization.model_version_id != model.id
            or quantization.status != "succeeded"
        ):
            raise _error(409, "quantization_not_ready", "量化版本未完成或不属于所选模型")
    gate_result = {"status": "bypassed", "reason": "Legacy manifest-only API does not execute a release Gate"}
    release = Release(
        project_id=project_id,
        gate_result=gate_result,
        release_kind="legacy_manifest_only",
        **payload.model_dump(),
    )
    session.add(release)
    session.flush()
    source_artifact_id = quantization.artifact_id if quantization else model.artifact_id
    manifest = {
        "schema_version": 1,
        "release_kind": "legacy_manifest_only",
        "release_id": release.id,
        "project_id": project_id,
        "name": release.name,
        "version": release.version,
        "model_version_id": release.model_version_id,
        "quantization_run_id": release.quantization_run_id,
        "source_artifact_id": source_artifact_id,
        "target": release.target,
        "model_card": release.model_card,
        "license_manifest": release.license_manifest,
        "gate_result": gate_result,
    }
    manifest_artifact = artifact_storage.put_bytes(
        session,
        project_id=project_id,
        kind="release_manifest",
        payload=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        filename=f"{release.name}-{release.version}.json",
        media_type="application/json",
        metadata={"release_id": release.id, "source_artifact_id": source_artifact_id},
    )
    release.artifact_id = manifest_artifact.id
    release.manifest_artifact_id = manifest_artifact.id
    release.source_snapshot = {"source_artifact_id": source_artifact_id}
    add_audit(
        session,
        project_id=project_id,
        action="release.publish_legacy_manifest",
        subject_type="release",
        subject_id=release.id,
    )
    session.commit()
    return _release_view(session, release)
