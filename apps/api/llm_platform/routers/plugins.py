from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.inference import MICROLM_INFERENCE_FORMAT, MICROLM_INFERENCE_FORMATS
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ModelVersion,
    PluginVersion,
    Project,
    ProjectPluginBinding,
    QuantizationJob,
    QuantizationRun,
    Task,
    WorkerNode,
    WorkerPluginInstallation,
)
from ..schemas import (
    PluginInstallCreate,
    PluginSuiteInstallCreate,
    PluginUpdateCreate,
    ProjectPluginBindingUpsert,
    QuantizationJobCreate,
    QuantizationJobPatch,
)
from ..services.artifacts import artifact_storage
from ..services.plugin_suites import SUITE_KIND, deployment_view, get_suite, suite_view
from ..services.schema_validation import SchemaValidationError, validate_schema_instance
from ..services.tasks import create_task
from ..services.workers import worker_view
from .common import add_audit, get_or_404

router = APIRouter(tags=["plugins"])
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
_INSTALLABLE_PLUGIN_KINDS = {"quantization", "release_exporter", "release_publisher"}


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise _error(404, "project_not_found", "项目不存在")
    return project


def _installable_plugin(session: Session, plugin_version_id: str) -> PluginVersion:
    plugin = session.get(PluginVersion, plugin_version_id)
    if plugin is None:
        raise _error(404, "plugin_version_not_found", "插件版本不存在")
    if (
        plugin.kind not in _INSTALLABLE_PLUGIN_KINDS
        or (plugin.source or {}).get("type") != "builtin-catalog"
    ):
        raise _error(422, "plugin_not_installable", "该版本不是平台内置目录中的可安装插件")
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


def _semver(value: str) -> tuple[int, int, int]:
    match = _SEMVER.match(value)
    if not match:
        raise _error(422, "invalid_plugin_version", "插件目录版本必须使用语义化版本")
    return tuple(int(part) for part in match.groups())


def _installation_view(session: Session, row: WorkerPluginInstallation) -> dict:
    current = session.get(PluginVersion, row.current_plugin_version_id) if row.current_plugin_version_id else None
    target = session.get(PluginVersion, row.target_plugin_version_id) if row.target_plugin_version_id else None
    worker = session.get(WorkerNode, row.worker_id)
    return {
        **row.to_dict(),
        "worker": worker_view(worker) if worker else None,
        "current_plugin_version": current.to_dict() if current else None,
        "target_plugin_version": target.to_dict() if target else None,
    }


def _queue_install(
    session: Session,
    *,
    project_id: str,
    worker: WorkerNode,
    plugin: PluginVersion,
    installation: WorkerPluginInstallation,
    updating: bool,
    suite_ref: dict | None = None,
) -> dict:
    if worker.status != "online":
        raise _error(409, "worker_offline", "目标 Worker 当前不在线")
    if "builtin.plugin-manager" not in set(worker.capabilities or []):
        raise _error(409, "worker_plugin_manager_unavailable", "目标 Worker 不支持受控插件安装")
    if installation.state in {"installing", "updating"}:
        raise _error(409, "plugin_installation_busy", "该 Worker 上的插件安装任务仍在进行")
    runtime = plugin.runtime or {}
    requirements = runtime.get("requirements") or []
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) and item for item in requirements
    ):
        raise _error(422, "invalid_plugin_runtime", "插件目录没有冻结有效的依赖列表")
    task = create_task(
        session,
        project_id=project_id,
        kind="plugin_install",
        title=f"{'更新' if updating else '安装'}插件 {plugin.plugin_id}@{plugin.version}",
        adapter_id="builtin.plugin-manager",
        payload={
            "plugin": {
                "id": plugin.plugin_id,
                "version": plugin.version,
                "plugin_version_id": plugin.id,
                "entrypoint": plugin.entrypoint,
                "runtime": runtime,
                "source": plugin.source,
            },
            "installation_id": installation.id,
            **({"plugin_suite": suite_ref} if suite_ref else {}),
            "resource_request": {
                "worker_selector": {
                    "worker_id": worker.id,
                    "capabilities": ["builtin.plugin-manager"],
                }
            },
            "command": {},
        },
    )
    installation.target_plugin_version_id = plugin.id
    installation.state = "updating" if updating else "installing"
    installation.task_id = task.id
    installation.last_error = {}
    add_audit(
        session,
        project_id=project_id,
        action="plugin.update" if updating else "plugin.install",
        subject_type="worker_plugin_installation",
        subject_id=installation.id,
        payload={"worker_id": worker.id, "plugin_version_id": plugin.id, "task_id": task.id},
    )
    session.commit()
    return {**_installation_view(session, installation), "task": task.to_dict()}


@router.get("/plugin-installations")
def list_plugin_installations(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.scalars(
        select(WorkerPluginInstallation).order_by(WorkerPluginInstallation.created_at.desc())
    ).all()
    return [_installation_view(session, row) for row in rows]


@router.get("/plugin-suites")
def list_plugin_suites(session: Session = Depends(get_session)) -> list[dict]:
    return [suite_view(session, suite) for suite in session.scalars(select(PluginVersion).where(PluginVersion.kind == SUITE_KIND).order_by(PluginVersion.plugin_id, PluginVersion.version)).all()]


@router.get("/plugin-suites/{version_id}")
def get_plugin_suite(version_id: str, session: Session = Depends(get_session)) -> dict:
    return suite_view(session, get_suite(session, version_id))


@router.get("/plugin-suite-deployments")
def list_suite_deployments(session: Session = Depends(get_session)) -> list[dict]:
    suites = session.scalars(select(PluginVersion).where(PluginVersion.kind == SUITE_KIND)).all()
    workers = session.scalars(select(WorkerNode).order_by(WorkerNode.name)).all()
    return [deployment_view(session, suite, worker) for suite in suites for worker in workers]


@router.post("/projects/{project_id}/plugin-suites/{version_id}/installations", status_code=201)
def install_suite(project_id: str, version_id: str, payload: PluginSuiteInstallCreate, session: Session = Depends(get_session)) -> dict:
    _project(session, project_id)
    suite = get_suite(session, version_id)
    if not suite.enabled:
        raise _error(409, "suite_disabled", "套件版本已停用")
    worker = get_or_404(session, WorkerNode, payload.worker_id, "Worker")
    if worker.status != "online":
        raise _error(409, "worker_offline", "目标 Worker 当前不在线")
    if "builtin.plugin-manager" not in (worker.capabilities or []):
        raise _error(409, "worker_plugin_manager_unavailable", "目标 Worker 不支持插件安装")
    members = {member["key"]: member for member in suite.capabilities["manifest"]["components"] if member["deployment"] == "installable"}
    selected = list(dict.fromkeys(payload.component_keys if payload.component_keys is not None else members))
    if not selected or any(key not in members for key in selected):
        raise _error(422, "invalid_suite_components", "请选择该套件版本中的可安装组件")
    reference = {"suite_version_id": suite.id, "suite_id": suite.plugin_id, "version": suite.version, "manifest_hash": suite.capabilities["manifest_hash"]}
    results = []
    for key in selected:
        member = members[key]
        try:
            plugin = _installable_plugin(session, member["plugin_version_id"])
            row = session.scalar(select(WorkerPluginInstallation).where(WorkerPluginInstallation.worker_id == worker.id, WorkerPluginInstallation.plugin_id == plugin.plugin_id))
            if row and any(item.get("plugin_version_id") == plugin.id for item in row.installed_versions or []):
                results.append({"key": key, "status": "installed", "installation_id": row.id})
                continue
            active = session.get(Task, row.task_id) if row and row.task_id else None
            if row and active and active.observed_state not in {"succeeded", "failed", "cancelled", "lost"} and row.target_plugin_version_id == plugin.id:
                results.append({"key": key, "status": "in_progress", "task_id": active.id, "installation_id": row.id})
                continue
            if row is None:
                row = WorkerPluginInstallation(worker_id=worker.id, plugin_id=plugin.plugin_id, state="pending")
                session.add(row)
                session.flush()
            queued = _queue_install(session, project_id=project_id, worker=worker, plugin=plugin, installation=row, updating=bool(row.current_plugin_version_id), suite_ref={**reference, "component_key": key})
            results.append({"key": key, "status": "queued", "task_id": queued["task"]["id"], "installation_id": row.id})
        except HTTPException as exc:
            session.rollback()
            results.append({"key": key, "status": "failed", "error": exc.detail})
        except IntegrityError:
            session.rollback()
            results.append({"key": key, "status": "failed", "error": {"code": "installation_conflict", "message": "安装状态已变化，请刷新后重试"}})
    return {"suite": reference, "worker_id": worker.id, "results": results}


@router.post("/projects/{project_id}/plugin-installations", status_code=201)
def install_plugin(
    project_id: str, payload: PluginInstallCreate, session: Session = Depends(get_session)
) -> dict:
    _project(session, project_id)
    worker = get_or_404(session, WorkerNode, payload.worker_id, "Worker")
    plugin = _installable_plugin(session, payload.plugin_version_id)
    if worker.status != "online":
        raise _error(409, "worker_offline", "目标 Worker 当前不在线")
    if "builtin.plugin-manager" not in set(worker.capabilities or []):
        raise _error(409, "worker_plugin_manager_unavailable", "目标 Worker 不支持受控插件安装")
    installation = session.scalar(
        select(WorkerPluginInstallation).where(
            WorkerPluginInstallation.worker_id == worker.id,
            WorkerPluginInstallation.plugin_id == plugin.plugin_id,
        )
    )
    if installation is None:
        installation = WorkerPluginInstallation(
            worker_id=worker.id,
            plugin_id=plugin.plugin_id,
            state="pending",
        )
        session.add(installation)
        session.flush()
    installed_ids = {
        str(item.get("plugin_version_id") or "")
        for item in (installation.installed_versions or [])
        if isinstance(item, dict)
    }
    if plugin.id in installed_ids:
        raise _error(409, "plugin_version_already_installed", "该插件版本已安装")
    return _queue_install(
        session,
        project_id=project_id,
        worker=worker,
        plugin=plugin,
        installation=installation,
        updating=bool(installation.current_plugin_version_id),
    )


@router.post("/plugin-installations/{installation_id}/update", status_code=201)
def update_plugin(
    installation_id: str,
    payload: PluginUpdateCreate,
    session: Session = Depends(get_session),
) -> dict:
    _project(session, payload.project_id)
    installation = get_or_404(
        session, WorkerPluginInstallation, installation_id, "插件安装记录"
    )
    worker = get_or_404(session, WorkerNode, installation.worker_id, "Worker")
    current = (
        session.get(PluginVersion, installation.current_plugin_version_id)
        if installation.current_plugin_version_id
        else None
    )
    if current is None:
        raise _error(409, "plugin_not_installed", "插件尚未成功安装，不能执行更新")
    if payload.target_plugin_version_id:
        target = _installable_plugin(session, payload.target_plugin_version_id)
    else:
        candidates = session.scalars(
            select(PluginVersion).where(
                PluginVersion.plugin_id == installation.plugin_id,
                PluginVersion.enabled.is_(True),
            )
        ).all()
        newer = [item for item in candidates if _semver(item.version) > _semver(current.version)]
        if not newer:
            raise _error(409, "plugin_already_latest", "插件已经是目录中的最新版本")
        target = max(newer, key=lambda item: _semver(item.version))
    if target.plugin_id != installation.plugin_id:
        raise _error(422, "plugin_id_mismatch", "更新目标必须属于同一插件")
    if _semver(target.version) <= _semver(current.version):
        raise _error(422, "plugin_update_not_newer", "更新目标必须高于当前版本")
    return _queue_install(
        session,
        project_id=payload.project_id,
        worker=worker,
        plugin=target,
        installation=installation,
        updating=True,
    )


@router.get("/projects/{project_id}/plugin-bindings")
def list_project_plugin_bindings(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _project(session, project_id)
    bindings = session.scalars(
        select(ProjectPluginBinding)
        .where(ProjectPluginBinding.project_id == project_id)
        .order_by(ProjectPluginBinding.created_at)
    ).all()
    return [
        {
            **binding.to_dict(),
            "plugin_version": session.get(PluginVersion, binding.plugin_version_id).to_dict(),
        }
        for binding in bindings
    ]


@router.put("/projects/{project_id}/plugin-bindings/{plugin_id}")
def upsert_project_plugin_binding(
    project_id: str,
    plugin_id: str,
    payload: ProjectPluginBindingUpsert,
    session: Session = Depends(get_session),
) -> dict:
    _project(session, project_id)
    plugin = _installable_plugin(session, payload.plugin_version_id)
    if plugin.plugin_id != plugin_id:
        raise _error(422, "plugin_id_mismatch", "绑定路径与插件版本不一致")
    _validate_config(plugin, payload.default_config)
    binding = session.scalar(
        select(ProjectPluginBinding).where(
            ProjectPluginBinding.project_id == project_id,
            ProjectPluginBinding.plugin_id == plugin_id,
        )
    )
    if binding is None:
        binding = ProjectPluginBinding(project_id=project_id, plugin_id=plugin_id)
        session.add(binding)
    binding.plugin_version_id = plugin.id
    binding.enabled = payload.enabled
    binding.default_config = payload.default_config
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="plugin.bind",
        subject_type="project_plugin_binding",
        subject_id=binding.id,
        payload={"plugin_version_id": plugin.id, "enabled": payload.enabled},
    )
    session.commit()
    return {**binding.to_dict(), "plugin_version": plugin.to_dict()}


def _validate_job_values(
    session: Session,
    project_id: str,
    values: dict[str, Any],
) -> tuple[ModelVersion, PluginVersion]:
    model = session.get(ModelVersion, values["model_version_id"])
    if model is None or model.project_id != project_id:
        raise _error(404, "model_not_found", "模型版本不存在")
    if not model.artifact_id:
        raise _error(422, "model_artifact_required", "量化插件需要已登记的本地模型制品")
    artifact = session.get(Artifact, model.artifact_id)
    if artifact is None or artifact.availability_state != "available":
        raise _error(422, "model_artifact_unavailable", "模型制品不存在或不可用")
    if artifact.kind not in {"model", "model_weights", "quantized_model"}:
        raise _error(
            422,
            "model_artifact_not_model_directory",
            "量化插件只接受已封装的模型目录制品，不直接接受训练 Checkpoint",
        )
    plugin = _installable_plugin(session, values["plugin_version_id"])
    metadata = model.metadata_json or {}
    inference = metadata.get("inference") if isinstance(metadata.get("inference"), dict) else {}
    model_format = str(inference.get("format") or "")
    if plugin.plugin_id == "plugin.microlm.quantization":
        if model_format != MICROLM_INFERENCE_FORMAT:
            raise _error(
                422,
                "unsupported_model_format",
                "MicroLM 量化插件只接受未量化的 microlm-hybrid-safetensors 制品",
            )
    elif model_format in MICROLM_INFERENCE_FORMATS:
        raise _error(
            422,
            "unsupported_model_format",
            "所选量化插件不支持 MicroLM 自研模型格式",
        )
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
    _validate_config(plugin, values.get("config") or {})
    if (values.get("config") or {}).get("algorithm") == "GPTQ" and not values.get(
        "calibration_data_ref"
    ):
        raise _error(422, "calibration_required", "GPTQ 必须配置显式校准数据")
    calibration = values.get("calibration_data_ref") or {}
    if (values.get("config") or {}).get("algorithm") == "GPTQ":
        samples = calibration.get("samples") if isinstance(calibration, dict) else None
        if (
            not isinstance(samples, list)
            or not samples
            or len(samples) > 2048
            or any(not isinstance(item, str) or not item.strip() for item in samples)
        ):
            raise _error(
                422,
                "invalid_calibration_data",
                "GPTQ 首版需要 1–2048 条非空文本 samples，运行时不会隐式联网取数据",
            )
    return model, plugin


def _job_view(session: Session, job: QuantizationJob) -> dict:
    runs = session.scalars(
        select(QuantizationRun)
        .where(QuantizationRun.quantization_job_id == job.id)
        .order_by(QuantizationRun.run_number.desc())
        .limit(20)
    ).all()
    return {**job.to_dict(), "runs": [run.to_dict() for run in runs]}


@router.get("/projects/{project_id}/quantization-jobs")
def list_quantization_jobs(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _project(session, project_id)
    jobs = session.scalars(
        select(QuantizationJob)
        .where(QuantizationJob.project_id == project_id)
        .order_by(QuantizationJob.created_at.desc())
    ).all()
    return [_job_view(session, job) for job in jobs]


@router.post("/projects/{project_id}/quantization-jobs", status_code=201)
def create_quantization_job(
    project_id: str, payload: QuantizationJobCreate, session: Session = Depends(get_session)
) -> dict:
    _project(session, project_id)
    values = payload.model_dump()
    _validate_job_values(session, project_id, values)
    job = QuantizationJob(project_id=project_id, **values)
    session.add(job)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "quantization_job_name_exists", "项目中已存在同名量化任务") from exc
    add_audit(
        session,
        project_id=project_id,
        action="quantization_job.create",
        subject_type="quantization_job",
        subject_id=job.id,
    )
    session.commit()
    return _job_view(session, job)


@router.patch("/projects/{project_id}/quantization-jobs/{job_id}")
def update_quantization_job(
    project_id: str,
    job_id: str,
    payload: QuantizationJobPatch,
    session: Session = Depends(get_session),
) -> dict:
    _project(session, project_id)
    job = get_or_404(session, QuantizationJob, job_id, "量化任务")
    if job.project_id != project_id:
        raise _error(404, "quantization_job_not_found", "量化任务不存在")
    changes = payload.model_dump(exclude_unset=True)
    values = {**job.to_dict(), **changes}
    _validate_job_values(session, project_id, values)
    for key, value in changes.items():
        setattr(job, key, value)
    add_audit(
        session,
        project_id=project_id,
        action="quantization_job.update",
        subject_type="quantization_job",
        subject_id=job.id,
        payload={"fields": sorted(changes)},
    )
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise _error(409, "quantization_job_name_exists", "项目中已存在同名量化任务") from exc
    return _job_view(session, job)


@router.get("/projects/{project_id}/quantization-jobs/{job_id}/runs")
def list_quantization_job_runs(
    project_id: str, job_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    job = get_or_404(session, QuantizationJob, job_id, "量化任务")
    if job.project_id != project_id:
        raise _error(404, "quantization_job_not_found", "量化任务不存在")
    return [
        run.to_dict()
        for run in session.scalars(
            select(QuantizationRun)
            .where(QuantizationRun.quantization_job_id == job.id)
            .order_by(QuantizationRun.run_number.desc())
        ).all()
    ]


@router.post("/projects/{project_id}/quantization-jobs/{job_id}/runs", status_code=201)
def run_quantization_job(
    project_id: str, job_id: str, session: Session = Depends(get_session)
) -> dict:
    _project(session, project_id)
    job = get_or_404(session, QuantizationJob, job_id, "量化任务")
    if job.project_id != project_id:
        raise _error(404, "quantization_job_not_found", "量化任务不存在")
    if not job.enabled:
        raise _error(409, "quantization_job_disabled", "量化任务已停用")
    values = job.to_dict()
    model, plugin = _validate_job_values(session, project_id, values)
    version_capability = f"{plugin.entrypoint}@{plugin.version}"
    ready_installations = session.scalars(
        select(WorkerPluginInstallation).where(
            WorkerPluginInstallation.plugin_id == plugin.plugin_id,
            WorkerPluginInstallation.state == "ready",
        )
    ).all()
    plugin_worker_ids = {
        installation.worker_id
        for installation in ready_installations
        if any(
            isinstance(item, dict) and item.get("plugin_version_id") == plugin.id
            for item in (installation.installed_versions or [])
        )
    }
    if not plugin_worker_ids:
        raise _error(409, "plugin_version_not_installed", "该插件版本尚未在任何 Worker 上安装")
    artifact = session.get(Artifact, model.artifact_id)
    assert artifact is not None
    resource_request = dict(job.resource_request or {})
    selector = dict(resource_request.get("worker_selector") or {})
    access = artifact_storage.inference_access(session, artifact, project_id)
    if access["type"] == "unavailable":
        raise _error(422, "model_artifact_unavailable", "模型制品没有可供 Worker 使用的位置")
    eligible_worker_ids = set(plugin_worker_ids)
    if access["type"] == "external" and not access["globally_accessible"]:
        eligible_worker_ids &= set(access["worker_ids"])
    selected_worker_id = str(selector.get("worker_id") or "")
    if selected_worker_id:
        eligible_worker_ids &= {selected_worker_id}
    requested_worker_ids = {str(value) for value in selector.get("worker_ids") or []}
    if requested_worker_ids:
        eligible_worker_ids &= requested_worker_ids
    if not eligible_worker_ids:
        raise _error(
            422,
            "plugin_model_worker_mismatch",
            "没有同时安装该插件版本并能访问源模型制品的 Worker",
        )
    if not selected_worker_id:
        selector["worker_ids"] = sorted(eligible_worker_ids)
    capabilities = [str(value) for value in selector.get("capabilities") or []]
    selector["capabilities"] = list(
        dict.fromkeys([*capabilities, plugin.entrypoint, version_capability])
    )
    resource_request["worker_selector"] = selector
    source_snapshot = {
        "model_version_id": model.id,
        "name": model.name,
        "version": model.version,
        "kind": model.kind,
        "artifact_id": artifact.id,
        "artifact_kind": artifact.kind,
        "artifact_content_hash": artifact.content_hash,
        "metadata": model.metadata_json or {},
    }
    config_snapshot = {
        "plugin_id": plugin.plugin_id,
        "plugin_version_id": plugin.id,
        "plugin_version": plugin.version,
        "entrypoint": plugin.entrypoint,
        "config": job.config or {},
        "calibration_data_ref": job.calibration_data_ref or {},
        "resource_request": resource_request,
        "launcher_id": job.launcher_id,
    }
    run_number = int(
        session.scalar(
            select(func.max(QuantizationRun.run_number)).where(
                QuantizationRun.quantization_job_id == job.id
            )
        )
        or 0
    ) + 1
    task = create_task(
        session,
        project_id=project_id,
        kind="quantize",
        title=f"运行量化任务 {job.name} #{run_number}",
        adapter_id=plugin.entrypoint,
        launcher_id=job.launcher_id,
        payload={
            "quantization_job_id": job.id,
            "run_number": run_number,
            "plugin": {
                "id": plugin.plugin_id,
                "version": plugin.version,
                "plugin_version_id": plugin.id,
                "entrypoint": plugin.entrypoint,
                "runtime": plugin.runtime or {},
            },
            "model": source_snapshot,
            "config": job.config or {},
            "calibration_data_ref": job.calibration_data_ref or {},
            "resource_request": resource_request,
            "command": {},
        },
    )
    output_formats = (plugin.capabilities or {}).get("output_formats") or []
    output_format = (
        str(output_formats[0])
        if isinstance(output_formats, list) and output_formats
        else "quantized-model"
    )
    run = QuantizationRun(
        project_id=project_id,
        quantization_job_id=job.id,
        run_number=run_number,
        plugin_version_id=plugin.id,
        trigger_kind="manual",
        model_version_id=model.id,
        task_id=task.id,
        method=str((job.config or {}).get("algorithm") or ""),
        format=output_format,
        calibration_data_ref=job.calibration_data_ref or {},
        status="queued",
        config_snapshot=config_snapshot,
        source_snapshot=source_snapshot,
    )
    session.add(run)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="quantization_job.run",
        subject_type="quantization_run",
        subject_id=run.id,
        payload={"job_id": job.id, "run_number": run_number, "task_id": task.id},
    )
    session.commit()
    return {**run.to_dict(), "task": task.to_dict()}
