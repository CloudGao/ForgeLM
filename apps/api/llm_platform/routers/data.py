from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.datasets import parser_request_for_modality
from llm_platform_core.mixtures import (
    RESOLVER_VERSION,
    SAMPLER_VERSION,
    normalize_weight_units,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ArtifactLocation,
    DataConversionRun,
    DataMixture,
    DataMixtureVersion,
    DataProcessingRun,
    Dataset,
    DatasetAnalysisRevision,
    DatasetVersion,
    DatasetVersionIndex,
    PlanRevision,
    Project,
    ProjectDataBinding,
    SampleAnnotation,
    Task,
    TaskAttempt,
    TokenizerVersion,
    TrainingPlan,
)
from ..schemas import (
    AnnotationCreate,
    DatasetCreate,
    DatasetPreflightRequest,
    DatasetReanalyze,
    DatasetUpdate,
    DatasetVersionCreate,
    HuggingFaceDownloadPlanRequest,
    HuggingFaceRepositoryScanRequest,
    MixtureCreate,
    MixtureVersionCreate,
)
from ..services.artifacts import artifact_storage
from ..services.command_data_inputs import freeze_command_data_inputs
from ..services.data_sources import SUPPORTED_SOURCE_TYPES, preflight_source
from ..services.dataset_analyses import (
    TERMINAL_ANALYSIS_STATES,
    active_analysis,
    create_analysis_revision,
    effective_record_manifest,
    effective_supports_capability,
    latest_successful_analysis,
    overlay_analysis,
)
from ..services.dataset_materialization import freeze_import_materialization
from ..services.global_data import project_can_manage_asset
from ..services.project_storage import project_artifact_root
from ..services.tasks import create_task
from .common import add_audit, get_or_404, require_executable

router = APIRouter(tags=["data"])
TERMINAL_TASK_STATES = {"succeeded", "failed", "cancelled", "lost"}


def _with_hf_asset_capability(resource_request: dict) -> dict:
    request = {**resource_request}
    selector = dict(request.get("worker_selector") or {})
    capabilities = [str(value) for value in selector.get("capabilities") or []]
    selector["capabilities"] = list(dict.fromkeys([*capabilities, "builtin.hf-asset"]))
    request["worker_selector"] = selector
    return request


def _latest_task_attempt(session: Session, task_id: str) -> TaskAttempt | None:
    return session.scalar(
        select(TaskAttempt)
        .where(TaskAttempt.task_id == task_id)
        .order_by(TaskAttempt.attempt_number.desc())
    )


def _project_workspace_uri(project: Project) -> str:
    workspace_uri = str(project.workspace_uri or "").strip()
    if not workspace_uri:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "project_workspace_required",
                "message": "请先在项目总览配置项目地址，再接入 Hugging Face 数据资产",
            },
        )
    return workspace_uri


def _successful_hf_task(session: Session, project_id: str, task_id: str, kind: str) -> Task:
    task = get_or_404(session, Task, task_id, "Hugging Face 任务")
    if task.project_id != project_id or task.kind != kind:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_hf_plan",
                "message": "Hugging Face 任务与当前项目或阶段不匹配",
            },
        )
    if task.observed_state != "succeeded" or not isinstance(task.result, dict):
        raise HTTPException(
            status_code=409,
            detail={"code": "hf_plan_not_ready", "message": "Hugging Face 任务尚未成功完成"},
        )
    return task


def _freeze_hf_download_plan(
    session: Session,
    project_id: str,
    repo_id: str,
    source_config: dict,
    resource_request: dict,
) -> tuple[dict, dict]:
    plan_task_id = str(source_config.get("hf_plan_task_id") or "").strip()
    if not plan_task_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "hf_plan_required",
                "message": "新建 Hugging Face 数据资产必须先解析仓库、选择文件并完成下载预检",
            },
        )
    plan_task = _successful_hf_task(session, project_id, plan_task_id, "hf_download_plan")
    plan = plan_task.result or {}
    if str(plan.get("repo_id") or "") != repo_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "hf_plan_repo_mismatch", "message": "下载计划与 Repo ID 不匹配"},
        )
    if not plan.get("disk_space_sufficient"):
        raise HTTPException(
            status_code=409,
            detail={"code": "hf_disk_space_insufficient", "message": "目标 Worker 磁盘空间不足"},
        )
    selected_paths = [str(value) for value in plan.get("selected_paths") or []]
    plan_files = {
        str(item.get("path"))
        for item in plan.get("files") or []
        if isinstance(item, dict) and item.get("path")
    }
    commit_hash = str(plan.get("commit_hash") or "").strip()
    endpoint = str(plan.get("endpoint") or "").strip()
    project_workspace_uri = str(plan.get("project_workspace_uri") or "").strip()
    selected_bytes = int(plan.get("selected_bytes") or 0)
    download_bytes = int(plan.get("download_bytes") or 0)
    cached_bytes = int(plan.get("cached_bytes") or 0)
    if (
        not selected_paths
        or set(selected_paths) != plan_files
        or int(plan.get("file_count") or 0) != len(selected_paths)
        or not commit_hash
        or not endpoint
        or not project_workspace_uri
        or selected_bytes < 0
        or download_bytes < 0
        or cached_bytes < 0
        or cached_bytes + download_bytes != selected_bytes
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "invalid_hf_plan_result", "message": "下载计划结果不完整或不一致"},
        )
    plan_attempt = _latest_task_attempt(session, plan_task.id)
    if plan_attempt is None or not plan_attempt.worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "hf_plan_worker_missing", "message": "无法确定下载计划使用的 Worker"},
        )
    selector = dict(resource_request.get("worker_selector") or {})
    configured_worker = selector.get("worker_id")
    if configured_worker and str(configured_worker) != plan_attempt.worker_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "hf_plan_worker_mismatch", "message": "下载计划与所选 Worker 不匹配"},
        )
    selector["worker_id"] = plan_attempt.worker_id
    request = _with_hf_asset_capability({**resource_request, "worker_selector": selector})
    frozen_config = {
        "requested_revision": plan.get("requested_revision"),
        "revision": commit_hash,
        "commit_hash": commit_hash,
        "selected_paths": selected_paths,
        "allow_patterns": selected_paths,
        "hf_endpoint": endpoint,
        "requested_hf_endpoint": plan.get("requested_endpoint"),
        "endpoint_warning": plan.get("endpoint_warning"),
        "config": plan.get("config"),
        "split": plan.get("split"),
        "hf_plan_task_id": plan_task.id,
        "asset_key": plan_task.id,
        "project_workspace_uri": project_workspace_uri,
        "planned_file_count": plan.get("file_count"),
        "planned_selected_bytes": selected_bytes,
        "planned_cached_bytes": cached_bytes,
        "planned_download_bytes": download_bytes,
    }
    return frozen_config, request


def _contains_reference(value: object, references: set[str]) -> bool:
    if isinstance(value, str):
        return value in references
    if isinstance(value, dict):
        return any(_contains_reference(item, references) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference(item, references) for item in value)
    return False


def _dataset_delete_blockers(
    session: Session, dataset: Dataset, versions: list[DatasetVersion]
) -> list[str]:
    version_ids = {version.id for version in versions}
    task_ids = {version.task_id for version in versions if version.task_id}
    if version_ids:
        task_ids.update(
            task_id
            for task_id in session.scalars(
                select(DatasetAnalysisRevision.task_id).where(
                    DatasetAnalysisRevision.dataset_version_id.in_(version_ids),
                    DatasetAnalysisRevision.task_id.is_not(None),
                )
            ).all()
            if task_id
        )
    blockers: list[str] = []

    if task_ids:
        active_tasks = session.scalars(
            select(Task).where(
                Task.id.in_(task_ids), Task.observed_state.not_in(TERMINAL_TASK_STATES)
            )
        ).all()
        if active_tasks:
            blockers.append("仍有数据导入任务正在运行")

    binding = session.scalar(
        select(ProjectDataBinding.id).where(
            ProjectDataBinding.asset_type == "dataset_version",
            ProjectDataBinding.asset_version_id.in_(version_ids),
        )
    )
    if binding:
        blockers.append("已有项目绑定该数据集版本")

    mixture_versions = session.scalars(select(DataMixtureVersion)).all()
    if any(_contains_reference(version.entries, version_ids) for version in mixture_versions):
        blockers.append("已有数据混合引用该数据集版本")

    processing_run = session.scalar(
        select(DataProcessingRun.id).where(
            (DataProcessingRun.input_dataset_version_id.in_(version_ids))
            | (DataProcessingRun.output_dataset_version_id.in_(version_ids))
        )
    )
    if processing_run:
        blockers.append("已有数据处理运行引用该数据集版本")

    conversion_run = session.scalar(
        select(DataConversionRun.id).where(
            (DataConversionRun.input_dataset_version_id.in_(version_ids))
            | (DataConversionRun.output_dataset_version_id.in_(version_ids))
        )
    )
    if conversion_run:
        blockers.append("已有数据转换运行引用该数据集版本")

    tokenizer = session.scalar(
        select(TokenizerVersion.id).where(
            TokenizerVersion.data_ref_type == "dataset_version",
            TokenizerVersion.data_ref_id.in_(version_ids),
        )
    )
    if tokenizer:
        blockers.append("已有 Tokenizer 版本引用该数据集版本")

    plan_revisions = session.scalars(
        select(PlanRevision)
        .join(TrainingPlan, TrainingPlan.id == PlanRevision.plan_id)
    ).all()
    if any(
        _contains_reference(revision.stages, version_ids)
        or _contains_reference(revision.effective_config, version_ids)
        for revision in plan_revisions
    ):
        blockers.append("已有训练计划引用该数据集版本")

    downstream_tasks = session.scalars(
        select(Task).where(Task.kind != "data_import")
    ).all()
    if any(_contains_reference(task.payload, version_ids) for task in downstream_tasks):
        blockers.append("已有下游任务引用该数据集版本")
    return blockers


@router.post("/projects/{project_id}/datasets/preflight")
def preflight_dataset_source(
    project_id: str,
    payload: DatasetPreflightRequest,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    return preflight_source(
        payload.source_type,
        payload.source_uri,
        payload.source_config,
        storage_root=project_artifact_root(session, project_id),
    )


@router.post("/projects/{project_id}/datasets/huggingface/scan", status_code=202)
def scan_huggingface_repository(
    project_id: str,
    payload: HuggingFaceRepositoryScanRequest,
    session: Session = Depends(get_session),
) -> dict:
    project = get_or_404(session, Project, project_id, "项目")
    _project_workspace_uri(project)
    if "/" not in payload.repo_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_repo_id", "message": "Repo ID 必须使用 org/dataset 格式"},
        )
    source_config = {
        "revision": payload.revision,
        "config": payload.config,
        "split": payload.split,
    }
    if payload.hf_endpoint:
        source_config["hf_endpoint"] = payload.hf_endpoint
    task = create_task(
        session,
        project_id=project_id,
        kind="hf_repository_scan",
        title=f"解析 Hugging Face 仓库：{payload.repo_id}",
        adapter_id="builtin.data-import",
        payload={
            "operation": "hf_scan",
            "source_type": "huggingface",
            "source_uri": payload.repo_id,
            "source_config": source_config,
            "resource_request": _with_hf_asset_capability(payload.resource_request),
        },
    )
    session.commit()
    return task.to_dict()


@router.post("/projects/{project_id}/datasets/huggingface/plans", status_code=202)
def plan_huggingface_download(
    project_id: str,
    payload: HuggingFaceDownloadPlanRequest,
    session: Session = Depends(get_session),
) -> dict:
    project = get_or_404(session, Project, project_id, "项目")
    project_workspace_uri = _project_workspace_uri(project)
    scan_task = _successful_hf_task(session, project_id, payload.scan_task_id, "hf_repository_scan")
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
            detail={
                "code": "invalid_hf_selection",
                "message": "所选文件必须来自已解析并固定版本的仓库文件树",
            },
        )
    scan_attempt = _latest_task_attempt(session, scan_task.id)
    if scan_attempt is None or not scan_attempt.worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "hf_scan_worker_missing", "message": "无法确定仓库解析使用的 Worker"},
        )
    resource_request = _with_hf_asset_capability(
        {"gpus": 0, "worker_selector": {"worker_id": scan_attempt.worker_id}}
    )
    source_config = {
        "requested_revision": scan.get("requested_revision"),
        "commit_hash": scan.get("commit_hash"),
        "selected_paths": selected_paths,
        "hf_endpoint": scan.get("endpoint"),
        "config": scan.get("config"),
        "split": scan.get("split"),
    }
    task = create_task(
        session,
        project_id=project_id,
        kind="hf_download_plan",
        title=f"预检 Hugging Face 文件：{scan.get('repo_id')}",
        adapter_id="builtin.data-import",
        payload={},
    )
    task.payload = {
        "operation": "hf_plan",
        "asset_key": task.id,
        "source_type": "huggingface",
        "source_uri": scan.get("repo_id"),
        "source_config": source_config,
        "resource_request": resource_request,
        "scan_task_id": scan_task.id,
        "project_workspace_uri": project_workspace_uri,
    }
    session.commit()
    return task.to_dict()


def dataset_view(session: Session, dataset: Dataset) -> dict:
    versions = session.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset.id)
        .order_by(DatasetVersion.version.desc())
    ).all()
    artifact_ids = [version.artifact_id for version in versions if version.artifact_id]
    locations = (
        session.scalars(
            select(ArtifactLocation).where(ArtifactLocation.artifact_id.in_(artifact_ids))
        ).all()
        if artifact_ids
        else []
    )
    locations_by_artifact: dict[str, list[ArtifactLocation]] = {}
    for location in locations:
        locations_by_artifact.setdefault(location.artifact_id, []).append(location)
    current = (
        session.get(DatasetVersion, dataset.current_version_id)
        if dataset.current_version_id
        else None
    )
    version_ids = [version.id for version in versions]
    analyses = session.scalars(
        select(DatasetAnalysisRevision)
        .where(DatasetAnalysisRevision.dataset_version_id.in_(version_ids))
        .order_by(
            DatasetAnalysisRevision.dataset_version_id,
            DatasetAnalysisRevision.revision.desc(),
        )
    ).all() if version_ids else []
    indexes = session.scalars(
        select(DatasetVersionIndex).where(
            DatasetVersionIndex.dataset_version_id.in_(version_ids)
        )
    ).all() if versions else []
    analyses_by_version: dict[str, list[DatasetAnalysisRevision]] = {}
    for analysis in analyses:
        analyses_by_version.setdefault(analysis.dataset_version_id, []).append(analysis)
    indexes_by_analysis: dict[str, list[dict]] = {}
    legacy_indexes_by_version: dict[str, list[dict]] = {}
    for index in indexes:
        if index.analysis_revision_id:
            indexes_by_analysis.setdefault(index.analysis_revision_id, []).append(index.to_dict())
        else:
            legacy_indexes_by_version.setdefault(index.dataset_version_id, []).append(index.to_dict())

    def version_view(version: DatasetVersion) -> dict:
        history = analyses_by_version.get(version.id, [])
        latest = next((item for item in history if item.status == "succeeded"), None)
        active = next(
            (item for item in history if item.status not in TERMINAL_ANALYSIS_STATES), None
        )

        def analysis_view(analysis: DatasetAnalysisRevision) -> dict:
            return {
                **analysis.to_dict(),
                "indexes": indexes_by_analysis.get(analysis.id, []),
            }

        effective_indexes = (
            indexes_by_analysis.get(latest.id, [])
            if latest is not None
            else legacy_indexes_by_version.get(version.id, [])
        )
        view = {
            **overlay_analysis(version, latest),
            "indexes": effective_indexes,
            "latest_analysis": analysis_view(latest) if latest else None,
            "active_analysis": analysis_view(active) if active else None,
            "analysis_revisions": [analysis_view(item) for item in history],
        }
        view.pop("materialization_spec", None)
        version_locations = locations_by_artifact.get(str(version.artifact_id or ""), [])
        version_locations = [
            location
            for location in version_locations
            if location.project_id == dataset.project_id
            or (location.project_id is None and location.storage_type == "external")
        ]
        primary = next(
            (
                location
                for location in version_locations
                if location.id == version.primary_location_id
            ),
            None,
        )
        spec = version.materialization_spec or {}
        view["materialization"] = {
            "kind": version.materialization_kind,
            "state": version.materialization_state,
            "primary_location": primary.to_dict() if primary else None,
            "replicas": [
                location.to_dict()
                for location in version_locations
                if location.id != version.primary_location_id
                and location.availability_state == "available"
                and location.last_verified_at is not None
            ],
            "recovery": {
                "source_type": spec.get("source_type"),
                "source_uri": spec.get("source_uri"),
                "replay_adapter_id": spec.get("replay_adapter_id"),
                "source_task_id": spec.get("source_task_id"),
                "parent_dataset_version_id": spec.get("parent_dataset_version_id"),
                "can_replay": bool(spec.get("replay_adapter_id") and spec.get("replay_payload")),
            },
        }
        return view

    return {
        **dataset.to_dict(),
        "current_version": version_view(current) if current else None,
        "versions": [version_view(version) for version in versions],
    }


@router.get("/projects/{project_id}/datasets")
def list_datasets(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    bound_version_ids = set(
        session.scalars(
            select(ProjectDataBinding.asset_version_id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == "dataset_version",
            )
        ).all()
    )
    bound_dataset_ids = set(
        session.scalars(
            select(DatasetVersion.dataset_id).where(
                DatasetVersion.id.in_(bound_version_ids)
            )
        ).all()
    ) if bound_version_ids else set()
    datasets = session.scalars(
        select(Dataset)
        .where(
            (Dataset.project_id == project_id)
            | (Dataset.id.in_(bound_dataset_ids))
        )
        .order_by(Dataset.updated_at.desc())
    ).all()
    result: list[dict] = []
    for dataset in datasets:
        view = dataset_view(session, dataset)
        view["binding_scope"] = "owned" if dataset.project_id == project_id else "bound"
        if dataset.project_id != project_id:
            versions = [
                version
                for version in view["versions"]
                if version["id"] in bound_version_ids
            ]
            view["versions"] = versions
            view["current_version"] = versions[0] if versions else None
        result.append(view)
    return result


@router.post("/projects/{project_id}/datasets", status_code=201)
def create_dataset(
    project_id: str, payload: DatasetCreate, session: Session = Depends(get_session)
) -> dict:
    selected_adapter = payload.adapter_id or "builtin.data-import"
    frozen_command_inputs = freeze_command_data_inputs(session, project_id, payload)
    frozen_source_config = payload.source_config
    frozen_resource_request = payload.resource_request
    if payload.start_import and selected_adapter == "builtin.data-import":
        if payload.source_type not in SUPPORTED_SOURCE_TYPES:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_source", "message": "当前内置导入器不支持该来源类型"},
            )
        if not payload.source_uri or not payload.source_uri.strip():
            raise HTTPException(
                status_code=422,
                detail={"code": "missing_source_uri", "message": "真实数据导入必须提供来源地址"},
            )
        if payload.source_type == "huggingface":
            frozen_source_config, frozen_resource_request = _freeze_hf_download_plan(
                session,
                project_id,
                str(payload.source_uri),
                payload.source_config,
                payload.resource_request,
            )
    dataset = Dataset(
        project_id=project_id,
        name=payload.name,
        source_type=payload.source_type,
        modality=payload.modality,
        source_uri=payload.source_uri,
        source_config=frozen_source_config,
        license=payload.license,
        language=payload.language,
        tags=payload.tags,
    )
    session.add(dataset)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409, detail={"code": "duplicate_dataset", "message": "数据集名称已存在"}
        ) from exc
    initial = DatasetVersionCreate(**(payload.initial_version or {}))
    if payload.start_import:
        initial = initial.model_copy(
            update={
                "raw_size_bytes": 0,
                "document_count": 0,
                "token_estimate": 0,
                "automatic_quality_score": None,
                "human_score": None,
                "quality_report": {},
                "train_ready": False,
            }
        )
    parser_request = parser_request_for_modality(dataset.modality)
    version = DatasetVersion(
        dataset_id=dataset.id,
        version=1,
        modality=dataset.modality,
        parser_id=parser_request["id"] if parser_request else None,
        parser_version=parser_request["version"] if parser_request else None,
        parser_status="pending" if payload.start_import else "not_run",
        parser_config={},
        **initial.model_dump(),
    )
    session.add(version)
    session.flush()
    freeze_import_materialization(dataset, version)
    if payload.start_import:
        adapter_id = selected_adapter
        if adapter_id.startswith("builtin.command."):
            require_executable(adapter_id, payload.command)
        task = create_task(
            session,
            project_id=project_id,
            kind="data_import",
            title=f"导入数据：{dataset.name}",
            adapter_id=adapter_id,
            launcher_id=payload.launcher_id,
            payload={
                "dataset_id": dataset.id,
                "dataset_version_id": version.id,
                "modality": version.modality,
                "parser": parser_request,
                "parser_config": version.parser_config,
                "source_type": dataset.source_type,
                "source_uri": dataset.source_uri,
                "source_config": dataset.source_config,
                "resource_request": frozen_resource_request,
                "command": payload.command,
                **frozen_command_inputs,
            },
        )
        version.task_id = task.id
        analysis = create_analysis_revision(session, version, task_id=task.id)
        task.payload = {**task.payload, "analysis_revision_id": analysis.id}
        freeze_import_materialization(
            dataset,
            version,
            replay_payload=task.payload if adapter_id == "builtin.data-import" else None,
        )
        version.state = "queued"
    dataset.current_version_id = version.id
    dataset.status = (
        "queued" if payload.start_import else "ready" if version.train_ready else "registered"
    )
    add_audit(
        session,
        project_id=project_id,
        action="dataset.create",
        subject_type="dataset",
        subject_id=dataset.id,
    )
    session.commit()
    return dataset_view(session, dataset)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, session: Session = Depends(get_session)) -> dict:
    return dataset_view(session, get_or_404(session, Dataset, dataset_id, "数据集"))


@router.patch("/datasets/{dataset_id}")
def update_dataset(
    dataset_id: str, payload: DatasetUpdate, session: Session = Depends(get_session)
) -> dict:
    dataset = get_or_404(session, Dataset, dataset_id, "数据集")
    current = (
        session.get(DatasetVersion, dataset.current_version_id)
        if dataset.current_version_id
        else None
    )
    fields = payload.model_dump(exclude_unset=True, exclude={"resource_request"})
    requested_uri = fields.pop("source_uri", dataset.source_uri)
    requested_uri = requested_uri.strip() if requested_uri else requested_uri
    requested_config = fields.pop("source_config", dataset.source_config)
    requested_config = dataset.source_config if requested_config is None else requested_config
    source_changed = (
        requested_uri != dataset.source_uri or requested_config != dataset.source_config
    )
    frozen_resource_request = payload.resource_request

    if source_changed:
        if not requested_uri or not requested_uri.strip():
            raise HTTPException(
                status_code=422,
                detail={"code": "missing_source_uri", "message": "数据来源地址不能为空"},
            )
        running_analysis = active_analysis(session, current.id) if current else None
        if running_analysis is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_analysis_active",
                    "message": "当前数据版本正在重新分析，结束后才能修改来源",
                },
            )
        if current and current.task_id:
            current_task = session.get(Task, current.task_id)
            if current_task and current_task.observed_state not in TERMINAL_TASK_STATES:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "dataset_import_active",
                        "message": "当前导入任务仍在运行，结束后才能修改来源",
                    },
                )
        if dataset.source_type == "huggingface":
            requested_config, frozen_resource_request = _freeze_hf_download_plan(
                session,
                dataset.project_id,
                str(requested_uri),
                requested_config,
                payload.resource_request,
            )

    for key, value in fields.items():
        if value is not None:
            setattr(dataset, key, value)

    if source_changed:
        previous_uri = dataset.source_uri
        previous_config = dataset.source_config
        dataset.source_uri = requested_uri
        dataset.source_config = requested_config
        next_version = (
            int(
                session.scalar(
                    select(func.coalesce(func.max(DatasetVersion.version), 0)).where(
                        DatasetVersion.dataset_id == dataset.id
                    )
                )
                or 0
            )
            + 1
        )
        version = DatasetVersion(
            dataset_id=dataset.id,
            version=next_version,
            modality=dataset.modality,
            parser_id=(parser_request_for_modality(dataset.modality) or {}).get("id"),
            parser_version=(parser_request_for_modality(dataset.modality) or {}).get("version"),
            parser_status="pending",
            tier="bronze",
            state="queued",
            source_snapshot={
                "source_changed_from_version_id": current.id if current else None,
                "source_changed_from_uri": previous_uri,
                "source_changed_from_config": previous_config,
            },
            train_ready=False,
        )
        session.add(version)
        session.flush()
        task = create_task(
            session,
            project_id=dataset.project_id,
            kind="data_import",
            title=f"重新导入数据：{dataset.name} V{next_version}",
            adapter_id="builtin.data-import",
            launcher_id="builtin.local-process",
            payload={
                "dataset_id": dataset.id,
                "dataset_version_id": version.id,
                "modality": version.modality,
                "parser": parser_request_for_modality(version.modality),
                "parser_config": version.parser_config,
                "source_type": dataset.source_type,
                "source_uri": dataset.source_uri,
                "source_config": dataset.source_config,
                "resource_request": frozen_resource_request,
                "command": {},
            },
        )
        version.task_id = task.id
        analysis = create_analysis_revision(session, version, task_id=task.id)
        task.payload = {**task.payload, "analysis_revision_id": analysis.id}
        freeze_import_materialization(dataset, version, replay_payload=task.payload)
        dataset.current_version_id = version.id
        dataset.status = "queued"

    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.source.update" if source_changed else "dataset.update",
        subject_type="dataset",
        subject_id=dataset.id,
        payload={"source_changed": source_changed},
    )
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_dataset", "message": "数据集名称已存在"},
        ) from exc
    return dataset_view(session, dataset)


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, session: Session = Depends(get_session)) -> None:
    dataset = get_or_404(session, Dataset, dataset_id, "数据集")
    versions = session.scalars(
        select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id)
    ).all()
    blockers = _dataset_delete_blockers(session, dataset, list(versions))
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_in_use",
                "message": "该数据集暂时不能删除：" + "；".join(blockers),
                "blockers": blockers,
            },
        )
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.delete",
        subject_type="dataset",
        subject_id=dataset.id,
        payload={"name": dataset.name, "version_ids": [version.id for version in versions]},
    )
    session.delete(dataset)
    session.commit()


@router.post("/datasets/{dataset_id}/reanalyze", status_code=202)
def reanalyze_dataset(
    dataset_id: str, payload: DatasetReanalyze, session: Session = Depends(get_session)
) -> dict:
    dataset = get_or_404(session, Dataset, dataset_id, "数据集")
    current = get_or_404(session, DatasetVersion, dataset.current_version_id, "当前数据版本")
    running_analysis = active_analysis(session, current.id)
    if running_analysis is not None:
        raise HTTPException(
            status_code=409,
            detail={"code": "analysis_in_progress", "message": "当前数据版本已有分析任务正在运行"},
        )
    if current.task_id:
        current_task = session.get(Task, current.task_id)
        if current_task and current_task.observed_state not in {
            "succeeded",
            "failed",
            "cancelled",
            "lost",
        }:
            raise HTTPException(
                status_code=409,
                detail={"code": "analysis_in_progress", "message": "当前版本已有分析任务正在运行"},
            )

    if not current.artifact_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "snapshot_required",
                "message": "当前数据版本没有冻结的数据快照，不能安全地重新分析",
            },
        )
    snapshot = get_or_404(session, Artifact, current.artifact_id, "数据快照")
    if snapshot.kind != "dataset_snapshot":
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_snapshot", "message": "当前版本引用的制品不是数据快照"},
        )
    location = session.scalar(
        select(ArtifactLocation)
        .where(
            ArtifactLocation.artifact_id == snapshot.id,
            ArtifactLocation.storage_type == "external",
            ArtifactLocation.availability_state == "available",
        )
        .order_by(ArtifactLocation.worker_id.is_(None), ArtifactLocation.created_at)
    )
    previous_analysis = latest_successful_analysis(session, current.id)
    frozen_source_snapshot = (
        previous_analysis.source_snapshot
        if previous_analysis is not None and (previous_analysis.source_snapshot or {}).get("files")
        else current.source_snapshot or {}
    )
    expected_files = frozen_source_snapshot.get("files") or []
    if location is not None and not expected_files:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "snapshot_identity_missing",
                "message": "该外部数据版本没有冻结文件校验和，无法证明重新分析读取的是同一份数据；请重新导入为新版本",
            },
        )

    from .tokenizers import _route_data_request

    resource_request = _route_data_request(
        session, dataset.project_id, payload.resource_request, [snapshot.id]
    )
    selected_worker_id = (resource_request.get("worker_selector") or {}).get("worker_id")

    analysis = create_analysis_revision(session, current, task_id=None)
    task = create_task(
        session,
        project_id=dataset.project_id,
        kind="data_import",
        title=f"重新分析数据：{dataset.name} V{current.version} · A{analysis.revision}",
        adapter_id="builtin.data-import",
        launcher_id="builtin.local-process",
        payload={
            "dataset_id": dataset.id,
            "dataset_version_id": current.id,
            "analysis_revision_id": analysis.id,
            "modality": current.modality,
            "parser": parser_request_for_modality(current.modality),
            "parser_config": current.parser_config,
            "source_type": "local",
            "source_uri": "",
            "source_config": {},
            "analysis_only": True,
            "reuse_artifact_id": snapshot.id,
            "known_files": [],
            "expected_snapshot_files": expected_files,
            "analysis_input": {
                "artifact_id": snapshot.id,
                "artifact_hash": snapshot.content_hash,
            },
            "resource_request": resource_request,
            "command": {},
        },
    )
    analysis.task_id = task.id
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.reanalyze",
        subject_type="dataset_analysis_revision",
        subject_id=analysis.id,
        payload={
            "dataset_version_id": current.id,
            "dataset_version": current.version,
            "analysis_revision": analysis.revision,
            "snapshot_artifact_id": snapshot.id,
            "worker_id": selected_worker_id,
        },
    )
    session.commit()
    return {
        **dataset_view(session, dataset),
        "reanalyze": {
            "task_id": task.id,
            "analysis_revision_id": analysis.id,
            "analysis_revision": analysis.revision,
            "dataset_version_id": current.id,
            "analysis_only": True,
            "worker_id": selected_worker_id,
        },
    }


@router.post("/datasets/{dataset_id}/versions", status_code=201)
def create_dataset_version(
    dataset_id: str, payload: DatasetVersionCreate, session: Session = Depends(get_session)
) -> dict:
    dataset = get_or_404(session, Dataset, dataset_id, "数据集")
    latest = session.scalar(
        select(func.coalesce(func.max(DatasetVersion.version), 0)).where(
            DatasetVersion.dataset_id == dataset_id
        )
    )
    version = DatasetVersion(
        dataset_id=dataset_id,
        version=int(latest or 0) + 1,
        modality=dataset.modality,
        parser_status="not_run",
        **payload.model_dump(),
    )
    session.add(version)
    session.flush()
    freeze_import_materialization(dataset, version)
    dataset.current_version_id = version.id
    dataset.status = "ready" if version.train_ready else "processing"
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.version.create",
        subject_type="dataset_version",
        subject_id=version.id,
    )
    session.commit()
    return version.to_dict()


@router.post("/dataset-versions/{version_id}/annotations", status_code=201)
def annotate_sample(
    version_id: str, payload: AnnotationCreate, session: Session = Depends(get_session)
) -> dict:
    version = get_or_404(session, DatasetVersion, version_id, "数据版本")
    annotation = SampleAnnotation(dataset_version_id=version.id, **payload.model_dump())
    session.add(annotation)
    session.flush()
    existing = session.scalars(
        select(SampleAnnotation)
        .where(SampleAnnotation.dataset_version_id == version.id)
        .order_by(SampleAnnotation.created_at, SampleAnnotation.id)
    ).all()
    latest_by_sample = {item.sample_key: item for item in existing}
    latest_by_sample[annotation.sample_key] = annotation
    scores = [
        item.human_score for item in latest_by_sample.values() if item.human_score is not None
    ]
    version.human_score = sum(scores) / len(scores) if scores else None
    session.commit()
    return annotation.to_dict()


@router.get("/dataset-versions/{version_id}/annotations")
def list_annotations(version_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        item.to_dict()
        for item in session.scalars(
            select(SampleAnnotation)
            .where(SampleAnnotation.dataset_version_id == version_id)
            .order_by(SampleAnnotation.created_at.desc())
        ).all()
    ]


def mixture_view(session: Session, mixture: DataMixture) -> dict:
    versions = session.scalars(
        select(DataMixtureVersion)
        .where(DataMixtureVersion.mixture_id == mixture.id)
        .order_by(DataMixtureVersion.version.desc())
    ).all()
    current = (
        session.get(DataMixtureVersion, mixture.current_version_id)
        if mixture.current_version_id
        else None
    )
    return {
        **mixture.to_dict(),
        "current_version": current.to_dict() if current else None,
        "versions": [version.to_dict() for version in versions],
    }


def _resolved_mixture_entries(
    session: Session, project_id: str, raw_entries: list[dict]
) -> list[dict]:
    if not raw_entries:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_mixture_sources", "message": "数据混合至少需要一个来源"},
        )
    version_ids = [str(entry.get("dataset_version_id") or "") for entry in raw_entries]
    if any(not version_id for version_id in version_ids) or len(set(version_ids)) != len(
        version_ids
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_mixture_sources", "message": "数据来源版本不能为空或重复"},
        )
    try:
        weight_units = normalize_weight_units([entry.get("weight", 0) for entry in raw_entries])
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_weights", "message": "混合权重必须是大于零的有限数值"},
        ) from exc
    resolved: list[dict] = []
    for raw_entry, units, version_id in zip(raw_entries, weight_units, version_ids, strict=True):
        if float(raw_entry.get("weight") or 0) <= 0:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_weights", "message": "每个已选来源的权重必须大于零"},
            )
        version = session.get(DatasetVersion, version_id)
        dataset = session.get(Dataset, version.dataset_id) if version else None
        if (
            version is None
            or dataset is None
            or not project_can_manage_asset(
                session, project_id, "dataset_version", version.id
            )
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_dataset_version",
                    "message": "数据混合引用了不存在或不属于当前项目的数据版本",
                },
            )
        if version.state != "ready" or not effective_supports_capability(
            session, version, "text_pretraining"
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_version_not_ready",
                    "message": f"数据版本 {dataset.name} V{version.version} 尚不可用于混合",
                },
            )
        record_manifest = effective_record_manifest(session, version)
        dataset_artifact = (
            session.get(Artifact, version.artifact_id) if version.artifact_id else None
        )
        if dataset_artifact is None or dataset_artifact.kind != "dataset_snapshot":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_snapshot_required",
                    "message": f"数据版本 {dataset.name} V{version.version} 没有不可变数据快照",
                },
            )
        if (
            record_manifest is None
            or record_manifest.kind != "dataset_record_manifest"
            or artifact_storage.managed_payload_path(session, record_manifest, project_id) is None
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "record_manifest_required",
                    "message": f"数据版本 {dataset.name} V{version.version} 没有稳定文档清单，请重新分析后再使用",
                },
            )
        analysis = latest_successful_analysis(session, version.id)
        resolved.append(
            {
                **raw_entry,
                "dataset_version_id": version.id,
                "name": dataset.name,
                "weight_units": units,
                "normalized_weight": units / 1_000_000,
                "dataset_artifact_id": version.artifact_id,
                "dataset_artifact_hash": (
                    dataset_artifact.content_hash if dataset_artifact else None
                ),
                "record_manifest_artifact_id": record_manifest.id,
                "record_manifest_hash": record_manifest.content_hash,
                **(
                    {
                        "analysis_revision_id": analysis.id,
                        "analysis_revision": analysis.revision,
                    }
                    if analysis is not None
                    else {}
                ),
                "document_count": (
                    analysis.document_count if analysis is not None else version.document_count
                ),
                "token_estimate": (
                    analysis.token_estimate if analysis is not None else version.token_estimate
                ),
            }
        )
    return resolved


def _queue_mixture_resolution(
    session: Session,
    *,
    project_id: str,
    mixture: DataMixture,
    version: DataMixtureVersion,
) -> Task:
    task = create_task(
        session,
        project_id=project_id,
        kind="mixture_resolve",
        title=f"解析数据混合：{mixture.name} V{version.version}",
        adapter_id="builtin.data-mixture",
        launcher_id="builtin.local-process",
        payload={
            "mixture_id": mixture.id,
            "mixture_version_id": version.id,
            "seed": version.seed,
            "target_tokens": version.target_tokens,
            "entries": version.entries,
            "resource_request": {"gpus": 0},
        },
    )
    version.task_id = task.id
    return task


@router.get("/projects/{project_id}/mixtures")
def list_mixtures(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    bound_version_ids = set(
        session.scalars(
            select(ProjectDataBinding.asset_version_id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == "mixture_version",
            )
        ).all()
    )
    bound_mixture_ids = set(
        session.scalars(
            select(DataMixtureVersion.mixture_id).where(
                DataMixtureVersion.id.in_(bound_version_ids)
            )
        ).all()
    ) if bound_version_ids else set()
    mixtures = session.scalars(
        select(DataMixture)
        .where(
            (DataMixture.project_id == project_id)
            | (DataMixture.id.in_(bound_mixture_ids))
        )
        .order_by(DataMixture.updated_at.desc())
    ).all()
    result: list[dict] = []
    for mixture in mixtures:
        view = mixture_view(session, mixture)
        view["binding_scope"] = "owned" if mixture.project_id == project_id else "bound"
        if mixture.project_id != project_id:
            versions = [
                version
                for version in view["versions"]
                if version["id"] in bound_version_ids
            ]
            view["versions"] = versions
            view["current_version"] = versions[0] if versions else None
        result.append(view)
    return result


@router.post("/projects/{project_id}/mixtures", status_code=201)
def create_mixture(
    project_id: str, payload: MixtureCreate, session: Session = Depends(get_session)
) -> dict:
    consumer_project_id = project_id
    if payload.asset_scope == "global":
        from ..services.global_data import ensure_global_data_project, project_has_asset_access

        if not payload.entries or any(not project_has_asset_access(
            session, project_id, "dataset_version", str(entry.get("dataset_version_id") or "")
        ) for entry in payload.entries):
            raise HTTPException(422, "创建全局混合前，消费项目必须绑定每个来源版本")
        project_id = ensure_global_data_project(session).id
    normalized = _resolved_mixture_entries(session, project_id, payload.entries)
    mixture = DataMixture(project_id=project_id, name=payload.name, description=payload.description)
    session.add(mixture)
    session.flush()
    version = DataMixtureVersion(
        mixture_id=mixture.id,
        version=1,
        entries=normalized,
        target_tokens=payload.target_tokens,
        gate_result={"status": "passed", "issues": []},
        status="resolving",
        seed=payload.seed,
        resolver_version=RESOLVER_VERSION,
        sampler_version=SAMPLER_VERSION,
        locked=False,
    )
    session.add(version)
    session.flush()
    _queue_mixture_resolution(session, project_id=project_id, mixture=mixture, version=version)
    mixture.current_version_id = version.id
    if consumer_project_id != project_id:
        from ..schemas import ProjectDataBindingCreate
        from ..services.transactions import domain_transaction
        from .catalog import create_project_data_binding

        with domain_transaction(session):
            create_project_data_binding(consumer_project_id, ProjectDataBindingCreate(
                asset_type="mixture_version", asset_version_id=version.id, purpose="training", alias=payload.name,
            ), session)
    add_audit(
        session,
        project_id=project_id,
        action="mixture.create",
        subject_type="data_mixture",
        subject_id=mixture.id,
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return mixture_view(session, mixture)


@router.post("/mixtures/{mixture_id}/versions", status_code=201)
def create_mixture_version(
    mixture_id: str, payload: MixtureVersionCreate, session: Session = Depends(get_session)
) -> dict:
    mixture = get_or_404(session, DataMixture, mixture_id, "数据混合")
    return create_mixture_version_for_actor(
        session,
        mixture=mixture,
        payload=payload,
        access_project_id=mixture.project_id,
        execution_project_id=mixture.project_id,
    )


def create_mixture_version_for_actor(
    session: Session,
    *,
    mixture: DataMixture,
    payload: MixtureVersionCreate,
    access_project_id: str,
    execution_project_id: str,
) -> dict:
    """Create a logical mixture version using an explicit catalog/owner actor."""
    normalized = _resolved_mixture_entries(session, access_project_id, payload.entries)
    latest = int(
        session.scalar(
            select(func.coalesce(func.max(DataMixtureVersion.version), 0)).where(
                DataMixtureVersion.mixture_id == mixture.id
            )
        )
        or 0
    )
    version = DataMixtureVersion(
        mixture_id=mixture.id,
        version=latest + 1,
        entries=normalized,
        target_tokens=payload.target_tokens,
        estimated_steps=payload.estimated_steps,
        distribution=payload.distribution,
        gate_result=payload.gate_result,
        locked=False,
        status="resolving",
        seed=payload.seed,
        resolver_version=RESOLVER_VERSION,
        sampler_version=SAMPLER_VERSION,
    )
    session.add(version)
    session.flush()
    _queue_mixture_resolution(
        session, project_id=execution_project_id, mixture=mixture, version=version
    )
    mixture.current_version_id = version.id
    session.commit()
    return version.to_dict()
