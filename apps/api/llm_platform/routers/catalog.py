from __future__ import annotations

from copy import deepcopy
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataMixture,
    Dataset,
    DatasetLabelSnapshot,
    DatasetVersion,
    EvalDatasetVersion,
    EvalSuiteVersion,
    EvaluationRun,
    Project,
    ProjectDataBinding,
    SupervisionDataset,
    SupervisionDatasetVersion,
    SupervisionMixture,
    Task,
)
from ..schemas import (
    BenchmarkAssetCreate,
    DatasetCreate,
    DatasetExplorerMixtureCreate,
    DatasetExplorerQuery,
    DatasetPreflightRequest,
    DatasetViewerQuery,
    EvalDatasetCreate,
    EvalDatasetFromDataVersion,
    EvalDatasetVersionCreate,
    EvaluationDatasetViewerQuery,
    HuggingFaceDownloadPlanRequest,
    HuggingFaceRepositoryScanRequest,
    MixtureCreate,
    MixtureVersionCreate,
    ProjectDataBindingCreate,
    ProjectDataBindingUpdate,
    SupervisionDatasetCreate,
    SupervisionDatasetVersionCreate,
    SupervisionMixtureCreate,
    SupervisionMixtureVersionCreate,
)
from ..services.artifacts import artifact_storage
from ..services.benchmarks import benchmark_by_id, canonical_hash, freeze_benchmark
from ..services.dataset_explorer import query_dataset_records
from ..services.evaluation_viewer import create_evaluation_query
from ..services.global_data import (
    ASSET_TYPES,
    asset_descriptor,
    binding_usage_blockers,
    ensure_global_data_project,
    is_global_data_project,
)
from ..services.tasks import create_task
from . import data, evaluations, post_training
from .common import add_audit, get_or_404

router = APIRouter(tags=["global-data-catalog"])


def _contains_eval_version_reference(value: object, version_id: str) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_eval_version_reference(key, version_id)
            or _contains_eval_version_reference(item, version_id)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_eval_version_reference(item, version_id) for item in value)
    return isinstance(value, str) and value == version_id


def _evaluation_dataset_references(
    session: Session, version: EvalDatasetVersion
) -> list[dict]:
    references: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, object_id: str, label: str, **extra: object) -> None:
        key = (kind, object_id)
        if key in seen:
            return
        seen.add(key)
        references.append(
            {"kind": kind, "id": object_id, "label": label, **extra}
        )

    for binding in session.scalars(
        select(ProjectDataBinding).where(
            ProjectDataBinding.asset_type == "eval_dataset_version",
            ProjectDataBinding.asset_version_id == version.id,
        )
    ).all():
        project = session.get(Project, binding.project_id)
        add(
            "project_binding",
            binding.id,
            project.name if project else binding.project_id,
            purpose=binding.purpose,
            project_id=binding.project_id,
        )

    for suite in session.scalars(select(EvalSuiteVersion)).all():
        if _contains_eval_version_reference(suite.datasets, version.id):
            project = session.get(Project, suite.project_id)
            add(
                "eval_suite_version",
                suite.id,
                f"{suite.name} · V{suite.version}",
                project_id=suite.project_id,
                project_name=project.name if project else "",
            )

    for run in session.scalars(select(EvaluationRun)).all():
        if _contains_eval_version_reference(run.config_snapshot, version.id):
            project = session.get(Project, run.project_id)
            add(
                "evaluation_run",
                run.id,
                f"评测运行 {run.id[:8]}",
                project_id=run.project_id,
                project_name=project.name if project else "",
                status=run.status,
            )

    for task in session.scalars(select(Task).where(Task.kind == "evaluate")).all():
        if _contains_eval_version_reference(task.payload, version.id):
            add(
                "evaluation_task",
                task.id,
                task.title,
                project_id=task.project_id,
                status=task.observed_state,
            )
    for task in session.scalars(select(Task).where(Task.kind == "data_query", Task.observed_state.notin_(["succeeded", "failed", "cancelled", "lost"]))):
        if (task.payload or {}).get("eval_dataset_version_id") == version.id:
            add("evaluation_query", task.id, task.title, project_id=task.project_id, status=task.observed_state)
    return references


def _evaluation_dataset_detail(
    session: Session, version: EvalDatasetVersion
) -> dict:
    owner = session.get(Project, version.project_id)
    versions = session.scalars(
        select(EvalDatasetVersion)
        .where(
            EvalDatasetVersion.project_id == version.project_id,
            EvalDatasetVersion.name == version.name,
        )
        .order_by(EvalDatasetVersion.version.desc())
    ).all()
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    references = _evaluation_dataset_references(session, version)
    return {
        "version": {
            **version.to_dict(),
            "origin_project_name": owner.name if owner else "",
            "global_managed": is_global_data_project(owner),
        },
        "versions": [item.to_dict() for item in versions],
        "artifact": (
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
                "media_type": artifact.media_type,
                "availability_state": artifact.availability_state,
            }
            if artifact is not None
            else None
        ),
        "references": references,
        "can_delete": not references,
        "delete_blockers": references,
    }


@router.get("/catalog/context")
def catalog_context(session: Session = Depends(get_session)) -> dict:
    project = ensure_global_data_project(session)
    session.commit()
    return {
        "catalog_project_id": project.id,
        "workspace_uri": project.workspace_uri,
        "asset_types": sorted(ASSET_TYPES),
    }


@router.get("/catalog/datasets")
def list_catalog_datasets(
    q: str = Query(default="", max_length=240),
    tag: list[str] = Query(default=[]),
    language: str = Query(default="", max_length=80),
    license: str = Query(default="", max_length=120),
    state: str = Query(default="", max_length=40),
    train_ready: bool | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[dict]:
    datasets = session.scalars(select(Dataset).order_by(Dataset.updated_at.desc())).all()
    needle = q.strip().casefold()
    required_tags = {value.strip().casefold() for value in tag if value.strip()}
    result: list[dict] = []
    for dataset in datasets:
        view = data.dataset_view(session, dataset)
        versions = list(view.get("versions") or [])
        current = view.get("current_version") or {}
        haystack = " ".join(
            [
                dataset.name,
                str(dataset.source_uri or ""),
                dataset.language,
                dataset.license,
                *(str(value) for value in dataset.tags or []),
            ]
        ).casefold()
        dataset_tags = {str(value).casefold() for value in dataset.tags or []}
        if needle and needle not in haystack:
            continue
        if required_tags and not required_tags.issubset(dataset_tags):
            continue
        if language and dataset.language.casefold() != language.casefold():
            continue
        if license and dataset.license.casefold() != license.casefold():
            continue
        if state and not any(str(item.get("state") or "") == state for item in versions):
            continue
        if train_ready is not None and bool(current.get("train_ready")) is not train_ready:
            continue
        owner = session.get(Project, dataset.project_id)
        result.append(
            {
                **view,
                "origin_project_name": owner.name if owner else "",
                "global_managed": is_global_data_project(owner),
            }
        )
    return result


@router.post("/catalog/datasets/preflight")
def preflight_catalog_dataset(
    payload: DatasetPreflightRequest,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return data.preflight_dataset_source(project.id, payload, session)


@router.post("/catalog/datasets/huggingface/scan", status_code=202)
def scan_catalog_huggingface_repository(
    payload: HuggingFaceRepositoryScanRequest,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return data.scan_huggingface_repository(project.id, payload, session)


@router.post("/catalog/datasets/huggingface/plans", status_code=202)
def plan_catalog_huggingface_download(
    payload: HuggingFaceDownloadPlanRequest,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return data.plan_huggingface_download(project.id, payload, session)


@router.post("/catalog/datasets", status_code=201)
def create_catalog_dataset(
    payload: DatasetCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return data.create_dataset(project.id, payload, session)


@router.get("/catalog/mixtures")
def list_catalog_mixtures(session: Session = Depends(get_session)) -> list[dict]:
    mixtures = session.scalars(
        select(DataMixture).order_by(DataMixture.updated_at.desc())
    ).all()
    result: list[dict] = []
    for mixture in mixtures:
        owner = session.get(Project, mixture.project_id)
        result.append(
            {
                **data.mixture_view(session, mixture),
                "origin_project_name": owner.name if owner else "",
                "global_managed": is_global_data_project(owner),
            }
        )
    return result


@router.post("/catalog/mixtures", status_code=201)
def create_catalog_mixture(
    payload: MixtureCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return data.create_mixture(project.id, payload, session)


@router.post("/catalog/mixtures/{mixture_id}/versions", status_code=201)
def create_catalog_mixture_version(
    mixture_id: str,
    payload: MixtureVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    mixture = get_or_404(session, DataMixture, mixture_id, "数据混合")
    project = ensure_global_data_project(session)
    return data.create_mixture_version_for_actor(
        session,
        mixture=mixture,
        payload=payload,
        access_project_id=project.id,
        execution_project_id=project.id,
    )


@router.post("/catalog/explorer/query")
def query_catalog_records(
    payload: DatasetExplorerQuery,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return query_dataset_records(
        session,
        access_project_id=project.id,
        payload=payload,
    )


@router.post("/catalog/explorer/mixtures", status_code=201)
def create_catalog_mixture_from_query(
    payload: DatasetExplorerMixtureCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    query_payload = DatasetExplorerQuery.model_validate(
        payload.model_dump(
            exclude={"name", "description", "target_tokens", "weights"}
        )
    )
    result = query_dataset_records(
        session,
        access_project_id=project.id,
        payload=query_payload,
    )
    if not result["materializable_as_mixture"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "query_not_materializable",
                "message": "当前混合选择器还不能冻结语言条件，请清除语言条件后创建混合集",
            },
        )
    if not result["sources"]:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_dataset_query", "message": "检索结果为空，无法构建混合集"},
        )
    entries = [
        {
            "dataset_version_id": source["dataset_version_id"],
            "name": source["dataset_name"],
            "weight": float(payload.weights.get(source["dataset_version_id"], 1)),
            "selection": result["selection"],
        }
        for source in result["sources"]
    ]
    if any(float(entry["weight"]) <= 0 for entry in entries):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_weights", "message": "检索结果中的每个数据源权重必须大于零"},
        )
    return data.create_mixture(
        project.id,
        MixtureCreate(
            name=payload.name,
            description=payload.description,
            entries=entries,
            target_tokens=payload.target_tokens,
            seed=payload.seed,
        ),
        session,
    )


@router.get("/catalog/supervision-datasets")
def list_catalog_supervision_datasets(
    session: Session = Depends(get_session),
) -> list[dict]:
    logicals = session.scalars(
        select(SupervisionDataset).order_by(SupervisionDataset.updated_at.desc())
    ).all()
    result: list[dict] = []
    for logical in logicals:
        owner = session.get(Project, logical.project_id)
        versions = session.scalars(
            select(SupervisionDatasetVersion)
            .where(SupervisionDatasetVersion.supervision_dataset_id == logical.id)
            .order_by(SupervisionDatasetVersion.version.desc())
        ).all()
        result.append(
            {
                **logical.to_dict(),
                "versions": [
                    post_training._supervision_version_view(session, version)
                    for version in versions
                ],
                "origin_project_name": owner.name if owner else "",
                "global_managed": is_global_data_project(owner),
            }
        )
    return result


@router.post("/catalog/supervision-datasets", status_code=201)
def create_catalog_supervision_dataset(
    payload: SupervisionDatasetCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return post_training.create_supervision_dataset(project.id, payload, session)


@router.post("/catalog/supervision-datasets/{dataset_id}/versions", status_code=201)
def create_catalog_supervision_dataset_version(
    dataset_id: str,
    payload: SupervisionDatasetVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    logical = get_or_404(session, SupervisionDataset, dataset_id, "后训练数据")
    project = ensure_global_data_project(session)
    version = post_training._create_supervision_dataset_version(
        session,
        logical,
        payload,
        access_project_id=project.id,
        execution_project_id=project.id,
    )
    session.commit()
    return post_training._supervision_version_view(session, version)


@router.get("/catalog/supervision-mixtures")
def list_catalog_supervision_mixtures(
    session: Session = Depends(get_session),
) -> list[dict]:
    mixtures = session.scalars(
        select(SupervisionMixture).order_by(SupervisionMixture.updated_at.desc())
    ).all()
    result: list[dict] = []
    for mixture in mixtures:
        owner = session.get(Project, mixture.project_id)
        result.append(
            {
                **post_training._mixture_view(session, mixture),
                "origin_project_name": owner.name if owner else "",
                "global_managed": is_global_data_project(owner),
            }
        )
    return result


@router.post("/catalog/supervision-mixtures", status_code=201)
def create_catalog_supervision_mixture(
    payload: SupervisionMixtureCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return post_training.create_supervision_mixture(project.id, payload, session)


@router.post("/catalog/supervision-mixtures/{mixture_id}/versions", status_code=201)
def create_catalog_supervision_mixture_version(
    mixture_id: str,
    payload: SupervisionMixtureVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    mixture = get_or_404(session, SupervisionMixture, mixture_id, "后训练数据混合")
    project = ensure_global_data_project(session)
    version = post_training._create_mixture_version(
        session,
        mixture,
        payload,
        access_project_id=project.id,
    )
    session.commit()
    return post_training._mixture_version_view(session, version)


@router.get("/catalog/evaluation-datasets")
def list_catalog_evaluation_datasets(
    session: Session = Depends(get_session),
) -> list[dict]:
    result: list[dict] = []
    for row in session.scalars(
        select(EvalDatasetVersion).order_by(EvalDatasetVersion.updated_at.desc())
    ).all():
        owner = session.get(Project, row.project_id)
        result.append(
            {
                **row.to_dict(),
                "origin_project_name": owner.name if owner else "",
                "global_managed": is_global_data_project(owner),
            }
        )
    return result


@router.post("/catalog/dataset-versions/{version_id}/viewer/query", status_code=202)
def query_catalog_dataset_version(
    version_id: str,
    payload: DatasetViewerQuery,
    session: Session = Depends(get_session),
) -> dict:
    version = get_or_404(session, DatasetVersion, version_id, "数据版本")
    dataset = get_or_404(session, Dataset, version.dataset_id, "数据集")
    if version.state != "ready":
        raise HTTPException(
            status_code=409,
            detail={"code": "dataset_not_ready", "message": "数据版本尚未就绪，不能查询"},
        )
    if not version.artifact_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "snapshot_required", "message": "数据版本没有冻结快照，不能查询"},
        )
    snapshot = get_or_404(session, Artifact, version.artifact_id, "数据快照")
    if snapshot.kind != "dataset_snapshot" or snapshot.availability_state != "available":
        raise HTTPException(
            status_code=409,
            detail={"code": "snapshot_unavailable", "message": "数据快照当前不可用"},
        )
    access = artifact_storage.inference_access(session, snapshot, dataset.project_id)
    if access["type"] == "unavailable":
        raise HTTPException(
            status_code=409,
            detail={"code": "snapshot_location_unavailable", "message": "没有 Worker 可以访问该数据快照"},
        )
    selector: dict[str, object] = {"capabilities": ["builtin.dataset-viewer"]}
    if access["type"] == "external" and not access["globally_accessible"]:
        worker_ids = list(access["worker_ids"])
        if not worker_ids:
            raise HTTPException(
                status_code=409,
                detail={"code": "snapshot_worker_unavailable", "message": "保存快照的 Worker 当前不可选"},
            )
        selector["worker_ids"] = worker_ids
    source_snapshot = version.source_snapshot if isinstance(version.source_snapshot, dict) else {}
    snapshot_files = source_snapshot.get("files") if isinstance(source_snapshot, dict) else None
    logical_files = [
        str(item.get("path"))
        for item in snapshot_files or []
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    record_manifest = (
        session.get(Artifact, version.record_manifest_artifact_id)
        if version.record_manifest_artifact_id
        else None
    )
    label_snapshot = session.scalar(
        select(DatasetLabelSnapshot)
        .where(DatasetLabelSnapshot.dataset_version_id == version.id)
        .order_by(DatasetLabelSnapshot.version.desc())
        .limit(1)
    )
    label_artifact = (
        session.get(Artifact, label_snapshot.artifact_id) if label_snapshot else None
    )
    quality_report = version.quality_report if isinstance(version.quality_report, dict) else {}
    task = create_task(
        session,
        project_id=dataset.project_id,
        kind="data_query",
        title=f"查询数据：{dataset.name} V{version.version}",
        adapter_id="builtin.dataset-viewer",
        launcher_id="builtin.local-process",
        payload={
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "dataset_version_id": version.id,
            "dataset_version": version.version,
            "dataset_viewer_input": {
                "artifact_id": snapshot.id,
                "artifact_hash": snapshot.content_hash,
                "logical_files": logical_files,
            },
            "record_manifest_input": (
                {
                    "artifact_id": record_manifest.id,
                    "artifact_hash": record_manifest.content_hash,
                }
                if record_manifest
                else None
            ),
            "label_snapshot_input": (
                {
                    "artifact_id": label_artifact.id,
                    "artifact_hash": label_artifact.content_hash,
                    "snapshot_id": label_snapshot.id,
                }
                if label_snapshot and label_artifact
                else None
            ),
            "label_snapshot_id": label_snapshot.id if label_snapshot else None,
            "parsed_files": quality_report.get("parsed_files") or [],
            "sql": payload.sql,
            "search": payload.search,
            "configuration": payload.configuration,
            "split": payload.split,
            "offset": payload.offset,
            "limit": payload.limit,
            "timeout_seconds": payload.timeout_seconds,
            "resource_request": {
                "gpus": 0,
                "cpus": 1,
                "worker_selector": selector,
            },
            "command": {},
        },
    )
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.viewer.query",
        subject_type="dataset_version",
        subject_id=version.id,
        payload={"task_id": task.id, "split": payload.split, "offset": payload.offset},
    )
    session.commit()
    return task.to_dict()


@router.post("/catalog/evaluation-datasets/{version_id}/viewer/query", status_code=202)
def query_evaluation_dataset(version_id: str, payload: EvaluationDatasetViewerQuery, session: Session = Depends(get_session)) -> dict:
    version = get_or_404(session, EvalDatasetVersion, version_id, "评测集版本")
    task = create_evaluation_query(session, project_id=version.project_id, name=version.name, payload=payload, version=version)
    add_audit(session, project_id=version.project_id, action="evaluation_dataset.viewer.query", subject_type="eval_dataset_version", subject_id=version.id, payload={"task_id": task.id})
    session.commit()
    return task.to_dict()


@router.post("/catalog/benchmarks/{benchmark_id}/viewer/query", status_code=202)
def query_builtin_evaluation_dataset(benchmark_id: str, payload: EvaluationDatasetViewerQuery, session: Session = Depends(get_session)) -> dict:
    if benchmark_by_id(benchmark_id) is None:
        raise HTTPException(404, detail={"code": "benchmark_not_found"})
    benchmark = freeze_benchmark(benchmark_id)
    project = ensure_global_data_project(session)
    task = create_evaluation_query(session, project_id=project.id, name=benchmark["name"], payload=payload, benchmark=benchmark)
    add_audit(session, project_id=project.id, action="evaluation_dataset.viewer.query", subject_type="benchmark", subject_id=benchmark_id, payload={"task_id": task.id})
    session.commit()
    return task.to_dict()


@router.get("/catalog/evaluation-datasets/{version_id}")
def get_catalog_evaluation_dataset(
    version_id: str,
    session: Session = Depends(get_session),
) -> dict:
    version = get_or_404(
        session, EvalDatasetVersion, version_id, "评测数据版本"
    )
    return _evaluation_dataset_detail(session, version)


@router.post("/catalog/evaluation-datasets/{version_id}/versions", status_code=201)
def create_catalog_evaluation_dataset_version(
    version_id: str,
    payload: EvalDatasetVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    source = get_or_404(
        session, EvalDatasetVersion, version_id, "评测数据版本"
    )
    if source.source_type == "benchmark":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "benchmark_definition_immutable",
                "message": "标准 Benchmark 定义由目录冻结；请取消登记后按新的目录定义重新登记",
            },
        )
    clean_task_type = payload.task_type.strip()
    clean_license = payload.license.strip()
    if not clean_task_type or not clean_license:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_evaluation_metadata",
                "message": "任务类型和许可证不能为空",
            },
        )
    schema_config = evaluations._validate_schema_config(payload.schema_config)
    artifact = session.get(Artifact, source.artifact_id) if source.artifact_id else None
    source_identity = (
        str((source.source_config or {}).get("content_hash") or "")
        or (artifact.content_hash if artifact else "")
        or str(source.compatibility_hash or source.id)
    )
    compatibility_hash = canonical_hash(
        {
            "source_identity": source_identity,
            "task_type": clean_task_type,
            "schema_config": schema_config,
            "scoring_config": payload.scoring_config,
        }
    )
    row = EvalDatasetVersion(
        project_id=source.project_id,
        name=source.name,
        version=evaluations._latest_version(
            session, EvalDatasetVersion, source.project_id, source.name
        )
        + 1,
        source_type=source.source_type,
        task_type=clean_task_type,
        license=clean_license,
        dimensions=[value.strip() for value in payload.dimensions if value.strip()],
        sample_count=source.sample_count,
        contamination_tags=[
            value.strip() for value in payload.contamination_tags if value.strip()
        ],
        source_config=deepcopy(source.source_config or {}),
        schema_config=deepcopy(schema_config),
        scoring_config=deepcopy(payload.scoring_config),
        benchmark_config=deepcopy(source.benchmark_config or {}),
        compatibility_hash=compatibility_hash,
        artifact_id=source.artifact_id,
    )
    session.add(row)
    session.flush()
    add_audit(
        session,
        project_id=source.project_id,
        action="evaluation_dataset.version_create",
        subject_type="eval_dataset_version",
        subject_id=row.id,
        payload={
            "source_version_id": source.id,
            "source_version": source.version,
            "version": row.version,
            "artifact_id": row.artifact_id,
        },
    )
    session.commit()
    return _evaluation_dataset_detail(session, row)


@router.delete("/catalog/evaluation-datasets/{version_id}", status_code=204)
def delete_catalog_evaluation_dataset(
    version_id: str,
    session: Session = Depends(get_session),
) -> None:
    version = get_or_404(
        session, EvalDatasetVersion, version_id, "评测数据版本"
    )
    references = _evaluation_dataset_references(session, version)
    if references:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "evaluation_dataset_version_in_use",
                "message": "该评测版本仍被项目绑定、评测套件或历史运行引用",
                "blockers": references,
            },
        )
    add_audit(
        session,
        project_id=version.project_id,
        action="evaluation_dataset.version_delete",
        subject_type="eval_dataset_version",
        subject_id=version.id,
        payload={
            "name": version.name,
            "version": version.version,
            "artifact_id": version.artifact_id,
        },
    )
    session.delete(version)
    session.commit()


@router.post("/catalog/evaluation-datasets", status_code=201)
def create_catalog_evaluation_dataset(
    payload: EvalDatasetCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return evaluations.create_eval_dataset(project.id, payload, session)


@router.post("/catalog/evaluation-datasets/from-benchmark/{benchmark_id}", status_code=201)
def create_catalog_evaluation_dataset_from_benchmark(
    benchmark_id: str,
    payload: BenchmarkAssetCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    if benchmark_by_id(benchmark_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "benchmark_not_found", "id": benchmark_id},
        )
    try:
        benchmark = freeze_benchmark(benchmark_id, payload.overrides)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_benchmark_override", "message": str(exc)},
        ) from exc
    existing = session.scalar(
        select(EvalDatasetVersion).where(
            EvalDatasetVersion.project_id == project.id,
            EvalDatasetVersion.source_type == "benchmark",
            EvalDatasetVersion.compatibility_hash == benchmark["config_hash"],
        )
    )
    if existing is not None:
        return existing.to_dict()
    latest = evaluations._latest_version(
        session, EvalDatasetVersion, project.id, str(benchmark["name"])
    )
    row = EvalDatasetVersion(
        project_id=project.id,
        name=str(benchmark["name"]),
        version=latest + 1,
        source_type="benchmark",
        task_type=str(benchmark["task_type"]),
        license=str(benchmark["license"]),
        dimensions=list(benchmark["dimensions"]),
        sample_count=0,
        contamination_tags=["public-benchmark"],
        benchmark_config=benchmark,
        compatibility_hash=str(benchmark["config_hash"]),
    )
    session.add(row)
    session.flush()
    add_audit(
        session,
        project_id=project.id,
        action="benchmark_asset.create",
        subject_type="eval_dataset_version",
        subject_id=row.id,
        payload={"benchmark_id": benchmark_id, "config_hash": benchmark["config_hash"]},
    )
    session.commit()
    return row.to_dict()


@router.post("/catalog/evaluation-datasets/upload", status_code=201)
def upload_catalog_evaluation_dataset(
    name: Annotated[str, Form(min_length=1, max_length=180)],
    task_type: Annotated[str, Form()],
    license: Annotated[str, Form()],
    dimensions: Annotated[str, Form()],
    contamination_tags: Annotated[str, Form()],
    schema_config: Annotated[str, Form()],
    scoring_config: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return evaluations.upload_eval_dataset(
        project.id,
        name,
        task_type,
        license,
        dimensions,
        contamination_tags,
        schema_config,
        scoring_config,
        file,
        session,
    )


@router.post("/catalog/evaluation-datasets/inspect-upload")
def inspect_catalog_evaluation_dataset_upload(
    schema_config: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> dict:
    payload = file.file.read(evaluations.MAX_EVALUATION_UPLOAD_BYTES + 1)
    if len(payload) > evaluations.MAX_EVALUATION_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "evaluation_file_too_large", "message": "评测集不能超过 256 MiB"},
        )
    parsed_schema = evaluations._validate_schema_config(
        evaluations._parse_json_form(schema_config, "字段映射", dict)
    )
    return evaluations._inspect_uploaded_evaluation(
        file.filename or "evaluation.jsonl",
        payload,
        parsed_schema,
    )


@router.post("/catalog/evaluation-datasets/from-data-version", status_code=201)
def create_catalog_evaluation_dataset_from_data_version(
    payload: EvalDatasetFromDataVersion,
    session: Session = Depends(get_session),
) -> dict:
    project = ensure_global_data_project(session)
    return evaluations.create_eval_dataset_from_data_version(project.id, payload, session)


@router.get("/projects/{project_id}/data-bindings")
def list_project_data_bindings(
    project_id: str,
    asset_type: str = Query(default="", max_length=50),
    purpose: str = Query(default="", max_length=50),
    session: Session = Depends(get_session),
) -> list[dict]:
    project = get_or_404(session, Project, project_id, "项目")
    if is_global_data_project(project):
        raise HTTPException(status_code=404, detail="项目不存在")
    query = select(ProjectDataBinding).where(ProjectDataBinding.project_id == project_id)
    if asset_type:
        query = query.where(ProjectDataBinding.asset_type == asset_type)
    if purpose:
        query = query.where(ProjectDataBinding.purpose == purpose)
    bindings = session.scalars(query.order_by(ProjectDataBinding.created_at.desc())).all()
    return [
        {
            **binding.to_dict(),
            "asset": asset_descriptor(
                session, binding.asset_type, binding.asset_version_id
            ),
        }
        for binding in bindings
    ]


@router.post("/projects/{project_id}/data-bindings", status_code=201)
def create_project_data_binding(
    project_id: str,
    payload: ProjectDataBindingCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = get_or_404(session, Project, project_id, "项目")
    if is_global_data_project(project):
        raise HTTPException(status_code=404, detail="项目不存在")
    descriptor = asset_descriptor(session, payload.asset_type, payload.asset_version_id)
    if descriptor is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_global_asset", "message": "全局数据资产版本不存在"},
        )
    binding = ProjectDataBinding(
        project_id=project_id,
        asset_type=payload.asset_type,
        asset_version_id=payload.asset_version_id,
        purpose=payload.purpose,
        alias=payload.alias,
        metadata_json=payload.metadata,
    )
    session.add(binding)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_data_binding", "message": "项目已按该用途使用这个资产版本"},
        ) from exc
    add_audit(
        session,
        project_id=project_id,
        action="project_data_binding.create",
        subject_type="project_data_binding",
        subject_id=binding.id,
        payload={
            "asset_type": binding.asset_type,
            "asset_version_id": binding.asset_version_id,
            "purpose": binding.purpose,
        },
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return {**binding.to_dict(), "asset": descriptor}


@router.patch("/projects/{project_id}/data-bindings/{binding_id}")
def update_project_data_binding(
    project_id: str,
    binding_id: str,
    payload: ProjectDataBindingUpdate,
    session: Session = Depends(get_session),
) -> dict:
    project = get_or_404(session, Project, project_id, "项目")
    if is_global_data_project(project):
        raise HTTPException(status_code=404, detail="项目不存在")
    binding = get_or_404(session, ProjectDataBinding, binding_id, "项目数据集")
    if binding.project_id != project_id:
        raise HTTPException(status_code=404, detail="项目数据集不存在")

    fields = payload.model_dump(exclude_unset=True)
    if any(value is None for value in fields.values()):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_project_dataset_update", "message": "用途和项目别名不能为 null"},
        )
    before = {"purpose": binding.purpose, "alias": binding.alias}
    for key, value in fields.items():
        setattr(binding, key, value)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_data_binding", "message": "项目已按该用途使用这个资产版本"},
        ) from exc
    after = {"purpose": binding.purpose, "alias": binding.alias}
    if before != after:
        add_audit(
            session,
            project_id=project_id,
            action="project_data_binding.update",
            subject_type="project_data_binding",
            subject_id=binding.id,
            payload={"before": before, "after": after},
        )
    session.commit()
    return {
        **binding.to_dict(),
        "asset": asset_descriptor(session, binding.asset_type, binding.asset_version_id),
    }


@router.delete("/projects/{project_id}/data-bindings/{binding_id}")
def delete_project_data_binding(
    project_id: str,
    binding_id: str,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    binding = get_or_404(session, ProjectDataBinding, binding_id, "项目数据集")
    if binding.project_id != project_id:
        raise HTTPException(status_code=404, detail="项目数据集不存在")
    blockers = binding_usage_blockers(session, binding)
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "data_binding_in_use",
                "message": "该项目数据集仍被项目制品、计划或任务引用，请先移除下游引用",
                "blockers": blockers,
            },
        )
    snapshot = binding.to_dict()
    session.delete(binding)
    add_audit(
        session,
        project_id=project_id,
        action="project_data_binding.delete",
        subject_type="project_data_binding",
        subject_id=binding_id,
        payload=snapshot,
    )
    session.commit()
    return {"deleted": True, "id": binding_id}
