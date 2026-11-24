from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ArtifactLocation,
    Dataset,
    DatasetMaterialization,
    DatasetVersion,
    WorkerNode,
)
from ..schemas import DatasetMaterializationCreate
from ..services.dataset_materialization import (
    MaterializationRouteError,
    choose_materialization_route,
    materialization_routes,
    materialization_task_spec,
    materialization_view,
)
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(tags=["dataset-materialization"])


def _version_context(
    session: Session, dataset_version_id: str
) -> tuple[DatasetVersion, Dataset, Artifact]:
    version = get_or_404(session, DatasetVersion, dataset_version_id, "数据集版本")
    dataset = get_or_404(session, Dataset, version.dataset_id, "数据集")
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    if artifact is None or artifact.kind != "dataset_snapshot":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_snapshot_missing",
                "message": "该数据集版本还没有可物化的不可变快照",
            },
        )
    return version, dataset, artifact


@router.get("/dataset-versions/{dataset_version_id}/materialization-routes")
def preview_dataset_materialization_routes(
    dataset_version_id: str,
    target_worker_id: str = Query(),
    session: Session = Depends(get_session),
) -> dict:
    version, _dataset, artifact = _version_context(session, dataset_version_id)
    worker = get_or_404(session, WorkerNode, target_worker_id, "Worker")
    routes = materialization_routes(session, version, worker)
    return {
        "dataset_version_id": version.id,
        "materialization_kind": version.materialization_kind,
        "materialization_state": version.materialization_state,
        "artifact_id": artifact.id,
        "expected_content_hash": artifact.content_hash,
        "primary_location_id": version.primary_location_id,
        "target_worker_id": worker.id,
        "routes": routes,
    }


@router.get("/dataset-versions/{dataset_version_id}/materializations")
def list_dataset_materializations(
    dataset_version_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, DatasetVersion, dataset_version_id, "数据集版本")
    operations = session.scalars(
        select(DatasetMaterialization)
        .where(DatasetMaterialization.dataset_version_id == dataset_version_id)
        .order_by(DatasetMaterialization.created_at.desc())
    ).all()
    return [materialization_view(operation) for operation in operations]


@router.post("/dataset-versions/{dataset_version_id}/materializations", status_code=202)
def create_dataset_materialization(
    dataset_version_id: str,
    payload: DatasetMaterializationCreate,
    session: Session = Depends(get_session),
) -> dict:
    version, dataset, artifact = _version_context(session, dataset_version_id)
    worker = get_or_404(session, WorkerNode, payload.target_worker_id, "Worker")
    active = session.scalar(
        select(DatasetMaterialization).where(
            DatasetMaterialization.dataset_version_id == version.id,
            DatasetMaterialization.target_worker_id == worker.id,
            DatasetMaterialization.status.in_({"queued", "running"}),
        )
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail={"code": "materialization_active", "message": "目标 Worker 已有物化任务进行中"},
        )
    candidates = materialization_routes(session, version, worker)
    try:
        chosen = choose_materialization_route(
            candidates,
            selection_mode=payload.selection_mode,
            route_id=payload.route_id,
        )
    except MaterializationRouteError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "materialization_route_unavailable", "message": str(exc)},
        ) from exc
    operation = DatasetMaterialization(
        project_id=dataset.project_id,
        dataset_version_id=version.id,
        target_worker_id=worker.id,
        selection_mode=payload.selection_mode,
        requested_route_id=payload.route_id,
        candidate_routes=candidates,
        chosen_route=chosen,
        expected_content_hash=artifact.content_hash,
        status="queued",
    )
    session.add(operation)
    session.flush()
    if chosen["strategy"] == "already_local":
        location = get_or_404(
            session,
            ArtifactLocation,
            str(chosen.get("source_location_id") or ""),
            "数据副本位置",
        )
        operation.status = "succeeded"
        operation.result_location_id = location.id
        if not version.primary_location_id:
            version.primary_location_id = location.id
        version.materialization_state = "ready"
    else:
        try:
            adapter_id, task_payload = materialization_task_spec(operation, version, chosen)
        except MaterializationRouteError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "materialization_spec_unavailable", "message": str(exc)},
            ) from exc
        task = create_task(
            session,
            project_id=dataset.project_id,
            kind="dataset_materialize",
            title=f"同步数据：{dataset.name} V{version.version} → {worker.name}",
            adapter_id=adapter_id,
            launcher_id="builtin.local-process",
            payload=task_payload,
        )
        operation.task_id = task.id
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset_materialization.create",
        subject_type="dataset_materialization",
        subject_id=operation.id,
        payload={
            "dataset_version_id": version.id,
            "target_worker_id": worker.id,
            "route_id": chosen["route_id"],
            "selection_mode": payload.selection_mode,
        },
    )
    session.commit()
    return materialization_view(operation)


@router.get("/dataset-materializations/{materialization_id}")
def get_dataset_materialization(
    materialization_id: str, session: Session = Depends(get_session)
) -> dict:
    operation = get_or_404(session, DatasetMaterialization, materialization_id, "数据物化任务")
    return materialization_view(operation)
