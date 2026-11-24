from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    StorageCleanupPlan,
    StorageCleanupPlanItem,
    StorageInventoryItem,
    StorageScan,
)
from ..schemas import (
    StorageCleanupExecute,
    StorageCleanupPlanCreate,
    StoragePolicyUpdate,
    StorageScanCreate,
    WorkerStorageJobCreate,
)
from ..services.storage_management import (
    create_cleanup_plan,
    plan_is_expired,
    submit_cleanup_plan,
    submit_storage_scan,
)
from .common import get_or_404


def create_worker_storage_job(project_id, payload, session):
    from ..models import Artifact, ArtifactLocation, Project, Task, TaskAttempt, WorkerNode
    from ..services.tasks import create_task
    from .common import add_audit

    get_or_404(session, Project, project_id, "项目")
    worker = get_or_404(session, WorkerNode, payload.worker_id, "Worker")
    if "builtin.storage-maintenance" not in (worker.capabilities or []):
        raise HTTPException(409, "该 Worker 需要升级后才能执行存储维护")
    locations = session.execute(select(ArtifactLocation, Artifact).join(
        Artifact, Artifact.id == ArtifactLocation.artifact_id
    ).where(ArtifactLocation.worker_id == worker.id)).all()
    owners = session.execute(select(TaskAttempt, Task).join(
        Task, Task.id == TaskAttempt.task_id
    ).where(TaskAttempt.worker_id == worker.id,
            Task.observed_state.in_({"succeeded", "failed", "cancelled", "lost"}),
            TaskAttempt.status.in_({"succeeded", "failed", "cancelled", "lost"}))).all()
    specification = {
        "storage_action": payload.action,
        "terminal_attempts": [{"project_id": t.project_id, "task_id": t.id,
                               "attempt_id": a.id} for a, t in owners],
        "locations": [{"id": loc.id, "uri": loc.uri, "kind": artifact.kind}
                      for loc, artifact in locations],
        "protected_uris": [loc.uri for loc, _ in locations],
        "resource_request": {"cpus": 0, "gpus": 0,
                             "worker_selector": {"worker_id": worker.id}},
    }
    from ..services.storage_lifecycle import generation_storage_payload
    specification.update(generation_storage_payload(session, project_id, worker.id))
    if payload.action == "cleanup":
        from datetime import timedelta

        from ..models import utcnow
        from ..services.tasks import _utc
        scan = get_or_404(session, Task, payload.scan_task_id or "", "盘点任务")
        if (scan.project_id != project_id or scan.adapter_id != "builtin.storage-maintenance"
                or scan.observed_state != "succeeded"
                or scan.result.get("storage_action") != "scan"
                or (scan.payload.get("resource_request", {}).get("worker_selector", {})
                    .get("worker_id") != worker.id)
                or not scan.ended_at or _utc(scan.ended_at) < utcnow() - timedelta(minutes=30)):
            raise HTTPException(409, "需要同 Worker 最近 30 分钟内成功的盘点")
        candidates = [item for item in scan.result.get("items", [])
                      if item.get("reclaimable") and item.get("id") in payload.item_ids]
        if not candidates or len(candidates) != len(set(payload.item_ids)):
            raise HTTPException(422, "只能选择盘点已确认的可回收项目")
        specification.update(candidates=candidates, scan_task_id=scan.id)
    task = create_task(session, project_id=project_id, kind="storage_maintenance",
                       title=f"存储{'盘点' if payload.action == 'scan' else '回收'}：{worker.name}",
                       adapter_id="builtin.storage-maintenance", payload=specification)
    add_audit(session, project_id=project_id, action=f"storage.worker.{payload.action}",
              subject_type="task", subject_id=task.id,
              payload={"worker_id": worker.id, "scan_task_id": payload.scan_task_id,
                       "item_ids": payload.item_ids})
    session.commit()
    return {**task.to_dict(), "task_id": task.id}

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/projects/{project_id}/policy")
def get_storage_policy(project_id: str, session: Session = Depends(get_session)):
    from ..models import Project
    project = get_or_404(session, Project, project_id, "项目")
    return StoragePolicyUpdate.model_validate(project.storage_policy or {}).model_dump()


@router.put("/projects/{project_id}/policy")
def set_storage_policy(project_id: str, payload: StoragePolicyUpdate,
                       session: Session = Depends(get_session)):
    from ..models import Project
    from .common import add_audit
    project = get_or_404(session, Project, project_id, "项目")
    project.storage_policy = payload.model_dump()
    add_audit(session, project_id=project_id, action="storage.policy.update",
              subject_type="project", subject_id=project_id, payload=project.storage_policy)
    session.commit()
    return project.storage_policy


@router.post("/projects/{project_id}/worker-jobs", status_code=202)
def worker_storage_job(project_id: str, payload: WorkerStorageJobCreate,
                       session: Session = Depends(get_session)):
    return create_worker_storage_job(project_id, payload, session)


@router.get("/worker-jobs")
def worker_storage_jobs(session: Session = Depends(get_session)):
    from ..models import Task
    return [task.to_dict() for task in session.scalars(select(Task).where(
        Task.adapter_id == "builtin.storage-maintenance").order_by(Task.created_at.desc()).limit(40))]


@router.put("/projects/{project_id}/locations/{location_id}/retention")
def set_location_retention(project_id: str, location_id: str, payload: dict,
                           session: Session = Depends(get_session)):
    from ..models import ArtifactLocation
    from .common import add_audit
    location = get_or_404(session, ArtifactLocation, location_id, "制品位置")
    if location.project_id != project_id:
        raise HTTPException(404, "制品位置不属于项目")
    retention = payload.get("retention")
    if retention not in {"auto", "after_consumers", "retain"}:
        raise HTTPException(422, "不支持的保留策略")
    if (location.metadata_json or {}).get("lifecycle_state") == "reclaiming":
        raise HTTPException(409, "该位置正在回收，暂时不能修改策略")
    location.metadata_json = {**(location.metadata_json or {}), "retention": retention}
    add_audit(session, project_id=project_id, action="storage.retention.update",
              subject_type="artifact_location", subject_id=location.id,
              payload={"retention": retention})
    session.commit()
    return location.to_dict()


def scan_view(session: Session, scan: StorageScan, *, include_items: bool = False) -> dict:
    result = scan.to_dict()
    if include_items:
        result["items"] = [
            item.to_dict()
            for item in session.scalars(
                select(StorageInventoryItem)
                .where(StorageInventoryItem.scan_id == scan.id)
                .order_by(
                    StorageInventoryItem.reclaimable.desc(),
                    StorageInventoryItem.size_bytes.desc(),
                )
            ).all()
        ]
    return result


def cleanup_plan_view(session: Session, plan: StorageCleanupPlan) -> dict:
    if plan.status == "planned" and plan_is_expired(plan):
        plan.status = "expired"
        session.commit()
    result = plan.to_dict()
    result["items"] = [
        item.to_dict()
        for item in session.scalars(
            select(StorageCleanupPlanItem)
            .where(StorageCleanupPlanItem.plan_id == plan.id)
            .order_by(StorageCleanupPlanItem.size_bytes.desc())
        ).all()
    ]
    return result


@router.get("/overview")
def storage_overview(session: Session = Depends(get_session)) -> dict:
    storage_scans = select(StorageScan).where(func.coalesce(StorageScan.scope["kind"].as_string(), "").not_in({"model_deletion", "storage_relocation"}))
    current_scan = session.scalar(storage_scans.order_by(StorageScan.created_at.desc()))
    successful_scan = session.scalar(
        storage_scans
        .where(StorageScan.status == "succeeded")
        .order_by(StorageScan.ended_at.desc())
    )
    plans = session.scalars(
        select(StorageCleanupPlan).where(func.coalesce(StorageCleanupPlan.summary["kind"].as_string(), "") != "storage_relocation").order_by(StorageCleanupPlan.created_at.desc()).limit(10)
    ).all()
    return {
        "current_scan": scan_view(session, current_scan) if current_scan else None,
        "latest_successful_scan": (
            scan_view(session, successful_scan, include_items=True)
            if successful_scan
            else None
        ),
        "cleanup_plans": [cleanup_plan_view(session, plan) for plan in plans],
        "cleanup_selection_supported": True,
    }


@router.post("/scans", status_code=status.HTTP_202_ACCEPTED)
def create_storage_scan(
    payload: StorageScanCreate, session: Session = Depends(get_session)
) -> dict:
    scan = submit_storage_scan(session, payload.project_ids)
    return scan_view(session, scan)


@router.get("/scans/{scan_id}")
def get_storage_scan(scan_id: str, session: Session = Depends(get_session)) -> dict:
    scan = get_or_404(session, StorageScan, scan_id, "存储扫描")
    return scan_view(session, scan, include_items=True)


@router.post("/cleanup-plans", status_code=status.HTTP_201_CREATED)
def create_storage_cleanup_plan(
    payload: StorageCleanupPlanCreate, session: Session = Depends(get_session)
) -> dict:
    scan = get_or_404(session, StorageScan, payload.scan_id, "存储扫描")
    try:
        plan = create_cleanup_plan(session, scan, payload.project_ids, payload.item_ids)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "cleanup_plan_unavailable", "message": str(exc)},
        ) from exc
    return cleanup_plan_view(session, plan)


@router.get("/cleanup-plans/{plan_id}")
def get_storage_cleanup_plan(
    plan_id: str, session: Session = Depends(get_session)
) -> dict:
    plan = get_or_404(session, StorageCleanupPlan, plan_id, "清理计划")
    return cleanup_plan_view(session, plan)


@router.post("/cleanup-plans/{plan_id}/execute", status_code=status.HTTP_202_ACCEPTED)
def execute_storage_cleanup_plan(
    plan_id: str,
    payload: StorageCleanupExecute,
    session: Session = Depends(get_session),
) -> dict:
    plan = get_or_404(session, StorageCleanupPlan, plan_id, "清理计划")
    if (plan.summary or {}).get("kind") in {"model_deletion", "storage_relocation"}:
        raise HTTPException(409, "请从模型版本删除预览执行该计划")
    if plan.status != "planned":
        raise HTTPException(
            status_code=409,
            detail={"code": "cleanup_plan_not_planned", "message": "清理计划不再可执行"},
        )
    if plan_is_expired(plan):
        plan.status = "expired"
        session.commit()
        raise HTTPException(
            status_code=409,
            detail={"code": "cleanup_plan_expired", "message": "清理计划已过期，请重新扫描"},
        )
    if payload.confirmation != plan.confirmation_phrase:
        raise HTTPException(
            status_code=422,
            detail={"code": "cleanup_confirmation_mismatch", "message": "确认文字不匹配"},
        )
    if int((plan.summary or {}).get("item_count") or 0) == 0:
        raise HTTPException(
            status_code=409,
            detail={"code": "cleanup_plan_empty", "message": "清理计划没有可执行项目"},
        )
    from ..models import Project
    # Serialize with task submissions, model deletion and other cleanup requests.
    session.execute(update(Project).values(updated_at=Project.updated_at))
    session.refresh(plan)
    if plan.status != "planned":
        raise HTTPException(409, "该清理计划已提交")
    if session.scalar(select(StorageCleanupPlan.id).where(StorageCleanupPlan.status == "executing")):
        raise HTTPException(409, "另一项存储清理正在执行")
    plan.status = "executing"
    session.commit()
    submit_cleanup_plan(plan)
    return cleanup_plan_view(session, plan)


@router.get("/layout")
def get_storage_layout(session: Session = Depends(get_session)):
    from ..services.storage_relocation import layout_view
    return layout_view(session)


@router.post("/relocations/preview", status_code=201)
def preview_storage_relocation(payload: dict, session: Session = Depends(get_session)):
    from ..services.storage_relocation import preview
    try:
        plan = preview(session, str(payload.get("scope", "assets")), payload.get("project_ids"), payload.get("target_root"), payload.get("paused_training_task_ids"))
        return cleanup_plan_view(session, plan)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/relocations/{plan_id}/execute", status_code=202)
def execute_storage_relocation(plan_id: str, payload: StorageCleanupExecute, session: Session = Depends(get_session)):
    from ..services.storage_relocation import execute
    plan = get_or_404(session, StorageCleanupPlan, plan_id, "迁移计划")
    try:
        execute(session, plan, payload.confirmation)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return cleanup_plan_view(session, plan)
