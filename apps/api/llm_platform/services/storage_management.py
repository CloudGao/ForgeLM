from __future__ import annotations

import hashlib
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import (
    AuditEvent,
    Project,
    StorageCleanupPlan,
    StorageCleanupPlanItem,
    StorageInventoryItem,
    StorageScan,
    Task,
    TaskAttempt,
    WorkerNode,
    utcnow,
)
from .project_storage import ProjectStorageError, local_workspace_path

TERMINAL_TASK_STATES = {"succeeded", "failed", "cancelled", "lost"}
TERMINAL_ATTEMPT_STATES = {"succeeded", "failed", "cancelled", "lost"}
PLATFORM_CATEGORIES = {
    "artifacts": "durable_artifact",
    "data-assets": "dataset_asset",
    "model-weights": "model_weight",
    "runs": "run_state",
    "training-runs": "training_run",
}
SCRATCH_DIRECTORY_NAMES = {
    "inputs",
    "mixture-shards",
    "corpus-index",
    "validation-corpus-index",
}
SCRATCH_FILE_NAMES = {
    "data-mixture-manifest.jsonl",
    "data-mixture-manifest.jsonl.gz",
}
PLAN_TTL = timedelta(minutes=10)
SCAN_PLAN_MAX_AGE = timedelta(minutes=30)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="storage-maintenance")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def inspect_path(path: Path) -> dict[str, Any]:
    """Return deterministic logical size and identity without following symlinks."""
    digest = hashlib.sha256()
    if path.is_symlink() or (
        path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400
    ):
        stat = path.lstat()
        digest.update(f"L|{stat.st_size}|{stat.st_mtime_ns}".encode())
        return {"size_bytes": 0, "file_count": 0, "fingerprint": digest.hexdigest()}
    path = path.resolve()
    size_bytes = 0
    file_count = 0
    if path.is_file():
        stat = path.stat()
        digest.update(f".|{stat.st_size}|{stat.st_mtime_ns}".encode())
        return {
            "size_bytes": int(stat.st_size),
            "file_count": 1,
            "fingerprint": digest.hexdigest(),
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    for current_root, directory_names, file_names in os.walk(path, followlinks=False):
        directory_names.sort()
        file_names.sort()
        root_path = Path(current_root)
        for directory_name in list(directory_names):
            child = root_path / directory_name
            if child.is_symlink() or getattr(child.lstat(), "st_file_attributes", 0) & 0x400:
                directory_names.remove(directory_name)
                digest.update(f"L|{child.relative_to(path).as_posix()}".encode())
        for file_name in file_names:
            child = root_path / file_name
            relative = child.relative_to(path).as_posix()
            try:
                stat = child.lstat()
            except FileNotFoundError:
                continue
            if (
                child.is_symlink()
                or getattr(stat, "st_file_attributes", 0) & 0x400
                or not child.is_file()
            ):
                digest.update(f"L|{relative}".encode())
                continue
            size_bytes += int(stat.st_size)
            file_count += 1
            digest.update(f"F|{relative}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return {
        "size_bytes": size_bytes,
        "file_count": file_count,
        "fingerprint": digest.hexdigest(),
    }


def _volume_view(workspace: Path) -> tuple[str, dict[str, Any]]:
    usage = shutil.disk_usage(workspace)
    device = int(workspace.stat().st_dev)
    total = int(usage.total)
    free = int(usage.free)
    reserved = max(int(total * 0.1), 50 * 1024**3)
    return f"device:{device}", {
        "id": f"device:{device}",
        "source": "control_plane",
        "root": str(workspace),
        "total_bytes": total,
        "used_bytes": total - free,
        "free_bytes": free,
        "reserved_bytes": reserved,
        "available_for_tasks_bytes": max(free - reserved, 0),
        "severity": "critical"
        if free <= reserved
        else "warning"
        if free <= reserved * 2
        else "healthy",
        "managed_bytes": 0,
        "reclaimable_bytes": 0,
        "project_ids": [],
    }


def _inventory_item(
    scan: StorageScan,
    project: Project,
    *,
    item_type: str,
    category: str,
    path: Path,
    stats: dict[str, Any],
    reclaimable: bool = False,
    reason: str = "",
    task_id: str | None = None,
    attempt_id: str | None = None,
) -> StorageInventoryItem:
    return StorageInventoryItem(
        scan_id=scan.id,
        project_id=project.id,
        item_type=item_type,
        category=category,
        path=str(path.absolute()),
        size_bytes=stats["size_bytes"],
        file_count=stats["file_count"],
        fingerprint=stats["fingerprint"],
        reclaimable=reclaimable,
        reason=reason,
        task_id=task_id,
        attempt_id=attempt_id,
        metadata_json={},
    )


def _scan_project(
    session: Session,
    scan: StorageScan,
    project: Project,
) -> tuple[dict[str, Any], list[StorageInventoryItem], str, dict[str, Any]]:
    workspace = local_workspace_path(project.workspace_uri)
    from llm_platform_core.storage_cleanup import checked_path

    from llm_platform_core.storage_layout import project_state_root
    from ..config import get_settings
    candidate_root = project_state_root(workspace, get_settings().storage_root)
    platform_root = checked_path(candidate_root, workspace.parent if candidate_root == workspace else workspace)
    volume_id, volume = _volume_view(workspace)
    categories: dict[str, int] = defaultdict(int)
    items: list[StorageInventoryItem] = []
    managed_bytes = 0
    reclaimable_bytes = 0
    if platform_root.is_dir():
        known_names = set(PLATFORM_CATEGORIES)
        for child in sorted(platform_root.iterdir(), key=lambda value: value.name):
            category = PLATFORM_CATEGORIES.get(child.name, "unknown")
            stats = inspect_path(child)
            categories[category] += stats["size_bytes"]
            managed_bytes += stats["size_bytes"]
            items.append(
                _inventory_item(
                    scan,
                    project,
                    item_type="category",
                    category=category,
                    path=child,
                    stats=stats,
                    reason=(
                        "持久内容受引用保护"
                        if child.name in known_names and category != "run_state"
                        else "运行目录只允许清理明确的 scratch 子目录"
                        if category == "run_state"
                        else "未知目录不自动清理"
                    ),
                )
            )

    historical = workspace / "projects"
    if historical.is_dir():
        stats = inspect_path(historical)
        categories["historical_workspace"] += stats["size_bytes"]
        managed_bytes += stats["size_bytes"]
        items.append(
            _inventory_item(
                scan,
                project,
                item_type="category",
                category="historical_workspace",
                path=historical,
                stats=stats,
                reason="历史布局按真实所有者识别，未知内容保留",
            )
        )
    # Discover exact cleanup objects across current and registered historical layouts.
    from .data_cleanup import discover

    for candidate in discover(session, project):
        items.append(
            StorageInventoryItem(
                scan_id=scan.id,
                project_id=project.id,
                item_type="cleanup_candidate",
                **candidate,
            )
        )
        if candidate["reclaimable"]:
            reclaimable_bytes += candidate["size_bytes"]
    project_view = {
        "project_id": project.id,
        "project_name": project.name,
        "workspace": str(workspace),
        "volume_id": volume_id,
        "managed_bytes": managed_bytes,
        "reclaimable_bytes": reclaimable_bytes,
        "categories": dict(categories),
        "scanned": True,
        "error": None,
    }
    volume["managed_bytes"] = managed_bytes
    volume["reclaimable_bytes"] = reclaimable_bytes
    volume["project_ids"] = [project.id]
    return project_view, items, volume_id, volume


def run_storage_scan(scan_id: str) -> None:
    with SessionLocal() as session:
        scan = session.get(StorageScan, scan_id)
        if scan is None:
            return
        scan.status = "running"
        scan.started_at = utcnow()
        session.commit()
        try:
            requested_ids = set((scan.scope or {}).get("project_ids") or [])
            statement = select(Project).order_by(Project.name)
            if requested_ids:
                statement = statement.where(Project.id.in_(requested_ids))
            projects = session.scalars(statement).all()
            project_views: list[dict[str, Any]] = []
            volumes: dict[str, dict[str, Any]] = {}
            seen_workspaces: set[str] = set()
            all_items: list[StorageInventoryItem] = []
            for project in projects:
                try:
                    workspace = local_workspace_path(project.workspace_uri)
                    workspace_key = str(workspace).casefold()
                    if workspace_key in seen_workspaces:
                        raise ProjectStorageError("项目工作区与另一个项目重复，已阻止重复计量")
                    seen_workspaces.add(workspace_key)
                    project_view, items, volume_id, volume = _scan_project(session, scan, project)
                    project_views.append(project_view)
                    all_items.extend(items)
                    if volume_id not in volumes:
                        volumes[volume_id] = volume
                    else:
                        volumes[volume_id]["managed_bytes"] += volume["managed_bytes"]
                        volumes[volume_id]["reclaimable_bytes"] += volume["reclaimable_bytes"]
                        volumes[volume_id]["project_ids"].append(project.id)
                except (OSError, ValueError, ProjectStorageError) as exc:
                    project_views.append(
                        {
                            "project_id": project.id,
                            "project_name": project.name,
                            "workspace": project.workspace_uri,
                            "managed_bytes": 0,
                            "reclaimable_bytes": 0,
                            "categories": {},
                            "scanned": False,
                            "error": str(exc),
                        }
                    )
            session.add_all(all_items)
            workers = session.scalars(select(WorkerNode).order_by(WorkerNode.name)).all()
            worker_views = []
            for worker in workers:
                workspace = (worker.resources or {}).get("workspace") or {}
                worker_views.append(
                    {
                        "worker_id": worker.id,
                        "worker_name": worker.name,
                        "status": worker.status,
                        "root": workspace.get("root"),
                        "total_bytes": int(workspace.get("total_bytes") or 0),
                        "free_bytes": int(workspace.get("free_bytes") or 0),
                        "scannable": False,
                        "last_heartbeat_at": (
                            worker.last_heartbeat_at.isoformat()
                            if worker.last_heartbeat_at
                            else None
                        ),
                    }
                )
            managed_bytes = sum(item["managed_bytes"] for item in project_views)
            reclaimable_bytes = sum(item["reclaimable_bytes"] for item in project_views)
            scan.summary = {
                "managed_bytes": managed_bytes,
                "reclaimable_bytes": reclaimable_bytes,
                "candidate_count": sum(item.reclaimable for item in all_items),
                "project_count": len(project_views),
                "unscanned_project_count": sum(not item["scanned"] for item in project_views),
                "projects": project_views,
                "volumes": list(volumes.values()),
                "workers": worker_views,
            }
            scan.status = "succeeded"
            scan.ended_at = utcnow()
            session.commit()
        except Exception as exc:
            session.rollback()
            failed = session.get(StorageScan, scan_id)
            if failed is not None:
                failed.status = "failed"
                failed.error = {"type": type(exc).__name__, "message": str(exc)}
                failed.ended_at = utcnow()
                session.commit()


def submit_storage_scan(session: Session, project_ids: list[str]) -> StorageScan:
    active = session.scalar(
        select(StorageScan).where(StorageScan.status.in_(["queued", "running"]))
    )
    if active is not None:
        return active
    scan = StorageScan(status="queued", scope={"project_ids": project_ids})
    session.add(scan)
    session.commit()
    _executor.submit(run_storage_scan, scan.id)
    return scan


def create_cleanup_plan(
    session: Session, scan: StorageScan, project_ids: list[str], item_ids: list[str] | None = None
) -> StorageCleanupPlan:
    if scan.status != "succeeded":
        raise ValueError("只能基于成功的存储扫描创建清理计划")
    ended_at = scan.ended_at
    if ended_at is None:
        raise ValueError("存储扫描没有完成时间")
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=UTC)
    if utcnow() - ended_at > SCAN_PLAN_MAX_AGE:
        raise ValueError("存储扫描已超过三十分钟，请重新扫描")
    statement = select(StorageInventoryItem).where(
        StorageInventoryItem.scan_id == scan.id,
        StorageInventoryItem.item_type == "cleanup_candidate",
        StorageInventoryItem.reclaimable.is_(True),
    )
    if project_ids:
        statement = statement.where(StorageInventoryItem.project_id.in_(project_ids))
    if item_ids is not None:
        if not item_ids or len(item_ids) != len(set(item_ids)):
            raise ValueError("请选择至少一个不重复的可清理候选")
        statement = statement.where(StorageInventoryItem.id.in_(item_ids))
    candidates = session.scalars(statement.order_by(StorageInventoryItem.size_bytes.desc())).all()
    if item_ids is not None and {item.id for item in candidates} != set(item_ids):
        raise ValueError("只能选择同一扫描和项目范围内的可清理候选")
    deleted_artifacts = {
        aid for item in candidates for aid in item.metadata_json.get("artifact_ids", [])
    }
    if any(
        (item.metadata_json.get("recovery_copy") or {}).get("artifact_id") in deleted_artifacts
        for item in candidates
    ):
        raise ValueError("本次选择包含中间恢复点及其最终恢复副本，请保留最终恢复副本")
    phrase = f"清理 {len(candidates)} 项"
    plan = StorageCleanupPlan(
        scan_id=scan.id,
        status="planned",
        confirmation_phrase=phrase,
        expires_at=utcnow() + PLAN_TTL,
        summary={
            "kind": "data_cleanup"
            if candidates
            and all(i.metadata_json.get("cleanup_kind") == "data_cleanup" for i in candidates)
            else "scratch",
            "item_count": len(candidates),
            "estimated_reclaimable_bytes": sum(item.size_bytes for item in candidates),
            "actual_reclaimed_bytes": 0,
            "deleted_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
        },
    )
    session.add(plan)
    session.flush()
    session.add_all(
        [
            StorageCleanupPlanItem(
                plan_id=plan.id,
                inventory_item_id=item.id,
                project_id=item.project_id,
                category=item.category,
                path=item.path,
                size_bytes=item.size_bytes,
                file_count=item.file_count,
                fingerprint=item.fingerprint,
                task_id=item.task_id,
                attempt_id=item.attempt_id,
                status="planned",
                reason=item.reason,
                metadata_json=item.metadata_json or {},
            )
            for item in candidates
        ]
    )
    session.add(
        AuditEvent(
            project_id=None,
            action="storage.cleanup.plan_created",
            subject_type="storage_cleanup_plan",
            subject_id=plan.id,
            payload=plan.summary,
        )
    )
    session.commit()
    return plan


def _skip(item: StorageCleanupPlanItem, code: str, message: str) -> None:
    item.status = "skipped"
    item.error = {"code": code, "message": message}


def _validate_cleanup_item(session: Session, item: StorageCleanupPlanItem) -> Path | None:
    project = session.get(Project, item.project_id) if item.project_id else None
    if project is None:
        _skip(item, "project_missing", "项目已不存在")
        return None
    task = session.get(Task, item.task_id) if item.task_id else None
    attempt = session.get(TaskAttempt, item.attempt_id) if item.attempt_id else None
    if (
        task is None
        or attempt is None
        or task.project_id != project.id
        or attempt.task_id != task.id
        or task.observed_state not in TERMINAL_TASK_STATES
        or attempt.status not in TERMINAL_ATTEMPT_STATES
    ):
        _skip(item, "attempt_not_terminal", "任务或 attempt 不再满足终态保护条件")
        return None
    try:
        workspace = local_workspace_path(project.workspace_uri)
    except ProjectStorageError as exc:
        _skip(item, "workspace_unavailable", str(exc))
        return None
    from llm_platform_core.storage_layout import project_state_root
    from ..config import get_settings
    run_root = (project_state_root(workspace, get_settings().storage_root) / "runs").resolve()
    attempt_root = (run_root / task.id / attempt.id).resolve()
    path = Path(item.path)
    from llm_platform_core.storage_cleanup import checked_path

    try:
        checked_path(path, workspace)
    except (OSError, ValueError) as exc:
        _skip(item, "symlink_rejected", str(exc))
        return None
    if path.is_symlink() or (
        path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400
    ):
        _skip(item, "symlink_rejected", "清理候选不能是符号链接")
        return None
    resolved = path.resolve()
    if not _is_relative_to(resolved, attempt_root) or resolved.parent != attempt_root:
        _skip(item, "path_boundary_changed", "候选不再位于精确 attempt scratch 边界")
        return None
    if resolved.name not in SCRATCH_DIRECTORY_NAMES | SCRATCH_FILE_NAMES:
        _skip(item, "unsupported_scratch_name", "候选名称不在可重建 scratch 白名单")
        return None
    if not resolved.exists():
        _skip(item, "already_missing", "候选已经不存在")
        return None
    stats = inspect_path(resolved)
    if stats["fingerprint"] != item.fingerprint or stats["size_bytes"] != item.size_bytes:
        _skip(item, "candidate_changed", "候选在扫描后发生变化，请重新扫描")
        return None
    return resolved


def run_cleanup_plan(plan_id: str) -> None:
    with SessionLocal() as session:
        plan = session.get(StorageCleanupPlan, plan_id)
        if plan is None or plan.status != "executing":
            return
        items = session.scalars(
            select(StorageCleanupPlanItem)
            .where(StorageCleanupPlanItem.plan_id == plan.id)
            .order_by(StorageCleanupPlanItem.created_at)
        ).all()
        for item in items:
            try:
                target = _validate_cleanup_item(session, item)
                if target is None:
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                item.status = "deleted"
                item.actual_reclaimed_bytes = item.size_bytes
            except Exception as exc:
                item.status = "failed"
                item.error = {"type": type(exc).__name__, "message": str(exc)}
        deleted = [item for item in items if item.status == "deleted"]
        skipped = [item for item in items if item.status == "skipped"]
        failed = [item for item in items if item.status == "failed"]
        plan.summary = {
            **(plan.summary or {}),
            "actual_reclaimed_bytes": sum(item.actual_reclaimed_bytes for item in deleted),
            "deleted_count": len(deleted),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
        }
        plan.status = "succeeded" if not skipped and not failed else "partial"
        plan.executed_at = utcnow()
        session.add(
            AuditEvent(
                project_id=None,
                action="storage.cleanup.executed",
                subject_type="storage_cleanup_plan",
                subject_id=plan.id,
                payload=plan.summary,
            )
        )
        session.commit()


def submit_cleanup_plan(plan: StorageCleanupPlan) -> None:
    if (plan.summary or {}).get("kind") == "data_cleanup":
        from .data_cleanup import run_plan

        _executor.submit(run_plan, plan.id)
    else:
        _executor.submit(run_cleanup_plan, plan.id)


def plan_is_expired(plan: StorageCleanupPlan) -> bool:
    expires_at = plan.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= utcnow()
