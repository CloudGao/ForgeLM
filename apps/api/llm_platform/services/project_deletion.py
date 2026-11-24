"""Reviewed, project-scoped deletion; never accept caller-supplied disk paths."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import HTTPException
from sqlalchemy import JSON, delete, or_, select, text, update
from sqlalchemy.orm import Session

from ..database import Base
from ..models import Artifact, ArtifactLocation, AuditEvent, Project, Task, TaskAttempt, WorkerNode
from .project_storage import local_workspace_path
from .storage_management import inspect_path

TERMINAL = {"succeeded", "failed", "cancelled", "lost"}
HISTORY = {
    "audit_events",
    "storage_inventory_items",
    "storage_cleanup_plan_items",
    "storage_scans",
    "storage_cleanup_plans",
    "outbox_events",
}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _owned(session: Session, project_id: str) -> dict[str, set]:
    tables = Base.metadata.tables
    owned = {"projects": {project_id}}
    changed = True
    while changed:
        changed = False
        for name, table in tables.items():
            if name in HISTORY or name in {"projects", "artifacts", "artifact_locations"}:
                continue
            clauses = []
            for fk in table.foreign_keys:
                parent = fk.column.table.name
                if fk.ondelete == "CASCADE" and owned.get(parent):
                    clauses.append(fk.parent.in_(owned[parent]))
            if not clauses:
                continue
            query = select(table.c.id).where(or_(*clauses))
            # A cross-project reference never transfers ownership.
            if "project_id" in table.c:
                query = query.where(table.c.project_id == project_id)
            ids = set(session.scalars(query))
            previous = owned.setdefault(name, set())
            if not ids <= previous:
                previous.update(ids)
                changed = True
    return {name: ids for name, ids in owned.items() if ids}


def _worker_path(raw: str, worker: WorkerNode) -> Path:
    labels = worker.labels or {}
    if os.name == "nt" and raw.startswith("/"):
        distro = str(labels.get("distro") or "")
        if (
            labels.get("location") != "local"
            or labels.get("environment") != "wsl"
            or worker.hostname.casefold() != platform.node().casefold()
            or not distro
            or any(c in distro for c in "/\\:")
        ):
            raise ValueError("Worker 目录无法由当前控制面安全访问")
        drive_path = re.fullmatch(r"/mnt/([a-zA-Z])/(.*)", raw)
        if drive_path:
            return Path(drive_path[1] + ":/" + drive_path[2])
        return Path("\\\\wsl.localhost\\" + distro + raw.replace("/", "\\"))
    if labels.get("location") != "local":
        raise ValueError("远程 Worker 工作目录须先通过其存储维护入口清理")
    return Path(raw)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def preview(session: Session, project: Project) -> dict:
    owned = _owned(session, project.id)
    blockers: list[dict] = []
    tables = Base.metadata.tables
    owned_ids = {str(i) for ids in owned.values() for i in ids if isinstance(i, str)}
    active = session.scalars(
        select(Task).where(Task.project_id == project.id, Task.observed_state.not_in(TERMINAL))
    ).all()
    blockers.extend({"kind": "active_task", "id": t.id} for t in active)
    for name in ("project_pipeline_runs", "task_attempts"):
        table = tables[name]
        ids = owned.get(name, set())
        if ids:
            for row in session.execute(
                select(table.c.id, table.c.status).where(table.c.id.in_(ids))
            ):
                allowed = TERMINAL | ({"paused"} if name == "task_attempts" else set())
                if row.status not in allowed:
                    blockers.append({"kind": "active_execution", "id": row.id})

    artifacts = session.scalars(select(Artifact)).all()
    candidate_ids = {
        a.id
        for a in artifacts
        if a.created_by_task_attempt_id in owned.get("task_attempts", set())
        or (a.metadata_json or {}).get("project_id") == project.id
    }
    for name, ids in owned.items():
        table = tables[name]
        for fk in table.foreign_keys:
            if fk.column.table.name == "artifacts":
                candidate_ids.update(
                    v for v in session.scalars(select(fk.parent).where(table.c.id.in_(ids))) if v
                )
    outside_artifact_refs: set[str] = set()
    # Frozen JSON manifests and polymorphic bindings are references too.
    # Event/audit history is retained as history, not a live consumer.
    for name, table in tables.items():
        if name in HISTORY or name in {"artifacts", "artifact_locations", "task_events"}:
            continue
        columns = [
            c for c in table.c if isinstance(c.type, JSON) or getattr(c.type, "length", None) == 36
        ]
        if not columns:
            continue
        query = select(table.c.id, *[c for c in columns if c.name != "id"])
        if owned.get(name):
            query = query.where(table.c.id.not_in(owned[name]))
        for row in session.execute(query).mappings():
            refs = set(_strings(dict(row)))
            hits = refs & owned_ids
            if hits:
                blockers.append(
                    {
                        "kind": "external_reference",
                        "table": name,
                        "id": row["id"],
                        "references": sorted(hits),
                    }
                )
            outside_artifact_refs.update(refs & candidate_ids)
    locations = session.scalars(select(ArtifactLocation)).all()
    for location in locations:
        if location.project_id and location.project_id != project.id:
            outside_artifact_refs.add(location.artifact_id)
    removable_artifacts = candidate_ids - outside_artifact_refs

    paths: set[Path] = set()
    if project.workspace_uri:
        paths.add(local_workspace_path(project.workspace_uri))
    for attempt in session.scalars(
        select(TaskAttempt).where(TaskAttempt.id.in_(owned.get("task_attempts", set())))
    ):
        worker = session.get(WorkerNode, attempt.worker_id) if attempt.worker_id else None
        raw = str((attempt.runtime_handle or {}).get("run_dir") or "")
        marker = f"/projects/{project.id}/"
        normalized = raw.replace("\\", "/")
        if worker and marker in normalized:
            root = normalized.split(marker, 1)[0] + marker.rstrip("/")
            try:
                paths.add(_worker_path(root, worker))
            except ValueError as exc:
                blockers.append(
                    {"kind": "unreachable_workspace", "worker_id": worker.id, "message": str(exc)}
                )
    path_rows = []
    for path in sorted(paths, key=str):
        try:
            linked = path.is_symlink() or (
                path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400
            )
        except OSError as exc:
            blockers.append(
                {"kind": "unreachable_workspace", "path": str(path), "message": str(exc)}
            )
            continue
        if linked:
            blockers.append({"kind": "linked_workspace", "path": str(path)})
            continue
        resolved = path.resolve()
        if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
            blockers.append({"kind": "unsafe_workspace", "path": str(resolved)})
            continue
        for other in session.scalars(select(Project).where(Project.id != project.id)):
            if other.workspace_uri:
                other_path = local_workspace_path(other.workspace_uri).resolve()
                if _within(other_path, resolved):
                    blockers.append(
                        {
                            "kind": "overlapping_workspace",
                            "project_id": other.id,
                            "path": str(resolved),
                        }
                    )
        # Shared artifacts inside this directory must prevent directory removal.
        for loc in locations:
            if loc.artifact_id in removable_artifacts:
                continue
            parsed = urlparse(loc.uri)
            if parsed.scheme != "file":
                continue
            try:
                worker = session.get(WorkerNode, loc.worker_id) if loc.worker_id else None
                location_path = (
                    _worker_path(unquote(parsed.path), worker)
                    if worker
                    and parsed.path.startswith("/")
                    and not (len(parsed.path) > 2 and parsed.path[2] == ":")
                    else local_workspace_path(loc.uri)
                )
                if _within(location_path.resolve(), resolved):
                    blockers.append(
                        {
                            "kind": "shared_artifact_path",
                            "artifact_id": loc.artifact_id,
                            "path": str(resolved),
                        }
                    )
            except (ValueError, OSError):
                continue
        path_rows.append(
            {
                "path": str(resolved),
                "exists": path.exists(),
                **(
                    inspect_path(path)
                    if path.exists()
                    else {"size_bytes": 0, "file_count": 0, "fingerprint": "missing"}
                ),
            }
        )
    facts = {
        "project_id": project.id,
        "project_name": project.name,
        "records": {k: sorted(v, key=str) for k, v in owned.items()},
        "artifact_ids": sorted(removable_artifacts),
        "paths": path_rows,
    }
    fingerprint = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return {
        **facts,
        "record_counts": {k: len(v) for k, v in owned.items()},
        "fingerprint": fingerprint,
        "blockers": blockers,
        "can_delete": not blockers,
    }


def execute(session: Session, project: Project, confirmation: str, fingerprint: str) -> dict:
    if confirmation != project.name:
        raise HTTPException(422, "请输入完整项目名称确认删除")
    # Serialize with task submission; recheck all references and file identities.
    session.execute(
        update(Project).where(Project.id == project.id).values(updated_at=Project.updated_at)
    )
    plan = preview(session, project)
    if plan["blockers"]:
        raise HTTPException(409, {"code": "project_deletion_blocked", "blockers": plan["blockers"]})
    if fingerprint != plan["fingerprint"]:
        raise HTTPException(409, {"code": "project_deletion_preview_stale"})
    # Validate the complete relational deletion before touching any files.
    session.execute(text("PRAGMA defer_foreign_keys=ON"))
    session.execute(delete(Project).where(Project.id == project.id))
    if plan["artifact_ids"]:
        session.execute(delete(Artifact).where(Artifact.id.in_(plan["artifact_ids"])))
    violations = session.execute(text("PRAGMA foreign_key_check")).fetchmany(1)
    if violations:
        session.rollback()
        raise HTTPException(409, {"code": "project_deletion_reference_remaining"})
    removed = []
    try:
        for item in plan["paths"]:
            path = Path(item["path"])
            if path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(item)
    except OSError as exc:
        session.rollback()
        raise HTTPException(
            409,
            {"code": "project_files_cleanup_failed", "message": str(exc), "removed_paths": removed},
        ) from exc
    receipt = {
        "deleted": True,
        "project_id": project.id,
        "project_name": project.name,
        "record_counts": plan["record_counts"],
        "artifact_ids": plan["artifact_ids"],
        "removed_paths": removed,
    }
    session.add(
        AuditEvent(
            project_id=None,
            action="project.delete",
            subject_type="project",
            subject_id=project.id,
            payload=receipt,
        )
    )
    session.commit()
    return receipt
