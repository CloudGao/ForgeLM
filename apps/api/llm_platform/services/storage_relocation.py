"""Asset-aware relocation through the existing storage plan and audit lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import HTTPException
from llm_platform_core.storage_cleanup import (
    checked_path,
    filesystem_path,
    inspect_target,
    remove_target,
)
from llm_platform_core.storage_layout import (
    PUBLIC_DATA_KINDS,
    project_directory,
    unified_storage_root,
)
from sqlalchemy import select, update

from .. import models as m
from ..config import get_settings
from .model_deletion import local_path
from .project_deletion import _worker_path

KIND = "storage_relocation"
ACTIVE = {"assigned", "starting", "running", "stopping", "pausing"}


def root_path():
    root = unified_storage_root(get_settings().storage_root)
    if root is None:
        raise HTTPException(409, "请先配置统一存储根目录")
    return root


def active_plan(session):
    return session.scalar(
        select(m.StorageCleanupPlan.id)
        .where(
            m.StorageCleanupPlan.status.in_({"executing", "interrupted", "awaiting_cutover"}),
            m.StorageCleanupPlan.summary["kind"].as_string() == KIND,
        )
        .limit(1)
    )


def guard_writes(session):
    if active_plan(session):
        raise HTTPException(409, "存储迁移正在执行，请等待位置校验和切换完成")


def _idle(session):
    if session.scalar(select(m.Task.id).where(m.Task.observed_state.in_(ACTIVE)).limit(1)):
        raise ValueError("仍有执行中任务；须等待其安全停止")
    if session.scalar(
        select(m.ResourceLease.id).where(m.ResourceLease.released_at.is_(None)).limit(1)
    ):
        raise ValueError("仍有未释放的资源租约")


def location_path(session, location):
    worker = session.get(m.WorkerNode, location.worker_id) if location.worker_id else None
    if worker and (worker.labels or {}).get("location") != "local":
        raise ValueError("远程位置由节点保留，未做本机搬迁")
    path = local_path(location.uri, worker)
    if path is None and worker:
        path = _worker_path(unquote(urlparse(location.uri).path), worker)
    if path is None:
        raise ValueError("位置不是本机可访问的文件路径")
    return path.absolute()


def project_target(project):
    root = root_path()
    if project.system_kind == "global_data_catalog":
        return root / "platform" / "global-data-jobs"
    if project.system_kind:
        return root / "platform" / "packages" / project_directory(project.id, project.system_kind)
    old = Path(project.workspace_uri or "")
    if old.is_absolute() and old.is_relative_to(root / "projects"):
        return old
    return root / "projects" / project_directory(project.id, project.name)


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tree_hash(path):
    path = filesystem_path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p.relative_to(path)) if p != path else ".": _file_hash(p) for p in files}


def _uri_for_path(path, old_uri):
    # Existing local WSL worker ownership is preserved, with its native path syntax.
    raw = unquote(urlparse(old_uri).path)
    if os.name == "nt" and raw.startswith("/") and not (len(raw) > 2 and raw[2] == ":"):
        target = str(path).replace("\\", "/")
        if len(target) > 2 and target[1] == ":":
            return "file:///mnt/" + target[0].lower() + target[2:]
    return path.as_uri()


def _journal(source, target, old_uris, plan_id):
    path = root_path() / "platform" / "config" / "path-relocations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # Flatten earlier verified migrations when their destination moves again.
    for row in rows:
        previous = Path(row["target"])
        if previous.is_relative_to(source):
            row["target"] = str(target / previous.relative_to(source))
    names = {str(source).replace("\\", "/")}
    for uri in old_uris:
        raw = unquote(urlparse(uri).path)
        if raw.startswith("/") and not (len(raw) > 2 and raw[2] == ":"):
            names.add(raw)
    for name in names:
        row = {"source": name, "target": str(target), "plan_id": plan_id}
        rows = [r for r in rows if r["source"] != name] + [row]
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_catalog(session):
    """Human-readable inventory; the database remains the authoritative catalog."""
    datasets = {d.id: d for d in session.scalars(select(m.Dataset))}
    artifacts = {a.id: a for a in session.scalars(select(m.Artifact))}
    locations = {}
    for loc in session.scalars(select(m.ArtifactLocation)):
        locations.setdefault(loc.artifact_id, []).append(
            {"id": loc.id, "uri": loc.uri, "availability": loc.availability_state}
        )
    rows = []
    for version in session.scalars(select(m.DatasetVersion)):
        dataset = datasets.get(version.dataset_id)
        artifact = artifacts.get(version.artifact_id)
        rows.append(
            {
                "name": dataset.name if dataset else "",
                "dataset_id": version.dataset_id,
                "version_id": version.id,
                "version": version.version,
                "artifact_id": version.artifact_id,
                "content_hash": artifact.content_hash if artifact else None,
                "primary_location_id": version.primary_location_id,
                "locations": locations.get(version.artifact_id, []),
            }
        )
    directory = root_path() / "dataset" / "catalog"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".index.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "generated_at": m.utcnow().isoformat(),
                "description": "迁移完成时的目录盘点；实时版本、选择和血缘以平台数据中心为准。",
                "datasets": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(directory / "index.json")


def layout_view(session):
    root = root_path()
    return {
        "root": str(root),
        "directories": {
            "dataset": str(root / "dataset"),
            "projects": str(root / "projects"),
            "weights": str(root / "weights"),
            "platform": str(root / "platform"),
        },
        "database_url": get_settings().database_url,
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "current": p.workspace_uri,
                "target": str(project_target(p)),
            }
            for p in session.scalars(select(m.Project))
        ],
        "plans": [
            p.to_dict()
            for p in session.scalars(
                select(m.StorageCleanupPlan)
                .where(m.StorageCleanupPlan.summary["kind"].as_string() == KIND)
                .order_by(m.StorageCleanupPlan.created_at.desc())
                .limit(20)
            )
        ],
    }


def _validated_paused_training(session, task_ids, project_ids):
    if not isinstance(task_ids, list) or any(not isinstance(t, str) for t in task_ids):
        raise ValueError("暂停训练选择必须是任务 ID 列表")
    selected = set(task_ids)
    for task_id in selected:
        task = session.get(m.Task, task_id)
        if (task is None or task.kind != "train" or task.observed_state != "paused"
                or (project_ids and task.project_id not in project_ids)):
            raise ValueError("只能选择指定项目内的暂停训练任务")
        attempt = session.scalar(
            select(m.TaskAttempt).where(m.TaskAttempt.task_id == task_id)
            .order_by(m.TaskAttempt.attempt_number.desc()).limit(1)
        )
        if attempt is None or attempt.status != "paused" or attempt.ended_at is None:
            raise ValueError("暂停训练的最新执行必须已安全结束")
    return selected


def preview(session, scope="assets", project_ids=None, target_root=None, paused_training_task_ids=None):
    if paused_training_task_ids and scope != "projects":
        raise ValueError("只有项目迁移支持选择暂停训练任务")
    selected_paused = _validated_paused_training(session, paused_training_task_ids or [], project_ids)
    source_root = root_path()
    root = source_root
    if scope == "root":
        root = unified_storage_root(target_root)
        if not target_root or root is None or root == source_root:
            raise ValueError("整根迁移必须指定不同的绝对目标目录")
        if root.is_relative_to(source_root) or source_root.is_relative_to(root):
            raise ValueError("源和目标目录不得嵌套")
        if root.exists():
            raise ValueError("整根迁移目标必须尚不存在")
        if project_ids:
            raise ValueError("整根迁移不能筛选项目")
    elif target_root:
        raise ValueError("只有整根迁移支持目标根参数")
    _idle(session)
    guard_writes(session)
    projects = {p.id: p for p in session.scalars(select(m.Project))}
    locations = list(session.scalars(select(m.ArtifactLocation)))
    artifacts = {a.id: a for a in session.scalars(select(m.Artifact))}
    candidates, blocked, seen = [], [], set()
    targets = set()

    def add(source, target, metadata, project_id=None):
        source, target = source.absolute(), target.absolute()
        if source == target or not source.exists():
            return
        if str(source).casefold() in seen:
            return
        seen.add(str(source).casefold())
        try:
            checked_path(target, root)
            stats = inspect_target(source, source.parent)
            if target.exists() or str(target).casefold() in targets:
                raise ValueError("目标已有位置或本批有相同内容，保留重复位置供独立核验")
            targets.add(str(target).casefold())
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "stats": stats,
                    "metadata": metadata,
                    "project_id": project_id,
                }
            )
        except (OSError, ValueError) as exc:
            blocked.append({"source": str(source), "target": str(target), "reason": str(exc)})

    if scope == "root":
        deferred = {"db", "config", "logs", "control-plane-supervisor"}
        sources = [p for p in source_root.iterdir() if p.name != "platform"]
        platform = source_root / "platform"
        if platform.is_dir():
            sources.extend(p for p in platform.iterdir() if p.name not in deferred)
        for source in sources:
            locs, relatives = [], {}
            for loc in locations:
                if loc.availability_state != "available":
                    continue
                try:
                    physical = location_path(session, loc)
                except ValueError:
                    continue
                if physical.is_relative_to(source):
                    if not filesystem_path(physical).exists():
                        raise ValueError("登记位置不存在: " + loc.id)
                    locs.append(loc)
                    relatives[loc.id] = physical.relative_to(source).as_posix()
            add(
                source,
                root / source.relative_to(source_root),
                {
                    "operation": "root",
                    "location_ids": [loc.id for loc in locs],
                    "old_uris": [loc.uri for loc in locs],
                    "location_relative_paths": relatives,
                },
            )
        if blocked:
            raise ValueError("整根迁移预检未通过: " + str(blocked))
    elif scope == "assets":
        grouped = {}
        for loc in locations:
            if loc.availability_state != "available" or (
                project_ids and loc.project_id not in project_ids
            ):
                continue
            artifact = artifacts[loc.artifact_id]
            try:
                source = location_path(session, loc)
                if not source.exists():
                    raise ValueError("登记位置当前不存在，保留记录待修复")
            except (ValueError, OSError) as exc:
                blocked.append({"location_id": loc.id, "source": loc.uri, "reason": str(exc)})
                continue
            if source.is_relative_to(root):
                continue
            group = grouped.setdefault(str(source).casefold(), (source, artifact, []))
            if group[1].id != artifact.id:
                raise ValueError("同一路径登记为不同制品，须人工核对")
            group[2].append(loc)
        for source, artifact, locs in grouped.values():
            project = projects.get(locs[0].project_id)
            if artifact.kind in PUBLIC_DATA_KINDS:
                base = root / "dataset" / "objects"
            elif artifact.kind == "model_weights":
                base = root / "weights" / "objects"
            elif project:
                base = project_target(project) / "artifacts"
            else:
                blocked.append({"source": str(source), "reason": "制品缺少可核验的项目归属"})
                continue
            target = (
                base
                / artifact.kind
                / artifact.content_hash[:2]
                / artifact.content_hash
                / ("payload" if source.is_dir() else source.name)
            )
            add(
                source,
                target,
                {
                    "location_ids": [loc.id for loc in locs],
                    "artifact_id": artifact.id,
                    "old_uris": [loc.uri for loc in locs],
                    "expected_hash": artifact.content_hash,
                    "is_file": source.is_file(),
                    "operation": "asset",
                },
                locs[0].project_id,
            )
    elif scope == "projects":
        registered_ancestors = set()
        for loc in locations:
            if loc.availability_state != "available":
                continue
            try:
                registered = location_path(session, loc)
                registered_ancestors.update(
                    str(p).casefold() for p in (registered, *registered.parents)
                )
            except (OSError, ValueError):
                continue
        for project in projects.values():
            if project_ids and project.id not in project_ids:
                continue
            target = project_target(project)
            from .project_storage import workspace_history

            bases = {
                Path(uri).absolute() / ".llm-platform"
                for uri in workspace_history(session, project)
            }
            if project.workspace_uri:
                workspace = Path(project.workspace_uri).absolute()
                bases.add(workspace / ".llm-platform")
                # User-owned configs/exports are moved independently, never entire unknown workspaces.
                for name in (
                    "configs",
                    "exports",
                    "reports",
                    "evaluations",
                    "imports",
                    "README.md",
                    "preflight-result.json",
                ):
                    add(
                        workspace / name,
                        target / name,
                        {"operation": "runtime", "old_uris": []},
                        project.id,
                    )
            for host in projects.values():
                if host.workspace_uri:
                    for branch in (
                        Path(host.workspace_uri) / "projects" / project.id / ".llm-platform",
                        Path(host.workspace_uri)
                        / ".llm-platform/worker-workspaces/projects"
                        / project.id
                        / ".llm-platform",
                    ):
                        bases.add(branch.absolute())
            # Explicit task attempts discover additional local Worker roots without scanning arbitrary disks.
            for attempt, _task in session.execute(
                select(m.TaskAttempt, m.Task)
                .join(m.Task, m.Task.id == m.TaskAttempt.task_id)
                .where(m.Task.project_id == project.id)
            ):
                raw = str((attempt.runtime_handle or {}).get("run_dir") or "")
                marker = f"/projects/{project.id}/.llm-platform/"
                worker = session.get(m.WorkerNode, attempt.worker_id) if attempt.worker_id else None
                if worker and marker in raw:
                    try:
                        bases.add(
                            _worker_path(
                                raw.split(marker)[0] + marker.rstrip("/"), worker
                            ).absolute()
                        )
                    except ValueError:
                        pass
            paused = list(
                session.scalars(
                    select(m.Task).where(
                        m.Task.project_id == project.id, m.Task.observed_state == "paused",
                        m.Task.id.not_in(selected_paused)
                    )
                )
            )
            for base in sorted(bases, key=str):
                for branch in ("runs", "training-runs"):
                    directory = base / branch
                    if not directory.is_dir():
                        continue
                    for child in directory.iterdir():
                        if not child.is_dir():
                            continue
                        owner = (
                            session.get(m.Task, child.name)
                            if branch == "runs"
                            else session.get(m.TrainingRun, child.name)
                        )
                        if owner is None or owner.project_id != project.id:
                            blocked.append(
                                {"source": str(child), "reason": "目录没有匹配的项目执行记录"}
                            )
                            continue
                        overlaps_location = str(child).casefold() in registered_ancestors
                        if overlaps_location:
                            blocked.append(
                                {"source": str(child), "reason": "目录仍含已登记资产，须先迁移资产"}
                            )
                            continue
                        # Paused directories remain protected unless an ended training attempt was explicitly selected.
                        if (branch == "runs" and any(t.id == child.name for t in paused)) or (
                            branch == "training-runs"
                            and any(
                                (t.payload or {}).get("training_run_id") == child.name
                                for t in paused
                            )
                        ):
                            blocked.append(
                                {
                                    "source": str(child),
                                    "reason": "暂停任务恢复目录保留，等待恢复验证",
                                }
                            )
                            continue
                        if branch == "runs":
                            for attempt_dir in child.iterdir():
                                attempt = (
                                    session.get(m.TaskAttempt, attempt_dir.name)
                                    if attempt_dir.is_dir()
                                    else None
                                )
                                if (
                                    attempt_dir.name == "dataset"
                                    and attempt_dir.is_dir()
                                    and owner.kind in {"data_import", "dataset_import"}
                                ):
                                    add(
                                        attempt_dir,
                                        root / "dataset" / "staging" / "legacy-imports" / owner.id,
                                        {
                                            "operation": "legacy_import_copy",
                                            "old_uris": [],
                                            "task_id": owner.id,
                                        },
                                        project.id,
                                    )
                                elif attempt_dir.is_file() and attempt_dir.name.endswith(
                                    "-workflow-state.json"
                                ):
                                    add(
                                        attempt_dir,
                                        target / branch / child.name / attempt_dir.name,
                                        {
                                            "operation": "runtime",
                                            "old_uris": [],
                                            "task_id": owner.id,
                                        },
                                        project.id,
                                    )
                                elif attempt and attempt.task_id == owner.id:
                                    add(
                                        attempt_dir,
                                        target / branch / child.name / attempt_dir.name,
                                        {"operation": "runtime", "old_uris": []},
                                        project.id,
                                    )
                                else:
                                    blocked.append(
                                        {
                                            "source": str(attempt_dir),
                                            "reason": "未匹配具体执行记录，保留原位",
                                        }
                                    )
                        else:
                            add(
                                child,
                                target / branch / child.name,
                                {"operation": "runtime", "old_uris": []},
                                project.id,
                            )
            # Location references must be moved first; leftover unknown data never follows a workspace rename.
        # Workspace pointer changes happen only after each selected project's safe records are moved.
    elif scope == "platform":
        legacy = Path.cwd() / ".llm-platform"
        directories = {
            "backups": "backups",
            "runtime-wheel-cache": "cache/runtime-wheels",
            "ssh-credentials": "config/ssh-credentials",
            "deploy": "config/deploy",
            "previews": "cache/previews",
            "tools": "tools",
        }
        for name, destination in directories.items():
            source, target = legacy / name, root / "platform" / destination
            if source.is_dir() and target.is_dir():
                for child in source.iterdir():
                    add(child, target / child.name, {"operation": "platform", "old_uris": []})
            else:
                add(source, target, {"operation": "platform", "old_uris": []})
        add(
            legacy / "experimental-models",
            root / "weights" / "sources" / "legacy-experimental-models",
            {"operation": "legacy_shared_weights", "old_uris": []},
        )
        # Source files are provenance inputs, not new DatasetVersions. Keep their
        # identity/history and relocate only known source roots with database evidence.
        datasets = list(session.scalars(select(m.Dataset)))
        source_roots = [
            (legacy / "fact-sources", root / "dataset" / "sources" / "legacy-fact-sources")
        ]
        from .project_storage import workspace_history

        source_roots.extend(
            (Path(uri) / "datasets", root / "dataset" / "sources" / p.id)
            for p in projects.values()
            for uri in workspace_history(session, p)
        )
        for source, target in source_roots:
            prefix = str(source.absolute()).replace("\\", "/").casefold().rstrip("/")
            referenced = []
            for dataset in datasets:
                uri = str(dataset.source_uri or "").replace("\\", "/")
                if uri.startswith("/mnt/") and len(uri) > 7:
                    uri = uri[5] + ":/" + uri[7:]
                if uri.casefold().startswith(prefix + "/"):
                    referenced.append(dataset.id)
            if referenced:
                add(
                    source,
                    target,
                    {"operation": "source_inputs", "old_uris": [], "dataset_ids": referenced},
                )
    else:
        raise ValueError("不支持的迁移范围")
    # Move nested registrations together. Pulling a registered child out of a
    # registered directory would invalidate the parent's content identity.
    by_source = {str(c["source"]).casefold(): c for c in candidates}
    for child in candidates:
        ancestors = [
            by_source[str(p).casefold()]
            for p in child["source"].parents
            if str(p).casefold() in by_source
        ]
        if not ancestors:
            continue
        parent = ancestors[-1]
        if (
            parent["metadata"].get("operation") != "asset"
            or child["metadata"].get("operation") != "asset"
        ):
            raise ValueError("迁移目录相互嵌套，请分别预览")
        relative = child["source"].relative_to(parent["source"]).as_posix()
        parent["metadata"].setdefault("location_relative_paths", {}).update(
            {lid: relative for lid in child["metadata"]["location_ids"]}
        )
        parent["metadata"]["location_ids"].extend(child["metadata"]["location_ids"])
        parent["metadata"]["old_uris"].extend(child["metadata"]["old_uris"])
        child["nested"] = True
    candidates = [c for c in candidates if not c.get("nested")]
    scan = m.StorageScan(
        status="succeeded", scope={"kind": KIND, "scope": scope}, ended_at=m.utcnow()
    )
    session.add(scan)
    session.flush()
    plan = m.StorageCleanupPlan(
        scan_id=scan.id,
        status="planned",
        confirmation_phrase="迁移已核验的存储位置",
        summary={
            "kind": KIND,
            "scope": scope,
            "target_root": str(root),
            "source_root": str(source_root),
            "item_count": len(candidates),
            "bytes": sum(c["stats"]["size_bytes"] for c in candidates),
            "blocked": blocked,
            "project_ids": project_ids or list(projects),
            "paused_training_task_ids": sorted(selected_paused),
        },
        expires_at=m.utcnow() + timedelta(hours=2),
    )
    session.add(plan)
    session.flush()
    for c in candidates:
        session.add(
            m.StorageCleanupPlanItem(
                plan_id=plan.id,
                project_id=c["project_id"],
                category="relocation",
                path=str(c["source"]),
                **c["stats"],
                metadata_json={
                    **c["metadata"],
                    "destination": str(c["target"]),
                    "root": str(c["source"].parent),
                },
            )
        )
    session.add(
        m.AuditEvent(
            action="storage.relocation.preview",
            subject_type="storage_cleanup_plan",
            subject_id=plan.id,
            payload=plan.summary,
        )
    )
    session.commit()
    return plan


def execute(session, plan, confirmation):
    if plan.summary.get("kind") != KIND or plan.status not in {"planned", "interrupted"}:
        raise ValueError("迁移计划不可执行")
    if confirmation != plan.confirmation_phrase:
        raise ValueError("迁移确认文字不匹配")
    from .tasks import _utc

    if plan.status == "planned" and _utc(plan.expires_at) < m.utcnow():
        raise ValueError("迁移计划已过期")
    session.execute(update(m.Project).values(updated_at=m.Project.updated_at))
    if active_plan(session) not in {None, plan.id}:
        raise ValueError("其他迁移正在执行")
    _idle(session)
    _validated_paused_training(session, plan.summary.get("paused_training_task_ids", []), plan.summary.get("project_ids"))
    if session.scalar(
        select(m.StorageCleanupPlan.id).where(m.StorageCleanupPlan.status == "executing").limit(1)
    ):
        raise ValueError("其他存储计划正在执行")
    plan.status = "executing"
    session.commit()
    from .storage_management import _executor

    _executor.submit(run_plan, plan.id)


def recover_interrupted():
    from ..database import SessionLocal

    with SessionLocal() as session:
        cutover_completed = False
        for plan in session.scalars(
            select(m.StorageCleanupPlan).where(
                m.StorageCleanupPlan.status == "executing",
                m.StorageCleanupPlan.summary["kind"].as_string() == KIND,
            )
        ):
            plan.status = "interrupted"
        for plan in session.scalars(
            select(m.StorageCleanupPlan).where(
                m.StorageCleanupPlan.status == "awaiting_cutover",
                m.StorageCleanupPlan.summary["kind"].as_string() == KIND,
            )
        ):
            if Path(plan.summary["target_root"]) == root_path():
                plan.status = "succeeded"
                plan.summary = {**plan.summary, "cutover_at": m.utcnow().isoformat()}
                cutover_completed = True
        session.commit()
        if cutover_completed:
            write_catalog(session)


def _same_device(source, target):
    return source.stat().st_dev == target.parent.stat().st_dev


def _expected(item):
    return {k: getattr(item, k) for k in ("size_bytes", "file_count", "fingerprint")}


def _finish_registered(session, plan, item, source, target):
    if not target.exists():
        raise ValueError("已切换位置的目标不可用，保留迁移锁等待恢复")
    relatives = item.metadata_json.get("location_relative_paths", {})
    ids = item.metadata_json.get("location_ids", [])
    uris = item.metadata_json.get("old_uris", [])
    _journal(
        source,
        target,
        [uri for lid, uri in zip(ids, uris, strict=True) if lid not in relatives],
        plan.id,
    )
    for lid, uri in zip(ids, uris, strict=True):
        if lid in relatives:
            _journal(source / relatives[lid], target / relatives[lid], [uri], plan.id)
    if source.exists():
        if item.metadata_json["same_filesystem"]:
            raise ValueError("源位置在切换后被重新创建")
        verified = item.metadata_json.get("verified_target")
        source_state = inspect_target(source, source.parent)
        if verified:
            if inspect_target(target, target.parent) != verified:
                raise ValueError("校验后的目标发生变化，保留源文件")
            if source_state != _expected(item):
                # A previous deletion may have stopped partway through. Prove
                # every remaining file is still present unchanged in the verified copy.
                remaining = _tree_hash(source)
                target_files = _tree_hash(target)
                if any(target_files.get(name) != digest for name, digest in remaining.items()):
                    raise ValueError("校验后的源发生变化，保留源文件")
        elif _tree_hash(source) != _tree_hash(target):
            raise ValueError("切换后的副本不一致，保留源文件")
        remove_target({"path": str(source), "root": str(source.parent), **source_state})
    item.metadata_json = {**item.metadata_json, "phase": "complete"}
    item.error = {}
    session.commit()


def run_plan(plan_id):
    from ..database import SessionLocal

    with SessionLocal() as session:
        plan = session.get(m.StorageCleanupPlan, plan_id)
        if not plan or plan.status != "executing":
            return
        destination_root = Path(plan.summary["target_root"])
        items = list(
            session.scalars(
                select(m.StorageCleanupPlanItem).where(m.StorageCleanupPlanItem.plan_id == plan_id)
            )
        )
        interrupted = False
        for item in items:
            if item.metadata_json.get("phase") == "complete":
                continue
            source, target = Path(item.path), Path(item.metadata_json["destination"])
            try:
                _idle(session)
                checked_path(source, item.metadata_json["root"])
                checked_path(target, destination_root)
                if item.status == "moved":
                    _finish_registered(session, plan, item, source, target)
                    continue
                before = _expected(item)
                phase = item.metadata_json.get("phase")
                locations = [
                    session.get(m.ArtifactLocation, lid)
                    for lid in item.metadata_json.get("location_ids", [])
                ]
                if any(
                    not loc
                    or loc.availability_state != "available"
                    or location_path(session, loc)
                    != source
                    / item.metadata_json.get("location_relative_paths", {}).get(loc.id, "")
                    for loc in locations
                ):
                    raise ValueError("登记位置在预览后变化")
                if phase != "moving":
                    if inspect_target(source, source.parent) != before:
                        raise ValueError("源内容在预览后变化")
                    if target.exists():
                        raise ValueError("目标已存在，拒绝覆盖")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    same_device = _same_device(source, target)
                    if (
                        not same_device
                        and shutil.disk_usage(target.parent).free < item.size_bytes + 5 * 1024**3
                    ):
                        raise ValueError("目标空间不足以保留可回滚副本")
                    item.metadata_json = {
                        **item.metadata_json,
                        "phase": "moving",
                        "same_filesystem": same_device,
                    }
                    session.commit()
                same_device = item.metadata_json["same_filesystem"]
                if same_device:
                    if source.exists() and not target.exists():
                        if inspect_target(source, source.parent) != before:
                            raise ValueError("源内容在移动前变化")
                        source.rename(target)
                    if source.exists() or inspect_target(target, target.parent) != before:
                        raise ValueError("改名后文件身份变化或源目标同时存在")
                else:
                    if inspect_target(source, source.parent) != before:
                        raise ValueError("源内容在复制前变化")
                    expected = _tree_hash(source)
                    if not target.exists():
                        stage = target.parent / (".relocation-" + item.id)
                        checked_path(stage, destination_root)
                        if stage.exists():
                            remove_target(
                                {
                                    "path": str(stage),
                                    "root": str(destination_root),
                                    **inspect_target(stage, destination_root),
                                }
                            )
                        if source.is_dir():
                            shutil.copytree(
                                filesystem_path(source),
                                filesystem_path(stage),
                                copy_function=shutil.copy2,
                            )
                        else:
                            shutil.copy2(filesystem_path(source), filesystem_path(stage))
                        if _tree_hash(stage) != expected:
                            raise ValueError("复制校验失败，源文件保留")
                        stage.rename(target)
                    elif _tree_hash(target) != expected:
                        raise ValueError("既有目标内容校验失败，源文件保留")
                    if inspect_target(source, source.parent) != before:
                        raise ValueError("复制校验失败或源内容发生变化")
                    item.metadata_json = {
                        **item.metadata_json,
                        "verified_target": inspect_target(target, target.parent),
                    }
                _idle(session)
                for location in locations:
                    old = location.uri
                    location.uri = _uri_for_path(
                        target
                        / item.metadata_json.get("location_relative_paths", {}).get(
                            location.id, ""
                        ),
                        old,
                    )
                    location.metadata_json = {
                        **(location.metadata_json or {}),
                        "relocation_plan_id": plan.id,
                        "previous_uri": old,
                    }
                    artifact = session.get(m.Artifact, location.artifact_id)
                    if artifact.uri == old:
                        artifact.uri = location.uri
                item.status = "moved"
                item.metadata_json = {**item.metadata_json, "phase": "registered"}
                session.add(
                    m.AuditEvent(
                        project_id=item.project_id,
                        action="storage.location.relocated",
                        subject_type="storage_cleanup_plan_item",
                        subject_id=item.id,
                        payload={
                            "source": str(source),
                            "target": str(target),
                            "same_filesystem": same_device,
                            "bytes": item.size_bytes,
                        },
                    )
                )
                session.commit()
                _finish_registered(session, plan, item, source, target)
            except Exception as exc:
                session.rollback()
                item.error = {"message": str(exc)}
                if item.metadata_json.get("phase") in {"moving", "registered"}:
                    interrupted = True
                    session.commit()
                    break
                item.status = "failed"
                session.commit()
        if (
            not interrupted
            and plan.summary.get("scope") == "root"
            and all(i.status == "moved" for i in items)
        ):
            source_root = Path(plan.summary["source_root"])
            for project in session.scalars(select(m.Project)):
                previous = project.workspace_uri
                workspace = Path(previous or "")
                if not workspace.is_absolute() or not workspace.is_relative_to(source_root):
                    continue
                target = destination_root / workspace.relative_to(source_root)
                target.mkdir(parents=True, exist_ok=True)
                project.workspace_uri = str(target)
                session.add(
                    m.AuditEvent(
                        project_id=project.id,
                        action="project.workspace.relocated",
                        subject_type="project",
                        subject_id=project.id,
                        payload={
                            "previous_workspace": previous,
                            "workspace": str(target),
                            "plan_id": plan.id,
                        },
                    )
                )
            session.commit()
        if not interrupted and plan.summary.get("scope") == "projects":
            for pid in plan.summary["project_ids"]:
                project = session.get(m.Project, pid)
                if not project or any(i.project_id == pid and i.status != "moved" for i in items):
                    continue
                target = project_target(project)
                target.mkdir(parents=True, exist_ok=True)
                previous = project.workspace_uri
                project.workspace_uri = str(target)
                session.add(
                    m.AuditEvent(
                        project_id=pid,
                        action="project.workspace.relocated",
                        subject_type="project",
                        subject_id=pid,
                        payload={
                            "previous_workspace": previous,
                            "workspace": str(target),
                            "plan_id": plan.id,
                        },
                    )
                )
            session.commit()
        plan.summary = {
            **plan.summary,
            "moved_count": sum(i.status == "moved" for i in items),
            "moved_bytes": sum(i.size_bytes for i in items if i.status == "moved"),
            "failed_count": sum(i.status == "failed" for i in items),
        }
        plan.status = (
            "interrupted"
            if interrupted
            else "succeeded"
            if all(i.status == "moved" for i in items)
            else "partial"
        )
        plan.executed_at = m.utcnow()
        if plan.status == "succeeded" and plan.summary.get("scope") == "root":
            plan.status = "awaiting_cutover"
        session.commit()
        if not interrupted and plan.summary.get("scope") != "root":
            write_catalog(session)
