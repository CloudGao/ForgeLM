"""Model tombstones and explicit, reference-aware physical deletion plans."""

from __future__ import annotations

import os
import socket
from collections import defaultdict
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from fastapi import HTTPException
from llm_platform_core.storage_cleanup import inspect_target, remove_target
from sqlalchemy import select, update

from .. import models as m
from .project_storage import local_workspace_path

TERMINAL = {"succeeded", "failed", "cancelled", "lost", "skipped"}
DERIVED = {
    "tokenized_corpus",
    "training_data_index",
    "supervision_corpus",
    "supervision_training_index",
}
RETAINED = {
    "dataset_snapshot",
    "dataset_record_manifest",
    "source_code",
    "tokenizer",
    "model_weights",
    "evaluation_report",
    "runtime_metrics",
}
PLAN_KIND = "model_deletion"
WORKER_CAPABILITY = "storage.model-deletion.v1"
DERIVED_RECORDS = (
    m.TokenizedCorpusVersion,
    m.TrainingDataViewVersion,
    m.SupervisionCorpusVersion,
    m.SupervisionTrainingViewVersion,
)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for entry in value.values():
            yield from strings(entry)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            yield from strings(entry)


def fail(code, message):
    raise HTTPException(409, {"code": code, "message": message})


def model_record(session, identity):
    model = session.get(m.ModelVersion, identity)
    if model is None:
        raise HTTPException(404, "模型版本不存在")
    return model


def local_path(uri, worker=None):
    if worker and worker.hostname.casefold() != socket.gethostname().casefold():
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    raw = unquote(parsed.path)
    if os.name == "nt":
        if raw.startswith("/mnt/") and len(raw) > 7 and raw[6] == "/":
            raw = raw[5].upper() + ":/" + raw[7:]
        elif len(raw) > 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        elif not (len(raw) > 2 and raw[1] == ":"):
            return None
    elif not raw.startswith("/") or (len(raw) > 2 and raw[2] == ":"):
        return None
    return Path(raw).absolute()


def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def _run(session, model):
    checkpoint = (
        session.get(m.Checkpoint, model.source_checkpoint_id)
        if model.source_checkpoint_id
        else None
    )
    return session.get(m.TrainingRun, checkpoint.training_run_id) if checkpoint else None


def _references(session, model, run):
    """Historical ancestry is not a live dependency; direct consumers remain protected."""
    protected = defaultdict(list)
    blockers = []
    project = session.get(m.Project, model.project_id)
    if model.is_official or project.current_model_version_id == model.id:
        blockers.append("当前正式模型：请先切换正式版本")
    checkpoints = list(session.scalars(select(m.Checkpoint)))
    own_checkpoints = {cp.id for cp in checkpoints if run and cp.training_run_id == run.id}
    sibling = False
    for other in session.scalars(
        select(m.ModelVersion).where(
            m.ModelVersion.id != model.id, m.ModelVersion.status != "deleted"
        )
    ):
        if other.artifact_id:
            protected[other.artifact_id].append(f"模型 {other.name} V{other.version} 正在引用")
        if other.source_checkpoint_id:
            protected[other.source_checkpoint_id].append(f"模型 V{other.version} 的来源恢复点")
            if other.source_checkpoint_id in own_checkpoints:
                sibling = True
    main = {model.id, model.artifact_id, model.source_checkpoint_id} - {None}
    for task in session.scalars(select(m.Task).where(~m.Task.observed_state.in_(TERMINAL))):
        if task.adapter_id == "builtin.storage-maintenance":
            continue
        refs = set(strings(task.payload)) | set(strings(task.result))
        for ref in refs:
            protected[ref].append(f"未完成任务：{task.title}")
        if refs & main or (run and task.id == run.task_id):
            blockers.append(f"未完成任务：{task.title}")
    if run and run.status not in TERMINAL:
        blockers.append(f"来源训练运行尚未结束（{run.status}）")
    for pipeline in session.scalars(
        select(m.ProjectPipelineRun).where(~m.ProjectPipelineRun.status.in_(TERMINAL))
    ):
        refs = set(strings(pipeline.to_dict()))
        for ref in refs:
            protected[ref].append("未完成训练流程仍有引用")
        if refs & main:
            blockers.append("未完成训练流程仍使用该模型")
    for lease in session.scalars(
        select(m.ResourceLease).where(m.ResourceLease.released_at.is_(None))
    ):
        for ref in strings(lease.allocation):
            protected[ref].append("活动读取租约")
    for job in session.scalars(
        select(m.QuantizationJob).where(
            m.QuantizationJob.model_version_id == model.id, m.QuantizationJob.enabled.is_(True)
        )
    ):
        blockers.append(f"已启用的量化任务：{job.name}")
    for job in session.scalars(
        select(m.ReleaseJob).where(
            m.ReleaseJob.source_kind == "model_version",
            m.ReleaseJob.source_id == model.id,
            m.ReleaseJob.enabled.is_(True),
        )
    ):
        blockers.append(f"已启用的发布任务：{job.name}")
    for release in session.scalars(select(m.Release).where(m.Release.model_version_id == model.id)):
        if release.status not in {"archived", "deleted", "failed"}:
            blockers.append("模型仍有发布登记，请先撤销发布")
    # Dataset/Tokenizer/code registrations always retain their physical payloads.
    # Completed evaluations and ancestry snapshots retain history, not file leases.
    for cls in (m.DatasetVersion, m.TokenizerVersion, m.CodeVersion, m.ModelWeightVersion):
        for record in session.scalars(select(cls)):
            for key, value in record.to_dict().items():
                if key.endswith("artifact_id") and value:
                    protected[value].append("原始/全局数据、Tokenizer、代码或外部权重登记保留")
    for cp in checkpoints:
        if cp.id in protected and cp.artifact_id:
            protected[cp.artifact_id].extend(protected[cp.id])
    for other in session.scalars(select(m.TrainingRun).where(~m.TrainingRun.status.in_(TERMINAL))):
        for ref in strings(other.config_snapshot):
            protected[ref].append("未完成训练运行依赖")
    for cls in DERIVED_RECORDS:
        for record in session.scalars(select(cls)):
            if record.id in protected and record.artifact_id:
                protected[record.artifact_id].extend(protected[record.id])
    return protected, list(dict.fromkeys(blockers)), sibling


def _derived_ids(session, run):
    if not run:
        return set(), {}
    refs = set(strings(run.config_snapshot))
    records = {}
    for cls in DERIVED_RECORDS:
        for record in session.scalars(select(cls)):
            records[record.id] = record.to_dict()
    return _expand_derived(refs, records), records


def _expand_derived(refs, records):
    # Resolve typed data-view/corpus references, never traverse into raw datasets.
    for _ in range(4):
        expanded = set(refs)
        for identity in refs & records.keys():
            expanded.update(strings(records[identity]))
        if expanded == refs:
            break
        refs = expanded
    return refs


def candidates(session, model, mode):
    run = _run(session, model)
    protected, blockers, sibling = _references(session, model, run)
    artifacts = {a.id: a for a in session.scalars(select(m.Artifact))}
    selected = {model.artifact_id} - {None}
    if mode == "cascade" and run:
        selected.update(
            cp.artifact_id
            for cp in session.scalars(
                select(m.Checkpoint).where(m.Checkpoint.training_run_id == run.id)
            )
            if cp.artifact_id
        )
        refs, derived_records = _derived_ids(session, run)
        selected.update(a.id for a in artifacts.values() if a.id in refs and a.kind in DERIVED)
        # A reusable view outside this run's graph can still need a selected corpus.
        for identity, record in derived_records.items():
            if identity in refs or record.get("status") == "missing":
                continue
            record_refs = _expand_derived(set(strings(record)), derived_records)
            for aid in selected & record_refs:
                if artifacts[aid].kind in DERIVED:
                    protected[aid].append("其他编码语料或训练视图仍引用该派生产物")
        for other in session.scalars(select(m.TrainingRun).where(m.TrainingRun.id != run.id)):
            other_refs, _ = _derived_ids(session, other)
            for identity in selected & other_refs:
                if artifacts[identity].kind in DERIVED:
                    protected[identity].append("其他训练运行共用的编码语料或索引")
        # Reusable plans and other typed views can hold data independently of a model.
        for revision in session.scalars(select(m.PlanRevision)):
            if revision.id == run.plan_revision_id:
                continue
            revision_refs = _expand_derived(set(strings(revision.to_dict())), derived_records)
            for identity in selected & revision_refs:
                if artifacts[identity].kind in DERIVED:
                    protected[identity].append("其他训练计划仍引用该派生产物")
    root = local_workspace_path(session.get(m.Project, model.project_id).workspace_uri)
    locations = list(session.scalars(select(m.ArtifactLocation)))
    all_paths = []
    remote_paths = []
    for ref, reasons in list(protected.items()):
        if ref.startswith("file://"):
            path = local_path(ref)
            if path:
                all_paths.append((path, reasons, None))
    outputs = []
    for artifact in artifacts.values():
        entries = [loc for loc in locations if loc.artifact_id == artifact.id]
        sources = [(loc.uri, loc) for loc in entries]
        if artifact.uri not in {uri for uri, _ in sources}:
            sources.append((artifact.uri, None))
        for uri, loc in sources:
            worker = session.get(m.WorkerNode, loc.worker_id) if loc and loc.worker_id else None
            path = local_path(uri, worker)
            reasons = list(protected.get(artifact.id, []))
            if loc:
                reasons.extend(protected.get(loc.id, []))
                if loc.project_id and loc.project_id != model.project_id:
                    reasons.append("该物理位置属于其他项目")
                if (loc.metadata_json or {}).get("retention") == "retain":
                    reasons.append("该位置已设为长期保留")
            if artifact.kind in RETAINED or (
                artifact.id == model.artifact_id and artifact.kind != "model"
            ):
                reasons.append("独立登记的原始/共享制品不随模型删除")
            if artifact.id not in selected:
                reasons.append("其他已登记制品")
            if path:
                all_paths.append((path, reasons, artifact.id))
            elif worker and urlparse(uri).scheme == "file":
                remote_paths.append(
                    (worker.id, PurePosixPath(unquote(urlparse(uri).path)), reasons)
                )
            if artifact.id not in selected:
                continue
            category = artifact.kind
            boundary = root
            remote = False
            remote_root = (
                ((worker.resources or {}).get("workspace") or {}).get("root") if worker else None
            )
            if (
                not path
                and worker
                and remote_root
                and urlparse(uri).scheme == "file"
                and not urlparse(uri).netloc
            ):
                remote_path = PurePosixPath(unquote(urlparse(uri).path))
                boundary = str(remote_root)
                if (
                    worker.status == "online"
                    and WORKER_CAPABILITY in (worker.capabilities or [])
                    and remote_path.is_relative_to(PurePosixPath(boundary))
                ):
                    path = remote_path
                    remote = True
            allowed = False
            if path:
                parts = path.parts
                for pos, part in enumerate(parts[:-1]):
                    if part == ".llm-platform" and pos + 1 < len(parts):
                        branch = parts[pos + 1]
                        allowed |= (
                            branch == "artifacts"
                            and pos + 2 < len(parts)
                            and parts[pos + 2] == artifact.kind
                        )
                        allowed |= branch == "data-assets" and artifact.kind in DERIVED
                        allowed |= (
                            branch == "training-runs"
                            and run is not None
                            and pos + 2 < len(parts)
                            and parts[pos + 2] == run.id
                        )
            if path and (
                not path.is_relative_to(PurePosixPath(str(boundary)) if remote else root)
                or not allowed
            ):
                reasons.append("不是项目平台托管目录，保留外部文件")
            if not path:
                reasons.append("位置不在控制面可访问的本机；需在所属 Worker 盘点回收")
            outputs.append(
                {
                    "path": str(path) if path else uri,
                    "root": str(boundary),
                    "category": category,
                    "artifact_ids": [artifact.id],
                    "location_ids": [loc.id] if loc else [],
                    "worker_id": worker.id if worker else None,
                    "worker_name": worker.name if worker else "控制面本机",
                    "reason": "；".join(dict.fromkeys(reasons)),
                    "reclaimable": not reasons,
                    "remote": remote,
                    "size_bytes": artifact.size_bytes,
                    "file_count": 0,
                    "fingerprint": "",
                }
            )
    if mode == "cascade" and run:
        # Historical roots are discovered from registered checkpoint/model locations,
        # plus the current project's two documented workspace layouts.
        run_roots = {
            root / ".llm-platform" / "training-runs" / run.id,
            root / "projects" / model.project_id / ".llm-platform" / "training-runs" / run.id,
            root
            / ".llm-platform"
            / "worker-workspaces"
            / "projects"
            / model.project_id
            / ".llm-platform"
            / "training-runs"
            / run.id,
        }
        for path, _, _ in all_paths:
            for parent in path.parents:
                if (
                    parent.name == run.id
                    and parent.parent.name == "training-runs"
                    and parent.is_relative_to(root)
                ):
                    run_roots.add(parent)
        for run_root in sorted(run_roots):
            if not run_root.is_dir():
                continue
            # Do not delete logs/configuration, nor arbitrary files in a run directory.
            targets = list((run_root / "trainer").glob("checkpoint-*"))
            if (run_root / "checkpoint-final").exists():
                targets.append(run_root / "checkpoint-final")
            for path in targets:
                reasons = ["同一次训练仍有其他模型版本，保留恢复状态"] if sibling else []
                outputs.append(
                    {
                        "path": str(path),
                        "root": str(root),
                        "category": "training_state",
                        "artifact_ids": [],
                        "location_ids": [],
                        "worker_id": None,
                        "worker_name": "控制面本机",
                        "reason": "；".join(reasons),
                        "reclaimable": not reasons,
                        "size_bytes": 0,
                        "file_count": 0,
                        "fingerprint": "",
                    }
                )
        for attempt in session.scalars(
            select(m.TaskAttempt).where(m.TaskAttempt.task_id == run.task_id)
        ):
            if attempt.status not in TERMINAL:
                continue
            for base in (
                root,
                root / "projects" / model.project_id,
                root / ".llm-platform" / "worker-workspaces" / "projects" / model.project_id,
            ):
                for name in ("inputs", "mixture-shards", "corpus-index", "validation-corpus-index"):
                    path = base / ".llm-platform" / "runs" / run.task_id / attempt.id / name
                    if path.exists():
                        outputs.append(
                            {
                                "path": str(path),
                                "root": str(root),
                                "category": "scratch",
                                "artifact_ids": [],
                                "location_ids": [],
                                "worker_id": None,
                                "worker_name": "控制面本机",
                                "reason": "",
                                "reclaimable": True,
                                "size_bytes": 0,
                                "file_count": 0,
                                "fingerprint": "",
                            }
                        )
    if mode == "cascade" and run:
        for source in list(outputs):
            if not source.get("remote"):
                continue
            run_root = next(
                (
                    parent
                    for parent in PurePosixPath(source["path"]).parents
                    if parent.name == run.id and parent.parent.name == "training-runs"
                ),
                None,
            )
            if run_root is None:
                continue
            for name in ("trainer", "checkpoint-final"):
                outputs.append(
                    {
                        **source,
                        "path": str(run_root / name),
                        "category": "training_state",
                        "artifact_ids": [],
                        "location_ids": [],
                        "size_bytes": 0,
                        "reclaimable": source["reclaimable"] and not sibling,
                        "reason": "同一次训练仍有其他模型版本，保留恢复状态"
                        if sibling
                        else source["reason"],
                        "checkpoint_children": name == "trainer",
                    }
                )
    # Directory overlap matters even when two distinct artifact IDs name nested files.
    for item in outputs:
        if not item["reclaimable"]:
            continue
        if item.get("remote"):
            remote_path = PurePosixPath(item["path"])
            for worker_id, other, reasons in remote_paths:
                if worker_id == item["worker_id"] and reasons and overlaps(remote_path, other):
                    item.update(
                        reclaimable=False,
                        reason="包含或重叠受保护的 Worker 制品：" + "；".join(reasons),
                    )
            continue
        path = Path(item["path"])
        conflict = next(
            (reasons for other, reasons, _ in all_paths if reasons and overlaps(path, other)), None
        )
        if conflict:
            item.update(reclaimable=False, reason="包含或重叠受保护位置：" + "；".join(conflict))
    # Merge identical locations and absorb children only under a deletable parent.
    unique = {}
    for item in sorted(outputs, key=lambda value: len(value["path"])):
        item.setdefault("remote", False)
        item.setdefault("checkpoint_children", False)
        key = (item["worker_id"] if item.get("remote") else None, item["path"])
        previous = unique.get(key)
        if previous:
            previous["artifact_ids"] = sorted(set(previous["artifact_ids"] + item["artifact_ids"]))
            previous["location_ids"] = sorted(set(previous["location_ids"] + item["location_ids"]))
            if not item["reclaimable"]:
                previous.update(reclaimable=False, reason=item["reason"])
            continue
        parent = next(
            (
                entry
                for entry in unique.values()
                if entry["reclaimable"]
                and item["reclaimable"]
                and entry.get("remote") == item.get("remote")
                and entry["worker_id"] == item["worker_id"]
                and Path(item["path"]).is_relative_to(Path(entry["path"]))
            ),
            None,
        )
        if parent:
            parent["artifact_ids"] += item["artifact_ids"]
            parent["location_ids"] += item["location_ids"]
        else:
            unique[key] = item
    return list(unique.values()), blockers


def _items(session, plan):
    return list(
        session.scalars(
            select(m.StorageCleanupPlanItem).where(m.StorageCleanupPlanItem.plan_id == plan.id)
        )
    )


def plan_view(session, plan):
    from .storage_management import plan_is_expired

    reconcile_worker_results(session, plan)
    if plan.status == "planned" and plan_is_expired(plan):
        plan.status = "expired"
        session.commit()
    return {**plan.to_dict(), "items": [item.to_dict() for item in _items(session, plan)]}


def preview(session, model, mode):
    if mode not in {"normal", "cascade"}:
        raise HTTPException(422, "不支持的删除模式")
    items, blockers = candidates(session, model, mode)
    scan = m.StorageScan(
        status="succeeded",
        scope={"kind": PLAN_KIND, "model_id": model.id},
        started_at=m.utcnow(),
        ended_at=m.utcnow(),
    )
    session.add(scan)
    session.flush()
    plan = m.StorageCleanupPlan(
        scan_id=scan.id,
        status="planned",
        expires_at=m.utcnow() + timedelta(minutes=10),
        confirmation_phrase=f"{'级联删除' if mode == 'cascade' else '删除'} V{model.version}",
        summary={
            "kind": PLAN_KIND,
            "model_id": model.id,
            "mode": mode,
            "model_name": model.name,
            "version": model.version,
            "blockers": blockers,
            "estimated_reclaimable_bytes": 0,
            "actual_reclaimed_bytes": 0,
        },
    )
    session.add(plan)
    session.flush()
    for item in items:
        if item["reclaimable"] and not item.get("remote"):
            try:
                item.update(inspect_target(item["path"], item["root"]))
            except (OSError, ValueError) as exc:
                item.update(reclaimable=False, reason=str(exc))
        session.add(
            m.StorageCleanupPlanItem(
                plan_id=plan.id,
                project_id=model.project_id,
                category=item["category"],
                path=item["path"],
                size_bytes=item["size_bytes"],
                file_count=item["file_count"],
                fingerprint=item["fingerprint"],
                status=("inspecting" if item.get("remote") else "planned")
                if item["reclaimable"]
                else "protected",
                reason=item["reason"] or "该版本独占的可回收产物",
                metadata_json={
                    **{
                        key: item[key]
                        for key in (
                            "root",
                            "artifact_ids",
                            "location_ids",
                            "worker_id",
                            "worker_name",
                        )
                    },
                    "remote": item.get("remote", False),
                    "checkpoint_children": item.get("checkpoint_children", False),
                },
            )
        )
    plan.summary = {
        **plan.summary,
        "item_count": sum(i["reclaimable"] for i in items),
        "estimated_reclaimable_bytes": sum(i["size_bytes"] for i in items if i["reclaimable"]),
    }
    session.add(
        m.AuditEvent(
            project_id=model.project_id,
            action="model.deletion.preview",
            subject_type="model_version",
            subject_id=model.id,
            payload={"plan_id": plan.id, **plan.summary},
        )
    )
    session.commit()
    if any(item.status == "inspecting" for item in _items(session, plan)):
        plan.status = "scanning"
        submit_worker_jobs(session, plan, "scan")
        session.commit()
    return plan_view(session, plan)


def get_plan(session, model_id, plan_id):
    plan = session.get(m.StorageCleanupPlan, plan_id)
    if (
        not plan
        or plan.summary.get("kind") != PLAN_KIND
        or plan.summary.get("model_id") != model_id
    ):
        raise HTTPException(404, "删除预览不存在")
    return plan


def execute(session, model, plan, confirmation):
    from .storage_management import _executor, plan_is_expired

    if plan.status in {"executing", "succeeded", "partial", "failed"}:
        return plan_view(session, plan)
    if plan.status != "planned" or plan_is_expired(plan):
        fail("deletion_plan_expired", "删除预览已过期，请重新预览")
    if confirmation != plan.confirmation_phrase:
        raise HTTPException(422, "确认文字不匹配")
    # Acquire the same DB write lock used by task submission before revalidating.
    session.execute(
        update(m.Project)
        .where(m.Project.id == model.project_id)
        .values(updated_at=m.Project.updated_at)
    )
    if session.scalar(
        select(m.StorageCleanupPlan.id).where(
            m.StorageCleanupPlan.status == "executing",
            m.StorageCleanupPlan.summary["kind"].as_string().in_({PLAN_KIND, "data_cleanup"}),
        )
    ):
        fail("model_deletion_busy", "另一项模型清理正在执行，请完成后重新预览")
    current, blockers = candidates(session, model, plan.summary["mode"])
    if blockers:
        raise HTTPException(
            409, {"code": "model_in_use", "message": "模型仍在使用", "blockers": blockers}
        )
    by_path = {
        (item["worker_id"] if item.get("remote") else None, item["path"]): item
        for item in current
        if item["reclaimable"]
    }
    for item in _items(session, plan):
        if item.status != "planned":
            continue
        candidate = by_path.get(
            (
                item.metadata_json["worker_id"] if item.metadata_json.get("remote") else None,
                item.path,
            )
        )
        if not candidate or sorted(candidate["artifact_ids"]) != sorted(
            item.metadata_json["artifact_ids"]
        ):
            fail("deletion_references_changed", "预览后引用发生变化，请重新预览")
        if item.metadata_json.get("remote"):
            continue
        try:
            stats = inspect_target(item.path, candidate["root"])
        except (OSError, ValueError) as exc:
            fail("deletion_path_changed", str(exc))
        if stats != {
            "size_bytes": item.size_bytes,
            "file_count": item.file_count,
            "fingerprint": item.fingerprint,
        }:
            fail("deletion_files_changed", "预览后文件发生变化，请重新预览")
    model.status = "deleted"
    model.is_best = model.is_candidate = model.is_official = False
    model.metadata_json = {
        **(model.metadata_json or {}),
        "deletion": {
            "plan_id": plan.id,
            "mode": plan.summary["mode"],
            "requested_at": m.utcnow().isoformat(),
        },
    }
    plan.status = "executing"
    for item in _items(session, plan):
        if item.status == "planned":
            for aid in item.metadata_json["artifact_ids"]:
                artifact = session.get(m.Artifact, aid)
                artifact.availability_state = "unknown"
                artifact.metadata_json = {**artifact.metadata_json, "deletion_plan_id": plan.id}
            for lid in item.metadata_json["location_ids"]:
                session.get(m.ArtifactLocation, lid).availability_state = "unknown"
    session.add(
        m.AuditEvent(
            project_id=model.project_id,
            action="model.deletion.requested",
            subject_type="model_version",
            subject_id=model.id,
            payload={"plan_id": plan.id, "mode": plan.summary["mode"]},
        )
    )
    session.commit()
    submit_worker_jobs(session, plan, "cleanup")
    session.commit()
    _executor.submit(run_plan, plan.id)
    return plan_view(session, plan)


def run_plan(plan_id):
    from ..database import SessionLocal

    with SessionLocal() as session:
        plan = session.get(m.StorageCleanupPlan, plan_id)
        if not plan or plan.status != "executing":
            return
        model = model_record(session, plan.summary["model_id"])
        for item in _items(session, plan):
            if item.status != "planned" or item.metadata_json.get("remote"):
                continue
            try:
                current, blockers = candidates(session, model, plan.summary["mode"])
                candidate = next(
                    (v for v in current if v["path"] == item.path and v["reclaimable"]), None
                )
                if blockers or candidate is None:
                    raise ValueError("执行前发现新的使用者或路径保护，已跳过")
                result = remove_target(
                    {
                        "path": item.path,
                        "root": item.metadata_json["root"],
                        "size_bytes": item.size_bytes,
                        "file_count": item.file_count,
                        "fingerprint": item.fingerprint,
                        "checkpoint_children": item.metadata_json.get("checkpoint_children", False),
                    }
                )
                item.status = result["status"]
                item.actual_reclaimed_bytes = result["actual_reclaimed_bytes"]
            except (OSError, ValueError) as exc:
                item.status = "failed"
                item.error = {"message": str(exc)}
            for aid in item.metadata_json["artifact_ids"]:
                artifact = session.get(m.Artifact, aid)
                # Successful sibling copies may still exist; recompute below.
                artifact.availability_state = "unknown"
            for lid in item.metadata_json["location_ids"]:
                loc = session.get(m.ArtifactLocation, lid)
                loc.availability_state = (
                    "missing" if item.status in {"deleted", "missing"} else "unknown"
                )
                loc.last_verified_at = None
                loc.metadata_json = {**loc.metadata_json, "deletion_plan_id": plan.id}
            session.commit()
        finish_plan(session, plan, model)


def finish_plan(session, plan, model):
    # Local and Worker completion can race in sessions that retain loaded objects.
    # Serialize finalization and reload items after flushing this executor's work.
    session.flush()
    claimed = session.execute(
        update(m.StorageCleanupPlan)
        .where(m.StorageCleanupPlan.id == plan.id, m.StorageCleanupPlan.status == "executing")
        .values(updated_at=m.StorageCleanupPlan.updated_at)
        .execution_options(synchronize_session=False)
    )
    if not claimed.rowcount:
        return
    session.refresh(plan)
    rows = list(
        session.scalars(
            select(m.StorageCleanupPlanItem)
            .where(m.StorageCleanupPlanItem.plan_id == plan.id)
            .execution_options(populate_existing=True)
        )
    )
    if any(row.status in {"planned", "inspecting", "queued"} for row in rows):
        return
    touched = {
        aid
        for row in rows
        if row.status != "protected"
        for aid in row.metadata_json["artifact_ids"]
    }
    for aid in touched:
        artifact = session.get(m.Artifact, aid)
        related = [row for row in rows if aid in row.metadata_json["artifact_ids"]]
        if all(row.status in {"deleted", "missing"} for row in related):
            artifact.availability_state = "missing"
            for cp in session.scalars(select(m.Checkpoint).where(m.Checkpoint.artifact_id == aid)):
                cp.status = "missing"
            for cls in DERIVED_RECORDS:
                for record in session.scalars(select(cls).where(cls.artifact_id == aid)):
                    record.status = "missing"
                    record.failure_reason = "模型版本级联删除已回收该派生产物；历史配置保留"
        elif session.scalar(
            select(m.ArtifactLocation.id)
            .where(
                m.ArtifactLocation.artifact_id == aid,
                m.ArtifactLocation.availability_state == "available",
            )
            .limit(1)
        ):
            artifact.availability_state = "available"
    plan.status = "partial" if any(row.status == "failed" for row in rows) else "succeeded"
    plan.executed_at = m.utcnow()
    plan.summary = {
        **plan.summary,
        "actual_reclaimed_bytes": sum(row.actual_reclaimed_bytes for row in rows),
        "deleted_count": sum(row.status == "deleted" for row in rows),
        "failed_count": sum(row.status == "failed" for row in rows),
        "protected_count": sum(row.status == "protected" for row in rows),
    }
    session.add(
        m.AuditEvent(
            project_id=model.project_id,
            action="model.deletion.completed",
            subject_type="model_version",
            subject_id=model.id,
            payload={"plan_id": plan.id, **plan.summary},
        )
    )
    session.commit()


def submit_worker_jobs(session, plan, action):
    from .tasks import create_task

    grouped = defaultdict(list)
    for item in _items(session, plan):
        if item.metadata_json.get("remote") and item.status == (
            "inspecting" if action == "scan" else "planned"
        ):
            grouped[item.metadata_json["worker_id"]].append(item)
    for worker_id, items in grouped.items():
        task = create_task(
            session,
            project_id=items[0].project_id,
            kind="storage_maintenance",
            title=f"模型版本{'盘点' if action == 'scan' else '删除回收'} V{plan.summary['version']}",
            adapter_id="builtin.storage-maintenance",
            payload={
                "model_deletion_plan_id": plan.id,
                "storage_action": action,
                "resource_request": {
                    "cpus": 0,
                    "gpus": 0,
                    "worker_selector": {
                        "worker_id": worker_id,
                        "capabilities": [WORKER_CAPABILITY],
                    },
                },
                "deletion_items": [
                    {
                        "item_id": item.id,
                        "path": item.path,
                        "size_bytes": item.size_bytes,
                        "file_count": item.file_count,
                        "fingerprint": item.fingerprint,
                        "checkpoint_children": item.metadata_json.get("checkpoint_children", False),
                    }
                    for item in items
                ],
            },
        )
        for item in items:
            item.metadata_json = {**item.metadata_json, "maintenance_task_id": task.id}
            if action == "cleanup":
                item.status = "queued"


def refresh_worker_payload(session, task, worker_id):
    plan = session.get(m.StorageCleanupPlan, task.payload["model_deletion_plan_id"])
    if not plan or plan.summary.get("kind") != PLAN_KIND:
        fail("invalid_deletion_job", "存储任务没有有效的模型删除计划")
    expected_status = "scanning" if task.payload.get("storage_action") == "scan" else "executing"
    if plan.status != expected_status:
        fail("invalid_deletion_job", "该模型删除计划已不再接受执行任务")
    model = model_record(session, plan.summary["model_id"])
    current, blockers = candidates(session, model, plan.summary["mode"])
    allowed = {
        entry["path"]
        for entry in current
        if entry["reclaimable"] and entry["worker_id"] == worker_id
    }
    payload = dict(task.payload)
    owned = {
        item.id: item
        for item in _items(session, plan)
        if item.metadata_json.get("maintenance_task_id") == task.id
        and item.metadata_json.get("worker_id") == worker_id
    }
    payload["deletion_items"] = [
        {
            "item_id": item.id,
            "path": item.path,
            "size_bytes": item.size_bytes,
            "file_count": item.file_count,
            "fingerprint": item.fingerprint,
            "checkpoint_children": item.metadata_json.get("checkpoint_children", False),
            "blocked": bool(blockers) or item.path not in allowed,
        }
        for item in owned.values()
    ]
    if not owned or task.project_id != model.project_id:
        fail("invalid_deletion_job", "存储任务与删除计划的所有权不匹配")
    task.payload = payload


def reconcile_worker_results(session, plan):
    if plan.status not in {"scanning", "executing"}:
        return
    changed = False
    for item in _items(session, plan):
        if item.status not in {"inspecting", "queued"}:
            continue
        task = session.get(m.Task, item.metadata_json.get("maintenance_task_id"))
        if (
            task
            and task.observed_state == "queued"
            and task.created_at.replace(tzinfo=None)
            < (m.utcnow() - timedelta(minutes=10)).replace(tzinfo=None)
        ):
            # Only unclaimed work can time out here. An executing Worker must finish
            # or pass through the existing attempt/lease recovery protocol.
            expired = session.execute(
                update(m.Task)
                .where(
                    m.Task.id == task.id,
                    m.Task.observed_state == "queued",
                    m.Task.created_at < m.utcnow() - timedelta(minutes=10),
                )
                .values(desired_state="cancelled")
                .execution_options(synchronize_session=False)
            )
            if expired.rowcount:
                from .tasks import transition_task

                task.desired_state = "cancelled"
                task.error = {"message": "Worker 十分钟内未领取任务，文件保留，请重新预览"}
                transition_task(session, task, "cancelled")
        if not task or task.observed_state not in TERMINAL:
            continue
        changed = True
        result = next(
            (
                entry
                for entry in (task.result or {}).get("items", [])
                if entry.get("item_id") == item.id
            ),
            None,
        )
        if task.observed_state != "succeeded" or not result or result.get("path") != item.path:
            result = {
                "status": "failed",
                "reason": (task.error or {}).get("message")
                or "Worker 任务未完成或未返回该文件的验证结果",
            }
        if plan.status == "scanning":
            if result.get("status") == "inspected":
                item.size_bytes = result["size_bytes"]
                item.file_count = result["file_count"]
                item.fingerprint = result["fingerprint"]
                item.status = "planned"
            else:
                item.status = "protected"
                item.reason = result.get("reason", "Worker 无法验证此位置")
        else:
            item.status = (
                result["status"] if result.get("status") in {"deleted", "missing"} else "failed"
            )
            item.actual_reclaimed_bytes = item.size_bytes if item.status == "deleted" else 0
            item.error = {"message": result.get("reason", "")} if item.status == "failed" else {}
            for lid in item.metadata_json["location_ids"]:
                location = session.get(m.ArtifactLocation, lid)
                location.availability_state = (
                    "missing" if item.status in {"deleted", "missing"} else "unknown"
                )
                location.last_verified_at = None
                location.metadata_json = {**location.metadata_json, "deletion_plan_id": plan.id}
    if changed:
        rows = _items(session, plan)
        if plan.status == "scanning" and all(item.status != "inspecting" for item in rows):
            plan.status = "planned"
            plan.expires_at = m.utcnow() + timedelta(minutes=10)
            plan.summary = {
                **plan.summary,
                "estimated_reclaimable_bytes": sum(
                    item.size_bytes for item in rows if item.status == "planned"
                ),
                "item_count": sum(item.status == "planned" for item in rows),
            }
        session.commit()
    if plan.status == "executing":
        finish_plan(session, plan, model_record(session, plan.summary["model_id"]))


def guard_task_inputs(session, payload):
    """All executable task adapters share the same deletion boundary."""
    if session.scalar(
        select(m.StorageCleanupPlan.id).where(
            m.StorageCleanupPlan.status == "executing",
            m.StorageCleanupPlan.summary["kind"].as_string().in_({PLAN_KIND, "data_cleanup"}),
        )
    ):
        fail("model_deletion_busy", "存储产物正在清理，完成后可提交新任务")
    refs = set(strings(payload))
    if not refs:
        return
    for model in session.scalars(select(m.ModelVersion).where(m.ModelVersion.status == "deleted")):
        if model.id in refs:
            fail("model_deleted", "任务引用了已删除的模型版本")
    for artifact in session.scalars(
        select(m.Artifact).where(m.Artifact.availability_state != "available")
    ):
        if artifact.id in refs and (artifact.metadata_json or {}).get("deletion_plan_id"):
            fail("model_artifact_deleted", "任务引用的模型或训练产物已清理")
    for checkpoint in session.scalars(select(m.Checkpoint).where(m.Checkpoint.status.in_({"missing", "unknown"}))):
        if checkpoint.id in refs:
            fail("checkpoint_deleted", "任务引用的 Checkpoint 已清理或可用性尚未验证")
    for cls in DERIVED_RECORDS:
        for record in session.scalars(select(cls).where(cls.id.in_(refs))):
            artifact = session.get(m.Artifact, record.artifact_id) if record.artifact_id else None
            if (
                artifact
                and artifact.availability_state != "available"
                and (artifact.metadata_json or {}).get("deletion_plan_id")
            ):
                fail("training_data_deleted", "任务引用的编码语料或训练视图已清理，请重新准备数据")


def recover_interrupted_plans():
    from ..database import SessionLocal

    with SessionLocal() as session:
        for plan in session.scalars(
            select(m.StorageCleanupPlan).where(
                m.StorageCleanupPlan.status == "executing",
                m.StorageCleanupPlan.summary["kind"].as_string() == PLAN_KIND,
            )
        ):
            finish_plan(session, plan, model_record(session, plan.summary["model_id"]))
            if plan.status != "executing":
                continue
            if any(item.status == "queued" for item in _items(session, plan)):
                # Remote jobs survive control-plane restarts; only interrupted local
                # items need manual reinspection. Their unknown locations stay fenced.
                for item in _items(session, plan):
                    if item.status == "planned":
                        item.status = "failed"
                        item.error = {"message": "本机回收被重启中断，请重新预览"}
                continue
            plan.status = "partial"
            plan.error = {"message": "服务重启中断了文件回收，请重新预览余留产物"}
            for item in _items(session, plan):
                if item.status == "planned":
                    item.status = "failed"
                    item.error = plan.error
        session.commit()


def reconcile_plans(session):
    for plan in list(
        session.scalars(
            select(m.StorageCleanupPlan).where(
                m.StorageCleanupPlan.status.in_({"scanning", "executing"}),
                m.StorageCleanupPlan.summary["kind"].as_string() == PLAN_KIND,
            )
        )
    ):
        reconcile_worker_results(session, plan)
