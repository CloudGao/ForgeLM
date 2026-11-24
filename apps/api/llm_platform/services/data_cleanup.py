"""Local storage cleanup derived from ownership, references and retained recovery copies."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from llm_platform_core.storage_cleanup import checked_path, inspect_target, remove_target
from sqlalchemy import select, update

from .. import models as m
from .model_deletion import TERMINAL, local_path, overlaps, strings
from .project_storage import ProjectStorageError, local_workspace_path

PLAN_KIND = "data_cleanup"
CATEGORIES = {
    "scratch": "终态任务缓存",
    "training_inputs": "终态训练输入副本",
    "cancelled_checkpoint": "取消或丢失训练的恢复点",
    "intermediate_checkpoint": "训练中间恢复点",
    "orphan_artifact": "已删除模型的遗留制品",
}


def _normal(value):
    return str(value).replace("\\", "/").casefold()


class Context:
    def __init__(self, session, proof_cache=None):
        self.session = session
        self.proof_cache = proof_cache if proof_cache is not None else {}
        self.artifacts = {a.id: a for a in session.scalars(select(m.Artifact))}
        self.locations = list(session.scalars(select(m.ArtifactLocation)))
        self.models = list(session.scalars(select(m.ModelVersion)))
        self.checkpoints = list(session.scalars(select(m.Checkpoint)))
        self.tasks = {t.id: t for t in session.scalars(select(m.Task))}
        self.attempts = list(session.scalars(select(m.TaskAttempt)))
        self.leases = list(
            session.scalars(select(m.ResourceLease).where(m.ResourceLease.released_at.is_(None)))
        )
        self.registered = []
        for entry in [*self.artifacts.values(), *self.locations]:
            path = local_path(entry.uri)
            if path:
                self.registered.append(
                    (path, entry.artifact_id if isinstance(entry, m.ArtifactLocation) else entry.id)
                )
        self.refs = set()
        for task in self.tasks.values():
            if task.observed_state not in TERMINAL:
                self.refs.update(strings(task.payload))
                self.refs.update(strings(task.result))
        for lease in self.leases:
            self.refs.update(strings(lease.allocation))
        for cls in (
            m.PlanRevision,
            m.ProjectPipelineRevision,
            m.ProjectPipelineInstance,
            m.Release,
            m.ReleaseJob,
            m.QuantizationJob,
        ):
            for row in session.scalars(select(cls)):
                self.refs.update(strings(row.to_dict()))
        for run in session.scalars(
            select(m.ProjectPipelineRun).where(~m.ProjectPipelineRun.status.in_(TERMINAL))
        ):
            self.refs.update(strings(run.to_dict()))
        self.global_artifacts = set()
        for cls in (
            m.DatasetVersion,
            m.TokenizerVersion,
            m.CodeVersion,
            m.ModelWeightVersion,
            m.SupervisionDatasetVersion,
            m.EvalDatasetVersion,
            m.TrainingValidationSet,
            m.TokenizedCorpusVersion,
            m.TrainingDataViewVersion,
            m.SupervisionCorpusVersion,
            m.SupervisionTrainingViewVersion,
        ):
            for record in session.scalars(select(cls)):
                for key, value in record.to_dict().items():
                    if key.endswith("artifact_id") and isinstance(value, str):
                        self.global_artifacts.add(value)
        self.ref_paths = [local_path(ref) for ref in self.refs if ref.startswith("file:")]
        self.ref_text = [_normal(ref) for ref in self.refs]

    def path_reason(self, path, ignored_artifacts=()):
        if any(
            aid not in ignored_artifacts and overlaps(path, other) for other, aid in self.registered
        ):
            return "路径与登记制品或共享副本重叠"
        if any(other and overlaps(path, other) for other in self.ref_paths):
            return "任务、流程、计划或读取租约引用此路径"
        normalized = _normal(path)
        # Also recognize WSL paths and paths embedded in command arguments.
        alternatives = [normalized]
        if len(normalized) > 2 and normalized[1] == ":":
            alternatives.append("/mnt/" + normalized[0] + normalized[2:])
        if any(value in ref for value in alternatives for ref in self.ref_text):
            return "任务或复用配置显式引用此路径"
        # Bare directory references can refer to any child checkpoint.
        for ref in self.refs:
            if ref.startswith(("/", "\\")) or (len(ref) > 2 and ref[1] == ":"):
                ref_path = local_path(
                    "file://" + ("/" if len(ref) > 2 and ref[1] == ":" else "") + ref
                )
                if ref_path and overlaps(path, ref_path):
                    return "任务或复用配置引用此目录"
        return ""

    def owner_reason(self, task, attempt=None):
        if not task or task.observed_state not in TERMINAL:
            return "所属任务尚未结束"
        attempts = [attempt] if attempt else [a for a in self.attempts if a.task_id == task.id]
        if not attempts:
            return "缺少可核验的执行记录"
        for owner in attempts:
            # A resumed task retains its closed paused attempts as history.
            # The task must be terminal; an open attempt or lease still blocks cleanup.
            ended_pause = owner.status == "paused" and owner.ended_at is not None
            if owner.status not in TERMINAL and not ended_pause:
                return "所属执行记录尚未结束"
            if any(lease.task_attempt_id == owner.id for lease in self.leases):
                return "所属执行记录仍有资源租约"
        return ""

    def cache_reason(self, task, project, attempt=None):
        reason = self.owner_reason(task, attempt)
        if reason:
            return reason
        if (task.payload.get("storage_policy") or {}).get("cache_cleanup") == "retain":
            return "任务缓存设置为长期保留"
        if (project.storage_policy or {}).get("cache_cleanup") == "retain":
            return "项目缓存设置为长期保留"
        return ""

    def final_checkpoint(self, run):
        for cp in self.checkpoints:
            if cp.training_run_id != run.id or cp.status != "available":
                continue
            artifact = self.artifacts.get(cp.artifact_id)
            if (
                not artifact
                or artifact.kind != "checkpoint"
                or artifact.availability_state != "available"
            ):
                continue
            # The final archive must be produced by this run, not merely a row alias.
            source = _normal((artifact.metadata_json or {}).get("source_path", ""))
            if f"/training-runs/{run.id}/checkpoint-final" not in source or not source.endswith(
                "/checkpoint-final"
            ):
                continue
            path = local_path(artifact.uri)
            if not path or not path.is_file():
                continue
            try:
                stats = inspect_target(path, path.parent)
                key = (artifact.id, artifact.content_hash, stats["fingerprint"])
                if key not in self.proof_cache:
                    with path.open("rb") as stream:
                        self.proof_cache[key] = (
                            hashlib.file_digest(stream, "sha256").hexdigest()
                            == artifact.content_hash
                        )
                if (
                    self.proof_cache[key]
                    and stats["size_bytes"] == artifact.size_bytes
                    and inspect_target(path, path.parent) == stats
                ):
                    return {
                        "artifact_id": artifact.id,
                        "path": str(path),
                        "content_hash": artifact.content_hash,
                    }
            except (OSError, ValueError):
                continue
        return None


def layouts(session, project):
    """Only known project layouts beneath registered local workspaces."""
    seen = set()
    from .project_storage import workspace_history
    hosts = [(host, uri) for host in session.scalars(select(m.Project)) for uri in workspace_history(session, host)]
    for host, uri in hosts:
        try:
            root = local_workspace_path(uri)
        except (OSError, ProjectStorageError):
            continue
        branches = [
            root / "projects" / project.id / ".llm-platform",
            root
            / ".llm-platform"
            / "worker-workspaces"
            / "projects"
            / project.id
            / ".llm-platform",
        ]
        if host.id == project.id:
            from llm_platform_core.storage_layout import project_state_root
            from ..config import get_settings
            branches.insert(0, project_state_root(root, get_settings().storage_root))
        for branch in branches:
            if str(branch).casefold() not in seen and branch.exists():
                boundary = root.parent if branch == root else root
                checked_path(branch, boundary)
                seen.add(str(branch).casefold())
                yield branch, boundary


def historical_input_copies(ctx, project, only_path=None, covered_paths=()):
    """Only exact manifest copies in recorded local attempt directories."""
    from urllib.parse import unquote, urlparse

    from .project_deletion import _worker_path

    workers = {w.id: w for w in ctx.session.scalars(select(m.WorkerNode))}

    def digest(path):
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
        if key not in ctx.proof_cache:
            with path.open("rb") as stream:
                ctx.proof_cache[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        return ctx.proof_cache[key]

    for attempt in ctx.attempts:
        task = ctx.tasks.get(attempt.task_id)
        worker = workers.get(attempt.worker_id)
        if not task or task.project_id != project.id or not worker:
            continue
        if ctx.cache_reason(task, project, attempt):
            continue
        raw = str((attempt.runtime_handle or {}).get("run_dir") or "")
        suffix = f"/projects/{project.id}/.llm-platform/runs/{task.id}/{attempt.id}"
        if not raw.replace("\\", "/").endswith(suffix):
            continue
        try:
            base = _worker_path(raw, worker)
            root = _worker_path(raw.replace("\\", "/")[:-len(suffix)], worker)
            checked_path(base, root)
            for path in (base / "inputs" / "data").glob("*/dataset-records.jsonl"):
                if only_path and path.absolute() != Path(only_path):
                    continue
                if any(path.is_relative_to(Path(covered)) for covered in covered_paths):
                    continue
                if not re.fullmatch(r"[0-9]{3}-[0-9a-f-]{36}", path.parent.name):
                    continue
                artifact = ctx.artifacts.get(path.parent.name[4:])
                if not artifact or artifact.kind != "dataset_record_manifest":
                    continue
                source = local_path(artifact.uri)
                if (not source or not source.is_file() or source == path
                        or artifact.availability_state != "available"):
                    continue
                checked_path(path, root)
                if path.stat().st_size != artifact.size_bytes:
                    continue
                # Linux paths must also protect their Windows WSL UNC aliases.
                raw_target = raw.rstrip("/") + "/" + path.relative_to(base).as_posix()
                refs = list(ctx.refs) + [e.uri for e in [*ctx.artifacts.values(), *ctx.locations]]
                protected = False
                for ref in refs:
                    value = unquote(urlparse(ref).path) if ref.startswith("file:") else ref
                    value = value.replace("\\", "/").rstrip("/")
                    if (value and (value == raw_target or value.startswith(raw_target + "/")
                                   or raw_target.startswith(value + "/") or raw_target in value)):
                        protected = True
                        break
                if protected or ctx.path_reason(path):
                    continue
                if digest(source) == artifact.content_hash == digest(path):
                    yield path, root, task, attempt, artifact
        except (OSError, ValueError):
            continue


def discover(session, project, *, only_path=None, proof_cache=None):
    from .storage_management import SCRATCH_DIRECTORY_NAMES, SCRATCH_FILE_NAMES

    ctx = Context(session, proof_cache)
    output = []
    seen = set()

    def add(path, root, category, task=None, attempt=None, reason="", impact="", extra=None):
        path = path.absolute()
        if only_path and path != Path(only_path):
            return
        if str(path) in seen or not (path.exists() or path.is_symlink()):
            return
        seen.add(str(path))
        metadata = {
            "cleanup_kind": PLAN_KIND,
            "root": str(root),
            "impact": impact,
            "project_name": project.name,
            "owner_label": task.title if task else "",
            **(extra or {}),
        }
        reason = reason or ctx.path_reason(path, metadata.get("artifact_ids", []))
        try:
            stats = inspect_target(path, root)
        except (OSError, ValueError) as exc:
            stats = {"size_bytes": 0, "file_count": 0, "fingerprint": ""}
            reason = str(exc)
        if not stats["size_bytes"] and not reason:
            return
        output.append(
            dict(
                path=str(path),
                category=category,
                **stats,
                reclaimable=not bool(reason),
                reason=reason or "所有权与引用检查通过",
                task_id=task.id if task else None,
                attempt_id=attempt.id if attempt else None,
                metadata_json=metadata,
            )
        )

    project_tasks = {t.id: t for t in ctx.tasks.values() if t.project_id == project.id}
    project_runs = list(
        session.scalars(select(m.TrainingRun).where(m.TrainingRun.project_id == project.id))
    )
    for platform, root in layouts(session, project):
        for attempt in ctx.attempts:
            task = project_tasks.get(attempt.task_id)
            if not task:
                continue
            if only_path and not Path(only_path).is_relative_to(
                platform / "runs" / task.id / attempt.id
            ):
                continue
            owner = ctx.cache_reason(task, project, attempt)
            for name in sorted(SCRATCH_DIRECTORY_NAMES | SCRATCH_FILE_NAMES):
                add(
                    platform / "runs" / task.id / attempt.id / name,
                    root,
                    "scratch",
                    task,
                    attempt,
                    reason=owner,
                    impact="删除输入或计算缓存；保留任务配置、日志和评测结果",
                )
        for run in project_runs:
            run_root = platform / "training-runs" / run.id
            if not run_root.exists() or (
                only_path and not Path(only_path).is_relative_to(run_root)
            ):
                continue
            task = project_tasks.get(run.task_id)
            owner = ctx.owner_reason(task)
            run_status = run.status
            # Older control-plane cancellation did not refresh TrainingRun.status.
            # Interpret only a proven completed cancellation, without rewriting history.
            if (run_status == "paused" and task and task.observed_state == "cancelled"
                    and task.ended_at is not None and not owner):
                run_status = "cancelled"
            if run_status not in TERMINAL:
                owner = "训练运行尚未结束"
            trainer = run_root / "trainer"
            try:
                checked_path(trainer, root)
            except (OSError, ValueError):
                add(
                    trainer,
                    root,
                    "intermediate_checkpoint",
                    task,
                    reason="训练目录包含链接或越界路径",
                )
                continue
            checkpoints = sorted(trainer.glob("checkpoint-*"))
            proof = None
            if checkpoints and run_status == "succeeded" and not owner:
                proof = ctx.final_checkpoint(run)
                if not proof:
                    owner = "没有同次训练可用且哈希校验通过的最终 Checkpoint 包"
            if run_status not in {"succeeded", "cancelled", "lost", "failed"}:
                owner = owner or "训练状态不支持清理"
            for checkpoint in checkpoints:
                if not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name):
                    continue
                add(
                    checkpoint,
                    root,
                    "intermediate_checkpoint"
                    if run_status == "succeeded"
                    else "cancelled_checkpoint",
                    task,
                    reason=owner,
                    impact="不能恢复到该中间步数；最终恢复目录与模型包保留"
                    if run_status == "succeeded"
                    else "删除该历史运行的恢复状态；无法从此恢复点继续训练",
                    extra={
                        "training_run_id": run.id,
                        "recorded_training_status": run.status,
                        "effective_training_status": run_status,
                        "recovery_copy": proof,
                        "owner_label": "、".join(
                            f"{v.name} V{v.version}"
                            for v in ctx.models
                            if v.status != "deleted"
                            and any(
                                cp.id == v.source_checkpoint_id and cp.training_run_id == run.id
                                for cp in ctx.checkpoints
                            )
                        )
                        or (task.title if task else run.id),
                    },
                )
            add(
                run_root / "inputs" / "resume-checkpoint",
                root,
                "training_inputs",
                task,
                reason=ctx.cache_reason(task, project)
                or ("训练运行尚未结束" if run_status not in TERMINAL else ""),
                impact="删除终态训练任务的输入恢复包副本；规范制品与训练记录保留",
                extra={"training_run_id": run.id},
            )

    for path, boundary, task, attempt, source in historical_input_copies(ctx, project, only_path, tuple(seen)):
        add(path, boundary, "scratch", task, attempt,
            impact="删除已逐字节哈希核验的历史输入副本；规范制品和执行记录保留",
            extra={"source_artifact_id": source.id, "source_hash": source.content_hash})

    try:
        root = local_workspace_path(project.workspace_uri)
    except (OSError, ProjectStorageError):
        return output
    for artifact in ctx.artifacts.values():
        if artifact.kind not in {"model", "checkpoint"}:
            continue
        if not (artifact.metadata_json or {}).get("archived_directory"):
            # Historical task JSON summaries can also have kind=model; retain history.
            continue
        attempt = next(
            (a for a in ctx.attempts if a.id == artifact.created_by_task_attempt_id), None
        )
        task = project_tasks.get(attempt.task_id) if attempt else None
        if not task:
            continue
        model_id = (task.result or {}).get("model_version_id")
        deleted = next(
            (
                v
                for v in ctx.models
                if v.id == model_id and v.status == "deleted" and v.project_id == project.id
            ),
            None,
        )
        if not deleted:
            continue
        path = local_path(artifact.uri)
        if not path or not path.is_relative_to(
            root / ".llm-platform" / "artifacts" / artifact.kind
        ):
            continue
        if path.parent.name != artifact.content_hash:
            continue
        reason = ctx.owner_reason(task, attempt)
        cp_ids = {cp.id for cp in ctx.checkpoints if cp.artifact_id == artifact.id}
        if any(
            v.status != "deleted"
            and (v.artifact_id == artifact.id or v.source_checkpoint_id in cp_ids)
            for v in ctx.models
        ):
            reason = "保留模型仍使用该制品或来源 Checkpoint"
        if ({artifact.id} | cp_ids) & ctx.refs:
            reason = "任务、复用计划、流程或发布仍引用该制品"
        locations = [loc for loc in ctx.locations if loc.artifact_id == artifact.id]
        if any(local_path(loc.uri) != path for loc in locations):
            reason = "存在其他位置或远端副本，请使用模型版本制品治理"
        if artifact.id in ctx.global_artifacts:
            reason = "全局数据、编码语料、代码、Tokenizer 或外部权重登记保留"
        add(
            path,
            root,
            "orphan_artifact",
            task,
            attempt,
            reason=reason,
            impact="回收已删除模型的遗留文件并更新可用性；保留版本、血缘与任务历史",
            extra={
                "artifact_ids": [artifact.id],
                "location_ids": [loc.id for loc in locations],
                "deleted_model_id": deleted.id,
            },
        )
    return output


def validate(session, item, proof_cache=None):
    project = session.get(m.Project, item.project_id)
    if not project:
        return None, "项目已不存在"
    candidates = discover(session, project, only_path=item.path, proof_cache=proof_cache)
    candidate = next((row for row in candidates if row["path"] == item.path), None)
    if not candidate or not candidate["reclaimable"]:
        return None, candidate["reason"] if candidate else "候选所有权或目录布局已变化"
    for key in ("size_bytes", "file_count", "fingerprint", "category", "task_id", "attempt_id"):
        if candidate[key] != getattr(item, key):
            return None, "文件在扫描后发生变化，请重新扫描"
    for key in ("root", "artifact_ids", "location_ids", "training_run_id", "recovery_copy", "source_artifact_id", "source_hash"):
        if candidate["metadata_json"].get(key) != item.metadata_json.get(key):
            return None, "候选引用或最终恢复副本已变化，请重新扫描"
    return candidate, ""


def _availability(session, item, state, plan_id):
    for aid in item.metadata_json.get("artifact_ids", []):
        artifact = session.get(m.Artifact, aid)
        artifact.availability_state = state
        artifact.metadata_json = {**artifact.metadata_json, "deletion_plan_id": plan_id}
        for cp in session.scalars(select(m.Checkpoint).where(m.Checkpoint.artifact_id == aid)):
            if state in {"missing", "unknown"}:
                cp.status = state
    for lid in item.metadata_json.get("location_ids", []):
        loc = session.get(m.ArtifactLocation, lid)
        loc.availability_state = state
        loc.last_verified_at = None


def run_plan(plan_id):
    from ..database import SessionLocal

    with SessionLocal() as session:
        plan = session.get(m.StorageCleanupPlan, plan_id)
        if not plan or plan.status != "executing":
            return
        proof_cache = {}
        items = list(
            session.scalars(
                select(m.StorageCleanupPlanItem).where(m.StorageCleanupPlanItem.plan_id == plan.id)
            )
        )
        for item in items:
            try:
                # Same SQLite write boundary as task creation/claiming and model deletion.
                session.execute(
                    update(m.Project)
                    .where(m.Project.id == item.project_id)
                    .values(updated_at=m.Project.updated_at)
                )
                session.expire_all()
                candidate, reason = validate(session, item, proof_cache)
                if not candidate:
                    item.status, item.error = (
                        "skipped",
                        {"code": "candidate_changed", "message": reason},
                    )
                    session.commit()
                    continue
                result = remove_target(
                    {
                        "path": item.path,
                        "root": candidate["metadata_json"]["root"],
                        "size_bytes": item.size_bytes,
                        "file_count": item.file_count,
                        "fingerprint": item.fingerprint,
                    }
                )
                item.status = result["status"]
                item.actual_reclaimed_bytes = result["actual_reclaimed_bytes"]
                _availability(session, item, "missing", plan.id)
                session.commit()
            except Exception as exc:
                session.rollback()
                item.status = "failed"
                item.error = {"message": str(exc)}
                # A partially removed archive cannot be advertised as available.
                if item.metadata_json.get("artifact_ids"):
                    _availability(
                        session, item, "unknown" if Path(item.path).exists() else "missing", plan.id
                    )
                session.commit()
        finish(session, plan, items)


def finish(session, plan, items):
    plan.summary = {
        **plan.summary,
        "actual_reclaimed_bytes": sum(i.actual_reclaimed_bytes for i in items),
        "deleted_count": sum(i.status == "deleted" for i in items),
        "skipped_count": sum(i.status == "skipped" for i in items),
        "failed_count": sum(i.status == "failed" for i in items),
    }
    plan.status = (
        "succeeded" if all(i.status in {"deleted", "missing"} for i in items) else "partial"
    )
    plan.executed_at = m.utcnow()
    session.add(
        m.AuditEvent(
            project_id=None,
            action="storage.data_cleanup.completed",
            subject_type="storage_cleanup_plan",
            subject_id=plan.id,
            payload=plan.summary,
        )
    )
    session.commit()


def recover_interrupted():
    from ..database import SessionLocal

    with SessionLocal() as session:
        for plan in session.scalars(
            select(m.StorageCleanupPlan).where(
                m.StorageCleanupPlan.status == "executing",
                m.StorageCleanupPlan.summary["kind"].as_string() == PLAN_KIND,
            )
        ):
            items = list(
                session.scalars(
                    select(m.StorageCleanupPlanItem).where(
                        m.StorageCleanupPlanItem.plan_id == plan.id
                    )
                )
            )
            for item in items:
                if item.status != "planned":
                    continue
                item.status = "failed" if Path(item.path).exists() else "missing"
                item.error = {"message": "清理被重启中断，请重新扫描余留文件"}
                if item.metadata_json.get("artifact_ids"):
                    _availability(
                        session, item, "unknown" if Path(item.path).exists() else "missing", plan.id
                    )
            finish(session, plan, items)
