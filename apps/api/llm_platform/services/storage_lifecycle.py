"""Conservative retention decisions; logical versions outlive physical replicas."""
from datetime import timedelta

from sqlalchemy import select

from ..models import (
    Artifact,
    ArtifactLocation,
    DatasetVersion,
    Project,
    ProjectPipelineRun,
    ResourceLease,
    Task,
    TaskAttempt,
    utcnow,
)

TERMINAL = {"succeeded", "failed", "cancelled", "lost", "skipped"}
INTERMEDIATE_KINDS = {"dataset_snapshot", "tokenized_corpus", "supervision_corpus",
                      "data_mixture_manifest", "supervision_record_index"}


def replica_eligibility(session, location, artifact):
    """Only surplus verified copies after all known consumers finish are eligible."""
    if artifact.kind not in INTERMEDIATE_KINDS:
        return False, "原始/模型/Tokenizer/Checkpoint 或其他保留制品"
    if location.uri == artifact.uri:
        return False, "兼容读取使用的制品规范位置保留"
    if session.scalar(select(DatasetVersion.id).where(
        DatasetVersion.artifact_id == artifact.id, DatasetVersion.parent_version_id.is_(None)).limit(1)):
        return False, "原始数据默认保留"
    project = session.get(Project, location.project_id) if location.project_id else None
    policy = (project.storage_policy or {}) if project else {}
    retention = (location.metadata_json or {}).get("retention", "auto")
    if retention == "retain" or (retention == "auto" and policy.get("intermediate_retention") == "retain"):
        return False, "保留策略：长期保留"
    if location.availability_state != "available" or not location.last_verified_at:
        return False, "未验证或不可用的位置不自动回收"
    # Primary locations anchor version identity; callers must explicitly migrate them first.
    if session.scalar(select(DatasetVersion.id).where(
        DatasetVersion.primary_location_id == location.id).limit(1)):
        return False, "数据集主位置保留；仅回收多余的已验证副本"
    if session.scalar(select(Task.id).where(~Task.observed_state.in_(TERMINAL),
        Task.adapter_id != "builtin.storage-maintenance").limit(1)):
        return False, "存在未完成任务，保留所有潜在共享输入"
    if session.scalar(select(ProjectPipelineRun.id).where(
        ~ProjectPipelineRun.status.in_(TERMINAL)).limit(1)):
        return False, "存在未完成流程，等待下游与分支消费完成"
    from .tasks import _utc
    latest_use = max(_utc(location.updated_at), _utc(artifact.updated_at))
    # This conservative global last-use bound includes cross-project consumers.
    ended = session.scalars(select(Task.ended_at).where(Task.ended_at.is_not(None),
        Task.adapter_id != "builtin.storage-maintenance")).all()
    if ended:
        latest_use = max(latest_use, max(_utc(value) for value in ended))
    if latest_use + timedelta(hours=int(policy.get("grace_hours", 24))) > utcnow():
        return False, "消费完成后的保留宽限期尚未结束"
    for lease in session.scalars(select(ResourceLease).where(ResourceLease.released_at.is_(None))):
        if location.id in (lease.allocation or {}).get("artifact_location_ids", []):
            return False, "位置仍有读取租约"
    copies = session.scalars(select(ArtifactLocation.id).where(
        ArtifactLocation.artifact_id == artifact.id, ArtifactLocation.availability_state == "available",
        ArtifactLocation.last_verified_at.is_not(None)).order_by(ArtifactLocation.id)).all()
    if len(copies) < 2 or location.id == copies[0]:
        return False, "唯一副本或尚无已验证恢复副本，继续保留"
    return True, "下游已结束、宽限期已过且存在相同内容的已验证副本"


def guard_generation_cleanup(session, payload):
    from fastapi import HTTPException

    from .model_deletion import strings

    refs = set(strings(payload))
    for maintenance in session.scalars(select(Task).where(
        Task.adapter_id == "builtin.storage-maintenance", ~Task.observed_state.in_(TERMINAL),
    )):
        for item in (maintenance.payload or {}).get("candidates", []):
            if item.get("category") != "generation_duplicate":
                continue
            path = item["path"].replace("\\", "/")
            attempt_id = path.rsplit("/", 1)[-1][-36:]
            if attempt_id in refs or any(path in ref for ref in refs):
                raise HTTPException(409, "生成恢复副本正在回收，请等待存储维护完成后重新选择来源")


def generation_storage_payload(session, project_id, worker_id):
    """Frozen ownership for manual generation-cache maintenance; no caller paths."""
    from .. import models as m
    from .model_deletion import strings

    refs = set()
    for row in session.scalars(select(Task)):
        if row.adapter_id == "builtin.storage-maintenance":
            continue  # Maintenance inventory is not a consumer or continuation pin.
        # Any explicit continuation binding remains meaningful even after completion.
        refs.update(strings(row.payload))
        if row.observed_state not in TERMINAL:
            refs.update(strings(row.result))
    for cls in (m.PlanRevision, m.ProjectPipelineRevision, m.ProjectPipelineInstance,
                m.Release, m.ReleaseJob, m.QuantizationJob):
        for row in session.scalars(select(cls)):
            refs.update(strings(row.to_dict()))
    project = session.get(Project, project_id)
    owners = []
    if project and (project.storage_policy or {}).get("cache_cleanup") != "retain":
        for attempt, owner in session.execute(select(TaskAttempt, Task).join(Task, Task.id == TaskAttempt.task_id).where(
            Task.project_id == project_id, TaskAttempt.worker_id == worker_id,
            Task.observed_state.in_(TERMINAL | {"paused"}),
            TaskAttempt.status.in_(TERMINAL | {"paused"}), TaskAttempt.ended_at.is_not(None),
        )):
            payload = owner.payload or {}
            if not all(key in payload for key in ("output_dataset_version_id", "generation", "generation_input", "generation_model")):
                continue
            if (payload.get("storage_policy") or {}).get("cache_cleanup") == "retain":
                continue
            if session.scalar(select(ResourceLease.id).where(ResourceLease.task_attempt_id == attempt.id,
                                                             ResourceLease.released_at.is_(None)).limit(1)):
                continue
            owners.append({"project_id": project_id, "attempt_id": attempt.id,
                           "created_at": attempt.created_at.isoformat(), "pinned": attempt.id in refs,
                           "task": {"id": owner.id, "payload": {key: payload[key] for key in (
                               "output_dataset_version_id", "generation", "generation_input", "generation_model")}}})
    return {"generation_attempts": owners, "generation_path_references": sorted(
        ref for ref in refs if "/" in ref or "\\" in ref)}


def refresh_maintenance_payload(session, task, worker_id):
    if task.payload.get("model_deletion_plan_id"):
        from .model_deletion import refresh_worker_payload
        refresh_worker_payload(session, task, worker_id)
        return
    payload = dict(task.payload)
    payload.update(generation_storage_payload(session, task.project_id, worker_id))
    locations = session.execute(select(ArtifactLocation, Artifact).join(
        Artifact, Artifact.id == ArtifactLocation.artifact_id
    )).all()
    payload["protected_uris"] = [loc.uri for loc, _ in locations]
    payload["locations"] = []
    for loc, artifact in locations:
        if loc.worker_id != worker_id:
            continue
        eligible, reason = replica_eligibility(session, loc, artifact)
        payload["locations"].append({"id": loc.id, "uri": loc.uri, "kind": artifact.kind,
                                     "reclaimable": eligible, "reason": reason,
                                     "retention": (loc.metadata_json or {}).get("retention", "auto")})
    owners = session.execute(select(TaskAttempt, Task).join(Task, Task.id == TaskAttempt.task_id)
        .where(TaskAttempt.worker_id == worker_id, Task.observed_state.in_(TERMINAL),
               TaskAttempt.status.in_(TERMINAL))).all()
    payload["terminal_attempts"] = [{"project_id": t.project_id, "task_id": t.id,
                                    "attempt_id": a.id} for a, t in owners
                                    if (t.payload.get("storage_policy") or {}).get("cache_cleanup") != "retain"]
    if payload.get("storage_action") == "cleanup":
        eligible = {x["id"] for x in payload["locations"] if x["reclaimable"]}
        for item in payload.get("candidates", []):
            identity = item.get("location_id")
            if not identity:
                continue
            if identity not in eligible:
                continue
            loc = session.get(ArtifactLocation, identity)
            loc.availability_state = "offline"
            loc.metadata_json = {**(loc.metadata_json or {}), "lifecycle_state": "reclaiming",
                                 "cleanup_task_id": task.id}
            payload["protected_uris"].remove(loc.uri)
    task.payload = payload


def finalize_storage_cleanup(session, task, result):
    if task.payload.get("model_deletion_plan_id"):
        if result.get("storage_action") != task.payload.get("storage_action"):
            raise ValueError("模型删除任务返回的动作不匹配")
        return
    if result.get("storage_action") != task.payload.get("storage_action"):
        raise ValueError("存储结果动作与任务不匹配")
    if result.get("storage_action") == "scan":
        return
    for item in result.get("items", []):
        identity = item.get("location_id")
        if not identity:
            continue
        loc = session.get(ArtifactLocation, identity)
        if loc is not None and item.get("status") == "skipped" and (loc.metadata_json or {}).get("cleanup_task_id") != task.id:
            continue
        if loc is None or (loc.metadata_json or {}).get("cleanup_task_id") != task.id:
            raise ValueError("清理结果与登记位置不匹配")
        deleted = item.get("status") == "deleted"
        loc.availability_state = "missing" if deleted else "available"
        loc.metadata_json = {**loc.metadata_json, "lifecycle_state": "evicted" if deleted else "retained",
                             "reclaimed_bytes": item.get("reclaimed_bytes", 0),
                             "cleanup_completed_at": utcnow().isoformat()}


def estimated_write_bytes(session, task):
    request = task.payload.get("resource_request") or {}
    if request.get("disk_bytes") is not None:
        return max(0, int(request["disk_bytes"]))
    if task.adapter_id == "builtin.data-juicer":
        selection = (task.payload.get("data_processing_input") or {}).get("file_selection")
        if selection and selection.get("schema") == "forgelm-processing-file-selection-v1":
            return sum(int(item["size_bytes"]) for item in selection["files"]) * 6
        identity = task.payload.get("input_dataset_version_id")
        version = session.get(DatasetVersion, identity) if identity else None
        if version:
            # Compressed input expands into Arrow and several materialized stages.
            return int(version.raw_size_bytes or 0) * 6
    if task.adapter_id == "builtin.tokenize-corpus":
        source = task.payload.get("data_input") or {}
        return int(source.get("token_estimate", 0) or 0) * 8
    if task.adapter_id == "builtin.supervision-corpus":
        return sum(int(source.get("token_estimate", 0) or 0)
                   for source in task.payload.get("data_inputs", [])) * 16
    return 0


def volume_identity(worker):
    layout = (worker.resources or {}).get("storage_layout") or {}
    disk = layout.get("data_disk") or layout.get("system_disk") or {}
    return f"{worker.hostname}:{disk.get('device') or worker.id}"


def reserved_disk_bytes(session, worker):
    from ..models import WorkerNode
    identity = volume_identity(worker)
    total = 0
    for lease in session.scalars(select(ResourceLease).where(ResourceLease.released_at.is_(None))):
        owner = session.get(WorkerNode, lease.worker_id)
        if owner and volume_identity(owner) == identity:
            total += int((lease.allocation or {}).get("disk_bytes", 0))
    return total


def reconcile_storage_maintenance(session):
    """Idle maintenance is a normal visible Task, never a shell side channel."""
    from ..models import WorkerNode
    from ..routers.storage import create_worker_storage_job
    from ..schemas import WorkerStorageJobCreate
    from .model_deletion import reconcile_plans
    from .tasks import _utc, worker_matches_selector
    reconcile_plans(session)
    for loc in session.scalars(select(ArtifactLocation).where(
        ArtifactLocation.metadata_json["lifecycle_state"].as_string() == "reclaiming")):
        owner = session.get(Task, (loc.metadata_json or {}).get("cleanup_task_id"))
        if owner and owner.observed_state in TERMINAL:
            loc.metadata_json = {**loc.metadata_json, "lifecycle_state": "verification_required"}
            loc.availability_state = "unknown"
            loc.last_verified_at = None
    session.commit()
    for worker in session.scalars(select(WorkerNode).where(WorkerNode.status == "online")):
        if "builtin.storage-maintenance" not in (worker.capabilities or []):
            continue
        if session.scalar(select(ResourceLease.id).where(
            ResourceLease.worker_id == worker.id, ResourceLease.released_at.is_(None)).limit(1)):
            continue
        jobs = [t for t in session.scalars(select(Task).where(
            Task.adapter_id == "builtin.storage-maintenance").order_by(Task.created_at.desc()))
            if t.payload.get("resource_request", {}).get("worker_selector", {}).get("worker_id") == worker.id]
        if any(job.observed_state not in TERMINAL for job in jobs):
            continue
        latest = jobs[0] if jobs else None
        waiting_for_space = any(worker_matches_selector(worker, task.payload.get("resource_request") or {})
            for task in session.scalars(select(Task).where(Task.observed_state == "queued",
                                                           Task.current_phase == "waiting_for_storage")))
        if (latest and latest.payload.get("automatic") and latest.observed_state == "succeeded"
                and latest.result.get("storage_action") == "scan" and latest.ended_at
                and _utc(latest.ended_at) > utcnow() - timedelta(minutes=30)):
            ids = [item["id"] for item in latest.result.get("items", [])
                   if item.get("reclaimable") and item.get("category") != "computation_cache"]
            if ids:
                response = create_worker_storage_job(latest.project_id, WorkerStorageJobCreate(
                    worker_id=worker.id, action="cleanup", scan_task_id=latest.id, item_ids=ids), session)
                created = session.get(Task, response["id"])
                created.payload = {**created.payload, "automatic": True}
                session.commit()
                continue
        interval = timedelta(minutes=5) if waiting_for_space else timedelta(hours=1)
        if latest and _utc(latest.created_at) > utcnow() - interval:
            continue
        if not waiting_for_space and session.scalar(select(Task.id).where(~Task.observed_state.in_(TERMINAL),
            Task.adapter_id != "builtin.storage-maintenance").limit(1)):
            continue
        owner = session.scalar(select(Task).join(TaskAttempt, TaskAttempt.task_id == Task.id)
            .where(TaskAttempt.worker_id == worker.id,
                   Task.adapter_id != "builtin.storage-maintenance").order_by(Task.created_at.desc()))
        if not owner or not owner.project_id:
            continue
        response = create_worker_storage_job(owner.project_id, WorkerStorageJobCreate(worker_id=worker.id), session)
        created = session.get(Task, response["id"])
        created.payload = {**created.payload, "automatic": True}
        session.commit()
