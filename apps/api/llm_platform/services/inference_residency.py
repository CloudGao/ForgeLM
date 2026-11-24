"""Coordinate resident models through existing Worker heartbeats and GPU leases."""
from datetime import UTC, timedelta

from sqlalchemy import select

from ..models import ResourceLease, WorkerNode, utcnow


def gpu_keys(worker, indexes=None):
    keys = set()
    for position, gpu in enumerate((worker.resources or {}).get("gpus") or []):
        index = int(gpu.get("index", position)) if isinstance(gpu, dict) else int(gpu)
        if indexes is not None and index not in indexes:
            continue
        uuid = gpu.get("uuid") if isinstance(gpu, dict) else None
        if uuid:
            keys.add("uuid:" + str(uuid))
        keys.add(f"host:{worker.hostname or worker.id}:{index}")
    return keys


def live(worker):
    return (worker.status == "online" and worker.last_heartbeat_at is not None
            and worker.last_heartbeat_at.replace(tzinfo=UTC) > utcnow() - timedelta(seconds=35))


def resident_matches(worker, task):
    model = (task.payload or {}).get("model") or {}
    config = (task.payload or {}).get("config") or {}
    dtype = config.get("dtype", "auto")
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if ((task.payload or {}).get("resource_request") or {}).get("gpus") else "cpu"
    return any(r.get("adapter_id") == task.adapter_id and r.get("model_id") == model.get("id")
               and r.get("model_id") and dtype in {r.get("requested_dtype"), r.get("dtype")}
               and str(r.get("device", "")).startswith(device)
               for r in (worker.resources or {}).get("inference_runtimes", []))


def prefer_warm_worker(session, worker, task):
    from .tasks import worker_matches_selector
    if task.kind != "inference" or resident_matches(worker, task):
        return False
    request = (task.payload or {}).get("resource_request") or {}
    for other in session.scalars(select(WorkerNode).where(WorkerNode.id != worker.id)):
        caps = set(other.capabilities or [])
        if not live(other) or not resident_matches(other, task) or not worker_matches_selector(other, request):
            continue
        if "*" not in caps and not {task.adapter_id, task.launcher_id}.issubset(caps):
            continue
        if len((other.resources or {}).get("gpus") or []) < int(request.get("gpus") or 0):
            continue
        if session.scalar(select(ResourceLease.id).where(ResourceLease.worker_id == other.id,
                ResourceLease.status == "active", ResourceLease.expires_at > utcnow()).limit(1)) is None:
            return True
    return False


def reclaim_before_allocation(session, worker, indexes):
    """Do not grant a lease until other processes acknowledge releasing caches."""
    requested = gpu_keys(worker, indexes)
    waiting = False
    for other in session.scalars(select(WorkerNode).where(WorkerNode.id != worker.id)):
        for runtime in (other.resources or {}).get("inference_runtimes", []):
            if str(runtime.get("device", "")).startswith("cuda") and requested & gpu_keys(other, runtime.get("gpu_indexes") or None):
                other.labels = {**(other.labels or {}), "inference_reclaim_requested": True}
                waiting = True
    return waiting
