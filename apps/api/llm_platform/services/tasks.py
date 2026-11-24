from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from llm_platform_core.datasets import TEXT_PARSER_ID, TEXT_PARSER_VERSION, TEXT_USAGE_CAPABILITIES
from llm_platform_core.domain import TaskStatus, ensure_task_transition
from llm_platform_core.tokenizers import evaluate_tokenizer_gate_policy
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..models import (
    Artifact,
    ArtifactLocation,
    AuditEvent,
    Checkpoint,
    CodeSource,
    CodeVersion,
    DataConversionRun,
    DataMixtureVersion,
    DataProcessingRun,
    Dataset,
    DatasetAnalysisRevision,
    DatasetLabelOperation,
    DatasetLabelSnapshot,
    DatasetMaterialization,
    DatasetVersion,
    DatasetVersionIndex,
    EvaluationRun,
    InferenceRecord,
    ModelVersion,
    ModelWeightAsset,
    ModelWeightVersion,
    OutboxEvent,
    PlanRevision,
    Project,
    QuantizationRun,
    Release,
    ReleaseBuildRun,
    ReleaseDeliveryRun,
    RemoteWorkerEnvironment,
    RemoteWorkerOperation,
    ResourceLease,
    SupervisionCorpusVersion,
    SupervisionDatasetVersion,
    SupervisionTrainingViewVersion,
    Task,
    TaskAttempt,
    TaskEvent,
    TokenizedCorpusVersion,
    TokenizedDatasetVersion,
    TokenizerVersion,
    TrainingDataViewVersion,
    TrainingPlan,
    TrainingRun,
    WorkerNode,
    WorkerPluginInstallation,
    utcnow,
)
from .artifacts import artifact_storage
from .dataset_analyses import create_analysis_revision
from .dataset_materialization import (
    ensure_version_primary_location,
    finalize_dataset_materialization,
)
from .project_storage import local_workspace_path

TERMINAL_TASK_STATES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.LOST,
}
TIMING_SAMPLE_WINDOW = timedelta(minutes=30)
MIN_TIMING_SAMPLE_SPAN_SECONDS = 30


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _attempt_runtime_seconds(
    attempts: list[TaskAttempt], now: datetime, *, include_active: bool = True
) -> float:
    total = 0.0
    for attempt in attempts:
        if attempt.started_at is None:
            continue
        end = attempt.ended_at
        if end is None:
            if not include_active:
                continue
            end = now
        total += max(0.0, (_utc(end) - _utc(attempt.started_at)).total_seconds())
    return total


def estimate_task_timing(
    task: Task,
    attempts: list[TaskAttempt],
    progress_samples: list[tuple[datetime, float]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a truthful task timing view from bounded, attempt-local progress samples."""
    now = _utc(now or utcnow())
    state = TaskStatus(task.observed_state)
    actual_runtime = _attempt_runtime_seconds(attempts, now)
    if not attempts and task.started_at is not None:
        fallback_end = task.ended_at or (now if state not in TERMINAL_TASK_STATES else None)
        if fallback_end is not None:
            actual_runtime = max(
                0.0, (_utc(fallback_end) - _utc(task.started_at)).total_seconds()
            )

    if state in TERMINAL_TASK_STATES:
        return {
            "status": "complete",
            "actual_duration_seconds": round(actual_runtime),
            "estimated_total_seconds": None,
            "remaining_seconds": 0 if state == TaskStatus.SUCCEEDED else None,
            "estimated_completion_at": None,
            "sample_count": 0,
        }
    if state in {TaskStatus.DRAFT, TaskStatus.QUEUED, TaskStatus.ASSIGNED, TaskStatus.STARTING}:
        return {
            "status": "waiting",
            "actual_duration_seconds": round(actual_runtime),
            "estimated_total_seconds": None,
            "remaining_seconds": None,
            "estimated_completion_at": None,
            "sample_count": 0,
        }
    if state in {TaskStatus.PAUSING, TaskStatus.STOPPING}:
        return {
            "status": "unavailable",
            "actual_duration_seconds": round(actual_runtime),
            "estimated_total_seconds": None,
            "remaining_seconds": None,
            "estimated_completion_at": None,
            "sample_count": 0,
        }

    current_attempt = attempts[0] if attempts else None
    normalized_samples: list[tuple[datetime, float]] = []
    for created_at, progress in progress_samples:
        try:
            normalized_progress = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            continue
        normalized_samples.append((_utc(created_at), normalized_progress))
    normalized_samples.sort(key=lambda item: item[0])
    sample_window_end = (
        normalized_samples[-1][0]
        if state == TaskStatus.PAUSED and normalized_samples
        else now
    )
    cutoff = sample_window_end - TIMING_SAMPLE_WINDOW
    normalized_samples = [item for item in normalized_samples if item[0] >= cutoff]

    # Only the first attempt has a truthful 0% baseline. A resumed attempt inherits
    # task progress, so it must report two real samples before its rate is known.
    if (
        current_attempt is not None
        and current_attempt.attempt_number == 1
        and current_attempt.started_at is not None
    ):
        started_at = _utc(current_attempt.started_at)
        if started_at >= cutoff and (
            not normalized_samples or started_at <= normalized_samples[0][0]
        ):
            normalized_samples.insert(0, (started_at, 0.0))

    # A task may truthfully spend minutes at the same progress while downloading
    # or extracting inputs. Using the beginning of that plateau as the rate anchor
    # multiplies one-time preparation across all remaining work. Keep the latest
    # sample immediately before the first increase, while preserving the ordinary
    # start baseline for tasks that do not emit preparation progress.
    if len(normalized_samples) > 2:
        initial_progress = normalized_samples[0][1]
        first_increase = next(
            (
                index
                for index, (_created_at, progress) in enumerate(normalized_samples[1:], 1)
                if progress > initial_progress
            ),
            None,
        )
        if first_increase is not None and first_increase > 1:
            normalized_samples = normalized_samples[first_increase - 1 :]

    latest_progress = max(0.0, min(1.0, float(task.progress or 0)))
    if normalized_samples and latest_progress > normalized_samples[-1][1]:
        normalized_samples.append((now, latest_progress))

    anchor: tuple[datetime, float] | None = None
    latest: tuple[datetime, float] | None = None
    if normalized_samples:
        latest = normalized_samples[-1]
        for candidate in normalized_samples:
            if latest[1] > candidate[1]:
                anchor = candidate
                break

    status = "paused" if state == TaskStatus.PAUSED else "estimating"
    if anchor is None or latest is None:
        return {
            "status": status,
            "actual_duration_seconds": round(actual_runtime),
            "estimated_total_seconds": None,
            "remaining_seconds": None,
            "estimated_completion_at": None,
            "sample_count": len(normalized_samples),
        }
    sample_span = (latest[0] - anchor[0]).total_seconds()
    progress_delta = latest[1] - anchor[1]
    if sample_span < MIN_TIMING_SAMPLE_SPAN_SECONDS or progress_delta <= 0:
        return {
            "status": status,
            "actual_duration_seconds": round(actual_runtime),
            "estimated_total_seconds": None,
            "remaining_seconds": None,
            "estimated_completion_at": None,
            "sample_count": len(normalized_samples),
        }

    rate_per_second = progress_delta / sample_span
    remaining_seconds = max(0.0, (1.0 - latest_progress) / rate_per_second)
    estimated_total = actual_runtime + remaining_seconds
    return {
        "status": "paused" if state == TaskStatus.PAUSED else "available",
        "actual_duration_seconds": round(actual_runtime),
        "estimated_total_seconds": round(estimated_total),
        "remaining_seconds": round(remaining_seconds),
        "estimated_completion_at": (
            None
            if state == TaskStatus.PAUSED
            else (now + timedelta(seconds=remaining_seconds)).isoformat()
        ),
        "sample_count": len(normalized_samples),
    }


def append_task_event(
    session: Session,
    task: Task,
    event_type: str,
    payload: dict[str, Any] | None = None,
    attempt_id: str | None = None,
) -> TaskEvent:
    sequence = session.scalar(
        select(func.coalesce(func.max(TaskEvent.sequence), 0)).where(TaskEvent.task_id == task.id)
    )
    event = TaskEvent(
        task_id=task.id,
        attempt_id=attempt_id,
        sequence=int(sequence or 0) + 1,
        event_type=event_type,
        payload=payload or {},
    )
    session.add(event)
    return event


def create_task(
    session: Session,
    *,
    project_id: str | None,
    kind: str,
    title: str,
    adapter_id: str,
    launcher_id: str = "builtin.local-process",
    payload: dict[str, Any] | None = None,
) -> Task:
    from .storage_relocation import guard_writes
    guard_writes(session)
    if not project_id:
        raise ValueError("executable tasks must belong to a project")
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError("task project does not exist")
    if adapter_id != "builtin.storage-maintenance":
        from .model_deletion import guard_task_inputs
        session.execute(update(Project).where(Project.id == project_id).values(updated_at=Project.updated_at))
        guard_task_inputs(session, payload or {})
        from .storage_lifecycle import guard_generation_cleanup
        guard_generation_cleanup(session, payload or {})
    if adapter_id != "builtin.storage-maintenance" and session.scalar(select(ArtifactLocation.id)
        .where(ArtifactLocation.metadata_json["lifecycle_state"].as_string() == "reclaiming").limit(1)):
        from fastapi import HTTPException
        raise HTTPException(409, "制品副本正在回收，请在维护任务结束后重新提交")
    local_workspace_path(project.workspace_uri)
    task = Task(
        project_id=project_id,
        kind=kind,
        title=title,
        adapter_id=adapter_id,
        launcher_id=launcher_id,
        desired_state=TaskStatus.RUNNING,
        observed_state=TaskStatus.QUEUED,
        payload={"storage_policy": dict(project.storage_policy or {}), **(payload or {})},
    )
    session.add(task)
    session.flush()
    append_task_event(session, task, "task.queued", {"title": title, "kind": kind})
    session.add(
        OutboxEvent(topic="task.queued", aggregate_id=task.id, payload={"task_id": task.id})
    )
    from .worker_dispatch import notify_after_commit
    notify_after_commit(session)
    return task


def transition_task(
    session: Session,
    task: Task,
    target: str,
    *,
    event_type: str | None = None,
    payload: dict[str, Any] | None = None,
    attempt_id: str | None = None,
) -> None:
    if task.observed_state != target:
        ensure_task_transition(task.observed_state, target)
    task.observed_state = target
    task.state_version += 1
    if target == TaskStatus.RUNNING:
        task.started_at = task.started_at or utcnow()
    if target in {
        status.value
        for status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.LOST,
        )
    }:
        task.ended_at = utcnow()
    append_task_event(session, task, event_type or f"task.{target}", payload, attempt_id)


def _available_gpu_indexes(session: Session, worker: WorkerNode) -> list[int]:
    reported = (worker.resources or {}).get("gpus") or []
    indexes = [
        int(gpu.get("index", position)) if isinstance(gpu, dict) else int(gpu)
        for position, gpu in enumerate(reported)
    ]
    from .inference_residency import gpu_keys
    used_keys = set()
    for lease in session.scalars(select(ResourceLease).where(
            ResourceLease.status == "active", ResourceLease.expires_at > utcnow())):
        owner = session.get(WorkerNode, lease.worker_id)
        if owner:
            used_keys.update(gpu_keys(owner, (lease.allocation or {}).get("gpu_indexes", [])))
    return [index for index in indexes if not gpu_keys(worker, [index]) & used_keys]


def worker_matches_selector(worker: WorkerNode, request: dict[str, Any]) -> bool:
    selector = request.get("worker_selector") or {}
    if not isinstance(selector, dict):
        return False
    worker_id = selector.get("worker_id")
    if worker_id and worker.id != str(worker_id):
        return False
    worker_name = selector.get("worker_name")
    if worker_name and worker.name != str(worker_name):
        return False
    worker_ids = selector.get("worker_ids") or []
    if worker_ids and worker.id not in {str(value) for value in worker_ids}:
        return False
    labels = selector.get("labels") or {}
    if not isinstance(labels, dict):
        return False
    reported_labels = worker.labels or {}
    if any(reported_labels.get(str(key)) != value for key, value in labels.items()):
        return False
    required_capabilities = selector.get("capabilities") or []
    if not isinstance(required_capabilities, list):
        return False
    if not set(str(value) for value in required_capabilities).issubset(
        set(worker.capabilities or [])
    ):
        return False
    return True


def _task_allocation(session: Session, worker: WorkerNode, task: Task) -> dict[str, Any] | None:
    from .suite_packages import can_allocate
    if not can_allocate(session, worker, task):
        return None
    # Workers on one host may share computation-cache paths. Maintenance must
    # exclude those sibling agents too, even though CPU leases remain per Worker.
    maintenance_scope = ResourceLease.worker_id == worker.id
    if getattr(worker, "hostname", None):
        maintenance_scope = ResourceLease.worker_id.in_(
            select(WorkerNode.id).where(WorkerNode.hostname == worker.hostname)
        )
    leases = session.scalars(select(ResourceLease).where(
        maintenance_scope, ResourceLease.released_at.is_(None)
    )).all()
    if leases and (task.adapter_id == "builtin.storage-maintenance" or any(
        (owner := session.get(TaskAttempt, lease.task_attempt_id)) is not None
        and (active := session.get(Task, owner.task_id)) is not None
        and active.adapter_id == "builtin.storage-maintenance" for lease in leases
    )):
        return None
    capabilities = set(worker.capabilities or [])
    if "*" not in capabilities and (
        task.adapter_id not in capabilities or task.launcher_id not in capabilities
    ):
        return None

    request = (task.payload or {}).get("resource_request") or {}
    if not isinstance(request, dict) or not worker_matches_selector(worker, request):
        return None
    gpu_count = max(0, int(request.get("gpus", 0) or 0))
    available_gpus = _available_gpu_indexes(session, worker)
    if gpu_count > len(available_gpus):
        return None
    requested_cpus = max(0, int(request.get("cpus", 0) or 0))
    reported_cpus = int((worker.resources or {}).get("cpu_count", 0) or 0)
    leased_cpus = sum(int((lease.allocation or {}).get("cpus") or 0) for lease in session.scalars(
        select(ResourceLease).where(ResourceLease.worker_id == worker.id,
            ResourceLease.status == "active", ResourceLease.expires_at > utcnow())
    ))
    available_cpus = max(0, reported_cpus - leased_cpus)
    if (requested_cpus and requested_cpus > available_cpus) or (reported_cpus and not available_cpus):
        return None
    from .storage_lifecycle import estimated_write_bytes, reserved_disk_bytes
    write_bytes = estimated_write_bytes(session, task)
    workspace = (worker.resources or {}).get("workspace") or {}
    if write_bytes and workspace.get("free_bytes") is not None:
        reserved = reserved_disk_bytes(session, worker)
        available = max(int(workspace["free_bytes"]) - reserved - 8 * 1024**3, 0)
        if write_bytes > available:
            if task.current_phase != "waiting_for_storage":
                append_task_event(session, task, "storage.waiting", {
                    "message": "数据盘空间不足，等待回收或扩容后自动接续",
                    "required_bytes": write_bytes, "available_bytes": available,
                    "reserved_bytes": reserved,
                })
            task.current_phase = "waiting_for_storage"
            return None
    from .inference_residency import prefer_warm_worker, reclaim_before_allocation
    if prefer_warm_worker(session, worker, task):
        return None
    if gpu_count and reclaim_before_allocation(session, worker, available_gpus[:gpu_count]):
        task.current_phase = "waiting_for_gpu_release"
        return None
    return {
        "worker_id": worker.id,
        "gpu_indexes": available_gpus[:gpu_count],
        "gpus": gpu_count,
        "cpus": requested_cpus or available_cpus,
        "disk_bytes": write_bytes,
    }


def claim_task(session: Session, worker_id: str, *, direct: bool = False) -> tuple[Task, TaskAttempt, ResourceLease] | None:
    from .storage_relocation import active_plan
    if active_plan(session):
        return None
    worker = session.get(WorkerNode, worker_id)
    if worker is None or worker.status != "online":
        return None
    if session.scalar(select(RemoteWorkerOperation.id).join(
        RemoteWorkerEnvironment, RemoteWorkerOperation.environment_id == RemoteWorkerEnvironment.id
    ).where(RemoteWorkerEnvironment.worker_id == worker_id,
            RemoteWorkerOperation.kind != "probe",
            RemoteWorkerOperation.status.in_({"queued", "running"})).limit(1)):
        return None

    # One execution slot per Worker, including CPU and duplicate delivery connections.
    if session.scalar(select(ResourceLease.id).where(
        ResourceLease.worker_id == worker_id, ResourceLease.status == "active").limit(1)):
        return None

    candidates = session.scalars(
        select(Task)
        .where(Task.observed_state == TaskStatus.QUEUED)
        .order_by((Task.adapter_id == "builtin.storage-maintenance").desc(), Task.created_at)
        .limit(100)
    ).all()
    if direct:
        targeted = session.scalars(select(Task).where(
            Task.observed_state == TaskStatus.QUEUED,
            Task.kind == "inference",
            Task.payload["session_id"].as_string().is_not(None),
            Task.payload["resource_request"]["worker_selector"]["worker_id"].as_string() == worker_id,
        ).order_by(Task.created_at).limit(100)).all()
        targeted_ids = {task.id for task in targeted}
        candidates = [*targeted, *(task for task in candidates if task.id not in targeted_ids)]
    candidate: Task | None = None
    allocation: dict[str, Any] | None = None
    for queued_task in candidates:
        if queued_task.adapter_id != "builtin.storage-maintenance":
            from fastapi import HTTPException

            from .model_deletion import guard_task_inputs
            try:
                guard_task_inputs(session, queued_task.payload)
            except HTTPException as exc:
                if isinstance(exc.detail, dict) and exc.detail.get("code") != "model_deletion_busy":
                    queued_task.error = exc.detail
                    transition_task(session, queued_task, TaskStatus.FAILED)
                    session.commit()
                continue
        allocation = _task_allocation(session, worker, queued_task)
        if allocation is not None:
            candidate = queued_task
            break
    if candidate is None or allocation is None:
        return None

    claimed = session.execute(
        update(Task)
        .where(Task.id == candidate.id, Task.observed_state == TaskStatus.QUEUED)
        .values(observed_state=TaskStatus.ASSIGNED, state_version=Task.state_version + 1)
    )
    if claimed.rowcount != 1:
        session.rollback()
        return None

    session.refresh(candidate)
    from .suite_packages import claim_environment
    try:
        claim_environment(session, worker, candidate)
    except ValueError:
        session.rollback()
        return None
    if candidate.adapter_id == "builtin.storage-maintenance":
        from .storage_lifecycle import refresh_maintenance_payload
        refresh_maintenance_payload(session, candidate, worker_id)
    else:
        from .model_deletion import guard_task_inputs
        guard_task_inputs(session, candidate.payload)
        allocation["artifact_location_ids"] = list(session.scalars(select(ArtifactLocation.id)
            .where(ArtifactLocation.worker_id == worker_id,
                   ArtifactLocation.availability_state == "available")))
    attempt_number = (
        int(
            session.scalar(
                select(func.count(TaskAttempt.id)).where(TaskAttempt.task_id == candidate.id)
            )
            or 0
        )
        + 1
    )
    attempt = TaskAttempt(
        task_id=candidate.id,
        attempt_number=attempt_number,
        worker_id=worker_id,
        status=TaskStatus.ASSIGNED,
        lease_expires_at=utcnow() + timedelta(seconds=45),
        heartbeat_at=utcnow(),
    )
    session.add(attempt)
    session.flush()
    lease = ResourceLease(
        task_attempt_id=attempt.id,
        worker_id=worker_id,
        allocation=allocation,
        status="active",
        expires_at=attempt.lease_expires_at,
    )
    session.add(lease)
    candidate.active_attempt_id = attempt.id
    if direct and candidate.kind == "inference":
        candidate.current_phase = "dispatching"
    from .execution_provenance import freeze_intent
    freeze_intent(session, candidate, attempt, allocation)
    append_task_event(
        session,
        candidate,
        "task.assigned",
        {
            "worker_id": worker_id,
            "attempt_number": attempt_number,
            "allocation": allocation,
            "delivery": "direct" if direct else "claim",
        },
        attempt.id,
    )
    return candidate, attempt, lease


def update_task_from_progress(session: Session, task: Task, payload: dict[str, Any]) -> None:
    if "progress" in payload:
        reported_progress = max(0, min(1, float(payload["progress"])))
        # A resumed attempt starts by reporting its own preparing/starting phases.
        # Those events must not make the task-level progress bar jump backwards.
        task.progress = max(float(task.progress or 0), reported_progress)
    if "phase" in payload:
        task.current_phase = str(payload["phase"])

    if task.kind == "train":
        run = session.scalar(select(TrainingRun).where(TrainingRun.task_id == task.id))
        if run:
            run.status = task.observed_state
            run.current_stage = payload.get("phase", run.current_stage)
            run.current_step = int(payload.get("step", run.current_step))
            run.current_tokens = int(payload.get("tokens", run.current_tokens))
            counters = payload.get("counters")
            if isinstance(counters, dict):
                run.progress_counters = {**(run.progress_counters or {}), **counters}
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                current_metrics = run.latest_metrics or {}
                history = list(current_metrics.get("history", []))
                history.append(
                    {
                        "step": run.current_step,
                        "tokens": run.current_tokens,
                        **metrics,
                    }
                )
                run.latest_metrics = {
                    **current_metrics,
                    **metrics,
                    "history": history[-500:],
                }


def _collect_result_artifacts(
    session: Session,
    task: Task,
    attempt: TaskAttempt,
    result: dict[str, Any],
) -> tuple[list[Artifact], dict[str, str], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    for raw_record in result.get("output_artifacts", []):
        if not isinstance(raw_record, dict) or not raw_record.get("artifact_id"):
            continue
        records.append(
            {
                "ref": str(raw_record.get("ref") or raw_record["artifact_id"]),
                "artifact_id": str(raw_record["artifact_id"]),
                "kind": str(raw_record.get("kind") or "task_output"),
                "storage_type": str(raw_record.get("storage_type") or "managed"),
            }
        )
    known_ids = {record["artifact_id"] for record in records}
    for artifact_id in result.get("output_artifact_ids", []):
        if isinstance(artifact_id, str) and artifact_id not in known_ids:
            records.append(
                {
                    "ref": artifact_id,
                    "artifact_id": artifact_id,
                    "kind": "task_output",
                    "storage_type": "managed",
                }
            )
            known_ids.add(artifact_id)

    for index, raw_output in enumerate(result.get("outputs", [])):
        if not isinstance(raw_output, dict):
            continue
        storage_type = raw_output.get("storage_type") or raw_output.get("storage_mode")
        if storage_type != "external":
            continue
        artifact, _location = artifact_storage.register_external(
            session,
            project_id=task.project_id,
            kind=str(raw_output.get("kind") or "task_output"),
            uri=str(raw_output.get("uri") or ""),
            content_hash=str(raw_output.get("content_hash") or ""),
            size_bytes=int(raw_output.get("size_bytes", 0) or 0),
            media_type=str(raw_output.get("media_type") or "application/octet-stream"),
            availability_state=str(raw_output.get("availability_state") or "available"),
            worker_id=str(raw_output.get("worker_id") or attempt.worker_id or "") or None,
            created_by_task_attempt_id=attempt.id,
            metadata={
                **(raw_output.get("metadata") or {}),
                "task_id": task.id,
                "output_ref": str(raw_output.get("ref") or f"output-{index}"),
                "retention": task.payload.get("output_retention", "auto"),
            },
        )
        records.append(
            {
                "ref": str(raw_output.get("ref") or f"output-{index}"),
                "artifact_id": artifact.id,
                "kind": artifact.kind,
                "storage_type": "external",
            }
        )

    artifact_ids = list(dict.fromkeys(record["artifact_id"] for record in records))
    artifacts = (
        session.scalars(select(Artifact).where(Artifact.id.in_(artifact_ids))).all()
        if artifact_ids
        else []
    )
    found_ids = {artifact.id for artifact in artifacts}
    missing_ids = [artifact_id for artifact_id in artifact_ids if artifact_id not in found_ids]
    if missing_ids:
        raise ValueError(f"task result references unknown artifacts: {missing_ids}")
    artifact_by_ref = {record["ref"]: record["artifact_id"] for record in records}
    return artifacts, artifact_by_ref, records


def _resolve_artifact_id(
    session: Session,
    spec: dict[str, Any],
    artifact_by_ref: dict[str, str],
    default_artifact_id: str | None,
) -> str | None:
    artifact_id = spec.get("artifact_id")
    artifact_ref = spec.get("artifact_ref")
    if artifact_ref:
        artifact_id = artifact_by_ref.get(str(artifact_ref))
        if artifact_id is None:
            raise ValueError(f"unknown artifact_ref: {artifact_ref}")
    artifact_id = str(artifact_id) if artifact_id else default_artifact_id
    if artifact_id and session.get(Artifact, artifact_id) is None:
        raise ValueError(f"unknown artifact_id: {artifact_id}")
    return artifact_id


def normalize_training_result_tokens(
    task_payload: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Convert legacy stage-local token results to the cumulative plan count.

    Early stage-continuation adapters reported the selected stages' token count
    even though their frozen resume binding proved that those tokens followed an
    earlier checkpoint.  Only normalize when the final reported count exactly
    matches the frozen selected-stage budget, so arbitrary adapter results are
    never guessed or silently changed.
    """
    resume_checkpoint = task_payload.get("resume_checkpoint")
    stages = task_payload.get("stages")
    if not isinstance(resume_checkpoint, dict) or not isinstance(stages, list):
        return result

    resume_tokens = int(resume_checkpoint.get("token_count") or 0)
    incremental_tokens = sum(
        int(stage.get("target_tokens") or 0) for stage in stages if isinstance(stage, dict)
    )
    reported_tokens = int(result.get("tokens") or 0)
    if resume_tokens <= 0 or incremental_tokens <= 0 or reported_tokens != incremental_tokens:
        return result

    normalized = {**result, "tokens": resume_tokens + reported_tokens}
    raw_checkpoints = result.get("checkpoints")
    if isinstance(raw_checkpoints, list):
        normalized_checkpoints: list[Any] = []
        for raw_checkpoint in raw_checkpoints:
            if not isinstance(raw_checkpoint, dict):
                normalized_checkpoints.append(raw_checkpoint)
                continue
            checkpoint = dict(raw_checkpoint)
            token_key = (
                "tokens"
                if checkpoint.get("tokens") is not None
                else "token_count"
                if checkpoint.get("token_count") is not None
                else None
            )
            if token_key is not None:
                local_tokens = int(checkpoint[token_key] or 0)
                checkpoint[token_key] = resume_tokens + local_tokens
            normalized_checkpoints.append(checkpoint)
        normalized["checkpoints"] = normalized_checkpoints
    return normalized


def normalize_training_result_model_versions(
    session: Session,
    task: Task,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Assign an unused immutable registry version without losing requested intent."""

    reserved: set[tuple[str, int]] = set()

    def assign(name: str, requested_version: int) -> int:
        existing = session.scalar(
            select(ModelVersion).where(
                ModelVersion.project_id == task.project_id,
                ModelVersion.name == name,
                ModelVersion.version == requested_version,
            )
        )
        existing_task_id = (
            str((existing.metadata_json or {}).get("task_id") or "")
            if existing is not None
            else ""
        )
        if (
            (existing is None or existing_task_id == task.id)
            and (name, requested_version) not in reserved
        ):
            reserved.add((name, requested_version))
            return requested_version
        highest_version = int(
            session.scalar(
                select(func.max(ModelVersion.version)).where(
                    ModelVersion.project_id == task.project_id,
                    ModelVersion.name == name,
                )
            )
            or 0
        )
        assigned = max(requested_version, highest_version + 1)
        while (name, assigned) in reserved:
            assigned += 1
        reserved.add((name, assigned))
        return assigned

    raw_models = result.get("models")
    if isinstance(raw_models, list) and raw_models:
        normalized_models: list[Any] = []
        changed = False
        for index, raw_spec in enumerate(raw_models):
            if not isinstance(raw_spec, dict):
                normalized_models.append(raw_spec)
                continue
            spec = dict(raw_spec)
            name = str(spec.get("name") or f"Model-{index + 1}")
            requested_version = int(spec.get("version", 1) or 1)
            assigned_version = assign(name, requested_version)
            if assigned_version != requested_version:
                spec["version"] = assigned_version
                spec["metadata"] = {
                    **(spec.get("metadata") or {}),
                    "requested_version": requested_version,
                    "version_assignment": "next_available",
                }
                changed = True
            normalized_models.append(spec)
        return {**result, "models": normalized_models} if changed else result

    model_count = int(
        session.scalar(
            select(func.count(ModelVersion.id)).where(
                ModelVersion.project_id == task.project_id
            )
        )
        or 0
    )
    name = str(result.get("model_name") or f"Model-{model_count + 1}")
    requested_version = int(result.get("model_version", 1) or 1)
    assigned_version = assign(name, requested_version)
    if assigned_version == requested_version:
        return result
    return {
        **result,
        "model_version": assigned_version,
        "requested_model_version": requested_version,
    }


def _finalize_training_outputs(
    session: Session,
    task: Task,
    run: TrainingRun,
    result: dict[str, Any],
    artifact_by_ref: dict[str, str],
    default_artifact: Artifact,
) -> tuple[list[Checkpoint], list[ModelVersion]]:
    raw_checkpoints = result.get("checkpoints")
    checkpoint_specs = raw_checkpoints if isinstance(raw_checkpoints, list) else []
    if not checkpoint_specs:
        checkpoint_specs = [
            {
                "ref": "final",
                "step": result.get("step", run.current_step or 1000),
                "tokens": result.get("tokens", run.current_tokens or 0),
                "metrics": result.get("metrics", {}),
                "artifact_id": default_artifact.id,
            }
        ]
    checkpoints: list[Checkpoint] = []
    checkpoint_by_ref: dict[str, str] = {}
    for index, raw_spec in enumerate(checkpoint_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError("checkpoint result entries must be objects")
        step = int(raw_spec.get("step", 0))
        artifact_id = _resolve_artifact_id(session, raw_spec, artifact_by_ref, default_artifact.id)
        checkpoint = session.scalar(
            select(Checkpoint).where(
                Checkpoint.training_run_id == run.id,
                Checkpoint.step == step,
            )
        )
        if checkpoint is None:
            checkpoint = Checkpoint(
                project_id=run.project_id,
                training_run_id=run.id,
                parent_model_version_id=raw_spec.get("parent_model_version_id"),
                step=step,
                token_count=int(raw_spec.get("tokens", raw_spec.get("token_count", 0)) or 0),
                artifact_id=artifact_id,
                metrics=raw_spec.get("metrics") or {},
                training_position=raw_spec.get("training_position") or {},
                status=str(raw_spec.get("status") or "available"),
            )
            session.add(checkpoint)
            session.flush()
        checkpoints.append(checkpoint)
        checkpoint_by_ref[str(raw_spec.get("ref") or f"checkpoint-{index}")] = checkpoint.id

    raw_models = result.get("models")
    model_specs = raw_models if isinstance(raw_models, list) else []
    if not model_specs:
        model_count = int(
            session.scalar(
                select(func.count(ModelVersion.id)).where(ModelVersion.project_id == run.project_id)
            )
            or 0
        )
        final_checkpoint_ref = next(reversed(checkpoint_by_ref))
        model_specs = [
            {
                "ref": "final",
                "name": str(result.get("model_name") or f"Model-{model_count + 1}"),
                "version": int(result.get("model_version", 1) or 1),
                "kind": str(result.get("model_kind") or "base"),
                "source_checkpoint_ref": final_checkpoint_ref,
                "status": "candidate",
            }
        ]

    models: list[ModelVersion] = []
    model_by_ref: dict[str, str] = {}
    deferred_parents: list[tuple[ModelVersion, str]] = []
    for index, raw_spec in enumerate(model_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError("model result entries must be objects")
        name = str(raw_spec.get("name") or f"Model-{index + 1}")
        version = int(raw_spec.get("version", 1) or 1)
        model = session.scalar(
            select(ModelVersion).where(
                ModelVersion.project_id == run.project_id,
                ModelVersion.name == name,
                ModelVersion.version == version,
            )
        )
        if model is not None and str((model.metadata_json or {}).get("task_id") or "") != task.id:
            raise ValueError(
                f"model version collision was not resolved: {name} version {version}"
            )
        source_checkpoint_id = raw_spec.get("source_checkpoint_id")
        source_checkpoint_ref = raw_spec.get("source_checkpoint_ref")
        if source_checkpoint_ref:
            source_checkpoint_id = checkpoint_by_ref.get(str(source_checkpoint_ref))
            if source_checkpoint_id is None:
                raise ValueError(f"unknown source_checkpoint_ref: {source_checkpoint_ref}")
        source_checkpoint = (
            session.get(Checkpoint, str(source_checkpoint_id)) if source_checkpoint_id else None
        )
        if source_checkpoint_id and source_checkpoint is None:
            raise ValueError(f"unknown source_checkpoint_id: {source_checkpoint_id}")
        artifact_id = _resolve_artifact_id(
            session,
            raw_spec,
            artifact_by_ref,
            source_checkpoint.artifact_id if source_checkpoint else default_artifact.id,
        )
        if model is None:
            model = ModelVersion(
                project_id=run.project_id,
                name=name,
                version=version,
                kind=str(raw_spec.get("kind") or "base"),
                parent_model_version_id=raw_spec.get("parent_model_version_id"),
                source_checkpoint_id=source_checkpoint.id if source_checkpoint else None,
                artifact_id=artifact_id,
                tokenizer_version_id=raw_spec.get("tokenizer_version_id"),
                chat_template_version_id=raw_spec.get("chat_template_version_id"),
                objective_kind=raw_spec.get("objective_kind"),
                parameter_count=int(raw_spec.get("parameter_count", 0) or 0),
                status=str(raw_spec.get("status") or "candidate"),
                is_best=bool(raw_spec.get("is_best", False)),
                is_candidate=bool(raw_spec.get("is_candidate", True)),
                is_official=bool(raw_spec.get("is_official", False)),
                metadata_json={**(raw_spec.get("metadata") or {}), "task_id": task.id},
            )
            session.add(model)
            session.flush()
        ref = str(raw_spec.get("ref") or f"model-{index}")
        model_by_ref[ref] = model.id
        if raw_spec.get("parent_model_ref"):
            deferred_parents.append((model, str(raw_spec["parent_model_ref"])))
        models.append(model)

    for model, parent_ref in deferred_parents:
        parent_id = model_by_ref.get(parent_ref)
        if parent_id is None:
            raise ValueError(f"unknown parent_model_ref: {parent_ref}")
        model.parent_model_version_id = parent_id
    official_models = [model for model in models if model.is_official]
    if official_models:
        official = official_models[-1]
        session.execute(
            update(ModelVersion)
            .where(
                ModelVersion.project_id == run.project_id,
                ModelVersion.id != official.id,
            )
            .values(is_official=False)
        )
        official.is_official = True
        official.is_candidate = False
        project = session.get(Project, run.project_id)
        if project:
            project.current_model_version_id = official.id
    return checkpoints, models


def _upsert_dataset_version_index(
    session: Session,
    version: DatasetVersion,
    analysis: DatasetAnalysisRevision,
    *,
    index_kind: str,
    artifact_id: str,
    format: str = "",
    metadata: dict[str, Any] | None = None,
) -> DatasetVersionIndex:
    version_index = session.scalar(
        select(DatasetVersionIndex).where(
            DatasetVersionIndex.dataset_version_id == version.id,
            DatasetVersionIndex.analysis_revision_id == analysis.id,
            DatasetVersionIndex.index_kind == index_kind,
        )
    )
    if version_index is None:
        version_index = DatasetVersionIndex(
            dataset_version_id=version.id,
            analysis_revision_id=analysis.id,
            index_kind=index_kind,
            artifact_id=artifact_id,
        )
        session.add(version_index)
    version_index.artifact_id = artifact_id
    version_index.format = format
    version_index.parser_id = analysis.parser_id
    version_index.parser_version = analysis.parser_version
    version_index.status = "ready"
    version_index.metadata_json = metadata or {}
    return version_index


def _analysis_for_task(
    session: Session, task: Task, version: DatasetVersion
) -> DatasetAnalysisRevision:
    analysis = session.scalar(
        select(DatasetAnalysisRevision).where(DatasetAnalysisRevision.task_id == task.id)
    )
    if analysis is None:
        analysis = create_analysis_revision(
            session, version, task_id=task.id, status="running"
        )
    elif analysis.dataset_version_id != version.id:
        raise ValueError("dataset analysis revision points to a different data version")
    return analysis


def _populate_analysis_result(
    analysis: DatasetAnalysisRevision,
    version: DatasetVersion,
    snapshot: Artifact,
    record_manifest: Artifact | None,
    result: dict[str, Any],
) -> None:
    analysis.status = "succeeded"
    analysis.error = {}
    analysis.snapshot_artifact_id = snapshot.id
    analysis.snapshot_content_hash = snapshot.content_hash
    analysis.record_manifest_artifact_id = record_manifest.id if record_manifest else None
    analysis.raw_size_bytes = int(result.get("raw_size_bytes") or 0)
    analysis.document_count = int(result.get("document_count") or 0)
    analysis.record_count = int(
        result.get("record_count")
        if result.get("record_count") is not None
        else analysis.document_count
    )
    analysis.token_estimate = int(result.get("token_estimate") or 0)
    analysis.parser_id = str(result.get("parser_id") or version.parser_id or "") or None
    analysis.parser_version = (
        str(result.get("parser_version") or version.parser_version or "") or None
    )
    analysis.parser_status = str(result.get("parser_status") or "succeeded")
    analysis.parser_config = result.get("parser_config") or version.parser_config or {}
    analysis.detected_formats = result.get("detected_formats") or []
    analysis.data_schema = result.get("data_schema") or {}
    analysis.statistics = result.get("statistics") or {}
    analysis.usage_capabilities = result.get("usage_capabilities") or []
    analysis.train_ready = bool(result.get("train_ready", False))
    analysis.automatic_quality_score = (
        float(result["automatic_quality_score"])
        if result.get("automatic_quality_score") is not None
        else None
    )
    analysis.quality_report = result.get("quality_report") or {}
    analysis.sample_preview = result.get("sample_preview") or []
    analysis.source_snapshot = result.get("source_snapshot") or {}


def _copy_analysis_to_legacy_version(
    version: DatasetVersion, analysis: DatasetAnalysisRevision
) -> None:
    version.record_manifest_artifact_id = analysis.record_manifest_artifact_id
    version.raw_size_bytes = analysis.raw_size_bytes
    version.document_count = analysis.document_count
    version.record_count = analysis.record_count
    version.token_estimate = analysis.token_estimate
    version.parser_id = analysis.parser_id
    version.parser_version = analysis.parser_version
    version.parser_status = analysis.parser_status
    version.parser_config = analysis.parser_config
    version.detected_formats = analysis.detected_formats
    version.data_schema = analysis.data_schema
    version.statistics = analysis.statistics
    version.usage_capabilities = analysis.usage_capabilities
    version.train_ready = analysis.train_ready
    version.automatic_quality_score = analysis.automatic_quality_score
    version.quality_report = analysis.quality_report
    version.sample_preview = analysis.sample_preview
    version.source_snapshot = analysis.source_snapshot


def finalize_task_domain(
    session: Session, task: Task, attempt: TaskAttempt, result: dict[str, Any]
) -> None:
    if not task.project_id:
        raise ValueError("executable tasks must belong to a project")
    if task.kind == "data_query":
        payload_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        if len(payload_bytes) > 10 * 1024 * 1024:
            raise ValueError("dataset query result exceeds the 10 MiB control-plane limit")
        task.result = result
        return
    if task.kind == "train":
        result = normalize_training_result_tokens(task.payload or {}, result)
        result = normalize_training_result_model_versions(session, task, result)
    elif task.kind == "tokenizer_train":
        tokenizer_config = (task.payload or {}).get("config") or {}
        gate_evaluation = evaluate_tokenizer_gate_policy(
            tokenizer_config.get("gate_policy"), result.get("metrics")
        )
        result = {**result, **gate_evaluation}
    payload_bytes = json.dumps(
        {"task_id": task.id, "kind": task.kind, "result": result}, ensure_ascii=False, indent=2
    ).encode("utf-8")
    artifact_kind = {
        "train": "checkpoint",
        "data_tokenize": "tokenized_dataset",
        "data_tokenize_corpus": "tokenized_corpus",
        "training_data_index": "training_data_index",
        "supervision_scan": "supervision_record_index",
        "supervision_corpus": "supervision_corpus",
        "supervision_training_index": "supervision_training_index",
        "tokenizer_train": "tokenizer",
        "evaluate": "evaluation_report",
        "quantize": "quantized_model",
        "release_export": "release_runtime",
        "release_publish": "release_delivery_report",
        "data_label": "data_label_report",
    }.get(task.kind, "task_output")
    report_artifact = artifact_storage.put_bytes(
        session,
        project_id=task.project_id,
        kind=artifact_kind,
        payload=payload_bytes,
        filename=f"{task.id}.json",
        media_type="application/json",
        created_by_task_attempt_id=attempt.id,
        metadata={"task_kind": task.kind},
    )
    uploaded, artifact_by_ref, output_records = _collect_result_artifacts(
        session, task, attempt, result
    )
    expected_kind = {
        "code_import": "source_code",
        "data_import": "dataset_snapshot",
        "data_generate": "dataset_snapshot",
        "data_convert": "dataset_snapshot",
        "data_process": "dataset_snapshot",
        "dataset_materialize": "dataset_snapshot",
        "mixture_resolve": "data_mixture_manifest",
        "data_tokenize": "tokenized_dataset",
        "data_tokenize_corpus": "tokenized_corpus",
        "training_data_index": "training_data_index",
        "supervision_scan": "supervision_record_index",
        "supervision_corpus": "supervision_corpus",
        "supervision_training_index": "supervision_training_index",
        "train": "checkpoint",
        "tokenizer_train": "tokenizer",
        "evaluate": "evaluation_report",
        "quantize": "quantized_model",
        "release_export": "release_runtime",
        "release_publish": "release_delivery_report",
        "model_weight_download": "model_weights",
        "data_label": "data_label_snapshot",
    }.get(task.kind)
    artifact = next(
        (candidate for candidate in uploaded if candidate.kind == expected_kind),
        uploaded[0] if uploaded else report_artifact,
    )
    task.result = {
        **result,
        "artifact_id": artifact.id,
        "report_artifact_id": report_artifact.id,
        "output_artifact_ids": [candidate.id for candidate in uploaded],
        "output_artifacts": output_records,
    }
    if task.adapter_id == "builtin.storage-maintenance":
        from .storage_lifecycle import finalize_storage_cleanup
        finalize_storage_cleanup(session, task, result)

    if task.adapter_id == "builtin.suite-manager":
        from .suite_packages import finalize_operation
        finalize_operation(session, task, result)
    elif task.adapter_id == "builtin.suite-runner":
        from .suite_packages import finalize_run
        finalize_run(session, task)

    if task.kind == "data_label":
        operation = session.scalar(
            select(DatasetLabelOperation).where(DatasetLabelOperation.task_id == task.id)
        )
        if operation is None:
            raise ValueError("data-label task has no operation record")
        current = session.scalar(
            select(DatasetLabelSnapshot)
            .where(DatasetLabelSnapshot.dataset_version_id == operation.dataset_version_id)
            .order_by(DatasetLabelSnapshot.version.desc())
            .limit(1)
        )
        if (current.id if current else None) != operation.base_snapshot_id:
            raise ValueError("data-label base snapshot is stale")
        selection_id = artifact_by_ref.get("label_selection")
        delta_id = artifact_by_ref.get("label_delta")
        snapshot_id = artifact_by_ref.get("label_snapshot")
        selection_artifact = session.get(Artifact, selection_id) if selection_id else None
        delta_artifact = session.get(Artifact, delta_id) if delta_id else None
        snapshot_artifact = session.get(Artifact, snapshot_id) if snapshot_id else None
        if selection_artifact is None or selection_artifact.kind != "data_label_selection":
            raise ValueError("data-label task did not produce a selection artifact")
        if delta_artifact is None or delta_artifact.kind != "data_label_delta":
            raise ValueError("data-label task did not produce a delta artifact")
        if snapshot_artifact is None or snapshot_artifact.kind != "data_label_snapshot":
            raise ValueError("data-label task did not produce a snapshot artifact")
        snapshot_version = int((task.payload or {}).get("output_snapshot_version") or 0)
        if snapshot_version != (current.version if current else 0) + 1:
            raise ValueError("data-label snapshot version is not the next immutable version")
        label_snapshot = DatasetLabelSnapshot(
            dataset_version_id=operation.dataset_version_id,
            version=snapshot_version,
            parent_snapshot_id=current.id if current else None,
            operation_id=operation.id,
            artifact_id=snapshot_artifact.id,
            content_hash=snapshot_artifact.content_hash,
            assignment_count=int(result.get("assignment_count") or 0),
            labeled_record_count=int(result.get("labeled_record_count") or 0),
            summary={
                "selection_count": int(result.get("selection_count") or 0),
                "scope": result.get("scope"),
                "action": result.get("action"),
                "engine": result.get("engine"),
            },
            immutable=True,
        )
        session.add(label_snapshot)
        session.flush()
        operation.output_snapshot_id = label_snapshot.id
        operation.selection_artifact_id = selection_artifact.id
        operation.delta_artifact_id = delta_artifact.id
        operation.selection_count = int(result.get("selection_count") or 0)
        operation.status = "succeeded"
        operation.error = {}
        task.result = {**task.result, "label_snapshot_id": label_snapshot.id}

    elif task.kind == "plugin_install":
        installation_id = str((task.payload or {}).get("installation_id") or "")
        installation = (
            session.get(WorkerPluginInstallation, installation_id)
            if installation_id
            else None
        )
        if installation is None:
            raise ValueError("plugin installation task has no installation record")
        plugin = (task.payload or {}).get("plugin") or {}
        plugin_version_id = str(plugin.get("plugin_version_id") or "")
        if plugin_version_id != installation.target_plugin_version_id:
            raise ValueError("plugin installation result does not match its frozen target")
        manifest = result.get("installation") or {}
        if (
            manifest.get("plugin_id") != plugin.get("id")
            or manifest.get("version") != plugin.get("version")
            or manifest.get("entrypoint") != plugin.get("entrypoint")
            or manifest.get("runner") != ((plugin.get("runtime") or {}).get("runner"))
            or manifest.get("isolation")
            != ((plugin.get("runtime") or {}).get("isolation"))
            or manifest.get("requirements")
            != ((plugin.get("runtime") or {}).get("requirements") or [])
        ):
            raise ValueError("plugin installation manifest does not match its frozen target")
        installed = [
            item
            for item in (installation.installed_versions or [])
            if isinstance(item, dict) and item.get("plugin_version_id") != plugin_version_id
        ]
        installed.append(
            {
                "plugin_version_id": plugin_version_id,
                "version": plugin.get("version"),
                "entrypoint": plugin.get("entrypoint"),
                "manifest_path": manifest.get("manifest_path"),
                "python": manifest.get("python"),
                "runner": manifest.get("runner"),
                "isolation": manifest.get("isolation"),
                "requirements": manifest.get("requirements") or [],
                "package_version": manifest.get("package_version"),
                "runtime_versions": manifest.get("runtime_versions") or {},
                "installed_at": manifest.get("installed_at"),
            }
        )
        installation.installed_versions = installed
        installation.current_plugin_version_id = plugin_version_id
        installation.target_plugin_version_id = plugin_version_id
        installation.state = "ready"
        installation.runtime = manifest
        installation.last_error = {}
        installation.last_verified_at = utcnow()
        worker = session.get(WorkerNode, installation.worker_id)
        if worker is not None:
            worker.capabilities = list(
                dict.fromkeys(
                    [
                        *(worker.capabilities or []),
                        str(plugin.get("entrypoint") or ""),
                        f"{plugin.get('entrypoint')}@{plugin.get('version')}",
                    ]
                )
            )

    elif task.kind == "code_import":
        version = session.scalar(select(CodeVersion).where(CodeVersion.task_id == task.id))
        if version:
            version.state = "ready"
            version.artifact_id = artifact.id
            version.git_commit = result.get("git_commit") or None
            version.source_snapshot = {
                **(version.source_snapshot or {}),
                **(result.get("source_snapshot") or {}),
                "content_hash": artifact.content_hash,
                "artifact_id": artifact.id,
            }
            source = session.get(CodeSource, version.source_id)
            if source and source.current_version_id == version.id:
                source.status = "ready"

    elif task.kind == "model_weight_download":
        version = session.scalar(
            select(ModelWeightVersion).where(ModelWeightVersion.task_id == task.id)
        )
        if version:
            output_id = artifact_by_ref.get("model_weight")
            output = session.get(Artifact, output_id) if output_id else artifact
            if output is None or output.kind != "model_weights":
                raise ValueError("model-weight download did not produce a model_weights artifact")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("model-weight identity does not match the frozen version")
            version.state = "ready"
            version.artifact_id = output.id
            version.source_manifest = {
                **(version.source_manifest or {}),
                **(result.get("source_manifest") or {}),
                "content_hash": output.content_hash,
            }
            version.format_summary = result.get("format_summary") or {}
            version.runtime_capabilities = result.get("runtime_capabilities") or []
            version.failure_reason = None
            asset = session.get(ModelWeightAsset, version.asset_id)
            if asset:
                asset.current_version_id = version.id

    elif task.kind == "model_weight_evict":
        version_id = str((task.payload or {}).get("model_weight_version_id") or "")
        version = session.get(ModelWeightVersion, version_id) if version_id else None
        location_id = str((task.payload or {}).get("location_id") or "")
        location = session.get(ArtifactLocation, location_id) if location_id else None
        if location:
            artifact_id = location.artifact_id
            session.delete(location)
            session.flush()
            remaining = session.scalar(
                select(ArtifactLocation.id).where(
                    ArtifactLocation.artifact_id == artifact_id,
                    ArtifactLocation.availability_state == "available",
                )
            )
            if not remaining:
                output = session.get(Artifact, artifact_id)
                if output:
                    output.availability_state = "unavailable"
                if version:
                    version.state = "evicted"
            elif version:
                version.state = "ready"

    elif task.kind == "dataset_materialize":
        finalize_dataset_materialization(
            session, task, attempt, result, artifact_by_ref
        )

    elif task.kind in {"data_import", "data_generate"}:
        analysis = session.scalar(
            select(DatasetAnalysisRevision).where(DatasetAnalysisRevision.task_id == task.id)
        )
        version = (
            session.get(DatasetVersion, analysis.dataset_version_id)
            if analysis is not None
            else session.scalar(select(DatasetVersion).where(DatasetVersion.task_id == task.id))
        )
        if version:
            is_initial_import = version.task_id == task.id
            if is_initial_import:
                version.state = "ready"
                version.artifact_id = artifact.id
            elif version.artifact_id != artifact.id:
                raise ValueError(
                    "dataset reanalysis returned a different snapshot; import it as a new data version"
                )
            if task.kind == "data_generate":
                manifest = session.get(Artifact, artifact_by_ref.get("record_manifest"))
                empty_generation = (result.get("document_count") == 0
                                    and (result.get("source_snapshot") or {}).get("generation_empty") is True
                                    and not result.get("train_ready"))
                if artifact.kind != "dataset_snapshot" or manifest is None or manifest.kind != "dataset_record_manifest" or (not result.get("train_ready") and not empty_generation):
                    raise ValueError("Generated data requires a train-ready snapshot and record manifest")
            record_manifest_id = artifact_by_ref.get("record_manifest")
            record_manifest = (
                session.get(Artifact, record_manifest_id) if record_manifest_id else None
            )
            analysis = _analysis_for_task(session, task, version)
            _populate_analysis_result(analysis, version, artifact, record_manifest, result)
            if is_initial_import:
                version.modality = str(result.get("modality") or version.modality or "text")
                _copy_analysis_to_legacy_version(version, analysis)
                if version.materialization_kind == "download":
                    spec = dict(version.materialization_spec or {})
                    replay = dict(spec.get("replay_payload") or {})
                    transfer = (result.get("source_snapshot") or {}).get("transfer") or {}
                    if replay.get("source_type") == "git" and transfer.get("commit_hash"):
                        source_config = dict(replay.get("source_config") or {})
                        source_config["commit_hash"] = str(transfer["commit_hash"])
                        replay["source_config"] = source_config
                        spec["replay_payload"] = replay
                        version.materialization_spec = spec
                ensure_version_primary_location(
                    session, version, artifact, preferred_worker_id=attempt.worker_id
                )
            for index_result in result.get("indexes") or []:
                index_artifact_id = artifact_by_ref.get(str(index_result.get("ref") or ""))
                if not index_artifact_id:
                    continue
                index_kind = str(index_result.get("kind") or "").strip()
                if not index_kind:
                    continue
                _upsert_dataset_version_index(
                    session,
                    version,
                    analysis,
                    index_kind=index_kind,
                    artifact_id=index_artifact_id,
                    format=str(index_result.get("format") or ""),
                    metadata=index_result.get("metadata") or {},
                )
            dataset = session.get(Dataset, version.dataset_id)
            if dataset and task.kind == "data_generate":
                dataset.current_version_id = version.id
            if dataset and dataset.current_version_id == version.id:
                dataset.status = (
                    "ready"
                    if analysis.train_ready or analysis.parser_status == "unavailable"
                    else "needs_review"
                )

    elif task.kind == "data_convert":
        run = session.scalar(select(DataConversionRun).where(DataConversionRun.task_id == task.id))
        if run:
            version = session.get(DatasetVersion, run.output_dataset_version_id)
            if version is None:
                raise ValueError("data conversion run has no output dataset version")
            snapshot_id = artifact_by_ref.get("dataset")
            snapshot = session.get(Artifact, snapshot_id) if snapshot_id else artifact
            if snapshot is None or snapshot.kind != "dataset_snapshot":
                raise ValueError("data conversion did not produce a dataset snapshot")
            record_manifest_id = artifact_by_ref.get("record_manifest")
            record_manifest = (
                session.get(Artifact, record_manifest_id) if record_manifest_id else None
            )
            if record_manifest is None or record_manifest.kind != "dataset_record_manifest":
                raise ValueError("data conversion did not produce a dataset record manifest")
            conversion_report_id = artifact_by_ref.get("conversion_report")
            conversion_report = (
                session.get(Artifact, conversion_report_id) if conversion_report_id else None
            )
            if conversion_report is None or conversion_report.kind != "data_conversion_report":
                raise ValueError("data conversion did not produce a conversion report")
            version.state = "ready"
            version.artifact_id = snapshot.id
            version.record_manifest_artifact_id = record_manifest.id
            version.raw_size_bytes = int(result.get("raw_size_bytes") or 0)
            version.document_count = int(result.get("document_count") or 0)
            version.token_estimate = int(result.get("token_estimate") or 0)
            version.automatic_quality_score = (
                float(result["automatic_quality_score"])
                if result.get("automatic_quality_score") is not None
                else None
            )
            version.quality_report = result.get("quality_report") or {}
            version.sample_preview = result.get("sample_preview") or []
            version.source_snapshot = result.get("source_snapshot") or {}
            version.modality = "text"
            version.record_count = version.document_count
            version.parser_id = TEXT_PARSER_ID
            version.parser_version = TEXT_PARSER_VERSION
            version.parser_status = "succeeded"
            version.data_schema = {
                "type": "text_records",
                "record_manifest_format": "llm.dataset-records.v1",
            }
            version.statistics = {
                "record_count": version.document_count,
                "document_count": version.document_count,
                "token_estimate": version.token_estimate,
                "raw_size_bytes": version.raw_size_bytes,
            }
            version.usage_capabilities = (
                list(TEXT_USAGE_CAPABILITIES) if version.document_count > 0 else ["source_archive"]
            )
            version.train_ready = bool(result.get("train_ready", False))
            ensure_version_primary_location(
                session, version, snapshot, preferred_worker_id=attempt.worker_id
            )
            analysis = _analysis_for_task(session, task, version)
            _populate_analysis_result(
                analysis,
                version,
                snapshot,
                record_manifest,
                {
                    **result,
                    "record_count": version.record_count,
                    "parser_id": version.parser_id,
                    "parser_version": version.parser_version,
                    "parser_status": version.parser_status,
                    "data_schema": version.data_schema,
                    "statistics": version.statistics,
                    "usage_capabilities": version.usage_capabilities,
                },
            )
            _upsert_dataset_version_index(
                session,
                version,
                analysis,
                index_kind="record_manifest",
                artifact_id=record_manifest.id,
                format="llm.dataset-records.v1",
            )
            run.status = "succeeded"
            run.engine_version = str(result.get("engine_version") or "") or None
            run.summary = result.get("conversion_summary") or {}
            run.failure_reason = None
            dataset = session.get(Dataset, version.dataset_id)
            if dataset:
                dataset.current_version_id = version.id
                dataset.status = "ready" if version.train_ready else "needs_review"

    elif task.kind == "data_process":
        run = session.scalar(select(DataProcessingRun).where(DataProcessingRun.task_id == task.id))
        if run:
            version = session.get(DatasetVersion, run.output_dataset_version_id)
            if version is None:
                raise ValueError("data processing run has no output dataset version")
            snapshot_id = artifact_by_ref.get("dataset")
            snapshot = session.get(Artifact, snapshot_id) if snapshot_id else artifact
            if snapshot is None or snapshot.kind != "dataset_snapshot":
                raise ValueError("data processing did not produce a dataset snapshot")
            record_manifest_id = artifact_by_ref.get("record_manifest")
            record_manifest = (
                session.get(Artifact, record_manifest_id) if record_manifest_id else None
            )
            if record_manifest is None or record_manifest.kind != "dataset_record_manifest":
                raise ValueError("data processing did not produce a dataset record manifest")
            version.state = "ready"
            version.artifact_id = snapshot.id
            version.record_manifest_artifact_id = record_manifest.id
            version.raw_size_bytes = int(result.get("raw_size_bytes") or 0)
            version.document_count = int(result.get("document_count") or 0)
            version.token_estimate = int(result.get("token_estimate") or 0)
            version.automatic_quality_score = (
                float(result["automatic_quality_score"])
                if result.get("automatic_quality_score") is not None
                else None
            )
            version.quality_report = result.get("quality_report") or {}
            version.sample_preview = result.get("sample_preview") or []
            version.source_snapshot = result.get("source_snapshot") or {}
            version.modality = "text"
            version.record_count = version.document_count
            version.parser_id = TEXT_PARSER_ID
            version.parser_version = TEXT_PARSER_VERSION
            version.parser_status = "succeeded"
            version.data_schema = {
                "type": "text_records",
                "record_manifest_format": "llm.dataset-records.v1",
            }
            version.statistics = {
                "record_count": version.document_count,
                "document_count": version.document_count,
                "token_estimate": version.token_estimate,
                "raw_size_bytes": version.raw_size_bytes,
            }
            version.usage_capabilities = (
                list(TEXT_USAGE_CAPABILITIES) if version.document_count > 0 else ["source_archive"]
            )
            version.train_ready = bool(result.get("train_ready", False))
            ensure_version_primary_location(
                session, version, snapshot, preferred_worker_id=attempt.worker_id
            )
            analysis = _analysis_for_task(session, task, version)
            _populate_analysis_result(
                analysis,
                version,
                snapshot,
                record_manifest,
                {
                    **result,
                    "record_count": version.record_count,
                    "parser_id": version.parser_id,
                    "parser_version": version.parser_version,
                    "parser_status": version.parser_status,
                    "data_schema": version.data_schema,
                    "statistics": version.statistics,
                    "usage_capabilities": version.usage_capabilities,
                },
            )
            _upsert_dataset_version_index(
                session,
                version,
                analysis,
                index_kind="record_manifest",
                artifact_id=record_manifest.id,
                format="llm.dataset-records.v1",
            )
            run.status = "succeeded"
            run.engine_version = str(result.get("engine_version") or "") or None
            run.summary = result.get("processing_summary") or {}
            run.failure_reason = None
            dataset = session.get(Dataset, version.dataset_id)
            if dataset:
                dataset.current_version_id = version.id
                dataset.status = "ready" if version.train_ready else "needs_review"

    elif task.kind == "mixture_resolve":
        version = session.scalar(
            select(DataMixtureVersion).where(DataMixtureVersion.task_id == task.id)
        )
        if version:
            manifest_id = artifact_by_ref.get("mixture_manifest")
            manifest = session.get(Artifact, manifest_id) if manifest_id else artifact
            if manifest is None:
                raise ValueError("mixture resolution did not produce a manifest artifact")
            version.status = "ready"
            version.locked = True
            version.manifest_artifact_id = manifest.id
            version.manifest_hash = manifest.content_hash
            version.resolver_version = str(
                result.get("resolver_version") or version.resolver_version
            )
            version.sampler_version = str(result.get("sampler_version") or version.sampler_version)
            version.resolved_document_count = int(result.get("resolved_document_count") or 0)
            version.resolved_token_estimate = int(result.get("resolved_token_estimate") or 0)
            version.entries = result.get("resolved_entries") or version.entries
            version.failure_reason = None

    elif task.kind == "tokenizer_train":
        tokenizer = session.scalar(
            select(TokenizerVersion).where(TokenizerVersion.task_id == task.id)
        )
        if tokenizer:
            tokenizer.status = "passed" if result.get("gate_passed", True) else "failed_gate"
            tokenizer.metrics = result.get("metrics", {})
            tokenizer.artifact_id = artifact.id

    elif task.kind == "data_tokenize":
        version = session.scalar(
            select(TokenizedDatasetVersion).where(
                TokenizedDatasetVersion.task_id == task.id
            )
        )
        if version:
            tokenized_id = artifact_by_ref.get("tokenized_dataset")
            tokenized_artifact = (
                session.get(Artifact, tokenized_id) if tokenized_id else artifact
            )
            if tokenized_artifact is None or tokenized_artifact.kind != "tokenized_dataset":
                raise ValueError("pretokenization did not produce a tokenized_dataset artifact")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("pretokenization identity does not match the frozen version")
            version.status = "ready"
            version.artifact_id = tokenized_artifact.id
            version.manifest_hash = tokenized_artifact.content_hash
            version.token_count = int(result.get("token_count") or 0)
            version.sequence_count = int(result.get("sequence_count") or 0)
            version.shard_count = int(result.get("shard_count") or 0)
            version.size_bytes = int(result.get("size_bytes") or 0)
            version.failure_reason = None

    elif task.kind == "data_tokenize_corpus":
        version = session.scalar(
            select(TokenizedCorpusVersion).where(
                TokenizedCorpusVersion.task_id == task.id
            )
        )
        if version:
            corpus_id = artifact_by_ref.get("tokenized_corpus")
            corpus_artifact = session.get(Artifact, corpus_id) if corpus_id else artifact
            if corpus_artifact is None or corpus_artifact.kind != "tokenized_corpus":
                raise ValueError("tokenization did not produce a tokenized_corpus artifact")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("tokenized corpus identity does not match the frozen version")
            version.status = "ready"
            version.artifact_id = corpus_artifact.id
            version.manifest_hash = corpus_artifact.content_hash
            version.document_count = int(result.get("document_count") or 0)
            version.token_count = int(result.get("token_count") or 0)
            version.shard_count = int(result.get("shard_count") or 0)
            version.size_bytes = int(result.get("size_bytes") or 0)
            version.warnings = list(result.get("warnings") or [])
            version.failure_reason = None

    elif task.kind == "training_data_index":
        version = session.scalar(
            select(TrainingDataViewVersion).where(
                TrainingDataViewVersion.task_id == task.id
            )
        )
        if version:
            view_id = artifact_by_ref.get("training_data_view")
            view_artifact = session.get(Artifact, view_id) if view_id else artifact
            if view_artifact is None or view_artifact.kind != "training_data_index":
                raise ValueError("index build did not produce a training_data_index artifact")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("training data view identity does not match the frozen version")
            version.status = "ready"
            version.artifact_id = view_artifact.id
            version.manifest_hash = view_artifact.content_hash
            version.token_count = int(result.get("token_count") or 0)
            version.sequence_count = int(result.get("sequence_count") or 0)
            version.size_bytes = int(result.get("size_bytes") or 0)
            version.warnings = list(result.get("warnings") or [])
            version.failure_reason = None

    elif task.kind == "supervision_scan":
        version = session.scalar(
            select(SupervisionDatasetVersion).where(
                SupervisionDatasetVersion.task_id == task.id
            )
        )
        if version:
            output_id = artifact_by_ref.get("supervision_record_index")
            output = session.get(Artifact, output_id) if output_id else artifact
            if output is None or output.kind != "supervision_record_index":
                raise ValueError("supervision scan did not produce a record index")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("supervision record identity does not match its version")
            version.status = "ready"
            version.artifact_id = output.id
            version.manifest_hash = output.content_hash
            version.accepted_record_count = int(result.get("accepted_record_count") or 0)
            version.rejected_record_count = int(result.get("rejected_record_count") or 0)
            version.validation_report = result.get("validation_report") or {}
            version.sample_preview = result.get("sample_preview") or []
            version.warnings = result.get("warnings") or []
            version.failure_reason = None

    elif task.kind == "supervision_corpus":
        version = session.scalar(
            select(SupervisionCorpusVersion).where(
                SupervisionCorpusVersion.task_id == task.id
            )
        )
        if version:
            output_id = artifact_by_ref.get("supervision_corpus")
            output = session.get(Artifact, output_id) if output_id else artifact
            if output is None or output.kind != "supervision_corpus":
                raise ValueError("supervision corpus build did not produce a corpus artifact")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("supervision corpus identity does not match its version")
            version.status = "ready"
            version.artifact_id = output.id
            version.manifest_hash = output.content_hash
            version.example_count = int(result.get("example_count") or 0)
            version.pair_count = int(result.get("pair_count") or 0)
            version.raw_token_count = int(result.get("raw_token_count") or 0)
            version.supervised_token_count = int(result.get("supervised_token_count") or 0)
            version.size_bytes = int(result.get("size_bytes") or 0)
            version.warnings = result.get("warnings") or []
            version.failure_reason = None

    elif task.kind == "supervision_training_index":
        version = session.scalar(
            select(SupervisionTrainingViewVersion).where(
                SupervisionTrainingViewVersion.task_id == task.id
            )
        )
        if version:
            output_id = artifact_by_ref.get("supervision_training_view")
            output = session.get(Artifact, output_id) if output_id else artifact
            if output is None or output.kind != "supervision_training_index":
                raise ValueError("supervision index build did not produce a training index")
            if str(result.get("identity_hash") or "") != version.identity_hash:
                raise ValueError("supervision training view identity does not match its version")
            version.status = "ready"
            version.artifact_id = output.id
            version.manifest_hash = output.content_hash
            version.example_count = int(result.get("example_count") or 0)
            version.pair_count = int(result.get("pair_count") or 0)
            version.raw_token_count = int(result.get("raw_token_count") or 0)
            version.supervised_token_count = int(result.get("supervised_token_count") or 0)
            version.size_bytes = int(result.get("size_bytes") or 0)
            version.warnings = result.get("warnings") or []
            version.failure_reason = None

    elif task.kind == "train":
        run = session.scalar(select(TrainingRun).where(TrainingRun.task_id == task.id))
        if run:
            run.status = "succeeded"
            run.ended_at = utcnow()
            run.current_step = int(result.get("step", run.current_step or 0) or 0)
            run.current_tokens = int(result.get("tokens", run.current_tokens or 0) or 0)
            from .training_metrics import merge_training_metrics

            run.latest_metrics = merge_training_metrics(
                run.latest_metrics, result.get("metrics")
            )
            checkpoints, models = _finalize_training_outputs(
                session, task, run, result, artifact_by_ref, artifact
            )
            if checkpoints and isinstance(checkpoints[-1].training_position, dict):
                counters = checkpoints[-1].training_position.get("counters")
                if isinstance(counters, dict):
                    run.progress_counters = {
                        **(run.progress_counters or {}),
                        **counters,
                    }
            task.result = {
                **task.result,
                "checkpoint_ids": [checkpoint.id for checkpoint in checkpoints],
                "model_version_ids": [model.id for model in models],
                "checkpoint_id": checkpoints[-1].id,
                "model_version_id": models[-1].id,
            }
            revision = session.get(PlanRevision, run.plan_revision_id)
            if revision:
                plan = session.get(TrainingPlan, revision.plan_id)
                if plan:
                    plan.status = "succeeded"

    elif task.kind == "evaluate":
        evaluation = session.scalar(select(EvaluationRun).where(EvaluationRun.task_id == task.id))
        if evaluation:
            evaluation.status = "succeeded"
            evaluation.summary = result.get("summary", {})
            evaluation.sample_results = result.get("samples", [])
            evaluation.benchmark_signature = result.get(
                "benchmark_signature", evaluation.benchmark_signature or {}
            )
            evaluation.compatibility_hash = evaluation.benchmark_signature.get(
                "compatibility_hash", evaluation.compatibility_hash
            )

    elif task.kind == "quantize":
        quantization = session.scalar(
            select(QuantizationRun).where(QuantizationRun.task_id == task.id)
        )
        if quantization:
            quantization.status = "succeeded"
            quantization.metrics_before = result.get("metrics_before", {})
            quantization.metrics_after = result.get("metrics_after", {})
            quantization.artifact_id = artifact.id
            quantization.started_at = task.started_at
            quantization.ended_at = utcnow()

    elif task.kind == "release_export":
        build = session.scalar(
            select(ReleaseBuildRun).where(ReleaseBuildRun.task_id == task.id)
        )
        if build is None:
            raise ValueError("release export task has no build run")
        runtime_id = artifact_by_ref.get("release_runtime")
        runtime_artifact = session.get(Artifact, runtime_id) if runtime_id else None
        if runtime_artifact is None or runtime_artifact.kind != "release_runtime":
            raise ValueError("release exporter did not produce a release_runtime artifact")
        runtime = result.get("runtime") or {}
        runtime_format = str(runtime.get("format") or "")
        if not runtime_format:
            raise ValueError("release exporter did not return a runtime format")
        source = build.source_snapshot or {}
        existing = session.scalar(
            select(Release).where(
                Release.project_id == build.project_id,
                Release.version == build.release_version,
            )
        )
        if existing is not None:
            raise ValueError("release version already exists")
        release = Release(
            project_id=build.project_id,
            name=build.release_name,
            version=build.release_version,
            model_version_id=str(source.get("model_version_id") or ""),
            quantization_run_id=(
                str(source.get("quantization_run_id"))
                if source.get("quantization_run_id")
                else None
            ),
            target="runtime",
            model_card=build.model_card_snapshot,
            license_manifest=build.license_snapshot or {},
            gate_result=build.gate_snapshot or {},
            release_kind="runtime",
            release_build_run_id=build.id,
            runtime_artifact_id=runtime_artifact.id,
            source_snapshot=source,
            runtime_snapshot={
                **runtime,
                "artifact_id": runtime_artifact.id,
                "artifact_content_hash": runtime_artifact.content_hash,
                "plugin": build.plugin_snapshot or {},
            },
            status="published",
        )
        session.add(release)
        session.flush()
        manifest = {
            "schema": "forgelm-release-manifest-v2",
            "release_id": release.id,
            "project_id": release.project_id,
            "name": release.name,
            "version": release.version,
            "source": release.source_snapshot,
            "runtime": release.runtime_snapshot,
            "model_card": release.model_card,
            "license_manifest": release.license_manifest,
            "gate_result": release.gate_result,
            "release_build_run_id": build.id,
        }
        manifest_artifact = artifact_storage.put_bytes(
            session,
            project_id=release.project_id,
            kind="release_manifest",
            payload=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            filename=f"{release.name}-{release.version}.json",
            media_type="application/json",
            created_by_task_attempt_id=attempt.id,
            metadata={
                "schema": "forgelm-release-manifest-v2",
                "release_id": release.id,
                "runtime_artifact_id": runtime_artifact.id,
            },
        )
        release.artifact_id = manifest_artifact.id
        release.manifest_artifact_id = manifest_artifact.id
        build.status = "succeeded"
        build.runtime_artifact_id = runtime_artifact.id
        build.validation_report = result.get("validation_report") or {}
        build.error = {}
        build.started_at = task.started_at
        build.ended_at = utcnow()
        session.add(
            AuditEvent(
                project_id=release.project_id,
                action="release.publish",
                subject_type="release",
                subject_id=release.id,
                payload={
                    "release_build_run_id": build.id,
                    "runtime_artifact_id": runtime_artifact.id,
                    "manifest_artifact_id": manifest_artifact.id,
                },
            )
        )
        task.result = {
            **(task.result or {}),
            "release_id": release.id,
            "release_manifest_artifact_id": manifest_artifact.id,
            "runtime_artifact_id": runtime_artifact.id,
        }

    elif task.kind == "release_publish":
        delivery = session.scalar(
            select(ReleaseDeliveryRun).where(ReleaseDeliveryRun.task_id == task.id)
        )
        if delivery is None:
            raise ValueError("release publish task has no delivery run")
        delivery_result = result.get("delivery") or {}
        if not delivery_result.get("immutable_locator") or not delivery_result.get(
            "content_hash"
        ):
            raise ValueError("release publisher did not return an immutable locator and hash")
        delivery.status = "succeeded"
        delivery.result = delivery_result
        delivery.report_artifact_id = report_artifact.id
        delivery.error = {}
        delivery.started_at = task.started_at
        delivery.ended_at = utcnow()
        session.add(
            AuditEvent(
                project_id=delivery.project_id,
                action="release.delivery_succeeded",
                subject_type="release_delivery_run",
                subject_id=delivery.id,
                payload={
                    "immutable_locator": delivery_result.get("immutable_locator"),
                    "content_hash": delivery_result.get("content_hash"),
                },
            )
        )

    elif task.kind == "inference":
        record = session.scalar(select(InferenceRecord).where(InferenceRecord.task_id == task.id))
        if record:
            record.rendered_prompt = result.get("rendered_prompt", record.prompt)
            record.output = result.get("output", "")
            record.token_ids = result.get("token_ids", [])
            record.input_tokens = int(result.get("input_tokens", 0))
            record.output_tokens = int(result.get("output_tokens", 0))
            record.stop_reason = result.get("stop_reason", "stop")
            record.performance = result.get("performance", {})


def sync_task_domain_status(session: Session, task: Task) -> None:
    if task.adapter_id in {"builtin.suite-manager", "builtin.suite-runner"}:
        from .suite_packages import sync_status
        sync_status(session, task)
    status = task.observed_state
    if task.kind == "plugin_install":
        installation_id = str((task.payload or {}).get("installation_id") or "")
        installation = (
            session.get(WorkerPluginInstallation, installation_id)
            if installation_id
            else None
        )
        if installation and status != "succeeded":
            if status in {"failed", "cancelled", "lost"}:
                installation.state = (
                    "ready" if installation.current_plugin_version_id else "failed"
                )
                installation.last_error = task.error or {
                    "code": f"plugin_install_{status}",
                    "message": f"插件安装任务状态：{status}",
                }
            elif status in {"assigned", "starting", "running"}:
                installation.state = (
                    "updating" if installation.current_plugin_version_id else "installing"
                )
    elif task.kind == "data_label":
        operation = session.scalar(
            select(DatasetLabelOperation).where(DatasetLabelOperation.task_id == task.id)
        )
        if operation and status != "succeeded":
            operation.status = status
            operation.error = task.error or {}
    elif task.kind == "dataset_materialize":
        operation = session.scalar(
            select(DatasetMaterialization).where(DatasetMaterialization.task_id == task.id)
        )
        if operation and status != "succeeded":
            if status in {"queued", "assigned"}:
                operation.status = "queued"
            elif status in {"failed", "lost"}:
                operation.status = "failed"
            elif status == "cancelled":
                operation.status = "cancelled"
            else:
                operation.status = "running"
            operation.error = task.error or {}
    elif task.kind == "code_import":
        version = session.scalar(select(CodeVersion).where(CodeVersion.task_id == task.id))
        if version and status != "succeeded":
            version.state = status
            source = session.get(CodeSource, version.source_id)
            if source and source.current_version_id == version.id:
                source.status = status
    elif task.kind == "model_weight_download":
        version = session.scalar(
            select(ModelWeightVersion).where(ModelWeightVersion.task_id == task.id)
        )
        if version and status != "succeeded":
            version.state = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "model_weight_evict" and status in {"failed", "cancelled", "lost"}:
        location_id = str((task.payload or {}).get("location_id") or "")
        location = session.get(ArtifactLocation, location_id) if location_id else None
        if location and location.availability_state == "deleting":
            location.availability_state = "available"
    elif task.kind in {"data_import", "data_generate"}:
        analysis = session.scalar(
            select(DatasetAnalysisRevision).where(DatasetAnalysisRevision.task_id == task.id)
        )
        version = (
            session.get(DatasetVersion, analysis.dataset_version_id)
            if analysis is not None
            else session.scalar(select(DatasetVersion).where(DatasetVersion.task_id == task.id))
        )
        if analysis is not None and status != "succeeded":
            analysis.status = status
            analysis.error = task.error or {}
        if version and status != "succeeded" and version.task_id == task.id:
            version.state = status
            dataset = session.get(Dataset, version.dataset_id)
            if dataset and (task.kind != "data_generate" or dataset.current_version_id == version.id):
                dataset.status = status
    elif task.kind == "data_convert":
        run = session.scalar(select(DataConversionRun).where(DataConversionRun.task_id == task.id))
        if run and status != "succeeded":
            run.status = status
            run.failure_reason = str((task.error or {}).get("message") or "") or None
            version = session.get(DatasetVersion, run.output_dataset_version_id)
            if version:
                version.state = status
    elif task.kind == "data_process":
        run = session.scalar(select(DataProcessingRun).where(DataProcessingRun.task_id == task.id))
        if run and status != "succeeded":
            run.status = status
            run.failure_reason = str((task.error or {}).get("message") or "") or None
            version = session.get(DatasetVersion, run.output_dataset_version_id)
            if version:
                version.state = status
    elif task.kind == "mixture_resolve":
        version = session.scalar(
            select(DataMixtureVersion).where(DataMixtureVersion.task_id == task.id)
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "tokenizer_train":
        tokenizer = session.scalar(
            select(TokenizerVersion).where(TokenizerVersion.task_id == task.id)
        )
        if tokenizer and status != "succeeded":
            tokenizer.status = status
    elif task.kind == "data_tokenize":
        version = session.scalar(
            select(TokenizedDatasetVersion).where(
                TokenizedDatasetVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "data_tokenize_corpus":
        version = session.scalar(
            select(TokenizedCorpusVersion).where(
                TokenizedCorpusVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "training_data_index":
        version = session.scalar(
            select(TrainingDataViewVersion).where(
                TrainingDataViewVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "supervision_scan":
        version = session.scalar(
            select(SupervisionDatasetVersion).where(
                SupervisionDatasetVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "supervision_corpus":
        version = session.scalar(
            select(SupervisionCorpusVersion).where(
                SupervisionCorpusVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "supervision_training_index":
        version = session.scalar(
            select(SupervisionTrainingViewVersion).where(
                SupervisionTrainingViewVersion.task_id == task.id
            )
        )
        if version and status != "succeeded":
            version.status = status
            version.failure_reason = str((task.error or {}).get("message") or "") or None
    elif task.kind == "train":
        run = session.scalar(select(TrainingRun).where(TrainingRun.task_id == task.id))
        if run:
            run.status = status
            run.started_at = task.started_at
            run.ended_at = task.ended_at
    elif task.kind == "evaluate":
        evaluation = session.scalar(select(EvaluationRun).where(EvaluationRun.task_id == task.id))
        if evaluation and status != "succeeded":
            evaluation.status = status
    elif task.kind == "quantize":
        quantization = session.scalar(
            select(QuantizationRun).where(QuantizationRun.task_id == task.id)
        )
        if quantization and status != "succeeded":
            quantization.status = status
            quantization.started_at = task.started_at
            quantization.ended_at = task.ended_at
    elif task.kind == "release_export":
        build = session.scalar(
            select(ReleaseBuildRun).where(ReleaseBuildRun.task_id == task.id)
        )
        if build and status != "succeeded":
            build.status = status
            build.error = task.error or {}
            build.started_at = task.started_at
            build.ended_at = task.ended_at
    elif task.kind == "release_publish":
        delivery = session.scalar(
            select(ReleaseDeliveryRun).where(ReleaseDeliveryRun.task_id == task.id)
        )
        if delivery and status != "succeeded":
            delivery.status = status
            delivery.error = task.error or {}
            delivery.started_at = task.started_at
            delivery.ended_at = task.ended_at

    # Pipeline task nodes reuse the existing Task state machine. Keep this import
    # local so the DAG scheduler can call create_task() without a module cycle.
    from .pipeline_dags import sync_pipeline_task_status

    sync_pipeline_task_status(session, task)


def reap_expired_leases(session: Session) -> int:
    leases = session.scalars(
        select(ResourceLease).where(
            ResourceLease.status == "active", ResourceLease.expires_at < utcnow()
        )
    ).all()
    reaped = 0
    for lease in leases:
        attempt = session.get(TaskAttempt, lease.task_attempt_id)
        if attempt is None:
            lease.status = "expired"
            lease.released_at = utcnow()
            continue
        task = session.get(Task, attempt.task_id)
        lease.status = "expired"
        lease.released_at = utcnow()
        attempt.status = TaskStatus.LOST
        attempt.ended_at = utcnow()
        attempt.error = {"code": "lease_expired", "message": "Worker heartbeat lease expired"}
        if task and task.observed_state in {
            TaskStatus.ASSIGNED,
            TaskStatus.STARTING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSING,
            TaskStatus.STOPPING,
        }:
            transition_task(
                session,
                task,
                TaskStatus.LOST,
                event_type="task.lease_expired",
                payload={"worker_id": attempt.worker_id, "attempt_id": attempt.id},
                attempt_id=attempt.id,
            )
            task.error = attempt.error
            sync_task_domain_status(session, task)
            reaped += 1
    return reaped


def recover_expired_attempt_lease(
    session: Session,
    *,
    worker_id: str,
    attempt_id: str,
    now: datetime | None = None,
) -> bool:
    """Recover a live Worker attempt that the lease reaper marked lost."""
    attempt = session.get(TaskAttempt, attempt_id)
    if attempt is None or attempt.worker_id != worker_id:
        return False
    task = session.get(Task, attempt.task_id)
    lease = session.scalar(
        select(ResourceLease).where(ResourceLease.task_attempt_id == attempt.id)
    )
    if (
        task is None
        or lease is None
        or task.active_attempt_id != attempt.id
        or task.observed_state != TaskStatus.LOST
        or task.desired_state != TaskStatus.RUNNING
        or (task.error or {}).get("code") != "lease_expired"
        or attempt.status != TaskStatus.LOST
        or attempt.ended_at is None
        or (attempt.error or {}).get("code") != "lease_expired"
        or lease.status != "expired"
    ):
        return False

    recovered_at = now or utcnow()
    task.observed_state = TaskStatus.RUNNING
    task.state_version += 1
    task.ended_at = None
    task.error = {}
    attempt.status = TaskStatus.RUNNING
    attempt.ended_at = None
    attempt.exit_code = None
    attempt.error = {}
    attempt.heartbeat_at = recovered_at
    attempt.lease_expires_at = recovered_at + timedelta(seconds=45)
    lease.status = "active"
    lease.expires_at = attempt.lease_expires_at
    lease.released_at = None
    sync_task_domain_status(session, task)
    return True


def mark_stale_workers_offline(session: Session, *, timeout_seconds: int = 30) -> int:
    cutoff = utcnow() - timedelta(seconds=timeout_seconds)
    workers = session.scalars(
        select(WorkerNode).where(
            WorkerNode.status == "online",
            WorkerNode.last_heartbeat_at.is_not(None),
            WorkerNode.last_heartbeat_at < cutoff,
        )
    ).all()
    for worker in workers:
        worker.status = "offline"
    if workers:
        session.execute(
            update(RemoteWorkerEnvironment)
            .where(
                RemoteWorkerEnvironment.worker_id.in_([worker.id for worker in workers]),
                RemoteWorkerEnvironment.status == "online",
            )
            .values(status="offline")
        )
    return len(workers)
