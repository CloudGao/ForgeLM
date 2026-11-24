from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from llm_platform_core.domain import TaskStatus
from llm_platform_core.tokenizers import (
    validate_builtin_tokenizer_sample_tokens,
    validate_disk_bpe_config,
)
from sqlalchemy import not_, or_, select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import SessionLocal, get_session
from ..models import (
    Artifact,
    Project,
    ProjectPipelineInstance,
    ProjectPipelineNodeRun,
    ProjectPipelineRun,
    RemoteWorkerEnvironment,
    RemoteWorkerOperation,
    ResourceLease,
    Task,
    TaskAttempt,
    TaskEvent,
    TrainingRun,
    WorkerNode,
    numeric_task_progress,
    utcnow,
    worker_event_sequence,
)
from ..schemas import (
    AgentClaim,
    AgentComplete,
    AgentEvent,
    AgentStart,
    TaskControl,
    WorkerEnroll,
    WorkerHeartbeat,
    WorkerRegister,
)
from ..services.artifacts import artifact_storage, artifact_view, artifact_view_for_worker
from ..services.tasks import (
    TIMING_SAMPLE_WINDOW,
    append_task_event,
    claim_task,
    estimate_task_timing,
    finalize_task_domain,
    recover_expired_attempt_lease,
    sync_task_domain_status,
    transition_task,
    update_task_from_progress,
)
from ..services.transactions import commit_or_flush
from ..services.workers import worker_view
from .common import add_audit, get_or_404

router = APIRouter(tags=["tasks"])

SAFE_TRAINING_PAUSE_PROTOCOL = "forgelm-command-control-file-v1"


def _supports_safe_training_pause(task: Task) -> bool:
    if task.kind != "train":
        return False
    if not (task.payload or {}).get("training_run_id"):
        return False
    command = (task.payload or {}).get("command") or {}
    if not isinstance(command, dict):
        return False
    control = command.get("control") or {}
    if not isinstance(control, dict):
        return False
    safe_pause = control.get("safe_pause") or {}
    if not isinstance(safe_pause, dict):
        return False
    return safe_pause.get("protocol") == SAFE_TRAINING_PAUSE_PROTOCOL


def task_event_view(event: TaskEvent) -> dict:
    return {
        "id": event.id,
        "task_id": event.task_id,
        "attempt_id": event.attempt_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "payload": event.payload,
        "created_at": event.created_at.isoformat(),
    }


@dataclass(frozen=True)
class AgentIdentity:
    worker_id: str | None
    legacy: bool = False


def verify_agent(
    x_agent_token: str = Header(default=""),
    x_worker_id: str = Header(default=""),
    session: Session = Depends(get_session),
) -> AgentIdentity:
    if hmac.compare_digest(x_agent_token, get_settings().agent_token):
        return AgentIdentity(worker_id=None, legacy=True)
    if not x_agent_token or not x_worker_id:
        raise HTTPException(
            status_code=401, detail={"code": "invalid_agent_token", "message": "Agent 凭证无效"}
        )
    worker = session.get(WorkerNode, x_worker_id)
    candidate_hash = hashlib.sha256(x_agent_token.encode("utf-8")).hexdigest()
    if worker is None or not worker.token_hash or not hmac.compare_digest(
        candidate_hash, worker.token_hash
    ):
        raise HTTPException(
            status_code=401, detail={"code": "invalid_agent_token", "message": "Agent 凭证无效"}
        )
    return AgentIdentity(worker_id=worker.id)


def _require_agent_scope(identity: AgentIdentity, worker_id: str) -> None:
    if identity.worker_id is not None and identity.worker_id != worker_id:
        raise HTTPException(
            status_code=403,
            detail={"code": "agent_worker_scope_mismatch", "message": "Agent 只能操作自己的 Worker"},
        )


def _mark_remote_worker_online(
    session: Session, worker: WorkerNode
) -> RemoteWorkerEnvironment | None:
    environment = session.scalar(
        select(RemoteWorkerEnvironment).where(RemoteWorkerEnvironment.worker_id == worker.id)
    )
    if environment is not None:
        environment.status = "online"
        environment.last_error = {}
        operation = session.scalar(
            select(RemoteWorkerOperation)
            .where(
                RemoteWorkerOperation.environment_id == environment.id,
                RemoteWorkerOperation.status == "succeeded",
                RemoteWorkerOperation.phase == "waiting_for_agent",
            )
            .order_by(RemoteWorkerOperation.created_at.desc())
            .limit(1)
        )
        if operation is not None:
            operation.phase = "completed"
            operation.result = {**(operation.result or {}), "waiting_for_agent": False}
    return environment


def task_view(session: Session, task: Task, include_events: bool = False) -> dict:
    now = utcnow()
    attempts = session.scalars(
        select(TaskAttempt)
        .where(TaskAttempt.task_id == task.id)
        .order_by(TaskAttempt.attempt_number.desc())
    ).all()
    progress_events: list[TaskEvent] = []
    timing_attempt = attempts[0] if attempts else None
    if task.observed_state in {TaskStatus.RUNNING, TaskStatus.PAUSED} and timing_attempt:
        numeric_progress = numeric_task_progress()
        progress_events = list(
            reversed(
                session.scalars(
                    select(TaskEvent)
                    .where(
                        TaskEvent.task_id == task.id,
                        TaskEvent.attempt_id == timing_attempt.id,
                        numeric_progress,
                    )
                    .order_by(TaskEvent.sequence.desc())
                    .limit(120)
                ).all()
            )
        )
        if len(progress_events) == 120:
            window_end = (
                progress_events[-1].created_at
                if task.observed_state == TaskStatus.PAUSED
                else now
            )
            anchor = session.scalar(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.attempt_id == timing_attempt.id,
                    numeric_progress,
                    TaskEvent.sequence < progress_events[0].sequence,
                    TaskEvent.created_at >= window_end - TIMING_SAMPLE_WINDOW,
                )
                .order_by(TaskEvent.sequence)
                .limit(1)
            )
            if anchor is not None:
                progress_events.insert(0, anchor)
    progress_samples = [
        (event.created_at, event.payload.get("progress"))
        for event in progress_events
        if "progress" in (event.payload or {})
    ]
    result = {
        **task.to_dict(),
        "attempts": [attempt.to_dict() for attempt in attempts],
        "timing_estimate": estimate_task_timing(
            task, attempts, progress_samples, now=now
        ),
    }
    if include_events:
        events = session.scalars(
            select(TaskEvent).where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.sequence.desc()).limit(301)
        ).all()
        result["events"] = [task_event_view(event) for event in reversed(events[:300])]
        result["events_has_more"] = len(events) > 300
    return result


@router.get("/tasks")
def list_tasks(
    project_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    include_events: bool = Query(default=False),
    exclude_kind: list[str] | None = Query(default=None),
    scope: Literal["all", "pipeline"] = Query(default="all"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> list[dict]:
    query = select(Task)
    if kind:
        query = query.where(Task.kind == kind)
    if project_id:
        query = query.where(Task.project_id == project_id)
    if status:
        query = query.where(Task.observed_state == status)
    if exclude_kind:
        query = query.where(Task.kind.not_in(set(exclude_kind)))
    if scope == "pipeline":
        # Execution ownership, not kind/duration or unverified payload hints,
        # defines the task center. EXISTS also deduplicates resumed node runs.
        node_execution = (
            select(ProjectPipelineNodeRun.id)
            .join(ProjectPipelineRun)
            .where(
                ProjectPipelineNodeRun.task_id == Task.id,
                ProjectPipelineRun.project_id == Task.project_id,
            )
            .exists()
        )
        # Imported training instances expose real TaskAttempts without creating
        # synthetic NodeRuns; retain that existing compatibility path.
        history_execution = (
            select(TrainingRun.id)
            .join(
                ProjectPipelineInstance,
                or_(
                    ProjectPipelineInstance.variables["__training_history"]["training_run_id"].as_string() == TrainingRun.id,
                    ProjectPipelineInstance.variables["__forgelm"]["source_training_run_id"].as_string() == TrainingRun.id,
                ),
            )
            .where(
                TrainingRun.task_id == Task.id,
                TrainingRun.project_id == Task.project_id,
                ProjectPipelineInstance.project_id == Task.project_id,
            )
            .exists()
        )
        query = query.where(or_(node_execution, history_execution))
    query = query.order_by(Task.created_at.desc(), Task.id.desc()).offset(offset).limit(limit)
    return [
        task_view(session, task, include_events=include_events)
        for task in session.scalars(query).all()
    ]


@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    include_events: bool = Query(default=True),
    session: Session = Depends(get_session),
) -> dict:
    return task_view(
        session,
        get_or_404(session, Task, task_id, "任务"),
        include_events=include_events,
    )


@router.get("/tasks/{task_id}/event-window")
def list_task_events(
    task_id: str,
    event_group: Literal["logs", "resources", "timeline", "all"] = Query(default="all"),
    attempt_id: str | None = Query(default=None),
    limit: int = Query(default=300, ge=1, le=2000),
    before_sequence: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Task, task_id, "任务")
    log_filter = or_(
        TaskEvent.event_type == "log",
        TaskEvent.event_type.contains("stdout"),
        TaskEvent.event_type.contains("stderr"),
    )
    resource_filter = TaskEvent.event_type.in_(("resource.sample", "resource.summary"))
    query = select(TaskEvent).where(TaskEvent.task_id == task_id)
    if before_sequence is not None:
        query = query.where(TaskEvent.sequence < before_sequence)
    if attempt_id:
        query = query.where(TaskEvent.attempt_id == attempt_id)
    if event_group == "logs":
        query = query.where(log_filter)
    elif event_group == "resources":
        query = query.where(resource_filter)
    elif event_group == "timeline":
        query = query.where(not_(or_(log_filter, resource_filter)))
    newest = list(
        session.scalars(query.order_by(TaskEvent.sequence.desc()).limit(limit + 1)).all()
    )
    has_more = len(newest) > limit
    items = list(reversed(newest[:limit]))
    return {
        "items": [task_event_view(event) for event in items],
        "has_more": has_more,
        "next_before_sequence": items[0].sequence if has_more and items else None,
    }


@router.post("/tasks/{task_id}/control")
def control_task(
    task_id: str, payload: TaskControl, session: Session = Depends(get_session)
) -> dict:
    task = get_or_404(session, Task, task_id, "任务")
    current = task.observed_state
    if task.adapter_id in {"builtin.suite-manager", "builtin.suite-runner"} and payload.action == "pause" and current != TaskStatus.QUEUED and not task.payload.get("suite_execution", {}).get("training"):
        raise HTTPException(409, detail={"code": "suite_pause_unsupported", "message": "套件运行时未声明安全暂停/恢复契约，请使用取消操作"})
    if task.adapter_id == "builtin.suite-manager" and payload.action == "retry":
        raise HTTPException(409, detail={"code": "suite_operation_retry_required", "message": "请从套件操作记录重试，以保留部署代际与对账关系"})
    if payload.action in {"retry", "resume"} and current in {"failed", "lost", "cancelled"} and not session.info.get("training_attempt_control"):
        from ..models import ProjectPipelineInstance, ProjectPipelineNodeRun
        from ..services.training_instances import META, start_attempt
        node = session.scalar(select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.task_id == task.id).order_by(ProjectPipelineNodeRun.created_at.desc()).limit(1))
        if node:
            from ..models import ProjectPipelineRun
            attempt = session.get(ProjectPipelineRun, node.pipeline_run_id)
            instance = session.get(ProjectPipelineInstance, attempt.instance_id) if attempt else None
            if instance and (attempt.context_snapshot or {}).get(META):
                start_attempt(session, instance, attempt.id)
                commit_or_flush(session)
                return task_view(session, task, include_events=True)
    if payload.action == "pause":
        if task.kind == "train" and not _supports_safe_training_pause(task):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "safe_training_pause_not_supported",
                    "message": "该训练任务没有声明安全暂停协议，请使用立即停止",
                },
            )
        if current == TaskStatus.QUEUED:
            task.desired_state = TaskStatus.PAUSED
            transition_task(session, task, TaskStatus.PAUSED, event_type="task.paused")
        elif current in {TaskStatus.ASSIGNED, TaskStatus.STARTING}:
            task.desired_state = TaskStatus.PAUSED
            task.state_version += 1
            append_task_event(
                session,
                task,
                "task.pause_requested",
                {"waiting_for_worker": True},
                task.active_attempt_id,
            )
        elif current == TaskStatus.RUNNING:
            task.desired_state = TaskStatus.PAUSED
            transition_task(session, task, TaskStatus.PAUSING, event_type="task.pause_requested")
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "not_pauseable",
                    "message": "只有排队、已分配、启动中或运行中的任务可以暂停",
                },
            )
    elif payload.action == "resume":
        if current == TaskStatus.PAUSED:
            task.desired_state = TaskStatus.RUNNING
            transition_task(session, task, TaskStatus.QUEUED, event_type="task.resume_requested")
        elif (current == TaskStatus.CANCELLED and task.kind != "train"
              and session.info.get("training_attempt_control")
              and task.adapter_id.startswith("builtin.")
              and not task.adapter_id.startswith("builtin.command.")):
            # Unified workflow retries restart native data nodes from their frozen
            # inputs; they do not imply that an interrupted engine has a checkpoint.
            task.desired_state = TaskStatus.RUNNING
            task.error = {}
            task.ended_at = None
            task.progress = 0
            task.current_phase = None
            transition_task(session, task, TaskStatus.QUEUED, event_type="task.retry_queued",
                            payload={"from_state": "cancelled", "restart_from_frozen_inputs": True})
        elif current == TaskStatus.CANCELLED and task.kind == "train":
            training_run_id = str((task.payload or {}).get("training_run_id") or "")
            training_run = session.get(TrainingRun, training_run_id) if training_run_id else None
            if training_run is None or training_run.task_id != task.id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "training_run_context_missing",
                        "message": "训练任务缺少可恢复的训练运行上下文",
                    },
                )
            task_payload = task.payload or {}
            tokenizer_vocab_size = int(
                (task_payload.get("tokenizer") or {}).get("vocab_size") or 0
            )
            model_vocab_size = int(task_payload.get("model_vocab_size") or 0)
            if (
                tokenizer_vocab_size
                and model_vocab_size
                and tokenizer_vocab_size != model_vocab_size
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "tokenizer_model_vocab_mismatch",
                        "message": (
                            f"模型配置要求 {model_vocab_size:,} 词表，但任务 Tokenizer 为 "
                            f"{tokenizer_vocab_size:,}，不能继续该配置"
                        ),
                    },
                )
            task.desired_state = TaskStatus.RUNNING
            task.error = {}
            task.ended_at = None
            transition_task(
                session,
                task,
                TaskStatus.QUEUED,
                event_type="task.resume_requested",
                payload={"from_state": TaskStatus.CANCELLED},
            )
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "not_resumable",
                    "message": "只有暂停任务或已停止的训练任务可以继续执行",
                },
            )
    elif payload.action in {"safe_stop", "cancel"}:
        task.desired_state = TaskStatus.CANCELLED
        if current in {TaskStatus.DRAFT, TaskStatus.QUEUED, TaskStatus.PAUSED}:
            transition_task(session, task, TaskStatus.CANCELLED, event_type="task.cancelled")
        elif current in {
            TaskStatus.ASSIGNED,
            TaskStatus.STARTING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSING,
        }:
            if current != TaskStatus.STOPPING:
                if current == TaskStatus.ASSIGNED:
                    transition_task(
                        session, task, TaskStatus.STARTING, event_type="task.stop_requested"
                    )
                transition_task(
                    session, task, TaskStatus.STOPPING, event_type="task.stop_requested"
                )
    elif payload.action == "retry":
        if current not in {TaskStatus.FAILED, TaskStatus.LOST}:
            raise HTTPException(
                status_code=409, detail={"code": "not_retryable", "message": "该任务当前不能重试"}
            )
        if task.kind == "train":
            task_payload = task.payload or {}
            tokenizer_vocab_size = int(
                (task_payload.get("tokenizer") or {}).get("vocab_size") or 0
            )
            model_vocab_size = int(task_payload.get("model_vocab_size") or 0)
            if (
                tokenizer_vocab_size
                and model_vocab_size
                and tokenizer_vocab_size != model_vocab_size
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "tokenizer_model_vocab_mismatch",
                        "message": (
                            f"模型配置要求 {model_vocab_size:,} 词表，但任务 Tokenizer 为 "
                            f"{tokenizer_vocab_size:,}，不能重试该配置"
                        ),
                    },
                )
        if task.adapter_id in {"builtin.hf-tokenizer", "builtin.disk-bpe-tokenizer"}:
            task_payload = dict(task.payload or {})
            try:
                config = task_payload.get("config") or {}
                validate_builtin_tokenizer_sample_tokens(config.get("sample_tokens"))
                if task.adapter_id == "builtin.disk-bpe-tokenizer":
                    if str(task_payload.get("algorithm") or "BPE").lower() != "bpe":
                        raise ValueError("Disk-backed tokenizer supports BPE only")
                    validate_disk_bpe_config(config)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": (
                            "invalid_disk_bpe_config"
                            if task.adapter_id == "builtin.disk-bpe-tokenizer"
                            else "invalid_tokenizer_sample_tokens"
                        ),
                        "message": str(exc),
                    },
                ) from exc
            resource_request = dict(task_payload.get("resource_request") or {})
            resource_request["gpus"] = 0
            resource_request.pop("gpu_indexes", None)
            task.payload = {**task_payload, "resource_request": resource_request}
        task.desired_state = TaskStatus.RUNNING
        task.error = {}
        task.ended_at = None
        transition_task(session, task, TaskStatus.QUEUED, event_type="task.retry_queued")
    add_audit(
        session,
        project_id=task.project_id,
        action=f"task.control.{payload.action}",
        subject_type="task",
        subject_id=task.id,
        reason=payload.reason,
    )
    sync_task_domain_status(session, task)
    commit_or_flush(session)
    return task_view(session, task, include_events=True)


@router.get("/tasks/{task_id}/events")
async def stream_task_events(
    task_id: str, after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    try:
        resume_cursor = max(after, int(last_event_id or 0))
    except ValueError:
        raise HTTPException(422, detail={"message": "Invalid stream cursor"})
    async def generate():
        cursor = resume_cursor
        idle_ticks = 0
        while idle_ticks < 300:
            with SessionLocal() as session:
                task = session.get(Task, task_id)
                if task is None:
                    yield 'event: error\ndata: {"code":"not_found"}\n\n'
                    return
                events = session.scalars(
                    select(TaskEvent)
                    .where(TaskEvent.task_id == task_id, TaskEvent.sequence > cursor)
                    .order_by(TaskEvent.sequence).limit(128)
                ).all()
                for event in events:
                    cursor = event.sequence
                    data = {
                        "id": event.id,
                        "sequence": event.sequence,
                        "attempt_id": event.attempt_id,
                        "event_type": event.event_type,
                        "payload": event.payload,
                        "created_at": event.created_at.isoformat(),
                    }
                    yield f"id: {cursor}\nevent: task-event\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if events:
                    idle_ticks = 0
                else:
                    idle_ticks += 1
                    yield ": heartbeat\n\n"
                if (
                    task.observed_state in {"succeeded", "failed", "cancelled", "lost"}
                    and not events
                ):
                    return
            await asyncio.sleep(0.25)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent/enroll")
def enroll_worker(
    payload: WorkerEnroll,
    x_agent_token: str = Header(default=""),
    session: Session = Depends(get_session),
) -> dict:
    environment = get_or_404(
        session, RemoteWorkerEnvironment, payload.environment_id, "远程 Worker 环境"
    )
    expires_at = environment.enrollment_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    supplied_hash = hashlib.sha256(x_agent_token.encode("utf-8")).hexdigest()
    if (
        not x_agent_token
        or not environment.enrollment_token_hash
        or environment.enrollment_used_at is not None
        or expires_at is None
        or expires_at <= utcnow()
        or not hmac.compare_digest(supplied_hash, environment.enrollment_token_hash)
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_enrollment_token",
                "message": "Worker 注册凭证无效、已使用或已过期",
            },
        )
    if payload.name != environment.name:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "enrollment_environment_mismatch",
                "message": "Worker 名称与注册环境不一致",
            },
        )
    enrollment_used_at = utcnow()
    claimed = session.execute(
        update(RemoteWorkerEnvironment)
        .where(
            RemoteWorkerEnvironment.id == environment.id,
            RemoteWorkerEnvironment.enrollment_token_hash == supplied_hash,
            RemoteWorkerEnvironment.enrollment_used_at.is_(None),
        )
        .values(enrollment_used_at=enrollment_used_at)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        session.rollback()
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_enrollment_token",
                "message": "Worker 注册凭证无效、已使用或已过期",
            },
        )
    worker = session.get(WorkerNode, environment.worker_id) if environment.worker_id else None
    if worker is None and environment.worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "bound_worker_missing", "message": "远程环境绑定的 Worker 不存在"},
        )
    if worker is None:
        name_collision = session.scalar(
            select(WorkerNode).where(WorkerNode.name == payload.name)
        )
        if name_collision is not None:
            raise HTTPException(
                status_code=409,
                detail={"code": "worker_name_in_use", "message": "Worker 名称已被占用"},
            )
        worker = WorkerNode(name=payload.name, hostname=payload.hostname)
        session.add(worker)
        session.flush()
    if worker.name != environment.name:
        raise HTTPException(
            status_code=409,
            detail={"code": "environment_already_bound", "message": "远程环境已绑定其他 Worker"},
        )
    agent_token = secrets.token_urlsafe(48)
    worker.token_hash = hashlib.sha256(agent_token.encode("utf-8")).hexdigest()
    worker.hostname = payload.hostname
    worker.labels = {
        **payload.labels,
        **(environment.labels or {}),
        "environment": "remote-ssh",
        "remote_environment_id": environment.id,
    }
    worker.resources = payload.resources
    worker.capabilities = payload.capabilities
    worker.status = "online"
    worker.last_heartbeat_at = utcnow()
    environment.worker_id = worker.id
    environment.enrollment_used_at = enrollment_used_at
    _mark_remote_worker_online(session, worker)
    session.commit()
    return {"worker": worker_view(worker), "agent_token": agent_token}


@router.post("/agent/register")
def register_worker(
    payload: WorkerRegister,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    if identity.worker_id is not None:
        worker = get_or_404(session, WorkerNode, identity.worker_id, "Worker")
        if worker.name != payload.name:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "agent_worker_scope_mismatch",
                    "message": "节点凭证不能注册为其他 Worker",
                },
            )
    else:
        worker = session.scalar(select(WorkerNode).where(WorkerNode.name == payload.name))
        if worker is None:
            worker = WorkerNode(name=payload.name, hostname=payload.hostname)
            session.add(worker)
    worker.hostname = payload.hostname
    environment = _mark_remote_worker_online(session, worker)
    worker.labels = (
        {
            **payload.labels,
            **(environment.labels or {}),
            "environment": "remote-ssh",
            "remote_environment_id": environment.id,
        }
        if environment is not None
        else payload.labels
    )
    worker.resources = payload.resources
    worker.capabilities = payload.capabilities
    worker.status = "online"
    worker.last_heartbeat_at = utcnow()
    session.commit()
    return worker_view(worker)


@router.post("/agent/workers/{worker_id}/heartbeat")
def worker_heartbeat(
    worker_id: str,
    payload: WorkerHeartbeat,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    _require_agent_scope(identity, worker_id)
    worker = get_or_404(session, WorkerNode, worker_id, "Worker")
    worker.status = "online"
    worker.resources = payload.resources or worker.resources
    if payload.capabilities is not None:
        worker.capabilities = payload.capabilities
    worker.last_heartbeat_at = utcnow()
    _mark_remote_worker_online(session, worker)
    controls: list[dict] = []
    recovered_attempt_ids: list[str] = []
    for attempt_id in payload.running_attempt_ids:
        if recover_expired_attempt_lease(
            session,
            worker_id=worker_id,
            attempt_id=attempt_id,
        ):
            recovered_attempt_ids.append(attempt_id)
        attempt = session.get(TaskAttempt, attempt_id)
        if attempt and attempt.worker_id == worker_id:
            attempt.heartbeat_at = utcnow()
            attempt.lease_expires_at = utcnow() + timedelta(seconds=45)
            lease = session.scalar(
                select(ResourceLease).where(ResourceLease.task_attempt_id == attempt.id)
            )
            if lease and lease.status == "active":
                lease.expires_at = attempt.lease_expires_at
            task = session.get(Task, attempt.task_id)
            if task and task.desired_state != TaskStatus.RUNNING:
                controls.append(
                    {
                        "task_id": task.id,
                        "attempt_id": attempt.id,
                        "desired_state": task.desired_state,
                    }
                )
    if (worker.labels or {}).get("inference_reclaim_requested"):
        if (worker.resources or {}).get("inference_runtimes"):
            controls.append({"action": "release_idle_inference"})
        else:
            worker.labels = {k: v for k, v in (worker.labels or {}).items() if k != "inference_reclaim_requested"}
    session.commit()
    from ..services.worker_dispatch import notify
    notify()
    return {
        "worker": worker_view(worker),
        "controls": controls,
        "recovered_attempt_ids": recovered_attempt_ids,
    }


@router.post("/agent/tasks/claim")
def agent_claim(
    payload: AgentClaim,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict | None:
    from ..services.worker_dispatch import allocation_lock
    with allocation_lock:
        return _assign_worker(payload.worker_id, identity, session)


def _assign_worker(worker_id: str, identity: AgentIdentity, session: Session, *, direct: bool = False) -> dict | None:
    _require_agent_scope(identity, worker_id)
    get_or_404(session, WorkerNode, worker_id, "Worker")
    claimed = claim_task(session, worker_id, direct=direct)
    if not claimed:
        # Capacity waits are persisted even when no task can be assigned.
        session.commit()
        return None
    task, attempt, lease = claimed
    project = session.get(Project, task.project_id) if task.project_id else None
    task_view = task.to_dict()
    task_view["project_workspace_uri"] = project.workspace_uri if project else ""
    task_view["project_name"] = project.name if project else ""
    task_view["project_system_kind"] = project.system_kind if project else ""
    from pathlib import PureWindowsPath
    key = PureWindowsPath(str(project.workspace_uri or "")).name if project else ""
    task_view["project_storage_key"] = key if project and (key == project.id or key.endswith("--" + project.id)) else ""
    if task.kind == "data_generate":
        roots = set()
        source_task_ids = {task.id}
        continuation = (task.payload or {}).get("generation_continuation") or {}
        if continuation.get("task", {}).get("id"):
            source_task_ids.add(continuation["task"]["id"])
        for previous in session.scalars(select(TaskAttempt).where(TaskAttempt.task_id.in_(source_task_ids), TaskAttempt.worker_id == worker_id)):
            raw = str((previous.runtime_handle or {}).get("run_dir") or "").replace("\\", "/")
            marker = f"/.llm-platform/runs/{previous.task_id}/"
            if marker in raw:
                roots.add(raw.split(marker)[0] + "/.llm-platform/data-assets/data-generation")
        task_view["storage_resume_roots"] = sorted(roots)
    from ..services.training_continuation import latest_resume
    resume = latest_resume(session, task)
    if resume:
        task_view["suite_resume"] = resume
    if task.payload.get("suite_execution"):
        from ..models import ExecutionSnapshot
        resolved = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.task_id == task.id, ExecutionSnapshot.phase == "resolved").order_by(ExecutionSnapshot.created_at.desc()).limit(1))
        if resolved:
            task_view["suite_resolution"] = resolved.snapshot
    session.commit()
    return {
        "task": task_view,
        "attempt": attempt.to_dict(),
        "lease": lease.to_dict(),
    }


@router.get("/agent/workers/{worker_id}/dispatch")
async def agent_dispatch(
    worker_id: str,
    request: Request,
    identity: AgentIdentity = Depends(verify_agent),
) -> dict | None:
    from ..services.worker_dispatch import allocation_lock, subscribe
    _require_agent_scope(identity, worker_id)

    def assign():
        with allocation_lock, SessionLocal() as session:
            return _assign_worker(worker_id, identity, session, direct=True)

    # Subscribe before reading the queue; a commit during assignment cannot be lost.
    with subscribe(worker_id) as changed:
        deadline = asyncio.get_running_loop().time() + 25
        while True:
            if await request.is_disconnected():
                return None
            changed.clear()
            assigned = await run_in_threadpool(assign)
            if assigned:
                return assigned
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(changed.wait(), remaining)
            except TimeoutError:
                return None


@router.get("/agent/artifacts/{artifact_id}")
def agent_get_artifact(
    artifact_id: str,
    project_id: str = Query(),
    worker_id: str | None = Query(default=None),
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    artifact = get_or_404(session, Artifact, artifact_id, "制品")
    get_or_404(session, Project, project_id, "项目")
    if identity.worker_id is not None:
        if worker_id is not None:
            _require_agent_scope(identity, worker_id)
        else:
            worker_id = identity.worker_id
    if worker_id:
        worker = get_or_404(session, WorkerNode, worker_id, "Worker")
        return artifact_view_for_worker(session, artifact, project_id, worker)
    return artifact_view(session, artifact, project_id)


@router.get("/agent/artifacts/{artifact_id}/payload")
def agent_download_artifact(
    artifact_id: str,
    project_id: str = Query(),
    _identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> FileResponse:
    artifact = get_or_404(session, Artifact, artifact_id, "制品")
    payload_path = artifact_storage.managed_payload_path(session, artifact, project_id)
    if payload_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "managed_artifact_payload_not_found",
                "message": "该制品没有可下载的托管副本",
            },
        )
    return FileResponse(
        payload_path,
        media_type=artifact.media_type,
        filename=payload_path.name,
    )


@router.post("/agent/tasks/{task_id}/start")
def agent_start(
    task_id: str,
    payload: AgentStart,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    _require_agent_scope(identity, payload.worker_id)
    task = get_or_404(session, Task, task_id, "任务")
    attempt = get_or_404(session, TaskAttempt, payload.attempt_id, "任务尝试")
    if attempt.task_id != task.id or attempt.worker_id != payload.worker_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "attempt_mismatch", "message": "任务尝试与 Worker 不匹配"},
        )
    transition_task(session, task, TaskStatus.STARTING, attempt_id=attempt.id)
    transition_task(session, task, TaskStatus.RUNNING, attempt_id=attempt.id)
    if task.desired_state == TaskStatus.PAUSED:
        transition_task(
            session,
            task,
            TaskStatus.PAUSING,
            event_type="task.pause_requested",
            payload={"requested_before_start": True},
            attempt_id=attempt.id,
        )
    attempt.status = TaskStatus.RUNNING
    attempt.runtime_handle = payload.runtime_handle
    attempt.environment_snapshot = payload.environment_snapshot
    attempt.started_at = utcnow()
    attempt.heartbeat_at = utcnow()
    attempt.lease_expires_at = utcnow() + timedelta(seconds=45)
    sync_task_domain_status(session, task)
    session.commit()
    return task.to_dict()


@router.post("/agent/tasks/{task_id}/events")
def agent_event(
    task_id: str,
    payload: AgentEvent,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    _require_agent_scope(identity, payload.worker_id)
    task = get_or_404(session, Task, task_id, "任务")
    attempt = get_or_404(session, TaskAttempt, payload.attempt_id, "任务尝试")
    if attempt.task_id != task.id or attempt.worker_id != payload.worker_id:
        raise HTTPException(status_code=409, detail={"code": "attempt_mismatch"})
    event_payload = {**payload.payload}
    if payload.worker_sequence is not None:
        event_payload["worker_sequence"] = payload.worker_sequence
        existing_event = session.scalar(
            select(TaskEvent)
            .where(
                TaskEvent.task_id == task.id,
                TaskEvent.attempt_id == attempt.id,
                worker_event_sequence() == payload.worker_sequence,
            )
            .order_by(TaskEvent.id.desc())
        )
        if existing_event is not None:
            if (
                existing_event.event_type != payload.event_type
                or existing_event.payload != event_payload
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "worker_sequence_conflict",
                        "message": "同一 Worker 事件序号不能用于不同内容",
                    },
                )
            return {
                "sequence": existing_event.sequence,
                "desired_state": task.desired_state,
            }
    if payload.event_type in {"execution.resolved", "training.checkpoint", "execution.interval"}:
        from ..services.execution_provenance import (
            record_interval,
            record_receipt,
            register_checkpoint,
        )
        try:
            if payload.event_type == "execution.resolved":
                record_receipt(session, task, attempt, payload.payload)
            elif payload.event_type == "execution.interval":
                record_interval(session, task, attempt, payload.payload)
            else:
                register_checkpoint(session, task, attempt, payload.payload)
        except ValueError as exc:
            raise HTTPException(422, detail={"code": "invalid_training_receipt", "message": str(exc)}) from exc
    if payload.event_type in {"inference.snapshot", "inference.output"}:
        if task.active_attempt_id != attempt.id or attempt.ended_at is not None:
            raise HTTPException(409, detail={"message": "Inference attempt is no longer active"})
        if task.kind != "inference" or len(json.dumps(event_payload)) > 2 * 1024 * 1024:
            raise HTTPException(422, detail={"message": "Invalid inference stream"})
        from ..models import InferenceRecord
        record = session.scalar(select(InferenceRecord).where(InferenceRecord.task_id == task.id))
        if record:
            # Partial output never becomes conversation history or feedback input.
            record.performance = {**(record.performance or {}), "stream": {
                **event_payload, "attempt_id": attempt.id}}
        task.current_phase = str(event_payload.get("phase") or "generating")
    event = append_task_event(session, task, payload.event_type, event_payload, attempt.id)
    if payload.event_type in {"progress", "metric", "checkpoint"}:
        update_task_from_progress(session, task, event_payload)
    attempt.heartbeat_at = utcnow()
    attempt.lease_expires_at = utcnow() + timedelta(seconds=45)
    session.commit()
    return {"sequence": event.sequence, "desired_state": task.desired_state}


@router.post("/agent/tasks/{task_id}/artifacts")
def agent_upload_artifact(
    task_id: str,
    worker_id: str = Form(),
    attempt_id: str = Form(),
    kind: str = Form(),
    metadata_json: str = Form(default="{}"),
    file: UploadFile = File(),
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    _require_agent_scope(identity, worker_id)
    task = get_or_404(session, Task, task_id, "任务")
    attempt = get_or_404(session, TaskAttempt, attempt_id, "任务尝试")
    if attempt.task_id != task.id or attempt.worker_id != worker_id:
        raise HTTPException(status_code=409, detail={"code": "attempt_mismatch"})
    if (
        task.active_attempt_id != attempt.id
        or attempt.ended_at is not None
        or task.observed_state
        not in {
            TaskStatus.STARTING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSING,
            TaskStatus.STOPPING,
        }
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "task_attempt_not_active",
                "message": "只有未结束的当前任务尝试可以上传制品",
            },
        )
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError
        artifact = artifact_storage.put_stream(
            session,
            project_id=str(task.project_id or ""),
            kind=kind,
            source=file.file,
            filename=file.filename or "payload.bin",
            media_type=file.content_type or "application/octet-stream",
            created_by_task_attempt_id=attempt.id,
            metadata={**metadata, "task_id": task.id},
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_artifact", "message": "制品参数不合法"},
        ) from exc
    append_task_event(
        session,
        task,
        "artifact.created",
        {"artifact_id": artifact.id, "kind": artifact.kind, "size_bytes": artifact.size_bytes},
        attempt.id,
    )
    session.commit()
    return artifact_view(session, artifact, str(task.project_id or ""))


@router.post("/agent/tasks/{task_id}/complete")
def agent_complete(
    task_id: str,
    payload: AgentComplete,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    _require_agent_scope(identity, payload.worker_id)
    task = get_or_404(session, Task, task_id, "任务")
    attempt = get_or_404(session, TaskAttempt, payload.attempt_id, "任务尝试")
    if attempt.task_id != task.id or attempt.worker_id != payload.worker_id:
        raise HTTPException(status_code=409, detail={"code": "attempt_mismatch"})

    if attempt.ended_at is not None:
        if attempt.status == payload.status and task.observed_state == payload.status:
            return task_view(session, task, include_events=True)
        raise HTTPException(
            status_code=409,
            detail={
                "code": "attempt_already_completed",
                "message": "任务尝试已经以不同状态结束",
            },
        )

    if payload.runtime_metrics_artifact_id:
        metrics_artifact = session.get(Artifact, payload.runtime_metrics_artifact_id)
        uploaded_events = session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.attempt_id == attempt.id,
                TaskEvent.event_type == "artifact.created",
            )
        ).all()
        uploaded_by_attempt = any(
            event.payload.get("artifact_id") == payload.runtime_metrics_artifact_id
            for event in uploaded_events
        )
        if (
            metrics_artifact is None
            or metrics_artifact.kind != "runtime_metrics"
            or not uploaded_by_attempt
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_runtime_metrics_artifact",
                    "message": "运行指标制品不存在、类型错误或不属于当前任务尝试",
                },
            )

    target = payload.status
    if target == TaskStatus.PAUSED and task.payload.get("suite_execution", {}).get("training"):
        from ..models import TrainingStateCheckpoint
        if not session.scalar(select(TrainingStateCheckpoint.id).where(TrainingStateCheckpoint.attempt_id == attempt.id)):
            raise HTTPException(422, detail={"code": "checkpoint_required_for_pause", "message": "安全暂停需要本次尝试已登记的 Checkpoint"})
    if target == TaskStatus.CANCELLED and task.observed_state != TaskStatus.STOPPING:
        transition_task(session, task, TaskStatus.STOPPING, attempt_id=attempt.id)
    if target == TaskStatus.SUCCEEDED:
        try:
            finalize_task_domain(session, task, attempt, payload.result)
        except ValueError as exc:
            session.rollback()
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_task_result", "message": str(exc)},
            ) from exc
        if task.kind == "train":
            append_task_event(
                session,
                task,
                "system.validation",
                {
                    "validation_id": "output_artifact_integrity",
                    "status": "passed",
                    "phase": "output",
                    "message": "Checkpoint、模型和训练结果制品已完成身份校验与登记。",
                },
                attempt_id=attempt.id,
            )
    transition_task(
        session,
        task,
        target,
        event_type=f"task.{target}",
        payload=payload.result if target == TaskStatus.SUCCEEDED else payload.error,
        attempt_id=attempt.id,
    )
    if target == TaskStatus.SUCCEEDED:
        task.progress = 1.0
        task.current_phase = "completed"
    task.error = payload.error
    attempt.status = target
    attempt.exit_code = payload.exit_code
    attempt.error = payload.error
    attempt.resource_summary = payload.resource_summary
    attempt.runtime_metrics_artifact_id = payload.runtime_metrics_artifact_id
    attempt.log_summary = payload.log_summary
    attempt.ended_at = utcnow()
    from ..services.execution_provenance import append_snapshot
    append_snapshot(session, attempt, "completed", {"status": target, "result": payload.result, "error": payload.error,
                                                  "environment": attempt.environment_snapshot, "resource_summary": payload.resource_summary})
    lease = session.scalar(select(ResourceLease).where(ResourceLease.task_attempt_id == attempt.id))
    if lease:
        lease.status = "released"
        lease.released_at = utcnow()
    sync_task_domain_status(session, task)
    session.commit()
    from ..services.worker_dispatch import notify
    notify()
    return task_view(session, task, include_events=True)


@router.get("/agent/tasks/{task_id}/desired-state")
def desired_state(
    task_id: str,
    identity: AgentIdentity = Depends(verify_agent),
    session: Session = Depends(get_session),
) -> dict:
    task = get_or_404(session, Task, task_id, "任务")
    if identity.worker_id is not None:
        attempt = session.get(TaskAttempt, task.active_attempt_id) if task.active_attempt_id else None
        if attempt is None or attempt.worker_id != identity.worker_id:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "agent_worker_scope_mismatch",
                    "message": "Agent 只能查看自己的任务控制状态",
                },
            )
    return {"desired_state": task.desired_state, "observed_state": task.observed_state}
