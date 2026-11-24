from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import Task, TaskAttempt
from ..services.suite_packages import get
from ..suite_models import SuiteServiceRequest
from .tasks import AgentIdentity, _require_agent_scope, verify_agent

router = APIRouter(tags=["suite-services"])


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=100000)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)


class ServiceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str
    attempt_id: str
    sequence: int = Field(ge=1, le=20000)
    event: str
    data: dict


def scope(session: Session, task_id: str, worker_id: str, attempt_id: str, identity: AgentIdentity):
    _require_agent_scope(identity, worker_id)
    task = get(session, Task, task_id)
    attempt = get(session, TaskAttempt, attempt_id)
    if task.kind != "suite_service" or task.active_attempt_id != attempt_id or attempt.task_id != task_id or attempt.worker_id != worker_id or attempt.ended_at is not None:
        raise HTTPException(409, {"code": "service_attempt_mismatch"})
    return task


@router.post("/suite-services/{task_id}/generate", status_code=201)
def generate(task_id: str, payload: Generation, session: Session = Depends(get_session)):
    task = get(session, Task, task_id)
    if task.kind != "suite_service" or task.observed_state != "running":
        raise HTTPException(409, {"message": "推理服务尚未运行"})
    pending = session.scalars(select(SuiteServiceRequest).where(SuiteServiceRequest.task_id == task_id, SuiteServiceRequest.status.in_({"queued", "running"}))).all()
    if len(pending) >= 32:
        raise HTTPException(429, {"message": "推理服务队列已满"})
    request = SuiteServiceRequest(task_id=task_id, payload=payload.model_dump())
    session.add(request)
    session.commit()
    return request.to_dict()


@router.get("/suite-service-requests/{request_id}")
def request_status(request_id: str, session: Session = Depends(get_session)):
    return get(session, SuiteServiceRequest, request_id).to_dict()


@router.post("/suite-service-requests/{request_id}/cancel")
def cancel(request_id: str, session: Session = Depends(get_session)):
    request = get(session, SuiteServiceRequest, request_id)
    request.cancel_requested = True
    if request.status == "queued":
        request.status = "cancelled"
    session.commit()
    return request.to_dict()


@router.get("/agent/suite-services/{task_id}/requests")
def pending_requests(task_id: str, worker_id: str, attempt_id: str, identity: AgentIdentity = Depends(verify_agent), session: Session = Depends(get_session)):
    scope(session, task_id, worker_id, attempt_id, identity)
    requests = session.scalars(select(SuiteServiceRequest).where(SuiteServiceRequest.task_id == task_id, SuiteServiceRequest.status.in_({"queued", "running"})).order_by(SuiteServiceRequest.created_at).limit(32)).all()
    for request in requests:
        if request.status == "queued":
            request.status, request.attempt_id = "running", attempt_id
        elif request.attempt_id != attempt_id:
            request.status, request.error = "failed", {"message": "原服务进程已结束，请重新发起请求"}
    session.commit()
    return [r.to_dict() for r in requests if r.status == "running"]


@router.post("/agent/suite-service-requests/{request_id}/events")
def event(request_id: str, payload: ServiceEvent, identity: AgentIdentity = Depends(verify_agent), session: Session = Depends(get_session)):
    request = get(session, SuiteServiceRequest, request_id)
    scope(session, request.task_id, payload.worker_id, payload.attempt_id, identity)
    if request.attempt_id != payload.attempt_id or payload.event not in {"chunk", "done", "error"}:
        raise HTTPException(409, {"message": "请求与服务尝试不匹配"})
    events = request.chunks
    record = {"event": payload.event, "data": payload.data}
    if payload.sequence <= len(events):
        if events[payload.sequence - 1] != record:
            raise HTTPException(409, {"message": "同一流序号内容冲突"})
        return request.to_dict()
    if payload.sequence != len(events) + 1 or request.status != "running":
        raise HTTPException(409, {"message": "无效的流序号或终态"})
    if len(json.dumps([*events, record])) > 2 * 1024 * 1024:
        request.status, request.cancel_requested, request.error = "failed", True, {"message": "输出超过 2 MiB 限制"}
    else:
        request.chunks = [*events, record]
        if payload.event == "done":
            request.status = "cancelled" if request.cancel_requested else "succeeded"
        elif payload.event == "error":
            request.status, request.error = "failed", payload.data
    session.commit()
    return request.to_dict()
