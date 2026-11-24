from __future__ import annotations

from typing import TypeVar

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import AuditEvent

T = TypeVar("T")


def get_or_404(session: Session, model: type[T], object_id: str, label: str) -> T:
    record = session.get(model, object_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": f"{label}不存在"}
        )
    if getattr(model, "__tablename__", "") == "model_versions" and record.status == "deleted":
        raise HTTPException(410, {"code": "model_deleted", "message": "模型版本已删除，不能再次使用或恢复"})
    return record


def require_executable(adapter_id: str, command: dict) -> None:
    if adapter_id.startswith("builtin.command.") and not command.get("executable"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "missing_executable",
                "message": "用户命令适配器必须配置可执行文件",
            },
        )


def add_audit(
    session: Session,
    *,
    project_id: str | None,
    action: str,
    subject_type: str,
    subject_id: str,
    reason: str | None = None,
    payload: dict | None = None,
) -> None:
    session.add(
        AuditEvent(
            project_id=project_id,
            action=action,
            subject_type=subject_type,
            subject_id=subject_id,
            reason=reason,
            payload=payload or {},
        )
    )
