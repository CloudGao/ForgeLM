from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import HistoryImport, Project
from ..schemas import HistoryImportCreate
from ..services.history_imports import history_import_view, import_history
from .common import get_or_404

router = APIRouter(tags=["imports"])


@router.get("/projects/{project_id}/history-imports")
def list_history_imports(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    return [
        history_import_view(row)
        for row in session.scalars(
            select(HistoryImport)
            .where(HistoryImport.project_id == project_id)
            .order_by(HistoryImport.created_at.desc())
        ).all()
    ]


@router.post("/projects/{project_id}/history-imports", status_code=201)
def create_history_import(
    project_id: str,
    payload: HistoryImportCreate,
    session: Session = Depends(get_session),
) -> dict:
    project = get_or_404(session, Project, project_id, "项目")
    try:
        result = import_history(session, project=project, payload=payload)
        session.commit()
        return result
    except (ValueError, IntegrityError) as exc:
        session.rollback()
        message = str(exc.orig) if isinstance(exc, IntegrityError) else str(exc)
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_history_import", "message": message},
        ) from exc
