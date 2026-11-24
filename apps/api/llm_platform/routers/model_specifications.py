from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    CodeSource,
    CodeVersion,
    ModelSpecification,
    ModelSpecificationRevision,
    Project,
)
from ..schemas import (
    ModelSpecificationCreate,
    ModelSpecificationResolve,
    ModelSpecificationRevisionCreate,
)
from ..services.model_specifications import architecture_catalog, resolve_model_specification
from .common import add_audit, get_or_404

router = APIRouter(tags=["model-specifications"])


def _code_version_view(
    session: Session, project_id: str, code_version_id: str | None
) -> dict | None:
    if not code_version_id:
        return None
    version = session.get(CodeVersion, code_version_id)
    source = session.get(CodeSource, version.source_id) if version else None
    if version is None or source is None or source.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_code_version", "message": "代码版本不存在或不属于当前项目"},
        )
    if version.state != "ready" or not version.artifact_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "code_version_not_ready", "message": "自定义模型代码版本尚未就绪"},
        )
    artifact = session.get(Artifact, version.artifact_id)
    if artifact is None or artifact.kind != "source_code":
        raise HTTPException(
            status_code=422,
            detail={"code": "code_artifact_unavailable", "message": "代码版本没有可用 source_code 制品"},
        )
    return {
        "id": version.id,
        "source_id": source.id,
        "source_name": source.name,
        "version": version.version,
        "git_commit": version.git_commit,
        "artifact_id": version.artifact_id,
        "content_hash": artifact.content_hash,
        "entrypoints": version.entrypoints or {},
        "config_schema": version.config_schema or {},
        "dependency_spec": version.dependency_spec,
    }


def revision_view(revision: ModelSpecificationRevision) -> dict:
    runtime_requirements = revision.runtime_requirements or {}
    contract_version = int(runtime_requirements.get("contract_version") or 1)
    implementation = (
        runtime_requirements
        if contract_version >= 2
        else runtime_requirements.get("implementation")
    )
    generated_json = {
        "schema_version": f"forgelm.model-spec.v{contract_version}",
        "source_type": revision.source_type,
        "architecture": {
            "family_id": revision.architecture_family_id,
            "family_version": revision.architecture_version,
            **(
                {"category": runtime_requirements.get("category")}
                if contract_version >= 2
                else {}
            ),
            "implementation": implementation,
            **({"preset_id": revision.preset_id} if revision.preset_id else {}),
        },
        "config": revision.resolved_config or {},
        "initialization": revision.initialization or {},
        "parameters": revision.parameter_summary or {},
        "runtime_requirements": runtime_requirements,
        "config_signature": revision.config_signature,
    }
    return {**revision.to_dict(), "generated_json": generated_json}


def specification_view(session: Session, specification: ModelSpecification) -> dict:
    revisions = session.scalars(
        select(ModelSpecificationRevision)
        .where(ModelSpecificationRevision.specification_id == specification.id)
        .order_by(ModelSpecificationRevision.revision.desc())
    ).all()
    current = (
        session.get(ModelSpecificationRevision, specification.current_revision_id)
        if specification.current_revision_id
        else None
    )
    return {
        **specification.to_dict(),
        "current_revision": revision_view(current) if current else None,
        "revisions": [revision_view(item) for item in revisions],
    }


def _resolve(session: Session, project_id: str, payload: ModelSpecificationResolve) -> dict:
    get_or_404(session, Project, project_id, "项目")
    code_version = _code_version_view(session, project_id, payload.code_version_id)
    resolved = resolve_model_specification(
        source_type=payload.source_type,
        architecture_family_id=payload.architecture_family_id,
        preset_id=payload.preset_id,
        config=payload.config,
        initialization=payload.initialization,
        code_version=code_version,
    )
    return resolved


def _resolved_or_422(session: Session, project_id: str, payload: ModelSpecificationResolve) -> dict:
    resolved = _resolve(session, project_id, payload)
    if resolved.get("status") != "ready":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_specification_invalid",
                "message": "模型规格配置未通过校验",
                "issues": resolved.get("issues", []),
            },
        )
    return resolved


def _new_revision(
    *,
    specification_id: str,
    revision_number: int,
    code_version_id: str | None,
    resolved: dict,
) -> ModelSpecificationRevision:
    return ModelSpecificationRevision(
        specification_id=specification_id,
        revision=revision_number,
        source_type=resolved["source_type"],
        architecture_family_id=resolved["architecture_family_id"],
        architecture_version=resolved["architecture_version"],
        preset_id=resolved.get("preset_id"),
        code_version_id=code_version_id,
        user_config=resolved["user_config"],
        resolved_config=resolved["resolved_config"],
        initialization=resolved["initialization"],
        parameter_summary=resolved["parameter_summary"],
        runtime_requirements=resolved["runtime_requirements"],
        generated_code=resolved["generated_code"],
        config_signature=resolved["config_signature"],
    )


@router.get("/model-architectures")
def list_model_architectures() -> dict:
    return {"schema_version": 1, "architectures": architecture_catalog()}


@router.post("/projects/{project_id}/model-specifications/resolve")
def resolve_model_specification_preview(
    project_id: str,
    payload: ModelSpecificationResolve,
    session: Session = Depends(get_session),
) -> dict:
    return _resolve(session, project_id, payload)


@router.get("/projects/{project_id}/model-specifications")
def list_model_specifications(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    specifications = session.scalars(
        select(ModelSpecification)
        .where(ModelSpecification.project_id == project_id)
        .order_by(ModelSpecification.updated_at.desc())
    ).all()
    return [specification_view(session, item) for item in specifications]


@router.post("/projects/{project_id}/model-specifications", status_code=201)
def create_model_specification(
    project_id: str,
    payload: ModelSpecificationCreate,
    session: Session = Depends(get_session),
) -> dict:
    existing = session.scalar(
        select(ModelSpecification).where(
            ModelSpecification.project_id == project_id,
            ModelSpecification.name == payload.name,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={"code": "model_specification_name_exists", "message": "模型规格名称已存在"},
        )
    resolved = _resolved_or_422(session, project_id, payload)
    specification = ModelSpecification(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        status="ready",
    )
    session.add(specification)
    session.flush()
    revision = _new_revision(
        specification_id=specification.id,
        revision_number=1,
        code_version_id=payload.code_version_id,
        resolved=resolved,
    )
    session.add(revision)
    session.flush()
    specification.current_revision_id = revision.id
    add_audit(
        session,
        project_id=project_id,
        action="model_specification.create",
        subject_type="model_specification",
        subject_id=specification.id,
    )
    session.commit()
    return specification_view(session, specification)


@router.post("/model-specifications/{specification_id}/revisions", status_code=201)
def create_model_specification_revision(
    specification_id: str,
    payload: ModelSpecificationRevisionCreate,
    session: Session = Depends(get_session),
) -> dict:
    specification = get_or_404(session, ModelSpecification, specification_id, "模型规格")
    resolved = _resolved_or_422(session, specification.project_id, payload)
    latest = int(
        session.scalar(
            select(func.coalesce(func.max(ModelSpecificationRevision.revision), 0)).where(
                ModelSpecificationRevision.specification_id == specification.id
            )
        )
        or 0
    )
    revision = _new_revision(
        specification_id=specification.id,
        revision_number=latest + 1,
        code_version_id=payload.code_version_id,
        resolved=resolved,
    )
    session.add(revision)
    session.flush()
    specification.current_revision_id = revision.id
    specification.status = "ready"
    add_audit(
        session,
        project_id=specification.project_id,
        action="model_specification.revise",
        subject_type="model_specification_revision",
        subject_id=revision.id,
    )
    session.commit()
    return specification_view(session, specification)
