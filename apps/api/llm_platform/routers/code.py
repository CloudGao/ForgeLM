from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import Artifact, CodeSource, CodeVersion, Project
from ..schemas import CodeSourceCreate, CodeVersionCreate
from ..services.artifacts import artifact_storage
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(tags=["code"])


def version_view(session: Session, version: CodeVersion) -> dict[str, Any]:
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    return {
        **version.to_dict(),
        "artifact": artifact.to_dict() if artifact else None,
    }


def source_view(session: Session, source: CodeSource) -> dict[str, Any]:
    versions = session.scalars(
        select(CodeVersion)
        .where(CodeVersion.source_id == source.id)
        .order_by(CodeVersion.version.desc())
    ).all()
    current = (
        session.get(CodeVersion, source.current_version_id) if source.current_version_id else None
    )
    return {
        **source.to_dict(),
        "current_version": version_view(session, current) if current else None,
        "versions": [version_view(session, version) for version in versions],
    }


def _next_version(session: Session, source_id: str) -> int:
    return (
        int(
            session.scalar(
                select(func.coalesce(func.max(CodeVersion.version), 0)).where(
                    CodeVersion.source_id == source_id
                )
            )
            or 0
        )
        + 1
    )


def _validate_entrypoints(entrypoints: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for kind, path in entrypoints.items():
        clean_kind = str(kind).strip()
        clean_path = str(path).strip().replace("\\", "/")
        if not clean_kind or not clean_path:
            continue
        if clean_path.startswith("/") or clean_path.startswith("../") or ":/" in clean_path:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_code_entrypoint",
                    "message": f"代码入口必须是代码根目录内的相对路径：{path}",
                },
            )
        normalized[clean_kind] = clean_path
    return normalized


def _queue_version(
    session: Session,
    source: CodeSource,
    *,
    source_config: dict[str, Any],
    entrypoints: dict[str, str],
    config_schema: dict[str, Any],
    training_contract: dict[str, Any],
    dependency_spec: str,
    resource_request: dict[str, Any],
) -> CodeVersion:
    version = CodeVersion(
        source_id=source.id,
        version=_next_version(session, source.id),
        state="queued",
        entrypoints=_validate_entrypoints(entrypoints),
        config_schema=config_schema,
        training_contract=training_contract,
        dependency_spec=dependency_spec,
        source_snapshot={
            "source_type": source.source_type,
            "source_uri": source.source_uri,
            "source_config": source_config,
        },
    )
    session.add(version)
    session.flush()
    task = create_task(
        session,
        project_id=source.project_id,
        kind="code_import",
        title=f"导入代码：{source.name} v{version.version}",
        adapter_id="builtin.code-import",
        launcher_id="builtin.local-process",
        payload={
            "code_version_id": version.id,
            "source_type": source.source_type,
            "source_uri": source.source_uri,
            "source_config": source_config,
            "entrypoints": version.entrypoints,
            "resource_request": {"gpus": 0, **resource_request},
        },
    )
    version.task_id = task.id
    source.current_version_id = version.id
    source.status = "queued"
    return version


@router.get("/projects/{project_id}/code-sources")
def list_code_sources(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    sources = session.scalars(
        select(CodeSource)
        .where(CodeSource.project_id == project_id)
        .order_by(CodeSource.updated_at.desc())
    ).all()
    return [source_view(session, source) for source in sources]


@router.post("/projects/{project_id}/code-sources", status_code=201)
def create_code_source(
    project_id: str, payload: CodeSourceCreate, session: Session = Depends(get_session)
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    source = CodeSource(
        project_id=project_id,
        name=payload.name.strip(),
        source_type=payload.source_type,
        source_uri=payload.source_uri.strip(),
        source_config=payload.source_config,
        status="registered",
    )
    session.add(source)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_code_source", "message": "项目内代码源名称已存在"},
        ) from exc
    version = _queue_version(
        session,
        source,
        source_config=payload.source_config,
        entrypoints=payload.entrypoints,
        config_schema=payload.config_schema,
        training_contract=payload.training_contract,
        dependency_spec=payload.dependency_spec,
        resource_request=payload.resource_request,
    )
    add_audit(
        session,
        project_id=project_id,
        action="code_source.create",
        subject_type="code_source",
        subject_id=source.id,
        payload={"code_version_id": version.id, "source_type": source.source_type},
    )
    session.commit()
    return source_view(session, source)


@router.post("/code-sources/{source_id}/versions", status_code=201)
def create_code_version(
    source_id: str, payload: CodeVersionCreate, session: Session = Depends(get_session)
) -> dict:
    source = get_or_404(session, CodeSource, source_id, "代码源")
    if source.source_type == "upload":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "upload_source_requires_file",
                "message": "上传来源创建新版本时需要重新选择归档文件",
            },
        )
    previous = (
        session.get(CodeVersion, source.current_version_id) if source.current_version_id else None
    )
    source_config = payload.source_config or source.source_config or {}
    source.source_config = source_config
    version = _queue_version(
        session,
        source,
        source_config=source_config,
        entrypoints=payload.entrypoints
        if payload.entrypoints is not None
        else (previous.entrypoints if previous else {}),
        config_schema=payload.config_schema
        if payload.config_schema is not None
        else (previous.config_schema if previous else {}),
        training_contract=payload.training_contract
        if payload.training_contract is not None
        else (previous.training_contract if previous else {}),
        dependency_spec=payload.dependency_spec
        if payload.dependency_spec is not None
        else (previous.dependency_spec if previous else ""),
        resource_request=payload.resource_request,
    )
    add_audit(
        session,
        project_id=source.project_id,
        action="code_version.create",
        subject_type="code_version",
        subject_id=version.id,
    )
    session.commit()
    return version_view(session, version)


def _parse_json_object(raw: str, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_json", "message": f"{field_name} 不是合法 JSON"},
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_json", "message": f"{field_name} 必须是 JSON 对象"},
        )
    return value


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise HTTPException(
            status_code=422,
            detail={"code": "unsafe_code_archive", "message": f"代码归档包含不安全路径：{name}"},
        )
    return path


def _inspect_archive(upload: UploadFile) -> set[str]:
    upload.file.seek(0)
    files: list[PurePosixPath] = []
    if zipfile.is_zipfile(upload.file):
        upload.file.seek(0)
        with zipfile.ZipFile(upload.file) as archive:
            for member in archive.infolist():
                mode = member.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "unsafe_code_archive",
                            "message": f"代码归档包含符号链接：{member.filename}",
                        },
                    )
                if not member.is_dir():
                    files.append(_safe_archive_name(member.filename))
        upload.file.seek(0)
    else:
        upload.file.seek(0)
        try:
            with tarfile.open(fileobj=upload.file, mode="r:*") as archive:
                for member in archive.getmembers():
                    if member.issym() or member.islnk():
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "code": "unsafe_code_archive",
                                "message": f"代码归档包含链接：{member.name}",
                            },
                        )
                    if member.isfile():
                        files.append(_safe_archive_name(member.name))
        except tarfile.TarError as exc:
            upload.file.seek(0)
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_code_archive",
                    "message": "代码文件必须是 zip 或 tar 归档",
                },
            ) from exc
    upload.file.seek(0)
    if not files:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_code_archive", "message": "代码归档中没有文件"},
        )
    top_level = {path.parts[0] for path in files if path.parts}
    strip_root = len(top_level) == 1 and all(len(path.parts) > 1 for path in files)
    return {"/".join(path.parts[1:] if strip_root else path.parts) for path in files}


@router.post("/projects/{project_id}/code-sources/upload", status_code=201)
def upload_code_source(
    project_id: str,
    name: Annotated[str, Form(min_length=1, max_length=180)],
    file: Annotated[UploadFile, File()],
    entrypoints_json: Annotated[str, Form()] = "{}",
    config_schema_json: Annotated[str, Form()] = "{}",
    training_contract_json: Annotated[str, Form()] = "{}",
    dependency_spec: Annotated[str, Form(max_length=4096)] = "",
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    if session.scalar(
        select(CodeSource).where(
            CodeSource.project_id == project_id,
            CodeSource.name == name.strip(),
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_code_source", "message": "项目内代码源名称已存在"},
        )
    entrypoints = _validate_entrypoints(_parse_json_object(entrypoints_json, "入口点"))
    config_schema = _parse_json_object(config_schema_json, "配置 Schema")
    training_contract = _parse_json_object(training_contract_json, "训练契约")
    archive_files = _inspect_archive(file)
    missing_entrypoints = [path for path in entrypoints.values() if path not in archive_files]
    if missing_entrypoints:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "code_entrypoint_missing",
                "message": f"代码归档缺少入口文件：{', '.join(missing_entrypoints)}",
            },
        )
    artifact = artifact_storage.put_stream(
        session,
        project_id=project_id,
        kind="source_code",
        source=file.file,
        filename=file.filename or "source-code.zip",
        media_type=file.content_type or "application/octet-stream",
        metadata={"upload_filename": file.filename or "source-code.zip"},
    )
    source = CodeSource(
        project_id=project_id,
        name=name.strip(),
        source_type="upload",
        source_uri=f"upload://{artifact.content_hash}",
        source_config={"filename": file.filename or "source-code.zip"},
        status="ready",
    )
    session.add(source)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_code_source", "message": "项目内代码源名称已存在"},
        ) from exc
    version = CodeVersion(
        source_id=source.id,
        version=1,
        state="ready",
        artifact_id=artifact.id,
        source_snapshot={
            "source_type": "upload",
            "filename": file.filename or "source-code.zip",
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
        },
        entrypoints=entrypoints,
        config_schema=config_schema,
        training_contract=training_contract,
        dependency_spec=dependency_spec,
    )
    session.add(version)
    session.flush()
    source.current_version_id = version.id
    add_audit(
        session,
        project_id=project_id,
        action="code_source.upload",
        subject_type="code_version",
        subject_id=version.id,
        payload={"artifact_id": artifact.id, "content_hash": artifact.content_hash},
    )
    session.commit()
    return source_view(session, source)


@router.post("/code-sources/{source_id}/versions/upload", status_code=201)
def upload_code_version(
    source_id: str,
    file: Annotated[UploadFile, File()],
    entrypoints_json: Annotated[str, Form()] = "",
    config_schema_json: Annotated[str, Form()] = "",
    training_contract_json: Annotated[str, Form()] = "",
    dependency_spec: Annotated[str | None, Form(max_length=4096)] = None,
    session: Session = Depends(get_session),
) -> dict:
    source = get_or_404(session, CodeSource, source_id, "代码源")
    if source.source_type != "upload":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "code_source_not_upload",
                "message": "只有浏览器上传来源使用此接口",
            },
        )
    previous = (
        session.get(CodeVersion, source.current_version_id) if source.current_version_id else None
    )
    entrypoints = (
        _validate_entrypoints(_parse_json_object(entrypoints_json, "入口点"))
        if entrypoints_json.strip()
        else (previous.entrypoints if previous else {})
    )
    config_schema = (
        _parse_json_object(config_schema_json, "配置 Schema")
        if config_schema_json.strip()
        else (previous.config_schema if previous else {})
    )
    training_contract = (
        _parse_json_object(training_contract_json, "训练契约")
        if training_contract_json.strip()
        else (previous.training_contract if previous else {})
    )
    archive_files = _inspect_archive(file)
    missing_entrypoints = [path for path in entrypoints.values() if path not in archive_files]
    if missing_entrypoints:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "code_entrypoint_missing",
                "message": f"代码归档缺少入口文件：{', '.join(missing_entrypoints)}",
            },
        )
    artifact = artifact_storage.put_stream(
        session,
        project_id=source.project_id,
        kind="source_code",
        source=file.file,
        filename=file.filename or "source-code.zip",
        media_type=file.content_type or "application/octet-stream",
        metadata={"upload_filename": file.filename or "source-code.zip"},
    )
    version = CodeVersion(
        source_id=source.id,
        version=_next_version(session, source.id),
        state="ready",
        artifact_id=artifact.id,
        source_snapshot={
            "source_type": "upload",
            "filename": file.filename or "source-code.zip",
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
        },
        entrypoints=entrypoints,
        config_schema=config_schema,
        training_contract=training_contract,
        dependency_spec=(
            dependency_spec
            if dependency_spec is not None
            else (previous.dependency_spec if previous else "")
        ),
    )
    session.add(version)
    session.flush()
    source.current_version_id = version.id
    source.source_uri = f"upload://{artifact.content_hash}"
    source.source_config = {"filename": file.filename or "source-code.zip"}
    source.status = "ready"
    add_audit(
        session,
        project_id=source.project_id,
        action="code_version.upload",
        subject_type="code_version",
        subject_id=version.id,
        payload={"artifact_id": artifact.id, "content_hash": artifact.content_hash},
    )
    session.commit()
    return version_view(session, version)
