from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from forgelm_plugin_sdk.package import MAX_PACKAGE, inspect_package
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import Task
from ..services import suite_packages as service
from ..services.artifacts import artifact_storage
from ..suite_models import (
    Suite,
    SuiteComponent,
    SuiteConfigRevision,
    SuiteOperation,
    SuiteOperationPlan,
    SuitePackageImport,
    SuiteVersion,
)
from .common import add_audit

router = APIRouter(tags=["suite-packages"])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanCreate(StrictModel):
    action: str
    version_id: str
    worker_ids: list[str] = Field(min_length=1, max_length=100)
    version_ids: list[str] | None = None
    drain: bool = False


class OperationCreate(StrictModel):
    plan_id: str
    idempotency_key: str = Field(min_length=1, max_length=180)
    trust_digest: str | None = None
    accept_impact: bool = False


class VersionState(StrictModel):
    enabled: bool | None = None
    trusted: bool | None = None


class ArchiveState(StrictModel):
    archived: bool


class RevisionCreate(StrictModel):
    suite_version_id: str
    node_id: str
    config: dict = Field(default_factory=dict)
    previous_revision_id: str | None = None
    description: str = Field(default="", max_length=500)


class Continuation(StrictModel):
    checkpoint_id: str
    mode: str = "weights"
    keep_optimizer: bool = False


class RunCreate(StrictModel):
    revision_id: str
    worker_id: str
    inputs: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=86400, ge=1, le=2592000)
    continuation: Continuation | None = None


@router.post("/projects/{project_id}/suite-inputs", status_code=201)
def upload_input(project_id: str, suite_version_id: str = Form(...), node_id: str = Form(...), port: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    from ..models import Project
    service.get(session, Project, project_id)
    version = service.get(session, SuiteVersion, suite_version_id)
    node = next((n for n in version.package["nodes"] if n["id"] == node_id), None)
    declaration = next((p for p in node["inputs"] if p["name"] == port), None) if node else None
    if not declaration:
        service.fail("invalid_input_port", "套件中不存在此输入端口", 422)
    artifact = artifact_storage.put_stream(session, project_id=project_id, kind="suite_input", source=file.file, filename=Path(file.filename or "input.bin").name, media_type=file.content_type or "application/octet-stream", metadata={"format": declaration["format"], "suite_version_id": version.id, "node_id": node_id, "port": port})
    session.commit()
    return artifact.to_dict()


@router.post("/suite-package-imports", status_code=201)
def import_package(file: UploadFile = File(...), session: Session = Depends(get_session)):
    with tempfile.TemporaryDirectory(prefix="suite-import-") as temp:
        target = Path(temp) / "package.flsuite"
        total = 0
        with target.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_PACKAGE:
                    service.fail("package_too_large", "套件包超过 8 GiB", 413)
                output.write(chunk)
        try:
            package = inspect_package(target)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            service.fail("invalid_suite_package", str(exc), 422)
        project = service.system_project(session)
        with target.open("rb") as source:
            artifact = artifact_storage.put_stream(session, project_id=project.id, kind="suite_package", source=source, filename=package["digest"] + ".flsuite", media_type="application/zip", metadata={"suite_id": package["manifest"]["id"], "version": package["manifest"]["version"]})
        row = SuitePackageImport(artifact_id=artifact.id, package=package)
        session.add(row)
        session.commit()
        return row.to_dict()


@router.post("/suite-package-imports/{import_id}/register", status_code=201)
def register_package(import_id: str, session: Session = Depends(get_session)):
    imported = service.get(session, SuitePackageImport, import_id)
    package = imported.package
    manifest = package["manifest"]
    suite = session.scalar(select(Suite).where(Suite.suite_id == manifest["id"]))
    if not suite:
        suite = Suite(suite_id=manifest["id"], name=manifest["name"], publisher=manifest["publisher"])
        session.add(suite)
        session.flush()
    version = session.scalar(select(SuiteVersion).where(SuiteVersion.suite_id == suite.id, SuiteVersion.version == manifest["version"]))
    if version:
        if version.digest != package["digest"]:
            service.fail("immutable_version_conflict", "同一套件版本已存在不同内容，请发布新版本")
    else:
        version = SuiteVersion(suite_id=suite.id, version=manifest["version"], artifact_id=imported.artifact_id, digest=package["digest"], package=package)
        session.add(version)
        session.flush()
        for component in manifest["components"]:
            session.add(SuiteComponent(suite_version_id=version.id, component_id=component["id"], spec=component))
    imported.registered_version_id = version.id
    add_audit(session, project_id=None, action="suite.register", subject_type="suite_version", subject_id=version.id, payload={"digest": version.digest})
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        service.fail("registration_conflict", "套件目录已变化，请刷新后重试")
    return service.version_view(session, version)


@router.get("/suite-packages")
def list_packages(session: Session = Depends(get_session)):
    return [service.version_view(session, v) for v in session.scalars(select(SuiteVersion).order_by(SuiteVersion.created_at.desc()))]


@router.get("/suite-package-deployments")
def list_deployments(session: Session = Depends(get_session)):
    return service.deployments(session)


@router.get("/suite-nodes")
def list_nodes(session: Session = Depends(get_session)):
    return [{"suite_version_id": v.id, "suite_id": service.get(session, Suite, v.suite_id).suite_id, "version": v.version, "digest": v.digest, **node} for v in session.scalars(select(SuiteVersion).where(SuiteVersion.enabled.is_(True), SuiteVersion.trusted.is_(True))) if not service.get(session, Suite, v.suite_id).archived for node in v.package["nodes"]]


@router.patch("/suite-packages/{version_id}")
def state(version_id: str, payload: VersionState, session: Session = Depends(get_session)):
    version = service.get(session, SuiteVersion, version_id)
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(version, key, value)
    add_audit(session, project_id=None, action="suite.state", subject_type="suite_version", subject_id=version.id, payload=payload.model_dump(exclude_none=True))
    session.commit()
    return service.version_view(session, version)


@router.post("/suite-packages/{suite_id}/archive")
def archive(suite_id: str, payload: ArchiveState, session: Session = Depends(get_session)):
    suite = service.get(session, Suite, suite_id)
    suite.archived = payload.archived
    add_audit(session, project_id=None, action="suite.archive", subject_type="suite", subject_id=suite.id, payload={"archived": payload.archived})
    session.commit()
    return suite.to_dict()


@router.post("/suite-operation-plans", status_code=201)
def plan(payload: PlanCreate, session: Session = Depends(get_session)):
    row = service.make_plan(session, **payload.model_dump())
    session.commit()
    return row.to_dict()


@router.post("/suite-operations", status_code=201)
def operate(payload: OperationCreate, session: Session = Depends(get_session)):
    row = service.start_operation(session, service.get(session, SuiteOperationPlan, payload.plan_id), payload.idempotency_key, payload.trust_digest, payload.accept_impact)
    add_audit(session, project_id=None, action="suite." + row.action, subject_type="suite_operation", subject_id=row.id, payload={"plan_id": payload.plan_id})
    session.commit()
    return service.operation_view(session, row)


@router.get("/suite-operations")
def operations(session: Session = Depends(get_session)):
    return [service.operation_view(session, row) for row in session.scalars(select(SuiteOperation).order_by(SuiteOperation.created_at.desc()).limit(100))]


@router.get("/suite-operations/{operation_id}")
def operation(operation_id: str, session: Session = Depends(get_session)):
    return service.operation_view(session, service.get(session, SuiteOperation, operation_id))


@router.post("/suite-operations/{operation_id}/retry")
def retry(operation_id: str, session: Session = Depends(get_session)):
    operation = service.retry_operation(session, service.get(session, SuiteOperation, operation_id))
    session.commit()
    return service.operation_view(session, operation)


@router.post("/projects/{project_id}/suite-config-revisions", status_code=201)
def revision(project_id: str, payload: RevisionCreate, session: Session = Depends(get_session)):
    row = service.create_revision(session, project_id, payload.suite_version_id, payload.node_id, payload.config, payload.previous_revision_id)
    row.description = payload.description
    session.commit()
    return row.to_dict()


@router.get("/projects/{project_id}/suite-config-revisions")
def revisions(project_id: str, session: Session = Depends(get_session)):
    return [r.to_dict() for r in session.scalars(select(SuiteConfigRevision).where(SuiteConfigRevision.project_id == project_id).order_by(SuiteConfigRevision.created_at.desc()))]


@router.post("/projects/{project_id}/suite-runs", status_code=201)
def run(project_id: str, payload: RunCreate, session: Session = Depends(get_session)):
    revision = service.get(session, SuiteConfigRevision, payload.revision_id)
    if revision.project_id != project_id:
        service.fail("wrong_project", "配置不属于此项目", 422)
    task = service.run_revision(session, revision, payload.inputs, payload.worker_id, payload.timeout_seconds, payload.continuation.model_dump() if payload.continuation else None)
    session.commit()
    return task.to_dict()


@router.get("/projects/{project_id}/suite-runs")
def runs(project_id: str, session: Session = Depends(get_session)):
    return [r.to_dict() for r in session.scalars(select(Task).where(Task.project_id == project_id, Task.adapter_id == "builtin.suite-runner").order_by(Task.created_at.desc()).limit(100))]


@router.get("/projects/{project_id}/training-state-checkpoints")
def training_checkpoints(project_id: str, session: Session = Depends(get_session)):
    from ..models import Project, TrainingStateCheckpoint
    service.get(session, Project, project_id)
    return [r.to_dict() for r in session.scalars(select(TrainingStateCheckpoint).where(TrainingStateCheckpoint.project_id == project_id).order_by(TrainingStateCheckpoint.created_at.desc()).limit(200))]


@router.get("/tasks/{task_id}/execution-snapshots")
def execution_snapshots(task_id: str, session: Session = Depends(get_session)):
    from ..services.execution_provenance import snapshots_for_task
    service.get(session, Task, task_id)
    return snapshots_for_task(session, task_id)


@router.get("/projects/{project_id}/training-executions")
def training_executions(project_id: str, suite_id: str | None = None, component_version: str | None = None,
                        provider_id: str | None = None, provider_version: str | None = None,
                        offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200), session: Session = Depends(get_session)):
    from itertools import islice

    from sqlalchemy.orm import aliased

    from ..models import ExecutionSnapshot, Project, TaskAttempt
    service.get(session, Project, project_id)
    query = select(ExecutionSnapshot, TaskAttempt, Task).join(TaskAttempt, TaskAttempt.id == ExecutionSnapshot.attempt_id).join(Task, Task.id == ExecutionSnapshot.task_id).where(Task.project_id == project_id, ExecutionSnapshot.phase == "intent")
    if suite_id:
        query = query.where(ExecutionSnapshot.snapshot["payload"]["suite_execution"]["suite_id"].as_string() == suite_id)
    if component_version:
        query = query.where(ExecutionSnapshot.snapshot["component"]["implementationVersion"].as_string() == component_version)
    query = query.order_by(ExecutionSnapshot.created_at.desc())
    if provider_id or provider_version:
        resolved = aliased(ExecutionSnapshot)
        joined = query.add_columns(resolved).join(resolved, (resolved.attempt_id == TaskAttempt.id) & (resolved.phase == "resolved"))
        # Exact provider/version pairing; portable across SQLite and PostgreSQL.
        matches = (row[:3] for row in session.execute(joined).yield_per(100) if any(
            (not provider_id or backend.get("id") == provider_id) and (not provider_version or backend.get("version") == provider_version)
            for backend in row[3].snapshot.get("backends", [])))
        rows = list(islice(matches, offset, offset + limit + 1))
    else:
        rows = session.execute(query.offset(offset).limit(limit + 1)).all()
    return {"has_more": len(rows) > limit, "items": [{"id": row.id, "task_id": task.id, "title": task.title, "attempt_id": attempt.id,
            "attempt_number": attempt.attempt_number, "status": attempt.status, "digest": row.digest,
            "suite": row.snapshot["payload"].get("suite_execution"), "component": row.snapshot.get("component"), "created_at": row.created_at.isoformat()} for row, attempt, task in rows[:limit]]}
