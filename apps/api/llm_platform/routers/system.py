from __future__ import annotations

import asyncio
from ipaddress import ip_address
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import Artifact, PluginVersion, Project, WorkerNode
from ..schemas import (
    ControlPlaneRestart,
    ExternalArtifactReferenceCreate,
    LocalEnvironmentInject,
    LocalEnvironmentStart,
    PluginCreate,
)
from ..services.artifacts import artifact_storage, artifact_view
from ..services.control_plane import (
    control_plane_instance_id,
    control_plane_status,
    submit_restart_request,
)
from ..services.local_environments import (
    LocalEnvironmentError,
    discover_local_environments,
    local_injection_manager,
    local_native_worker_supervisor,
    local_wsl_lifecycle_supervisor,
)
from ..services.workers import worker_view
from .common import add_audit, get_or_404

router = APIRouter(tags=["system"])


def require_local_control(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host == "testclient":
        return
    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "local_control_required",
                "message": "本地执行环境注入只能从控制面主机操作",
            },
        )


def _has_online_native_worker(session: Session) -> bool:
    online_workers = session.scalars(
        select(WorkerNode).where(WorkerNode.status == "online")
    ).all()
    return any(
        (worker.labels or {}).get("environment") == "native"
        and (worker.labels or {}).get("location") == "local"
        for worker in online_workers
    )


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "local-llm-platform",
        "instance_id": control_plane_instance_id(),
    }


@router.get("/control-plane", dependencies=[Depends(require_local_control)])
def get_control_plane(request: Request) -> dict:
    server_attached = getattr(request.app.state, "control_plane_server", None) is not None
    return control_plane_status(server_attached=server_attached)


async def _stop_server_after_response(server: Any) -> None:
    await asyncio.sleep(0.5)
    server.should_exit = True


@router.post(
    "/control-plane/restart",
    status_code=202,
    dependencies=[Depends(require_local_control)],
)
async def restart_control_plane(payload: ControlPlaneRestart, request: Request) -> dict:
    server = getattr(request.app.state, "control_plane_server", None)
    status = control_plane_status(server_attached=server is not None)
    if not status["restart_supported"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "control_plane_not_supervised",
                "message": "控制面不是由 Supervisor 管理，请通过受管启动脚本启动后再重启",
            },
        )
    try:
        restart_request = submit_restart_request(payload.reason)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "control_plane_restart_unavailable",
                "message": str(exc),
            },
        ) from exc

    task = asyncio.create_task(_stop_server_after_response(server))
    tasks = getattr(request.app.state, "control_plane_tasks", None)
    if tasks is None:
        tasks = set()
        request.app.state.control_plane_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"accepted": True, **restart_request}


@router.get("/plugins")
def list_plugins(
    kind: str | None = Query(default=None), session: Session = Depends(get_session)
) -> list[dict]:
    query = select(PluginVersion)
    if kind:
        query = query.where(PluginVersion.kind == kind)
    return [
        plugin.to_dict()
        for plugin in session.scalars(
            query.order_by(PluginVersion.kind, PluginVersion.plugin_id, PluginVersion.version)
        ).all()
    ]


@router.post("/plugins", status_code=201)
def register_plugin(payload: PluginCreate, session: Session = Depends(get_session)) -> dict:
    if (payload.source or {}).get("type") in {"builtin-catalog", "builtin-suite"} or payload.kind == "plugin_suite":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "builtin_plugin_catalog_read_only",
                "message": "内置可安装插件目录只能随控制面版本登记",
            },
        )
    plugin = PluginVersion(**payload.model_dump())
    session.add(plugin)
    session.commit()
    return plugin.to_dict()


@router.get("/workers")
def list_workers(session: Session = Depends(get_session)) -> list[dict]:
    return [
        worker_view(worker)
        for worker in session.scalars(
            select(WorkerNode).order_by(WorkerNode.last_heartbeat_at.desc())
        ).all()
    ]


@router.post("/workers/{worker_id}/inference/unload", status_code=202)
def unload_idle_inference(worker_id: str, session: Session = Depends(get_session)) -> dict:
    worker = get_or_404(session, WorkerNode, worker_id, "Worker")
    worker.labels = {**(worker.labels or {}), "inference_reclaim_requested": True}
    add_audit(session, project_id=None, action="inference.unload", subject_type="worker", subject_id=worker.id,
              payload={"policy": "after_active_request"})
    session.commit()
    return {"accepted": True, "worker_id": worker.id, "policy": "after_active_request"}


@router.get("/execution-environments/local", dependencies=[Depends(require_local_control)])
def list_local_execution_environments(session: Session = Depends(get_session)) -> list[dict]:
    workers = session.scalars(select(WorkerNode)).all()
    environments = discover_local_environments()
    wsl_environment_count = sum(environment["kind"] == "wsl" for environment in environments)
    for environment in environments:
        matching = []
        for worker in workers:
            labels = worker.labels or {}
            if environment["kind"] == "wsl" and (
                labels.get("distro") == environment["name"]
                or (
                    labels.get("environment") == "wsl"
                    and worker.name.startswith("microlm-wsl-")
                    and wsl_environment_count == 1
                )
            ):
                matching.append(worker)
            elif (
                environment["id"] == "host:native"
                and labels.get("environment") == "native"
                and labels.get("location") == "local"
            ):
                matching.append(worker)
        matching.sort(
            key=lambda worker: worker.last_heartbeat_at or worker.created_at, reverse=True
        )
        worker_views = [worker_view(worker) for worker in matching]
        environment["workers"] = worker_views
        environment["worker"] = worker_views[0] if worker_views else None
        if environment["kind"] == "wsl":
            environment["supervision"] = local_wsl_lifecycle_supervisor.view(
                environment["name"]
            )
        elif environment["id"] == "host:native":
            environment["supervision"] = local_native_worker_supervisor.view()
        else:
            environment["supervision"] = None
    return environments


@router.get(
    "/execution-environments/local/operations",
    dependencies=[Depends(require_local_control)],
)
def list_local_environment_operations() -> list[dict]:
    return local_injection_manager.list()


@router.post(
    "/execution-environments/local/start",
    status_code=202,
    dependencies=[Depends(require_local_control)],
)
def start_local_environment(
    payload: LocalEnvironmentStart,
    session: Session = Depends(get_session),
) -> dict:
    try:
        local_native_worker_supervisor.validate(payload.environment_id)
    except LocalEnvironmentError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "local_environment_start_failed", "message": str(exc)},
        ) from exc
    if _has_online_native_worker(session):
        return local_native_worker_supervisor.view() | {
            "already_online": True,
            "worker_status": "online",
        }
    try:
        return local_native_worker_supervisor.start(payload.environment_id) | {
            "already_online": False,
            "worker_status": "offline",
        }
    except LocalEnvironmentError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "local_environment_start_failed", "message": str(exc)},
        ) from exc


@router.post(
    "/execution-environments/local/inject",
    status_code=202,
    dependencies=[Depends(require_local_control)],
)
def inject_local_environment(payload: LocalEnvironmentInject) -> dict:
    try:
        return local_injection_manager.start(
            environment_id=payload.environment_id,
            profile=payload.profile,
            workspace_root=payload.workspace_root,
        )
    except LocalEnvironmentError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "local_environment_injection_failed", "message": str(exc)},
        ) from exc


@router.get("/artifacts")
def list_artifacts(session: Session = Depends(get_session)) -> list[dict]:
    return [
        artifact_view(session, artifact)
        for artifact in session.scalars(
            select(Artifact).order_by(Artifact.created_at.desc()).limit(500)
        ).all()
    ]


@router.get("/projects/{project_id}/artifacts/{artifact_id}/payload")
def download_managed_artifact(
    project_id: str,
    artifact_id: str,
    session: Session = Depends(get_session),
) -> FileResponse:
    get_or_404(session, Project, project_id, "项目")
    artifact = get_or_404(session, Artifact, artifact_id, "制品")
    payload_path = artifact_storage.managed_payload_path(session, artifact, project_id)
    if payload_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "managed_artifact_payload_not_found",
                "message": "该项目没有此制品的可下载托管副本",
            },
        )
    return FileResponse(
        payload_path,
        media_type=artifact.media_type,
        filename=payload_path.name,
    )


@router.post("/projects/{project_id}/artifacts/references", status_code=201)
def register_external_artifact(
    project_id: str,
    payload: ExternalArtifactReferenceCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    try:
        artifact, location = artifact_storage.register_external(
            session,
            project_id=project_id,
            kind=payload.kind,
            uri=payload.uri,
            content_hash=payload.content_hash,
            size_bytes=payload.size_bytes,
            media_type=payload.media_type,
            availability_state=payload.availability_state,
            worker_id=payload.worker_id,
            metadata={**payload.metadata, "project_id": project_id},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_external_artifact", "message": str(exc)},
        ) from exc
    add_audit(
        session,
        project_id=project_id,
        action="artifact.reference.register",
        subject_type="artifact",
        subject_id=artifact.id,
        payload={"location_id": location.id, "uri": location.uri},
    )
    session.commit()
    return artifact_view(session, artifact)
