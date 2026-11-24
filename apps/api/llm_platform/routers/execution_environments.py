from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import RemoteWorkerEnvironment, RemoteWorkerOperation, ResourceLease, WorkerNode
from ..schemas import (
    RemoteDownloadPolicyUpdate,
    RemoteWorkerEnvironmentCreate,
    RemoteWorkerOperationCreate,
)
from ..services.remote_workers import (
    RemoteWorkerError,
    capture_host_key,
    remote_environment_view,
    remote_operation_view,
    resolve_private_key,
    start_remote_worker_operation,
)
from ..services.ssh_credentials import prepare_credential
from ..services.ssh_tunnels import tunnel_manager
from .common import get_or_404
from .system import require_local_control

router = APIRouter(tags=["execution-environments"])


def _environment_with_latest_operation(
    session: Session, environment: RemoteWorkerEnvironment
) -> dict:
    latest = session.scalar(
        select(RemoteWorkerOperation)
        .where(RemoteWorkerOperation.environment_id == environment.id)
        .order_by(RemoteWorkerOperation.created_at.desc())
        .limit(1)
    )
    return {
        **remote_environment_view(environment),
        "latest_operation": remote_operation_view(latest) if latest else None,
    }


@router.get("/execution-environments/remote", dependencies=[Depends(require_local_control)])
def list_remote_worker_environments(session: Session = Depends(get_session)) -> list[dict]:
    environments = session.scalars(
        select(RemoteWorkerEnvironment).order_by(RemoteWorkerEnvironment.created_at.desc())
    ).all()
    return [_environment_with_latest_operation(session, item) for item in environments]


@router.post(
    "/execution-environments/remote",
    status_code=201,
    dependencies=[Depends(require_local_control)],
)
def create_remote_worker_environment(
    payload: RemoteWorkerEnvironmentCreate, session: Session = Depends(get_session)
) -> dict:
    if session.scalar(select(WorkerNode).where(WorkerNode.name == payload.name)) is not None:
        raise HTTPException(
            status_code=409,
            detail={"code": "worker_name_in_use", "message": "Worker 名称已被占用"},
        )
    existing_endpoint = session.scalar(
        select(RemoteWorkerEnvironment).where(
            RemoteWorkerEnvironment.ssh_host == payload.ssh_host,
            RemoteWorkerEnvironment.ssh_port == payload.ssh_port,
        )
    )
    if existing_endpoint is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "remote_endpoint_in_use",
                "message": "该 SSH 主机与端口已有受管 Worker 环境",
            },
        )
    if payload.workspace_root == "/" or payload.install_root == "/":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsafe_remote_root",
                "message": "远程安装目录和工作目录不能是根目录",
            },
        )
    try:
        if payload.credential_ref:
            resolve_private_key(payload.credential_ref)
        host_key, fingerprint = capture_host_key(
            payload.ssh_host,
            payload.ssh_port,
            expected_fingerprint=payload.expected_host_key_fingerprint,
        )
        credential_ref = prepare_credential(payload, host_key)
    except (RemoteWorkerError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "remote_worker_preflight_failed", "message": str(exc)},
        ) from exc
    environment = RemoteWorkerEnvironment(
        name=payload.name,
        ssh_host=payload.ssh_host,
        ssh_port=payload.ssh_port,
        ssh_user=payload.ssh_user,
        credential_ref=credential_ref,
        host_key=host_key,
        host_key_fingerprint=fingerprint,
        control_plane_url=payload.control_plane_url,
        workspace_root=payload.workspace_root,
        install_root=payload.install_root,
        profile=payload.profile,
        labels=payload.labels,
        desired_release=payload.desired_release,
        status="registered",
        metadata_json={
            "deployment": {
                "transport": payload.transport,
                "runtime": payload.runtime,
                "service_mode": payload.service_mode,
            }
        },
    )
    session.add(environment)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_remote_environment", "message": "远程环境名称已存在"},
        ) from exc
    if payload.auto_provision:
        operation = RemoteWorkerOperation(
            environment_id=environment.id,
            kind="provision",
            status="queued",
            phase="queued",
        )
        session.add(operation)
        session.commit()
        start_remote_worker_operation(operation.id)
    return _environment_with_latest_operation(session, environment)


@router.get(
    "/execution-environments/remote/{environment_id}",
    dependencies=[Depends(require_local_control)],
)
def get_remote_worker_environment(
    environment_id: str, session: Session = Depends(get_session)
) -> dict:
    environment = get_or_404(session, RemoteWorkerEnvironment, environment_id, "远程 Worker 环境")
    view = _environment_with_latest_operation(session, environment)
    operations = session.scalars(
        select(RemoteWorkerOperation)
        .where(RemoteWorkerOperation.environment_id == environment.id)
        .order_by(RemoteWorkerOperation.created_at.desc())
        .limit(20)
    ).all()
    view["operations"] = [remote_operation_view(item) for item in operations]
    return view


@router.delete(
    "/execution-environments/remote/{environment_id}",
    status_code=204,
    dependencies=[Depends(require_local_control)],
)
def delete_remote_worker_environment(
    environment_id: str, session: Session = Depends(get_session)
) -> Response:
    environment = get_or_404(session, RemoteWorkerEnvironment, environment_id, "远程 Worker 环境")
    running = session.scalar(
        select(RemoteWorkerOperation).where(
            RemoteWorkerOperation.environment_id == environment.id,
            RemoteWorkerOperation.status.in_({"queued", "running"}),
        )
    )
    if running:
        raise HTTPException(
            status_code=409,
            detail={"code": "remote_operation_running", "message": "远程操作进行中，暂不能删除"},
        )
    session.delete(environment)
    session.commit()
    tunnel_manager.remove(environment_id)
    return Response(status_code=204)


@router.put("/execution-environments/remote/{environment_id}/download-policy",
            dependencies=[Depends(require_local_control)])
def set_remote_download_policy(environment_id: str, payload: RemoteDownloadPolicyUpdate,
                               session: Session = Depends(get_session)) -> dict:
    from .common import add_audit
    environment = get_or_404(session, RemoteWorkerEnvironment, environment_id, "远程 Worker 环境")
    if session.scalar(select(RemoteWorkerOperation.id).where(
        RemoteWorkerOperation.environment_id == environment.id,
        RemoteWorkerOperation.status.in_({"queued", "running"})).limit(1)):
        raise HTTPException(409, "部署进行中，请在完成后修改下载策略")
    metadata = dict(environment.metadata_json or {})
    metadata["deployment"] = {**metadata.get("deployment", {}), **payload.model_dump()}
    environment.metadata_json = metadata
    add_audit(session, project_id=None, action="remote.download_policy.update",
              subject_type="remote_worker_environment", subject_id=environment.id,
              payload=payload.model_dump())
    session.commit()
    return _environment_with_latest_operation(session, environment)


@router.post(
    "/execution-environments/remote/{environment_id}/operations",
    status_code=202,
    dependencies=[Depends(require_local_control)],
)
def create_remote_worker_operation(
    environment_id: str,
    payload: RemoteWorkerOperationCreate,
    session: Session = Depends(get_session),
) -> dict:
    environment = get_or_404(session, RemoteWorkerEnvironment, environment_id, "远程 Worker 环境")
    running = session.scalar(
        select(RemoteWorkerOperation).where(
            RemoteWorkerOperation.environment_id == environment.id,
            RemoteWorkerOperation.status.in_({"queued", "running"}),
        )
    )
    if running:
        raise HTTPException(
            status_code=409,
            detail={"code": "remote_operation_running", "message": "该环境已有远程操作进行中"},
        )
    if (
        payload.kind != "probe"
        and not payload.wait_until_idle
        and environment.worker_id
        and session.scalar(
            select(ResourceLease.id)
            .where(
                ResourceLease.worker_id == environment.worker_id,
                ResourceLease.released_at.is_(None),
            )
            .limit(1)
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "remote_worker_busy",
                "message": "Worker 仍有任务租约，请结束任务后再升级或修复",
            },
        )
    operation = RemoteWorkerOperation(
        environment_id=environment.id,
        kind=payload.kind,
        status="queued",
        phase="queued",
    )
    session.add(operation)
    session.commit()
    operation_id = operation.id
    start_remote_worker_operation(operation_id)
    return remote_operation_view(operation)


@router.get(
    "/execution-environments/remote-operations/{operation_id}",
    dependencies=[Depends(require_local_control)],
)
def get_remote_worker_operation(operation_id: str, session: Session = Depends(get_session)) -> dict:
    operation = get_or_404(session, RemoteWorkerOperation, operation_id, "远程 Worker 操作")
    return remote_operation_view(operation)
