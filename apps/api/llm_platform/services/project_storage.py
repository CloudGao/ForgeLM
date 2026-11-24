from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy.orm import Session

from ..models import Project


class ProjectStorageError(ValueError):
    pass


def local_workspace_path(workspace_uri: str) -> Path:
    value = str(workspace_uri or "").strip()
    if not value:
        raise ProjectStorageError("项目必须先配置本地工作区")
    parsed = urlparse(value)
    if parsed.scheme and len(parsed.scheme) > 1:
        if parsed.scheme != "file":
            raise ProjectStorageError("项目工作区只支持本地路径或 file:// URI")
        raw_path = unquote(parsed.path)
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        value = raw_path
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ProjectStorageError(f"项目工作区不是控制面可访问的本地目录：{path}")
    return path


def project_workspace(session: Session, project_id: str) -> tuple[Project, Path]:
    project = session.get(Project, project_id)
    if project is None:
        raise ProjectStorageError("项目不存在")
    return project, local_workspace_path(project.workspace_uri)


def project_artifact_root(session: Session, project_id: str, *, create: bool = True) -> Path:
    project, workspace = project_workspace(session, project_id)
    from llm_platform_core.storage_layout import project_directory, unified_storage_root

    from ..config import get_settings
    unified = unified_storage_root(get_settings().storage_root)
    if unified:
        if project.system_kind == "global_data_catalog":
            workspace = unified / "platform" / "global-data-jobs"
        elif project.system_kind:
            workspace = unified / "platform" / "packages" / project_directory(project.id, project.system_kind)
        elif not workspace.is_relative_to(unified / "projects"):
            workspace = unified / "projects" / project_directory(project.id, project.name)
        root = workspace / "artifacts"
    else:
        root = workspace / ".llm-platform" / "artifacts"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def project_run_root(workspace_uri: str, *, create: bool = True) -> Path:
    from llm_platform_core.storage_layout import project_state_root

    from ..config import get_settings
    root = project_state_root(local_workspace_path(workspace_uri), get_settings().storage_root) / "runs"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def artifact_root(session: Session, project_id: str, kind: str, *, create: bool = True) -> Path:
    from llm_platform_core.storage_layout import PUBLIC_DATA_KINDS, unified_storage_root

    from ..config import get_settings
    root = unified_storage_root(get_settings().storage_root)
    if root and kind in PUBLIC_DATA_KINDS:
        result = root / "dataset" / "objects"
    elif root and kind == "model_weights":
        result = root / "weights" / "objects"
    else:
        return project_artifact_root(session, project_id, create=create)
    if create:
        result.mkdir(parents=True, exist_ok=True)
    return result.resolve()


def workspace_history(session: Session, project: Project) -> list[str]:
    """Registered current and audited former workspaces, for storage accounting."""
    from sqlalchemy import select

    from ..models import AuditEvent
    values = [project.workspace_uri] if project.workspace_uri else []
    for event in session.scalars(select(AuditEvent).where(
        AuditEvent.project_id == project.id, AuditEvent.action == "project.workspace.relocated")):
        previous = (event.payload or {}).get("previous_workspace")
        if previous and previous not in values:
            values.append(previous)
    return values
