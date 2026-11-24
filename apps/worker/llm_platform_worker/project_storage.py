from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def _local_workspace(workspace_uri: str) -> Path:
    value = str(workspace_uri or "").strip()
    if not value:
        raise ValueError("Task project has no configured workspace")
    parsed = urlparse(value)
    if parsed.scheme and len(parsed.scheme) > 1:
        if parsed.scheme != "file":
            raise ValueError("Task project workspace must be a local path or file URI")
        raw_path = unquote(parsed.path)
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        value = raw_path
    workspace = Path(value).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Task project workspace is not accessible on this Worker: {workspace}")
    return workspace


def _project_platform_root(workspace_uri: str) -> Path:
    workspace = _local_workspace(workspace_uri)
    from llm_platform_core.storage_layout import project_state_root
    root = project_state_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def worker_project_workspace(
    project_id: str,
    *,
    workspace_root: str | Path | None,
    legacy_workspace_uri: str,
    project_name: str = "",
    system_kind: str = "",
    project_storage_key: str = "",
) -> Path:
    """Resolve the project workspace that is local to this Worker.

    A configured Worker root deliberately takes precedence over the control
    plane's workspace URI, which may use a path syntax that is invalid on this
    operating system. Without a root we retain the original shared-filesystem
    behavior for existing local installations.
    """
    from llm_platform_core.storage_layout import project_directory, unified_storage_root
    unified = unified_storage_root()
    if unified:
        if system_kind == "global_data_catalog":
            target = unified / "platform" / "global-data-jobs"
        elif system_kind:
            target = unified / "platform" / "packages" / project_directory(project_id, system_kind)
        else:
            key = project_storage_key or project_directory(project_id, project_name)
            if Path(key).name != key or not (key == project_id or key.endswith("--" + project_id)):
                raise ValueError("Invalid project storage key")
            target = unified / "projects" / key
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve()
    if workspace_root is None or not str(workspace_root).strip():
        return _local_workspace(legacy_workspace_uri)

    root = Path(workspace_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    projects_root = root / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    project_workspace = scoped_runtime_path(projects_root, project_id)
    project_workspace.mkdir(parents=True, exist_ok=True)
    return project_workspace.resolve()


def project_run_root(workspace_uri: str) -> Path:
    root = _project_platform_root(workspace_uri) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def project_data_asset_root(workspace_uri: str) -> Path:
    from llm_platform_core.storage_layout import data_asset_root
    root = data_asset_root(workspace_uri)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def project_training_run_root(workspace_uri: str) -> Path:
    root = _project_platform_root(workspace_uri) / "training-runs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def scoped_runtime_path(root: Path, *components: str) -> Path:
    if not components or any(not str(component).strip() for component in components):
        raise ValueError("Runtime path components must not be empty")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*(str(component) for component in components)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Runtime path escapes its project-scoped root") from exc
    if candidate == resolved_root:
        raise ValueError("Runtime path must be below its project-scoped root")
    return candidate
