"""Mount-based storage defaults shared by managed deployment and Worker reporting."""
from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path, PurePosixPath


def mount_table(text: str) -> list[dict[str, str]]:
    mounts = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or "-" not in parts:
            continue
        separator = parts.index("-")
        if len(parts) <= separator + 2:
            continue
        def decode(value: str) -> str:
            return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
        mounts.append({"mount": decode(parts[4]), "device": parts[2],
                       "source": decode(parts[separator + 2]), "fstype": parts[separator + 1]})
    return mounts


def storage_layout(workspace: str, mountinfo: str) -> dict:
    mounts = mount_table(mountinfo)
    system = next((m for m in mounts if m["mount"] == "/"), None)
    data = next((m for m in mounts if m["mount"] == "/root/autodl-tmp"), None)
    # A plain directory on / is not a separate AutoDL data disk.
    if data and system and data["device"] == system["device"]:
        data = None
    effective = workspace
    if data and not PurePosixPath(workspace).is_relative_to(data["mount"]):
        effective = f'{data["mount"]}/forgelm'
    return {"provider": "autodl" if data else "generic", "system_disk": system,
            "data_disk": data, "workspace_root": effective,
            "runtime_root": f"{effective}/.runtime", "cache_root": f"{effective}/.cache",
            "temporary_root": f"{effective}/.tmp"}


def local_storage_layout(workspace: str) -> dict:
    source = Path("/proc/self/mountinfo")
    layout = storage_layout(workspace, source.read_text() if source.is_file() else "")
    # Reporting describes the actual configured workspace, never an imaginary move.
    layout["workspace_root"] = workspace
    for key in ("system_disk", "data_disk"):
        disk = layout[key]
        if disk:
            usage = shutil.disk_usage(disk["mount"])
            disk.update(total_bytes=usage.total, free_bytes=usage.free)
    return layout


# Payload ownership is independent of its producer project's historical identity.
PUBLIC_DATA_KINDS = frozenset({
    "dataset_snapshot", "dataset_record_manifest", "data_mixture_manifest",
    "supervision_record_index", "supervision_mixture_manifest", "dataset_index",
    "eval_dataset", "evaluation_dataset", "data_processing_recipe",
})


def unified_storage_root(configured: str | None = None) -> Path | None:
    import os
    value = configured if configured is not None else os.getenv("LLM_PLATFORM_STORAGE_ROOT", "")
    if not value or not str(value).strip():
        return None
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError("Storage root must be absolute")
    return root.resolve()


def project_directory(project_id: str, name: str = "") -> str:
    from uuid import UUID
    UUID(project_id)
    slug = re.sub(r"[^\w.-]+", "-", name, flags=re.UNICODE).strip(".-")[:60]
    return f"{slug}--{project_id}" if slug else project_id


def project_state_root(workspace: str | Path, configured: str | None = None) -> Path:
    workspace = Path(workspace).resolve()
    root = unified_storage_root(configured)
    return workspace if root and workspace.is_relative_to(root) else workspace / ".llm-platform"


def data_asset_root(workspace: str | Path, configured: str | None = None) -> Path:
    root = unified_storage_root(configured)
    return root / "dataset" / "objects" / "external" if root else Path(workspace) / ".llm-platform" / "data-assets"


def derived_asset_root(workspace: str | Path) -> Path:
    root = unified_storage_root()
    return project_state_root(workspace) / "artifacts" / "derived" if root else Path(workspace) / ".llm-platform" / "data-assets"


def platform_state_root(configured: str | None = None) -> Path:
    root = unified_storage_root(configured)
    return root / "platform" if root else Path.cwd() / ".llm-platform"



@lru_cache(maxsize=8)
def _relocations(path: str, modified: int, size: int):
    import json
    return sorted(json.loads(Path(path).read_text(encoding="utf-8")), key=lambda e: len(e["source"]), reverse=True)


def resolve_relocated_path(value: str) -> str:
    """Resolve only explicit path prefixes recorded by verified storage migrations.

    This is a runtime view. Frozen task specifications and lineage stay unchanged.
    """
    import os
    from urllib.parse import unquote, urlparse
    root = unified_storage_root()
    if not root or not isinstance(value, str):
        return value
    uri = value.startswith("file:")
    raw = unquote(urlparse(value).path) if uri else value
    norm = raw.replace("\\", "/")
    if re.match(r"^/[A-Za-z]:/", norm):
        norm = norm[1:]
    if re.match(r"^/mnt/[a-zA-Z]/", norm):
        norm = norm[5] + ":/" + norm[7:]
    if not (norm.startswith("/") or re.match(r"^[A-Za-z]:/", norm)):
        return value
    journal = root / "platform" / "config" / "path-relocations.json"
    if not journal.is_file():
        return value
    stat = journal.stat()
    for entry in _relocations(str(journal), stat.st_mtime_ns, stat.st_size):
        old = entry["source"].replace("\\", "/").rstrip("/")
        if old.startswith("//wsl.localhost/"):
            parts = old.split("/", 4)
            if len(parts) == 5 and os.environ.get("WSL_DISTRO_NAME") == parts[3]:
                old = "/" + parts[4]
        a, b = (norm.casefold(), old.casefold()) if re.match(r"^[A-Za-z]:/", old) else (norm, old)
        if a != b and not a.startswith(b + "/"):
            continue
        target = entry["target"].replace("\\", "/").rstrip("/") + norm[len(old):]
        if os.name != "nt" and re.match(r"^[A-Za-z]:/", target):
            target = "/mnt/" + target[0].lower() + target[2:]
        if not Path(target).exists():
            raise ValueError("Migrated path is unavailable: " + target)
        return Path(target).as_uri() if uri else target
    return value


def relocated_runtime_view(value):
    if isinstance(value, dict):
        return {k: relocated_runtime_view(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated_runtime_view(v) for v in value]
    return resolve_relocated_path(value) if isinstance(value, str) else value
