"""Worker-local inventory and exact, platform-owned cache reclamation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from llm_platform_core.storage_cleanup import (
    checkpoint_group,
    inspect_target,
    remove_checkpoint_group,
    remove_target,
)


def model_deletion_storage(root: Path, payload: dict) -> dict:
    """Only exact server-planned targets under this Worker's configured root."""
    results = []
    for item in payload.get("deletion_items", []):
        result = {"item_id": item["item_id"], "path": item["path"]}
        try:
            if item.get("blocked"):
                raise ValueError("模型或文件在盘点后出现新的引用，已保留")
            path = Path(item["path"])
            relative = path.relative_to(root)
            parts = relative.parts
            if ".llm-platform" not in parts or parts.index(".llm-platform") + 1 >= len(parts):
                raise ValueError("不是平台托管的产物目录")
            branch = parts[parts.index(".llm-platform") + 1]
            if branch not in {"artifacts", "training-runs", "data-assets"}:
                raise ValueError("不支持回收此目录")
            if payload["storage_action"] == "scan":
                stats = checkpoint_group(path, root)[0] if item.get("checkpoint_children") else inspect_target(path, root)
                result.update(stats, status="inspected")
            else:
                remove = remove_checkpoint_group if item.get("checkpoint_children") else remove_target
                result.update(remove({**item, "root": str(root)}))
        except (OSError, ValueError) as exc:
            result.update(status="failed", reason=str(exc))
        results.append(result)
    return {"items": results, "reclaimed_bytes": sum(item.get("actual_reclaimed_bytes", 0) for item in results)}


def inspect_tree(path: Path) -> dict:
    digest = hashlib.sha256()
    size = count = 0
    if path.is_symlink():
        raise ValueError("Symlink storage candidates are protected")
    entries = [path] if path.is_file() else sorted(path.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("Storage tree contains a symlink")
        if entry.is_file():
            stat = entry.stat()
            count += 1
            size += stat.st_size
            digest.update(f"{entry.relative_to(path) if entry != path else '.'}|"
                          f"{stat.st_size}|{stat.st_mtime_ns}|{stat.st_ino}\n".encode())
    return {"size_bytes": size, "file_count": count, "fingerprint": digest.hexdigest()}


def overlaps_artifact(path: Path, uris: list[str]) -> bool:
    for uri in uris:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            continue
        protected = Path(url2pathname(parsed.path)).resolve()
        if path == protected or path in protected.parents or protected in path.parents:
            return True
    return False


def candidate_paths(root: Path, payload: dict):
    from ..generation_storage import generation_candidates

    yield from generation_candidates(root, payload)
    # HF datasets builder/Arrow cache, never the Hub download/weights directory.
    shared = root / ".cache" / "huggingface" / "datasets"
    if shared.exists():
        yield shared, "computation_cache", True, "可重建的 HF Datasets 计算缓存"
    for owner in payload.get("terminal_attempts", []):
        project = str(owner["project_id"])
        task = str(owner["task_id"])
        attempt = str(owner["attempt_id"])
        for value in (project, task, attempt):
            if not value or any(c not in "0123456789abcdef-" for c in value):
                raise ValueError("Invalid task ownership identifier")
        base = root / "projects" / project / ".llm-platform" / "runs" / task / attempt
        from llm_platform_core.storage_layout import unified_storage_root
        if unified_storage_root() == root:
            matches = list((root / "projects").glob("*--" + project))
            workspace = matches[0] if len(matches) == 1 else root / "projects" / project
            base = workspace / "runs" / task / attempt
        for name in ("data-juicer-scratch", "data-juicer-export", "analysis-shards"):
            path = base / name
            if path.exists():
                yield path, "scratch", True, "已终态任务的可重建工作文件"
    for location in payload.get("locations", []):
        parsed = urlparse(location["uri"])
        if parsed.scheme != "file":
            continue
        path = Path(url2pathname(parsed.path))
        if path.exists() and path.resolve().is_relative_to(root):
            yield path, location.get("kind", "artifact"), bool(location.get("reclaimable")), location.get("reason", "已登记制品受保留策略保护")


def inventory(root: Path, payload: dict) -> dict:
    entries = []
    seen = set()
    for path, category, reclaimable, reason in candidate_paths(root, payload):
        if str(path) in seen:
            continue
        seen.add(str(path))
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or path.is_symlink():
            continue
        try:
            stats = inspect_tree(path)
        except (OSError, ValueError) as exc:
            entries.append({"path": str(path), "category": category,
                            "reclaimable": False, "reason": str(exc), "size_bytes": 0})
            continue
        location = next((loc for loc in payload.get("locations", [])
                         if Path(url2pathname(urlparse(loc["uri"]).path)).resolve() == resolved), None)
        protected = [uri for uri in payload.get("protected_uris", [])
                     if not (location and location.get("reclaimable") and uri == location["uri"])]
        if reclaimable and overlaps_artifact(resolved, protected):
            reclaimable, reason = False, "与已登记制品位置重叠，禁止回收"
        entries.append({"id": hashlib.sha256(str(path).encode()).hexdigest(),
                        "path": str(path), "category": category, "reclaimable": reclaimable,
                        "reason": reason, "location_id": location["id"] if location else None,
                        "retention": location.get("retention", "auto") if location else "auto", **stats})
    # Top-level accounting includes unknown content, without authorizing its deletion.
    categories = []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            continue
        try:
            categories.append({"path": str(child), **inspect_tree(child)})
        except (OSError, ValueError) as exc:
            categories.append({"path": str(child), "error": str(exc)})
    usage = shutil.disk_usage(root)
    return {"root": str(root), "items": entries, "categories": categories,
            "free_bytes": usage.free, "total_bytes": usage.total,
            "reclaimable_bytes": sum(x["size_bytes"] for x in entries if x["reclaimable"])}


def cleanup(root: Path, payload: dict) -> dict:
    allowed = {str(p.resolve()): p for p, _, eligible, _ in candidate_paths(root, payload)
               if eligible and not p.is_symlink() and p.resolve().is_relative_to(root)}
    before = shutil.disk_usage(root).free
    results = []
    for item in payload.get("candidates", []):
        path = allowed.get(item["path"])
        result = {"path": item["path"], "location_id": item.get("location_id"),
                  "status": "skipped", "reclaimed_bytes": 0}
        try:
            if path is None or overlaps_artifact(path.resolve(), payload.get("protected_uris", [])):
                raise ValueError("Candidate is no longer eligible")
            current = inspect_tree(path)
            if current != {key: item[key] for key in current}:
                raise ValueError("Candidate changed after inventory; scan again")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            result.update(status="deleted", reclaimed_bytes=current["size_bytes"])
        except (OSError, ValueError) as exc:
            result["reason"] = str(exc)
        results.append(result)
    return {"items": results, "reclaimed_bytes": sum(x["reclaimed_bytes"] for x in results),
            "disk_free_delta_bytes": shutil.disk_usage(root).free - before}


class StorageMaintenanceAdapter:
    async def run(self, task, emit, get_desired_state=None):
        context = task.get("runtime_context") or {}
        configured = context.get("worker_workspace")
        if not configured:
            raise ValueError("Storage maintenance requires a configured Worker workspace")
        root = Path(configured).resolve()
        if not root.is_dir() or root == Path(root.anchor):
            raise ValueError("Invalid Worker workspace")
        payload = task.get("payload") or {}
        action = payload.get("storage_action")
        if action not in {"scan", "cleanup"}:
            raise ValueError("Unsupported storage maintenance action")
        if get_desired_state and get_desired_state() in {"cancelled", "paused"}:
            return {"controlled_state": get_desired_state()}
        await emit("progress", {"phase": f"storage_{action}", "progress": 0.1})
        operation = model_deletion_storage if payload.get("model_deletion_plan_id") else inventory if action == "scan" else cleanup
        result = await asyncio.to_thread(operation, root, payload)
        await emit("log", {"stream": "stdout", "message": json.dumps(
            {"action": action, "items": len(result["items"]),
             "bytes": result.get("reclaimed_bytes", result.get("reclaimable_bytes", 0))})})
        return {"storage_action": action, **result}
