"""Verified, atomic model materialization shared by attempts in a project."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if os.name == "nt":
        import msvcrt
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        # LK_LOCK has a short retry budget; callers retry without blocking heartbeats.
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            raise
    else:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise
    return handle


def _unlock(handle):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    handle.close()


def _ready(root: Path, identity: str) -> Path | None:
    try:
        manifest = json.loads((root / "ready.json").read_text())
        if manifest["identity"] != identity:
            return None
        for name, size, modified in manifest["files"]:
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
                return None
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                return None
        model = root / manifest["model"]
        if model.resolve().is_relative_to(root.resolve()) and (model / "config.json").is_file():
            return model
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


async def cached_model(agent, task, run_dir, artifact, source_path, emit=None, desired_state=lambda: "running"):
    """Authorization is performed by materialize_inference_model before this call."""
    identity = str(artifact.get("content_hash") or "")
    if not identity:
        raise ValueError("Model artifact has no immutable content hash")
    key = hashlib.sha256((str(artifact["id"]) + ":" + identity).encode()).hexdigest()
    project_root = run_dir.parents[2] if run_dir.parent.parent.name == "runs" else run_dir.parent
    base = project_root / ".cache" / "inference"
    base.mkdir(parents=True, exist_ok=True)
    root = base / key
    lock = None
    try:
        while lock is None:
            if desired_state() in {"cancelled", "paused"}:
                raise RuntimeError("Model preparation cancelled")
            try:
                lock = await asyncio.to_thread(_lock, base / (key + ".lock"))
            except OSError:
                await asyncio.sleep(0.1)
        model = await asyncio.to_thread(_ready, root, identity)
        if model is not None:
            return model, True
        temporary = Path(tempfile.mkdtemp(prefix=key + ".", dir=base))
        try:
            if source_path is None:
                if emit:
                    await emit("progress", {"phase": "downloading_model", "progress": 0.02})
                source_path = temporary / "payload.bin"
                digest = hashlib.sha256()
                async with agent.client.stream("GET", f"/api/agent/artifacts/{artifact['id']}/payload",
                        params={"project_id": str(task.get("project_id") or "")}, timeout=None) as response:
                    response.raise_for_status()
                    with source_path.open("wb") as output:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            if desired_state() in {"cancelled", "paused"}:
                                raise RuntimeError("Model preparation cancelled")
                            digest.update(chunk)
                            await asyncio.to_thread(output.write, chunk)
                if digest.hexdigest() != identity:
                    raise ValueError("Downloaded model artifact failed SHA-256 verification")
            if emit:
                await emit("progress", {"phase": "extracting_model", "progress": 0.04})
            model = await asyncio.to_thread(agent._extract_model_archive, source_path, temporary / "model")
            if model is None:
                raise ValueError("Model artifact must be a directory or a tar/zip archive")
            if desired_state() in {"cancelled", "paused"}:
                raise RuntimeError("Model preparation cancelled")
            # Keep only the verified expanded model, not a second archive copy.
            if source_path.parent == temporary:
                source_path.unlink()
            def publish():
                files = [(str(p.relative_to(temporary)), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in temporary.rglob("*") if p.is_file()]
                relative = model.relative_to(temporary)
                (temporary / "ready.json").write_text(json.dumps({"identity": identity,
                    "model": str(relative), "files": files}), encoding="utf-8")
                if root.exists():
                    shutil.rmtree(root)
                temporary.rename(root)
                return root / relative
            return await asyncio.to_thread(publish), False
        finally:
            if temporary.exists():
                await asyncio.to_thread(shutil.rmtree, temporary)
    finally:
        if lock is not None:
            await asyncio.to_thread(_unlock, lock)
