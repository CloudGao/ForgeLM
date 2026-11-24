from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_SUBDIRECTORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _load_spec() -> dict[str, Any]:
    spec_path = Path(os.environ["LLM_PLATFORM_TASK_SPEC"]).resolve()
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    return value.get("task") or {}


def _slug(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return (normalized or fallback)[:120]


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Release runtime contains a symbolic link: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    task = _load_spec()
    payload = task.get("payload") or {}
    release = payload.get("release") or {}
    model = payload.get("model") or {}
    source = Path(str(model.get("runtime_source") or "")).resolve()
    if not source.is_dir() or not (source / "forgelm-runtime.json").is_file():
        raise ValueError("Local publisher requires a materialized ForgeLM runtime directory")
    root = Path(os.environ["LLM_PLATFORM_RELEASE_ROOT"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Administrator-configured release root does not exist")
    config = payload.get("config") or {}
    subdirectory = str(config.get("target_subdirectory") or "")
    if not _SAFE_SUBDIRECTORY.fullmatch(subdirectory):
        raise ValueError("target_subdirectory must be one safe path segment")
    target_parent = (root / subdirectory / str(task.get("project_id") or "")).resolve()
    target_parent.relative_to(root)
    target_parent.mkdir(parents=True, exist_ok=True)
    destination = (
        target_parent
        / _slug(str(release.get("name") or ""), "release")
        / _slug(str(release.get("version") or ""), "version")
    ).resolve()
    destination.relative_to(root)
    marker_name = "forgelm-delivery.json"
    artifact = payload.get("artifact") or {}
    identity = {
        "release_id": release.get("id"),
        "release_version": release.get("version"),
        "runtime_artifact_id": artifact.get("id"),
        "runtime_artifact_content_hash": artifact.get("content_hash"),
    }
    if destination.exists():
        marker = destination / marker_name
        existing = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if all(existing.get(key) == value for key, value in identity.items()):
            _emit(
                "result",
                delivery={
                    **identity,
                    "target_kind": "local_directory",
                    "immutable_locator": f"{destination.as_uri()}#sha256={existing.get('tree_hash')}",
                    "uri": destination.as_uri(),
                    "content_hash": existing.get("tree_hash"),
                    "idempotent": True,
                },
            )
            return
        raise FileExistsError("Delivery destination already contains different immutable content")

    _emit("progress", phase="validating_target", progress=0.1)
    source_hash = _tree_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forgelm-release-", dir=destination.parent)).resolve()
    staging.relative_to(root)
    try:
        shutil.rmtree(staging)
        _emit("progress", phase="publishing", progress=0.25)
        shutil.copytree(source, staging, symlinks=False)
        copied_hash = _tree_hash(staging)
        if copied_hash != source_hash:
            raise RuntimeError("Published directory failed content verification")
        marker = {
            "schema": "forgelm-release-delivery-v1",
            **identity,
            "tree_hash": copied_hash,
            "published_at": datetime.now(UTC).isoformat(),
        }
        (staging / marker_name).write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(destination)
        _emit("progress", phase="verifying_remote", progress=1.0)
        _emit(
            "result",
            delivery={
                **identity,
                "target_kind": "local_directory",
                "immutable_locator": f"{destination.as_uri()}#sha256={copied_hash}",
                "uri": destination.as_uri(),
                "content_hash": copied_hash,
                "idempotent": False,
            },
        )
    finally:
        if staging.exists():
            staging.relative_to(root)
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
