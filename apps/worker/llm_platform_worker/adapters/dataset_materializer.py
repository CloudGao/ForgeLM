from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..project_storage import project_data_asset_root, scoped_runtime_path
from .data_import import DataImportAdapter

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
GetDesiredState = Callable[[], str]


def _file_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _directory_hash(path: Path) -> tuple[str, int]:
    records: list[dict[str, Any]] = []
    size = 0
    symbolic_link = next(
        (candidate for candidate in path.rglob("*") if candidate.is_symlink()), None
    )
    if symbolic_link is not None:
        raise ValueError(f"Dataset snapshot contains a symbolic link: {symbolic_link}")
    for child in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        digest, file_size = _file_hash(child)
        size += file_size
        records.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size_bytes": file_size,
                "sha256": digest,
            }
        )
    return DataImportAdapter._manifest_hash(records), size


def _content_identity(path: Path) -> tuple[str, int]:
    return _directory_hash(path) if path.is_dir() else _file_hash(path)


class DatasetMaterializerAdapter:
    """Copy a verified snapshot into the target Worker's persistent replica store."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime_context = task.get("runtime_context") or {}
        binding = payload.get("materialization_input") or {}
        source = Path(str(binding.get("runtime_path") or "")).resolve()
        expected_hash = str(payload.get("expected_content_hash") or "")
        materialization_id = str(payload.get("materialization_id") or "")
        dataset_version_id = str(payload.get("dataset_version_id") or "")
        worker_id = str(runtime_context.get("worker_id") or "") or None
        project_workspace = str(runtime_context.get("project_workspace") or "")
        if not source.exists():
            raise FileNotFoundError("Dataset materialization source is unavailable")
        if not expected_hash or not materialization_id or not dataset_version_id:
            raise ValueError("Dataset materialization task is missing its frozen identity")
        if get_desired_state and get_desired_state() in {"paused", "cancelled"}:
            return {"controlled_state": get_desired_state()}

        replica_root = scoped_runtime_path(
            project_data_asset_root(project_workspace), "dataset-replicas"
        )
        replica_root.mkdir(parents=True, exist_ok=True)
        version_root = scoped_runtime_path(replica_root, dataset_version_id)
        version_root.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix if source.is_file() else ""
        target = scoped_runtime_path(version_root, f"{expected_hash}{suffix}")
        source_hash, source_size = await asyncio.to_thread(_content_identity, source)
        if source_hash != expected_hash:
            raise ValueError("Dataset materialization source failed content verification")
        if target.exists():
            target_hash, target_size = await asyncio.to_thread(_content_identity, target)
            if target_hash == expected_hash:
                await emit(
                    "progress",
                    {"progress": 0.98, "phase": "reusing_verified_replica"},
                )
                return self._result(
                    target,
                    expected_hash,
                    target_size,
                    materialization_id,
                    dataset_version_id,
                    worker_id,
                )

        staging = scoped_runtime_path(
            version_root, f".{expected_hash}.{materialization_id}.staging"
        )
        if staging.exists():
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()
        await emit(
            "progress",
            {"progress": 0.15, "phase": "copying_dataset_replica", "size_bytes": source_size},
        )
        if source.is_dir():
            await asyncio.to_thread(shutil.copytree, source, staging)
        else:
            await asyncio.to_thread(shutil.copy2, source, staging)
        if get_desired_state and get_desired_state() in {"paused", "cancelled"}:
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                staging.unlink(missing_ok=True)
            return {"controlled_state": get_desired_state()}
        copied_hash, copied_size = await asyncio.to_thread(_content_identity, staging)
        if copied_hash != expected_hash:
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                staging.unlink(missing_ok=True)
            raise ValueError("Copied dataset replica failed content verification")
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(staging, target)
        await emit(
            "progress",
            {"progress": 0.98, "phase": "dataset_replica_verified", "size_bytes": copied_size},
        )
        return self._result(
            target,
            expected_hash,
            copied_size,
            materialization_id,
            dataset_version_id,
            worker_id,
        )

    @staticmethod
    def _result(
        target: Path,
        content_hash: str,
        size_bytes: int,
        materialization_id: str,
        dataset_version_id: str,
        worker_id: str | None,
    ) -> dict[str, Any]:
        return {
            "materialized_uri": target.as_uri(),
            "outputs": [
                {
                    "ref": "dataset",
                    "kind": "dataset_snapshot",
                    "storage_type": "external",
                    "uri": target.as_uri(),
                    "content_hash": content_hash,
                    "size_bytes": size_bytes,
                    "media_type": (
                        "application/vnd.llm-platform.dataset-directory"
                        if target.is_dir()
                        else "application/octet-stream"
                    ),
                    "availability_state": "available",
                    "worker_id": worker_id,
                    "metadata": {
                        "materialization_id": materialization_id,
                        "dataset_version_id": dataset_version_id,
                        "verified": True,
                    },
                }
            ],
        }
