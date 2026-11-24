from __future__ import annotations

import copy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Artifact,
    ArtifactLocation,
    Dataset,
    DatasetMaterialization,
    DatasetVersion,
    Task,
    TaskAttempt,
    WorkerNode,
    utcnow,
)
from .artifacts import artifact_storage
from .global_data import project_can_access_artifact

DOWNLOAD_SOURCE_TYPES = {
    "http",
    "https",
    "huggingface",
    "git",
}


class MaterializationRouteError(ValueError):
    pass


def classify_materialization(dataset: Dataset, version: DatasetVersion) -> str:
    if (
        version.parent_version_id
        or version.conversion_recipe_version_id
        or version.processing_recipe_version_id
    ):
        return "derived"
    if dataset.source_type in DOWNLOAD_SOURCE_TYPES:
        return "download"
    return "imported"


def freeze_import_materialization(
    dataset: Dataset,
    version: DatasetVersion,
    *,
    replay_payload: dict[str, Any] | None = None,
) -> None:
    kind = classify_materialization(dataset, version)
    version.materialization_kind = kind
    version.materialization_state = "pending"
    version.materialization_spec = {
        "schema_version": 1,
        "kind": kind,
        "source_type": dataset.source_type,
        "source_uri": dataset.source_uri,
        "source_config": copy.deepcopy(dataset.source_config or {}),
        **(
            {
                "replay_adapter_id": "builtin.data-import",
                "replay_payload": copy.deepcopy(replay_payload),
            }
            if kind == "download" and replay_payload
            else {}
        ),
    }


def freeze_derived_materialization(version: DatasetVersion, task: Task) -> None:
    version.materialization_kind = "derived"
    version.materialization_state = "pending"
    version.materialization_spec = {
        "schema_version": 1,
        "kind": "derived",
        "replay_adapter_id": task.adapter_id,
        "replay_payload": copy.deepcopy(task.payload or {}),
        "source_task_id": task.id,
        "parent_dataset_version_id": version.parent_version_id,
    }


def ensure_version_primary_location(
    session: Session,
    version: DatasetVersion,
    artifact: Artifact,
    *,
    preferred_worker_id: str | None = None,
) -> ArtifactLocation | None:
    spec = dict(version.materialization_spec or {})
    spec["expected_content_hash"] = artifact.content_hash
    spec["artifact_id"] = artifact.id
    if version.source_snapshot.get("files"):
        spec["expected_snapshot_files"] = copy.deepcopy(version.source_snapshot["files"])
    version.materialization_spec = spec
    if version.primary_location_id:
        current = session.get(ArtifactLocation, version.primary_location_id)
        if current is None:
            version.primary_location_id = None
        elif current.artifact_id != artifact.id:
            raise ValueError("dataset primary location belongs to different content")
        elif current.availability_state == "available":
            current.last_verified_at = current.last_verified_at or utcnow()
            version.materialization_state = "ready"
            return current
        else:
            version.materialization_state = "degraded"
            return current
    locations = session.scalars(
        select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.availability_state == "available",
        )
    ).all()
    dataset = session.get(Dataset, version.dataset_id)
    if dataset is not None:
        locations = [
            location
            for location in locations
            if location.project_id == dataset.project_id
            or (location.project_id is None and location.storage_type == "external")
        ]
    locations.sort(
        key=lambda location: (
            0 if preferred_worker_id and location.worker_id == preferred_worker_id else 1,
            0 if location.storage_type == "managed" else 1,
            location.created_at,
            location.id,
        )
    )
    if not locations:
        version.materialization_state = "degraded"
        return None
    primary = locations[0]
    primary.last_verified_at = primary.last_verified_at or utcnow()
    version.primary_location_id = primary.id
    version.materialization_state = "ready"
    return primary


def _route(
    route_id: str,
    strategy: str,
    *,
    adapter_id: str | None,
    cost: int,
    feasible: bool,
    reason: str,
    source_location: ArtifactLocation | None = None,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "strategy": strategy,
        "adapter_id": adapter_id,
        "cost": cost,
        "feasible": feasible,
        "reason": reason,
        "source_location_id": source_location.id if source_location else None,
        "source_worker_id": source_location.worker_id if source_location else None,
        "source_uri": source_location.uri if source_location else None,
    }


def _worker_supports(worker: WorkerNode, adapter_id: str) -> bool:
    capabilities = set(worker.capabilities or [])
    return "*" in capabilities or {
        adapter_id,
        "builtin.local-process",
    }.issubset(capabilities)


def _artifact_is_accessible_on_worker(
    session: Session, artifact_id: str, worker_id: str, project_id: str
) -> bool:
    locations = session.scalars(
        select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact_id,
            ArtifactLocation.availability_state == "available",
        )
    ).all()
    if not project_can_access_artifact(session, project_id, artifact_id):
        locations = [
            location
            for location in locations
            if location.project_id == project_id
            or (location.project_id is None and location.storage_type == "external")
        ]
    return any(
        location.storage_type == "managed"
        or location.worker_id is None
        or location.worker_id == worker_id
        for location in locations
    )


def _derived_inputs_are_accessible(
    session: Session,
    version: DatasetVersion,
    replay_payload: dict[str, Any],
    worker_id: str,
) -> bool:
    artifact_ids = {
        str(binding.get("artifact_id") or "")
        for key in ("data_conversion_input", "data_processing_input")
        if isinstance((binding := replay_payload.get(key)), dict) and binding.get("artifact_id")
    }
    if not artifact_ids and version.parent_version_id:
        parent = session.get(DatasetVersion, version.parent_version_id)
        if parent is not None and parent.artifact_id:
            artifact_ids.add(parent.artifact_id)
    dataset = session.get(Dataset, version.dataset_id)
    if dataset is None:
        return False
    return bool(artifact_ids) and all(
        _artifact_is_accessible_on_worker(session, artifact_id, worker_id, dataset.project_id)
        for artifact_id in artifact_ids
    )


def materialization_routes(
    session: Session, version: DatasetVersion, worker: WorkerNode
) -> list[dict[str, Any]]:
    if not version.artifact_id:
        return []
    artifact = session.get(Artifact, version.artifact_id)
    if artifact is None or artifact.kind != "dataset_snapshot":
        return []
    locations = session.scalars(
        select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.availability_state == "available",
        )
    ).all()
    dataset = session.get(Dataset, version.dataset_id)
    if dataset is not None:
        locations = [
            location
            for location in locations
            if location.project_id == dataset.project_id
            or (location.project_id is None and location.storage_type == "external")
        ]
    routes: list[dict[str, Any]] = []
    for location in locations:
        if location.worker_id == worker.id:
            if location.last_verified_at is not None:
                routes.append(
                    _route(
                        f"already-local:{location.id}",
                        "already_local",
                        adapter_id=None,
                        cost=0,
                        feasible=True,
                        reason="目标 Worker 已有通过校验的副本",
                        source_location=location,
                    )
                )
            elif location.storage_type == "external":
                routes.append(
                    _route(
                        f"local-verify:{location.id}",
                        "visible_copy",
                        adapter_id="builtin.dataset-materializer",
                        cost=5,
                        feasible=_worker_supports(worker, "builtin.dataset-materializer"),
                        reason="校验目标 Worker 的现有位置并登记稳定副本",
                        source_location=location,
                    )
                )
    copy_supported = _worker_supports(worker, "builtin.dataset-materializer")
    for location in locations:
        if location.storage_type == "managed":
            routes.append(
                _route(
                    f"managed-copy:{location.id}",
                    "managed_copy",
                    adapter_id="builtin.dataset-materializer",
                    cost=10,
                    feasible=copy_supported,
                    reason=(
                        "从控制面托管位置下载并校验"
                        if copy_supported
                        else "目标 Worker 未声明 dataset-materializer 能力"
                    ),
                    source_location=location,
                )
            )
        elif location.worker_id is None:
            routes.append(
                _route(
                    f"visible-copy:{location.id}",
                    "visible_copy",
                    adapter_id="builtin.dataset-materializer",
                    cost=15,
                    feasible=copy_supported,
                    reason=(
                        "从目标 Worker 可见的共享位置复制并校验"
                        if copy_supported
                        else "目标 Worker 未声明 dataset-materializer 能力"
                    ),
                    source_location=location,
                )
            )
        elif location.worker_id != worker.id:
            routes.append(
                _route(
                    f"unreachable-worker:{location.id}",
                    "worker_copy",
                    adapter_id=None,
                    cost=1000,
                    feasible=False,
                    reason="该外部位置只对另一台 Worker 可见，未配置共享传输通道",
                    source_location=location,
                )
            )
    spec = version.materialization_spec or {}
    replay_adapter = str(spec.get("replay_adapter_id") or "")
    replay_payload = spec.get("replay_payload")
    if version.materialization_kind == "download":
        feasible = bool(
            replay_adapter == "builtin.data-import" and replay_payload
        ) and _worker_supports(worker, "builtin.data-import")
        routes.append(
            _route(
                "download:frozen-source",
                "download",
                adapter_id="builtin.data-import",
                cost=30,
                feasible=feasible,
                reason=(
                    "按冻结互联网来源重新下载并校验"
                    if feasible
                    else "历史版本没有可重放的冻结下载规格，或目标 Worker 缺少能力"
                ),
            )
        )
    elif version.materialization_kind == "derived":
        has_replay = bool(replay_adapter and isinstance(replay_payload, dict) and replay_payload)
        supports_adapter = has_replay and _worker_supports(worker, replay_adapter)
        inputs_accessible = bool(
            supports_adapter
            and _derived_inputs_are_accessible(session, version, replay_payload, worker.id)
        )
        feasible = bool(supports_adapter and inputs_accessible)
        if not has_replay:
            reason = "历史版本没有可重放 Recipe"
        elif not supports_adapter:
            reason = "目标 Worker 缺少 Recipe 执行能力"
        elif not inputs_accessible:
            reason = "冻结输入快照在目标 Worker 不可达，请先同步输入版本"
        else:
            reason = "按冻结 Recipe 和输入身份重新加工并校验"
        routes.append(
            _route(
                "rebuild:frozen-recipe",
                "rebuild",
                adapter_id=replay_adapter or None,
                cost=40,
                feasible=feasible,
                reason=reason,
            )
        )
    return sorted(routes, key=lambda item: (int(item["cost"]), str(item["route_id"])))


def choose_materialization_route(
    routes: list[dict[str, Any]], *, selection_mode: str, route_id: str | None
) -> dict[str, Any]:
    if selection_mode == "manual":
        selected = next((route for route in routes if route["route_id"] == route_id), None)
        if selected is None:
            raise MaterializationRouteError("指定的数据物化路线不存在")
        if not selected["feasible"]:
            raise MaterializationRouteError(str(selected["reason"]))
        return selected
    selected = next((route for route in routes if route["feasible"]), None)
    if selected is None:
        raise MaterializationRouteError("当前没有到目标 Worker 的可行数据物化路线")
    return selected


def materialization_task_spec(
    materialization: DatasetMaterialization,
    version: DatasetVersion,
    route: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    artifact_id = str(version.artifact_id or "")
    expected_hash = materialization.expected_content_hash
    strategy = str(route["strategy"])
    if strategy in {"managed_copy", "visible_copy"}:
        return "builtin.dataset-materializer", {
            "materialization_id": materialization.id,
            "dataset_version_id": version.id,
            "artifact_id": artifact_id,
            "expected_content_hash": expected_hash,
            "chosen_route": copy.deepcopy(route),
            "materialization_input": {
                "artifact_id": artifact_id,
                "artifact_hash": expected_hash,
                "preferred_location_id": route.get("source_location_id"),
            },
            "resource_request": {
                "gpus": 0,
                "worker_selector": {"worker_id": materialization.target_worker_id},
            },
            "command": {},
        }
    spec = version.materialization_spec or {}
    adapter_id = str(spec.get("replay_adapter_id") or route.get("adapter_id") or "")
    replay = copy.deepcopy(spec.get("replay_payload") or {})
    if not adapter_id or not replay:
        raise MaterializationRouteError("该版本没有可重放的物化规格")
    request = dict(replay.get("resource_request") or {})
    selector = dict(request.get("worker_selector") or {})
    selector.pop("worker_ids", None)
    selector.pop("worker_name", None)
    selector["worker_id"] = materialization.target_worker_id
    request["worker_selector"] = selector
    replay.update(
        {
            "materialization_id": materialization.id,
            "dataset_version_id": version.id,
            "expected_content_hash": expected_hash,
            "chosen_route": copy.deepcopy(route),
            "resource_request": request,
        }
    )
    if strategy == "download" and version.source_snapshot.get("files"):
        replay["expected_snapshot_files"] = copy.deepcopy(version.source_snapshot["files"])
    return adapter_id, replay


def finalize_dataset_materialization(
    session: Session,
    task: Task,
    attempt: TaskAttempt,
    result: dict[str, Any],
    artifact_by_ref: dict[str, str],
) -> None:
    materialization = session.scalar(
        select(DatasetMaterialization).where(DatasetMaterialization.task_id == task.id)
    )
    if materialization is None:
        raise ValueError("dataset materialization task has no operation record")
    version = session.get(DatasetVersion, materialization.dataset_version_id)
    expected_artifact = session.get(Artifact, version.artifact_id) if version else None
    output_id = artifact_by_ref.get("dataset")
    output = session.get(Artifact, output_id) if output_id else None
    if version is None or expected_artifact is None:
        raise ValueError("dataset materialization references a missing dataset version")
    if output is None or output.kind != "dataset_snapshot":
        raise ValueError("dataset materialization did not produce a dataset snapshot")
    if (
        output.id != expected_artifact.id
        or output.content_hash != materialization.expected_content_hash
    ):
        raise ValueError("materialized dataset content does not match the immutable version")

    materialized_uri = str(result.get("materialized_uri") or "").strip()
    location = (
        session.scalar(
            select(ArtifactLocation)
            .where(
                ArtifactLocation.artifact_id == output.id,
                ArtifactLocation.project_id == materialization.project_id,
                ArtifactLocation.worker_id == materialization.target_worker_id,
                ArtifactLocation.availability_state == "available",
                ArtifactLocation.uri == materialized_uri,
            )
            .limit(1)
        )
        if materialized_uri
        else session.scalar(
            select(ArtifactLocation)
            .where(
                ArtifactLocation.artifact_id == output.id,
                ArtifactLocation.project_id == materialization.project_id,
                ArtifactLocation.worker_id == materialization.target_worker_id,
                ArtifactLocation.availability_state == "available",
            )
            .order_by(ArtifactLocation.updated_at.desc())
            .limit(1)
        )
    )
    if location is None and materialized_uri:
        registered_artifact, location = artifact_storage.register_external(
            session,
            project_id=materialization.project_id,
            kind="dataset_snapshot",
            uri=materialized_uri,
            content_hash=output.content_hash,
            size_bytes=output.size_bytes,
            media_type=output.media_type,
            availability_state="available",
            worker_id=materialization.target_worker_id,
            created_by_task_attempt_id=attempt.id,
            metadata={
                "materialization_id": materialization.id,
                "dataset_version_id": version.id,
                "route": materialization.chosen_route,
            },
        )
        if registered_artifact.id != expected_artifact.id:
            raise ValueError("materialized location resolved to a different artifact")
    if location is None:
        raise ValueError("dataset materialization produced no target Worker location")
    if location.worker_id != materialization.target_worker_id:
        raise ValueError("dataset materialization location belongs to another Worker")
    location.last_verified_at = utcnow()
    location.availability_state = "available"
    location.metadata_json = {
        **(location.metadata_json or {}),
        "materialization_id": materialization.id,
        "dataset_version_id": version.id,
        "verified": True,
    }
    materialization.result_location_id = location.id
    materialization.status = "succeeded"
    materialization.error = {}
    if not version.primary_location_id:
        version.primary_location_id = location.id
    version.materialization_state = "ready"


def materialization_view(materialization: DatasetMaterialization) -> dict[str, Any]:
    return materialization.to_dict()
