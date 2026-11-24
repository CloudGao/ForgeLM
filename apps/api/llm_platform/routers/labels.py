from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    Dataset,
    DatasetLabelOperation,
    DatasetLabelSnapshot,
    DatasetVersion,
    LabelDefinition,
    LabelGroup,
    LabelScheme,
    LabelSchemeVersion,
    Task,
)
from ..schemas import (
    DatasetLabelOperationCreate,
    LabelSchemeCreate,
    LabelSchemeUpdate,
    LabelSchemeVersionCreate,
)
from ..services.artifacts import artifact_storage
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(tags=["data-labeling"])


def latest_label_snapshot(session: Session, dataset_version_id: str) -> DatasetLabelSnapshot | None:
    return session.scalar(
        select(DatasetLabelSnapshot)
        .where(DatasetLabelSnapshot.dataset_version_id == dataset_version_id)
        .order_by(DatasetLabelSnapshot.version.desc())
        .limit(1)
    )


def _version_view(session: Session, version: LabelSchemeVersion) -> dict:
    groups = session.scalars(
        select(LabelGroup)
        .where(LabelGroup.scheme_version_id == version.id)
        .order_by(LabelGroup.sort_order, LabelGroup.created_at)
    ).all()
    return {
        **version.to_dict(),
        "groups": [
            {
                **group.to_dict(),
                "labels": [
                    label.to_dict()
                    for label in session.scalars(
                        select(LabelDefinition)
                        .where(LabelDefinition.group_id == group.id)
                        .order_by(LabelDefinition.sort_order, LabelDefinition.created_at)
                    ).all()
                ],
            }
            for group in groups
        ],
    }


def _scheme_view(session: Session, scheme: LabelScheme, *, include_versions: bool = False) -> dict:
    versions = session.scalars(
        select(LabelSchemeVersion)
        .where(LabelSchemeVersion.scheme_id == scheme.id)
        .order_by(LabelSchemeVersion.version.desc())
    ).all()
    current = next((item for item in versions if item.id == scheme.current_version_id), None)
    result = {
        **scheme.to_dict(),
        "current_version": _version_view(session, current) if current else None,
        "version_count": len(versions),
    }
    if include_versions:
        result["versions"] = [_version_view(session, item) for item in versions]
    return result


def _create_version_rows(
    session: Session,
    scheme: LabelScheme,
    payload: LabelSchemeCreate | LabelSchemeVersionCreate,
    version_number: int,
) -> LabelSchemeVersion:
    version = LabelSchemeVersion(
        scheme_id=scheme.id,
        version=version_number,
        description=payload.description,
        created_by=payload.created_by,
        immutable=True,
    )
    session.add(version)
    session.flush()
    for group_index, group_payload in enumerate(payload.groups):
        group = LabelGroup(
            scheme_version_id=version.id,
            code=group_payload.code,
            name=group_payload.name,
            description=group_payload.description,
            selection_mode=group_payload.selection_mode,
            required=group_payload.required,
            sort_order=group_index,
        )
        session.add(group)
        session.flush()
        for label_index, label_payload in enumerate(group_payload.labels):
            session.add(
                LabelDefinition(
                    group_id=group.id,
                    code=label_payload.code,
                    name=label_payload.name,
                    description=label_payload.description,
                    color=label_payload.color,
                    sort_order=label_index,
                )
            )
    session.flush()
    scheme.current_version_id = version.id
    return version


@router.get("/catalog/label-schemes")
def list_label_schemes(
    modality: str = Query(default="", max_length=24),
    include_archived: bool = False,
    session: Session = Depends(get_session),
) -> list[dict]:
    query = select(LabelScheme)
    if modality:
        query = query.where(LabelScheme.modality == modality)
    if not include_archived:
        query = query.where(LabelScheme.status == "active")
    schemes = session.scalars(query.order_by(LabelScheme.modality, LabelScheme.name)).all()
    return [_scheme_view(session, scheme) for scheme in schemes]


@router.get("/catalog/label-schemes/{scheme_id}")
def get_label_scheme(scheme_id: str, session: Session = Depends(get_session)) -> dict:
    return _scheme_view(
        session, get_or_404(session, LabelScheme, scheme_id, "标签体系"), include_versions=True
    )


@router.post("/catalog/label-schemes", status_code=201)
def create_label_scheme(
    payload: LabelSchemeCreate, session: Session = Depends(get_session)
) -> dict:
    scheme = LabelScheme(
        code=payload.code,
        name=payload.name,
        modality=payload.modality,
        description=payload.description,
        status="active",
    )
    session.add(scheme)
    try:
        session.flush()
        _create_version_rows(session, scheme, payload, 1)
        add_audit(
            session,
            project_id=None,
            action="label_scheme.create",
            subject_type="label_scheme",
            subject_id=scheme.id,
            payload={"modality": scheme.modality, "version": 1},
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "label_scheme_conflict", "message": "该模态下的标签体系编码已存在"},
        ) from exc
    return _scheme_view(session, scheme, include_versions=True)


@router.patch("/catalog/label-schemes/{scheme_id}")
def update_label_scheme(
    scheme_id: str,
    payload: LabelSchemeUpdate,
    session: Session = Depends(get_session),
) -> dict:
    scheme = get_or_404(session, LabelScheme, scheme_id, "标签体系")
    changes = payload.model_dump(exclude_unset=True)
    for name, value in changes.items():
        setattr(scheme, name, value)
    add_audit(
        session,
        project_id=None,
        action="label_scheme.update",
        subject_type="label_scheme",
        subject_id=scheme.id,
        payload=changes,
    )
    session.commit()
    return _scheme_view(session, scheme, include_versions=True)


@router.post("/catalog/label-schemes/{scheme_id}/versions", status_code=201)
def create_label_scheme_version(
    scheme_id: str,
    payload: LabelSchemeVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    scheme = get_or_404(session, LabelScheme, scheme_id, "标签体系")
    if scheme.status != "active":
        raise HTTPException(
            status_code=409,
            detail={"code": "label_scheme_archived", "message": "归档标签体系不能创建新版本"},
        )
    next_version = int(
        session.scalar(
            select(func.coalesce(func.max(LabelSchemeVersion.version), 0)).where(
                LabelSchemeVersion.scheme_id == scheme.id
            )
        )
        or 0
    ) + 1
    version = _create_version_rows(session, scheme, payload, next_version)
    add_audit(
        session,
        project_id=None,
        action="label_scheme.version.create",
        subject_type="label_scheme_version",
        subject_id=version.id,
        payload={"scheme_id": scheme.id, "version": next_version},
    )
    session.commit()
    return _scheme_view(session, scheme, include_versions=True)


@router.get("/catalog/dataset-versions/{version_id}/labeling")
def get_dataset_labeling(version_id: str, session: Session = Depends(get_session)) -> dict:
    get_or_404(session, DatasetVersion, version_id, "数据版本")
    snapshots = session.scalars(
        select(DatasetLabelSnapshot)
        .where(DatasetLabelSnapshot.dataset_version_id == version_id)
        .order_by(DatasetLabelSnapshot.version.desc())
    ).all()
    operations = session.scalars(
        select(DatasetLabelOperation)
        .where(DatasetLabelOperation.dataset_version_id == version_id)
        .order_by(DatasetLabelOperation.created_at.desc())
        .limit(100)
    ).all()
    return {
        "current_snapshot": snapshots[0].to_dict() if snapshots else None,
        "snapshots": [item.to_dict() for item in snapshots],
        "operations": [item.to_dict() for item in operations],
    }


def _frozen_label_definitions(
    session: Session, scheme: LabelScheme, version: LabelSchemeVersion, label_ids: list[str]
) -> list[dict]:
    groups = session.scalars(
        select(LabelGroup).where(LabelGroup.scheme_version_id == version.id)
    ).all()
    group_by_id = {group.id: group for group in groups}
    labels = session.scalars(
        select(LabelDefinition).where(LabelDefinition.id.in_(label_ids))
    ).all()
    if len(labels) != len(label_ids) or any(label.group_id not in group_by_id for label in labels):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_label_selection", "message": "所选标签不属于指定的标签体系版本"},
        )
    by_group: dict[str, list[LabelDefinition]] = defaultdict(list)
    for label in labels:
        by_group[label.group_id].append(label)
    for group_id, selected in by_group.items():
        if group_by_id[group_id].selection_mode == "single" and len(selected) > 1:
            raise HTTPException(
                status_code=422,
                detail={"code": "single_group_conflict", "message": "单选标签组一次只能选择一个标签"},
            )
    label_by_id = {label.id: label for label in labels}
    return [
        {
            "scheme_id": scheme.id,
            "scheme_code": scheme.code,
            "scheme_version_id": version.id,
            "scheme_version": version.version,
            "group_id": group.id,
            "group_code": group.code,
            "selection_mode": group.selection_mode,
            "label_id": label.id,
            "label_code": label.code,
        }
        for label_id in label_ids
        for label in [label_by_id[label_id]]
        for group in [group_by_id[label.group_id]]
    ]


@router.post("/catalog/dataset-versions/{version_id}/label-operations", status_code=202)
def create_dataset_label_operation(
    version_id: str,
    payload: DatasetLabelOperationCreate,
    session: Session = Depends(get_session),
) -> dict:
    version = get_or_404(session, DatasetVersion, version_id, "数据版本")
    dataset = get_or_404(session, Dataset, version.dataset_id, "数据集")
    if version.state != "ready" or not version.artifact_id or not version.record_manifest_artifact_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "dataset_not_labelable", "message": "数据版本需就绪并具有稳定 Record Manifest"},
        )
    scheme_version = get_or_404(
        session, LabelSchemeVersion, payload.scheme_version_id, "标签体系版本"
    )
    scheme = get_or_404(session, LabelScheme, scheme_version.scheme_id, "标签体系")
    if scheme.status != "active":
        raise HTTPException(
            status_code=409,
            detail={"code": "label_scheme_archived", "message": "归档标签体系不能用于新打标"},
        )
    if scheme.modality != version.modality:
        raise HTTPException(
            status_code=422,
            detail={"code": "label_modality_mismatch", "message": "标签体系模态与数据版本不一致"},
        )
    frozen_labels = _frozen_label_definitions(
        session, scheme, scheme_version, payload.label_ids
    )
    source_task = get_or_404(session, Task, payload.source_query_task_id, "来源查询任务")
    source_payload = source_task.payload or {}
    source_result = source_task.result or {}
    if (
        source_task.kind != "data_query"
        or source_task.observed_state != "succeeded"
        or source_payload.get("dataset_version_id") != version.id
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "invalid_source_query", "message": "只能基于该数据版本已成功的 Data Studio 查询打标"},
        )
    if not source_result.get("labelable"):
        raise HTTPException(
            status_code=422,
            detail={"code": "query_not_labelable", "message": "查询结果没有保留完整的稳定行身份"},
        )
    current_snapshot = latest_label_snapshot(session, version.id)
    frozen_base_id = source_payload.get("label_snapshot_id") or None
    if frozen_base_id != (current_snapshot.id if current_snapshot else None):
        raise HTTPException(
            status_code=409,
            detail={"code": "stale_label_query", "message": "标签状态已变化，请重新查询后再打标"},
        )
    active_operation = session.scalar(
        select(DatasetLabelOperation.id).where(
            DatasetLabelOperation.dataset_version_id == version.id,
            DatasetLabelOperation.status.in_(
                ["queued", "assigned", "starting", "running", "pausing", "paused", "stopping"]
            ),
        )
    )
    if active_operation:
        raise HTTPException(
            status_code=409,
            detail={"code": "label_operation_active", "message": "该数据版本已有进行中的打标任务"},
        )
    page_identities = source_result.get("row_identities") or []
    if payload.scope in {"selected", "page"}:
        allowed = {
            (str(item.get("record_id") or ""), str(item.get("record_hash") or ""))
            for item in page_identities
            if isinstance(item, dict)
        }
        requested = {(item.record_id, item.record_hash) for item in payload.records}
        if not requested.issubset(allowed):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_record_selection", "message": "所选记录不属于来源查询的当前结果页"},
            )
    snapshot = get_or_404(session, Artifact, version.artifact_id, "数据快照")
    manifest = get_or_404(session, Artifact, version.record_manifest_artifact_id, "Record Manifest")
    access = artifact_storage.inference_access(session, snapshot, dataset.project_id)
    if access["type"] == "unavailable":
        raise HTTPException(status_code=409, detail={"code": "snapshot_unavailable", "message": "数据快照不可用"})
    selector: dict[str, object] = {"capabilities": ["builtin.dataset-labeler"]}
    if access["type"] == "external" and not access["globally_accessible"]:
        selector["worker_ids"] = list(access["worker_ids"])
    next_snapshot_version = (current_snapshot.version if current_snapshot else 0) + 1
    task = create_task(
        session,
        project_id=dataset.project_id,
        kind="data_label",
        title=f"打标数据：{dataset.name} V{version.version}",
        adapter_id="builtin.dataset-labeler",
        payload={
            "dataset_id": dataset.id,
            "dataset_version_id": version.id,
            "dataset_snapshot_input": {
                "artifact_id": snapshot.id,
                "artifact_hash": snapshot.content_hash,
                "logical_files": (source_payload.get("dataset_viewer_input") or {}).get("logical_files") or [],
            },
            "record_manifest_input": {
                "artifact_id": manifest.id,
                "artifact_hash": manifest.content_hash,
            },
            "label_snapshot_input": (
                {
                    "artifact_id": current_snapshot.artifact_id,
                    "artifact_hash": current_snapshot.content_hash,
                    "snapshot_id": current_snapshot.id,
                }
                if current_snapshot
                else None
            ),
            "base_label_snapshot_id": current_snapshot.id if current_snapshot else None,
            "output_snapshot_version": next_snapshot_version,
            "source_query": {
                "task_id": source_task.id,
                "sql": source_result.get("sql") or source_payload.get("sql"),
                "split": source_result.get("split") or source_payload.get("split") or "",
            },
            "scope": payload.scope,
            "action": payload.action,
            "records": [item.model_dump() for item in payload.records],
            "labels": frozen_labels,
            "annotator": payload.annotator,
            "note": payload.note,
            "resource_request": {
                "gpus": 0,
                "cpus": 1,
                "worker_selector": selector,
            },
            "command": {},
        },
    )
    operation = DatasetLabelOperation(
        dataset_version_id=version.id,
        scheme_version_id=scheme_version.id,
        task_id=task.id,
        base_snapshot_id=current_snapshot.id if current_snapshot else None,
        source_query_task_id=source_task.id,
        status="queued",
        scope=payload.scope,
        action=payload.action,
        label_ids=payload.label_ids,
        query_snapshot={
            "sql": source_result.get("sql"),
            "split": source_result.get("split"),
            "source_query_task_id": source_task.id,
            "base_label_snapshot_id": current_snapshot.id if current_snapshot else None,
        },
        annotator=payload.annotator,
        note=payload.note,
        selection_count=len(payload.records),
    )
    session.add(operation)
    session.flush()
    task.payload = {**(task.payload or {}), "operation_id": operation.id}
    add_audit(
        session,
        project_id=dataset.project_id,
        action="dataset.label.create",
        subject_type="dataset_label_operation",
        subject_id=operation.id,
        payload={"task_id": task.id, "scope": payload.scope, "action": payload.action},
    )
    session.commit()
    return {**operation.to_dict(), "task": task.to_dict()}
