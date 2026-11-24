from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ChatTemplate,
    ChatTemplateVersion,
    Dataset,
    DatasetVersion,
    PlanRevision,
    ProjectDataBinding,
    SupervisionCorpusVersion,
    SupervisionDataset,
    SupervisionDatasetVersion,
    SupervisionMixture,
    SupervisionMixtureVersion,
    SupervisionTrainingViewVersion,
    Task,
    TokenizerVersion,
    TrainingPlan,
    TrainingRun,
)
from ..schemas import (
    ChatTemplateCreate,
    ChatTemplateValidate,
    ChatTemplateVersionCreate,
    SupervisionCorpusCreate,
    SupervisionDatasetCreate,
    SupervisionDatasetVersionCreate,
    SupervisionMixtureCreate,
    SupervisionMixtureUpdate,
    SupervisionMixtureVersionCreate,
    SupervisionTrainingViewCreate,
)
from ..services.dataset_analyses import (
    effective_record_manifest,
    latest_successful_analysis,
)
from ..services.global_data import project_can_manage_asset, project_has_asset_access
from ..services.tasks import create_task
from ..services.transactions import commit_or_flush
from .common import add_audit, get_or_404
from .tokenizers import _route_data_request
from .training import _freeze_tokenizer_binding

router = APIRouter(tags=["post-training"])

SUPERVISION_RECORD_FORMAT = "forgelm-supervision-record-index-v1"
SFT_CORPUS_FORMAT = "forgelm-sft-corpus-v1"
PREFERENCE_CORPUS_FORMAT = "forgelm-preference-corpus-v1"
SFT_VIEW_FORMAT = "forgelm-sft-training-index-v1"
PREFERENCE_VIEW_FORMAT = "forgelm-preference-training-index-v1"


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _contains_reference(value: object, references: set[str]) -> bool:
    if isinstance(value, str):
        return value in references
    if isinstance(value, dict):
        return any(_contains_reference(item, references) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference(item, references) for item in value)
    return False


def _project_for_supervision_version(
    session: Session, version: SupervisionDatasetVersion
) -> tuple[SupervisionDataset, DatasetVersion, Dataset]:
    logical = session.get(SupervisionDataset, version.supervision_dataset_id)
    dataset_version = session.get(DatasetVersion, version.dataset_version_id)
    dataset = session.get(Dataset, dataset_version.dataset_id) if dataset_version else None
    if logical is None or dataset_version is None or dataset is None:
        raise HTTPException(status_code=404, detail={"code": "supervision_source_missing"})
    return logical, dataset_version, dataset


def _template_version_view(session: Session, version: ChatTemplateVersion) -> dict:
    template = session.get(ChatTemplate, version.template_id)
    return {
        **version.to_dict(),
        "template_name": template.name if template else None,
        "project_id": template.project_id if template else None,
    }


def _template_view(session: Session, template: ChatTemplate) -> dict:
    versions = session.scalars(
        select(ChatTemplateVersion)
        .where(ChatTemplateVersion.template_id == template.id)
        .order_by(ChatTemplateVersion.version.desc())
    ).all()
    return {
        **template.to_dict(),
        "versions": [_template_version_view(session, version) for version in versions],
    }


def _template_report(payload: ChatTemplateVersionCreate | ChatTemplateCreate) -> dict:
    body = payload.template_body
    generation_open = bool(re.search(r"{%[-+]?\s*generation\s*[-+]?%}", body))
    generation_close = bool(re.search(r"{%[-+]?\s*endgeneration\s*[-+]?%}", body))
    issues: list[dict[str, str]] = []
    if payload.assistant_mask_mode == "generation_spans_v1" and not (
        generation_open and generation_close
    ):
        issues.append(
            {
                "code": "generation_region_not_declared",
                "message": "模板未同时包含 generation/endgeneration；原生 assistant-only 构建会返回直接错误。",
            }
        )
    return {
        "renderer_version": "hf_chat_template_v1",
        "assistant_mask_capable": bool(generation_open and generation_close),
        "issues": issues,
    }


def _create_template_version(
    session: Session,
    template: ChatTemplate,
    payload: ChatTemplateVersionCreate | ChatTemplateCreate,
) -> ChatTemplateVersion:
    identity = {
        "template_body": payload.template_body,
        "role_schema": payload.role_schema,
        "required_special_tokens": payload.required_special_tokens,
        "assistant_mask_mode": payload.assistant_mask_mode,
        "renderer_version": "hf_chat_template_v1",
    }
    content_hash = _hash(identity)
    existing = session.scalar(
        select(ChatTemplateVersion).where(
            ChatTemplateVersion.template_id == template.id,
            ChatTemplateVersion.content_hash == content_hash,
        )
    )
    if existing is not None:
        template.current_version_id = existing.id
        return existing
    next_version = int(
        session.scalar(
            select(func.max(ChatTemplateVersion.version)).where(
                ChatTemplateVersion.template_id == template.id
            )
        )
        or 0
    ) + 1
    version = ChatTemplateVersion(
        template_id=template.id,
        version=next_version,
        template_body=payload.template_body,
        role_schema=payload.role_schema,
        required_special_tokens=payload.required_special_tokens,
        assistant_mask_mode=payload.assistant_mask_mode,
        renderer_version="hf_chat_template_v1",
        validation_report=_template_report(payload),
        content_hash=content_hash,
        immutable=True,
    )
    session.add(version)
    session.flush()
    template.current_version_id = version.id
    return version


@router.get("/projects/{project_id}/chat-templates")
def list_chat_templates(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    templates = session.scalars(
        select(ChatTemplate)
        .where(ChatTemplate.project_id == project_id)
        .order_by(ChatTemplate.created_at.desc())
    ).all()
    return [_template_view(session, template) for template in templates]


@router.post("/projects/{project_id}/chat-templates", status_code=201)
def create_chat_template(
    project_id: str, payload: ChatTemplateCreate, session: Session = Depends(get_session)
) -> dict:
    template = ChatTemplate(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description,
    )
    session.add(template)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_chat_template", "message": "项目内模板名称已存在"},
        ) from exc
    version = _create_template_version(session, template, payload)
    add_audit(
        session,
        project_id=project_id,
        action="chat_template.create",
        subject_type="chat_template",
        subject_id=template.id,
        payload={"version_id": version.id, "content_hash": version.content_hash},
    )
    commit_or_flush(session)
    return _template_view(session, template)


@router.post("/chat-templates/{template_id}/versions", status_code=201)
def create_chat_template_version(
    template_id: str,
    payload: ChatTemplateVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    template = get_or_404(session, ChatTemplate, template_id, "Chat Template")
    version = _create_template_version(session, template, payload)
    add_audit(
        session,
        project_id=template.project_id,
        action="chat_template.version.create",
        subject_type="chat_template_version",
        subject_id=version.id,
        payload={"content_hash": version.content_hash},
    )
    session.commit()
    return _template_version_view(session, version)


@router.post("/chat-template-versions/{version_id}/validate")
def validate_chat_template_version(
    version_id: str,
    payload: ChatTemplateValidate,
    session: Session = Depends(get_session),
) -> dict:
    version = get_or_404(session, ChatTemplateVersion, version_id, "Chat Template 版本")
    template = get_or_404(session, ChatTemplate, version.template_id, "Chat Template")
    available_special_tokens: set[str] = set()
    tokenizer = None
    if payload.tokenizer_version_id:
        tokenizer = session.get(TokenizerVersion, payload.tokenizer_version_id)
        if tokenizer is None or tokenizer.project_id != template.project_id:
            raise HTTPException(status_code=422, detail={"code": "invalid_tokenizer_version"})
        available_special_tokens = {
            str(value) for value in (tokenizer.config or {}).get("special_tokens") or []
        }
    missing = [
        token for token in version.required_special_tokens if token not in available_special_tokens
    ] if tokenizer else []
    report = {
        **(version.validation_report or {}),
        "messages": payload.messages,
        "tokenizer_version_id": tokenizer.id if tokenizer else None,
        "missing_special_tokens": missing,
        "compatible": not missing,
        "note": "完整 token 与 assistant mask 预览由监督 Corpus 构建器使用冻结 Tokenizer 生成。",
    }
    return report


def _supervision_version_view(session: Session, version: SupervisionDatasetVersion) -> dict:
    logical, dataset_version, dataset = _project_for_supervision_version(session, version)
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    return {
        **version.to_dict(),
        "name": logical.name,
        "description": logical.description,
        "project_id": logical.project_id,
        "dataset_name": dataset.name,
        "dataset_version": dataset_version.version,
        "artifact_content_hash": artifact.content_hash if artifact else None,
    }


def _dataset_input(
    session: Session,
    dataset: Dataset,
    version: DatasetVersion,
    snapshot: Artifact,
    record_manifest: Artifact,
) -> dict:
    analysis = latest_successful_analysis(session, version.id)
    return {
        "format": "llm-dataset-input-v1",
        "dataset_version_id": version.id,
        "dataset_artifact_id": snapshot.id,
        "dataset_artifact_hash": snapshot.content_hash,
        "record_manifest_artifact_id": record_manifest.id,
        "record_manifest_hash": record_manifest.content_hash,
        **(
            {
                "analysis_revision_id": analysis.id,
                "analysis_revision": analysis.revision,
            }
            if analysis is not None
            else {}
        ),
        "document_count": int(
            analysis.document_count if analysis is not None else version.document_count or 0
        ),
        "token_estimate": int(
            analysis.token_estimate if analysis is not None else version.token_estimate or 0
        ),
        "source_config": dataset.source_config or {},
    }


def _queue_supervision_scan(
    session: Session,
    logical: SupervisionDataset,
    version: SupervisionDatasetVersion,
    dataset: Dataset,
    dataset_version: DatasetVersion,
    *,
    execution_project_id: str | None = None,
    launcher_id: str,
    resource_request: dict[str, Any],
) -> None:
    snapshot = session.get(Artifact, dataset_version.artifact_id) if dataset_version.artifact_id else None
    record_manifest = effective_record_manifest(session, dataset_version)
    if snapshot is None or record_manifest is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "dataset_snapshot_unavailable", "message": "数据快照或记录索引不可用"},
        )
    data_input = _dataset_input(session, dataset, dataset_version, snapshot, record_manifest)
    data_inputs = [data_input]
    lookup_artifact_ids: list[str] = []
    for lookup in version.mapping_config.get("_lookups") or []:
        lookup_version = session.get(DatasetVersion, str(lookup.get("dataset_version_id") or ""))
        lookup_dataset = session.get(Dataset, lookup_version.dataset_id) if lookup_version else None
        lookup_snapshot = session.get(Artifact, str(lookup.get("dataset_artifact_id") or ""))
        lookup_manifest = session.get(Artifact, str(lookup.get("record_manifest_artifact_id") or ""))
        if (
            lookup_dataset is None
            or lookup_snapshot is None
            or lookup_manifest is None
            or lookup_snapshot.content_hash != lookup.get("dataset_artifact_hash")
            or lookup_manifest.content_hash != lookup.get("record_manifest_hash")
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "supervision_lookup_unavailable", "name": lookup.get("name")},
            )
        data_inputs.append(
            _dataset_input(
                session, lookup_dataset, lookup_version, lookup_snapshot, lookup_manifest
            )
        )
        lookup_artifact_ids.extend([lookup_snapshot.id, lookup_manifest.id])
    actor_project_id = execution_project_id or logical.project_id
    routed = _route_data_request(
        session,
        actor_project_id,
        {**resource_request, "gpus": 0},
        [snapshot.id, record_manifest.id, *lookup_artifact_ids],
    )
    task = create_task(
        session,
        project_id=actor_project_id,
        kind="supervision_scan",
        title=f"扫描后训练数据：{logical.name} V{version.version}",
        adapter_id="builtin.supervision-scan",
        launcher_id=launcher_id,
        payload={
            "supervision_dataset_version_id": version.id,
            "identity_hash": version.identity_hash,
            "format": SUPERVISION_RECORD_FORMAT,
            "kind": version.kind,
            "mapping": version.mapping_config,
            "filters": version.filter_config,
            "data_inputs": data_inputs,
            "resource_request": routed,
            "command": {},
        },
    )
    version.task_id = task.id
    version.status = "queued"


def _create_supervision_dataset_version(
    session: Session,
    logical: SupervisionDataset,
    payload: SupervisionDatasetVersionCreate | SupervisionDatasetCreate,
    *,
    access_project_id: str,
    execution_project_id: str,
) -> SupervisionDatasetVersion:
    dataset_version = session.get(DatasetVersion, payload.dataset_version_id)
    dataset = session.get(Dataset, dataset_version.dataset_id) if dataset_version else None
    if dataset is None or not project_can_manage_asset(
        session, access_project_id, "dataset_version", payload.dataset_version_id
    ):
        raise HTTPException(status_code=422, detail={"code": "invalid_dataset_version"})
    snapshot = session.get(Artifact, dataset_version.artifact_id) if dataset_version.artifact_id else None
    record_manifest = effective_record_manifest(session, dataset_version)
    if snapshot is None or record_manifest is None:
        raise HTTPException(status_code=409, detail={"code": "dataset_snapshot_unavailable"})
    lookup_configs: list[dict[str, Any]] = []
    lookup_names: set[str] = set()
    for requested in payload.lookups:
        lookup = requested.model_dump()
        if lookup["name"] in lookup_names:
            raise HTTPException(status_code=422, detail={"code": "duplicate_supervision_lookup_name"})
        lookup_names.add(lookup["name"])
        if any(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", str(alias)) is None
            or not str(path).strip()
            for alias, path in lookup["fields"].items()
        ):
            raise HTTPException(status_code=422, detail={"code": "invalid_supervision_lookup_fields"})
        lookup_version = session.get(DatasetVersion, lookup["dataset_version_id"])
        lookup_dataset = session.get(Dataset, lookup_version.dataset_id) if lookup_version else None
        if lookup_dataset is None or not project_has_asset_access(
            session, access_project_id, "dataset_version", lookup["dataset_version_id"]
        ):
            raise HTTPException(status_code=422, detail={"code": "invalid_supervision_lookup_dataset"})
        lookup_snapshot = session.get(Artifact, lookup_version.artifact_id) if lookup_version.artifact_id else None
        lookup_manifest = effective_record_manifest(session, lookup_version)
        if lookup_snapshot is None or lookup_manifest is None:
            raise HTTPException(status_code=409, detail={"code": "supervision_lookup_snapshot_unavailable"})
        lookup_configs.append(
            {
                **lookup,
                "dataset_artifact_id": lookup_snapshot.id,
                "dataset_artifact_hash": lookup_snapshot.content_hash,
                "record_manifest_artifact_id": lookup_manifest.id,
                "record_manifest_hash": lookup_manifest.content_hash,
            }
        )
    mapping_config = {
        **{key: value for key, value in payload.mapping.items() if key != "_lookups"},
        **({"_lookups": lookup_configs} if lookup_configs else {}),
    }
    identity_snapshot = {
        "format": SUPERVISION_RECORD_FORMAT,
        "dataset_version_id": dataset_version.id,
        "dataset_artifact_hash": snapshot.content_hash,
        "record_manifest_hash": record_manifest.content_hash,
        "kind": payload.kind,
        "mapping": mapping_config,
        "filters": payload.filters,
        "validator_version": "supervision_schema_v1",
    }
    identity_hash = _hash(identity_snapshot)
    existing = session.scalar(
        select(SupervisionDatasetVersion).where(
            SupervisionDatasetVersion.supervision_dataset_id == logical.id,
            SupervisionDatasetVersion.identity_hash == identity_hash,
        )
    )
    if existing is not None:
        logical.current_version_id = existing.id
        return existing
    next_version = int(
        session.scalar(
            select(func.max(SupervisionDatasetVersion.version)).where(
                SupervisionDatasetVersion.supervision_dataset_id == logical.id
            )
        )
        or 0
    ) + 1
    version = SupervisionDatasetVersion(
        supervision_dataset_id=logical.id,
        version=next_version,
        dataset_version_id=dataset_version.id,
        kind=payload.kind,
        mapping_config=mapping_config,
        filter_config=payload.filters,
        identity_hash=identity_hash,
        status="draft",
    )
    session.add(version)
    session.flush()
    logical.current_version_id = version.id
    if payload.start_scan:
        _queue_supervision_scan(
            session,
            logical,
            version,
            dataset,
            dataset_version,
            execution_project_id=execution_project_id,
            launcher_id=payload.launcher_id,
            resource_request=payload.resource_request,
        )
    return version


@router.get("/projects/{project_id}/supervision-datasets")
def list_supervision_datasets(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    bound_version_ids = set(
        session.scalars(
            select(ProjectDataBinding.asset_version_id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == "supervision_dataset_version",
            )
        ).all()
    )
    bound_logical_ids = set(
        session.scalars(
            select(SupervisionDatasetVersion.supervision_dataset_id).where(
                SupervisionDatasetVersion.id.in_(bound_version_ids)
            )
        ).all()
    ) if bound_version_ids else set()
    logicals = session.scalars(
        select(SupervisionDataset)
        .where(
            (SupervisionDataset.project_id == project_id)
            | (SupervisionDataset.id.in_(bound_logical_ids))
        )
        .order_by(SupervisionDataset.created_at.desc())
    ).all()
    result: list[dict] = []
    for logical in logicals:
        versions = session.scalars(
            select(SupervisionDatasetVersion)
            .where(SupervisionDatasetVersion.supervision_dataset_id == logical.id)
            .order_by(SupervisionDatasetVersion.version.desc())
        ).all()
        if logical.project_id != project_id:
            versions = [version for version in versions if version.id in bound_version_ids]
        result.append(
            {
                **logical.to_dict(),
                "binding_scope": "owned" if logical.project_id == project_id else "bound",
                "versions": [_supervision_version_view(session, v) for v in versions],
            }
        )
    return result


@router.post("/projects/{project_id}/supervision-datasets", status_code=201)
def create_supervision_dataset(
    project_id: str,
    payload: SupervisionDatasetCreate,
    session: Session = Depends(get_session),
) -> dict:
    logical = SupervisionDataset(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description,
    )
    session.add(logical)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail={"code": "duplicate_supervision_dataset"}) from exc
    version = _create_supervision_dataset_version(
        session,
        logical,
        payload,
        access_project_id=project_id,
        execution_project_id=project_id,
    )
    add_audit(
        session,
        project_id=project_id,
        action="supervision_dataset.create",
        subject_type="supervision_dataset_version",
        subject_id=version.id,
        payload={"identity_hash": version.identity_hash, "kind": version.kind},
    )
    commit_or_flush(session)
    return _supervision_version_view(session, version)


@router.post("/supervision-datasets/{dataset_id}/versions", status_code=201)
def create_supervision_dataset_version(
    dataset_id: str,
    payload: SupervisionDatasetVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    logical = get_or_404(session, SupervisionDataset, dataset_id, "后训练数据")
    version = _create_supervision_dataset_version(
        session,
        logical,
        payload,
        access_project_id=logical.project_id,
        execution_project_id=logical.project_id,
    )
    session.commit()
    return _supervision_version_view(session, version)


@router.post("/supervision-dataset-versions/{version_id}/scan", status_code=202)
def scan_supervision_dataset_version(
    version_id: str, session: Session = Depends(get_session)
) -> dict:
    version = get_or_404(session, SupervisionDatasetVersion, version_id, "后训练数据版本")
    logical, dataset_version, dataset = _project_for_supervision_version(session, version)
    _queue_supervision_scan(
        session,
        logical,
        version,
        dataset,
        dataset_version,
        launcher_id="builtin.local-process",
        resource_request={"gpus": 0},
    )
    session.commit()
    return _supervision_version_view(session, version)


@router.get("/supervision-dataset-versions/{version_id}/validation-report")
def supervision_validation_report(
    version_id: str, session: Session = Depends(get_session)
) -> dict:
    version = get_or_404(session, SupervisionDatasetVersion, version_id, "后训练数据版本")
    return {
        "version_id": version.id,
        "status": version.status,
        "accepted_record_count": version.accepted_record_count,
        "rejected_record_count": version.rejected_record_count,
        "report": version.validation_report,
        "sample_preview": version.sample_preview,
        "warnings": version.warnings,
    }


def _mixture_version_view(session: Session, version: SupervisionMixtureVersion) -> dict:
    mixture = session.get(SupervisionMixture, version.mixture_id)
    entries: list[dict[str, Any]] = []
    for entry in version.entries or []:
        source_version = session.get(
            SupervisionDatasetVersion, entry.get("supervision_dataset_version_id")
        )
        logical = (
            session.get(SupervisionDataset, source_version.supervision_dataset_id)
            if source_version
            else None
        )
        dataset_version = (
            session.get(DatasetVersion, source_version.dataset_version_id)
            if source_version
            else None
        )
        dataset = (
            session.get(Dataset, dataset_version.dataset_id) if dataset_version else None
        )
        entries.append(
            {
                **entry,
                "supervision_name": logical.name if logical else None,
                "supervision_version": source_version.version if source_version else None,
                "dataset_name": dataset.name if dataset else None,
                "dataset_version": dataset_version.version if dataset_version else None,
            }
        )
    return {
        **version.to_dict(),
        "entries": entries,
        "name": mixture.name if mixture else None,
        "description": mixture.description if mixture else None,
        "project_id": mixture.project_id if mixture else None,
    }


def _mixture_view(session: Session, mixture: SupervisionMixture) -> dict:
    versions = session.scalars(
        select(SupervisionMixtureVersion)
        .where(SupervisionMixtureVersion.mixture_id == mixture.id)
        .order_by(SupervisionMixtureVersion.version.desc())
    ).all()
    return {
        **mixture.to_dict(),
        "versions": [_mixture_version_view(session, version) for version in versions],
    }


def _mixture_delete_blockers(
    session: Session,
    mixture: SupervisionMixture,
    versions: list[SupervisionMixtureVersion],
) -> list[str]:
    version_ids = {version.id for version in versions}
    if not version_ids:
        return []
    blockers: list[str] = []
    binding = session.scalar(
        select(ProjectDataBinding.id).where(
            ProjectDataBinding.asset_type == "supervision_mixture_version",
            ProjectDataBinding.asset_version_id.in_(version_ids),
        )
    )
    if binding:
        blockers.append("已有项目绑定该监督混合版本")
    corpus = session.scalar(
        select(SupervisionCorpusVersion.id).where(
            SupervisionCorpusVersion.mixture_version_id.in_(version_ids)
        )
    )
    if corpus:
        blockers.append("已有监督 Corpus 引用该混合版本")
    plan_revisions = session.scalars(
        select(PlanRevision)
        .join(TrainingPlan, TrainingPlan.id == PlanRevision.plan_id)
    ).all()
    if any(
        _contains_reference(revision.stages, version_ids)
        or _contains_reference(revision.effective_config, version_ids)
        for revision in plan_revisions
    ):
        blockers.append("已有训练计划引用该混合版本")
    tasks = session.scalars(select(Task)).all()
    if any(
        _contains_reference(task.payload, version_ids)
        or _contains_reference(task.result, version_ids)
        for task in tasks
    ):
        blockers.append("已有任务记录引用该混合版本")
    runs = session.scalars(select(TrainingRun)).all()
    if any(_contains_reference(run.config_snapshot, version_ids) for run in runs):
        blockers.append("已有训练运行引用该混合版本")
    return blockers


def _create_mixture_version(
    session: Session,
    mixture: SupervisionMixture,
    payload: SupervisionMixtureVersionCreate | SupervisionMixtureCreate,
    *,
    access_project_id: str,
) -> SupervisionMixtureVersion:
    entries = [entry.model_dump(mode="json") for entry in payload.entries]
    versions = session.scalars(
        select(SupervisionDatasetVersion).where(
            SupervisionDatasetVersion.id.in_(
                [entry["supervision_dataset_version_id"] for entry in entries]
            )
        )
    ).all()
    if len(versions) != len({entry["supervision_dataset_version_id"] for entry in entries}):
        raise HTTPException(status_code=422, detail={"code": "invalid_supervision_entries"})
    all_accessible = True
    kinds = {version.kind for version in versions}
    by_id = {version.id: version for version in versions}
    normalized: list[dict[str, Any]] = []
    total_weight = sum(int(entry["weight"]) for entry in entries)
    for entry in entries:
        version = by_id[entry["supervision_dataset_version_id"]]
        logical = session.get(SupervisionDataset, version.supervision_dataset_id)
        if logical is None or not project_can_manage_asset(
            session,
            access_project_id,
            "supervision_dataset_version",
            version.id,
        ):
            all_accessible = False
        normalized.append(
            {
                **entry,
                "identity_hash": version.identity_hash,
                "accepted_record_count": version.accepted_record_count,
                "normalized_weight": int(entry["weight"]) / total_weight,
            }
        )
    if not all_accessible or len(kinds) != 1:
        raise HTTPException(
            status_code=422,
            detail={"code": "incompatible_supervision_entries", "message": "混合来源必须属于同一项目和同一语义类型"},
        )
    identity_snapshot = {
        "kind": next(iter(kinds)),
        "entries": normalized,
        "seed": payload.seed,
        "sampling_policy": "weighted_deficit_v1",
    }
    identity_hash = _hash(identity_snapshot)
    existing = session.scalar(
        select(SupervisionMixtureVersion).where(
            SupervisionMixtureVersion.mixture_id == mixture.id,
            SupervisionMixtureVersion.identity_hash == identity_hash,
        )
    )
    if existing is not None:
        mixture.current_version_id = existing.id
        return existing
    next_version = int(
        session.scalar(
            select(func.max(SupervisionMixtureVersion.version)).where(
                SupervisionMixtureVersion.mixture_id == mixture.id
            )
        )
        or 0
    ) + 1
    version = SupervisionMixtureVersion(
        mixture_id=mixture.id,
        version=next_version,
        kind=next(iter(kinds)),
        entries=normalized,
        seed=payload.seed,
        identity_hash=identity_hash,
        status="ready",
        locked=True,
    )
    session.add(version)
    session.flush()
    mixture.current_version_id = version.id
    return version


@router.get("/projects/{project_id}/supervision-mixtures")
def list_supervision_mixtures(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    bound_version_ids = set(
        session.scalars(
            select(ProjectDataBinding.asset_version_id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == "supervision_mixture_version",
            )
        ).all()
    )
    bound_mixture_ids = set(
        session.scalars(
            select(SupervisionMixtureVersion.mixture_id).where(
                SupervisionMixtureVersion.id.in_(bound_version_ids)
            )
        ).all()
    ) if bound_version_ids else set()
    mixtures = session.scalars(
        select(SupervisionMixture)
        .where(
            (SupervisionMixture.project_id == project_id)
            | (SupervisionMixture.id.in_(bound_mixture_ids))
        )
        .order_by(SupervisionMixture.created_at.desc())
    ).all()
    result: list[dict] = []
    for mixture in mixtures:
        view = _mixture_view(session, mixture)
        view["binding_scope"] = "owned" if mixture.project_id == project_id else "bound"
        if mixture.project_id != project_id:
            view["versions"] = [
                version
                for version in view["versions"]
                if version["id"] in bound_version_ids
            ]
        result.append(view)
    return result


@router.get("/supervision-mixtures/{mixture_id}")
def get_supervision_mixture(
    mixture_id: str, session: Session = Depends(get_session)
) -> dict:
    return _mixture_view(
        session, get_or_404(session, SupervisionMixture, mixture_id, "后训练数据混合")
    )


@router.post("/projects/{project_id}/supervision-mixtures", status_code=201)
def create_supervision_mixture(
    project_id: str,
    payload: SupervisionMixtureCreate,
    session: Session = Depends(get_session),
) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_supervision_mixture_name", "message": "混合名称不能为空"},
        )
    mixture = SupervisionMixture(
        project_id=project_id,
        name=name,
        description=payload.description,
    )
    session.add(mixture)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail={"code": "duplicate_supervision_mixture"}) from exc
    version = _create_mixture_version(
        session, mixture, payload, access_project_id=project_id
    )
    add_audit(
        session,
        project_id=project_id,
        action="supervision_mixture.create",
        subject_type="supervision_mixture",
        subject_id=mixture.id,
        payload={"name": mixture.name, "version_id": version.id},
    )
    commit_or_flush(session)
    return _mixture_version_view(session, version)


@router.post("/supervision-mixtures/{mixture_id}/versions", status_code=201)
def create_supervision_mixture_version(
    mixture_id: str,
    payload: SupervisionMixtureVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    mixture = get_or_404(session, SupervisionMixture, mixture_id, "后训练数据混合")
    version = _create_mixture_version(
        session, mixture, payload, access_project_id=mixture.project_id
    )
    add_audit(
        session,
        project_id=mixture.project_id,
        action="supervision_mixture.version.create",
        subject_type="supervision_mixture",
        subject_id=mixture.id,
        payload={"version_id": version.id, "version": version.version},
    )
    session.commit()
    return _mixture_version_view(session, version)


@router.patch("/supervision-mixtures/{mixture_id}")
def update_supervision_mixture(
    mixture_id: str,
    payload: SupervisionMixtureUpdate,
    session: Session = Depends(get_session),
) -> dict:
    mixture = get_or_404(session, SupervisionMixture, mixture_id, "后训练数据混合")
    changes = payload.model_dump(exclude_unset=True)
    if "name" in changes:
        raw_name = changes["name"]
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_supervision_mixture_name", "message": "混合名称不能为空"},
            )
        mixture.name = name
    if "description" in changes:
        mixture.description = changes["description"] or ""
    add_audit(
        session,
        project_id=mixture.project_id,
        action="supervision_mixture.update",
        subject_type="supervision_mixture",
        subject_id=mixture.id,
        payload={"fields": sorted(changes)},
    )
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_supervision_mixture", "message": "后训练数据混合名称已存在"},
        ) from exc
    return _mixture_view(session, mixture)


@router.delete("/supervision-mixtures/{mixture_id}", status_code=204)
def delete_supervision_mixture(
    mixture_id: str, session: Session = Depends(get_session)
) -> None:
    mixture = get_or_404(session, SupervisionMixture, mixture_id, "后训练数据混合")
    versions = session.scalars(
        select(SupervisionMixtureVersion).where(
            SupervisionMixtureVersion.mixture_id == mixture.id
        )
    ).all()
    blockers = _mixture_delete_blockers(session, mixture, list(versions))
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "supervision_mixture_in_use",
                "message": "该后训练数据混合正在被使用：" + "；".join(blockers),
                "blockers": blockers,
            },
        )
    add_audit(
        session,
        project_id=mixture.project_id,
        action="supervision_mixture.delete",
        subject_type="supervision_mixture",
        subject_id=mixture.id,
        payload={
            "name": mixture.name,
            "version_ids": [version.id for version in versions],
        },
    )
    session.delete(mixture)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "supervision_mixture_in_use",
                "message": "该后训练数据混合仍有下游引用，不能删除",
            },
        ) from exc


def _corpus_view(session: Session, version: SupervisionCorpusVersion) -> dict:
    mixture_version = session.get(SupervisionMixtureVersion, version.mixture_version_id)
    mixture = session.get(SupervisionMixture, mixture_version.mixture_id) if mixture_version else None
    template_version = session.get(ChatTemplateVersion, version.chat_template_version_id)
    template = session.get(ChatTemplate, template_version.template_id) if template_version else None
    tokenizer = session.get(TokenizerVersion, version.tokenizer_version_id)
    return {
        **version.to_dict(),
        "mixture_name": mixture.name if mixture else None,
        "mixture_version": mixture_version.version if mixture_version else None,
        "chat_template_name": template.name if template else None,
        "chat_template_version": template_version.version if template_version else None,
        "tokenizer_name": tokenizer.name if tokenizer else None,
        "tokenizer_version": tokenizer.version if tokenizer else None,
    }


@router.get("/projects/{project_id}/supervision-corpora")
def list_supervision_corpora(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    versions = session.scalars(
        select(SupervisionCorpusVersion)
        .where(SupervisionCorpusVersion.project_id == project_id)
        .order_by(SupervisionCorpusVersion.created_at.desc())
    ).all()
    return [_corpus_view(session, version) for version in versions]


@router.post("/projects/{project_id}/supervision-corpora", status_code=201)
def create_supervision_corpus(
    project_id: str,
    payload: SupervisionCorpusCreate,
    session: Session = Depends(get_session),
) -> dict:
    mixture_version = session.get(SupervisionMixtureVersion, payload.mixture_version_id)
    mixture = session.get(SupervisionMixture, mixture_version.mixture_id) if mixture_version else None
    template_version = session.get(ChatTemplateVersion, payload.chat_template_version_id)
    template = session.get(ChatTemplate, template_version.template_id) if template_version else None
    tokenizer_binding = _freeze_tokenizer_binding(session, project_id, payload.tokenizer_version_id)
    if mixture is None or not project_has_asset_access(
        session,
        project_id,
        "supervision_mixture_version",
        payload.mixture_version_id,
    ):
        raise HTTPException(status_code=422, detail={"code": "invalid_supervision_mixture"})
    if template is None or template.project_id != project_id:
        raise HTTPException(status_code=422, detail={"code": "invalid_chat_template"})
    if tokenizer_binding is None:
        raise HTTPException(status_code=409, detail={"code": "tokenizer_artifact_unavailable"})
    expected_kind = "conversation" if payload.objective_kind == "sft" else "preference"
    if mixture_version.kind != expected_kind:
        raise HTTPException(status_code=422, detail={"code": "objective_data_kind_mismatch"})

    sources: list[dict[str, Any]] = []
    data_inputs: list[dict[str, Any]] = []
    artifact_ids: list[str] = [tokenizer_binding["artifact_id"]]
    for source_index, entry in enumerate(mixture_version.entries or []):
        version = session.get(
            SupervisionDatasetVersion, str(entry.get("supervision_dataset_version_id") or "")
        )
        if version is None:
            raise HTTPException(status_code=422, detail={"code": "supervision_source_missing"})
        logical, dataset_version, dataset = _project_for_supervision_version(session, version)
        index_artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
        snapshot = session.get(Artifact, dataset_version.artifact_id) if dataset_version.artifact_id else None
        record_manifest = effective_record_manifest(session, dataset_version)
        if index_artifact is None or snapshot is None or record_manifest is None:
            raise HTTPException(
                status_code=409,
                detail={"code": "supervision_source_not_prepared", "version_id": version.id},
            )
        data_input = _dataset_input(
            session, dataset, dataset_version, snapshot, record_manifest
        )
        data_inputs.append(data_input)
        sources.append(
            {
                "source_index": source_index,
                "version_id": version.id,
                "identity_hash": version.identity_hash,
                "weight": int(entry.get("weight") or 1),
                "mapping": version.mapping_config,
                "kind": version.kind,
                "data_input_index": source_index,
                "record_index": {
                    "artifact_id": index_artifact.id,
                    "content_hash": index_artifact.content_hash,
                    "format": SUPERVISION_RECORD_FORMAT,
                    "identity_hash": version.identity_hash,
                },
            }
        )
        artifact_ids.extend([snapshot.id, record_manifest.id, index_artifact.id])
    config = {
        "max_length": payload.max_length,
        **({"deduplicate_questions": True} if payload.deduplicate_questions else {}),
        "overflow_policy": payload.overflow_policy,
        "min_completion_tokens": payload.min_completion_tokens,
        "token_dtype": payload.token_dtype,
    }
    format_name = SFT_CORPUS_FORMAT if payload.objective_kind == "sft" else PREFERENCE_CORPUS_FORMAT
    identity_snapshot = {
        "format": format_name,
        "mixture_version_id": mixture_version.id,
        "mixture_identity_hash": mixture_version.identity_hash,
        "tokenizer_version_id": payload.tokenizer_version_id,
        "tokenizer_content_hash": tokenizer_binding["content_hash"],
        "chat_template_version_id": template_version.id,
        "chat_template_content_hash": template_version.content_hash,
        "objective_kind": payload.objective_kind,
        "sources": sources,
        "configuration": config,
    }
    identity_hash = _hash(identity_snapshot)
    version = session.scalar(
        select(SupervisionCorpusVersion).where(
            SupervisionCorpusVersion.project_id == project_id,
            SupervisionCorpusVersion.identity_hash == identity_hash,
        )
    )
    if version is not None and version.status in {"queued", "assigned", "running", "ready"}:
        return _corpus_view(session, version)
    if version is None:
        version = SupervisionCorpusVersion(
            project_id=project_id,
            mixture_version_id=mixture_version.id,
            tokenizer_version_id=payload.tokenizer_version_id,
            chat_template_version_id=template_version.id,
            objective_kind=payload.objective_kind,
            identity_hash=identity_hash,
            format=format_name,
            status="queued",
            config_snapshot=identity_snapshot,
        )
        session.add(version)
        session.flush()
    else:
        version.status = "queued"
        version.failure_reason = None
    routed = _route_data_request(
        session,
        project_id,
        {**payload.resource_request, "gpus": 0},
        list(dict.fromkeys(artifact_ids)),
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="supervision_corpus",
        title=f"构建 {payload.objective_kind.upper()} Corpus：{mixture.name}",
        adapter_id="builtin.supervision-corpus",
        launcher_id=payload.launcher_id,
        payload={
            "supervision_corpus_version_id": version.id,
            "identity_hash": identity_hash,
            "identity_snapshot": identity_snapshot,
            "format": format_name,
            "objective_kind": payload.objective_kind,
            "configuration": config,
            "data_inputs": data_inputs,
            "supervision_sources": sources,
            "tokenizer": tokenizer_binding,
            "chat_template": _template_version_view(session, template_version),
            "resource_request": routed,
            "command": {},
        },
    )
    version.task_id = task.id
    commit_or_flush(session)
    return _corpus_view(session, version)


def _training_view(session: Session, version: SupervisionTrainingViewVersion) -> dict:
    corpus = session.get(SupervisionCorpusVersion, version.corpus_version_id)
    return {
        **version.to_dict(),
        "corpus_identity_hash": corpus.identity_hash if corpus else None,
        "corpus_format": corpus.format if corpus else None,
        "chat_template_version_id": corpus.chat_template_version_id if corpus else None,
        "tokenizer_version_id": corpus.tokenizer_version_id if corpus else None,
    }


@router.get("/projects/{project_id}/supervision-training-views")
def list_supervision_training_views(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    versions = session.scalars(
        select(SupervisionTrainingViewVersion)
        .where(SupervisionTrainingViewVersion.project_id == project_id)
        .order_by(SupervisionTrainingViewVersion.created_at.desc())
    ).all()
    return [_training_view(session, version) for version in versions]


@router.post("/projects/{project_id}/supervision-training-views", status_code=201)
def create_supervision_training_view(
    project_id: str,
    payload: SupervisionTrainingViewCreate,
    session: Session = Depends(get_session),
) -> dict:
    corpus = session.get(SupervisionCorpusVersion, payload.corpus_version_id)
    artifact = session.get(Artifact, corpus.artifact_id) if corpus and corpus.artifact_id else None
    if corpus is None or corpus.project_id != project_id or artifact is None:
        raise HTTPException(status_code=409, detail={"code": "supervision_corpus_unavailable"})
    config = payload.config.model_dump(mode="json")
    if not config.get("reshuffle_each_epoch"):
        config.pop("reshuffle_each_epoch", None)
    if corpus.objective_kind == "sft" and config["budget_unit"] not in {
        "supervised_tokens",
        "examples",
        "steps",
    }:
        raise HTTPException(status_code=422, detail={"code": "invalid_sft_budget_unit"})
    if corpus.objective_kind == "dpo" and config["budget_unit"] not in {
        "preference_pairs",
        "steps",
    }:
        raise HTTPException(status_code=422, detail={"code": "invalid_dpo_budget_unit"})
    format_name = SFT_VIEW_FORMAT if corpus.objective_kind == "sft" else PREFERENCE_VIEW_FORMAT
    identity_snapshot = {
        "format": format_name,
        "corpus_version_id": corpus.id,
        "corpus_identity_hash": corpus.identity_hash,
        "corpus_manifest_hash": artifact.content_hash,
        "objective_kind": corpus.objective_kind,
        "configuration": config,
    }
    identity_hash = _hash(identity_snapshot)
    version = session.scalar(
        select(SupervisionTrainingViewVersion).where(
            SupervisionTrainingViewVersion.project_id == project_id,
            SupervisionTrainingViewVersion.identity_hash == identity_hash,
        )
    )
    if version is not None and version.status in {"queued", "assigned", "running", "ready"}:
        return _training_view(session, version)
    if version is None:
        version = SupervisionTrainingViewVersion(
            project_id=project_id,
            corpus_version_id=corpus.id,
            objective_kind=corpus.objective_kind,
            identity_hash=identity_hash,
            format=format_name,
            budget_unit=config["budget_unit"],
            budget_value=config["budget_value"],
            status="queued",
            config_snapshot=identity_snapshot,
        )
        session.add(version)
        session.flush()
    else:
        version.status = "queued"
        version.failure_reason = None
    routed = _route_data_request(
        session,
        project_id,
        {**payload.resource_request, "gpus": 0},
        [artifact.id],
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="supervision_training_index",
        title=f"构建 {corpus.objective_kind.upper()} Training View",
        adapter_id="builtin.build-supervision-index",
        launcher_id=payload.launcher_id,
        payload={
            "supervision_training_view_version_id": version.id,
            "identity_hash": identity_hash,
            "identity_snapshot": identity_snapshot,
            "format": format_name,
            "objective_kind": corpus.objective_kind,
            "configuration": config,
            "supervision_corpus": {
                "version_id": corpus.id,
                "identity_hash": corpus.identity_hash,
                "format": corpus.format,
                "artifact_id": artifact.id,
                "content_hash": artifact.content_hash,
            },
            "resource_request": routed,
            "command": {},
        },
    )
    version.task_id = task.id
    commit_or_flush(session)
    return _training_view(session, version)
