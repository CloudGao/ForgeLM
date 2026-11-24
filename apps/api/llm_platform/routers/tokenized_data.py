from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataMixture,
    DataMixtureVersion,
    TokenizedDatasetVersion,
    TokenizerVersion,
)
from ..schemas import TokenizedDatasetCreate
from ..services.global_data import project_has_asset_access
from ..services.tasks import create_task
from .common import add_audit, get_or_404
from .training import (
    _freeze_data_inputs,
    _freeze_tokenizer_binding,
    _route_training_resource_request,
    _training_source_artifact_ids,
)

router = APIRouter(tags=["tokenized-data"])
TOKENIZED_DATA_FORMAT = "forgelm-packed-tokens-v1"


def _identity(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _view(session: Session, version: TokenizedDatasetVersion) -> dict:
    mixture_version = session.get(DataMixtureVersion, version.mixture_version_id)
    mixture = (
        session.get(DataMixture, mixture_version.mixture_id)
        if mixture_version is not None
        else None
    )
    tokenizer = session.get(TokenizerVersion, version.tokenizer_version_id)
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    return {
        **version.to_dict(),
        "mixture_name": mixture.name if mixture is not None else None,
        "mixture_version": mixture_version.version if mixture_version is not None else None,
        "tokenizer_name": tokenizer.name if tokenizer is not None else None,
        "tokenizer_version": tokenizer.version if tokenizer is not None else None,
        "artifact_content_hash": artifact.content_hash if artifact is not None else None,
    }


@router.get("/projects/{project_id}/tokenized-datasets")
def list_tokenized_datasets(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    versions = session.scalars(
        select(TokenizedDatasetVersion)
        .where(TokenizedDatasetVersion.project_id == project_id)
        .order_by(TokenizedDatasetVersion.created_at.desc())
    ).all()
    return [_view(session, version) for version in versions]


@router.get("/tokenized-datasets/{version_id}")
def get_tokenized_dataset(
    version_id: str, session: Session = Depends(get_session)
) -> dict:
    version = get_or_404(
        session, TokenizedDatasetVersion, version_id, "Tokenized Dataset"
    )
    return _view(session, version)


@router.post("/projects/{project_id}/tokenized-datasets", status_code=201)
def create_tokenized_dataset(
    project_id: str,
    payload: TokenizedDatasetCreate,
    session: Session = Depends(get_session),
) -> dict:
    del project_id, payload, session
    raise HTTPException(
        status_code=410,
        detail={
            "code": "legacy_tokenized_dataset_read_only",
            "message": (
                "forgelm-packed-tokens-v1 已转为历史只读格式；"
                "请创建 Tokenized Corpus，再创建 Training Data View"
            ),
        },
    )


def _create_legacy_tokenized_dataset(
    project_id: str,
    payload: TokenizedDatasetCreate,
    session: Session,
) -> dict:
    """Retained for explicit migration tooling; not exposed as a creation route."""
    mixture_version = session.get(DataMixtureVersion, payload.mixture_version_id)
    mixture = (
        session.get(DataMixture, mixture_version.mixture_id)
        if mixture_version is not None
        else None
    )
    if mixture is None or not project_has_asset_access(
        session, project_id, "mixture_version", payload.mixture_version_id
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_data_mixture", "message": "数据混合不存在或不属于当前项目"},
        )
    data_inputs = _freeze_data_inputs(
        session, project_id, [{"data_ref_id": payload.mixture_version_id}]
    )
    if len(data_inputs) != 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "data_mixture_not_ready",
                "message": "数据混合尚未解析、锁定或没有可用 Manifest",
            },
        )
    tokenizer = session.get(TokenizerVersion, payload.tokenizer_version_id)
    tokenizer_binding = _freeze_tokenizer_binding(
        session, project_id, payload.tokenizer_version_id
    )
    if tokenizer is None or tokenizer.project_id != project_id or tokenizer_binding is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "tokenizer_artifact_unavailable",
                "message": "Tokenizer 未锁定、制品不可用或不属于当前项目",
            },
        )
    data_input = data_inputs[0]
    identity_snapshot = {
        "format": TOKENIZED_DATA_FORMAT,
        "packing_version": payload.packing_version,
        "mixture_version_id": mixture_version.id,
        "mixture_manifest_hash": data_input["manifest_hash"],
        "resolver_version": data_input["resolver_version"],
        "sampler_version": data_input["sampler_version"],
        "seed": data_input["seed"],
        "tokenizer_version_id": tokenizer.id,
        "tokenizer_content_hash": tokenizer_binding["content_hash"],
        "vocab_size": tokenizer.vocab_size,
        "sequence_length": payload.sequence_length,
        "target_tokens": payload.target_tokens,
        "eos_policy": "append_eos_per_document",
        "document_boundary": "isolated",
    }
    identity_hash = _identity(identity_snapshot)
    version = session.scalar(
        select(TokenizedDatasetVersion).where(
            TokenizedDatasetVersion.project_id == project_id,
            TokenizedDatasetVersion.identity_hash == identity_hash,
        )
    )
    if version is not None and version.status in {"queued", "assigned", "running", "ready"}:
        return _view(session, version)
    if version is None:
        version = TokenizedDatasetVersion(
            project_id=project_id,
            mixture_version_id=mixture_version.id,
            tokenizer_version_id=tokenizer.id,
            identity_hash=identity_hash,
            format=TOKENIZED_DATA_FORMAT,
            packing_version=payload.packing_version,
            sequence_length=payload.sequence_length,
            target_tokens=payload.target_tokens,
            status="queued",
            config_snapshot=identity_snapshot,
        )
        session.add(version)
        session.flush()
    else:
        version.status = "queued"
        version.failure_reason = None

    artifact_ids = [
        *_training_source_artifact_ids(data_inputs, None),
        str(tokenizer_binding["artifact_id"]),
    ]
    resource_request = _route_training_resource_request(
        session,
        project_id,
        {**payload.resource_request, "gpus": 0},
        artifact_ids,
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="data_tokenize",
        title=(
            f"预分词：{mixture.name} V{mixture_version.version} · "
            f"{tokenizer.name} V{tokenizer.version}"
        ),
        adapter_id="builtin.pretokenize",
        launcher_id=payload.launcher_id,
        payload={
            "tokenized_dataset_version_id": version.id,
            "identity_hash": identity_hash,
            "format": TOKENIZED_DATA_FORMAT,
            "packing_version": payload.packing_version,
            "sequence_length": payload.sequence_length,
            "target_tokens": payload.target_tokens,
            "shard_sequences": payload.shard_sequences,
            "data_input": data_input,
            "tokenizer": tokenizer_binding,
            "resource_request": resource_request,
            "command": {},
        },
    )
    version.task_id = task.id
    add_audit(
        session,
        project_id=project_id,
        action="tokenized_dataset.create",
        subject_type="tokenized_dataset_version",
        subject_id=version.id,
        payload={"identity_hash": identity_hash, "target_tokens": payload.target_tokens},
    )
    session.commit()
    return _view(session, version)
