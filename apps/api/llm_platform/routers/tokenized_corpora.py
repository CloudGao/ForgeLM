from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.mixtures import is_identity_selection
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataMixtureVersion,
    Dataset,
    DatasetVersion,
    TokenizedCorpusVersion,
    TokenizerVersion,
)
from ..schemas import TokenizedCorpusCreate
from ..services.artifacts import artifact_storage
from ..services.dataset_analyses import (
    effective_record_manifest,
    effective_supports_capability,
    latest_successful_analysis,
)
from ..services.global_data import project_has_asset_access
from ..services.tasks import create_task
from ..services.training_selection import freeze_training_selection
from ..services.transactions import commit_or_flush
from .common import add_audit, get_or_404
from .tokenizers import _route_data_request
from .training import _freeze_tokenizer_binding

router = APIRouter(tags=["tokenized-corpora"])
TOKENIZED_CORPUS_FORMAT = "forgelm-token-corpus-v2"


def _identity(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _view(session: Session, version: TokenizedCorpusVersion) -> dict:
    dataset_version = session.get(DatasetVersion, version.dataset_version_id)
    dataset = (
        session.get(Dataset, dataset_version.dataset_id) if dataset_version is not None else None
    )
    tokenizer = session.get(TokenizerVersion, version.tokenizer_version_id)
    artifact = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    return {
        **version.to_dict(),
        "dataset_name": dataset.name if dataset is not None else None,
        "dataset_version": dataset_version.version if dataset_version is not None else None,
        "dataset_token_estimate": (
            dataset_version.token_estimate if dataset_version is not None else None
        ),
        "tokenizer_name": tokenizer.name if tokenizer is not None else None,
        "tokenizer_version": tokenizer.version if tokenizer is not None else None,
        "artifact_content_hash": artifact.content_hash if artifact is not None else None,
    }


@router.get("/projects/{project_id}/tokenized-corpora")
def list_tokenized_corpora(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    versions = session.scalars(
        select(TokenizedCorpusVersion)
        .where(TokenizedCorpusVersion.project_id == project_id)
        .order_by(TokenizedCorpusVersion.created_at.desc())
    ).all()
    return [_view(session, version) for version in versions]


@router.get("/tokenized-corpora/{version_id}")
def get_tokenized_corpus(version_id: str, session: Session = Depends(get_session)) -> dict:
    version = get_or_404(session, TokenizedCorpusVersion, version_id, "Tokenized Corpus")
    return _view(session, version)


@router.post("/projects/{project_id}/tokenized-corpora", status_code=201)
def create_tokenized_corpus(
    project_id: str,
    payload: TokenizedCorpusCreate,
    session: Session = Depends(get_session),
) -> dict:
    return _create_tokenized_corpus(
        project_id,
        payload,
        session,
        trigger_context={
            "source": "direct_api",
            "action": "创建编码数据 API",
        },
    )


def _create_tokenized_corpus(
    project_id: str,
    payload: TokenizedCorpusCreate,
    session: Session,
    *,
    trigger_context: dict,
) -> dict:
    dataset_version = session.get(DatasetVersion, payload.dataset_version_id)
    dataset = (
        session.get(Dataset, dataset_version.dataset_id) if dataset_version is not None else None
    )
    dataset_artifact = (
        session.get(Artifact, dataset_version.artifact_id)
        if dataset_version is not None and dataset_version.artifact_id
        else None
    )
    record_manifest = (
        effective_record_manifest(session, dataset_version) if dataset_version is not None else None
    )
    if dataset is None or not project_has_asset_access(
        session, project_id, "dataset_version", payload.dataset_version_id
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_dataset_version", "message": "数据版本不存在或不属于当前项目"},
        )
    if not (
        dataset_version is not None
        and dataset_version.state == "ready"
        and effective_supports_capability(session, dataset_version, "text_pretraining")
        and dataset_artifact is not None
        and dataset_artifact.kind == "dataset_snapshot"
        and record_manifest is not None
        and record_manifest.kind == "dataset_record_manifest"
        and artifact_storage.managed_payload_path(session, record_manifest, project_id) is not None
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_version_not_ready",
                "message": "数据版本的不可变快照或记录清单尚不可用",
            },
        )
    tokenizer = session.get(TokenizerVersion, payload.tokenizer_version_id)
    tokenizer_binding = _freeze_tokenizer_binding(session, project_id, payload.tokenizer_version_id)
    if tokenizer is None or tokenizer_binding is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "tokenizer_artifact_unavailable",
                "message": "Tokenizer 未锁定、制品不可用或不属于当前项目",
            },
        )

    selection = payload.selection.model_dump(mode="json")
    if selection["mode"] != "mixture":
        selection.pop("mixture_version_id", None)
    selection_input = None
    if selection["mode"] == "mixture":
        mixture = session.get(DataMixtureVersion, selection["mixture_version_id"])
        matches = (
            [
                index
                for index, entry in enumerate(mixture.entries or [])
                if entry.get("dataset_version_id") == dataset_version.id
            ]
            if mixture
            else []
        )
        if len(matches) != 1:
            raise HTTPException(422, detail={"code": "invalid_corpus_selection_source"})
        frozen_selection = freeze_training_selection(session, project_id, mixture, matches[0])
        if is_identity_selection(mixture.entries[matches[0]].get("selection")):
            selection = {"mode": "all", "estimated_token_limit": None}
        else:
            selection_input = frozen_selection
        if (
            frozen_selection["record_manifest_hash"] != record_manifest.content_hash
            or frozen_selection["dataset_artifact_hash"] != dataset_artifact.content_hash
        ):
            raise HTTPException(409, detail={"code": "corpus_selection_source_changed"})
    encoding = payload.encoding.model_dump(mode="json")
    analysis = latest_successful_analysis(session, dataset_version.id)
    data_input = {
        "format": "llm-dataset-input-v1",
        "dataset_version_id": dataset_version.id,
        "dataset_artifact_id": dataset_artifact.id,
        "dataset_artifact_hash": dataset_artifact.content_hash,
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
            analysis.document_count if analysis is not None else dataset_version.document_count or 0
        ),
        "token_estimate": int(
            analysis.token_estimate if analysis is not None else dataset_version.token_estimate or 0
        ),
    }
    identity_snapshot = {
        "format": TOKENIZED_CORPUS_FORMAT,
        "dataset_version_id": dataset_version.id,
        "dataset_artifact_hash": dataset_artifact.content_hash,
        "record_manifest_hash": record_manifest.content_hash,
        "tokenizer_version_id": tokenizer.id,
        "tokenizer_content_hash": tokenizer_binding["content_hash"],
        "selection": selection,
        "encoding": encoding,
        **({"selection_input": selection_input} if selection_input else {}),
    }
    if selection_input:
        data_input.update(
            document_count=selection_input["selected_document_count"],
            token_estimate=selection_input["selected_token_estimate"],
        )
    identity_hash = _identity(identity_snapshot)
    if selection_input:
        # Full-source encoding is independent of a mixture's later selections.
        # Compare its complete identity, including the frozen record manifest.
        for candidate in session.scalars(
            select(TokenizedCorpusVersion).where(
                TokenizedCorpusVersion.project_id == project_id,
                TokenizedCorpusVersion.dataset_version_id == dataset_version.id,
                TokenizedCorpusVersion.tokenizer_version_id == tokenizer.id,
                TokenizedCorpusVersion.status == "ready",
            )
        ).all():
            if (candidate.selection_config or {}).get("mode") != "all":
                continue
            candidate_identity = {**identity_snapshot, "selection": candidate.selection_config}
            candidate_identity.pop("selection_input", None)
            candidate_artifact = (
                session.get(Artifact, candidate.artifact_id) if candidate.artifact_id else None
            )
            ambiguous_legacy = (
                any(w.get("code") == "empty_documents_skipped" for w in candidate.warnings or [])
                and not (candidate_artifact.metadata_json or {}).get("document_identity_format")
                if candidate_artifact
                else True
            )
            if (
                candidate.identity_hash == _identity(candidate_identity)
                and candidate_artifact
                and candidate_artifact.kind == "tokenized_corpus"
                and candidate_artifact.content_hash == candidate.manifest_hash
                and not ambiguous_legacy
                and artifact_storage.inference_access(session, candidate_artifact, project_id)[
                    "type"
                ]
                != "unavailable"
            ):
                return _view(session, candidate)
    version = session.scalar(
        select(TokenizedCorpusVersion).where(
            TokenizedCorpusVersion.project_id == project_id,
            TokenizedCorpusVersion.identity_hash == identity_hash,
        )
    )
    if version is not None and version.status in {"queued", "assigned", "running", "ready"}:
        return _view(session, version)
    if version is None:
        version = TokenizedCorpusVersion(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            tokenizer_version_id=tokenizer.id,
            identity_hash=identity_hash,
            selection_config=selection,
            encoding_config=encoding,
            status="queued",
        )
        session.add(version)
        session.flush()
    else:
        version.status = "queued"
        version.failure_reason = None
        version.warnings = []

    resource_request = _route_data_request(
        session,
        project_id,
        {**payload.resource_request, "gpus": 0},
        [dataset_artifact.id, tokenizer_binding["artifact_id"]],
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="data_tokenize_corpus",
        title=f"Tokenized Corpus：{dataset.name} V{dataset_version.version} · {tokenizer.name} V{tokenizer.version}",
        adapter_id="builtin.tokenize-corpus",
        launcher_id=payload.launcher_id,
        payload={
            "tokenized_corpus_version_id": version.id,
            "trigger_context": trigger_context,
            "identity_hash": identity_hash,
            "identity_snapshot": identity_snapshot,
            "format": TOKENIZED_CORPUS_FORMAT,
            "selection": selection,
            "encoding": encoding,
            "data_input": data_input,
            **({"training_selection_inputs": [selection_input]} if selection_input else {}),
            "tokenizer": tokenizer_binding,
            "resource_request": resource_request,
            "command": {},
        },
    )
    version.task_id = task.id
    add_audit(
        session,
        project_id=project_id,
        action="tokenized_corpus.create",
        subject_type="tokenized_corpus_version",
        subject_id=version.id,
        payload={
            "identity_hash": identity_hash,
            "configuration": identity_snapshot,
            "trigger_context": trigger_context,
        },
    )
    commit_or_flush(session)
    return _view(session, version)
