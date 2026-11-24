from __future__ import annotations

import hashlib
import json
import math

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.mixtures import is_identity_selection
from llm_platform_core.training_data import (
    SEQUENCE_ORDER_STRATEGIES,
    materialized_sequence_size_bytes,
    sequence_order_strategy,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataMixture,
    DataMixtureVersion,
    TokenizedCorpusVersion,
    TokenizerVersion,
    TrainingDataViewVersion,
)
from ..schemas import TrainingDataViewCreate
from ..services.artifacts import artifact_storage
from ..services.global_data import project_has_asset_access
from ..services.tasks import create_task
from ..services.training_selection import freeze_training_selection
from ..services.transactions import commit_or_flush
from .common import add_audit, get_or_404
from .tokenizers import _route_data_request

router = APIRouter(tags=["training-data-views"])
TRAINING_DATA_VIEW_FORMAT = "forgelm-training-index-v1"


def _identity(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _configuration_snapshot(payload: TrainingDataViewCreate) -> dict:
    """Keep omitted V1 ordering fields absent so legacy API identities stay stable."""

    config = payload.config.model_dump(mode="json")
    for field in (
        "sequence_order_policy",
        "shuffle_block_sequences",
        "validation_holdout",
        "repeat_sequence_count",
    ):
        if field not in payload.config.model_fields_set:
            config.pop(field, None)
    return config


def _training_view_token_bytes(
    session: Session, tokenizer: TokenizerVersion, bindings: list[dict]
) -> int:
    sizes: list[int] = []
    for binding in bindings:
        corpus = session.get(TokenizedCorpusVersion, str(binding.get("version_id") or ""))
        configured = str((corpus.encoding_config if corpus else {}).get("token_dtype") or "auto")
        sizes.append(
            4
            if configured == "uint32"
            else 2
            if configured == "uint16"
            else (2 if tokenizer.vocab_size <= 65_535 else 4)
        )
    return max(sizes, default=2 if tokenizer.vocab_size <= 65_535 else 4)


def _view(session: Session, version: TrainingDataViewVersion) -> dict:
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


def _resolve_inputs(
    session: Session, project_id: str, payload: TrainingDataViewCreate
) -> tuple[DataMixture, DataMixtureVersion, TokenizerVersion, list[dict]]:
    mixture_version = session.get(DataMixtureVersion, payload.mixture_version_id)
    mixture = (
        session.get(DataMixture, mixture_version.mixture_id)
        if mixture_version is not None
        else None
    )
    if (
        mixture is None
        or not project_has_asset_access(
            session, project_id, "mixture_version", payload.mixture_version_id
        )
        or mixture_version is None
        or mixture_version.status != "ready"
        or not mixture_version.locked
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "mixture_not_ready", "message": "数据混合版本尚未解析并锁定"},
        )
    tokenizer = session.get(TokenizerVersion, payload.tokenizer_version_id)
    if tokenizer is None or tokenizer.project_id != project_id or not tokenizer.locked:
        raise HTTPException(
            status_code=409,
            detail={"code": "tokenizer_not_ready", "message": "Tokenizer 尚未锁定"},
        )
    corpus_versions = session.scalars(
        select(TokenizedCorpusVersion).where(
            TokenizedCorpusVersion.id.in_(payload.corpus_version_ids)
        )
    ).all()
    by_dataset = {version.dataset_version_id: version for version in corpus_versions}
    if len(corpus_versions) != len(set(payload.corpus_version_ids)):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_corpus_bindings", "message": "Corpus 列表包含无效或重复版本"},
        )
    bindings: list[dict] = []
    missing: list[str] = []
    for source_index, entry in enumerate(mixture_version.entries or []):
        dataset_version_id = str(entry.get("dataset_version_id") or "")
        selection_input = (
            freeze_training_selection(session, project_id, mixture_version, source_index)
            if not is_identity_selection(entry.get("selection"))
            else None
        )
        corpus = by_dataset.get(dataset_version_id)
        artifact = (
            session.get(Artifact, corpus.artifact_id)
            if corpus is not None and corpus.artifact_id
            else None
        )
        if (
            corpus is None
            or corpus.project_id != project_id
            or corpus.tokenizer_version_id != tokenizer.id
            or corpus.status != "ready"
            or corpus.format != "forgelm-token-corpus-v2"
            or artifact is None
            or artifact.kind != "tokenized_corpus"
            or artifact.content_hash != corpus.manifest_hash
            or artifact_storage.inference_access(session, artifact, project_id)["type"]
            == "unavailable"
        ):
            missing.append(dataset_version_id)
            continue
        bindings.append(
            {
                "source_index": source_index,
                "dataset_version_id": dataset_version_id,
                "weight_units": int(entry.get("weight_units") or 0),
                "normalized_weight": float(entry.get("normalized_weight") or 0),
                "selection": entry.get("selection") or {},
                "selected_document_count": int(entry.get("selected_document_count") or 0),
                "selected_token_estimate": int(entry.get("selected_tokens") or 0),
                "version_id": corpus.id,
                "identity_hash": corpus.identity_hash,
                "format": corpus.format,
                "document_count": corpus.document_count,
                "token_count": corpus.token_count,
                "artifact_id": artifact.id,
                "content_hash": artifact.content_hash,
                **(
                    {
                        "selection_input": selection_input,
                        "selected_capacity_known": (corpus.selection_config or {}).get(
                            "mixture_version_id"
                        )
                        == mixture_version.id,
                    }
                    if selection_input
                    else {}
                ),
            }
        )
    if missing or not bindings:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "tokenized_corpora_unavailable",
                "message": "混合中的每个数据集都必须绑定同一 Tokenizer 的就绪 Corpus",
                "missing_dataset_version_ids": missing,
            },
        )
    if any(item.get("selection_input") for item in bindings) and payload.config.validation_holdout:
        from .training import _validation_holdout_configuration

        request = payload.config.validation_holdout.model_dump(mode="json")
        validation = session.get(DataMixtureVersion, request["validation_mixture_version_id"])
        if (
            validation is None
            or not project_has_asset_access(session, project_id, "mixture_version", validation.id)
            or validation.status != "ready"
            or not validation.locked
        ):
            raise HTTPException(409, detail={"code": "validation_mixture_not_ready"})
        expected = _validation_holdout_configuration(mixture_version, validation)
        if request != expected:
            raise HTTPException(
                422,
                detail={
                    "code": "validation_holdout_selection_mismatch",
                    "message": "验证留出必须与训练筛选结果及冻结验证配置一致",
                },
            )
    return mixture, mixture_version, tokenizer, bindings


def _preflight_payload(
    payload: TrainingDataViewCreate, bindings: list[dict], *, token_bytes: int
) -> dict:
    config = _configuration_snapshot(payload)
    selection_pending = any(
        item.get("selection_input") and not item.get("selected_capacity_known") for item in bindings
    )
    strategy = sequence_order_strategy(str(config.get("sequence_order_policy") or "sequential_v1"))
    sequence_length = int(config["sequence_length"])
    raw_available_tokens = sum(int(item["token_count"]) for item in bindings)
    holdout = config.get("validation_holdout") or {}
    holdout_sequence_count = (
        math.ceil(int(holdout.get("target_tokens") or 0) / sequence_length) if holdout else 0
    )
    holdout_reserved_tokens = holdout_sequence_count * sequence_length
    available_tokens = max(raw_available_tokens - holdout_reserved_tokens, 0)
    unique_sequences = available_tokens // sequence_length
    if config["tail_policy"] == "pad" and available_tokens % sequence_length:
        unique_sequences += 1
    requested_sequences = math.ceil(int(config["requested_tokens"]) / sequence_length)
    sufficient = (
        config["consumption_policy"] == "repeat_epochs" and available_tokens > 0
    ) or requested_sequences <= unique_sequences
    required_epochs = (
        math.ceil(requested_sequences / unique_sequences)
        if config["consumption_policy"] == "repeat_epochs" and unique_sequences
        else 1
    )
    actual_sequence_estimate = (
        requested_sequences
        if config["consumption_policy"] == "repeat_epochs"
        else min(requested_sequences, unique_sequences)
    )
    materialized_bytes = (
        materialized_sequence_size_bytes(
            sequence_count=actual_sequence_estimate,
            sequence_length=sequence_length,
            token_bytes=token_bytes,
        )
        if strategy["materializes_sequences"]
        else 0
    )
    holdout_materialized_bytes = materialized_sequence_size_bytes(
        sequence_count=holdout_sequence_count,
        sequence_length=sequence_length,
        token_bytes=token_bytes,
    )
    total_materialized_bytes = materialized_bytes + holdout_materialized_bytes
    materialized_safety_bytes = (
        max(1_073_741_824, math.ceil(total_materialized_bytes * 0.05))
        if total_materialized_bytes
        else 0
    )
    storage_estimate = {
        "materialized_sequence_bytes": materialized_bytes,
        "worker_free_space_required_bytes": (total_materialized_bytes + materialized_safety_bytes),
        "token_bytes": token_bytes,
    }
    if holdout:
        storage_estimate["validation_holdout_bytes"] = holdout_materialized_bytes

    return {
        "ready": sufficient and not selection_pending,
        **(
            {"selection_capacity_pending": True, "capacity_is_upper_bound": True}
            if selection_pending
            else {}
        ),
        "requested_tokens": config["requested_tokens"],
        "requested_sequences": requested_sequences,
        "unique_token_capacity": available_tokens,
        "raw_unique_token_capacity": raw_available_tokens,
        "validation_holdout_reserved_tokens": holdout_reserved_tokens,
        "unique_sequence_capacity": unique_sequences,
        "required_epochs": required_epochs,
        "sequence_order_strategy": strategy,
        "storage_estimate": storage_estimate,
        "warnings": (
            [
                {
                    "code": "selection_capacity_pending",
                    "message": "筛选容量将在 Worker 按文档身份映射后计算；当前 Corpus 容量仅为上界。",
                }
            ]
            if selection_pending
            else []
            if sufficient
            else [
                {
                    "code": "unique_capacity_insufficient",
                    "message": "唯一消费策略的真实 Corpus 容量小于请求量；可降低请求量或显式选择循环 Epoch。",
                }
            ]
        ),
        "system_validations": [
            {
                "id": "corpus_identity_and_manifest_integrity",
                "status": "passed",
                "configurable": False,
                "category": "system_validation",
                "message": "Corpus 身份、格式、Manifest 哈希和可访问位置均有效。",
            },
            {
                "id": "training_capacity",
                "status": "pending" if selection_pending else "passed" if sufficient else "blocked",
                "configurable": False,
                "category": "system_validation",
                "message": "容量按真实 token 数和公开的数据消费/尾部策略计算。",
            },
        ],
        "configuration": config,
        "corpora": bindings,
    }


@router.get("/projects/{project_id}/training-data-views")
def list_training_data_views(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    versions = session.scalars(
        select(TrainingDataViewVersion)
        .where(TrainingDataViewVersion.project_id == project_id)
        .order_by(TrainingDataViewVersion.created_at.desc())
    ).all()
    return [_view(session, version) for version in versions]


@router.get("/training-data-view-strategies")
def list_training_data_view_strategies() -> list[dict]:
    return [dict(strategy) for strategy in SEQUENCE_ORDER_STRATEGIES.values()]


@router.get("/training-data-views/{version_id}")
def get_training_data_view(version_id: str, session: Session = Depends(get_session)) -> dict:
    version = get_or_404(session, TrainingDataViewVersion, version_id, "Training Data View")
    return _view(session, version)


@router.post("/projects/{project_id}/training-data-views/preflight")
def preflight_training_data_view(
    project_id: str,
    payload: TrainingDataViewCreate,
    session: Session = Depends(get_session),
) -> dict:
    _mixture, _mixture_version, tokenizer, bindings = _resolve_inputs(session, project_id, payload)
    return _preflight_payload(
        payload,
        bindings,
        token_bytes=_training_view_token_bytes(session, tokenizer, bindings),
    )


@router.post("/projects/{project_id}/training-data-views", status_code=201)
def create_training_data_view(
    project_id: str,
    payload: TrainingDataViewCreate,
    session: Session = Depends(get_session),
) -> dict:
    return _create_training_data_view(
        project_id,
        payload,
        session,
        trigger_context={
            "source": "direct_api",
            "action": "创建训练读取索引 API",
        },
    )


def _create_training_data_view(
    project_id: str,
    payload: TrainingDataViewCreate,
    session: Session,
    *,
    trigger_context: dict,
    prefix_training_data_view: dict | None = None,
) -> dict:
    mixture, mixture_version, tokenizer, bindings = _resolve_inputs(session, project_id, payload)
    preflight = _preflight_payload(
        payload,
        bindings,
        token_bytes=_training_view_token_bytes(session, tokenizer, bindings),
    )
    config = _configuration_snapshot(payload)
    prefix_lineage = None
    if prefix_training_data_view is not None:
        prefix_config = dict(prefix_training_data_view.get("configuration") or {})
        comparable_prefix = {
            key: value for key, value in prefix_config.items() if key != "requested_tokens"
        }
        comparable_config = {
            key: value for key, value in config.items() if key != "requested_tokens"
        }
        frozen_prefix_corpora = list(prefix_training_data_view.get("source_corpora") or [])
        current_corpora = [
            {key: item.get(key) for key in ("version_id", "identity_hash", "content_hash")}
            for item in bindings
        ]
        prefix_corpora = [
            {key: item.get(key) for key in ("version_id", "identity_hash", "content_hash")}
            for item in frozen_prefix_corpora
        ]
        if (
            prefix_training_data_view.get("mixture_version_id") != mixture_version.id
            or prefix_training_data_view.get("tokenizer_version_id") != tokenizer.id
            or comparable_prefix != comparable_config
            or prefix_corpora != current_corpora
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "training_data_view_prefix_incompatible",
                    "message": "Checkpoint 来源 View 与本次数据、Tokenizer 或读取策略不一致",
                },
            )
        base_sequence_count = int(prefix_training_data_view.get("sequence_count") or 0)
        if base_sequence_count <= 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "training_data_view_prefix_empty",
                    "message": "Checkpoint 来源 View 没有可继承的已发布序列",
                },
            )
        ancestor_views = list(
            (prefix_training_data_view.get("prefix_lineage") or {}).get("ancestor_views") or []
        )
        base_ancestor = {
            "version_id": str(prefix_training_data_view.get("version_id") or ""),
            "identity_hash": str(prefix_training_data_view.get("identity_hash") or ""),
            "content_hash": str(prefix_training_data_view.get("content_hash") or ""),
            "sequence_count": base_sequence_count,
        }
        ancestor_views = [
            {
                key: ancestor.get(key)
                for key in (
                    "version_id",
                    "identity_hash",
                    "content_hash",
                    "sequence_count",
                )
            }
            for ancestor in ancestor_views
            if ancestor.get("identity_hash") != base_ancestor["identity_hash"]
        ]
        ancestor_views.append(base_ancestor)
        prefix_lineage = {
            "contract": "forgelm.training-data-view-prefix.v1",
            "base_view": base_ancestor,
            "ancestor_views": ancestor_views,
        }
    identity_snapshot = {
        "format": TRAINING_DATA_VIEW_FORMAT,
        "mixture_version_id": mixture_version.id,
        "mixture_manifest_hash": mixture_version.manifest_hash,
        "tokenizer_version_id": tokenizer.id,
        "corpora": bindings,
        "configuration": config,
        **({"prefix_lineage": prefix_lineage} if prefix_lineage else {}),
    }
    identity_hash = _identity(identity_snapshot)
    version = session.scalar(
        select(TrainingDataViewVersion).where(
            TrainingDataViewVersion.project_id == project_id,
            TrainingDataViewVersion.identity_hash == identity_hash,
        )
    )
    if version is not None and version.status in {"queued", "assigned", "running", "ready"}:
        return _view(session, version)
    if version is None:
        version = TrainingDataViewVersion(
            project_id=project_id,
            mixture_version_id=mixture_version.id,
            tokenizer_version_id=tokenizer.id,
            identity_hash=identity_hash,
            sequence_length=config["sequence_length"],
            requested_tokens=config["requested_tokens"],
            status="queued",
            config_snapshot=identity_snapshot,
            warnings=preflight["warnings"],
        )
        session.add(version)
        session.flush()
    else:
        version.status = "queued"
        version.failure_reason = None
        version.warnings = preflight["warnings"]
    resource_request = _route_data_request(
        session,
        project_id,
        {**payload.resource_request, "gpus": 0},
        [str(item["artifact_id"]) for item in bindings]
        + (
            [str(prefix_training_data_view["artifact_id"])]
            if prefix_training_data_view is not None
            else []
        ),
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="training_data_index",
        title=f"Training Data View：{mixture.name} V{mixture_version.version}",
        adapter_id="builtin.build-training-index",
        launcher_id=payload.launcher_id,
        payload={
            "training_data_view_version_id": version.id,
            "trigger_context": trigger_context,
            "identity_hash": identity_hash,
            "identity_snapshot": identity_snapshot,
            "format": TRAINING_DATA_VIEW_FORMAT,
            "configuration": config,
            "tokenized_corpora": bindings,
            **(
                {
                    "training_selection_inputs": [
                        item["selection_input"] for item in bindings if item.get("selection_input")
                    ]
                }
                if any(item.get("selection_input") for item in bindings)
                else {}
            ),
            **(
                {"prefix_training_data_view": prefix_training_data_view}
                if prefix_training_data_view is not None
                else {}
            ),
            "preflight": preflight,
            "resource_request": resource_request,
            "command": {},
        },
    )
    version.task_id = task.id
    add_audit(
        session,
        project_id=project_id,
        action="training_data_view.create",
        subject_type="training_data_view_version",
        subject_id=version.id,
        payload={
            "identity_hash": identity_hash,
            "configuration": identity_snapshot,
            "trigger_context": trigger_context,
        },
    )
    commit_or_flush(session)
    return _view(session, version)
