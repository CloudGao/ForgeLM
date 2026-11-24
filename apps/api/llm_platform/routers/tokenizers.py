from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.mixtures import MIXTURE_MANIFEST_FORMAT
from llm_platform_core.tokenizers import (
    normalize_tokenizer_gate_policy,
    validate_builtin_tokenizer_sample_tokens,
    validate_disk_bpe_config,
)
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataMixture,
    DataMixtureVersion,
    Dataset,
    DatasetVersion,
    TokenizerVersion,
)
from ..schemas import TokenizerCreate
from ..services.artifacts import artifact_storage
from ..services.dataset_analyses import (
    effective_record_manifest,
    effective_supports_capability,
    latest_successful_analysis,
)
from ..services.global_data import project_has_asset_access
from ..services.tasks import create_task
from .common import add_audit, get_or_404, require_executable

router = APIRouter(tags=["tokenizers"])
BUILTIN_TOKENIZER_ADAPTERS = {"builtin.hf-tokenizer", "builtin.disk-bpe-tokenizer"}


def _route_data_request(
    session: Session, project_id: str, resource_request: dict, artifact_ids: list[str]
) -> dict:
    request = {**resource_request}
    eligible_worker_ids: set[str] | None = None
    for artifact_id in artifact_ids:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            raise HTTPException(
                status_code=409,
                detail={"code": "dataset_snapshot_missing", "message": "Tokenizer 输入快照不存在"},
            )
        access = artifact_storage.inference_access(session, artifact, project_id)
        if access["type"] == "unavailable":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_snapshot_unavailable",
                    "message": "Tokenizer 输入快照没有托管副本或可用的 Worker 位置",
                },
            )
        if access["type"] == "external" and not access["globally_accessible"]:
            source_workers = {str(value) for value in access["worker_ids"]}
            eligible_worker_ids = (
                source_workers
                if eligible_worker_ids is None
                else eligible_worker_ids & source_workers
            )
    if eligible_worker_ids is None:
        return request
    selector = dict(request.get("worker_selector") or {})
    selected_worker_id = selector.get("worker_id")
    if selected_worker_id:
        eligible_worker_ids &= {str(selected_worker_id)}
    selected_worker_ids = {str(value) for value in selector.get("worker_ids") or []}
    if selected_worker_ids:
        eligible_worker_ids &= selected_worker_ids
    if not eligible_worker_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_worker_location_mismatch",
                "message": "没有一个 Worker 能同时访问全部 Tokenizer 输入快照",
            },
        )
    if len(eligible_worker_ids) == 1:
        selector.pop("worker_ids", None)
        selector["worker_id"] = next(iter(eligible_worker_ids))
    else:
        selector.pop("worker_id", None)
        selector["worker_ids"] = sorted(eligible_worker_ids)
    request["worker_selector"] = selector
    return request


@router.get("/projects/{project_id}/tokenizers")
def list_tokenizers(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        item.to_dict()
        for item in session.scalars(
            select(TokenizerVersion)
            .where(TokenizerVersion.project_id == project_id)
            .order_by(TokenizerVersion.created_at.desc())
        ).all()
    ]


@router.post("/projects/{project_id}/tokenizers", status_code=201)
def create_tokenizer(
    project_id: str, payload: TokenizerCreate, session: Session = Depends(get_session)
) -> dict:
    try:
        config = {
            **payload.config,
            "gate_policy": normalize_tokenizer_gate_policy(payload.config.get("gate_policy")),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_tokenizer_gate_policy", "message": str(exc)},
        ) from exc
    adapter_id = payload.adapter_id or (
        "builtin.command.tokenizer" if payload.command.get("executable") else "builtin.hf-tokenizer"
    )
    if payload.start and adapter_id in BUILTIN_TOKENIZER_ADAPTERS:
        try:
            validate_builtin_tokenizer_sample_tokens(config.get("sample_tokens"))
            if adapter_id == "builtin.disk-bpe-tokenizer":
                if payload.algorithm.lower() != "bpe":
                    raise ValueError("Disk-backed tokenizer supports BPE only")
                validate_disk_bpe_config(config)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": (
                        "invalid_disk_bpe_config"
                        if adapter_id == "builtin.disk-bpe-tokenizer"
                        else "invalid_tokenizer_sample_tokens"
                    ),
                    "message": str(exc),
                },
            ) from exc
    data_input: dict | None = None
    dataset_artifact_ids: list[str] = []
    if payload.data_ref_type == "mixture_version":
        mixture_version = session.get(DataMixtureVersion, payload.data_ref_id)
        mixture = (
            session.get(DataMixture, mixture_version.mixture_id)
            if mixture_version is not None
            else None
        )
        manifest = (
            session.get(Artifact, mixture_version.manifest_artifact_id)
            if mixture_version is not None and mixture_version.manifest_artifact_id
            else None
        )
        if mixture is None or not project_has_asset_access(
            session, project_id, "mixture_version", payload.data_ref_id
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_data_mixture",
                    "message": "数据混合不存在或不属于当前项目",
                },
            )
        if (
            mixture_version is None
            or mixture_version.status != "ready"
            or not mixture_version.locked
            or manifest is None
            or manifest.kind != "data_mixture_manifest"
            or manifest.content_hash != mixture_version.manifest_hash
            or artifact_storage.managed_payload_path(session, manifest, project_id) is None
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "data_mixture_not_ready",
                    "message": "数据混合的稳定 Manifest 尚未就绪或不可用",
                },
            )
        data_input = {
            "format": str(
                (manifest.metadata_json or {}).get("format")
                or MIXTURE_MANIFEST_FORMAT
            ),
            "mixture_version_id": mixture_version.id,
            "manifest_artifact_id": manifest.id,
            "manifest_hash": manifest.content_hash,
            "resolver_version": mixture_version.resolver_version,
            "sampler_version": mixture_version.sampler_version,
            "seed": mixture_version.seed,
        }
        dataset_artifact_ids = [
            str(entry["dataset_artifact_id"])
            for entry in mixture_version.entries
            if entry.get("dataset_artifact_id")
        ]
    elif payload.data_ref_type == "dataset_version":
        dataset_version = session.get(DatasetVersion, payload.data_ref_id)
        dataset = (
            session.get(Dataset, dataset_version.dataset_id)
            if dataset_version is not None
            else None
        )
        dataset_artifact = (
            session.get(Artifact, dataset_version.artifact_id)
            if dataset_version is not None and dataset_version.artifact_id
            else None
        )
        record_manifest = (
            effective_record_manifest(session, dataset_version)
            if dataset_version is not None
            else None
        )
        if dataset is None or not project_has_asset_access(
            session, project_id, "dataset_version", payload.data_ref_id
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_dataset_version",
                    "message": "数据版本不存在或不属于当前项目",
                },
            )
        input_ready = bool(
            dataset_version is not None
            and dataset_version.state == "ready"
            and effective_supports_capability(
                session, dataset_version, "tokenizer_training"
            )
            and dataset_artifact is not None
            and dataset_artifact.kind == "dataset_snapshot"
            and record_manifest is not None
            and record_manifest.kind == "dataset_record_manifest"
            and artifact_storage.managed_payload_path(session, record_manifest, project_id)
            is not None
        )
        if not input_ready and payload.start and adapter_id in BUILTIN_TOKENIZER_ADAPTERS:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_version_not_ready",
                    "message": "数据版本的不可变快照或稳定文档清单尚未就绪",
                },
            )
        if input_ready:
            assert dataset_version is not None
            assert dataset_artifact is not None
            assert record_manifest is not None
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
            }
            dataset_artifact_ids = [dataset_artifact.id]
    latest = int(
        session.scalar(
            select(func.coalesce(func.max(TokenizerVersion.version), 0)).where(
                TokenizerVersion.project_id == project_id,
                TokenizerVersion.name == payload.name,
            )
        )
        or 0
    )
    tokenizer = TokenizerVersion(
        project_id=project_id,
        name=payload.name,
        version=latest + 1,
        algorithm=payload.algorithm,
        vocab_size=payload.vocab_size,
        data_ref_type=payload.data_ref_type,
        data_ref_id=payload.data_ref_id,
        config=config,
        status="queued" if payload.start else "draft",
    )
    session.add(tokenizer)
    session.flush()
    if payload.start:
        require_executable(adapter_id, payload.command)
        requested_resources = dict(payload.resource_request)
        if adapter_id in BUILTIN_TOKENIZER_ADAPTERS:
            requested_resources["gpus"] = 0
            requested_resources.pop("gpu_indexes", None)
        resource_request = (
            _route_data_request(session, project_id, requested_resources, dataset_artifact_ids)
            if data_input
            else requested_resources
        )
        task = create_task(
            session,
            project_id=project_id,
            kind="tokenizer_train",
            title=f"训练 Tokenizer：{tokenizer.name} v{tokenizer.version}",
            adapter_id=adapter_id,
            launcher_id=payload.launcher_id,
            payload={
                "tokenizer_version_id": tokenizer.id,
                "algorithm": tokenizer.algorithm,
                "vocab_size": tokenizer.vocab_size,
                "config": tokenizer.config,
                **({"data_input": data_input} if data_input else {}),
                "resource_request": resource_request,
                "command": payload.command,
            },
        )
        tokenizer.task_id = task.id
    add_audit(
        session,
        project_id=project_id,
        action="tokenizer.create",
        subject_type="tokenizer_version",
        subject_id=tokenizer.id,
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return tokenizer.to_dict()


@router.post("/tokenizers/{tokenizer_id}/lock")
def lock_tokenizer(tokenizer_id: str, session: Session = Depends(get_session)) -> dict:
    tokenizer = get_or_404(session, TokenizerVersion, tokenizer_id, "Tokenizer")
    tokenizer.locked = True
    add_audit(
        session,
        project_id=tokenizer.project_id,
        action="tokenizer.lock",
        subject_type="tokenizer_version",
        subject_id=tokenizer.id,
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return tokenizer.to_dict()


@router.post("/tokenizers/{tokenizer_id}/set-default")
def set_default_tokenizer(tokenizer_id: str, session: Session = Depends(get_session)) -> dict:
    tokenizer = get_or_404(session, TokenizerVersion, tokenizer_id, "Tokenizer")
    session.execute(
        update(TokenizerVersion)
        .where(TokenizerVersion.project_id == tokenizer.project_id)
        .values(is_default=False)
    )
    tokenizer.is_default = True
    tokenizer.locked = True
    add_audit(
        session,
        project_id=tokenizer.project_id,
        action="tokenizer.set_default",
        subject_type="tokenizer_version",
        subject_id=tokenizer.id,
    )
    session.commit()
    return tokenizer.to_dict()
