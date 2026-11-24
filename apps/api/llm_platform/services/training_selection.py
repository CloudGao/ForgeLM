from __future__ import annotations

from fastapi import HTTPException

from ..models import Artifact, DataMixtureVersion
from .global_data import project_has_asset_access


def freeze_training_selection(
    session, project_id: str, mixture: DataMixtureVersion, source_index: int
) -> dict:
    if not project_has_asset_access(session, project_id, "mixture_version", mixture.id):
        raise HTTPException(422, detail={"code": "invalid_mixture_version"})
    artifact = session.get(Artifact, mixture.manifest_artifact_id)
    if (
        mixture.status != "ready"
        or not mixture.locked
        or artifact is None
        or artifact.kind != "data_mixture_manifest"
        or artifact.content_hash != mixture.manifest_hash
    ):
        raise HTTPException(409, detail={"code": "mixture_not_ready"})
    entry = (mixture.entries or [])[source_index]
    required = (
        "dataset_version_id",
        "dataset_artifact_hash",
        "record_manifest_artifact_id",
        "record_manifest_hash",
        "selected_document_count",
    )
    if any(not entry.get(key) for key in required):
        raise HTTPException(409, detail={"code": "selection_manifest_identity_missing"})
    return {
        "contract": "forgelm.filtered-corpus-selection.v1",
        "mixture_version_id": mixture.id,
        "manifest_artifact_id": artifact.id,
        "manifest_hash": artifact.content_hash,
        "source_index": source_index,
        "selected_token_estimate": int(entry.get("selected_tokens") or 0),
        **{
            key: entry[key]
            for key in (
                "dataset_version_id",
                "dataset_artifact_hash",
                "record_manifest_artifact_id",
                "record_manifest_hash",
                "selected_document_count",
            )
        },
    }
