from __future__ import annotations

from collections.abc import Iterable

from llm_platform_core.datasets import version_supports_capability
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Artifact, DatasetAnalysisRevision, DatasetVersion

TERMINAL_ANALYSIS_STATES = {"succeeded", "failed", "cancelled", "lost"}

ANALYSIS_OVERLAY_FIELDS = (
    "raw_size_bytes",
    "document_count",
    "record_count",
    "token_estimate",
    "parser_id",
    "parser_version",
    "parser_status",
    "parser_config",
    "detected_formats",
    "data_schema",
    "statistics",
    "usage_capabilities",
    "train_ready",
    "automatic_quality_score",
    "quality_report",
    "sample_preview",
    "record_manifest_artifact_id",
)


def latest_successful_analysis(
    session: Session, dataset_version_id: str
) -> DatasetAnalysisRevision | None:
    return session.scalar(
        select(DatasetAnalysisRevision)
        .where(
            DatasetAnalysisRevision.dataset_version_id == dataset_version_id,
            DatasetAnalysisRevision.status == "succeeded",
        )
        .order_by(DatasetAnalysisRevision.revision.desc())
    )


def active_analysis(
    session: Session, dataset_version_id: str
) -> DatasetAnalysisRevision | None:
    return session.scalar(
        select(DatasetAnalysisRevision)
        .where(
            DatasetAnalysisRevision.dataset_version_id == dataset_version_id,
            DatasetAnalysisRevision.status.not_in(TERMINAL_ANALYSIS_STATES),
        )
        .order_by(DatasetAnalysisRevision.revision.desc())
    )


def next_analysis_revision(session: Session, dataset_version_id: str) -> int:
    latest = session.scalar(
        select(func.coalesce(func.max(DatasetAnalysisRevision.revision), 0)).where(
            DatasetAnalysisRevision.dataset_version_id == dataset_version_id
        )
    )
    return int(latest or 0) + 1


def create_analysis_revision(
    session: Session,
    version: DatasetVersion,
    *,
    task_id: str | None,
    status: str = "queued",
) -> DatasetAnalysisRevision:
    snapshot = session.get(Artifact, version.artifact_id) if version.artifact_id else None
    analysis = DatasetAnalysisRevision(
        dataset_version_id=version.id,
        revision=next_analysis_revision(session, version.id),
        task_id=task_id,
        status=status,
        snapshot_artifact_id=version.artifact_id,
        snapshot_content_hash=snapshot.content_hash if snapshot else None,
        parser_id=version.parser_id,
        parser_version=version.parser_version,
        parser_status="pending",
        parser_config=version.parser_config or {},
    )
    session.add(analysis)
    session.flush()
    return analysis


def effective_record_manifest_id(
    session: Session, version: DatasetVersion
) -> str | None:
    analysis = latest_successful_analysis(session, version.id)
    if analysis is not None:
        return analysis.record_manifest_artifact_id
    return version.record_manifest_artifact_id


def effective_record_manifest(
    session: Session, version: DatasetVersion
) -> Artifact | None:
    artifact_id = effective_record_manifest_id(session, version)
    return session.get(Artifact, artifact_id) if artifact_id else None


def effective_train_ready(session: Session, version: DatasetVersion) -> bool:
    analysis = latest_successful_analysis(session, version.id)
    return analysis.train_ready if analysis is not None else bool(version.train_ready)


def effective_supports_capability(
    session: Session, version: DatasetVersion, capability: str
) -> bool:
    analysis = latest_successful_analysis(session, version.id)
    if analysis is None:
        return version_supports_capability(version, capability)
    if analysis.usage_capabilities:
        return capability in analysis.usage_capabilities
    return analysis.train_ready and capability in {
        "record_explorer",
        "text_pretraining",
        "tokenizer_training",
        "text_evaluation",
        "supervision_mapping",
    }


def effective_sample_preview(session: Session, version: DatasetVersion) -> list:
    analysis = latest_successful_analysis(session, version.id)
    return analysis.sample_preview if analysis is not None else version.sample_preview


def overlay_analysis(
    version: DatasetVersion, analysis: DatasetAnalysisRevision | None
) -> dict:
    result = version.to_dict()
    if analysis is not None:
        for field in ANALYSIS_OVERLAY_FIELDS:
            result[field] = getattr(analysis, field)
        result["analysis_revision_id"] = analysis.id
        result["analysis_revision"] = analysis.revision
    else:
        result["analysis_revision_id"] = None
        result["analysis_revision"] = None
    return result


def files_identity(files: Iterable[dict]) -> list[tuple[str, int, str]]:
    return sorted(
        (
            str(item.get("path") or "").replace("\\", "/"),
            int(item.get("size_bytes") or 0),
            str(item.get("sha256") or ""),
        )
        for item in files
        if isinstance(item, dict) and item.get("path")
    )
