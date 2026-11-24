from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    literal_column,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def new_id() -> str:
    return str(uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class RecordMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for attribute in self.__mapper__.column_attrs:
            value = getattr(self, attribute.key)
            result[attribute.key] = value.isoformat() if isinstance(value, datetime) else value
        return result


class Project(RecordMixin, Base):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    target_model_name: Mapped[str] = mapped_column(String(160))
    workspace_uri: Mapped[str] = mapped_column(Text, default="")
    project_type: Mapped[str] = mapped_column(String(40), default="pretrain")
    target_parameters: Mapped[str] = mapped_column(String(40), default="300M")
    primary_languages: Mapped[list] = mapped_column(JSON, default=list)
    target_capabilities: Mapped[list] = mapped_column(JSON, default=list)
    default_hardware: Mapped[dict] = mapped_column(JSON, default=dict)
    training_budget: Mapped[dict] = mapped_column(JSON, default=dict)
    storage_policy: Mapped[dict] = mapped_column(JSON, default=dict)
    acceptance_goal: Mapped[dict] = mapped_column(JSON, default=dict)
    lifecycle_stage: Mapped[str] = mapped_column(String(40), default="data_preparation")
    current_model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    system_kind: Mapped[str] = mapped_column(String(40), default="", index=True)


class ProjectDataBinding(RecordMixin, Base):
    """A project-scoped reference to an immutable global data asset version."""

    __tablename__ = "project_data_bindings"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    asset_type: Mapped[str] = mapped_column(String(50), index=True)
    asset_version_id: Mapped[str] = mapped_column(String(36), index=True)
    purpose: Mapped[str] = mapped_column(String(50), default="training", index=True)
    alias: Mapped[str] = mapped_column(String(180), default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        CheckConstraint(
            "asset_type IN ('dataset_version', 'mixture_version', "
            "'supervision_dataset_version', 'supervision_mixture_version', "
            "'eval_dataset_version')",
            name="ck_project_data_binding_asset_type",
        ),
        CheckConstraint(
            "purpose IN ('training', 'validation', 'evaluation', 'reference')",
            name="ck_project_data_binding_purpose",
        ),
        UniqueConstraint(
            "project_id",
            "asset_type",
            "asset_version_id",
            "purpose",
            name="uq_project_data_binding_ref",
        ),
        Index(
            "ix_project_data_bindings_asset_ref",
            "asset_type",
            "asset_version_id",
        ),
    )


class ProjectPipeline(RecordMixin, Base):
    """A named project workflow with draft and immutable published revisions."""

    __tablename__ = "project_pipelines"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    objective: Mapped[str] = mapped_column(String(40), default="pretraining", index=True)
    current_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    draft_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        CheckConstraint(
            "objective IN ('pretraining', 'post_training', 'evaluation', 'release')",
            name="ck_project_pipeline_objective",
        ),
        UniqueConstraint("project_id", "name", name="uq_project_pipeline_name"),
    )


class ProjectPipelineRevision(RecordMixin, Base):
    """A mutable draft or immutable published graph and exact version mapping."""

    __tablename__ = "project_pipeline_revisions"

    pipeline_id: Mapped[str] = mapped_column(
        ForeignKey("project_pipelines.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    data_refs: Mapped[list] = mapped_column(JSON, default=list)
    tokenizer_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("tokenizer_versions.id"), nullable=True, index=True
    )
    code_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("code_versions.id"), nullable=True, index=True
    )
    plan_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("plan_revisions.id"), nullable=True, index=True
    )
    training_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_runs.id"), nullable=True, index=True
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    evaluation_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_runs.id"), nullable=True, index=True
    )
    quantization_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("quantization_runs.id"), nullable=True, index=True
    )
    release_id: Mapped[str | None] = mapped_column(
        ForeignKey("releases.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="published", index=True)
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    edges: Mapped[list] = mapped_column(JSON, default=list)
    variables: Mapped[dict] = mapped_column(JSON, default=dict)
    viewport: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_report: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    mapping_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    locked: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published')",
            name="ck_project_pipeline_revision_status",
        ),
        UniqueConstraint(
            "pipeline_id", "revision", name="uq_project_pipeline_revision"
        ),
    )


class ProjectPipelineInstance(RecordMixin, Base):
    """A concrete, lockable execution configuration based on one definition revision."""

    __tablename__ = "project_pipeline_instances"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("project_pipelines.id", ondelete="CASCADE"), index=True
    )
    definition_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("project_pipeline_revisions.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    edges: Mapped[list] = mapped_column(JSON, default=list)
    variables: Mapped[dict] = mapped_column(JSON, default=dict)
    viewport: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_report: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    locked: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'ready', 'archived')",
            name="ck_project_pipeline_instance_status",
        ),
        UniqueConstraint(
            "project_id", "name", name="uq_project_pipeline_instance_name"
        ),
    )


class ProjectPipelineRun(RecordMixin, Base):
    """One execution of a locked Pipeline instance and frozen graph snapshot."""

    __tablename__ = "project_pipeline_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    pipeline_id: Mapped[str | None] = mapped_column(
        ForeignKey("project_pipelines.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("project_pipeline_revisions.id"), index=True
    )
    instance_id: Mapped[str] = mapped_column(
        ForeignKey("project_pipeline_instances.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    trigger_kind: Mapped[str] = mapped_column(String(40), default="manual")
    input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    context_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    graph_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    node_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'cancelling', 'succeeded', "
            "'failed', 'cancelled')",
            name="ck_project_pipeline_run_status",
        ),
    )


class ProjectPipelineNodeRun(RecordMixin, Base):
    """Frozen execution state for one stable node in a Pipeline run."""

    __tablename__ = "project_pipeline_node_runs"

    pipeline_run_id: Mapped[str] = mapped_column(
        ForeignKey("project_pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    node_id: Mapped[str] = mapped_column(String(120))
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    label: Mapped[str] = mapped_column(String(180))
    status: Mapped[str] = mapped_column(String(24), default="waiting", index=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    output_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
    )
    branch_decision: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('waiting', 'queued', 'running', 'paused', 'succeeded', "
            "'failed', 'cancelled', 'skipped')",
            name="ck_project_pipeline_node_run_status",
        ),
        UniqueConstraint(
            "pipeline_run_id", "node_id", name="uq_project_pipeline_node_run"
        ),
    )


class LineageExecution(RecordMixin, Base):
    """Immutable lineage context for one terminal Pipeline node execution."""

    __tablename__ = "lineage_executions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    pipeline_run_id: Mapped[str] = mapped_column(
        ForeignKey("project_pipeline_runs.id", ondelete="CASCADE"), index=True
    )
    node_run_id: Mapped[str] = mapped_column(
        ForeignKey("project_pipeline_node_runs.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    node_id: Mapped[str] = mapped_column(String(120), index=True)
    node_type: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    execution_owner: Mapped[str] = mapped_column(String(40), default="pipeline_scheduler")
    execution_identity: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('succeeded', 'failed', 'cancelled', 'skipped')",
            name="ck_lineage_execution_terminal_status",
        ),
    )


class LineageAssetRelation(RecordMixin, Base):
    """A consumed or produced immutable asset reference for a lineage execution."""

    __tablename__ = "lineage_asset_relations"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    execution_id: Mapped[str] = mapped_column(
        ForeignKey("lineage_executions.id", ondelete="CASCADE"), index=True
    )
    relation_kind: Mapped[str] = mapped_column(String(24), index=True)
    asset_type: Mapped[str] = mapped_column(String(80), index=True)
    asset_id: Mapped[str] = mapped_column(String(120), index=True)
    role: Mapped[str] = mapped_column(String(80), default="")
    label: Mapped[str] = mapped_column(String(220))
    version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        CheckConstraint(
            "relation_kind IN ('consumed', 'produced')",
            name="ck_lineage_asset_relation_kind",
        ),
        UniqueConstraint(
            "execution_id",
            "relation_kind",
            "asset_type",
            "asset_id",
            "role",
            name="uq_lineage_asset_relation",
        ),
        Index(
            "ix_lineage_asset_relations_project_asset",
            "project_id",
            "asset_type",
            "asset_id",
        ),
    )


class CodeSource(RecordMixin, Base):
    __tablename__ = "code_sources"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    source_type: Mapped[str] = mapped_column(String(40))
    source_uri: Mapped[str] = mapped_column(Text, default="")
    source_config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="registered", index=True)
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_code_source_project_name"),)


class CodeVersion(RecordMixin, Base):
    __tablename__ = "code_versions"

    source_id: Mapped[str] = mapped_column(
        ForeignKey("code_sources.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    entrypoints: Mapped[dict] = mapped_column(JSON, default=dict)
    config_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    training_contract: Mapped[dict] = mapped_column(JSON, default=dict)
    dependency_spec: Mapped[str] = mapped_column(Text, default="")
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (UniqueConstraint("source_id", "version", name="uq_code_source_version"),)


class Artifact(RecordMixin, Base):
    __tablename__ = "artifacts"

    kind: Mapped[str] = mapped_column(String(60), index=True)
    uri: Mapped[str] = mapped_column(Text, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    media_type: Mapped[str] = mapped_column(String(160), default="application/octet-stream")
    manifest_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_by_task_attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    availability_state: Mapped[str] = mapped_column(String(30), default="available")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (UniqueConstraint("kind", "content_hash", name="uq_artifact_kind_hash"),)


class ArtifactLocation(RecordMixin, Base):
    __tablename__ = "artifact_locations"

    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("artifacts.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    storage_type: Mapped[str] = mapped_column(String(30), default="external", index=True)
    uri: Mapped[str] = mapped_column(Text, unique=True)
    worker_id: Mapped[str | None] = mapped_column(
        ForeignKey("worker_nodes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    availability_state: Mapped[str] = mapped_column(String(30), default="unknown", index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (UniqueConstraint("artifact_id", "uri", name="uq_artifact_location_uri"),)


class StorageScan(RecordMixin, Base):
    __tablename__ = "storage_scans"

    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    scope: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StorageInventoryItem(RecordMixin, Base):
    __tablename__ = "storage_inventory_items"

    scan_id: Mapped[str] = mapped_column(
        ForeignKey("storage_scans.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    item_type: Mapped[str] = mapped_column(String(30), default="category", index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    reclaimable: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_attempts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("scan_id", "path", "item_type", name="uq_storage_scan_path_type"),
    )


class StorageCleanupPlan(RecordMixin, Base):
    __tablename__ = "storage_cleanup_plans"

    scan_id: Mapped[str] = mapped_column(
        ForeignKey("storage_scans.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="planned", index=True)
    confirmation_phrase: Mapped[str] = mapped_column(String(120))
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StorageCleanupPlanItem(RecordMixin, Base):
    __tablename__ = "storage_cleanup_plan_items"

    plan_id: Mapped[str] = mapped_column(
        ForeignKey("storage_cleanup_plans.id", ondelete="CASCADE"), index=True
    )
    inventory_item_id: Mapped[str | None] = mapped_column(
        ForeignKey("storage_inventory_items.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    category: Mapped[str] = mapped_column(String(40), index=True)
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_attempts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="planned", index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    actual_reclaimed_bytes: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class HistoryImport(RecordMixin, Base):
    __tablename__ = "history_imports"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    source_key: Mapped[str] = mapped_column(String(180), index=True)
    manifest_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(30), default="succeeded")
    summary: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "source_key", "manifest_hash", name="uq_history_import_manifest"
        ),
    )


class HistoryImportEntity(RecordMixin, Base):
    __tablename__ = "history_import_entities"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    source_key: Mapped[str] = mapped_column(String(180), index=True)
    entity_type: Mapped[str] = mapped_column(String(60), index=True)
    external_id: Mapped[str] = mapped_column(String(240))
    local_id: Mapped[str] = mapped_column(String(36), index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_key",
            "entity_type",
            "external_id",
            name="uq_history_import_entity",
        ),
    )


class Dataset(RecordMixin, Base):
    __tablename__ = "datasets"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    source_type: Mapped[str] = mapped_column(String(40))
    modality: Mapped[str] = mapped_column(String(24), default="text", index=True)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_config: Mapped[dict] = mapped_column(JSON, default=dict)
    license: Mapped[str] = mapped_column(String(120), default="unknown")
    language: Mapped[str] = mapped_column(String(80), default="unknown")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(40), default="registered")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
            name="ck_dataset_modality",
        ),
        UniqueConstraint("project_id", "name", name="uq_dataset_project_name"),
    )


class DatasetVersion(RecordMixin, Base):
    __tablename__ = "dataset_versions"

    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    modality: Mapped[str] = mapped_column(String(24), default="text", index=True)
    tier: Mapped[str] = mapped_column(String(20), default="bronze")
    state: Mapped[str] = mapped_column(String(40), default="ready")
    raw_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    parser_id: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    parser_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    parser_status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    parser_config: Mapped[dict] = mapped_column(JSON, default=dict)
    detected_formats: Mapped[list] = mapped_column(JSON, default=list)
    data_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    statistics: Mapped[dict] = mapped_column(JSON, default=dict)
    usage_capabilities: Mapped[list] = mapped_column(JSON, default=list)
    automatic_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_report: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_preview: Mapped[list] = mapped_column(JSON, default=list)
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    record_manifest_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, index=True
    )
    conversion_recipe_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_conversion_recipe_versions.id"), nullable=True, index=True
    )
    conversion_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_conversion_runs.id"), nullable=True, index=True
    )
    processing_recipe_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_processing_recipe_versions.id"), nullable=True, index=True
    )
    processing_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_processing_runs.id"), nullable=True, index=True
    )
    materialization_kind: Mapped[str] = mapped_column(
        String(16), default="imported", index=True
    )
    materialization_spec: Mapped[dict] = mapped_column(JSON, default=dict)
    materialization_state: Mapped[str] = mapped_column(
        String(24), default="pending", index=True
    )
    primary_location_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifact_locations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)
    train_ready: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        CheckConstraint(
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
            name="ck_dataset_version_modality",
        ),
        CheckConstraint(
            "materialization_kind IN ('download', 'derived', 'imported')",
            name="ck_dataset_version_materialization_kind",
        ),
        CheckConstraint(
            "materialization_state IN ('pending', 'ready', 'degraded')",
            name="ck_dataset_version_materialization_state",
        ),
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    def to_dict(self) -> dict[str, Any]:
        view = super().to_dict()
        view.pop("materialization_spec", None)
        return view


class DatasetMaterialization(RecordMixin, Base):
    """One frozen plan for making a DatasetVersion available on a target Worker."""

    __tablename__ = "dataset_materializations"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    target_worker_id: Mapped[str] = mapped_column(
        ForeignKey("worker_nodes.id", ondelete="CASCADE"), index=True
    )
    selection_mode: Mapped[str] = mapped_column(String(16), default="auto", index=True)
    requested_route_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    candidate_routes: Mapped[list] = mapped_column(JSON, default=list)
    chosen_route: Mapped[dict] = mapped_column(JSON, default=dict)
    expected_content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    result_location_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifact_locations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    error: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        CheckConstraint(
            "selection_mode IN ('auto', 'manual')",
            name="ck_dataset_materialization_selection_mode",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_dataset_materialization_status",
        ),
    )


class DatasetAnalysisRevision(RecordMixin, Base):
    __tablename__ = "dataset_analysis_revisions"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    snapshot_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    snapshot_content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    raw_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    parser_id: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    parser_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    parser_status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    parser_config: Mapped[dict] = mapped_column(JSON, default=dict)
    detected_formats: Mapped[list] = mapped_column(JSON, default=list)
    data_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    statistics: Mapped[dict] = mapped_column(JSON, default=dict)
    usage_capabilities: Mapped[list] = mapped_column(JSON, default=list)
    train_ready: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    automatic_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_report: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_preview: Mapped[list] = mapped_column(JSON, default=list)
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    record_manifest_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    error: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id", "revision", name="uq_dataset_analysis_revision"
        ),
    )


class DatasetVersionIndex(RecordMixin, Base):
    __tablename__ = "dataset_version_indexes"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    analysis_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_analysis_revisions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    index_kind: Mapped[str] = mapped_column(String(80), index=True)
    format: Mapped[str] = mapped_column(String(160), default="")
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"), index=True)
    parser_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ready", index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "analysis_revision_id",
            "index_kind",
            name="uq_dataset_analysis_index_kind",
        ),
    )


class DataConversionRecipe(RecordMixin, Base):
    __tablename__ = "data_conversion_recipes"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_data_conversion_recipe_project_name"),
    )


class DataConversionRecipeVersion(RecordMixin, Base):
    __tablename__ = "data_conversion_recipe_versions"

    recipe_id: Mapped[str] = mapped_column(
        ForeignKey("data_conversion_recipes.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    engine_adapter_id: Mapped[str] = mapped_column(
        String(180), default="builtin.data-conversion"
    )
    operators: Mapped[list] = mapped_column(JSON, default=list)
    execution_config: Mapped[dict] = mapped_column(JSON, default=dict)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("recipe_id", "version", name="uq_data_conversion_recipe_version"),
    )


class DataConversionRun(RecordMixin, Base):
    __tablename__ = "data_conversion_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    recipe_version_id: Mapped[str] = mapped_column(
        ForeignKey("data_conversion_recipe_versions.id"), index=True
    )
    input_dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    output_dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, unique=True
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, unique=True
    )
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    engine_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DataProcessingRecipe(RecordMixin, Base):
    __tablename__ = "data_processing_recipes"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_data_processing_recipe_project_name"),
    )


class DataProcessingRecipeVersion(RecordMixin, Base):
    __tablename__ = "data_processing_recipe_versions"

    recipe_id: Mapped[str] = mapped_column(
        ForeignKey("data_processing_recipes.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    engine_adapter_id: Mapped[str] = mapped_column(String(180), default="builtin.data-juicer")
    operators: Mapped[list] = mapped_column(JSON, default=list)
    execution_config: Mapped[dict] = mapped_column(JSON, default=dict)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("recipe_id", "version", name="uq_data_processing_recipe_version"),
    )


class DataProcessingRun(RecordMixin, Base):
    __tablename__ = "data_processing_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    recipe_version_id: Mapped[str] = mapped_column(
        ForeignKey("data_processing_recipe_versions.id"), index=True
    )
    input_dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    output_dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, unique=True
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, unique=True
    )
    output_tier: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    engine_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class SampleAnnotation(RecordMixin, Base):
    __tablename__ = "sample_annotations"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    sample_key: Mapped[str] = mapped_column(String(160))
    decision: Mapped[str | None] = mapped_column(String(30), nullable=True)
    human_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)


class LabelScheme(RecordMixin, Base):
    __tablename__ = "label_schemes"

    code: Mapped[str] = mapped_column(String(80))
    name: Mapped[str] = mapped_column(String(180))
    modality: Mapped[str] = mapped_column(String(24), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
            name="ck_label_scheme_modality",
        ),
        CheckConstraint(
            "status IN ('active', 'archived')", name="ck_label_scheme_status"
        ),
        UniqueConstraint("modality", "code", name="uq_label_scheme_modality_code"),
    )


class LabelSchemeVersion(RecordMixin, Base):
    __tablename__ = "label_scheme_versions"

    scheme_id: Mapped[str] = mapped_column(
        ForeignKey("label_schemes.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(180), default="local-user")
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("scheme_id", "version", name="uq_label_scheme_version"),
    )


class LabelGroup(RecordMixin, Base):
    __tablename__ = "label_groups"

    scheme_version_id: Mapped[str] = mapped_column(
        ForeignKey("label_scheme_versions.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(80))
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    selection_mode: Mapped[str] = mapped_column(String(16), default="single")
    required: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        CheckConstraint(
            "selection_mode IN ('single', 'multiple')",
            name="ck_label_group_selection_mode",
        ),
        UniqueConstraint("scheme_version_id", "code", name="uq_label_group_version_code"),
    )


class LabelDefinition(RecordMixin, Base):
    __tablename__ = "label_definitions"

    group_id: Mapped[str] = mapped_column(
        ForeignKey("label_groups.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(80))
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(32), default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("group_id", "code", name="uq_label_definition_group_code"),
    )


class DatasetLabelOperation(RecordMixin, Base):
    __tablename__ = "dataset_label_operations"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    scheme_version_id: Mapped[str] = mapped_column(
        ForeignKey("label_scheme_versions.id"), index=True
    )
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True, index=True)
    base_snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    output_snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_query_task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    scope: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(16))
    label_ids: Mapped[list] = mapped_column(JSON, default=list)
    query_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    annotator: Mapped[str] = mapped_column(String(180), default="local-user")
    note: Mapped[str] = mapped_column(Text, default="")
    selection_count: Mapped[int] = mapped_column(Integer, default=0)
    selection_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    delta_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    error: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        CheckConstraint("scope IN ('selected', 'page', 'query')", name="ck_label_operation_scope"),
        CheckConstraint("action IN ('apply', 'remove')", name="ck_label_operation_action"),
    )


class DatasetLabelSnapshot(RecordMixin, Base):
    __tablename__ = "dataset_label_snapshots"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    parent_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_label_snapshots.id"), nullable=True, index=True
    )
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_label_operations.id", ondelete="CASCADE"), unique=True, index=True
    )
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"), index=True)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    assignment_count: Mapped[int] = mapped_column(Integer, default=0)
    labeled_record_count: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "version", name="uq_dataset_label_snapshot_version"),
    )


class DataMixture(RecordMixin, Base):
    __tablename__ = "data_mixtures"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_mixture_project_name"),)


class DataMixtureVersion(RecordMixin, Base):
    __tablename__ = "data_mixture_versions"

    mixture_id: Mapped[str] = mapped_column(
        ForeignKey("data_mixtures.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    entries: Mapped[list] = mapped_column(JSON, default=list)
    target_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_steps: Mapped[int] = mapped_column(Integer, default=0)
    distribution: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_result: Mapped[dict] = mapped_column(JSON, default=dict)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(40), default="resolving", index=True)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    resolver_version: Mapped[str] = mapped_column(String(80), default="stable_hash_selector_v1")
    sampler_version: Mapped[str] = mapped_column(String(80), default="weighted_hash_sampler_v1")
    manifest_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True
    )
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    resolved_document_count: Mapped[int] = mapped_column(Integer, default=0)
    resolved_token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("mixture_id", "version", name="uq_mixture_version"),)


class TokenizerVersion(RecordMixin, Base):
    __tablename__ = "tokenizer_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[int] = mapped_column(Integer)
    algorithm: Mapped[str] = mapped_column(String(80))
    vocab_size: Mapped[int] = mapped_column(Integer)
    data_ref_type: Mapped[str] = mapped_column(String(40))
    data_ref_id: Mapped[str] = mapped_column(String(36))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="draft")
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", "version", name="uq_tokenizer_version"),
    )


class ChatTemplate(RecordMixin, Base):
    __tablename__ = "chat_templates"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_chat_template_project_name"),
    )


class ChatTemplateVersion(RecordMixin, Base):
    __tablename__ = "chat_template_versions"

    template_id: Mapped[str] = mapped_column(
        ForeignKey("chat_templates.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    template_body: Mapped[str] = mapped_column(Text)
    role_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    required_special_tokens: Mapped[list] = mapped_column(JSON, default=list)
    assistant_mask_mode: Mapped[str] = mapped_column(
        String(40), default="generation_spans_v1"
    )
    renderer_version: Mapped[str] = mapped_column(String(80), default="hf_chat_template_v1")
    validation_report: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("template_id", "version", name="uq_chat_template_version"),
        UniqueConstraint("template_id", "content_hash", name="uq_chat_template_content"),
    )


class SupervisionDataset(RecordMixin, Base):
    __tablename__ = "supervision_datasets"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_supervision_dataset_project_name"),
    )


class SupervisionDatasetVersion(RecordMixin, Base):
    __tablename__ = "supervision_dataset_versions"

    supervision_dataset_id: Mapped[str] = mapped_column(
        ForeignKey("supervision_datasets.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30), index=True)
    mapping_config: Mapped[dict] = mapped_column(JSON, default=dict)
    filter_config: Mapped[dict] = mapped_column(JSON, default=dict)
    validator_version: Mapped[str] = mapped_column(String(80), default="supervision_schema_v1")
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    validation_report: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_preview: Mapped[list] = mapped_column(JSON, default=list)
    accepted_record_count: Mapped[int] = mapped_column(Integer, default=0)
    rejected_record_count: Mapped[int] = mapped_column(Integer, default=0)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint(
            "supervision_dataset_id", "version", name="uq_supervision_dataset_version"
        ),
        UniqueConstraint(
            "supervision_dataset_id", "identity_hash", name="uq_supervision_dataset_identity"
        ),
    )


class SupervisionMixture(RecordMixin, Base):
    __tablename__ = "supervision_mixtures"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_supervision_mixture_project_name"),
    )


class SupervisionMixtureVersion(RecordMixin, Base):
    __tablename__ = "supervision_mixture_versions"

    mixture_id: Mapped[str] = mapped_column(
        ForeignKey("supervision_mixtures.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    entries: Mapped[list] = mapped_column(JSON, default=list)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    sampling_policy: Mapped[str] = mapped_column(String(80), default="weighted_deficit_v1")
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(40), default="ready", index=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("mixture_id", "version", name="uq_supervision_mixture_version"),
        UniqueConstraint("mixture_id", "identity_hash", name="uq_supervision_mixture_identity"),
    )


class SupervisionCorpusVersion(RecordMixin, Base):
    __tablename__ = "supervision_corpus_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    mixture_version_id: Mapped[str] = mapped_column(
        ForeignKey("supervision_mixture_versions.id"), index=True
    )
    tokenizer_version_id: Mapped[str] = mapped_column(
        ForeignKey("tokenizer_versions.id"), index=True
    )
    chat_template_version_id: Mapped[str] = mapped_column(
        ForeignKey("chat_template_versions.id"), index=True
    )
    objective_kind: Mapped[str] = mapped_column(String(30), index=True)
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    format: Mapped[str] = mapped_column(String(80))
    example_count: Mapped[int] = mapped_column(Integer, default=0)
    pair_count: Mapped[int] = mapped_column(Integer, default=0)
    raw_token_count: Mapped[int] = mapped_column(Integer, default=0)
    supervised_token_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("project_id", "identity_hash", name="uq_supervision_corpus_identity"),
    )


class SupervisionTrainingViewVersion(RecordMixin, Base):
    __tablename__ = "supervision_training_view_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    corpus_version_id: Mapped[str] = mapped_column(
        ForeignKey("supervision_corpus_versions.id"), index=True
    )
    objective_kind: Mapped[str] = mapped_column(String(30), index=True)
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    format: Mapped[str] = mapped_column(String(80))
    budget_unit: Mapped[str] = mapped_column(String(40))
    budget_value: Mapped[int] = mapped_column(Integer)
    example_count: Mapped[int] = mapped_column(Integer, default=0)
    pair_count: Mapped[int] = mapped_column(Integer, default=0)
    raw_token_count: Mapped[int] = mapped_column(Integer, default=0)
    supervised_token_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "identity_hash", name="uq_supervision_training_view_identity"
        ),
    )


class TokenizedDatasetVersion(RecordMixin, Base):
    __tablename__ = "tokenized_dataset_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    mixture_version_id: Mapped[str] = mapped_column(
        ForeignKey("data_mixture_versions.id"), index=True
    )
    tokenizer_version_id: Mapped[str] = mapped_column(
        ForeignKey("tokenizer_versions.id"), index=True
    )
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    format: Mapped[str] = mapped_column(String(80), default="forgelm-packed-tokens-v1")
    packing_version: Mapped[str] = mapped_column(String(80), default="document-pack-v1")
    sequence_length: Mapped[int] = mapped_column(Integer)
    target_tokens: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    sequence_count: Mapped[int] = mapped_column(Integer, default=0)
    shard_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "identity_hash", name="uq_tokenized_dataset_identity"
        ),
    )


class TokenizedCorpusVersion(RecordMixin, Base):
    __tablename__ = "tokenized_corpus_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    tokenizer_version_id: Mapped[str] = mapped_column(
        ForeignKey("tokenizer_versions.id"), index=True
    )
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    format: Mapped[str] = mapped_column(String(80), default="forgelm-token-corpus-v2")
    selection_config: Mapped[dict] = mapped_column(JSON, default=dict)
    encoding_config: Mapped[dict] = mapped_column(JSON, default=dict)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    shard_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("project_id", "identity_hash", name="uq_tokenized_corpus_identity"),
    )


class TrainingDataViewVersion(RecordMixin, Base):
    __tablename__ = "training_data_view_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    mixture_version_id: Mapped[str] = mapped_column(
        ForeignKey("data_mixture_versions.id"), index=True
    )
    tokenizer_version_id: Mapped[str] = mapped_column(
        ForeignKey("tokenizer_versions.id"), index=True
    )
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    format: Mapped[str] = mapped_column(String(80), default="forgelm-training-index-v1")
    sequence_length: Mapped[int] = mapped_column(Integer)
    requested_tokens: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    sequence_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("project_id", "identity_hash", name="uq_training_data_view_identity"),
    )


class TrainingValidationSet(RecordMixin, Base):
    __tablename__ = "training_validation_sets"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[int] = mapped_column(Integer)
    source_type: Mapped[str] = mapped_column(String(40), default="upload")
    format: Mapped[str] = mapped_column(String(20))
    filename: Mapped[str] = mapped_column(String(255))
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="ready", index=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"), index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "name", "version", name="uq_training_validation_set_version"
        ),
    )


class TrainingPlan(RecordMixin, Base):
    __tablename__ = "training_plans"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(40), default="draft")
    current_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class ModelSpecification(RecordMixin, Base):
    __tablename__ = "model_specifications"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(40), default="ready", index=True)
    current_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_model_specification_project_name"),
    )


class ModelSpecificationRevision(RecordMixin, Base):
    __tablename__ = "model_specification_revisions"

    specification_id: Mapped[str] = mapped_column(
        ForeignKey("model_specifications.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    source_type: Mapped[str] = mapped_column(String(40), default="standard")
    architecture_family_id: Mapped[str] = mapped_column(String(120))
    architecture_version: Mapped[str] = mapped_column(String(40), default="1")
    preset_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    code_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("code_versions.id"), nullable=True, index=True
    )
    user_config: Mapped[dict] = mapped_column(JSON, default=dict)
    resolved_config: Mapped[dict] = mapped_column(JSON, default=dict)
    initialization: Mapped[dict] = mapped_column(JSON, default=dict)
    parameter_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_requirements: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_code: Mapped[str] = mapped_column(Text, default="")
    config_signature: Mapped[str] = mapped_column(String(64), index=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint(
            "specification_id", "revision", name="uq_model_specification_revision"
        ),
    )


class PlanRevision(RecordMixin, Base):
    __tablename__ = "plan_revisions"

    plan_id: Mapped[str] = mapped_column(
        ForeignKey("training_plans.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    stages: Mapped[list] = mapped_column(JSON, default=list)
    input_model_ref: Mapped[dict] = mapped_column(JSON, default=dict)
    model_specification_revision_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    tokenizer_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("tokenizer_versions.id"), nullable=True
    )
    validation_set_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_validation_sets.id"), nullable=True, index=True
    )
    validation_mixture_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_mixture_versions.id"), nullable=True, index=True
    )
    code_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("code_versions.id"), nullable=True, index=True
    )
    engine_plugin_id: Mapped[str] = mapped_column(String(180), default="builtin.command.training")
    launcher_plugin_id: Mapped[str] = mapped_column(String(180), default="builtin.local-process")
    resource_request: Mapped[dict] = mapped_column(JSON, default=dict)
    effective_config: Mapped[dict] = mapped_column(JSON, default=dict)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("plan_id", "revision", name="uq_plan_revision"),)


class Task(RecordMixin, Base):
    __tablename__ = "tasks"

    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(String(220))
    adapter_id: Mapped[str] = mapped_column(String(180), default="builtin.command.training")
    launcher_id: Mapped[str] = mapped_column(String(180), default="builtin.local-process")
    desired_state: Mapped[str] = mapped_column(String(30), default="queued")
    observed_state: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[float] = mapped_column(Float, default=0)
    current_phase: Mapped[str | None] = mapped_column(String(180), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    active_attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "idx_tasks_claimable",
            "observed_state",
            "created_at",
            sqlite_where=(observed_state == "queued"),
        ),
    )


class TaskAttempt(RecordMixin, Base):
    __tablename__ = "task_attempts"

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    worker_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="assigned")
    runtime_handle: Mapped[dict] = mapped_column(JSON, default=dict)
    environment_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    resource_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_metrics_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    log_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (UniqueConstraint("task_id", "attempt_number", name="uq_task_attempt"),)


class ExecutionSnapshot(RecordMixin, Base):
    __tablename__ = "execution_snapshots"
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("task_attempts.id", ondelete="CASCADE"), index=True)
    phase: Mapped[str] = mapped_column(String(30))
    digest: Mapped[str] = mapped_column(String(64), index=True)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (UniqueConstraint("attempt_id", "phase", name="uq_execution_snapshot_phase"),)


class TrainingStateCheckpoint(RecordMixin, Base):
    __tablename__ = "training_state_checkpoints"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("task_attempts.id", ondelete="CASCADE"), index=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"))
    step: Mapped[int] = mapped_column(Integer)
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    execution_digest: Mapped[str] = mapped_column(String(64), default="")
    __table_args__ = (UniqueConstraint("attempt_id", "step", name="uq_training_state_checkpoint_step"),)


class ResourceLease(RecordMixin, Base):
    __tablename__ = "resource_leases"

    task_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("task_attempts.id", ondelete="CASCADE"), unique=True, index=True
    )
    worker_id: Mapped[str] = mapped_column(
        ForeignKey("worker_nodes.id", ondelete="CASCADE"), index=True
    )
    allocation: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TaskEvent(Base):
    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_attempts.id", ondelete="SET NULL"), nullable=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("task_id", "sequence", name="uq_task_event_sequence"),)


def worker_event_sequence():
    # SQLite expression indexes require the same literal JSON path in queries.
    # A bound path parameter prevents index use even if its value is identical.
    return func.json_extract(TaskEvent.payload, literal_column("'$.\"worker_sequence\"'"))


Index("ix_task_event_worker_sequence", TaskEvent.task_id, TaskEvent.attempt_id,
      worker_event_sequence())


def numeric_task_progress():
    return literal_column(
        "event_type IN ('progress', 'metric', 'checkpoint') "
        "AND json_type(payload, '$.progress') IN ('integer', 'real')"
    )


Index("ix_task_event_numeric_progress", TaskEvent.task_id, TaskEvent.attempt_id,
      TaskEvent.sequence, sqlite_where=numeric_task_progress())
Index("ix_task_event_attempt_sequence", TaskEvent.task_id, TaskEvent.attempt_id, TaskEvent.sequence)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(80), index=True)
    aggregate_id: Mapped[str] = mapped_column(String(36), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingRun(RecordMixin, Base):
    __tablename__ = "training_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    plan_revision_id: Mapped[str] = mapped_column(ForeignKey("plan_revisions.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(120), nullable=True)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    current_tokens: Mapped[int] = mapped_column(Integer, default=0)
    progress_counters: Mapped[dict] = mapped_column(JSON, default=dict)
    latest_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Checkpoint(RecordMixin, Base):
    __tablename__ = "checkpoints"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), index=True
    )
    parent_model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    step: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    training_position: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="available")

    __table_args__ = (UniqueConstraint("training_run_id", "step", name="uq_checkpoint_run_step"),)


class ModelWeightAsset(RecordMixin, Base):
    __tablename__ = "model_weight_assets"

    name: Mapped[str] = mapped_column(String(180), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    source_provider: Mapped[str] = mapped_column(String(40), default="huggingface", index=True)
    repository_id: Mapped[str] = mapped_column(String(500), index=True)
    license: Mapped[str] = mapped_column(String(120), default="unknown")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "source_provider", "repository_id", name="uq_model_weight_asset_source"
        ),
    )


class ModelWeightVersion(RecordMixin, Base):
    __tablename__ = "model_weight_versions"

    asset_id: Mapped[str] = mapped_column(
        ForeignKey("model_weight_assets.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    identity_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    requested_revision: Mapped[str] = mapped_column(String(180), default="main")
    resolved_commit: Mapped[str] = mapped_column(String(64), index=True)
    source_endpoint: Mapped[str] = mapped_column(Text, default="https://huggingface.co")
    selected_paths: Mapped[list] = mapped_column(JSON, default=list)
    source_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    format_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_capabilities: Mapped[list] = mapped_column(JSON, default=list)
    parameter_count: Mapped[int] = mapped_column(Integer, default=0)
    quantization: Mapped[str | None] = mapped_column(String(80), nullable=True)
    state: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("asset_id", "version", name="uq_model_weight_asset_version"),
    )


class ModelVersion(RecordMixin, Base):
    __tablename__ = "model_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(40), default="base")
    parent_model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    source_checkpoint_id: Mapped[str | None] = mapped_column(
        ForeignKey("checkpoints.id"), nullable=True
    )
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    model_weight_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_weight_versions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    tokenizer_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("tokenizer_versions.id"), nullable=True
    )
    chat_template_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_template_versions.id"), nullable=True, index=True
    )
    objective_kind: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    parameter_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="candidate")
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    is_candidate: Mapped[bool] = mapped_column(Boolean, default=True)
    is_official: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (UniqueConstraint("project_id", "name", "version", name="uq_model_version"),)


class EvalDatasetVersion(RecordMixin, Base):
    __tablename__ = "eval_dataset_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[int] = mapped_column(Integer, default=1)
    source_type: Mapped[str] = mapped_column(String(60), default="custom")
    task_type: Mapped[str] = mapped_column(String(80), default="generation")
    license: Mapped[str] = mapped_column(String(120), default="unknown")
    dimensions: Mapped[list] = mapped_column(JSON, default=list)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    contamination_tags: Mapped[list] = mapped_column(JSON, default=list)
    source_config: Mapped[dict] = mapped_column(JSON, default=dict)
    schema_config: Mapped[dict] = mapped_column(JSON, default=dict)
    scoring_config: Mapped[dict] = mapped_column(JSON, default=dict)
    benchmark_config: Mapped[dict] = mapped_column(JSON, default=dict)
    compatibility_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)


class EvalSuiteVersion(RecordMixin, Base):
    __tablename__ = "eval_suite_versions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[int] = mapped_column(Integer, default=1)
    datasets: Mapped[list] = mapped_column(JSON, default=list)
    scoring_config: Mapped[dict] = mapped_column(JSON, default=dict)
    release_gate: Mapped[dict] = mapped_column(JSON, default=dict)


class EvaluationRun(RecordMixin, Base):
    __tablename__ = "evaluation_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True)
    model_version_ids: Mapped[list] = mapped_column(JSON, default=list)
    suite_version_ids: Mapped[list] = mapped_column(JSON, default=list)
    inference_config: Mapped[dict] = mapped_column(JSON, default=dict)
    scoring_config: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    benchmark_signature: Mapped[dict] = mapped_column(JSON, default=dict)
    compatibility_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    sample_results: Mapped[list] = mapped_column(JSON, default=list)


class QuantizationJob(RecordMixin, Base):
    """A reusable project quantization configuration."""

    __tablename__ = "quantization_jobs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    plugin_version_id: Mapped[str] = mapped_column(
        ForeignKey("plugin_versions.id"), index=True
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    calibration_data_ref: Mapped[dict] = mapped_column(JSON, default=dict)
    resource_request: Mapped[dict] = mapped_column(JSON, default=dict)
    launcher_id: Mapped[str] = mapped_column(String(180), default="builtin.local-process")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_project_quantization_job_name"),
    )


class QuantizationRun(RecordMixin, Base):
    __tablename__ = "quantization_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    quantization_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("quantization_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    run_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    plugin_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("plugin_versions.id"), nullable=True, index=True
    )
    trigger_kind: Mapped[str] = mapped_column(String(30), default="legacy")
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True)
    method: Mapped[str] = mapped_column(String(60))
    format: Mapped[str] = mapped_column(String(60))
    calibration_data_ref: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    metrics_before: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics_after: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "quantization_job_id",
            "run_number",
            name="uq_quantization_job_run_number",
        ),
    )


class ReleaseJob(RecordMixin, Base):
    """A reusable, editable release runtime build configuration."""

    __tablename__ = "release_jobs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text, default="")
    source_kind: Mapped[str] = mapped_column(String(30), index=True)
    source_id: Mapped[str] = mapped_column(String(36), index=True)
    exporter_plugin_version_id: Mapped[str] = mapped_column(
        ForeignKey("plugin_versions.id"), index=True
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_policy: Mapped[dict] = mapped_column(JSON, default=dict)
    model_card_template: Mapped[str] = mapped_column(Text, default="")
    license_policy: Mapped[dict] = mapped_column(JSON, default=dict)
    resource_request: Mapped[dict] = mapped_column(JSON, default=dict)
    launcher_id: Mapped[str] = mapped_column(String(180), default="builtin.local-process")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    __table_args__ = (
        CheckConstraint(
            "source_kind IN ('model_version', 'quantization_run')",
            name="ck_release_job_source_kind",
        ),
        UniqueConstraint("project_id", "name", name="uq_project_release_job_name"),
    )


class ReleaseBuildRun(RecordMixin, Base):
    """One immutable execution of a ReleaseJob."""

    __tablename__ = "release_build_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    release_job_id: Mapped[str] = mapped_column(
        ForeignKey("release_jobs.id", ondelete="CASCADE"), index=True
    )
    run_number: Mapped[int] = mapped_column(Integer)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    release_name: Mapped[str] = mapped_column(String(180))
    release_version: Mapped[str] = mapped_column(String(80))
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    plugin_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    model_card_snapshot: Mapped[str] = mapped_column(Text, default="")
    license_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    validation_report: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("release_job_id", "run_number", name="uq_release_job_run_number"),
        UniqueConstraint("project_id", "release_version", name="uq_release_build_version"),
    )


class Release(RecordMixin, Base):
    __tablename__ = "releases"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    version: Mapped[str] = mapped_column(String(80))
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    quantization_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("quantization_runs.id"), nullable=True
    )
    target: Mapped[str] = mapped_column(String(100), default="local")
    model_card: Mapped[str] = mapped_column(Text, default="")
    license_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_result: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True)
    release_kind: Mapped[str] = mapped_column(
        String(40), default="legacy_manifest_only", index=True
    )
    release_build_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("release_build_runs.id"), nullable=True, unique=True, index=True
    )
    runtime_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    manifest_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    source_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="published")

    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_release_version"),)


class ReleaseDeliveryJob(RecordMixin, Base):
    """A reusable target delivery configuration for one immutable Release."""

    __tablename__ = "release_delivery_jobs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    release_id: Mapped[str] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(180))
    publisher_plugin_version_id: Mapped[str] = mapped_column(
        ForeignKey("plugin_versions.id"), index=True
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    resource_request: Mapped[dict] = mapped_column(JSON, default=dict)
    launcher_id: Mapped[str] = mapped_column(String(180), default="builtin.local-process")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    __table_args__ = (
        UniqueConstraint("release_id", "name", name="uq_release_delivery_job_name"),
    )


class ReleaseDeliveryRun(RecordMixin, Base):
    """One immutable delivery attempt to one configured target."""

    __tablename__ = "release_delivery_runs"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    delivery_job_id: Mapped[str] = mapped_column(
        ForeignKey("release_delivery_jobs.id", ondelete="CASCADE"), index=True
    )
    run_number: Mapped[int] = mapped_column(Integer)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    release_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    plugin_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    report_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id"), nullable=True, index=True
    )
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "delivery_job_id", "run_number", name="uq_release_delivery_job_run_number"
        ),
    )


class PlaygroundSession(RecordMixin, Base):
    __tablename__ = "playground_sessions"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    mode: Mapped[str] = mapped_column(String(40), default="single")
    model_version_ids: Mapped[list] = mapped_column(JSON, default=list)
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class InferenceRecord(RecordMixin, Base):
    __tablename__ = "inference_records"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("playground_sessions.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), unique=True, nullable=True)
    role: Mapped[str] = mapped_column(String(30), default="user")
    prompt: Mapped[str] = mapped_column(Text)
    rendered_prompt: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text, default="")
    token_ids: Mapped[list] = mapped_column(JSON, default=list)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str] = mapped_column(String(40), default="stop")
    performance: Mapped[dict] = mapped_column(JSON, default=dict)


class FeedbackCandidate(RecordMixin, Base):
    __tablename__ = "feedback_candidates"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    inference_record_id: Mapped[str] = mapped_column(
        ForeignKey("inference_records.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(50))
    preferred_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    rejected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    labels: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(30), default="candidate")


class WorkerNode(RecordMixin, Base):
    __tablename__ = "worker_nodes"

    name: Mapped[str] = mapped_column(String(180), unique=True)
    hostname: Mapped[str] = mapped_column(String(240))
    status: Mapped[str] = mapped_column(String(30), default="online", index=True)
    labels: Mapped[dict] = mapped_column(JSON, default=dict)
    resources: Mapped[dict] = mapped_column(JSON, default=dict)
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        view = super().to_dict()
        view.pop("token_hash", None)
        return view


class RemoteWorkerEnvironment(RecordMixin, Base):
    """Desired state for a Linux Worker managed over SSH."""

    __tablename__ = "remote_worker_environments"

    name: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    ssh_host: Mapped[str] = mapped_column(String(240), index=True)
    ssh_port: Mapped[int] = mapped_column(Integer, default=22)
    ssh_user: Mapped[str] = mapped_column(String(120))
    credential_ref: Mapped[str] = mapped_column(String(180))
    host_key: Mapped[str] = mapped_column(Text)
    host_key_fingerprint: Mapped[str] = mapped_column(String(160), index=True)
    control_plane_url: Mapped[str] = mapped_column(Text)
    workspace_root: Mapped[str] = mapped_column(Text, default="/var/lib/forgelm/workspaces")
    install_root: Mapped[str] = mapped_column(Text, default="/opt/forgelm-worker")
    profile: Mapped[str] = mapped_column(String(24), default="general", index=True)
    labels: Mapped[dict] = mapped_column(JSON, default=dict)
    desired_release: Mapped[str] = mapped_column(String(120), default="source", index=True)
    deployed_release: Mapped[str | None] = mapped_column(String(120), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(
        ForeignKey("worker_nodes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="registered", index=True)
    enrollment_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enrollment_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enrollment_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        CheckConstraint("ssh_port > 0 AND ssh_port <= 65535", name="ck_remote_worker_ssh_port"),
        CheckConstraint(
            "profile IN ('general', 'data')", name="ck_remote_worker_profile"
        ),
        CheckConstraint(
            "status IN ('registered', 'provisioning', 'online', 'offline', 'failed')",
            name="ck_remote_worker_status",
        ),
        UniqueConstraint(
            "ssh_host", "ssh_port", name="uq_remote_worker_environment_endpoint"
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        view = super().to_dict()
        view.pop("enrollment_token_hash", None)
        return view


class RemoteWorkerOperation(RecordMixin, Base):
    __tablename__ = "remote_worker_operations"

    environment_id: Mapped[str] = mapped_column(
        ForeignKey("remote_worker_environments.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(24), index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    phase: Mapped[str] = mapped_column(String(80), default="queued")
    logs: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "kind IN ('probe', 'provision', 'upgrade', 'repair')",
            name="ck_remote_worker_operation_kind",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_remote_worker_operation_status",
        ),
    )


class PluginVersion(RecordMixin, Base):
    __tablename__ = "plugin_versions"

    plugin_id: Mapped[str] = mapped_column(String(180), index=True)
    version: Mapped[str] = mapped_column(String(60))
    kind: Mapped[str] = mapped_column(String(60), index=True)
    entrypoint: Mapped[str] = mapped_column(String(300))
    protocol_version: Mapped[int] = mapped_column(Integer, default=1)
    config_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    runtime: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    support_level: Mapped[str] = mapped_column(String(30), default="experimental")

    __table_args__ = (UniqueConstraint("plugin_id", "version", name="uq_plugin_version"),)


class WorkerPluginInstallation(RecordMixin, Base):
    """Worker-local installed versions for one installable plugin ID."""

    __tablename__ = "worker_plugin_installations"

    worker_id: Mapped[str] = mapped_column(
        ForeignKey("worker_nodes.id", ondelete="CASCADE"), index=True
    )
    plugin_id: Mapped[str] = mapped_column(String(180), index=True)
    current_plugin_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("plugin_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    target_plugin_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("plugin_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    installed_versions: Mapped[list] = mapped_column(JSON, default=list)
    state: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    runtime: Mapped[dict] = mapped_column(JSON, default=dict)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    last_error: Mapped[dict] = mapped_column(JSON, default=dict)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("worker_id", "plugin_id", name="uq_worker_plugin_installation"),
    )


class ProjectPluginBinding(RecordMixin, Base):
    """A project opt-in to one exact installable plugin version."""

    __tablename__ = "project_plugin_bindings"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    plugin_id: Mapped[str] = mapped_column(String(180), index=True)
    plugin_version_id: Mapped[str] = mapped_column(
        ForeignKey("plugin_versions.id"), index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    default_config: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("project_id", "plugin_id", name="uq_project_plugin_binding"),
    )


class GateResult(RecordMixin, Base):
    __tablename__ = "gate_results"

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    subject_type: Mapped[str] = mapped_column(String(60))
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    gate_type: Mapped[str] = mapped_column(String(80))
    rule_version: Mapped[str] = mapped_column(String(40), default="1")
    status: Mapped[str] = mapped_column(String(30))
    issues: Mapped[list] = mapped_column(JSON, default=list)
    input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    waiver: Mapped[dict] = mapped_column(JSON, default=dict)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    actor: Mapped[str] = mapped_column(String(120), default="local-user")
    action: Mapped[str] = mapped_column(String(100), index=True)
    subject_type: Mapped[str] = mapped_column(String(60))
    subject_id: Mapped[str] = mapped_column(String(36))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


from . import suite_models  # noqa: E402, F401
