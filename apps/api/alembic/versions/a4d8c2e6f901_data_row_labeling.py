"""add versioned row labeling

Revision ID: a4d8c2e6f901
Revises: f1c7a4d9e2b3
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d8c2e6f901"
down_revision: str | Sequence[str] | None = "f1c7a4d9e2b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _record_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def _indexes(table: str, names: list[str], *, unique: set[str] | None = None) -> None:
    unique = unique or set()
    for name in names:
        op.create_index(f"ix_{table}_{name}", table, [name], unique=name in unique)


def upgrade() -> None:
    op.create_table(
        "label_schemes",
        sa.Column("code", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("modality", sa.String(length=24), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        *_record_columns(),
        sa.CheckConstraint(
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
            name="ck_label_scheme_modality",
        ),
        sa.CheckConstraint("status IN ('active', 'archived')", name="ck_label_scheme_status"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("modality", "code", name="uq_label_scheme_modality_code"),
    )
    _indexes("label_schemes", ["modality", "status"])

    op.create_table(
        "label_scheme_versions",
        sa.Column("scheme_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=180), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        *_record_columns(),
        sa.ForeignKeyConstraint(["scheme_id"], ["label_schemes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scheme_id", "version", name="uq_label_scheme_version"),
    )
    _indexes("label_scheme_versions", ["scheme_id"])

    op.create_table(
        "label_groups",
        sa.Column("scheme_version_id", sa.String(length=36), nullable=False),
        sa.Column("code", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("selection_mode", sa.String(length=16), nullable=False),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        *_record_columns(),
        sa.CheckConstraint(
            "selection_mode IN ('single', 'multiple')", name="ck_label_group_selection_mode"
        ),
        sa.ForeignKeyConstraint(
            ["scheme_version_id"], ["label_scheme_versions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scheme_version_id", "code", name="uq_label_group_version_code"),
    )
    _indexes("label_groups", ["scheme_version_id"])

    op.create_table(
        "label_definitions",
        sa.Column("group_id", sa.String(length=36), nullable=False),
        sa.Column("code", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("color", sa.String(length=32), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        *_record_columns(),
        sa.ForeignKeyConstraint(["group_id"], ["label_groups.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("group_id", "code", name="uq_label_definition_group_code"),
    )
    _indexes("label_definitions", ["group_id"])

    op.create_table(
        "dataset_label_operations",
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("scheme_version_id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("base_snapshot_id", sa.String(length=36), nullable=True),
        sa.Column("output_snapshot_id", sa.String(length=36), nullable=True),
        sa.Column("source_query_task_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("label_ids", sa.JSON(), nullable=False),
        sa.Column("query_snapshot", sa.JSON(), nullable=False),
        sa.Column("annotator", sa.String(length=180), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("selection_count", sa.Integer(), nullable=False),
        sa.Column("selection_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("delta_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("error", sa.JSON(), nullable=False),
        *_record_columns(),
        sa.CheckConstraint("scope IN ('selected', 'page', 'query')", name="ck_label_operation_scope"),
        sa.CheckConstraint("action IN ('apply', 'remove')", name="ck_label_operation_action"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["scheme_version_id"], ["label_scheme_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["source_query_task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["selection_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["delta_artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
    )
    _indexes(
        "dataset_label_operations",
        [
            "dataset_version_id", "scheme_version_id", "task_id", "base_snapshot_id",
            "output_snapshot_id", "source_query_task_id", "status",
        ],
        unique={"task_id"},
    )

    op.create_table(
        "dataset_label_snapshots",
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("parent_snapshot_id", sa.String(length=36), nullable=True),
        sa.Column("operation_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("assignment_count", sa.Integer(), nullable=False),
        sa.Column("labeled_record_count", sa.Integer(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        *_record_columns(),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["parent_snapshot_id"], ["dataset_label_snapshots.id"]),
        sa.ForeignKeyConstraint(
            ["operation_id"], ["dataset_label_operations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_id"),
        sa.UniqueConstraint(
            "dataset_version_id", "version", name="uq_dataset_label_snapshot_version"
        ),
    )
    _indexes(
        "dataset_label_snapshots",
        ["dataset_version_id", "parent_snapshot_id", "operation_id", "artifact_id", "content_hash"],
        unique={"operation_id"},
    )


def downgrade() -> None:
    op.drop_table("dataset_label_snapshots")
    op.drop_table("dataset_label_operations")
    op.drop_table("label_definitions")
    op.drop_table("label_groups")
    op.drop_table("label_scheme_versions")
    op.drop_table("label_schemes")
