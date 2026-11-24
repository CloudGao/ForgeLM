"""add tokenized corpus and training data view versions

Revision ID: d3f8a1c5b790
Revises: a9f3c7d2e1b4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3f8a1c5b790"
down_revision: str | Sequence[str] | None = "a9f3c7d2e1b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tokenized_corpus_versions",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("tokenizer_version_id", sa.String(length=36), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("selection_config", sa.JSON(), nullable=False),
        sa.Column("encoding_config", sa.JSON(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("shard_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["tokenizer_version_id"], ["tokenizer_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "identity_hash", name="uq_tokenized_corpus_identity"),
    )
    for column in (
        "project_id", "dataset_version_id", "tokenizer_version_id", "identity_hash",
        "status", "manifest_hash",
    ):
        op.create_index(
            op.f(f"ix_tokenized_corpus_versions_{column}"),
            "tokenized_corpus_versions",
            [column],
            unique=False,
        )

    op.create_table(
        "training_data_view_versions",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("mixture_version_id", sa.String(length=36), nullable=False),
        sa.Column("tokenizer_version_id", sa.String(length=36), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("sequence_length", sa.Integer(), nullable=False),
        sa.Column("requested_tokens", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("sequence_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["mixture_version_id"], ["data_mixture_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["tokenizer_version_id"], ["tokenizer_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "identity_hash", name="uq_training_data_view_identity"),
    )
    for column in (
        "project_id", "mixture_version_id", "tokenizer_version_id", "identity_hash",
        "status", "manifest_hash",
    ):
        op.create_index(
            op.f(f"ix_training_data_view_versions_{column}"),
            "training_data_view_versions",
            [column],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("training_data_view_versions")
    op.drop_table("tokenized_corpus_versions")
