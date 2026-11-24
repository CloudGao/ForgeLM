"""add immutable tokenized dataset versions

Revision ID: a9f3c7d2e1b4
Revises: e1f7a9c2d4b6
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9f3c7d2e1b4"
down_revision: str | Sequence[str] | None = "e1f7a9c2d4b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tokenized_dataset_versions",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("mixture_version_id", sa.String(length=36), nullable=False),
        sa.Column("tokenizer_version_id", sa.String(length=36), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("packing_version", sa.String(length=80), nullable=False),
        sa.Column("sequence_length", sa.Integer(), nullable=False),
        sa.Column("target_tokens", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("sequence_count", sa.Integer(), nullable=False),
        sa.Column("shard_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
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
        sa.UniqueConstraint(
            "project_id", "identity_hash", name="uq_tokenized_dataset_identity"
        ),
    )
    for column in (
        "project_id",
        "mixture_version_id",
        "tokenizer_version_id",
        "identity_hash",
        "status",
        "manifest_hash",
    ):
        op.create_index(
            op.f(f"ix_tokenized_dataset_versions_{column}"),
            "tokenized_dataset_versions",
            [column],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("tokenized_dataset_versions")
