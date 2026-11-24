"""add global model weight library

Revision ID: d7b2f4a91e30
Revises: c6a1e7d4b290
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7b2f4a91e30"
down_revision: str | Sequence[str] | None = "c6a1e7d4b290"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_weight_assets",
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_provider", sa.String(length=40), nullable=False),
        sa.Column("repository_id", sa.String(length=500), nullable=False),
        sa.Column("license", sa.String(length=120), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_provider", "repository_id", name="uq_model_weight_asset_source"
        ),
    )
    op.create_index(
        op.f("ix_model_weight_assets_name"), "model_weight_assets", ["name"], unique=False
    )
    op.create_index(
        op.f("ix_model_weight_assets_source_provider"),
        "model_weight_assets",
        ["source_provider"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_weight_assets_repository_id"),
        "model_weight_assets",
        ["repository_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_weight_assets_archived"),
        "model_weight_assets",
        ["archived"],
        unique=False,
    )
    op.create_table(
        "model_weight_versions",
        sa.Column("asset_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("requested_revision", sa.String(length=180), nullable=False),
        sa.Column("resolved_commit", sa.String(length=64), nullable=False),
        sa.Column("source_endpoint", sa.Text(), nullable=False),
        sa.Column("selected_paths", sa.JSON(), nullable=False),
        sa.Column("source_manifest", sa.JSON(), nullable=False),
        sa.Column("format_summary", sa.JSON(), nullable=False),
        sa.Column("runtime_capabilities", sa.JSON(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        sa.Column("quantization", sa.String(length=80), nullable=True),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["asset_id"], ["model_weight_assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("asset_id", "version", name="uq_model_weight_asset_version"),
    )
    for name, columns, unique in (
        ("ix_model_weight_versions_asset_id", ["asset_id"], False),
        ("ix_model_weight_versions_identity_hash", ["identity_hash"], True),
        ("ix_model_weight_versions_resolved_commit", ["resolved_commit"], False),
        ("ix_model_weight_versions_state", ["state"], False),
        ("ix_model_weight_versions_artifact_id", ["artifact_id"], False),
        ("ix_model_weight_versions_task_id", ["task_id"], False),
    ):
        op.create_index(name, "model_weight_versions", columns, unique=unique)
    with op.batch_alter_table("model_versions") as batch_op:
        batch_op.add_column(sa.Column("model_weight_version_id", sa.String(length=36), nullable=True))
        batch_op.create_index(
            "ix_model_versions_model_weight_version_id", ["model_weight_version_id"], unique=False
        )
        batch_op.create_foreign_key(
            "fk_model_versions_model_weight_version_id",
            "model_weight_versions",
            ["model_weight_version_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    with op.batch_alter_table("model_versions") as batch_op:
        batch_op.drop_constraint("fk_model_versions_model_weight_version_id", type_="foreignkey")
        batch_op.drop_index("ix_model_versions_model_weight_version_id")
        batch_op.drop_column("model_weight_version_id")
    op.drop_table("model_weight_versions")
    op.drop_table("model_weight_assets")
