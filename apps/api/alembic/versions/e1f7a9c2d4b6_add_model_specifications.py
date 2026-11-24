"""add visual model specifications

Revision ID: e1f7a9c2d4b6
Revises: d8e4f1a2b6c9
Create Date: 2026-08-11 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f7a9c2d4b6"
down_revision: str | Sequence[str] | None = "d8e4f1a2b6c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_specifications",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("current_revision_id", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "name", name="uq_model_specification_project_name"
        ),
    )
    op.create_index(
        op.f("ix_model_specifications_project_id"),
        "model_specifications",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_specifications_status"),
        "model_specifications",
        ["status"],
        unique=False,
    )
    op.create_table(
        "model_specification_revisions",
        sa.Column("specification_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("architecture_family_id", sa.String(length=120), nullable=False),
        sa.Column("architecture_version", sa.String(length=40), nullable=False),
        sa.Column("preset_id", sa.String(length=120), nullable=True),
        sa.Column("code_version_id", sa.String(length=36), nullable=True),
        sa.Column("user_config", sa.JSON(), nullable=False),
        sa.Column("resolved_config", sa.JSON(), nullable=False),
        sa.Column("initialization", sa.JSON(), nullable=False),
        sa.Column("parameter_summary", sa.JSON(), nullable=False),
        sa.Column("runtime_requirements", sa.JSON(), nullable=False),
        sa.Column("generated_code", sa.Text(), nullable=False),
        sa.Column("config_signature", sa.String(length=64), nullable=False),
        sa.Column("locked", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["code_version_id"], ["code_versions.id"]),
        sa.ForeignKeyConstraint(
            ["specification_id"], ["model_specifications.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "specification_id", "revision", name="uq_model_specification_revision"
        ),
    )
    op.create_index(
        op.f("ix_model_specification_revisions_specification_id"),
        "model_specification_revisions",
        ["specification_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_specification_revisions_code_version_id"),
        "model_specification_revisions",
        ["code_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_specification_revisions_config_signature"),
        "model_specification_revisions",
        ["config_signature"],
        unique=False,
    )
    op.add_column(
        "plan_revisions",
        sa.Column("model_specification_revision_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_plan_revisions_model_specification_revision_id",
        "plan_revisions",
        ["model_specification_revision_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_plan_revisions_model_specification_revision_id", table_name="plan_revisions"
    )
    op.drop_column("plan_revisions", "model_specification_revision_id")
    op.drop_index(
        op.f("ix_model_specification_revisions_config_signature"),
        table_name="model_specification_revisions",
    )
    op.drop_index(
        op.f("ix_model_specification_revisions_code_version_id"),
        table_name="model_specification_revisions",
    )
    op.drop_index(
        op.f("ix_model_specification_revisions_specification_id"),
        table_name="model_specification_revisions",
    )
    op.drop_table("model_specification_revisions")
    op.drop_index(op.f("ix_model_specifications_status"), table_name="model_specifications")
    op.drop_index(
        op.f("ix_model_specifications_project_id"), table_name="model_specifications"
    )
    op.drop_table("model_specifications")
