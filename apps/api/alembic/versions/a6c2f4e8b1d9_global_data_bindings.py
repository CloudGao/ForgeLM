"""add global data catalog ownership and project bindings

Revision ID: a6c2f4e8b1d9
Revises: e4a1b7c9d2f3
Create Date: 2026-08-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6c2f4e8b1d9"
down_revision: str | Sequence[str] | None = "e4a1b7c9d2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.add_column(
            sa.Column(
                "system_kind",
                sa.String(length=40),
                nullable=False,
                server_default="",
            )
        )
        batch.create_index("ix_projects_system_kind", ["system_kind"])

    op.create_table(
        "project_data_bindings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("asset_type", sa.String(length=50), nullable=False),
        sa.Column("asset_version_id", sa.String(length=36), nullable=False),
        sa.Column("purpose", sa.String(length=50), nullable=False, server_default="training"),
        sa.Column("alias", sa.String(length=180), nullable=False, server_default=""),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.CheckConstraint(
            "asset_type IN ('dataset_version', 'mixture_version', "
            "'supervision_dataset_version', 'supervision_mixture_version', "
            "'eval_dataset_version')",
            name="ck_project_data_binding_asset_type",
        ),
        sa.CheckConstraint(
            "purpose IN ('training', 'validation', 'evaluation', 'reference')",
            name="ck_project_data_binding_purpose",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "asset_type",
            "asset_version_id",
            "purpose",
            name="uq_project_data_binding_ref",
        ),
    )
    op.create_index(
        "ix_project_data_bindings_project_id",
        "project_data_bindings",
        ["project_id"],
    )
    op.create_index(
        "ix_project_data_bindings_asset_type",
        "project_data_bindings",
        ["asset_type"],
    )
    op.create_index(
        "ix_project_data_bindings_asset_version_id",
        "project_data_bindings",
        ["asset_version_id"],
    )
    op.create_index(
        "ix_project_data_bindings_purpose",
        "project_data_bindings",
        ["purpose"],
    )
    op.create_index(
        "ix_project_data_bindings_asset_ref",
        "project_data_bindings",
        ["asset_type", "asset_version_id"],
    )


def downgrade() -> None:
    op.drop_table("project_data_bindings")
    with op.batch_alter_table("projects") as batch:
        batch.drop_index("ix_projects_system_kind")
        batch.drop_column("system_kind")
