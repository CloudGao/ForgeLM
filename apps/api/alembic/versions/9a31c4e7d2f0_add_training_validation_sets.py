"""add training validation sets

Revision ID: 9a31c4e7d2f0
Revises: 4f7d9a31c2b8
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9a31c4e7d2f0"
down_revision: str | Sequence[str] | None = "4f7d9a31c2b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_validation_sets",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("format", sa.String(length=20), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("estimated_tokens", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "name", "version", name="uq_training_validation_set_version"
        ),
    )
    op.create_index(
        "ix_training_validation_sets_project_id",
        "training_validation_sets",
        ["project_id"],
    )
    op.create_index(
        "ix_training_validation_sets_status", "training_validation_sets", ["status"]
    )
    op.create_index(
        "ix_training_validation_sets_artifact_id",
        "training_validation_sets",
        ["artifact_id"],
    )
    with op.batch_alter_table("plan_revisions") as batch:
        batch.add_column(sa.Column("validation_set_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_plan_revisions_validation_set_id_training_validation_sets",
            "training_validation_sets",
            ["validation_set_id"],
            ["id"],
        )
        batch.create_index("ix_plan_revisions_validation_set_id", ["validation_set_id"])


def downgrade() -> None:
    with op.batch_alter_table("plan_revisions") as batch:
        batch.drop_index("ix_plan_revisions_validation_set_id")
        batch.drop_constraint(
            "fk_plan_revisions_validation_set_id_training_validation_sets", type_="foreignkey"
        )
        batch.drop_column("validation_set_id")
    op.drop_index("ix_training_validation_sets_artifact_id", table_name="training_validation_sets")
    op.drop_index("ix_training_validation_sets_status", table_name="training_validation_sets")
    op.drop_index("ix_training_validation_sets_project_id", table_name="training_validation_sets")
    op.drop_table("training_validation_sets")
