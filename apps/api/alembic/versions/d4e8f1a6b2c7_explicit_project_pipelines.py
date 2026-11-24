"""add explicit versioned project pipelines

Revision ID: d4e8f1a6b2c7
Revises: c9e8f7a6b5d4
Create Date: 2026-08-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e8f1a6b2c7"
down_revision: str | Sequence[str] | None = "c9e8f7a6b5d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project_pipelines",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "objective", sa.String(length=40), nullable=False, server_default="pretraining"
        ),
        sa.Column("current_revision_id", sa.String(length=36), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.CheckConstraint(
            "objective IN ('pretraining', 'post_training', 'evaluation', 'release')",
            name="ck_project_pipeline_objective",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_project_pipeline_name"),
    )
    op.create_index(
        "ix_project_pipelines_project_id", "project_pipelines", ["project_id"]
    )
    op.create_index(
        "ix_project_pipelines_objective", "project_pipelines", ["objective"]
    )
    op.create_index(
        "ix_project_pipelines_archived", "project_pipelines", ["archived"]
    )

    op.create_table(
        "project_pipeline_revisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pipeline_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("data_refs", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("tokenizer_version_id", sa.String(length=36), nullable=True),
        sa.Column("code_version_id", sa.String(length=36), nullable=True),
        sa.Column("plan_revision_id", sa.String(length=36), nullable=True),
        sa.Column("training_run_id", sa.String(length=36), nullable=True),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("evaluation_run_id", sa.String(length=36), nullable=True),
        sa.Column("quantization_run_id", sa.String(length=36), nullable=True),
        sa.Column("release_id", sa.String(length=36), nullable=True),
        sa.Column("mapping_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("locked", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.ForeignKeyConstraint(
            ["pipeline_id"], ["project_pipelines.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["tokenizer_version_id"], ["tokenizer_versions.id"]),
        sa.ForeignKeyConstraint(["code_version_id"], ["code_versions.id"]),
        sa.ForeignKeyConstraint(["plan_revision_id"], ["plan_revisions.id"]),
        sa.ForeignKeyConstraint(["training_run_id"], ["training_runs.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["evaluation_runs.id"]),
        sa.ForeignKeyConstraint(["quantization_run_id"], ["quantization_runs.id"]),
        sa.ForeignKeyConstraint(["release_id"], ["releases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pipeline_id", "revision", name="uq_project_pipeline_revision"
        ),
    )
    for column in (
        "pipeline_id",
        "tokenizer_version_id",
        "code_version_id",
        "plan_revision_id",
        "training_run_id",
        "model_version_id",
        "evaluation_run_id",
        "quantization_run_id",
        "release_id",
    ):
        op.create_index(
            f"ix_project_pipeline_revisions_{column}",
            "project_pipeline_revisions",
            [column],
        )


def downgrade() -> None:
    op.drop_table("project_pipeline_revisions")
    op.drop_table("project_pipelines")
