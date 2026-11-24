"""add project pipeline DAG definitions and runs

Revision ID: e7f2a9c4d6b1
Revises: d4e8f1a6b2c7
Create Date: 2026-08-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7f2a9c4d6b1"
down_revision: str | Sequence[str] | None = "d4e8f1a6b2c7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("project_pipelines") as batch_op:
        batch_op.add_column(sa.Column("draft_revision_id", sa.String(length=36), nullable=True))

    with op.batch_alter_table("project_pipeline_revisions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "status",
                sa.String(length=24),
                nullable=False,
                server_default="published",
            )
        )
        batch_op.add_column(sa.Column("nodes", sa.JSON(), nullable=False, server_default="[]"))
        batch_op.add_column(sa.Column("edges", sa.JSON(), nullable=False, server_default="[]"))
        batch_op.add_column(sa.Column("variables", sa.JSON(), nullable=False, server_default="{}"))
        batch_op.add_column(sa.Column("viewport", sa.JSON(), nullable=False, server_default="{}"))
        batch_op.add_column(
            sa.Column("validation_report", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(sa.Column("content_hash", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_check_constraint(
            "ck_project_pipeline_revision_status",
            "status IN ('draft', 'published')",
        )
        batch_op.create_index(
            "ix_project_pipeline_revisions_status", ["status"], unique=False
        )
        batch_op.create_index(
            "ix_project_pipeline_revisions_content_hash", ["content_hash"], unique=False
        )

    op.execute(
        "UPDATE project_pipeline_revisions "
        "SET status = 'published', published_at = created_at WHERE locked = 1"
    )

    op.create_table(
        "project_pipeline_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_id", sa.String(length=36), nullable=False),
        sa.Column("revision_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="queued"),
        sa.Column("trigger_kind", sa.String(length=40), nullable=False, server_default="manual"),
        sa.Column("input_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("context_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("node_summary", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'cancelling', 'succeeded', "
            "'failed', 'cancelled')",
            name="ck_project_pipeline_run_status",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["pipeline_id"], ["project_pipelines.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["revision_id"], ["project_pipeline_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_project_pipeline_runs_project_id", "project_pipeline_runs", ["project_id"])
    op.create_index("ix_project_pipeline_runs_pipeline_id", "project_pipeline_runs", ["pipeline_id"])
    op.create_index("ix_project_pipeline_runs_revision_id", "project_pipeline_runs", ["revision_id"])
    op.create_index("ix_project_pipeline_runs_status", "project_pipeline_runs", ["status"])

    op.create_table(
        "project_pipeline_node_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pipeline_run_id", sa.String(length=36), nullable=False),
        sa.Column("node_id", sa.String(length=120), nullable=False),
        sa.Column("node_type", sa.String(length=40), nullable=False),
        sa.Column("label", sa.String(length=180), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="waiting"),
        sa.Column("config_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("input_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("output_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("branch_decision", sa.String(length=120), nullable=True),
        sa.Column("error", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('waiting', 'queued', 'running', 'paused', 'succeeded', "
            "'failed', 'cancelled', 'skipped')",
            name="ck_project_pipeline_node_run_status",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"], ["project_pipeline_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint(
            "pipeline_run_id", "node_id", name="uq_project_pipeline_node_run"
        ),
    )
    op.create_index(
        "ix_project_pipeline_node_runs_pipeline_run_id",
        "project_pipeline_node_runs",
        ["pipeline_run_id"],
    )
    op.create_index(
        "ix_project_pipeline_node_runs_node_type",
        "project_pipeline_node_runs",
        ["node_type"],
    )
    op.create_index(
        "ix_project_pipeline_node_runs_status",
        "project_pipeline_node_runs",
        ["status"],
    )
    op.create_index(
        "ix_project_pipeline_node_runs_task_id",
        "project_pipeline_node_runs",
        ["task_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("project_pipeline_node_runs")
    op.drop_table("project_pipeline_runs")
    with op.batch_alter_table("project_pipeline_revisions") as batch_op:
        batch_op.drop_index("ix_project_pipeline_revisions_content_hash")
        batch_op.drop_index("ix_project_pipeline_revisions_status")
        batch_op.drop_constraint("ck_project_pipeline_revision_status", type_="check")
        batch_op.drop_column("published_at")
        batch_op.drop_column("content_hash")
        batch_op.drop_column("validation_report")
        batch_op.drop_column("viewport")
        batch_op.drop_column("variables")
        batch_op.drop_column("edges")
        batch_op.drop_column("nodes")
        batch_op.drop_column("status")
    with op.batch_alter_table("project_pipelines") as batch_op:
        batch_op.drop_column("draft_revision_id")
