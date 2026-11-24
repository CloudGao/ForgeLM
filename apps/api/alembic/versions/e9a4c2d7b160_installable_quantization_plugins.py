"""add installable quantization plugins and reusable jobs

Revision ID: e9a4c2d7b160
Revises: d7b2f4a91e30
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e9a4c2d7b160"
down_revision: str | Sequence[str] | None = "d7b2f4a91e30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "worker_plugin_installations",
        sa.Column("worker_id", sa.String(length=36), nullable=False),
        sa.Column("plugin_id", sa.String(length=180), nullable=False),
        sa.Column("current_plugin_version_id", sa.String(length=36), nullable=True),
        sa.Column("target_plugin_version_id", sa.String(length=36), nullable=True),
        sa.Column("installed_versions", sa.JSON(), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("runtime", sa.JSON(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("last_error", sa.JSON(), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["current_plugin_version_id"], ["plugin_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["target_plugin_version_id"], ["plugin_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["worker_id"], ["worker_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("worker_id", "plugin_id", name="uq_worker_plugin_installation"),
    )
    for name, columns in (
        ("ix_worker_plugin_installations_worker_id", ["worker_id"]),
        ("ix_worker_plugin_installations_plugin_id", ["plugin_id"]),
        (
            "ix_worker_plugin_installations_current_plugin_version_id",
            ["current_plugin_version_id"],
        ),
        (
            "ix_worker_plugin_installations_target_plugin_version_id",
            ["target_plugin_version_id"],
        ),
        ("ix_worker_plugin_installations_state", ["state"]),
        ("ix_worker_plugin_installations_task_id", ["task_id"]),
    ):
        op.create_index(name, "worker_plugin_installations", columns, unique=False)

    op.create_table(
        "project_plugin_bindings",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("plugin_id", sa.String(length=180), nullable=False),
        sa.Column("plugin_version_id", sa.String(length=36), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("default_config", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plugin_version_id"], ["plugin_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "plugin_id", name="uq_project_plugin_binding"),
    )
    for name, columns in (
        ("ix_project_plugin_bindings_project_id", ["project_id"]),
        ("ix_project_plugin_bindings_plugin_id", ["plugin_id"]),
        ("ix_project_plugin_bindings_plugin_version_id", ["plugin_version_id"]),
        ("ix_project_plugin_bindings_enabled", ["enabled"]),
    ):
        op.create_index(name, "project_plugin_bindings", columns, unique=False)

    op.create_table(
        "quantization_jobs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("plugin_version_id", sa.String(length=36), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("calibration_data_ref", sa.JSON(), nullable=False),
        sa.Column("resource_request", sa.JSON(), nullable=False),
        sa.Column("launcher_id", sa.String(length=180), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["plugin_version_id"], ["plugin_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_project_quantization_job_name"),
    )
    for name, columns in (
        ("ix_quantization_jobs_project_id", ["project_id"]),
        ("ix_quantization_jobs_model_version_id", ["model_version_id"]),
        ("ix_quantization_jobs_plugin_version_id", ["plugin_version_id"]),
        ("ix_quantization_jobs_enabled", ["enabled"]),
    ):
        op.create_index(name, "quantization_jobs", columns, unique=False)

    with op.batch_alter_table("quantization_runs") as batch_op:
        batch_op.add_column(
            sa.Column("quantization_job_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(sa.Column("run_number", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("plugin_version_id", sa.String(length=36), nullable=True))
        batch_op.add_column(
            sa.Column("trigger_kind", sa.String(length=30), nullable=False, server_default="legacy")
        )
        batch_op.add_column(
            sa.Column("config_snapshot", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(
            sa.Column("source_snapshot", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_quantization_runs_quantization_job_id",
            "quantization_jobs",
            ["quantization_job_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_quantization_runs_plugin_version_id",
            "plugin_versions",
            ["plugin_version_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_quantization_runs_quantization_job_id", ["quantization_job_id"], unique=False
        )
        batch_op.create_index(
            "ix_quantization_runs_plugin_version_id", ["plugin_version_id"], unique=False
        )
        batch_op.create_unique_constraint(
            "uq_quantization_job_run_number", ["quantization_job_id", "run_number"]
        )


def downgrade() -> None:
    with op.batch_alter_table("quantization_runs") as batch_op:
        batch_op.drop_constraint("uq_quantization_job_run_number", type_="unique")
        batch_op.drop_index("ix_quantization_runs_plugin_version_id")
        batch_op.drop_index("ix_quantization_runs_quantization_job_id")
        batch_op.drop_constraint(
            "fk_quantization_runs_plugin_version_id", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_quantization_runs_quantization_job_id", type_="foreignkey"
        )
        batch_op.drop_column("ended_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("source_snapshot")
        batch_op.drop_column("config_snapshot")
        batch_op.drop_column("trigger_kind")
        batch_op.drop_column("plugin_version_id")
        batch_op.drop_column("run_number")
        batch_op.drop_column("quantization_job_id")
    op.drop_table("quantization_jobs")
    op.drop_table("project_plugin_bindings")
    op.drop_table("worker_plugin_installations")
