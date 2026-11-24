"""add pluginized release build and delivery pipeline

Revision ID: f1c7a4d9e2b3
Revises: e9a4c2d7b160
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f1c7a4d9e2b3"
down_revision: str | Sequence[str] | None = "e9a4c2d7b160"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _record_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "release_jobs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.String(length=30), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("exporter_plugin_version_id", sa.String(length=36), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("gate_policy", sa.JSON(), nullable=False),
        sa.Column("model_card_template", sa.Text(), nullable=False),
        sa.Column("license_policy", sa.JSON(), nullable=False),
        sa.Column("resource_request", sa.JSON(), nullable=False),
        sa.Column("launcher_id", sa.String(length=180), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        *_record_columns(),
        sa.CheckConstraint(
            "source_kind IN ('model_version', 'quantization_run')",
            name="ck_release_job_source_kind",
        ),
        sa.ForeignKeyConstraint(["exporter_plugin_version_id"], ["plugin_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_project_release_job_name"),
    )
    for name, columns in (
        ("ix_release_jobs_project_id", ["project_id"]),
        ("ix_release_jobs_source_kind", ["source_kind"]),
        ("ix_release_jobs_source_id", ["source_id"]),
        ("ix_release_jobs_exporter_plugin_version_id", ["exporter_plugin_version_id"]),
        ("ix_release_jobs_enabled", ["enabled"]),
    ):
        op.create_index(name, "release_jobs", columns, unique=False)

    op.create_table(
        "release_build_runs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("release_job_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("release_name", sa.String(length=180), nullable=False),
        sa.Column("release_version", sa.String(length=80), nullable=False),
        sa.Column("source_snapshot", sa.JSON(), nullable=False),
        sa.Column("plugin_snapshot", sa.JSON(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("gate_snapshot", sa.JSON(), nullable=False),
        sa.Column("model_card_snapshot", sa.Text(), nullable=False),
        sa.Column("license_snapshot", sa.JSON(), nullable=False),
        sa.Column("runtime_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("validation_report", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        *_record_columns(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["release_job_id"], ["release_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["runtime_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint(
            "release_job_id", "run_number", name="uq_release_job_run_number"
        ),
        sa.UniqueConstraint(
            "project_id", "release_version", name="uq_release_build_version"
        ),
    )
    for name, columns in (
        ("ix_release_build_runs_project_id", ["project_id"]),
        ("ix_release_build_runs_release_job_id", ["release_job_id"]),
        ("ix_release_build_runs_task_id", ["task_id"]),
        ("ix_release_build_runs_status", ["status"]),
        ("ix_release_build_runs_runtime_artifact_id", ["runtime_artifact_id"]),
    ):
        op.create_index(name, "release_build_runs", columns, unique=False)

    with op.batch_alter_table("releases") as batch_op:
        batch_op.add_column(
            sa.Column(
                "release_kind",
                sa.String(length=40),
                nullable=False,
                server_default="legacy_manifest_only",
            )
        )
        batch_op.add_column(sa.Column("release_build_run_id", sa.String(length=36)))
        batch_op.add_column(sa.Column("runtime_artifact_id", sa.String(length=36)))
        batch_op.add_column(sa.Column("manifest_artifact_id", sa.String(length=36)))
        batch_op.add_column(
            sa.Column("source_snapshot", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(
            sa.Column("runtime_snapshot", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.create_foreign_key(
            "fk_releases_release_build_run_id",
            "release_build_runs",
            ["release_build_run_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_releases_runtime_artifact_id",
            "artifacts",
            ["runtime_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_releases_manifest_artifact_id",
            "artifacts",
            ["manifest_artifact_id"],
            ["id"],
        )
        batch_op.create_unique_constraint(
            "uq_releases_release_build_run_id", ["release_build_run_id"]
        )
        batch_op.create_index("ix_releases_release_kind", ["release_kind"], unique=False)
        batch_op.create_index(
            "ix_releases_release_build_run_id", ["release_build_run_id"], unique=True
        )
        batch_op.create_index(
            "ix_releases_runtime_artifact_id", ["runtime_artifact_id"], unique=False
        )
        batch_op.create_index(
            "ix_releases_manifest_artifact_id", ["manifest_artifact_id"], unique=False
        )
    op.execute(
        "UPDATE releases SET manifest_artifact_id = artifact_id "
        "WHERE manifest_artifact_id IS NULL"
    )

    op.create_table(
        "release_delivery_jobs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("release_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("publisher_plugin_version_id", sa.String(length=36), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("resource_request", sa.JSON(), nullable=False),
        sa.Column("launcher_id", sa.String(length=180), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        *_record_columns(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["release_id"], ["releases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["publisher_plugin_version_id"], ["plugin_versions.id"]
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("release_id", "name", name="uq_release_delivery_job_name"),
    )
    for name, columns in (
        ("ix_release_delivery_jobs_project_id", ["project_id"]),
        ("ix_release_delivery_jobs_release_id", ["release_id"]),
        (
            "ix_release_delivery_jobs_publisher_plugin_version_id",
            ["publisher_plugin_version_id"],
        ),
        ("ix_release_delivery_jobs_enabled", ["enabled"]),
    ):
        op.create_index(name, "release_delivery_jobs", columns, unique=False)

    op.create_table(
        "release_delivery_runs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("delivery_job_id", sa.String(length=36), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("release_snapshot", sa.JSON(), nullable=False),
        sa.Column("artifact_snapshot", sa.JSON(), nullable=False),
        sa.Column("plugin_snapshot", sa.JSON(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("report_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        *_record_columns(),
        sa.ForeignKeyConstraint(
            ["delivery_job_id"], ["release_delivery_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["report_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id"),
        sa.UniqueConstraint(
            "delivery_job_id",
            "run_number",
            name="uq_release_delivery_job_run_number",
        ),
    )
    for name, columns in (
        ("ix_release_delivery_runs_project_id", ["project_id"]),
        ("ix_release_delivery_runs_delivery_job_id", ["delivery_job_id"]),
        ("ix_release_delivery_runs_task_id", ["task_id"]),
        ("ix_release_delivery_runs_status", ["status"]),
        ("ix_release_delivery_runs_report_artifact_id", ["report_artifact_id"]),
    ):
        op.create_index(name, "release_delivery_runs", columns, unique=False)


def downgrade() -> None:
    op.drop_table("release_delivery_runs")
    op.drop_table("release_delivery_jobs")
    with op.batch_alter_table("releases") as batch_op:
        batch_op.drop_index("ix_releases_manifest_artifact_id")
        batch_op.drop_index("ix_releases_runtime_artifact_id")
        batch_op.drop_index("ix_releases_release_build_run_id")
        batch_op.drop_index("ix_releases_release_kind")
        batch_op.drop_constraint("uq_releases_release_build_run_id", type_="unique")
        batch_op.drop_constraint("fk_releases_manifest_artifact_id", type_="foreignkey")
        batch_op.drop_constraint("fk_releases_runtime_artifact_id", type_="foreignkey")
        batch_op.drop_constraint("fk_releases_release_build_run_id", type_="foreignkey")
        batch_op.drop_column("runtime_snapshot")
        batch_op.drop_column("source_snapshot")
        batch_op.drop_column("manifest_artifact_id")
        batch_op.drop_column("runtime_artifact_id")
        batch_op.drop_column("release_build_run_id")
        batch_op.drop_column("release_kind")
    op.drop_table("release_build_runs")
    op.drop_table("release_jobs")
