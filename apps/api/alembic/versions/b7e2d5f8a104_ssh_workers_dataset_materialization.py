"""add SSH workers and dataset materialization

Revision ID: b7e2d5f8a104
Revises: a4d8c2e6f901
Create Date: 2026-08-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e2d5f8a104"
down_revision: str | Sequence[str] | None = "a4d8c2e6f901"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _record_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def _indexes(table: str, names: list[str]) -> None:
    for name in names:
        op.create_index(f"ix_{table}_{name}", table, [name], unique=False)


def upgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch:
        batch.add_column(
            sa.Column(
                "materialization_kind",
                sa.String(length=16),
                nullable=False,
                server_default="imported",
            )
        )
        batch.add_column(
            sa.Column("materialization_spec", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column(
                "materialization_state",
                sa.String(length=24),
                nullable=False,
                server_default="pending",
            )
        )
        batch.add_column(sa.Column("primary_location_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_dataset_versions_primary_location_id_artifact_locations",
            "artifact_locations",
            ["primary_location_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_check_constraint(
            "ck_dataset_version_materialization_kind",
            "materialization_kind IN ('download', 'derived', 'imported')",
        )
        batch.create_check_constraint(
            "ck_dataset_version_materialization_state",
            "materialization_state IN ('pending', 'ready', 'degraded')",
        )
    _indexes(
        "dataset_versions",
        ["materialization_kind", "materialization_state", "primary_location_id"],
    )

    op.execute(
        """
        UPDATE dataset_versions
        SET materialization_kind = CASE
            WHEN parent_version_id IS NOT NULL
              OR conversion_recipe_version_id IS NOT NULL
              OR processing_recipe_version_id IS NOT NULL THEN 'derived'
            WHEN EXISTS (
                SELECT 1 FROM datasets
                WHERE datasets.id = dataset_versions.dataset_id
                  AND datasets.source_type IN (
                    'http', 'https', 'huggingface', 'git'
                  )
            ) THEN 'download'
            ELSE 'imported'
        END
        """
    )
    op.execute(
        """
        UPDATE dataset_versions
        SET primary_location_id = (
            SELECT artifact_locations.id
            FROM artifact_locations
            WHERE artifact_locations.artifact_id = dataset_versions.artifact_id
              AND artifact_locations.availability_state = 'available'
              AND (
                artifact_locations.project_id = (
                  SELECT datasets.project_id FROM datasets
                  WHERE datasets.id = dataset_versions.dataset_id
                )
                OR (
                  artifact_locations.project_id IS NULL
                  AND artifact_locations.storage_type = 'external'
                )
              )
            ORDER BY
              CASE artifact_locations.storage_type WHEN 'managed' THEN 0 ELSE 1 END,
              artifact_locations.created_at,
              artifact_locations.id
            LIMIT 1
        )
        WHERE artifact_id IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE dataset_versions
        SET materialization_state = CASE
            WHEN primary_location_id IS NOT NULL THEN 'ready'
            WHEN artifact_id IS NOT NULL THEN 'degraded'
            ELSE 'pending'
        END
        """
    )

    op.create_table(
        "dataset_materializations",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("target_worker_id", sa.String(length=36), nullable=False),
        sa.Column("selection_mode", sa.String(length=16), nullable=False),
        sa.Column("requested_route_id", sa.String(length=160), nullable=True),
        sa.Column("candidate_routes", sa.JSON(), nullable=False),
        sa.Column("chosen_route", sa.JSON(), nullable=False),
        sa.Column("expected_content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("result_location_id", sa.String(length=36), nullable=True),
        sa.Column("error", sa.JSON(), nullable=False),
        *_record_columns(),
        sa.CheckConstraint(
            "selection_mode IN ('auto', 'manual')",
            name="ck_dataset_materialization_selection_mode",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_dataset_materialization_status",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["target_worker_id"], ["worker_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["result_location_id"], ["artifact_locations.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexes(
        "dataset_materializations",
        [
            "project_id",
            "dataset_version_id",
            "target_worker_id",
            "selection_mode",
            "expected_content_hash",
            "status",
            "task_id",
            "result_location_id",
        ],
    )

    op.create_table(
        "remote_worker_environments",
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("ssh_host", sa.String(length=240), nullable=False),
        sa.Column("ssh_port", sa.Integer(), nullable=False),
        sa.Column("ssh_user", sa.String(length=120), nullable=False),
        sa.Column("credential_ref", sa.String(length=180), nullable=False),
        sa.Column("host_key", sa.Text(), nullable=False),
        sa.Column("host_key_fingerprint", sa.String(length=160), nullable=False),
        sa.Column("control_plane_url", sa.Text(), nullable=False),
        sa.Column("workspace_root", sa.Text(), nullable=False),
        sa.Column("install_root", sa.Text(), nullable=False),
        sa.Column("profile", sa.String(length=24), nullable=False),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("desired_release", sa.String(length=120), nullable=False),
        sa.Column("deployed_release", sa.String(length=120), nullable=True),
        sa.Column("worker_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("enrollment_token_hash", sa.String(length=64), nullable=True),
        sa.Column("enrollment_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enrollment_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *_record_columns(),
        sa.CheckConstraint("ssh_port > 0 AND ssh_port <= 65535", name="ck_remote_worker_ssh_port"),
        sa.CheckConstraint("profile IN ('general', 'data')", name="ck_remote_worker_profile"),
        sa.CheckConstraint(
            "status IN ('registered', 'provisioning', 'online', 'offline', 'failed')",
            name="ck_remote_worker_status",
        ),
        sa.ForeignKeyConstraint(["worker_id"], ["worker_nodes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("ssh_host", "ssh_port", name="uq_remote_worker_environment_endpoint"),
    )
    _indexes(
        "remote_worker_environments",
        [
            "name",
            "ssh_host",
            "host_key_fingerprint",
            "profile",
            "desired_release",
            "worker_id",
            "status",
        ],
    )

    op.create_table(
        "remote_worker_operations",
        sa.Column("environment_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("phase", sa.String(length=80), nullable=False),
        sa.Column("logs", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        *_record_columns(),
        sa.CheckConstraint(
            "kind IN ('probe', 'provision', 'upgrade', 'repair')",
            name="ck_remote_worker_operation_kind",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_remote_worker_operation_status",
        ),
        sa.ForeignKeyConstraint(
            ["environment_id"], ["remote_worker_environments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    _indexes("remote_worker_operations", ["environment_id", "kind", "status"])


def downgrade() -> None:
    op.drop_table("remote_worker_operations")
    op.drop_table("remote_worker_environments")
    op.drop_table("dataset_materializations")
    op.drop_index("ix_dataset_versions_primary_location_id", table_name="dataset_versions")
    op.drop_index("ix_dataset_versions_materialization_state", table_name="dataset_versions")
    op.drop_index("ix_dataset_versions_materialization_kind", table_name="dataset_versions")
    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_constraint(
            "fk_dataset_versions_primary_location_id_artifact_locations", type_="foreignkey"
        )
        batch.drop_constraint("ck_dataset_version_materialization_state", type_="check")
        batch.drop_constraint("ck_dataset_version_materialization_kind", type_="check")
        batch.drop_column("primary_location_id")
        batch.drop_column("materialization_state")
        batch.drop_column("materialization_spec")
        batch.drop_column("materialization_kind")
