"""stable data mixture manifests

Revision ID: c8a4e21f9b7d
Revises: 8f4c2d91a6b3
Create Date: 2026-08-07 16:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8a4e21f9b7d"
down_revision: str | Sequence[str] | None = "8f4c2d91a6b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch:
        batch.add_column(sa.Column("record_manifest_artifact_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_dataset_versions_record_manifest_artifact_id_artifacts",
            "artifacts",
            ["record_manifest_artifact_id"],
            ["id"],
        )

    with op.batch_alter_table("data_mixture_versions") as batch:
        batch.add_column(
            sa.Column(
                "status",
                sa.String(length=40),
                nullable=False,
                server_default="legacy_unresolved",
            )
        )
        batch.add_column(sa.Column("seed", sa.Integer(), nullable=False, server_default="42"))
        batch.add_column(
            sa.Column(
                "resolver_version",
                sa.String(length=80),
                nullable=False,
                server_default="stable_hash_selector_v1",
            )
        )
        batch.add_column(
            sa.Column(
                "sampler_version",
                sa.String(length=80),
                nullable=False,
                server_default="weighted_hash_sampler_v1",
            )
        )
        batch.add_column(sa.Column("manifest_artifact_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("manifest_hash", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("resolved_document_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("resolved_token_estimate", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("task_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("failure_reason", sa.Text(), nullable=True))
        batch.create_foreign_key(
            "fk_data_mixture_versions_manifest_artifact_id_artifacts",
            "artifacts",
            ["manifest_artifact_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_data_mixture_versions_task_id_tasks", "tasks", ["task_id"], ["id"]
        )
        batch.create_index("ix_data_mixture_versions_status", ["status"])
        batch.create_index("ix_data_mixture_versions_manifest_hash", ["manifest_hash"])


def downgrade() -> None:
    with op.batch_alter_table("data_mixture_versions") as batch:
        batch.drop_index("ix_data_mixture_versions_manifest_hash")
        batch.drop_index("ix_data_mixture_versions_status")
        batch.drop_constraint("fk_data_mixture_versions_task_id_tasks", type_="foreignkey")
        batch.drop_constraint(
            "fk_data_mixture_versions_manifest_artifact_id_artifacts", type_="foreignkey"
        )
        batch.drop_column("failure_reason")
        batch.drop_column("task_id")
        batch.drop_column("resolved_token_estimate")
        batch.drop_column("resolved_document_count")
        batch.drop_column("manifest_hash")
        batch.drop_column("manifest_artifact_id")
        batch.drop_column("sampler_version")
        batch.drop_column("resolver_version")
        batch.drop_column("seed")
        batch.drop_column("status")

    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_constraint(
            "fk_dataset_versions_record_manifest_artifact_id_artifacts", type_="foreignkey"
        )
        batch.drop_column("record_manifest_artifact_id")
