"""external artifact locations and history imports

Revision ID: b7e31f8c2a10
Revises: d2a40b6e2c91
Create Date: 2026-08-02 23:05:00
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e31f8c2a10"
down_revision: str | Sequence[str] | None = "d2a40b6e2c91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "artifact_locations",
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("storage_type", sa.String(length=30), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("worker_id", sa.String(length=36), nullable=True),
        sa.Column("availability_state", sa.String(length=30), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["worker_id"], ["worker_nodes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_id", "uri", name="uq_artifact_location_uri"),
        sa.UniqueConstraint("uri"),
    )
    op.create_index(
        op.f("ix_artifact_locations_artifact_id"),
        "artifact_locations",
        ["artifact_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_artifact_locations_availability_state"),
        "artifact_locations",
        ["availability_state"],
        unique=False,
    )
    op.create_index(
        op.f("ix_artifact_locations_storage_type"),
        "artifact_locations",
        ["storage_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_artifact_locations_worker_id"),
        "artifact_locations",
        ["worker_id"],
        unique=False,
    )

    op.execute(
        sa.text(
            """
            INSERT INTO artifact_locations (
                artifact_id, storage_type, uri, worker_id, availability_state,
                metadata, last_verified_at, id, created_at, updated_at
            )
            SELECT
                id, 'managed', uri, NULL, availability_state,
                '{}', NULL, lower(hex(randomblob(16))),
                created_at, updated_at
            FROM artifacts
            """
        )
    )

    op.create_table(
        "history_imports",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("source_key", sa.String(length=180), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "source_key", "manifest_hash", name="uq_history_import_manifest"
        ),
    )
    op.create_index(
        op.f("ix_history_imports_manifest_hash"),
        "history_imports",
        ["manifest_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_history_imports_project_id"),
        "history_imports",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_history_imports_source_key"),
        "history_imports",
        ["source_key"],
        unique=False,
    )

    op.create_table(
        "history_import_entities",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("source_key", sa.String(length=180), nullable=False),
        sa.Column("entity_type", sa.String(length=60), nullable=False),
        sa.Column("external_id", sa.String(length=240), nullable=False),
        sa.Column("local_id", sa.String(length=36), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "source_key",
            "entity_type",
            "external_id",
            name="uq_history_import_entity",
        ),
    )
    op.create_index(
        op.f("ix_history_import_entities_entity_type"),
        "history_import_entities",
        ["entity_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_history_import_entities_local_id"),
        "history_import_entities",
        ["local_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_history_import_entities_project_id"),
        "history_import_entities",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_history_import_entities_source_key"),
        "history_import_entities",
        ["source_key"],
        unique=False,
    )

    with op.batch_alter_table("checkpoints") as batch_op:
        batch_op.create_unique_constraint(
            "uq_checkpoint_run_step", ["training_run_id", "step"]
        )


def downgrade() -> None:
    with op.batch_alter_table("checkpoints") as batch_op:
        batch_op.drop_constraint("uq_checkpoint_run_step", type_="unique")
    op.drop_table("history_import_entities")
    op.drop_table("history_imports")
    op.drop_table("artifact_locations")
