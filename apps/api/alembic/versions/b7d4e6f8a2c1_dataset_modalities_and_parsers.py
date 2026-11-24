"""add dataset modalities and parser contracts

Revision ID: b7d4e6f8a2c1
Revises: a6c2f4e8b1d9
Create Date: 2026-08-15
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "b7d4e6f8a2c1"
down_revision: str | Sequence[str] | None = "a6c2f4e8b1d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(
            sa.Column("modality", sa.String(length=24), nullable=False, server_default="text")
        )
        batch.create_index("ix_datasets_modality", ["modality"])
        batch.create_check_constraint(
            "ck_dataset_modality",
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
        )

    with op.batch_alter_table("dataset_versions") as batch:
        batch.add_column(
            sa.Column("modality", sa.String(length=24), nullable=False, server_default="text")
        )
        batch.add_column(
            sa.Column("record_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("parser_id", sa.String(length=160), nullable=True))
        batch.add_column(sa.Column("parser_version", sa.String(length=80), nullable=True))
        batch.add_column(
            sa.Column("parser_status", sa.String(length=30), nullable=False, server_default="pending")
        )
        batch.add_column(
            sa.Column("parser_config", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("detected_formats", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("data_schema", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("statistics", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("usage_capabilities", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.create_index("ix_dataset_versions_modality", ["modality"])
        batch.create_index("ix_dataset_versions_parser_id", ["parser_id"])
        batch.create_index("ix_dataset_versions_parser_status", ["parser_status"])
        batch.create_check_constraint(
            "ck_dataset_version_modality",
            "modality IN ('text', 'image', 'audio', 'video', 'multimodal', 'other')",
        )

    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE dataset_versions SET record_count = document_count, "
            "parser_id = CASE WHEN train_ready = 1 THEN 'builtin.text-parser' ELSE parser_id END, "
            "parser_version = CASE WHEN train_ready = 1 THEN '1.0.0' ELSE parser_version END, "
            "parser_status = CASE WHEN train_ready = 1 THEN 'succeeded' ELSE 'pending' END"
        )
    )
    ready_capabilities = (
        '["source_archive", "record_explorer", "text_pretraining", '
        '"tokenizer_training", "text_evaluation", "supervision_mapping"]'
    )
    connection.execute(
        sa.text(
            "UPDATE dataset_versions SET usage_capabilities = :capabilities "
            "WHERE train_ready = 1"
        ),
        {"capabilities": ready_capabilities},
    )

    op.create_table(
        "dataset_version_indexes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("index_kind", sa.String(length=80), nullable=False),
        sa.Column("format", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("artifact_id", sa.String(length=36), nullable=False),
        sa.Column("parser_id", sa.String(length=160), nullable=True),
        sa.Column("parser_version", sa.String(length=80), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="ready"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_version_id", "index_kind", name="uq_dataset_version_index_kind"
        ),
    )
    op.create_index(
        "ix_dataset_version_indexes_dataset_version_id",
        "dataset_version_indexes",
        ["dataset_version_id"],
    )
    op.create_index(
        "ix_dataset_version_indexes_index_kind",
        "dataset_version_indexes",
        ["index_kind"],
    )
    op.create_index(
        "ix_dataset_version_indexes_artifact_id",
        "dataset_version_indexes",
        ["artifact_id"],
    )
    op.create_index(
        "ix_dataset_version_indexes_status", "dataset_version_indexes", ["status"]
    )

    now = datetime.now(UTC)
    rows = connection.execute(
        sa.text(
            "SELECT id, record_manifest_artifact_id, parser_id, parser_version "
            "FROM dataset_versions WHERE record_manifest_artifact_id IS NOT NULL"
        )
    ).mappings()
    for row in rows:
        connection.execute(
            sa.text(
                "INSERT INTO dataset_version_indexes "
                "(id, created_at, updated_at, dataset_version_id, index_kind, format, "
                "artifact_id, parser_id, parser_version, status, metadata) "
                "VALUES (:id, :created_at, :updated_at, :version_id, 'record_manifest', "
                "'llm.dataset-records.v1', :artifact_id, :parser_id, :parser_version, "
                "'ready', '{}')"
            ),
            {
                "id": str(uuid4()),
                "created_at": now,
                "updated_at": now,
                "version_id": row["id"],
                "artifact_id": row["record_manifest_artifact_id"],
                "parser_id": row["parser_id"],
                "parser_version": row["parser_version"],
            },
        )


def downgrade() -> None:
    op.drop_index("ix_dataset_version_indexes_status", table_name="dataset_version_indexes")
    op.drop_index("ix_dataset_version_indexes_artifact_id", table_name="dataset_version_indexes")
    op.drop_index("ix_dataset_version_indexes_index_kind", table_name="dataset_version_indexes")
    op.drop_index(
        "ix_dataset_version_indexes_dataset_version_id", table_name="dataset_version_indexes"
    )
    op.drop_table("dataset_version_indexes")

    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_constraint("ck_dataset_version_modality", type_="check")
        batch.drop_index("ix_dataset_versions_parser_status")
        batch.drop_index("ix_dataset_versions_parser_id")
        batch.drop_index("ix_dataset_versions_modality")
        batch.drop_column("usage_capabilities")
        batch.drop_column("statistics")
        batch.drop_column("data_schema")
        batch.drop_column("detected_formats")
        batch.drop_column("parser_config")
        batch.drop_column("parser_status")
        batch.drop_column("parser_version")
        batch.drop_column("parser_id")
        batch.drop_column("record_count")
        batch.drop_column("modality")

    with op.batch_alter_table("datasets") as batch:
        batch.drop_constraint("ck_dataset_modality", type_="check")
        batch.drop_index("ix_datasets_modality")
        batch.drop_column("modality")
