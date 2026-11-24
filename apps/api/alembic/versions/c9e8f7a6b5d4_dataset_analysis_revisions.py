"""separate dataset analysis revisions from content versions

Revision ID: c9e8f7a6b5d4
Revises: b7d4e6f8a2c1
Create Date: 2026-08-15
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "c9e8f7a6b5d4"
down_revision: str | Sequence[str] | None = "b7d4e6f8a2c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_analysis_revisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="queued"),
        sa.Column("snapshot_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("snapshot_content_hash", sa.String(length=128), nullable=True),
        sa.Column("raw_size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("record_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("parser_id", sa.String(length=160), nullable=True),
        sa.Column("parser_version", sa.String(length=80), nullable=True),
        sa.Column("parser_status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("parser_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("detected_formats", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("data_schema", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("statistics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("usage_capabilities", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("train_ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("automatic_quality_score", sa.Float(), nullable=True),
        sa.Column("quality_report", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("sample_preview", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("source_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("record_manifest_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("error", sa.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["snapshot_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["record_manifest_artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", name="uq_dataset_analysis_task_id"),
        sa.UniqueConstraint(
            "dataset_version_id", "revision", name="uq_dataset_analysis_revision"
        ),
    )
    op.create_index(
        "ix_dataset_analysis_revisions_dataset_version_id",
        "dataset_analysis_revisions",
        ["dataset_version_id"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_task_id",
        "dataset_analysis_revisions",
        ["task_id"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_status",
        "dataset_analysis_revisions",
        ["status"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_snapshot_artifact_id",
        "dataset_analysis_revisions",
        ["snapshot_artifact_id"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_parser_id",
        "dataset_analysis_revisions",
        ["parser_id"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_parser_status",
        "dataset_analysis_revisions",
        ["parser_status"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_train_ready",
        "dataset_analysis_revisions",
        ["train_ready"],
    )
    op.create_index(
        "ix_dataset_analysis_revisions_record_manifest_artifact_id",
        "dataset_analysis_revisions",
        ["record_manifest_artifact_id"],
    )

    with op.batch_alter_table("dataset_version_indexes") as batch:
        batch.add_column(sa.Column("analysis_revision_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_dataset_version_indexes_analysis_revision_id",
            "dataset_analysis_revisions",
            ["analysis_revision_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.drop_constraint("uq_dataset_version_index_kind", type_="unique")
        batch.create_unique_constraint(
            "uq_dataset_analysis_index_kind",
            ["dataset_version_id", "analysis_revision_id", "index_kind"],
        )
        batch.create_index(
            "ix_dataset_version_indexes_analysis_revision_id", ["analysis_revision_id"]
        )

    connection = op.get_bind()
    artifact_hashes = {
        row["id"]: row["content_hash"]
        for row in connection.execute(
            sa.text("SELECT id, content_hash FROM artifacts")
        ).mappings()
    }
    now = datetime.now(UTC)
    analysis_ids: dict[str, str] = {}
    rows = connection.execute(
        sa.text(
            "SELECT id, task_id, state, artifact_id, raw_size_bytes, document_count, record_count, "
            "token_estimate, parser_id, parser_version, parser_status, parser_config, "
            "detected_formats, data_schema, statistics, usage_capabilities, "
            "train_ready, automatic_quality_score, quality_report, sample_preview, source_snapshot, "
            "record_manifest_artifact_id FROM dataset_versions"
        )
    ).mappings()
    for row in rows:
        analysis_id = str(uuid4())
        analysis_ids[row["id"]] = analysis_id
        connection.execute(
            sa.text(
                "INSERT INTO dataset_analysis_revisions ("
                "id, created_at, updated_at, dataset_version_id, revision, task_id, status, "
                "snapshot_artifact_id, snapshot_content_hash, raw_size_bytes, document_count, "
                "record_count, token_estimate, parser_id, parser_version, parser_status, "
                "parser_config, detected_formats, data_schema, statistics, usage_capabilities, train_ready, "
                "automatic_quality_score, quality_report, sample_preview, source_snapshot, "
                "record_manifest_artifact_id, error) VALUES ("
                ":id, :now, :now, :version_id, 1, :task_id, :status, :artifact_id, "
                ":content_hash, :raw_size_bytes, :document_count, :record_count, :token_estimate, "
                ":parser_id, :parser_version, :parser_status, :parser_config, :detected_formats, "
                ":data_schema, :statistics, :usage_capabilities, :train_ready, :automatic_quality_score, "
                ":quality_report, :sample_preview, :source_snapshot, :record_manifest_artifact_id, '{}')"
            ),
            {
                "id": analysis_id,
                "now": now,
                "version_id": row["id"],
                "task_id": row["task_id"],
                "status": "succeeded" if row["state"] == "ready" else row["state"],
                "artifact_id": row["artifact_id"],
                "content_hash": artifact_hashes.get(row["artifact_id"]),
                **{
                    key: row[key]
                    for key in row.keys()
                    if key not in {"id", "task_id", "state", "artifact_id"}
                },
            },
        )
    for version_id, analysis_id in analysis_ids.items():
        connection.execute(
            sa.text(
                "UPDATE dataset_version_indexes SET analysis_revision_id = :analysis_id "
                "WHERE dataset_version_id = :version_id"
            ),
            {"analysis_id": analysis_id, "version_id": version_id},
        )


def downgrade() -> None:
    with op.batch_alter_table("dataset_version_indexes") as batch:
        batch.drop_index("ix_dataset_version_indexes_analysis_revision_id")
        batch.drop_constraint("uq_dataset_analysis_index_kind", type_="unique")
        batch.create_unique_constraint(
            "uq_dataset_version_index_kind", ["dataset_version_id", "index_kind"]
        )
        batch.drop_constraint(
            "fk_dataset_version_indexes_analysis_revision_id", type_="foreignkey"
        )
        batch.drop_column("analysis_revision_id")

    op.drop_index(
        "ix_dataset_analysis_revisions_record_manifest_artifact_id",
        table_name="dataset_analysis_revisions",
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_train_ready",
        table_name="dataset_analysis_revisions",
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_parser_status",
        table_name="dataset_analysis_revisions",
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_parser_id",
        table_name="dataset_analysis_revisions",
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_snapshot_artifact_id",
        table_name="dataset_analysis_revisions",
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_status", table_name="dataset_analysis_revisions"
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_task_id", table_name="dataset_analysis_revisions"
    )
    op.drop_index(
        "ix_dataset_analysis_revisions_dataset_version_id",
        table_name="dataset_analysis_revisions",
    )
    op.drop_table("dataset_analysis_revisions")
