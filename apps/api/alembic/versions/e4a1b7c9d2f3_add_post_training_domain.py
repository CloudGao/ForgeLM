"""add first-class post-training domain

Revision ID: e4a1b7c9d2f3
Revises: d3f8a1c5b790
Create Date: 2026-08-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4a1b7c9d2f3"
down_revision: str | Sequence[str] | None = "d3f8a1c5b790"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _record_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    with op.batch_alter_table("code_versions") as batch:
        batch.add_column(sa.Column("training_contract", sa.JSON(), nullable=False, server_default="{}"))
    with op.batch_alter_table("training_runs") as batch:
        batch.add_column(sa.Column("progress_counters", sa.JSON(), nullable=False, server_default="{}"))
    with op.batch_alter_table("checkpoints") as batch:
        batch.add_column(sa.Column("training_position", sa.JSON(), nullable=False, server_default="{}"))

    op.create_table(
        "chat_templates",
        *_record_columns(),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_chat_template_project_name"),
    )
    op.create_index("ix_chat_templates_project_id", "chat_templates", ["project_id"])
    op.create_table(
        "chat_template_versions",
        *_record_columns(),
        sa.Column("template_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("template_body", sa.Text(), nullable=False),
        sa.Column("role_schema", sa.JSON(), nullable=False),
        sa.Column("required_special_tokens", sa.JSON(), nullable=False),
        sa.Column("assistant_mask_mode", sa.String(length=40), nullable=False),
        sa.Column("renderer_version", sa.String(length=80), nullable=False),
        sa.Column("validation_report", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["template_id"], ["chat_templates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("template_id", "version", name="uq_chat_template_version"),
        sa.UniqueConstraint("template_id", "content_hash", name="uq_chat_template_content"),
    )
    op.create_index("ix_chat_template_versions_template_id", "chat_template_versions", ["template_id"])
    op.create_index("ix_chat_template_versions_content_hash", "chat_template_versions", ["content_hash"])

    op.create_table(
        "supervision_datasets",
        *_record_columns(),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_supervision_dataset_project_name"),
    )
    op.create_index("ix_supervision_datasets_project_id", "supervision_datasets", ["project_id"])
    op.create_table(
        "supervision_dataset_versions",
        *_record_columns(),
        sa.Column("supervision_dataset_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("mapping_config", sa.JSON(), nullable=False),
        sa.Column("filter_config", sa.JSON(), nullable=False),
        sa.Column("validator_version", sa.String(length=80), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("validation_report", sa.JSON(), nullable=False),
        sa.Column("sample_preview", sa.JSON(), nullable=False),
        sa.Column("accepted_record_count", sa.Integer(), nullable=False),
        sa.Column("rejected_record_count", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["supervision_dataset_id"], ["supervision_datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("supervision_dataset_id", "version", name="uq_supervision_dataset_version"),
        sa.UniqueConstraint("supervision_dataset_id", "identity_hash", name="uq_supervision_dataset_identity"),
    )
    for name, column in (
        ("ix_supervision_dataset_versions_supervision_dataset_id", "supervision_dataset_id"),
        ("ix_supervision_dataset_versions_dataset_version_id", "dataset_version_id"),
        ("ix_supervision_dataset_versions_kind", "kind"),
        ("ix_supervision_dataset_versions_identity_hash", "identity_hash"),
        ("ix_supervision_dataset_versions_status", "status"),
        ("ix_supervision_dataset_versions_manifest_hash", "manifest_hash"),
    ):
        op.create_index(name, "supervision_dataset_versions", [column])

    op.create_table(
        "supervision_mixtures",
        *_record_columns(),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_supervision_mixture_project_name"),
    )
    op.create_index("ix_supervision_mixtures_project_id", "supervision_mixtures", ["project_id"])
    op.create_table(
        "supervision_mixture_versions",
        *_record_columns(),
        sa.Column("mixture_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("entries", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("sampling_policy", sa.String(length=80), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("locked", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["mixture_id"], ["supervision_mixtures.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("mixture_id", "version", name="uq_supervision_mixture_version"),
        sa.UniqueConstraint("mixture_id", "identity_hash", name="uq_supervision_mixture_identity"),
    )
    op.create_index("ix_supervision_mixture_versions_mixture_id", "supervision_mixture_versions", ["mixture_id"])
    op.create_index("ix_supervision_mixture_versions_kind", "supervision_mixture_versions", ["kind"])
    op.create_index("ix_supervision_mixture_versions_identity_hash", "supervision_mixture_versions", ["identity_hash"])
    op.create_index("ix_supervision_mixture_versions_status", "supervision_mixture_versions", ["status"])

    op.create_table(
        "supervision_corpus_versions",
        *_record_columns(),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("mixture_version_id", sa.String(length=36), nullable=False),
        sa.Column("tokenizer_version_id", sa.String(length=36), nullable=False),
        sa.Column("chat_template_version_id", sa.String(length=36), nullable=False),
        sa.Column("objective_kind", sa.String(length=30), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("pair_count", sa.Integer(), nullable=False),
        sa.Column("raw_token_count", sa.Integer(), nullable=False),
        sa.Column("supervised_token_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["mixture_version_id"], ["supervision_mixture_versions.id"]),
        sa.ForeignKeyConstraint(["tokenizer_version_id"], ["tokenizer_versions.id"]),
        sa.ForeignKeyConstraint(["chat_template_version_id"], ["chat_template_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "identity_hash", name="uq_supervision_corpus_identity"),
    )
    for column in (
        "project_id", "mixture_version_id", "tokenizer_version_id",
        "chat_template_version_id", "objective_kind", "identity_hash", "status", "manifest_hash",
    ):
        op.create_index(f"ix_supervision_corpus_versions_{column}", "supervision_corpus_versions", [column])

    op.create_table(
        "supervision_training_view_versions",
        *_record_columns(),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("corpus_version_id", sa.String(length=36), nullable=False),
        sa.Column("objective_kind", sa.String(length=30), nullable=False),
        sa.Column("identity_hash", sa.String(length=64), nullable=False),
        sa.Column("format", sa.String(length=80), nullable=False),
        sa.Column("budget_unit", sa.String(length=40), nullable=False),
        sa.Column("budget_value", sa.Integer(), nullable=False),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("pair_count", sa.Integer(), nullable=False),
        sa.Column("raw_token_count", sa.Integer(), nullable=False),
        sa.Column("supervised_token_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_id", sa.String(length=36), nullable=True),
        sa.Column("manifest_hash", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["corpus_version_id"], ["supervision_corpus_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "identity_hash", name="uq_supervision_training_view_identity"),
    )
    for column in (
        "project_id", "corpus_version_id", "objective_kind", "identity_hash", "status", "manifest_hash",
    ):
        op.create_index(
            f"ix_supervision_training_view_versions_{column}",
            "supervision_training_view_versions",
            [column],
        )

    with op.batch_alter_table("model_versions") as batch:
        batch.add_column(sa.Column("chat_template_version_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("objective_kind", sa.String(length=30), nullable=True))
        batch.create_foreign_key(
            "fk_model_versions_chat_template_version_id",
            "chat_template_versions",
            ["chat_template_version_id"],
            ["id"],
        )
        batch.create_index("ix_model_versions_chat_template_version_id", ["chat_template_version_id"])
        batch.create_index("ix_model_versions_objective_kind", ["objective_kind"])


def downgrade() -> None:
    with op.batch_alter_table("model_versions") as batch:
        batch.drop_index("ix_model_versions_objective_kind")
        batch.drop_index("ix_model_versions_chat_template_version_id")
        batch.drop_constraint("fk_model_versions_chat_template_version_id", type_="foreignkey")
        batch.drop_column("objective_kind")
        batch.drop_column("chat_template_version_id")

    op.drop_table("supervision_training_view_versions")
    op.drop_table("supervision_corpus_versions")
    op.drop_table("supervision_mixture_versions")
    op.drop_table("supervision_mixtures")
    op.drop_table("supervision_dataset_versions")
    op.drop_table("supervision_datasets")
    op.drop_table("chat_template_versions")
    op.drop_table("chat_templates")

    with op.batch_alter_table("checkpoints") as batch:
        batch.drop_column("training_position")
    with op.batch_alter_table("training_runs") as batch:
        batch.drop_column("progress_counters")
    with op.batch_alter_table("code_versions") as batch:
        batch.drop_column("training_contract")
