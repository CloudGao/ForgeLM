"""add Data-Juicer processing recipes and runs

Revision ID: b2d7e51a9c40
Revises: 6a8f1c2d4e90
Create Date: 2026-08-10 18:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2d7e51a9c40"
down_revision: str | Sequence[str] | None = "6a8f1c2d4e90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_processing_recipes",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "name", name="uq_data_processing_recipe_project_name"
        ),
    )
    op.create_index(
        op.f("ix_data_processing_recipes_project_id"),
        "data_processing_recipes",
        ["project_id"],
        unique=False,
    )
    op.create_table(
        "data_processing_recipe_versions",
        sa.Column("recipe_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("engine_adapter_id", sa.String(length=180), nullable=False),
        sa.Column("operators", sa.JSON(), nullable=False),
        sa.Column("execution_config", sa.JSON(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["recipe_id"], ["data_processing_recipes.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("recipe_id", "version", name="uq_data_processing_recipe_version"),
    )
    op.create_index(
        op.f("ix_data_processing_recipe_versions_recipe_id"),
        "data_processing_recipe_versions",
        ["recipe_id"],
        unique=False,
    )
    op.create_table(
        "data_processing_runs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("recipe_version_id", sa.String(length=36), nullable=False),
        sa.Column("input_dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("output_dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("output_tier", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("engine_version", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["recipe_version_id"], ["data_processing_recipe_versions.id"]
        ),
        sa.ForeignKeyConstraint(["input_dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["output_dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("output_dataset_version_id"),
        sa.UniqueConstraint("task_id"),
    )
    op.create_index(
        op.f("ix_data_processing_runs_project_id"),
        "data_processing_runs",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_data_processing_runs_recipe_version_id"),
        "data_processing_runs",
        ["recipe_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_data_processing_runs_input_dataset_version_id"),
        "data_processing_runs",
        ["input_dataset_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_data_processing_runs_status"),
        "data_processing_runs",
        ["status"],
        unique=False,
    )

    with op.batch_alter_table("dataset_versions") as batch:
        batch.add_column(sa.Column("parent_version_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column("processing_recipe_version_id", sa.String(length=36), nullable=True)
        )
        batch.add_column(sa.Column("processing_run_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            "fk_dataset_versions_parent_version_id_dataset_versions",
            "dataset_versions",
            ["parent_version_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_dataset_versions_processing_recipe_version_id_data_processing_recipe_versions",
            "data_processing_recipe_versions",
            ["processing_recipe_version_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_dataset_versions_processing_run_id_data_processing_runs",
            "data_processing_runs",
            ["processing_run_id"],
            ["id"],
        )
        batch.create_index("ix_dataset_versions_parent_version_id", ["parent_version_id"])
        batch.create_index(
            "ix_dataset_versions_processing_recipe_version_id",
            ["processing_recipe_version_id"],
        )
        batch.create_index("ix_dataset_versions_processing_run_id", ["processing_run_id"])


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_index("ix_dataset_versions_processing_run_id")
        batch.drop_index("ix_dataset_versions_processing_recipe_version_id")
        batch.drop_index("ix_dataset_versions_parent_version_id")
        batch.drop_constraint(
            "fk_dataset_versions_processing_run_id_data_processing_runs", type_="foreignkey"
        )
        batch.drop_constraint(
            "fk_dataset_versions_processing_recipe_version_id_data_processing_recipe_versions",
            type_="foreignkey",
        )
        batch.drop_constraint(
            "fk_dataset_versions_parent_version_id_dataset_versions", type_="foreignkey"
        )
        batch.drop_column("processing_run_id")
        batch.drop_column("processing_recipe_version_id")
        batch.drop_column("parent_version_id")

    op.drop_index(op.f("ix_data_processing_runs_status"), table_name="data_processing_runs")
    op.drop_index(
        op.f("ix_data_processing_runs_input_dataset_version_id"),
        table_name="data_processing_runs",
    )
    op.drop_index(
        op.f("ix_data_processing_runs_recipe_version_id"), table_name="data_processing_runs"
    )
    op.drop_index(
        op.f("ix_data_processing_runs_project_id"), table_name="data_processing_runs"
    )
    op.drop_table("data_processing_runs")
    op.drop_index(
        op.f("ix_data_processing_recipe_versions_recipe_id"),
        table_name="data_processing_recipe_versions",
    )
    op.drop_table("data_processing_recipe_versions")
    op.drop_index(
        op.f("ix_data_processing_recipes_project_id"), table_name="data_processing_recipes"
    )
    op.drop_table("data_processing_recipes")
