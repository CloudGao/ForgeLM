"""add data conversion stage

Revision ID: d8e4f1a2b6c9
Revises: b2d7e51a9c40
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8e4f1a2b6c9"
down_revision: str | Sequence[str] | None = "b2d7e51a9c40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_conversion_recipes",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "name", name="uq_data_conversion_recipe_project_name"),
    )
    op.create_index(
        op.f("ix_data_conversion_recipes_project_id"),
        "data_conversion_recipes",
        ["project_id"],
        unique=False,
    )
    op.create_table(
        "data_conversion_recipe_versions",
        sa.Column("recipe_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("engine_adapter_id", sa.String(length=180), nullable=False),
        sa.Column("operators", sa.JSON(), nullable=False),
        sa.Column("execution_config", sa.JSON(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["recipe_id"], ["data_conversion_recipes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("recipe_id", "version", name="uq_data_conversion_recipe_version"),
    )
    op.create_index(
        op.f("ix_data_conversion_recipe_versions_recipe_id"),
        "data_conversion_recipe_versions",
        ["recipe_id"],
        unique=False,
    )
    op.create_table(
        "data_conversion_runs",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("recipe_version_id", sa.String(length=36), nullable=False),
        sa.Column("input_dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("output_dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("engine_version", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["recipe_version_id"], ["data_conversion_recipe_versions.id"]),
        sa.ForeignKeyConstraint(["input_dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["output_dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("output_dataset_version_id"),
        sa.UniqueConstraint("task_id"),
    )
    for column in (
        "project_id",
        "recipe_version_id",
        "input_dataset_version_id",
        "status",
    ):
        op.create_index(
            op.f(f"ix_data_conversion_runs_{column}"),
            "data_conversion_runs",
            [column],
            unique=False,
        )
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.add_column(
            sa.Column("conversion_recipe_version_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(sa.Column("conversion_run_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_dataset_versions_conversion_recipe_version_id",
            "data_conversion_recipe_versions",
            ["conversion_recipe_version_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_dataset_versions_conversion_run_id",
            "data_conversion_runs",
            ["conversion_run_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_dataset_versions_conversion_recipe_version_id",
            ["conversion_recipe_version_id"],
        )
        batch_op.create_index("ix_dataset_versions_conversion_run_id", ["conversion_run_id"])


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.drop_index("ix_dataset_versions_conversion_run_id")
        batch_op.drop_index("ix_dataset_versions_conversion_recipe_version_id")
        batch_op.drop_constraint("fk_dataset_versions_conversion_run_id", type_="foreignkey")
        batch_op.drop_constraint(
            "fk_dataset_versions_conversion_recipe_version_id", type_="foreignkey"
        )
        batch_op.drop_column("conversion_run_id")
        batch_op.drop_column("conversion_recipe_version_id")
    op.drop_table("data_conversion_runs")
    op.drop_table("data_conversion_recipe_versions")
    op.drop_table("data_conversion_recipes")
