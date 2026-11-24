"""Allow independent immutable training instances and task recovery across attempts."""

import sqlalchemy as sa
from alembic import op

revision = "d9a4b8e0f306"
down_revision = "c8f3a7d9e205"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table, columns in (
        ("project_pipeline_instances", ("pipeline_id", "definition_revision_id")),
        ("project_pipeline_runs", ("pipeline_id", "revision_id")),
    ):
        with op.batch_alter_table(table) as batch:
            for column in columns:
                batch.alter_column(column, existing_type=sa.String(36), nullable=True)
    op.drop_index("ix_project_pipeline_node_runs_task_id", table_name="project_pipeline_node_runs")
    with op.batch_alter_table(
        "project_pipeline_node_runs",
        naming_convention={"uq": "uq_%(table_name)s_%(column_0_name)s"},
    ) as batch:
        batch.drop_constraint("uq_project_pipeline_node_runs_task_id", type_="unique")
    op.create_index(
        "ix_project_pipeline_node_runs_task_id",
        "project_pipeline_node_runs",
        ["task_id"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(
        sa.text("SELECT 1 FROM project_pipeline_instances WHERE pipeline_id IS NULL LIMIT 1")
    ).first():
        raise RuntimeError(
            "Independent training instances exist; downgrade would discard their identity"
        )
    if bind.execute(
        sa.text(
            "SELECT task_id FROM project_pipeline_node_runs WHERE task_id IS NOT NULL GROUP BY task_id HAVING COUNT(*) > 1 LIMIT 1"
        )
    ).first():
        raise RuntimeError(
            "Recovered attempts share tasks; downgrade would discard execution history"
        )
    op.drop_index("ix_project_pipeline_node_runs_task_id", table_name="project_pipeline_node_runs")
    op.create_index(
        "ix_project_pipeline_node_runs_task_id",
        "project_pipeline_node_runs",
        ["task_id"],
        unique=False,
    )
    with op.batch_alter_table("project_pipeline_node_runs") as batch:
        batch.create_unique_constraint("uq_project_pipeline_node_runs_task_id", ["task_id"])
    for table, columns in (
        ("project_pipeline_runs", ("pipeline_id", "revision_id")),
        ("project_pipeline_instances", ("pipeline_id", "definition_revision_id")),
    ):
        with op.batch_alter_table(table) as batch:
            for column in columns:
                batch.alter_column(column, existing_type=sa.String(36), nullable=False)
