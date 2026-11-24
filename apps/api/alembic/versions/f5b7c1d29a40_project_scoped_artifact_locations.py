"""project scoped artifact locations

Revision ID: f5b7c1d29a40
Revises: c8a4e21f9b7d
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "f5b7c1d29a40"
down_revision: str | Sequence[str] | None = "c8a4e21f9b7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("artifact_locations") as batch_op:
        batch_op.add_column(sa.Column("project_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_artifact_locations_project_id_projects",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_artifact_locations_project_id", ["project_id"])

    op.execute(
        """
        UPDATE artifact_locations
        SET project_id = (
            SELECT tasks.project_id
            FROM artifacts
            JOIN task_attempts
              ON task_attempts.id = artifacts.created_by_task_attempt_id
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE artifacts.id = artifact_locations.artifact_id
        )
        WHERE project_id IS NULL
          AND EXISTS (
            SELECT 1
            FROM artifacts
            JOIN task_attempts
              ON task_attempts.id = artifacts.created_by_task_attempt_id
            JOIN tasks ON tasks.id = task_attempts.task_id
            WHERE artifacts.id = artifact_locations.artifact_id
              AND tasks.project_id IS NOT NULL
          )
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("artifact_locations") as batch_op:
        batch_op.drop_index("ix_artifact_locations_project_id")
        batch_op.drop_constraint(
            "fk_artifact_locations_project_id_projects", type_="foreignkey"
        )
        batch_op.drop_column("project_id")
