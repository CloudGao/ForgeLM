"""add task attempt observability fields

Revision ID: 4f7d9a31c2b8
Revises: f5b7c1d29a40
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4f7d9a31c2b8"
down_revision: str | Sequence[str] | None = "f5b7c1d29a40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("task_attempts") as batch:
        batch.add_column(sa.Column("resource_summary", sa.JSON(), nullable=False, server_default="{}"))
        batch.add_column(
            sa.Column("runtime_metrics_artifact_id", sa.String(length=36), nullable=True)
        )
        batch.add_column(sa.Column("log_summary", sa.JSON(), nullable=False, server_default="{}"))
        batch.create_foreign_key(
            "fk_task_attempts_runtime_metrics_artifact_id_artifacts",
            "artifacts",
            ["runtime_metrics_artifact_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "ix_task_attempts_runtime_metrics_artifact_id",
            ["runtime_metrics_artifact_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("task_attempts") as batch:
        batch.drop_index("ix_task_attempts_runtime_metrics_artifact_id")
        batch.drop_constraint(
            "fk_task_attempts_runtime_metrics_artifact_id_artifacts", type_="foreignkey"
        )
        batch.drop_column("log_summary")
        batch.drop_column("runtime_metrics_artifact_id")
        batch.drop_column("resource_summary")
