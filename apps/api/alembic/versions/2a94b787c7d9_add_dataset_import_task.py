"""add_dataset_import_task

Revision ID: 2a94b787c7d9
Revises: f46ef510141b
Create Date: 2026-08-02 21:24:45.493711
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '2a94b787c7d9'
down_revision: Union[str, Sequence[str], None] = 'f46ef510141b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.add_column(sa.Column("task_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_dataset_versions_task_id_tasks", "tasks", ["task_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.drop_constraint("fk_dataset_versions_task_id_tasks", type_="foreignkey")
        batch_op.drop_column("task_id")
