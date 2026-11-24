"""add standard benchmark metadata and run signatures

Revision ID: c4b2a67d91ef
Revises: b7e31f8c2a10
Create Date: 2026-08-03 12:00:00
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4b2a67d91ef"
down_revision: str | Sequence[str] | None = "b7e31f8c2a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_dataset_versions") as batch_op:
        batch_op.add_column(sa.Column("benchmark_config", sa.JSON(), nullable=False, server_default="{}"))
        batch_op.add_column(sa.Column("compatibility_hash", sa.String(length=64), nullable=True))
        batch_op.create_index(
            "ix_eval_dataset_versions_compatibility_hash", ["compatibility_hash"], unique=False
        )
    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.add_column(sa.Column("config_snapshot", sa.JSON(), nullable=False, server_default="{}"))
        batch_op.add_column(sa.Column("benchmark_signature", sa.JSON(), nullable=False, server_default="{}"))
        batch_op.add_column(sa.Column("compatibility_hash", sa.String(length=64), nullable=True))
        batch_op.create_index(
            "ix_evaluation_runs_compatibility_hash", ["compatibility_hash"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.drop_index("ix_evaluation_runs_compatibility_hash")
        batch_op.drop_column("compatibility_hash")
        batch_op.drop_column("benchmark_signature")
        batch_op.drop_column("config_snapshot")
    with op.batch_alter_table("eval_dataset_versions") as batch_op:
        batch_op.drop_index("ix_eval_dataset_versions_compatibility_hash")
        batch_op.drop_column("compatibility_hash")
        batch_op.drop_column("benchmark_config")
