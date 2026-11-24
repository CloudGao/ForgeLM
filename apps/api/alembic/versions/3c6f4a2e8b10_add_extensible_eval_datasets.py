"""add extensible evaluation dataset metadata

Revision ID: 3c6f4a2e8b10
Revises: 7e2b4c91d5f0
Create Date: 2026-08-10 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3c6f4a2e8b10"
down_revision: str | Sequence[str] | None = "7e2b4c91d5f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("eval_dataset_versions") as batch_op:
        batch_op.add_column(
            sa.Column("source_config", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(
            sa.Column("schema_config", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.add_column(
            sa.Column("scoring_config", sa.JSON(), nullable=False, server_default="{}")
        )


def downgrade() -> None:
    with op.batch_alter_table("eval_dataset_versions") as batch_op:
        batch_op.drop_column("scoring_config")
        batch_op.drop_column("schema_config")
        batch_op.drop_column("source_config")
