"""add dataset source config and sample preview

Revision ID: 91d7cfdca62e
Revises: c4b2a67d91ef
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "91d7cfdca62e"
down_revision: str | Sequence[str] | None = "c4b2a67d91ef"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.add_column(sa.Column("source_config", sa.JSON(), nullable=False, server_default="{}"))
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.add_column(sa.Column("sample_preview", sa.JSON(), nullable=False, server_default="[]"))


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.drop_column("sample_preview")
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.drop_column("source_config")
