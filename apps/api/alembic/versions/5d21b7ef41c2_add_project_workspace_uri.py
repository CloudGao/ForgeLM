"""add_project_workspace_uri

Revision ID: 5d21b7ef41c2
Revises: 91d7cfdca62e
Create Date: 2026-08-04 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "5d21b7ef41c2"
down_revision: str | Sequence[str] | None = "91d7cfdca62e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column("workspace_uri", sa.Text(), nullable=False, server_default="")
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("workspace_uri")
