"""add release artifact

Revision ID: a4b95c7f142e
Revises: e6c27f56eb4d
Create Date: 2026-08-02 21:52:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a4b95c7f142e"
down_revision: str | Sequence[str] | None = "e6c27f56eb4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("releases") as batch_op:
        batch_op.add_column(sa.Column("artifact_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_releases_artifact_id_artifacts", "artifacts", ["artifact_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("releases") as batch_op:
        batch_op.drop_constraint("fk_releases_artifact_id_artifacts", type_="foreignkey")
        batch_op.drop_column("artifact_id")
