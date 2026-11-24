"""add validation mixture binding

Revision ID: c4b8d2e91a70
Revises: 9a31c4e7d2f0
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4b8d2e91a70"
down_revision: str | Sequence[str] | None = "9a31c4e7d2f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("plan_revisions") as batch:
        batch.add_column(
            sa.Column("validation_mixture_version_id", sa.String(length=36), nullable=True)
        )
        batch.create_foreign_key(
            "fk_plan_revisions_validation_mixture_version_id_data_mixture_versions",
            "data_mixture_versions",
            ["validation_mixture_version_id"],
            ["id"],
        )
        batch.create_index(
            "ix_plan_revisions_validation_mixture_version_id",
            ["validation_mixture_version_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("plan_revisions") as batch:
        batch.drop_index("ix_plan_revisions_validation_mixture_version_id")
        batch.drop_constraint(
            "fk_plan_revisions_validation_mixture_version_id_data_mixture_versions",
            type_="foreignkey",
        )
        batch.drop_column("validation_mixture_version_id")
