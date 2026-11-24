"""remove simulation plugins

Revision ID: d2a40b6e2c91
Revises: a4b95c7f142e
Create Date: 2026-08-02 22:15:00
"""

from typing import Sequence

from alembic import op

revision: str = "d2a40b6e2c91"
down_revision: str | Sequence[str] | None = "a4b95c7f142e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "DELETE FROM plugin_versions "
        "WHERE plugin_id LIKE 'builtin.simulated.%' OR plugin_id = 'builtin.command'"
    )


def downgrade() -> None:
    pass
