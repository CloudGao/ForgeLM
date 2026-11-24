"""Project storage lifecycle defaults."""
import sqlalchemy as sa
from alembic import op

revision = "e1b5c9d0f417"
down_revision = "d9a4b8e0f306"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("projects", sa.Column("storage_policy", sa.JSON(), nullable=False,
                                       server_default=sa.text("'{}'")))


def downgrade():
    op.drop_column("projects", "storage_policy")
