"""Index ordered Worker event idempotency lookups on SQLite."""
from alembic import op

revision = "b4e8f102ad73"
down_revision = "a3d7e9f102bc"
branch_labels = None
depends_on = None


def upgrade():
    op.execute('CREATE INDEX ix_task_event_worker_sequence ON task_events '
               '(task_id, attempt_id, json_extract(payload, \'$."worker_sequence"\'))')


def downgrade():
    op.drop_index("ix_task_event_worker_sequence", table_name="task_events")
