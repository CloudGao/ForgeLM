"""Index attempt windows and sparse numeric task progress."""
from alembic import op

revision = "c5f9a213be84"
down_revision = "b4e8f102ad73"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE INDEX ix_task_event_attempt_sequence ON task_events (task_id, attempt_id, sequence)")
    op.execute("CREATE INDEX ix_task_event_numeric_progress ON task_events (task_id, attempt_id, sequence) "
               "WHERE event_type IN ('progress', 'metric', 'checkpoint') "
               "AND json_type(payload, '$.progress') IN ('integer', 'real')")


def downgrade():
    op.drop_index("ix_task_event_numeric_progress", table_name="task_events")
    op.drop_index("ix_task_event_attempt_sequence", table_name="task_events")
