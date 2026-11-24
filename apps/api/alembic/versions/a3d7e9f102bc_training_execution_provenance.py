"""Immutable attempt snapshots and suite training checkpoint lineage."""
import sqlalchemy as sa
from alembic import op

revision = "a3d7e9f102bc"
down_revision = "f2c6d0e1a528"
branch_labels = None
depends_on = None


def base_columns():
    return [sa.Column("id", sa.String(36), primary_key=True), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    op.add_column("suite_config_revisions", sa.Column("requested_config", sa.JSON(), nullable=True))
    op.create_table("execution_snapshots", *base_columns(),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_id", sa.String(36), sa.ForeignKey("task_attempts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("phase", sa.String(30), nullable=False), sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False), sa.UniqueConstraint("attempt_id", "phase", name="uq_execution_snapshot_phase"))
    op.create_table("training_state_checkpoints", *base_columns(),
        sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_id", sa.String(36), sa.ForeignKey("task_attempts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_id", sa.String(36), sa.ForeignKey("artifacts.id"), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False), sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("execution_digest", sa.String(64), nullable=False),
        sa.UniqueConstraint("attempt_id", "step", name="uq_training_state_checkpoint_step"))
    for table, columns in [("execution_snapshots", ["task_id", "attempt_id", "digest"]), ("training_state_checkpoints", ["project_id", "task_id", "attempt_id"])]:
        for column in columns:
            op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade():
    op.drop_column("suite_config_revisions", "requested_config")
    op.drop_table("training_state_checkpoints")
    op.drop_table("execution_snapshots")
