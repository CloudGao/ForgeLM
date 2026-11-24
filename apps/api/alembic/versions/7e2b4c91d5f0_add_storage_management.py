"""add storage management

Revision ID: 7e2b4c91d5f0
Revises: c4b8d2e91a70
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7e2b4c91d5f0"
down_revision: str | Sequence[str] | None = "c4b8d2e91a70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "storage_scans",
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_storage_scans_status", "storage_scans", ["status"])
    op.create_table(
        "storage_inventory_items",
        sa.Column("scan_id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=True),
        sa.Column("item_type", sa.String(length=30), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("reclaimable", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["attempt_id"], ["task_attempts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["scan_id"], ["storage_scans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scan_id", "path", "item_type", name="uq_storage_scan_path_type"),
    )
    for column in ("scan_id", "project_id", "item_type", "category", "reclaimable", "task_id", "attempt_id"):
        op.create_index(f"ix_storage_inventory_items_{column}", "storage_inventory_items", [column])
    op.create_table(
        "storage_cleanup_plans",
        sa.Column("scan_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("confirmation_phrase", sa.String(length=120), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["scan_id"], ["storage_scans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_storage_cleanup_plans_scan_id", "storage_cleanup_plans", ["scan_id"])
    op.create_index("ix_storage_cleanup_plans_status", "storage_cleanup_plans", ["status"])
    op.create_index("ix_storage_cleanup_plans_expires_at", "storage_cleanup_plans", ["expires_at"])
    op.create_table(
        "storage_cleanup_plan_items",
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("inventory_item_id", sa.String(length=36), nullable=True),
        sa.Column("project_id", sa.String(length=36), nullable=True),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=True),
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actual_reclaimed_bytes", sa.Integer(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["attempt_id"], ["task_attempts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["inventory_item_id"], ["storage_inventory_items.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["plan_id"], ["storage_cleanup_plans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("plan_id", "project_id", "category", "task_id", "attempt_id", "status"):
        op.create_index(f"ix_storage_cleanup_plan_items_{column}", "storage_cleanup_plan_items", [column])


def downgrade() -> None:
    op.drop_table("storage_cleanup_plan_items")
    op.drop_table("storage_cleanup_plans")
    op.drop_table("storage_inventory_items")
    op.drop_table("storage_scans")
