"""add code sources and immutable versions

Revision ID: 8f4c2d91a6b3
Revises: 5d21b7ef41c2
Create Date: 2026-08-04 20:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8f4c2d91a6b3"
down_revision: str | Sequence[str] | None = "5d21b7ef41c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "code_sources" not in tables:
        op.create_table(
            "code_sources",
            sa.Column("project_id", sa.String(length=36), nullable=False),
            sa.Column("name", sa.String(length=180), nullable=False),
            sa.Column("source_type", sa.String(length=40), nullable=False),
            sa.Column("source_uri", sa.Text(), nullable=False, server_default=""),
            sa.Column("source_config", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("current_version_id", sa.String(length=36), nullable=True),
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("project_id", "name", name="uq_code_source_project_name"),
        )
        op.create_index(op.f("ix_code_sources_project_id"), "code_sources", ["project_id"])
        op.create_index(op.f("ix_code_sources_status"), "code_sources", ["status"])
    else:
        source_indexes = {item["name"] for item in inspector.get_indexes("code_sources")}
        if "ix_code_sources_project_id" not in source_indexes:
            op.create_index("ix_code_sources_project_id", "code_sources", ["project_id"])
        if "ix_code_sources_status" not in source_indexes:
            op.create_index("ix_code_sources_status", "code_sources", ["status"])

    if "code_versions" not in tables:
        op.create_table(
            "code_versions",
            sa.Column("source_id", sa.String(length=36), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("state", sa.String(length=40), nullable=False),
            sa.Column("artifact_id", sa.String(length=36), nullable=True),
            sa.Column("task_id", sa.String(length=36), nullable=True),
            sa.Column("git_commit", sa.String(length=64), nullable=True),
            sa.Column("source_snapshot", sa.JSON(), nullable=False),
            sa.Column("entrypoints", sa.JSON(), nullable=False),
            sa.Column("config_schema", sa.JSON(), nullable=False),
            sa.Column("dependency_spec", sa.Text(), nullable=False, server_default=""),
            sa.Column("immutable", sa.Boolean(), nullable=False),
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["source_id"], ["code_sources.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("source_id", "version", name="uq_code_source_version"),
        )
        op.create_index(op.f("ix_code_versions_source_id"), "code_versions", ["source_id"])
        op.create_index(op.f("ix_code_versions_state"), "code_versions", ["state"])
        op.create_index(op.f("ix_code_versions_git_commit"), "code_versions", ["git_commit"])
    else:
        version_indexes = {item["name"] for item in inspector.get_indexes("code_versions")}
        for name, columns in (
            ("ix_code_versions_source_id", ["source_id"]),
            ("ix_code_versions_state", ["state"]),
            ("ix_code_versions_git_commit", ["git_commit"]),
        ):
            if name not in version_indexes:
                op.create_index(name, "code_versions", columns)

    inspector = sa.inspect(bind)
    plan_columns = {item["name"] for item in inspector.get_columns("plan_revisions")}
    plan_indexes = {item["name"] for item in inspector.get_indexes("plan_revisions")}
    plan_foreign_keys = {
        tuple(item["constrained_columns"])
        for item in inspector.get_foreign_keys("plan_revisions")
    }

    def alter_plan_revisions() -> None:
        if inspector.has_table("_alembic_tmp_plan_revisions"):
            # SQLite batch DDL is non-transactional. If a prior attempt stopped after
            # creating the temporary copy, the authoritative table still exists and
            # the stale copy prevents Alembic from retrying the operation.
            op.drop_table("_alembic_tmp_plan_revisions")
        with op.batch_alter_table("plan_revisions") as batch_op:
            if "code_version_id" not in plan_columns:
                batch_op.add_column(
                    sa.Column("code_version_id", sa.String(length=36), nullable=True)
                )
            if ("code_version_id",) not in plan_foreign_keys:
                batch_op.create_foreign_key(
                    "fk_plan_revisions_code_version_id",
                    "code_versions",
                    ["code_version_id"],
                    ["id"],
                )
            if "ix_plan_revisions_code_version_id" not in plan_indexes:
                batch_op.create_index("ix_plan_revisions_code_version_id", ["code_version_id"])

    if bind.dialect.name == "sqlite":
        with op.get_context().autocommit_block():
            bind.exec_driver_sql("PRAGMA foreign_keys=OFF")
            try:
                alter_plan_revisions()
            finally:
                bind.exec_driver_sql("PRAGMA foreign_keys=ON")
    else:
        alter_plan_revisions()


def downgrade() -> None:
    with op.batch_alter_table("plan_revisions") as batch_op:
        batch_op.drop_index("ix_plan_revisions_code_version_id")
        batch_op.drop_constraint("fk_plan_revisions_code_version_id", type_="foreignkey")
        batch_op.drop_column("code_version_id")
    op.drop_index(op.f("ix_code_versions_git_commit"), table_name="code_versions")
    op.drop_index(op.f("ix_code_versions_state"), table_name="code_versions")
    op.drop_index(op.f("ix_code_versions_source_id"), table_name="code_versions")
    op.drop_table("code_versions")
    op.drop_index(op.f("ix_code_sources_status"), table_name="code_sources")
    op.drop_index(op.f("ix_code_sources_project_id"), table_name="code_sources")
    op.drop_table("code_sources")
