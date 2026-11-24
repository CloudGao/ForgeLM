"""add concrete project pipeline instances

Revision ID: f8a3b5c7d9e1
Revises: e7f2a9c4d6b1
Create Date: 2026-08-16
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "f8a3b5c7d9e1"
down_revision: str | Sequence[str] | None = "e7f2a9c4d6b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project_pipeline_instances",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_id", sa.String(length=36), nullable=False),
        sa.Column("definition_revision_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
        sa.Column("nodes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("edges", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("variables", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("viewport", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("validation_report", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.CheckConstraint(
            "status IN ('draft', 'ready', 'archived')",
            name="ck_project_pipeline_instance_status",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["pipeline_id"], ["project_pipelines.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["definition_revision_id"], ["project_pipeline_revisions.id"]
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "name", name="uq_project_pipeline_instance_name"
        ),
    )
    op.create_index(
        "ix_project_pipeline_instances_project_id",
        "project_pipeline_instances",
        ["project_id"],
    )
    op.create_index(
        "ix_project_pipeline_instances_pipeline_id",
        "project_pipeline_instances",
        ["pipeline_id"],
    )
    op.create_index(
        "ix_project_pipeline_instances_definition_revision_id",
        "project_pipeline_instances",
        ["definition_revision_id"],
    )
    op.create_index(
        "ix_project_pipeline_instances_status",
        "project_pipeline_instances",
        ["status"],
    )
    op.create_index(
        "ix_project_pipeline_instances_content_hash",
        "project_pipeline_instances",
        ["content_hash"],
    )

    with op.batch_alter_table("project_pipeline_runs") as batch_op:
        batch_op.add_column(sa.Column("instance_id", sa.String(length=36), nullable=True))
        batch_op.add_column(
            sa.Column("graph_snapshot", sa.JSON(), nullable=False, server_default="{}")
        )

    bind = op.get_bind()
    metadata = sa.MetaData()
    runs = sa.Table("project_pipeline_runs", metadata, autoload_with=bind)
    pipelines = sa.Table("project_pipelines", metadata, autoload_with=bind)
    revisions = sa.Table("project_pipeline_revisions", metadata, autoload_with=bind)
    instances = sa.Table("project_pipeline_instances", metadata, autoload_with=bind)
    rows = bind.execute(
        sa.select(
            runs.c.project_id,
            runs.c.pipeline_id,
            runs.c.revision_id,
            pipelines.c.name.label("pipeline_name"),
            revisions.c.revision,
            revisions.c.nodes,
            revisions.c.edges,
            revisions.c.variables,
            revisions.c.viewport,
            revisions.c.validation_report,
            revisions.c.content_hash,
        )
        .join(pipelines, pipelines.c.id == runs.c.pipeline_id)
        .join(revisions, revisions.c.id == runs.c.revision_id)
        .distinct()
    ).mappings()
    now = datetime.now(UTC)
    for row in rows:
        instance_id = str(uuid4())
        instance_name = (
            f"{row['pipeline_name'][:145]} · 迁移兼容实例 "
            f"R{row['revision']} [{row['revision_id'][:8]}]"
        )
        graph = {
            "nodes": row["nodes"] or [],
            "edges": row["edges"] or [],
            "variables": row["variables"] or {},
            "viewport": row["viewport"] or {},
        }
        bind.execute(
            instances.insert().values(
                id=instance_id,
                created_at=now,
                updated_at=now,
                project_id=row["project_id"],
                pipeline_id=row["pipeline_id"],
                definition_revision_id=row["revision_id"],
                name=instance_name,
                description="由迁移为已有 Pipeline 运行自动生成的只读实例",
                status="ready",
                nodes=graph["nodes"],
                edges=graph["edges"],
                variables=graph["variables"],
                viewport=graph["viewport"],
                validation_report=row["validation_report"] or {},
                content_hash=row["content_hash"],
                activated_at=now,
                locked=True,
            )
        )
        bind.execute(
            runs.update()
            .where(
                runs.c.pipeline_id == row["pipeline_id"],
                runs.c.revision_id == row["revision_id"],
            )
            .values(instance_id=instance_id, graph_snapshot=graph)
        )

    with op.batch_alter_table("project_pipeline_runs") as batch_op:
        batch_op.alter_column("instance_id", existing_type=sa.String(length=36), nullable=False)
        batch_op.create_foreign_key(
            "fk_project_pipeline_runs_instance_id_project_pipeline_instances",
            "project_pipeline_instances",
            ["instance_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "ix_project_pipeline_runs_instance_id", ["instance_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("project_pipeline_runs") as batch_op:
        batch_op.drop_index("ix_project_pipeline_runs_instance_id")
        batch_op.drop_constraint(
            "fk_project_pipeline_runs_instance_id_project_pipeline_instances",
            type_="foreignkey",
        )
        batch_op.drop_column("graph_snapshot")
        batch_op.drop_column("instance_id")
    op.drop_table("project_pipeline_instances")
