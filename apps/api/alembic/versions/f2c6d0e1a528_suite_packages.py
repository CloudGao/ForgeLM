"""Versioned suite packages and deployment lifecycle."""
import sqlalchemy as sa
from alembic import op

revision = "f2c6d0e1a528"
down_revision = "e1b5c9d0f417"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('suites',
        sa.Column('suite_id', sa.String(length=180), nullable=False),
        sa.Column('name', sa.String(length=180), nullable=False),
        sa.Column('publisher', sa.String(length=180), nullable=False),
        sa.Column('archived', sa.Boolean(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('suite_id'),
    )
    op.create_table('suite_package_imports',
        sa.Column('artifact_id', sa.String(length=36), sa.ForeignKey('artifacts.id'), nullable=False),
        sa.Column('package', sa.JSON(), nullable=False),
        sa.Column('registered_version_id', sa.String(length=36), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table('suite_versions',
        sa.Column('suite_id', sa.String(length=36), sa.ForeignKey('suites.id'), nullable=False),
        sa.Column('version', sa.String(length=80), nullable=False),
        sa.Column('artifact_id', sa.String(length=36), sa.ForeignKey('artifacts.id'), nullable=False),
        sa.Column('digest', sa.String(length=64), nullable=False),
        sa.Column('package', sa.JSON(), nullable=False),
        sa.Column('trusted', sa.Boolean(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('suite_id', 'version'),
    )
    op.create_index('ix_suite_versions_suite_id', 'suite_versions', ['suite_id'])
    op.create_table('suite_components',
        sa.Column('suite_version_id', sa.String(length=36), sa.ForeignKey('suite_versions.id'), nullable=False),
        sa.Column('component_id', sa.String(length=180), nullable=False),
        sa.Column('spec', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('suite_version_id', 'component_id'),
    )
    op.create_index('ix_suite_components_suite_version_id', 'suite_components', ['suite_version_id'])
    op.create_table('worker_suite_deployments',
        sa.Column('worker_id', sa.String(length=36), sa.ForeignKey('worker_nodes.id'), nullable=False),
        sa.Column('suite_id', sa.String(length=36), sa.ForeignKey('suites.id'), nullable=False),
        sa.Column('default_version_id', sa.String(length=36), sa.ForeignKey('suite_versions.id'), nullable=True),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('active_operation_id', sa.String(length=36), nullable=True),
        sa.Column('blocked_versions', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('worker_id', 'suite_id'),
    )
    op.create_index('ix_worker_suite_deployments_worker_id', 'worker_suite_deployments', ['worker_id'])
    op.create_table('suite_installations',
        sa.Column('deployment_id', sa.String(length=36), sa.ForeignKey('worker_suite_deployments.id'), nullable=False),
        sa.Column('suite_version_id', sa.String(length=36), sa.ForeignKey('suite_versions.id'), nullable=False),
        sa.Column('generation', sa.String(length=36), nullable=False),
        sa.Column('receipt', sa.JSON(), nullable=False),
        sa.Column('state', sa.String(length=40), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('deployment_id', 'suite_version_id'),
    )
    op.create_index('ix_suite_installations_suite_version_id', 'suite_installations', ['suite_version_id'])
    op.create_index('ix_suite_installations_deployment_id', 'suite_installations', ['deployment_id'])
    op.create_table('suite_operation_plans',
        sa.Column('action', sa.String(length=40), nullable=False),
        sa.Column('spec', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table('suite_operations',
        sa.Column('plan_id', sa.String(length=36), sa.ForeignKey('suite_operation_plans.id'), nullable=False),
        sa.Column('idempotency_key', sa.String(length=180), nullable=False),
        sa.Column('action', sa.String(length=40), nullable=False),
        sa.Column('status', sa.String(length=40), nullable=False),
        sa.Column('spec', sa.JSON(), nullable=False),
        sa.Column('results', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('idempotency_key'),
    )
    op.create_table('suite_operation_items',
        sa.Column('operation_id', sa.String(length=36), sa.ForeignKey('suite_operations.id'), nullable=False),
        sa.Column('deployment_id', sa.String(length=36), sa.ForeignKey('worker_suite_deployments.id'), nullable=False),
        sa.Column('task_id', sa.String(length=36), sa.ForeignKey('tasks.id'), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=40), nullable=False),
        sa.Column('result', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('task_id'),
    )
    op.create_index('ix_suite_operation_items_operation_id', 'suite_operation_items', ['operation_id'])
    op.create_table('suite_config_revisions',
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('suite_version_id', sa.String(length=36), sa.ForeignKey('suite_versions.id'), nullable=False),
        sa.Column('node_id', sa.String(length=180), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('previous_revision_id', sa.String(length=36), sa.ForeignKey('suite_config_revisions.id'), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_suite_config_revisions_suite_version_id', 'suite_config_revisions', ['suite_version_id'])
    op.create_index('ix_suite_config_revisions_project_id', 'suite_config_revisions', ['project_id'])
    op.create_table('suite_environment_leases',
        sa.Column('installation_id', sa.String(length=36), sa.ForeignKey('suite_installations.id'), nullable=False),
        sa.Column('task_id', sa.String(length=36), sa.ForeignKey('tasks.id'), nullable=False),
        sa.Column('state', sa.String(length=40), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('installation_id', 'task_id'),
    )
    op.create_index('ix_suite_environment_leases_task_id', 'suite_environment_leases', ['task_id'])
    op.create_index('ix_suite_environment_leases_installation_id', 'suite_environment_leases', ['installation_id'])

    op.create_table('suite_service_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('task_id', sa.String(36), sa.ForeignKey('tasks.id'), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(40), nullable=False),
        sa.Column('chunks', sa.JSON(), nullable=False),
        sa.Column('error', sa.JSON(), nullable=False),
        sa.Column('cancel_requested', sa.Boolean(), nullable=False),
        sa.Column('attempt_id', sa.String(36), nullable=True),
    )
    op.create_index('ix_suite_service_requests_task_id', 'suite_service_requests', ['task_id'])

def downgrade():
    op.drop_table('suite_service_requests')
    op.drop_table('suite_environment_leases')
    op.drop_table('suite_config_revisions')
    op.drop_table('suite_operation_items')
    op.drop_table('suite_operations')
    op.drop_table('suite_operation_plans')
    op.drop_table('suite_installations')
    op.drop_table('worker_suite_deployments')
    op.drop_table('suite_components')
    op.drop_table('suite_versions')
    op.drop_table('suite_package_imports')
    op.drop_table('suites')
