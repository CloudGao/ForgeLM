"""Package identities, deployment generations and immutable suite configuration."""
from sqlalchemy import JSON, Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base
from .models import RecordMixin


class Suite(RecordMixin, Base):
    __tablename__ = "suites"
    suite_id: Mapped[str] = mapped_column(String(180), unique=True)
    name: Mapped[str] = mapped_column(String(180))
    publisher: Mapped[str] = mapped_column(String(180))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)


class SuitePackageImport(RecordMixin, Base):
    __tablename__ = "suite_package_imports"
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"))
    package: Mapped[dict] = mapped_column(JSON)
    registered_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class SuiteVersion(RecordMixin, Base):
    __tablename__ = "suite_versions"
    suite_id: Mapped[str] = mapped_column(ForeignKey("suites.id"), index=True)
    version: Mapped[str] = mapped_column(String(80))
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"))
    digest: Mapped[str] = mapped_column(String(64))
    package: Mapped[dict] = mapped_column(JSON)
    trusted: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    __table_args__ = (UniqueConstraint("suite_id", "version"),)


class SuiteComponent(RecordMixin, Base):
    __tablename__ = "suite_components"
    suite_version_id: Mapped[str] = mapped_column(ForeignKey("suite_versions.id"), index=True)
    component_id: Mapped[str] = mapped_column(String(180))
    spec: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("suite_version_id", "component_id"),)


class WorkerSuiteDeployment(RecordMixin, Base):
    __tablename__ = "worker_suite_deployments"
    worker_id: Mapped[str] = mapped_column(ForeignKey("worker_nodes.id"), index=True)
    suite_id: Mapped[str] = mapped_column(ForeignKey("suites.id"))
    default_version_id: Mapped[str | None] = mapped_column(ForeignKey("suite_versions.id"), nullable=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    active_operation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    blocked_versions: Mapped[list] = mapped_column(JSON, default=list)
    __table_args__ = (UniqueConstraint("worker_id", "suite_id"),)


class SuiteInstallation(RecordMixin, Base):
    __tablename__ = "suite_installations"
    deployment_id: Mapped[str] = mapped_column(ForeignKey("worker_suite_deployments.id"), index=True)
    suite_version_id: Mapped[str] = mapped_column(ForeignKey("suite_versions.id"), index=True)
    generation: Mapped[str] = mapped_column(String(36))
    receipt: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(40), default="ready")
    __table_args__ = (UniqueConstraint("deployment_id", "suite_version_id"),)


class SuiteOperationPlan(RecordMixin, Base):
    __tablename__ = "suite_operation_plans"
    action: Mapped[str] = mapped_column(String(40))
    spec: Mapped[dict] = mapped_column(JSON)


class SuiteOperation(RecordMixin, Base):
    __tablename__ = "suite_operations"
    plan_id: Mapped[str] = mapped_column(ForeignKey("suite_operation_plans.id"))
    idempotency_key: Mapped[str] = mapped_column(String(180), unique=True)
    action: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(40), default="queued")
    spec: Mapped[dict] = mapped_column(JSON)
    results: Mapped[list] = mapped_column(JSON, default=list)


class SuiteOperationItem(RecordMixin, Base):
    __tablename__ = "suite_operation_items"
    operation_id: Mapped[str] = mapped_column(ForeignKey("suite_operations.id"), index=True)
    deployment_id: Mapped[str] = mapped_column(ForeignKey("worker_suite_deployments.id"))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True)
    generation: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40), default="queued")
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class SuiteConfigRevision(RecordMixin, Base):
    __tablename__ = "suite_config_revisions"
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    suite_version_id: Mapped[str] = mapped_column(ForeignKey("suite_versions.id"), index=True)
    node_id: Mapped[str] = mapped_column(String(180))
    config: Mapped[dict] = mapped_column(JSON)
    requested_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    previous_revision_id: Mapped[str | None] = mapped_column(ForeignKey("suite_config_revisions.id"), nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")


class SuiteEnvironmentLease(RecordMixin, Base):
    __tablename__ = "suite_environment_leases"
    installation_id: Mapped[str] = mapped_column(ForeignKey("suite_installations.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    state: Mapped[str] = mapped_column(String(40), default="active")
    __table_args__ = (UniqueConstraint("installation_id", "task_id"),)


class SuiteServiceRequest(RecordMixin, Base):
    __tablename__ = "suite_service_requests"
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(40), default="queued")
    chunks: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
