from datetime import timedelta

import pytest
from llm_platform.database import Base
from llm_platform.models import (
    Artifact,
    ArtifactLocation,
    Project,
    ResourceLease,
    Task,
    TaskAttempt,
    WorkerNode,
    utcnow,
)
from llm_platform.services.storage_lifecycle import (
    estimated_write_bytes,
    reconcile_storage_maintenance,
    replica_eligibility,
)
from llm_platform.services.tasks import _task_allocation
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


def test_replica_retention_keeps_one_copy_and_blocks_active_shared_consumers(session):
    old = utcnow() - timedelta(days=2)
    project = Project(name="p", target_model_name="m", storage_policy={"grace_hours": 0})
    artifact = Artifact(kind="tokenized_corpus", uri="file:///original", content_hash="a" * 64)
    session.add_all([project, artifact])
    session.flush()
    locations = [ArtifactLocation(id=str(i), artifact_id=artifact.id, project_id=project.id,
                 uri=f"file:///copy-{i}", availability_state="available", last_verified_at=old,
                 created_at=old, updated_at=old) for i in (1, 2)]
    session.add_all(locations)
    session.flush()
    assert not replica_eligibility(session, locations[0], artifact)[0]
    assert replica_eligibility(session, locations[1], artifact)[0]
    consumer = Task(project_id=project.id, kind="train", title="shared consumer", observed_state="queued")
    session.add(consumer)
    session.flush()
    assert not replica_eligibility(session, locations[1], artifact)[0]
    consumer.observed_state = "succeeded"
    locations[1].metadata_json = {"retention": "retain"}
    session.flush()
    assert not replica_eligibility(session, locations[1], artifact)[0]
    locations[1].metadata_json = {}
    locations[0].availability_state = "missing"
    session.flush()
    assert not replica_eligibility(session, locations[1], artifact)[0]


def test_space_budget_waits_without_blocking_read_only_jobs(session):
    worker = WorkerNode(name="w", hostname="h", resources={"cpu_count": 8,
        "workspace": {"free_bytes": 10 * 1024**3}}, capabilities=["a", "builtin.local-process"])
    project = Project(name="p", target_model_name="m")
    session.add_all([worker, project])
    session.flush()
    task = Task(project_id=project.id, kind="train", title="writer", adapter_id="a",
                observed_state="queued", payload={"resource_request": {"disk_bytes": 20 * 1024**3}})
    session.add(task)
    session.flush()
    assert estimated_write_bytes(session, task) == 20 * 1024**3
    assert _task_allocation(session, worker, task) is None
    assert task.current_phase == "waiting_for_storage"
    task.payload = {"resource_request": {}}
    assert _task_allocation(session, worker, task)["disk_bytes"] == 0


def test_expired_automatic_inventory_is_rescanned(session, tmp_path):
    from sqlalchemy import select
    old = utcnow() - timedelta(hours=2)
    project = Project(name="p", target_model_name="m", workspace_uri=str(tmp_path))
    worker = WorkerNode(name="w", hostname="h", capabilities=["builtin.storage-maintenance"])
    session.add_all([project, worker])
    session.flush()
    producer = Task(project_id=project.id, kind="data_process", title="done", observed_state="succeeded",
                    adapter_id="builtin.data-juicer", ended_at=old, created_at=old)
    session.add(producer)
    session.flush()
    session.add(TaskAttempt(task_id=producer.id, worker_id=worker.id, status="succeeded", attempt_number=1))
    stale = Task(project_id=project.id, kind="storage_maintenance", title="old scan",
                 adapter_id="builtin.storage-maintenance", observed_state="succeeded", ended_at=old,
                 created_at=old, payload={"automatic": True, "resource_request": {
                     "worker_selector": {"worker_id": worker.id}}}, result={
                     "storage_action": "scan", "items": [{"id": "x", "reclaimable": True}]})
    session.add(stale)
    session.commit()
    reconcile_storage_maintenance(session)
    fresh = session.scalar(select(Task).where(Task.observed_state == "queued"))
    assert fresh is not None
    assert fresh.payload["storage_action"] == "scan"
    assert fresh.payload["automatic"] is True


def test_maintenance_excludes_sibling_worker_using_same_host(session):
    project = Project(name="p", target_model_name="m")
    workers = [WorkerNode(name=f"w{i}", hostname="shared-host", resources={"cpu_count": 8},
                          capabilities=["builtin.storage-maintenance", "a", "builtin.local-process"])
               for i in range(2)]
    session.add_all([project, *workers])
    session.flush()
    active = Task(project_id=project.id, title="writer", kind="data_process", adapter_id="a")
    session.add(active)
    session.flush()
    attempt = TaskAttempt(task_id=active.id, worker_id=workers[0].id, attempt_number=1, status="running")
    session.add(attempt)
    session.flush()
    lease = ResourceLease(worker_id=workers[0].id, task_attempt_id=attempt.id,
                          expires_at=utcnow() + timedelta(hours=1), allocation={"cpus": 2})
    session.add(lease)
    maintenance = Task(project_id=project.id, title="cleanup", kind="storage_maintenance",
                        adapter_id="builtin.storage-maintenance", payload={})
    session.add(maintenance)
    session.flush()
    assert _task_allocation(session, workers[1], maintenance) is None
    active.adapter_id = "builtin.storage-maintenance"
    maintenance.adapter_id = "a"
    session.flush()
    assert _task_allocation(session, workers[1], maintenance) is None
    lease.released_at = utcnow()
    session.flush()
    assert _task_allocation(session, workers[1], maintenance) is not None
