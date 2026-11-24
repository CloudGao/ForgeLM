from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_platform.database import Base
from llm_platform.models import WorkerNode, Task, TaskAttempt, ResourceLease, utcnow
from llm_platform.services.tasks import _available_gpu_indexes
from llm_platform.services.inference_residency import prefer_warm_worker, reclaim_before_allocation


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def worker(session, name, host, uuid="GPU-1", warm=False):
    node = WorkerNode(name=name, hostname=host, status="online", last_heartbeat_at=utcnow(),
        resources={"gpus": [{"index": 0, "uuid": uuid}], "cpu_count": 4,
            "inference_runtimes": [{"model_id": "model", "adapter_id": "builtin.microlm-inference",
                "device": "cuda", "gpu_indexes": [0], "requested_dtype": "auto"}] if warm else []},
        capabilities=["builtin.microlm-inference", "builtin.local-process"])
    session.add(node)
    session.flush()
    return node


def test_physical_gpu_lease_blocks_sibling_with_different_hostname(session):
    a, b = worker(session, "a", "windows"), worker(session, "b", "wsl")
    c = worker(session, "c", "remote", "GPU-2")
    task = Task(kind="inference", title="test")
    session.add(task); session.flush()
    attempt = TaskAttempt(task_id=task.id, attempt_number=1, worker_id=a.id)
    session.add(attempt); session.flush()
    lease = ResourceLease(worker_id=a.id, task_attempt_id=attempt.id,
        allocation={"gpu_indexes": [0]}, expires_at=utcnow() + timedelta(seconds=45))
    session.add(lease); session.flush()
    assert _available_gpu_indexes(session, b) == []
    assert _available_gpu_indexes(session, c) == [0]
    lease.status = "released"; session.flush()
    assert _available_gpu_indexes(session, b) == [0]


def test_reclaim_requires_empty_cache_ack_and_warm_affinity_obeys_selector(session):
    a, b = worker(session, "a", "host", warm=True), worker(session, "b", "host")
    task = Task(kind="inference", title="test", adapter_id="builtin.microlm-inference",
        launcher_id="builtin.local-process", payload={"model": {"id": "model"}, "resource_request": {"gpus": 1}})
    assert prefer_warm_worker(session, b, task)
    task.payload = {**task.payload, "resource_request": {"gpus": 1, "worker_selector": {"worker_id": b.id}}}
    assert not prefer_warm_worker(session, b, task)
    assert reclaim_before_allocation(session, b, [0])
    assert a.labels["inference_reclaim_requested"]
    assert reclaim_before_allocation(session, b, [0])
    a.resources = {**a.resources, "inference_runtimes": []}
    assert not reclaim_before_allocation(session, b, [0])


def test_stale_warm_worker_does_not_capture_routing(session):
    a, b = worker(session, "a", "host", warm=True), worker(session, "b", "remote", "GPU-2")
    a.last_heartbeat_at = utcnow() - timedelta(minutes=1)
    task = Task(kind="inference", title="test", adapter_id="builtin.microlm-inference",
        launcher_id="builtin.local-process", payload={"model": {"id": "model"}})
    assert not prefer_warm_worker(session, b, task)
