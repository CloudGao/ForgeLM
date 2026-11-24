import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from llm_platform.database import SessionLocal
from llm_platform.models import ModelVersion, Project, ResourceLease, Task, TaskEvent, WorkerNode
from llm_platform.services import worker_dispatch
from llm_platform.services.tasks import create_task

HEADERS = {"X-Agent-Token": "local-development-token"}
_projects = []
_workers = []


@pytest.fixture(autouse=True)
def finish_test_queues():
    yield
    with SessionLocal() as session:
        session.execute(update(Task).where(Task.project_id.in_(_projects)).values(observed_state="cancelled"))
        session.execute(update(ResourceLease).where(ResourceLease.worker_id.in_(_workers)).values(status="released"))
        session.commit()
    _projects.clear()
    _workers.clear()


def setup(client, tmp_path):
    worker = client.post("/api/agent/register", headers=HEADERS, json={
        "name": f"dispatch-{uuid4()}", "hostname": str(uuid4()),
        "resources": {"cpu_count": 4, "gpus": []},
        "capabilities": ["builtin.transformers-inference", "builtin.local-process"],
    }).json()
    with SessionLocal() as session:
        project = Project(name=str(uuid4()), target_model_name="test", workspace_uri=str(tmp_path))
        session.add(project); session.flush()
        model = ModelVersion(project_id=project.id, name="tiny", version=1, kind="sft",
            metadata_json={"inference": {"pretrained": "test/model"}})
        session.add(model); session.commit()
        _projects.append(project.id)
        _workers.append(worker["id"])
        return worker["id"], project.id, model.id


def generate(client, worker, project, model):
    return client.post(f"/api/projects/{project}/playground/generate", json={
        "model_version_ids": [model], "prompt": "hello",
        "resource_request": {"gpus": 0, "worker_selector": {"worker_id": worker}},
        "config": {"device": "cpu", "max_tokens": 4},
    })


def wait_subscriber(worker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with worker_dispatch._lock:
            if worker_dispatch._listeners.get(worker):
                return
        time.sleep(.01)
    raise AssertionError("Worker did not establish its waiting connection")


def test_committed_playground_request_is_delivered_without_claim_poll(client, tmp_path):
    worker, project, model = setup(client, tmp_path)
    with ThreadPoolExecutor() as pool:
        waiting = pool.submit(client.get, f"/api/agent/workers/{worker}/dispatch", headers=HEADERS)
        wait_subscriber(worker)
        started = time.monotonic()
        created = generate(client, worker, project, model)
        assert created.status_code == 202, created.text
        response = waiting.result(timeout=3)
        assert time.monotonic() - started < 3  # Well below the 25-second connection renewal.
    assert response.status_code == 200
    assigned = response.json()
    assert assigned["task"]["id"] == created.json()["task"]["id"]
    assert assigned["task"]["project_id"] == project
    assert assigned["attempt"]["worker_id"] == worker
    with SessionLocal() as session:
        event = session.scalar(select(TaskEvent).where(TaskEvent.task_id == assigned["task"]["id"],
            TaskEvent.event_type == "task.assigned"))
        assert event.payload["delivery"] == "direct"
    # A duplicate/legacy connection cannot obtain another execution slot.
    second = generate(client, worker, project, model)
    assert second.status_code == 202
    assert client.post("/api/agent/tasks/claim", headers=HEADERS, json={"worker_id": worker}).json() is None
    # A busy Worker reconnects after completing its execution and receives its next request.
    complete = client.post(f"/api/agent/tasks/{assigned['task']['id']}/complete", headers=HEADERS,
        json={"worker_id": worker, "attempt_id": assigned["attempt"]["id"], "status": "failed",
              "error": {"message": "test completion"}})
    assert complete.status_code == 200, complete.text
    following = client.get(f"/api/agent/workers/{worker}/dispatch", headers=HEADERS).json()
    assert following["task"]["id"] == second.json()["task"]["id"]
    with SessionLocal() as session:
        assert len(session.scalars(select(ResourceLease).where(ResourceLease.worker_id == worker,
            ResourceLease.status == "active")).all()) == 1


def test_dispatch_auth_scope_and_explicit_offline_worker(client, tmp_path):
    worker, project, model = setup(client, tmp_path)
    with SessionLocal() as session:
        node = session.get(WorkerNode, worker)
        node.token_hash = hashlib.sha256(b"scoped-secret").hexdigest()
        node.status = "offline"
        session.commit()
    assert client.get(f"/api/agent/workers/{worker}/dispatch").status_code == 401
    denied = client.get("/api/agent/workers/another-worker/dispatch", headers={
        "X-Agent-Token": "scoped-secret", "X-Worker-ID": worker})
    assert denied.status_code == 403
    response = generate(client, worker, project, model)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "worker_unavailable"


def test_task_notification_waits_for_commit_and_rollback_discards_it(client, tmp_path, monkeypatch):
    worker, project, _ = setup(client, tmp_path)
    notifications = []
    monkeypatch.setattr(worker_dispatch, "notify", lambda: notifications.append(True))
    with SessionLocal() as session:
        create_task(session, project_id=project, kind="inference", title="rollback",
            adapter_id="builtin.transformers-inference")
        assert not notifications
        session.rollback()
        session.commit()
        assert not notifications
        create_task(session, project_id=project, kind="inference", title="commit",
            adapter_id="builtin.transformers-inference")
        assert not notifications
        session.commit()
        assert notifications == [True]


def test_targeted_dispatch_is_not_hidden_behind_unrelated_queue(client, tmp_path):
    worker, project, model = setup(client, tmp_path)
    with SessionLocal() as session:
        for index in range(105):
            session.add(Task(project_id=project, title=f"unrelated {index}", kind="test",
                observed_state="queued", adapter_id="not-supported", payload={}))
        session.commit()
    created = generate(client, worker, project, model)
    assert created.status_code == 202
    assigned = client.get(f"/api/agent/workers/{worker}/dispatch", headers=HEADERS).json()
    assert assigned["task"]["id"] == created.json()["task"]["id"]
