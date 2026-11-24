from uuid import uuid4

from llm_platform.database import SessionLocal
from llm_platform.models import Project, Task, WorkerNode
from llm_platform.services.tasks import claim_task


def test_worker_scan_and_exact_cleanup_api(client, tmp_path):
    with SessionLocal() as session:
        p = Project(name=uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        w = WorkerNode(name=uuid4().hex, hostname=uuid4().hex, resources={"cpu_count": 8},
                       capabilities=["builtin.storage-maintenance", "builtin.local-process"])
        session.add_all([p, w])
        session.commit()
        pid, wid = p.id, w.id
    base = f"/api/storage/projects/{pid}"
    assert client.get(f"{base}/policy").json()["checkpoint_retention"] == "retain"
    assert client.put(f"{base}/policy", json={"cache_cleanup": "retain"}).status_code == 200
    scan = client.post(f"{base}/worker-jobs", json={"worker_id": wid}).json()
    with SessionLocal() as session:
        claimed = claim_task(session, wid)
        assert claimed and claimed[0].id == scan["id"]
        attempt_id = claimed[1].id
        session.commit()
    queued = client.post(f"{base}/worker-jobs", json={"worker_id": wid}).json()
    with SessionLocal() as session:
        assert claim_task(session, wid) is None  # exclusive maintenance lease
    identity = {"worker_id": wid, "attempt_id": attempt_id}
    headers = {"X-Agent-Token": "local-development-token"}
    assert client.post(f"/api/agent/tasks/{scan['id']}/start", json=identity,
                       headers=headers).status_code == 200
    completed = client.post(f"/api/agent/tasks/{scan['id']}/complete", headers=headers,
        json={**identity, "status": "succeeded", "result": {"storage_action": "scan", "items": [
            {"id": "eligible", "path": "/safe/cache", "reclaimable": True},
            {"id": "protected", "path": "/weights", "reclaimable": False,
             "location_id": "registered-location-is-not-a-cleanup-result"},
        ]}})
    assert completed.status_code == 200, completed.text
    assert client.post(f"{base}/worker-jobs", json={"worker_id": wid, "action": "cleanup",
        "scan_task_id": scan["id"], "item_ids": ["protected"]}).status_code == 422
    accepted = client.post(f"{base}/worker-jobs", json={"worker_id": wid, "action": "cleanup",
        "scan_task_id": scan["id"], "item_ids": ["eligible"]})
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["payload"]["candidates"][0]["path"] == "/safe/cache"
    assert queued["id"] != scan["id"]


def test_unclaimed_storage_wait_is_persisted(client, tmp_path):
    with SessionLocal() as session:
        p = Project(name=uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        w = WorkerNode(name=uuid4().hex, hostname=uuid4().hex, resources={"cpu_count": 8,
            "workspace": {"free_bytes": 1024}}, capabilities=["budget-test", "builtin.local-process"])
        session.add_all([p, w])
        session.flush()
        t = Task(project_id=p.id, kind="data_process", title="needs space", adapter_id="budget-test",
                 observed_state="queued", desired_state="running", payload={"resource_request": {
                     "disk_bytes": 1000000, "worker_selector": {"worker_id": w.id}}})
        session.add(t)
        session.commit()
        wid, tid = w.id, t.id
    response = client.post("/api/agent/tasks/claim", json={"worker_id": wid},
                            headers={"X-Agent-Token": "local-development-token"})
    assert response.status_code == 200, response.text
    assert response.json() is None
    with SessionLocal() as session:
        assert session.get(Task, tid).current_phase == "waiting_for_storage"


def test_generation_maintenance_ownership_pins_and_submission_guard(client, tmp_path):
    import pytest
    from fastapi import HTTPException
    from llm_platform.models import TaskAttempt, utcnow
    from llm_platform.services.storage_lifecycle import (
        generation_storage_payload,
        guard_generation_cleanup,
    )

    with SessionLocal() as session:
        p = Project(name=uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        w = WorkerNode(name=uuid4().hex, hostname=uuid4().hex)
        session.add_all([p, w])
        session.flush()
        t = Task(project_id=p.id, title="paused", kind="data_generation", observed_state="paused",
                 payload={"output_dataset_version_id":str(uuid4()), "generation": {},
                          "generation_input": {}, "generation_model": {}})
        session.add(t)
        session.flush()
        a = TaskAttempt(task_id=t.id, worker_id=w.id, attempt_number=1, status="paused", ended_at=utcnow())
        session.add(a)
        session.commit()
        owners = generation_storage_payload(session, p.id, w.id)["generation_attempts"]
        assert len(owners) == 1 and not owners[0]["pinned"]
        assert not generation_storage_payload(session, p.id, str(uuid4()))["generation_attempts"]
        maintenance = Task(project_id=p.id, title="cleanup", kind="storage_maintenance",
            observed_state="queued", adapter_id="builtin.storage-maintenance", payload={"candidates": [{
                "category":"generation_duplicate", "path":f"/work/{uuid4()}-{a.id}"}]})
        session.add(maintenance)
        session.commit()
        # Inventory itself does not create a source dependency.
        assert not generation_storage_payload(session,p.id,w.id)["generation_attempts"][0]["pinned"]
        with pytest.raises(HTTPException):
            guard_generation_cleanup(session, {"generation_continuation":{"attempt_id":a.id}})
        maintenance.observed_state = "succeeded"
        continuation = Task(project_id=p.id,title="consumer",kind="data_generation",observed_state="queued",
                            payload={"generation_continuation":{"attempt_id":a.id}})
        session.add(continuation)
        session.commit()
        assert generation_storage_payload(session,p.id,w.id)["generation_attempts"][0]["pinned"]
        t.observed_state = "running"
        session.commit()
        assert not generation_storage_payload(session,p.id,w.id)["generation_attempts"]


def test_cancelled_native_data_node_restarts_only_under_workflow_retry(client, tmp_path):
    import pytest
    from fastapi import HTTPException
    from llm_platform.routers.tasks import control_task
    from llm_platform.schemas import TaskControl
    with SessionLocal() as session:
        p = Project(name=uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        session.add(p)
        session.flush()
        t = Task(project_id=p.id, kind="data_process", title="cancelled data node",
                 adapter_id="builtin.data-juicer", observed_state="cancelled",
                 desired_state="cancelled", payload={}, progress=0.8)
        session.add(t)
        session.commit()
        with pytest.raises(HTTPException):
            control_task(t.id, TaskControl(action="resume"), session)
        session.info["training_attempt_control"] = True
        result = control_task(t.id, TaskControl(action="resume"), session)
        assert result["observed_state"] == "queued"
        assert result["progress"] == 0
