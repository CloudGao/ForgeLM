from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import PlanRevision, ProjectPipeline, Task, TrainingRun, utcnow
from llm_platform.services.tasks import sync_task_domain_status
from sqlalchemy import func, select


def project(client, tmp_path):
    root = tmp_path / uuid4().hex
    root.mkdir()
    response = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "TestLM", "workspace_uri": str(root)},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def command_node(node_id="train"):
    return {
        "id": node_id,
        "type": "custom",
        "label": "执行命令",
        "config": {
            "command": {"executable": "__worker_python__", "args": ["-c", "print('test')"]},
            "resource_request": {"gpus": 0},
        },
    }


def finish(task_id, state, result=None):
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        task.observed_state = state
        task.result = result or {}
        task.error = {"message": "test failure"} if state == "failed" else {}
        task.ended_at = utcnow()
        sync_task_domain_status(session, task)
        session.commit()


def test_finish_running_keeps_active_task_and_stops_downstream(client, tmp_path):
    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    instance = client.post(base, json={"name": "drain", "nodes": [command_node("first"), command_node("second")],
        "edges": [{"id": "edge", "source": "first", "target": "second"}]}).json()
    run = client.post(f"{base}/{instance['id']}/attempts", json={}).json()
    task_id = next(n["task_id"] for n in run["node_runs"] if n["node_id"] == "first")
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        task.observed_state = "running"
        sync_task_domain_status(session, task)
        session.commit()
    response = client.post(f"{base}/{instance['id']}/attempts/{run['id']}/cancel?finish_running=true")
    assert response.status_code == 200, response.text
    with SessionLocal() as session:
        assert session.get(Task, task_id).desired_state == "running"
    finish(task_id, "succeeded")
    result = client.get(f"{base}/{instance['id']}/attempts/{run['id']}").json()
    assert result["status"] == "cancelled"
    assert {n["node_id"]: n["status"] for n in result["node_runs"]} == {"first": "succeeded", "second": "cancelled"}


def test_direct_instance_clone_archive_and_isolation(client: TestClient, tmp_path):
    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    with SessionLocal() as session:
        count = session.scalar(select(func.count(ProjectPipeline.id)))
    body = {"name": "实验 A", "nodes": [command_node()], "edges": []}
    created = client.post(base, json=body)
    assert created.status_code == 201, created.text
    instance = created.json()
    assert instance["locked"] and instance["status"] == "ready"
    assert instance["definition_revision_id"] is None and instance["pipeline_id"] is None
    with SessionLocal() as session:
        assert session.scalar(select(func.count(ProjectPipeline.id))) == count
    edit = client.put(
        f"/api/projects/{pid}/pipeline-instances/{instance['id']}/draft",
        json={**body, "name": "changed"},
    )
    assert edit.status_code == 409
    clone = client.post(
        base, json={**body, "name": "实验 B", "cloned_from_instance_id": instance["id"]}
    )
    assert clone.status_code == 201, clone.text
    assert clone.json()["id"] != instance["id"]
    assert clone.json()["cloned_from_instance_id"] == instance["id"]
    other = project(client, tmp_path)
    assert (
        client.get(f"/api/projects/{other}/training-instances/{instance['id']}").status_code == 404
    )
    assert (
        client.post(
            f"{base}/{instance['id']}/attempts", json={"inputs": {"changed": True}}
        ).status_code
        == 422
    )
    archived = client.patch(f"{base}/{instance['id']}/archive", json={"archived": True})
    assert archived.status_code == 200
    assert client.post(f"{base}/{instance['id']}/attempts", json={}).status_code == 409
    assert client.get(f"{base}/{instance['id']}").json()["content_hash"] == instance["content_hash"]


def test_retry_preserves_previous_execution_and_reuses_upstream(client, tmp_path):
    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    body = {
        "name": "retry",
        "nodes": [command_node("first"), command_node("second")],
        "edges": [
            {
                "id": "edge",
                "source": "first",
                "target": "second",
                "source_handle": "out",
                "target_handle": "in",
            }
        ],
    }
    response = client.post(base, json=body)
    assert response.status_code == 201, response.text
    iid = response.json()["id"]
    attempts = f"{base}/{iid}/attempts"
    started = client.post(attempts, json={})
    assert started.status_code == 201, started.text
    first = started.json()
    assert client.post(attempts, json={}).status_code == 409
    assert client.patch(f"{base}/{iid}/archive", json={"archived": True}).status_code == 409
    a = next(n for n in first["node_runs"] if n["node_id"] == "first")
    finish(
        a["task_id"],
        "succeeded",
        {
            "produced_assets": [
                {"asset_type": "model_version", "asset_id": "test-output", "label": "test model"}
            ]
        },
    )
    first = client.get(f"{attempts}/{first['id']}").json()
    b = next(n for n in first["node_runs"] if n["node_id"] == "second")
    finish(b["task_id"], "failed")
    frozen = client.get(f"{attempts}/{first['id']}").json()
    assert frozen["status"] == "failed"
    retried = client.post(attempts, json={"retry_from_attempt_id": first["id"]})
    assert retried.status_code == 201, retried.text
    second = retried.json()
    assert second["instance_id"] == iid and second["id"] != first["id"]
    reused = next(n for n in second["node_runs"] if n["node_id"] == "first")
    retried_node = next(n for n in second["node_runs"] if n["node_id"] == "second")
    assert reused["input_snapshot"]["reused_from_node_run_id"] == a["id"]
    assert reused["lineage"]["produced_assets"] == []
    assert retried_node["task_id"] == b["task_id"]
    finish(b["task_id"], "succeeded")
    assert client.get(f"{attempts}/{second['id']}").json()["status"] == "succeeded"
    after = client.get(f"{attempts}/{first['id']}").json()
    assert after["status"] == "failed"
    assert after["node_runs"] == frozen["node_runs"]


def test_native_node_creates_domain_training_task_and_retries_same_training_run(client, tmp_path):
    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    node = {
        "id": "training",
        "type": "training",
        "label": "结构化训练",
        "config": {
            "execution_mode": "native",
            "training_spec": {
                "stages": [
                    {
                        "name": "训练",
                        "type": "pretrain",
                        "target_tokens": 128,
                        "sequence_length": 8,
                        "global_batch_size": 1,
                    }
                ],
                "engine_config": {
                    "executable": "__worker_python__",
                    "args": ["-c", "print('train')"],
                },
                "resource_request": {"gpus": 0},
            },
        },
    }
    created = client.post(base, json={"name": "native", "nodes": [node], "edges": []})
    assert created.status_code == 201, created.text
    instance = created.json()
    config = instance["nodes"][0]["config"]
    with SessionLocal() as session:
        revision = session.get(PlanRevision, config["plan_revision_id"])
        assert revision.locked
        assert not session.scalar(
            select(TrainingRun).where(TrainingRun.plan_revision_id == revision.id)
        )
    assert (
        client.post(f"/api/training-plans/{config['_native_plan_id']}/runs", json={}).status_code
        == 409
    )
    attempts = f"{base}/{instance['id']}/attempts"
    started = client.post(attempts, json={})
    assert started.status_code == 201, started.text
    attempt = started.json()
    assert attempt["node_runs"][0]["status"] == "queued", attempt
    task = attempt["node_runs"][0]["task"]
    assert task["kind"] == "train"
    run_id = task["payload"]["training_run_id"]
    finish(task["id"], "failed")
    response = client.post(f"/api/tasks/{task['id']}/control", json={"action": "retry"})
    assert response.status_code == 200, response.text
    history = client.get(attempts).json()
    assert len(history) == 2
    assert history[0]["node_runs"][0]["task"]["payload"]["training_run_id"] == run_id
    assert history[1]["status"] == "failed"


def test_invalid_graph_has_no_partial_instance_or_domain_plan(client, tmp_path):
    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    invalid = {
        "name": "invalid",
        "nodes": [
            {
                "id": "training",
                "type": "training",
                "label": "训练",
                "config": {"execution_mode": "native", "training_spec": {"stages": []}},
            }
        ],
        "edges": [],
    }
    report = client.post(base + "/validate", json=invalid)
    assert report.status_code == 200 and not report.json()["valid"]
    assert client.post(base, json=invalid).status_code == 422
    assert client.get(base).json() == []
    assert client.get(f"/api/projects/{pid}/training-plans").json() == []


def test_late_heartbeat_recovers_same_execution_before_retry(client, tmp_path):
    from llm_platform.models import TaskAttempt

    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    iid = client.post(
        base, json={"name": "heartbeat", "nodes": [command_node()], "edges": []}
    ).json()["id"]
    attempts = f"{base}/{iid}/attempts"
    run = client.post(attempts, json={}).json()
    tid = run["node_runs"][0]["task_id"]
    with SessionLocal() as session:
        task = session.get(Task, tid)
        attempt = TaskAttempt(task_id=tid, attempt_number=1, status="lost")
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        task.observed_state = "lost"
        task.error = {"code": "lease_expired"}
        task.ended_at = utcnow()
        sync_task_domain_status(session, task)
        session.commit()
    lost = client.get(f"{attempts}/{run['id']}").json()
    assert lost["status"] == "failed"
    assert lost["node_runs"][0]["lineage"] is None
    with SessionLocal() as session:
        task = session.get(Task, tid)
        task.observed_state, task.error, task.ended_at = "running", {}, None
        sync_task_domain_status(session, task)
        session.commit()
    assert client.get(f"{attempts}/{run['id']}").json()["status"] == "running"
    assert len(client.get(attempts).json()) == 1
    finish(tid, "succeeded")
    done = client.get(f"{attempts}/{run['id']}").json()
    assert done["status"] == "succeeded"
    assert done["node_runs"][0]["lineage"]["status"] == "succeeded"
    assert client.post(f"{attempts}/{run['id']}/cancel").status_code == 409


def test_partial_outputs_belong_to_original_attempt_and_not_retry(client, tmp_path):
    from llm_platform.models import Artifact, TaskAttempt

    pid = project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    iid = client.post(
        base, json={"name": "partial-output", "nodes": [command_node()], "edges": []}
    ).json()["id"]
    attempts = f"{base}/{iid}/attempts"
    run = client.post(attempts, json={}).json()
    tid = run["node_runs"][0]["task_id"]
    with SessionLocal() as session:
        task = session.get(Task, tid)
        attempt = TaskAttempt(task_id=tid, attempt_number=1, status="failed")
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        artifact = Artifact(
            kind="checkpoint",
            uri=f"test://{uuid4()}",
            content_hash=uuid4().hex,
            created_by_task_attempt_id=attempt.id,
        )
        session.add(artifact)
        session.commit()
        artifact_id = artifact.id
    finish(tid, "failed")
    failed = client.get(f"{attempts}/{run['id']}").json()
    produced = failed["node_runs"][0]["lineage"]["produced_assets"]
    assert any(ref["asset_id"] == artifact_id for ref in produced)
    retried = client.post(attempts, json={"retry_from_attempt_id": run["id"]})
    assert retried.status_code == 201, retried.text
    with SessionLocal() as session:
        task = session.get(Task, tid)
        attempt = TaskAttempt(task_id=tid, attempt_number=2, status="succeeded")
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        session.commit()
    finish(tid, "succeeded")
    done = client.get(f"{attempts}/{retried.json()['id']}").json()
    assert done["node_runs"][0]["lineage"]["produced_assets"] == []
    assert client.get(f"{attempts}/{run['id']}").json()["node_runs"] == failed["node_runs"]


def test_training_view_lineage_uses_view_identity_not_mixture_identity(client):
    from llm_platform.services.training_instances import training_input_refs

    with SessionLocal() as session:
        refs = training_input_refs(
            session,
            {
                "training_data_view_inputs": [
                    {
                        "version_id": "view-1",
                        "mixture_version_id": "mixture-1",
                        "content_hash": "a" * 64,
                    }
                ]
            },
        )
    assert any(
        r["asset_type"] == "training_data_view_version" and r["asset_id"] == "view-1" for r in refs
    )
    assert any(r["asset_type"] == "mixture_version" and r["asset_id"] == "mixture-1" for r in refs)


def test_command_node_sends_frozen_code_in_worker_contract(client, tmp_path):
    from llm_platform.models import Artifact, CodeSource, CodeVersion

    pid = project(client, tmp_path)
    with SessionLocal() as session:
        artifact = Artifact(kind="source_code", uri=f"test://{uuid4()}", content_hash=uuid4().hex)
        source = CodeSource(
            project_id=pid, name="code", source_type="local", source_uri=str(tmp_path)
        )
        session.add_all([artifact, source])
        session.flush()
        version = CodeVersion(
            source_id=source.id, version=1, state="ready", artifact_id=artifact.id
        )
        session.add(version)
        session.commit()
        version_id, artifact_id, digest = version.id, artifact.id, artifact.content_hash
    node = command_node()
    node["config"]["code_version_id"] = version_id
    base = f"/api/projects/{pid}/training-instances"
    created = client.post(base, json={"name": "code-contract", "nodes": [node], "edges": []})
    assert created.status_code == 201, created.text
    started = client.post(f"{base}/{created.json()['id']}/attempts", json={})
    assert started.status_code == 201, started.text
    code = started.json()["node_runs"][0]["task"]["payload"]["code"]
    assert code["version_id"] == version_id
    assert code["artifact_id"] == artifact_id and code["content_hash"] == digest
