from uuid import uuid4

import pytest
from llm_platform.database import SessionLocal
from llm_platform.models import Artifact, ModelVersion, Task, utcnow
from llm_platform.services.tasks import sync_task_domain_status


def command(node_id):
    return {
        "id": node_id,
        "type": "custom",
        "label": node_id,
        "config": {
            "command": {"executable": "__worker_python__", "args": ["-c", "print('scope-test')"]},
            "resource_request": {"gpus": 0},
        },
    }


def edge(source, target, handle="out"):
    return {
        "id": f"{source}-{target}",
        "source": source,
        "target": target,
        "source_handle": handle,
        "target_handle": "in",
    }


@pytest.fixture
def graph(client, tmp_path):
    project = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "Test", "workspace_uri": str(tmp_path)},
    ).json()
    base = f"/api/projects/{project['id']}/training-instances"
    created = client.post(
        base,
        json={
            "name": "scope-test",
            "nodes": [command(n) for n in ("a", "b", "c", "d")],
            "edges": [edge("a", "b"), edge("b", "c")],
        },
    )
    assert created.status_code == 201, created.text
    instance = created.json()
    return base, instance, f"{base}/{instance['id']}/attempts"


def finish(task_id, status="succeeded", output=None):
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        task.observed_state = status
        task.ended_at = utcnow()
        task.result = output or {}
        task.error = {"message": "test failure"} if status == "failed" else {}
        sync_task_domain_status(session, task)
        session.commit()


def completed(client, base):
    response = client.post(base, json={})
    assert response.status_code == 201, response.text
    run = response.json()
    while run["status"] == "running":
        queued = [node for node in run["node_runs"] if node["status"] == "queued"]
        assert queued, run
        for node in queued:
            finish(node["task_id"], output={"marker": node["node_id"] + "-v1"})
        run = client.get(f"{base}/{run['id']}").json()
    assert run["status"] == "succeeded", run
    return run


def nodes(run):
    return {node["node_id"]: node for node in run["node_runs"]}


def test_from_node_reuses_boundary_and_excludes_other_branches(client, graph):
    base, instance, attempts = graph
    source = completed(client, attempts)
    body = {"selection_mode": "from_node", "node_ids": ["b"], "source_attempt_id": source["id"]}
    preview = client.post(attempts + "/preview", json=body).json()
    assert preview["valid"]
    assert preview["selected_node_ids"] == ["b", "c"]
    assert preview["reused_node_ids"] == ["a"]
    assert preview["excluded_node_ids"] == ["d"]
    assert len(client.get(attempts).json()) == 1  # Preview is read-only.
    response = client.post(attempts, json=body)
    assert response.status_code == 201, response.text
    run = response.json()
    rows = nodes(run)
    assert rows["a"]["task_id"] is None and rows["a"]["output_snapshot"] == {"marker": "a-v1"}
    assert rows["b"]["task_id"] != nodes(source)["b"]["task_id"]
    assert rows["b"]["input_snapshot"]["upstream"]["a"] == {"marker": "a-v1"}
    assert rows["d"]["task_id"] is None and rows["d"]["input_snapshot"]["excluded_from_execution"]
    assert rows["d"]["lineage"] is None
    assert rows["a"]["lineage"]["produced_assets"] == []
    assert run["node_summary"]["total"] == 2
    finish(rows["b"]["task_id"])
    run = client.get(f"{attempts}/{run['id']}").json()
    finish(nodes(run)["c"]["task_id"])
    assert client.get(f"{attempts}/{run['id']}").json()["status"] == "succeeded"
    after = client.get(f"{attempts}/{source['id']}").json()
    assert {k: v for k, v in after.items() if k != "instance"} == {
        k: v for k, v in source.items() if k != "instance"
    }
    assert client.get(f"{base}/{instance['id']}").json()["content_hash"] == instance["content_hash"]


def test_exact_selection_uses_explicit_historical_middle_output(client, graph):
    _, _, attempts = graph
    source = completed(client, attempts)
    response = client.post(
        attempts,
        json={
            "selection_mode": "selected",
            "node_ids": ["a", "c"],
            "source_attempt_id": source["id"],
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()
    rows = nodes(run)
    assert {n["node_id"] for n in run["node_runs"] if n["task_id"]} == {"a", "c"}
    assert rows["c"]["input_snapshot"]["upstream"]["b"] == {"marker": "b-v1"}
    assert rows["b"]["input_snapshot"]["reused_from_attempt_id"] == source["id"]
    assert rows["d"]["lineage"] is None


def test_first_partial_execution_does_not_create_reusable_excluded_results(client, graph):
    _, _, attempts = graph
    response = client.post(attempts, json={"selection_mode": "selected", "node_ids": ["a", "d"]})
    assert response.status_code == 201, response.text
    run = response.json()
    for row in run["node_runs"]:
        if row["task_id"]:
            finish(row["task_id"])
    source = client.get(f"{attempts}/{run['id']}").json()
    assert source["status"] == "succeeded"
    assert source["node_summary"]["total"] == 2
    body = {"selection_mode": "from_node", "node_ids": ["c"], "source_attempt_id": source["id"]}
    preview = client.post(attempts + "/preview", json=body).json()
    assert not preview["valid"] and preview["errors"][0]["node_id"] == "b"
    assert client.post(attempts, json=body).status_code == 409
    assert len(client.get(attempts).json()) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"selection_mode": "selected", "node_ids": []},
        {"selection_mode": "selected", "node_ids": ["unknown"]},
        {"selection_mode": "selected", "node_ids": ["a", "a"]},
        {"selection_mode": "from_node", "node_ids": ["a", "b"]},
        {"selection_mode": "all", "node_ids": ["a"]},
        {"selection_mode": "from_node", "node_ids": ["c"]},
        {"selection_mode": "selected", "node_ids": ["a"], "source_attempt_id": "missing"},
    ],
)
def test_invalid_scope_never_creates_an_attempt(client, graph, body):
    _, _, attempts = graph
    assert not client.post(attempts + "/preview", json=body).json()["valid"]
    assert client.post(attempts, json=body).status_code == 409
    assert client.get(attempts).json() == []


def test_sources_are_terminal_and_belong_to_same_instance(client, graph):
    base, _, attempts = graph
    active = client.post(attempts, json={}).json()
    body = {"selection_mode": "selected", "node_ids": ["c"], "source_attempt_id": active["id"]}
    assert (
        client.post(attempts + "/preview", json=body).json()["errors"][0]["code"]
        == "execution_source_active"
    )
    assert client.post(attempts, json=body).status_code == 409
    other = client.post(base, json={"name": "other", "nodes": [command("a")], "edges": []}).json()
    other_attempts = f"{base}/{other['id']}/attempts"
    body["node_ids"] = ["a"]
    assert not client.post(other_attempts + "/preview", json=body).json()["valid"]
    assert client.post(other_attempts, json=body).status_code == 409


def test_retry_of_partial_attempt_keeps_scope_and_history(client, graph):
    _, _, attempts = graph
    source = completed(client, attempts)
    partial = client.post(
        attempts,
        json={"selection_mode": "selected", "node_ids": ["b"], "source_attempt_id": source["id"]},
    ).json()
    task_id = nodes(partial)["b"]["task_id"]
    finish(task_id, "failed")
    failed = client.get(f"{attempts}/{partial['id']}").json()
    response = client.post(f"/api/tasks/{task_id}/control", json={"action": "retry"})
    assert response.status_code == 200, response.text
    retried = client.get(attempts).json()[0]
    assert retried["context_snapshot"]["__training_instance"]["execution_scope"][
        "selected_node_ids"
    ] == ["b"]
    assert nodes(retried)["b"]["task_id"] == task_id
    assert nodes(retried)["c"]["task_id"] is None and nodes(retried)["d"]["task_id"] is None
    finish(task_id)
    done = client.get(f"{attempts}/{retried['id']}").json()
    assert done["status"] == "succeeded" and done["node_summary"]["total"] == 1
    assert client.get(f"{attempts}/{failed['id']}").json()["node_runs"] == failed["node_runs"]


@pytest.mark.parametrize("asset_type", ["artifact", "model_version"])
def test_missing_registered_output_blocks_partial_execution(client, graph, asset_type):
    _, _, attempts = graph
    run = client.post(attempts, json={"selection_mode": "selected", "node_ids": ["a"]}).json()
    with SessionLocal() as session:
        artifact = Artifact(kind="dataset", uri=f"test://{uuid4()}", content_hash=uuid4().hex)
        session.add(artifact)
        session.commit()
        artifact_id = artifact.id
        asset_id = artifact_id
        if asset_type == "model_version":
            model = ModelVersion(
                project_id=run["project_id"],
                name="output model",
                version=1,
                artifact_id=artifact_id,
            )
            session.add(model)
            session.commit()
            asset_id = model.id
    finish(
        nodes(run)["a"]["task_id"],
        output={
            "produced_assets": [{"asset_type": asset_type, "asset_id": asset_id, "label": "output"}]
        },
    )
    body = {"selection_mode": "selected", "node_ids": ["b"], "source_attempt_id": run["id"]}
    assert client.post(attempts + "/preview", json=body).json()["valid"]
    with SessionLocal() as session:
        session.get(Artifact, artifact_id).availability_state = "missing"
        session.commit()
    assert not client.post(attempts + "/preview", json=body).json()["valid"]
    assert client.post(attempts, json=body).status_code == 409


@pytest.mark.parametrize("asset_type", ["dataset_version", "checkpoint", "model_version"])
def test_missing_asset_version_blocks_reuse(client, graph, asset_type):
    _, _, attempts = graph
    run = client.post(attempts, json={"selection_mode": "selected", "node_ids": ["a"]}).json()
    finish(
        nodes(run)["a"]["task_id"],
        output={"produced_assets": [{"asset_type": asset_type, "asset_id": str(uuid4())}]},
    )
    body = {"selection_mode": "selected", "node_ids": ["b"], "source_attempt_id": run["id"]}
    preview = client.post(attempts + "/preview", json=body).json()
    assert not preview["valid"]
    assert preview["errors"][0]["code"] == "execution_asset_unavailable"
    assert client.post(attempts, json=body).status_code == 409
    assert len(client.get(attempts).json()) == 1


def test_partial_execution_obeys_frozen_branch_and_reuses_real_skipped_path(client, graph):
    base, _, _ = graph
    branch = {
        "id": "branch",
        "label": "branch",
        "type": "branch",
        "config": {"path": "inputs.accept", "operator": "truthy"},
    }
    join = {"id": "join", "label": "join", "type": "join", "config": {}}
    created = client.post(
        base,
        json={
            "name": "branches",
            "variables": {"accept": True},
            "nodes": [branch, command("yes"), command("no"), join],
            "edges": [
                edge("branch", "yes", "true"),
                edge("branch", "no", "false"),
                edge("yes", "join"),
                edge("no", "join"),
            ],
        },
    )
    assert created.status_code == 201, created.text
    attempts = f"{base}/{created.json()['id']}/attempts"
    source = completed(client, attempts)
    assert nodes(source)["no"]["status"] == "skipped"
    response = client.post(
        attempts,
        json={
            "selection_mode": "from_node",
            "node_ids": ["yes"],
            "source_attempt_id": source["id"],
        },
    )
    assert response.status_code == 201, response.text
    partial = response.json()
    finish(nodes(partial)["yes"]["task_id"])
    assert client.get(f"{attempts}/{partial['id']}").json()["status"] == "succeeded"
    response = client.post(
        attempts,
        json={"selection_mode": "selected", "node_ids": ["no"], "source_attempt_id": source["id"]},
    )
    assert response.status_code == 201, response.text
    assert nodes(response.json())["no"]["status"] == "skipped"
    assert all(row["task_id"] is None for row in response.json()["node_runs"])
