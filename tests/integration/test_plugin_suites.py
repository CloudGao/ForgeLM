from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import PluginVersion, Project, Task, WorkerNode, WorkerPluginInstallation
from llm_platform.services.plugin_suites import register_builtin_suites


def setup_worker(tmp_path, capabilities=None):
    with SessionLocal() as session:
        project = Project(name=f"Suites-{uuid4().hex}", target_model_name="TestLM", workspace_uri=str(tmp_path))
        worker = WorkerNode(name=f"Suite worker {uuid4().hex}", hostname="test", status="online", capabilities=capabilities or ["builtin.plugin-manager"])
        session.add_all([project, worker])
        session.commit()
        return project.id, worker.id


def suite(client, name):
    return next(item for item in client.get("/api/plugin-suites").json() if item["name"] == name)


def test_suite_manifest_is_fixed_and_shared_components_are_references(client: TestClient):
    hf = suite(client, "Transformers")
    assert not any(item["name"] == "MicroLM" for item in client.get("/api/plugin-suites").json())
    assert not any(component["node_kind"] == "training" for component in hf["components"])
    with SessionLocal() as session:
        original = deepcopy(session.get(PluginVersion, hf["id"]).capabilities)
        register_builtin_suites(session)
        assert session.get(PluginVersion, hf["id"]).capabilities == original
    assert client.get(f"/api/plugin-suites/{hf['id']}").json()["manifest_hash"] == hf["manifest_hash"]
    response = client.post("/api/plugins", json={"plugin_id": "forged-suite", "kind": "plugin_suite", "version": "0.1.0", "entrypoint": ""})
    assert response.status_code == 403


def test_suite_install_freezes_origin_and_reuses_existing_work(client: TestClient, tmp_path):
    project_id, worker_id = setup_worker(tmp_path)
    selected_suite = hf = suite(client, "Transformers")
    url = f"/api/projects/{project_id}/plugin-suites/{selected_suite['id']}/installations"
    first = client.post(url, json={"worker_id": worker_id, "component_keys": ["quantization", "export"]})
    assert first.status_code == 201
    assert [item["status"] for item in first.json()["results"]] == ["queued", "queued"]
    repeated = client.post(url, json={"worker_id": worker_id, "component_keys": ["quantization", "export"]}).json()
    assert all(item["status"] == "in_progress" for item in repeated["results"])
    assert [item["task_id"] for item in repeated["results"]] == [item["task_id"] for item in first.json()["results"]]
    shared = client.post(f"/api/projects/{project_id}/plugin-suites/{hf['id']}/installations", json={"worker_id": worker_id, "component_keys": ["export"]}).json()
    assert shared["results"][0]["task_id"] == first.json()["results"][1]["task_id"]
    tasks = client.get(f"/api/tasks?project_id={project_id}&kind=plugin_install").json()
    assert len(tasks) == 2
    assert all(task["payload"]["plugin_suite"]["manifest_hash"] == selected_suite["manifest_hash"] for task in tasks)
    assert all(task["payload"]["plugin_suite"]["suite_version_id"] == selected_suite["id"] for task in tasks)
    assert client.get(f"/api/projects/{project_id}/plugin-bindings").json() == []


def test_suite_deployment_distinguishes_builtin_project_code_and_installation(client: TestClient, tmp_path):
    project_id, worker_id = setup_worker(tmp_path, ["builtin.plugin-manager", "builtin.command.training", "builtin.transformers-inference"])
    selected_suite = suite(client, "Transformers")
    quantization = next(item for item in selected_suite["components"] if item["key"] == "quantization")
    with SessionLocal() as session:
        session.add(WorkerPluginInstallation(worker_id=worker_id, plugin_id=quantization["plugin_id"], state="ready", current_plugin_version_id=quantization["plugin_version_id"], installed_versions=[{"plugin_version_id": quantization["plugin_version_id"]}]))
        session.commit()
    row = next(item for item in client.get("/api/plugin-suite-deployments").json() if item["worker_id"] == worker_id and item["suite_version_id"] == selected_suite["id"])
    states = {item["key"]: item["state"] for item in row["components"]}
    assert row["state"] == "partial"
    assert not row["project_configuration_required"]
    assert states["inference"] == "provided"
    assert "training" not in states
    assert states["quantization"] == "installed"
    assert states["evaluation"] == "missing"
    result = client.post(f"/api/projects/{project_id}/plugin-suites/{selected_suite['id']}/installations", json={"worker_id": worker_id, "component_keys": ["quantization"]}).json()
    assert result["results"][0]["status"] == "installed"
    assert client.get(f"/api/tasks?project_id={project_id}").json() == []


def test_invalid_suite_selection_has_no_side_effects(client: TestClient, tmp_path):
    project_id, worker_id = setup_worker(tmp_path)
    selected_suite = suite(client, "Transformers")
    url = f"/api/projects/{project_id}/plugin-suites/{selected_suite['id']}/installations"
    for keys in [["quantization", "foreign"], ["training"], []]:
        assert client.post(url, json={"worker_id": worker_id, "component_keys": keys}).status_code == 422
    assert client.get(f"/api/tasks?project_id={project_id}").json() == []
    with SessionLocal() as session:
        assert not session.query(Task).filter(Task.project_id == project_id).count()
