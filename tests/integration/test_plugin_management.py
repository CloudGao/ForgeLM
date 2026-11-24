from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import PluginVersion, Project, Task, WorkerNode, WorkerPluginInstallation


def test_plugin_task_filter_is_applied_before_pagination(client: TestClient, tmp_path) -> None:
    with SessionLocal() as session:
        project = Project(target_model_name="TestLM", name=f"Plugin UI {uuid4().hex}", workspace_uri=str(tmp_path))
        other_project = Project(target_model_name="TestLM", name=f"Other {uuid4().hex}", workspace_uri=str(tmp_path / "other"))
        session.add_all([project, other_project])
        session.flush()
        expected = []
        for i in range(4):
            task = Task(project_id=project.id, kind="plugin_install", title=f"Install {i}")
            session.add(task)
            session.flush()
            expected.append(task.id)
            session.add(Task(project_id=project.id, kind="train", title=f"Train {i}"))
        session.add(Task(project_id=other_project.id, kind="plugin_install", title="Other install"))
        session.commit()
        project_id = project.id
    url = f"/api/tasks?project_id={project_id}&kind=plugin_install&include_events=false&limit=2"
    first = client.get(url).json()
    second = client.get(url + "&offset=2").json()
    assert len(first) == len(second) == 2
    assert {task["id"] for task in first + second} == set(expected)
    assert all(task["kind"] == "plugin_install" and "events" not in task for task in first + second)
    default_rows = client.get(f"/api/tasks?project_id={project_id}").json()
    assert len(default_rows) == 8
    assert all("events" not in task for task in default_rows)
    assert client.get(url + "&exclude_kind=plugin_install").json() == []


@pytest.mark.parametrize(
    ("worker_status", "capabilities", "code"),
    [("offline", ["builtin.plugin-manager"], "worker_offline"),
     ("online", [], "worker_plugin_manager_unavailable")],
)
def test_update_checks_worker_before_creating_task(
    client: TestClient, tmp_path, worker_status: str, capabilities: list[str], code: str
) -> None:
    with SessionLocal() as session:
        project = Project(target_model_name="TestLM", name=f"Update {uuid4().hex}", workspace_uri=str(tmp_path))
        worker = WorkerNode(name=f"Unavailable-{uuid4().hex}", hostname="test", status=worker_status, capabilities=capabilities)
        plugin_id = f"test.plugin.{uuid4().hex}"
        versions = [PluginVersion(plugin_id=plugin_id, version=version, kind="quantization", entrypoint=plugin_id, source={"type": "builtin-catalog"}, runtime={"requirements": []}) for version in ["0.1.0", "0.2.0"]]
        session.add_all([project, worker, *versions])
        session.flush()
        installation = WorkerPluginInstallation(worker_id=worker.id, plugin_id=plugin_id, current_plugin_version_id=versions[0].id, state="ready", installed_versions=[{"plugin_version_id": versions[0].id}])
        session.add(installation)
        session.commit()
        installation_id, project_id, target_id = installation.id, project.id, versions[1].id
    response = client.post(f"/api/plugin-installations/{installation_id}/update", json={"project_id": project_id, "target_plugin_version_id": target_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == code
    assert client.get(f"/api/tasks?project_id={project_id}").json() == []
    with SessionLocal() as session:
        installation = session.get(WorkerPluginInstallation, installation_id)
        assert installation.state == "ready"
        assert installation.target_plugin_version_id is None
        assert installation.task_id is None
