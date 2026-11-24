from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import Dataset, DatasetVersion, Task, utcnow
from llm_platform.services.tasks import sync_task_domain_status


def _project_with_data(client: TestClient, tmp_path) -> tuple[str, str]:
    workspace = tmp_path / f"pipeline-dag-{uuid4().hex[:8]}"
    workspace.mkdir()
    response = client.post(
        "/api/projects",
        json={
            "name": f"Pipeline-DAG-{uuid4().hex[:8]}",
            "target_model_name": "GraphLM",
            "workspace_uri": str(workspace),
        },
    )
    assert response.status_code == 201
    project_id = response.json()["id"]
    with SessionLocal() as session:
        dataset = Dataset(
            project_id=project_id,
            name="DAG input",
            source_type="local",
            status="ready",
        )
        session.add(dataset)
        session.flush()
        version = DatasetVersion(
            dataset_id=dataset.id,
            version=1,
            state="ready",
            train_ready=True,
        )
        session.add(version)
        session.flush()
        dataset.current_version_id = version.id
        session.commit()
        return project_id, version.id


def _graph(dataset_version_id: str) -> tuple[list[dict], list[dict]]:
    return (
        [
            {
                "id": "source",
                "type": "data",
                "position": {"x": 40, "y": 120},
                "label": "训练数据",
                "config": {
                    "asset_type": "dataset_version",
                    "asset_version_id": dataset_version_id,
                },
            },
            {
                "id": "prepare",
                "type": "custom",
                "position": {"x": 340, "y": 120},
                "label": "真实准备命令",
                "config": {
                    "adapter_id": "builtin.command.data",
                    "command": {
                        "executable": "__worker_python__",
                        "args": ["-c", "print('pipeline')"],
                    },
                    "resource_request": {"cpus": 1, "gpus": 0},
                },
            },
        ],
        [
            {
                "id": "source-prepare",
                "source": "source",
                "target": "prepare",
                "source_handle": "out",
                "target_handle": "in",
            }
        ],
    )


def test_pipeline_definition_instance_and_task_backed_run(
    client: TestClient, tmp_path
) -> None:
    project_id, version_id = _project_with_data(client, tmp_path)
    nodes, edges = _graph(version_id)
    definition_nodes = [
        {**nodes[0], "config": {}},
        {
            **nodes[1],
            "config": {
                "adapter_id": "builtin.command.data",
                "command": {"executable": "", "args": []},
            },
        },
    ]
    created = client.post(
        f"/api/projects/{project_id}/pipeline-definitions",
        json={
            "name": "训练准备 DAG",
            "description": "可复用 Pipeline 定义",
            "nodes": definition_nodes,
            "edges": edges,
        },
    )
    assert created.status_code == 201, created.text
    pipeline = created.json()
    pipeline_id = pipeline["id"]
    assert pipeline["state"] == "draft"
    assert pipeline["draft_revision"]["revision"] == 1

    validation = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/validate"
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    published = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/publish"
    )
    assert published.status_code == 200, published.text
    current = published.json()["current_revision"]
    assert current["status"] == "published"
    assert current["locked"] is True
    assert len(current["content_hash"]) == 64

    direct_run = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/runs",
        json={"inputs": {"reason": "integration test"}},
    )
    assert direct_run.status_code == 409
    assert direct_run.json()["detail"]["code"] == "pipeline_instance_required"

    created_instance = client.post(
        f"/api/projects/{project_id}/pipeline-instances",
        json={
            "definition_revision_id": current["id"],
            "name": "训练准备 · 测试实例",
            "description": "绑定本次执行所需精确版本和命令",
        },
    )
    assert created_instance.status_code == 201, created_instance.text
    instance = created_instance.json()
    instance_id = instance["id"]
    assert instance["status"] == "draft"
    assert instance["validation_report"]["valid"] is False

    saved_instance = client.put(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/draft",
        json={
            "name": instance["name"],
            "description": instance["description"],
            "nodes": nodes,
            "edges": edges,
            "variables": {},
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        },
    )
    assert saved_instance.status_code == 200, saved_instance.text
    validation = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/validate"
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    activated = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/activate"
    )
    assert activated.status_code == 200, activated.text
    assert activated.json()["status"] == "ready"
    assert activated.json()["locked"] is True

    locked_update = client.put(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/draft",
        json={
            "name": instance["name"],
            "description": "should fail",
            "nodes": nodes,
            "edges": edges,
        },
    )
    assert locked_update.status_code == 409
    assert locked_update.json()["detail"]["code"] == "pipeline_instance_locked"

    started = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/runs",
        json={"inputs": {"reason": "integration test"}},
    )
    assert started.status_code == 201, started.text
    run = started.json()
    assert run["status"] == "running"
    assert run["instance_id"] == instance_id
    assert run["instance_name"] == instance["name"]
    assert [node["id"] for node in run["graph_snapshot"]["nodes"]] == [
        "source",
        "prepare",
    ]
    assert run["graph_snapshot"]["nodes"][0]["config"]["asset_version_id"] == version_id
    node_runs = {item["node_id"]: item for item in run["node_runs"]}
    assert node_runs["source"]["status"] == "succeeded"
    assert node_runs["prepare"]["status"] == "queued"
    task_id = node_runs["prepare"]["task_id"]

    with SessionLocal() as session:
        task = session.get(Task, task_id)
        assert task is not None
        assert task.kind == "pipeline_node"
        assert task.payload["pipeline_instance_id"] == instance_id
        assert task.payload["pipeline_context"]["upstream"]["source"][
            "asset_version_id"
        ] == version_id
        task.observed_state = "succeeded"
        task.desired_state = "succeeded"
        task.progress = 1
        task.result = {
            "message": "done",
            "execution_identity": {
                "engine": "fixture",
                "engine_version": "1",
            },
            "consumed_assets": [
                {
                    "asset_type": "dataset_version",
                    "asset_id": version_id,
                    "label": "DAG input · V1",
                    "version": 1,
                    "role": "training_data",
                }
            ],
            "produced_assets": [
                {
                    "asset_type": "model_version",
                    "asset_id": "model-output-1",
                    "label": "GraphLM · V1",
                    "version": 1,
                    "status": "candidate",
                }
            ],
        }
        task.ended_at = utcnow()
        sync_task_domain_status(session, task)
        sync_task_domain_status(session, task)
        session.commit()

    completed = client.get(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/runs/{run['id']}"
    )
    assert completed.status_code == 200
    completed_payload = completed.json()
    assert completed_payload["status"] == "succeeded"
    assert completed_payload["node_summary"]["progress"] == 1
    assert completed_payload["lineage"]["stored"] is True
    assert completed_payload["lineage"]["execution_count"] == 2
    completed_nodes = {
        item["node_id"]: item for item in completed_payload["node_runs"]
    }
    assert {
        item["asset_id"]
        for item in completed_nodes["source"]["lineage"]["consumed_assets"]
    } == {version_id}
    prepare_lineage = completed_nodes["prepare"]["lineage"]
    assert prepare_lineage["execution_identity"]["task_id"] == task_id
    assert prepare_lineage["execution_identity"]["engine"] == "fixture"
    assert {
        item["asset_id"] for item in prepare_lineage["consumed_assets"]
    } == {version_id}
    assert [
        item["asset_id"] for item in prepare_lineage["produced_assets"]
    ] == ["model-output-1"]

    lineage_response = client.get(
        f"/api/projects/{project_id}/pipeline-runs/{run['id']}/lineage"
    )
    assert lineage_response.status_code == 200
    assert lineage_response.json()["execution_count"] == 2
    asset_response = client.get(
        f"/api/projects/{project_id}/lineage/assets/model_version/model-output-1"
    )
    assert asset_response.status_code == 200
    assert [item["node_id"] for item in asset_response.json()["producers"]] == [
        "prepare"
    ]
    input_asset_response = client.get(
        f"/api/projects/{project_id}/lineage/assets/dataset_version/{version_id}"
    )
    assert input_asset_response.status_code == 200
    assert {item["node_id"] for item in input_asset_response.json()["consumers"]} == {
        "source",
        "prepare",
    }
    instance_runs = client.get(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/runs"
    )
    assert instance_runs.status_code == 200
    assert [item["id"] for item in instance_runs.json()] == [run["id"]]


def test_pipeline_publish_rejects_cycles_and_keeps_published_revision(
    client: TestClient, tmp_path
) -> None:
    project_id, version_id = _project_with_data(client, tmp_path)
    nodes, edges = _graph(version_id)
    created = client.post(
        f"/api/projects/{project_id}/pipeline-definitions",
        json={"name": "版本化 DAG", "nodes": nodes, "edges": edges},
    )
    pipeline_id = created.json()["id"]
    first = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/publish"
    )
    assert first.status_code == 200
    first_revision_id = first.json()["current_revision_id"]

    cyclic_edges = [
        *edges,
        {
            "id": "prepare-source",
            "source": "prepare",
            "target": "source",
            "source_handle": "out",
            "target_handle": "in",
        },
    ]
    updated = client.put(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/draft",
        json={"name": "版本化 DAG", "nodes": nodes, "edges": cyclic_edges},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["draft_revision"]["revision"] == 2
    assert updated.json()["current_revision_id"] == first_revision_id

    validation = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/validate"
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is False
    assert {item["code"] for item in validation.json()["errors"]} >= {
        "invalid_port",
        "pipeline_cycle",
    }

    rejected = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/publish"
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "pipeline_validation_failed"

    detail = client.get(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}"
    )
    assert detail.json()["current_revision_id"] == first_revision_id
    assert detail.json()["draft_revision"]["revision"] == 2


def test_pipeline_failure_finishes_parallel_tasks_before_closing_run(
    client: TestClient, tmp_path
) -> None:
    project_id, _ = _project_with_data(client, tmp_path)
    nodes = [
        {
            "id": "approval",
            "type": "check",
            "position": {"x": 40, "y": 80},
            "label": "发布检查",
            "config": {
                "path": "inputs.approved",
                "operator": "truthy",
                "failure_policy": "fail",
            },
        },
        {
            "id": "independent",
            "type": "custom",
            "position": {"x": 40, "y": 260},
            "label": "并行任务",
            "config": {
                "adapter_id": "builtin.command.data",
                "command": {
                    "executable": "__worker_python__",
                    "args": ["-c", "print('parallel')"],
                },
            },
        },
    ]
    created = client.post(
        f"/api/projects/{project_id}/pipeline-definitions",
        json={"name": "并行失败收敛", "nodes": nodes, "edges": []},
    )
    assert created.status_code == 201, created.text
    pipeline_id = created.json()["id"]
    published = client.post(
        f"/api/projects/{project_id}/pipeline-definitions/{pipeline_id}/publish"
    )
    assert published.status_code == 200, published.text

    instance = client.post(
        f"/api/projects/{project_id}/pipeline-instances",
        json={
            "definition_revision_id": published.json()["current_revision"]["id"],
            "name": "并行失败收敛 · 实例",
        },
    )
    assert instance.status_code == 201, instance.text
    instance_id = instance.json()["id"]
    activated = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/activate"
    )
    assert activated.status_code == 200, activated.text

    started = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/runs",
        json={"inputs": {"approved": False}},
    )
    assert started.status_code == 201, started.text
    run = started.json()
    node_runs = {item["node_id"]: item for item in run["node_runs"]}
    assert node_runs["approval"]["status"] == "failed"
    assert node_runs["independent"]["status"] == "queued"
    assert run["status"] == "running"

    with SessionLocal() as session:
        task = session.get(Task, node_runs["independent"]["task_id"])
        assert task is not None
        task.observed_state = "succeeded"
        task.desired_state = "succeeded"
        task.progress = 1
        task.result = {"message": "parallel completed"}
        task.ended_at = utcnow()
        sync_task_domain_status(session, task)
        session.commit()

    completed = client.get(
        f"/api/projects/{project_id}/pipeline-instances/{instance_id}/runs/{run['id']}"
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "failed"
    assert completed.json()["node_summary"]["progress"] == 1
