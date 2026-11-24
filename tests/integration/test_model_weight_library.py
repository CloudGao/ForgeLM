from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def _complete(client: TestClient, worker_id: str, task_id: str, result: dict) -> None:
    claimed = client.post(
        "/api/agent/tasks/claim", headers=AGENT_HEADERS, json={"worker_id": worker_id}
    ).json()
    assert claimed["task"]["id"] == task_id
    attempt_id = claimed["attempt"]["id"]
    assert (
        client.post(
            f"/api/agent/tasks/{task_id}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker_id, "attempt_id": attempt_id},
        ).status_code
        == 200
    )
    response = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": result,
            "exit_code": 0,
        },
    )
    assert response.status_code == 200, response.text


def test_global_weight_download_can_be_reused_by_project(
    client: TestClient, tmp_path: Path
) -> None:
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"weight-worker-{uuid4().hex[:8]}",
            "hostname": "weight-host",
            "labels": {"location": "local"},
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [
                "builtin.model-weight",
                "builtin.hf-asset",
                "builtin.local-process",
                "builtin.lm-eval",
                "builtin.transformers-inference",
            ],
        },
    ).json()
    scan = client.post(
        "/api/model-weights/huggingface/scan",
        json={
            "repo_id": "Qwen/example-0.8b",
            "revision": "main",
            "resource_request": {"gpus": 0, "worker_selector": {"worker_id": worker["id"]}},
        },
    )
    assert scan.status_code == 202
    commit = "a" * 40
    _complete(
        client,
        worker["id"],
        scan.json()["id"],
        {
            "repo_id": "Qwen/example-0.8b",
            "repo_type": "model",
            "requested_revision": "main",
            "commit_hash": commit,
            "endpoint": "https://huggingface.co",
            "files": [
                {"path": "config.json", "size_bytes": 10},
                {"path": "tokenizer.json", "size_bytes": 10},
                {"path": "model.safetensors", "size_bytes": 100},
            ],
            "file_count": 3,
            "total_size_bytes": 120,
        },
    )
    plan = client.post(
        "/api/model-weights/huggingface/plans",
        json={
            "scan_task_id": scan.json()["id"],
            "selected_paths": ["config.json", "tokenizer.json", "model.safetensors"],
        },
    )
    assert plan.status_code == 202, plan.text
    plan_payload = client.get(f"/api/tasks/{plan.json()['id']}").json()["payload"]
    identity_hash = plan_payload["identity_hash"]
    _complete(
        client,
        worker["id"],
        plan.json()["id"],
        {
            "repo_id": "Qwen/example-0.8b",
            "repo_type": "model",
            "requested_revision": "main",
            "commit_hash": commit,
            "endpoint": "https://huggingface.co",
            "selected_paths": ["config.json", "tokenizer.json", "model.safetensors"],
            "files": [
                {"path": "config.json", "size_bytes": 10, "will_download": True},
                {"path": "tokenizer.json", "size_bytes": 10, "will_download": True},
                {"path": "model.safetensors", "size_bytes": 100, "will_download": True},
            ],
            "file_count": 3,
            "selected_bytes": 120,
            "cached_bytes": 0,
            "download_bytes": 120,
            "disk_free_bytes": 1_000_000,
            "disk_space_sufficient": True,
            "identity_hash": identity_hash,
            "worker_id": worker["id"],
        },
    )
    created = client.post(
        "/api/model-weights",
        json={
            "name": "Qwen example 0.8B",
            "plan_task_id": plan.json()["id"],
            "parameter_count": 800_000_000,
        },
    )
    assert created.status_code == 202, created.text
    weight = created.json()["weight"]
    location = tmp_path / identity_hash
    location.mkdir()
    _complete(
        client,
        worker["id"],
        created.json()["task"]["id"],
        {
            "identity_hash": identity_hash,
            "source_manifest": {"repository_id": "Qwen/example-0.8b"},
            "format_summary": {"transformers": True, "tokenizer": True},
            "runtime_capabilities": ["lm_eval_hf", "generate", "loglikelihood"],
            "outputs": [
                {
                    "ref": "model_weight",
                    "kind": "model_weights",
                    "storage_type": "external",
                    "uri": location.resolve().as_uri(),
                    "content_hash": "b" * 64,
                    "size_bytes": 120,
                    "media_type": "application/vnd.forgelm.model-weight",
                    "availability_state": "available",
                    "worker_id": worker["id"],
                    "metadata": {
                        "format": "forgelm-model-weight-v1",
                        "identity_hash": identity_hash,
                    },
                }
            ],
        },
    )
    ready = client.get(f"/api/model-weights/{weight['id']}").json()
    assert ready["state"] == "ready"
    assert ready["locations"][0]["worker_id"] == worker["id"]

    project_workspace = tmp_path / "project"
    project_workspace.mkdir()
    project_response = client.post(
        "/api/projects",
        json={
            "name": f"weight-project-{uuid4().hex[:8]}",
            "target_model_name": "WeightLM",
            "workspace_uri": str(project_workspace),
            "project_type": "pretrain",
        },
    )
    assert project_response.status_code == 201, project_response.text
    project = project_response.json()
    used = client.post(
        f"/api/model-weights/{weight['id']}/project-models",
        json={"project_id": project["id"], "name": "Weight Baseline"},
    )
    assert used.status_code == 201, used.text
    models = client.get(f"/api/projects/{project['id']}/models").json()["models"]
    model = next(item for item in models if item["id"] == used.json()["id"])
    assert model["model_weight"]["id"] == weight["id"]
    assert model["inference_source"]["worker_ids"] == [worker["id"]]

    suite_response = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={"name": "Managed weight smoke", "benchmark_ids": ["lm-eval:gsm8k"]},
    )
    assert suite_response.status_code == 201, suite_response.text
    evaluation = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model["id"]],
            "suite_version_ids": [suite_response.json()["id"]],
            "inference_config": {"device": "cpu", "batch_size": 1, "sample_percentage": 1, "sample_seed": 42},
            "scoring_config": {"save_samples": True},
            "resource_request": {"gpus": 0},
        },
    )
    assert evaluation.status_code == 201, evaluation.text
    evaluation_task = evaluation.json()["task"]
    assert (
        evaluation_task["payload"]["model_weight_inputs"][0]["model_weight_version_id"]
        == weight["id"]
    )
    assert (
        evaluation_task["payload"]["inference_config"]["models"][model["id"]]["model_args"]
        == "pretrained=__FORGELM_MODEL_WEIGHT__"
    )
    assert (
        evaluation_task["payload"]["resource_request"]["worker_selector"]["worker_id"]
        == worker["id"]
    )
    assert (
        client.post(
            f"/api/tasks/{evaluation_task['id']}/control", json={"action": "cancel"}
        ).status_code
        == 200
    )
    assert client.delete(f"/api/model-weights/{weight['id']}").status_code == 409
