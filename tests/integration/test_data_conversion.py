from __future__ import annotations

import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import DatasetVersion, Task
from llm_platform.services.artifacts import artifact_storage

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"Data-Conversion-{uuid4().hex[:8]}",
            "target_model_name": "Document-LM",
            "workspace_uri": tempfile.mkdtemp(prefix="data-conversion-workspace-"),
            "project_type": "pretrain",
            "target_parameters": "42M",
        },
    )
    assert response.status_code == 201
    return response.json()


def _raw_dataset(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/datasets",
        json={
            "name": f"Documents-{uuid4().hex[:8]}",
            "source_type": "local",
            "source_uri": "F:/documents",
            "license": "MIT",
            "language": "zh",
            "initial_version": {
                "tier": "bronze",
                "raw_size_bytes": 1024,
                "document_count": 0,
                "token_estimate": 0,
                "quality_report": {"unsupported_files": ["manual.pdf"]},
                "train_ready": False,
            },
        },
    )
    assert response.status_code == 201
    dataset = response.json()
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project_id,
            kind="dataset_snapshot",
            payload=b"%PDF-raw-source",
            filename="manual.pdf",
            media_type="application/pdf",
        )
        version = session.get(DatasetVersion, dataset["current_version"]["id"])
        assert version is not None
        version.artifact_id = artifact.id
        version.state = "ready"
        session.commit()
    return client.get(f"/api/datasets/{dataset['id']}").json()


def _recipe(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/data-conversion-recipes",
        json={
            "name": "文档转文本",
            "description": "识别并转换三种文档格式",
            "operators": [
                {"detect_format": {}},
                {"extract_pdf": {}},
                {"extract_docx": {"include_tables": True}},
                {"extract_epub": {}},
                {"normalize_whitespace": {"max_blank_lines": 1}},
                {"filter_empty": {"min_characters": 1}},
            ],
        },
    )
    assert response.status_code == 201
    return response.json()


def test_conversion_recipe_run_and_success_create_new_bronze_version(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset = _raw_dataset(client, project["id"])
    recipe = _recipe(client, project["id"])
    original_version_id = dataset["current_version"]["id"]

    operators = client.get("/api/data-conversion/operators")
    assert operators.status_code == 200
    assert {item["name"] for item in operators.json()["operators"]} >= {
        "extract_pdf",
        "extract_docx",
        "extract_epub",
        "chunk_text",
    }

    created = client.post(
        f"/api/projects/{project['id']}/data-conversion-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": original_version_id,
            "resource_request": {"gpus": 0},
        },
    )
    assert created.status_code == 202
    run = created.json()
    assert run["status"] == "queued"
    assert run["output_version"]["tier"] == "bronze"
    assert run["output_version"]["parent_version_id"] == original_version_id
    assert run["output_version"]["conversion_run_id"] == run["id"]
    assert (
        client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]["id"]
        == original_version_id
    )
    with SessionLocal() as session:
        task = session.get(Task, run["task_id"])
        assert task is not None
        assert task.kind == "data_convert"
        assert task.adapter_id == "builtin.data-conversion"
        assert task.payload["resource_request"]["worker_selector"]["capabilities"] == [
            "builtin.data-conversion"
        ]
        assert task.payload["data_conversion_input"]["dataset_version_id"] == original_version_id

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"conversion-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "labels": {"os": "windows"},
            "capabilities": ["builtin.data-conversion", "builtin.local-process"],
            "resources": {"cpu_count": 4, "gpus": []},
            "runtime": {},
        },
    )
    assert worker.status_code == 200
    worker_id = worker.json()["id"]
    claim = client.post(
        "/api/agent/tasks/claim", headers=AGENT_HEADERS, json={"worker_id": worker_id}
    )
    assert claim.status_code == 200
    attempt = claim.json()["attempt"]
    task_id = run["task_id"]
    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "runtime_handle": {"type": "test"},
            "environment_snapshot": {},
        },
    )
    assert started.status_code == 200
    snapshot = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "kind": "dataset_snapshot",
        },
        files={"file": ("converted.jsonl", b'{"text":"converted"}\n', "application/x-ndjson")},
    )
    manifest = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "kind": "dataset_record_manifest",
        },
        files={"file": ("records.jsonl.gz", b"manifest", "application/gzip")},
    )
    report = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "kind": "data_conversion_report",
        },
        files={
            "file": ("conversion-report.json", b'{"converted_file_count":1}', "application/json")
        },
    )
    assert snapshot.status_code == 200
    assert manifest.status_code == 200
    assert report.status_code == 200
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "status": "succeeded",
            "exit_code": 0,
            "result": {
                "engine_version": "document-conversion-v1",
                "raw_size_bytes": 21,
                "document_count": 1,
                "token_estimate": 3,
                "automatic_quality_score": 98,
                "quality_report": {"analyzed_document_count": 1},
                "sample_preview": [{"sample_key": "converted:0", "raw_text": "converted"}],
                "source_snapshot": {"source_type": "conversion", "files": []},
                "train_ready": True,
                "conversion_summary": {
                    "format_counts": {"pdf": 1},
                    "converted_file_count": 1,
                    "output_record_count": 1,
                },
                "output_artifacts": [
                    {
                        "ref": "dataset",
                        "artifact_id": snapshot.json()["id"],
                        "kind": "dataset_snapshot",
                    },
                    {
                        "ref": "record_manifest",
                        "artifact_id": manifest.json()["id"],
                        "kind": "dataset_record_manifest",
                    },
                    {
                        "ref": "conversion_report",
                        "artifact_id": report.json()["id"],
                        "kind": "data_conversion_report",
                    },
                ],
            },
        },
    )
    assert completed.status_code == 200
    final_run = client.get(f"/api/projects/{project['id']}/data-conversion-runs").json()[0]
    assert final_run["status"] == "succeeded"
    assert final_run["summary"]["converted_file_count"] == 1
    final_dataset = client.get(f"/api/datasets/{dataset['id']}").json()
    assert final_dataset["current_version"]["id"] == run["output_dataset_version_id"]
    assert final_dataset["current_version"]["train_ready"] is True


def test_conversion_recipe_rejects_invalid_operator_order(client: TestClient) -> None:
    project = _project(client)
    response = client.post(
        f"/api/projects/{project['id']}/data-conversion-recipes",
        json={
            "name": "错误顺序",
            "operators": [{"extract_pdf": {}}, {"detect_format": {}}],
        },
    )
    assert response.status_code == 422


def test_cancelled_conversion_keeps_raw_version_current(client: TestClient) -> None:
    project = _project(client)
    dataset = _raw_dataset(client, project["id"])
    recipe = _recipe(client, project["id"])
    original_version_id = dataset["current_version"]["id"]
    created = client.post(
        f"/api/projects/{project['id']}/data-conversion-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": original_version_id,
        },
    )
    assert created.status_code == 202
    run = created.json()

    cancelled = client.post(
        f"/api/tasks/{run['task_id']}/control",
        json={"action": "cancel", "reason": "test cancellation"},
    )

    assert cancelled.status_code == 200
    final_dataset = client.get(f"/api/datasets/{dataset['id']}").json()
    assert final_dataset["current_version"]["id"] == original_version_id
    runs = client.get(f"/api/projects/{project['id']}/data-conversion-runs").json()
    final_run = next(item for item in runs if item["id"] == run["id"])
    assert final_run["status"] == "cancelled"
    assert final_run["output_version"]["state"] == "cancelled"
