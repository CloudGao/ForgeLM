from __future__ import annotations

import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import ModelVersion, WorkerPluginInstallation
from llm_platform.services.artifacts import artifact_storage

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"Release-{uuid4().hex[:8]}",
            "target_model_name": "Release-LM",
            "workspace_uri": tempfile.mkdtemp(prefix="release-plugin-workspace-"),
            "project_type": "pretrain",
            "target_parameters": "1M",
            "primary_languages": ["zh"],
            "target_capabilities": ["generate"],
        },
    )
    assert response.status_code == 201
    return response.json()


def _bind(client: TestClient, project_id: str, plugin: dict, config: dict) -> None:
    response = client.put(
        f"/api/projects/{project_id}/plugin-bindings/{plugin['plugin_id']}",
        json={
            "plugin_version_id": plugin["id"],
            "enabled": True,
            "default_config": config,
        },
    )
    assert response.status_code == 200, response.text


def _complete(
    client: TestClient, worker_id: str, expected_task_id: str, result: dict
) -> dict:
    claimed = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker_id},
    )
    assert claimed.status_code == 200, claimed.text
    body = claimed.json()
    assert body["task"]["id"] == expected_task_id
    task_id = body["task"]["id"]
    attempt_id = body["attempt"]["id"]
    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker_id, "attempt_id": attempt_id},
    )
    assert started.status_code == 200, started.text
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": result,
        },
    )
    assert completed.status_code == 200, completed.text
    return completed.json()


def test_native_build_creates_runtime_release_and_independent_delivery(
    client: TestClient,
) -> None:
    project = _project(client)
    exporter = next(
        item
        for item in client.get("/api/plugins?kind=release_exporter").json()
        if item["plugin_id"] == "plugin.forgelm.native-exporter"
    )
    publisher = next(
        item
        for item in client.get("/api/plugins?kind=release_publisher").json()
        if item["plugin_id"] == "plugin.local-directory.publisher"
    )
    _bind(client, project["id"], exporter, {"verify_smoke": True})
    _bind(
        client,
        project["id"],
        publisher,
        {"target_subdirectory": "published-models"},
    )

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"release-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {
                "cpu_count": 4,
                "gpus": [],
                "release_delivery": {"local_directory": True},
            },
            "capabilities": [
                "builtin.local-process",
                exporter["entrypoint"],
                f"{exporter['entrypoint']}@{exporter['version']}",
                publisher["entrypoint"],
                f"{publisher['entrypoint']}@{publisher['version']}",
            ],
        },
    ).json()
    with SessionLocal() as session:
        for plugin in (exporter, publisher):
            session.add(
                WorkerPluginInstallation(
                    worker_id=worker["id"],
                    plugin_id=plugin["plugin_id"],
                    current_plugin_version_id=plugin["id"],
                    target_plugin_version_id=plugin["id"],
                    installed_versions=[
                        {
                            "plugin_version_id": plugin["id"],
                            "version": plugin["version"],
                        }
                    ],
                    state="ready",
                )
            )
        source_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"managed-microlm-placeholder",
            filename="model.tar.gz",
            media_type="application/gzip",
            metadata={
                "format": "microlm-hybrid-safetensors",
                "inference": {"format": "microlm-hybrid-safetensors"},
            },
        )
        model = ModelVersion(
            project_id=project["id"],
            name="MicroLM Release Source",
            version=1,
            artifact_id=source_artifact.id,
            metadata_json={"inference": {"format": "microlm-hybrid-safetensors"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    job_response = client.post(
        f"/api/projects/{project['id']}/release-jobs",
        json={
            "name": "Native runtime",
            "source_kind": "model_version",
            "source_id": model_id,
            "exporter_plugin_version_id": exporter["id"],
            "config": {"verify_smoke": True},
            "gate_policy": {"mode": "bypassed", "reason": "本地集成测试"},
            "model_card_template": "# MicroLM",
            "license_policy": {"status": "reviewed"},
        },
    )
    assert job_response.status_code == 201, job_response.text
    job = job_response.json()
    build_response = client.post(
        f"/api/projects/{project['id']}/release-jobs/{job['id']}/runs",
        json={"release_name": "MicroLM", "release_version": "v1.0.0"},
    )
    assert build_response.status_code == 201, build_response.text
    build = build_response.json()
    assert build["source_snapshot"]["artifact_id"]
    assert build["gate_snapshot"]["status"] == "bypassed"

    with SessionLocal() as session:
        runtime_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="release_runtime",
            payload=b"runtime-directory-archive",
            filename="runtime.tar.gz",
            media_type="application/gzip",
            metadata={"format": "forgelm-native-model-directory-v1"},
        )
        session.commit()
        runtime_artifact_id = runtime_artifact.id
    _complete(
        client,
        worker["id"],
        build["task_id"],
        {
            "output_artifacts": [
                {
                    "ref": "release_runtime",
                    "artifact_id": runtime_artifact_id,
                    "kind": "release_runtime",
                    "storage_type": "managed",
                }
            ],
            "runtime": {
                "format": "forgelm-native-model-directory-v1",
                "payload_tree_hash": "a" * 64,
                "smoke_test": {"status": "passed"},
            },
            "validation_report": {"status": "passed"},
        },
    )
    releases = client.get(f"/api/projects/{project['id']}/releases").json()
    release = next(item for item in releases if item["version"] == "v1.0.0")
    assert release["release_kind"] == "runtime"
    assert release["runtime_artifact_id"] == runtime_artifact_id
    assert release["manifest_artifact_id"]
    assert release["gate_result"]["status"] == "bypassed"

    delivery_job_response = client.post(
        f"/api/projects/{project['id']}/releases/{release['id']}/delivery-jobs",
        json={
            "name": "Local immutable directory",
            "publisher_plugin_version_id": publisher["id"],
            "config": {"target_subdirectory": "published-models"},
        },
    )
    assert delivery_job_response.status_code == 201, delivery_job_response.text
    delivery_job = delivery_job_response.json()
    delivery_run_response = client.post(
        f"/api/projects/{project['id']}/releases/{release['id']}"
        f"/delivery-jobs/{delivery_job['id']}/runs"
    )
    assert delivery_run_response.status_code == 201, delivery_run_response.text
    delivery_run = delivery_run_response.json()
    assert delivery_run["task_id"] != build["task_id"]
    _complete(
        client,
        worker["id"],
        delivery_run["task_id"],
        {
            "delivery": {
                "target_kind": "local_directory",
                "immutable_locator": "file:///release/MicroLM/v1.0.0#sha256=" + "b" * 64,
                "uri": "file:///release/MicroLM/v1.0.0",
                "content_hash": "b" * 64,
                "idempotent": False,
            }
        },
    )
    detail = client.get(
        f"/api/projects/{project['id']}/releases/{release['id']}"
    ).json()
    completed_delivery = detail["delivery_jobs"][0]["runs"][0]
    assert completed_delivery["status"] == "succeeded"
    assert completed_delivery["result"]["immutable_locator"].endswith("b" * 64)
    assert completed_delivery["report_artifact_id"]


def test_release_job_rejects_a_fake_pass_and_requires_explicit_gate_decision(
    client: TestClient,
) -> None:
    project = _project(client)
    exporter = next(
        item
        for item in client.get("/api/plugins?kind=release_exporter").json()
        if item["plugin_id"] == "plugin.forgelm.native-exporter"
    )
    _bind(client, project["id"], exporter, {"verify_smoke": False})
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"placeholder",
            filename="model.tar.gz",
            metadata={"format": "microlm-hybrid-safetensors"},
        )
        model = ModelVersion(
            project_id=project["id"],
            name="Gate source",
            version=1,
            artifact_id=artifact.id,
            metadata_json={"inference": {"format": "microlm-hybrid-safetensors"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id
    response = client.post(
        f"/api/projects/{project['id']}/release-jobs",
        json={
            "name": "Invalid gate",
            "source_kind": "model_version",
            "source_id": model_id,
            "exporter_plugin_version_id": exporter["id"],
            "config": {"verify_smoke": False},
            "gate_policy": {"mode": "passed"},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_gate_policy"
