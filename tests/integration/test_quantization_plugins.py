from __future__ import annotations

import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    ModelVersion,
    PluginVersion,
    WorkerPluginInstallation,
)
from llm_platform.services.artifacts import artifact_storage

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"Quantization-{uuid4().hex[:8]}",
            "target_model_name": "Quantized-LM",
            "workspace_uri": tempfile.mkdtemp(prefix="quantization-plugin-workspace-"),
            "project_type": "pretrain",
            "target_parameters": "42M",
            "primary_languages": ["zh"],
            "target_capabilities": ["generate"],
        },
    )
    assert response.status_code == 201
    return response.json()


def test_install_bind_create_job_and_freeze_each_run(client: TestClient) -> None:
    project = _project(client)
    catalog_response = client.get("/api/plugins?kind=quantization")
    assert catalog_response.status_code == 200
    plugin = next(
        item
        for item in catalog_response.json()
        if item["plugin_id"] == "plugin.llm-compressor.quantization"
    )
    assert plugin["runtime"]["requirements"] == ["llmcompressor==0.10.0"]

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"quant-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": [{"index": 0}]},
            "capabilities": ["builtin.plugin-manager", "builtin.local-process"],
        },
    ).json()
    install = client.post(
        f"/api/projects/{project['id']}/plugin-installations",
        json={"worker_id": worker["id"], "plugin_version_id": plugin["id"]},
    )
    assert install.status_code == 201
    assert install.json()["task"]["adapter_id"] == "builtin.plugin-manager"
    frozen_plugin = install.json()["task"]["payload"]["plugin"]
    assert frozen_plugin["runtime"]["requirements"] == ["llmcompressor==0.10.0"]
    assert "requirements" not in install.request.content.decode()
    claimed = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claimed["task"]["id"] == install.json()["task"]["id"]
    task_id = claimed["task"]["id"]
    attempt_id = claimed["attempt"]["id"]
    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id},
    )
    assert started.status_code == 200
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "installation": {
                    "plugin_id": plugin["plugin_id"],
                    "version": plugin["version"],
                    "entrypoint": plugin["entrypoint"],
                    "runner": "llm_compressor",
                    "isolation": "venv",
                    "python": "/worker/plugins/venv/python",
                    "manifest_path": "/worker/plugins/installation.json",
                    "requirements": ["llmcompressor==0.10.0"],
                    "installed_at": "2026-08-23T00:00:00+00:00",
                }
            },
        },
    )
    assert completed.status_code == 200
    installation_view = next(
        item
        for item in client.get("/api/plugin-installations").json()
        if item["id"] == install.json()["id"]
    )
    assert installation_view["state"] == "ready"

    binding = client.put(
        f"/api/projects/{project['id']}/plugin-bindings/{plugin['plugin_id']}",
        json={
            "plugin_version_id": plugin["id"],
            "enabled": True,
            "default_config": {
                "algorithm": "RTN",
                "scheme": "FP8_DYNAMIC",
                "targets": ["Linear"],
            },
        },
    )
    assert binding.status_code == 200

    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"transformers-model-placeholder",
            filename="model.tar.gz",
            media_type="application/gzip",
        )
        model = ModelVersion(
            project_id=project["id"],
            name="Transformers Model",
            version=1,
            artifact_id=artifact.id,
            metadata_json={"inference": {"format": "transformers-pretrained"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    created = client.post(
        f"/api/projects/{project['id']}/quantization-jobs",
        json={
            "name": "Release FP8",
            "model_version_id": model_id,
            "plugin_version_id": plugin["id"],
            "config": {
                "algorithm": "RTN",
                "scheme": "FP8_DYNAMIC",
                "targets": ["Linear"],
                "ignore": ["lm_head"],
            },
            "resource_request": {"gpus": 1},
        },
    )
    assert created.status_code == 201
    job = created.json()

    first = client.post(
        f"/api/projects/{project['id']}/quantization-jobs/{job['id']}/runs"
    )
    assert first.status_code == 201
    first_run = first.json()
    assert first_run["run_number"] == 1
    assert first_run["config_snapshot"]["config"]["targets"] == ["Linear"]
    selector = first_run["task"]["payload"]["resource_request"]["worker_selector"]
    assert selector["worker_ids"] == [worker["id"]]
    assert selector["capabilities"] == [
        plugin["entrypoint"],
        f"{plugin['entrypoint']}@{plugin['version']}",
    ]

    patched = client.patch(
        f"/api/projects/{project['id']}/quantization-jobs/{job['id']}",
        json={
            "config": {
                "algorithm": "RTN",
                "scheme": "FP8_DYNAMIC",
                "targets": ["Conv1D"],
            }
        },
    )
    assert patched.status_code == 200
    second = client.post(
        f"/api/projects/{project['id']}/quantization-jobs/{job['id']}/runs"
    )
    assert second.status_code == 201
    assert second.json()["run_number"] == 2
    assert second.json()["config_snapshot"]["config"]["targets"] == ["Conv1D"]
    runs = client.get(
        f"/api/projects/{project['id']}/quantization-jobs/{job['id']}/runs"
    ).json()
    assert [item["run_number"] for item in runs] == [2, 1]
    assert runs[1]["config_snapshot"]["config"]["targets"] == ["Linear"]

    with SessionLocal() as session:
        session.add(
            PluginVersion(
                plugin_id=plugin["plugin_id"],
                version="0.2.0",
                kind="quantization",
                entrypoint=plugin["entrypoint"],
                protocol_version=1,
                config_schema=plugin["config_schema"],
                capabilities=plugin["capabilities"],
                runtime=plugin["runtime"],
                source={"type": "builtin-catalog"},
                support_level="experimental",
            )
        )
        session.commit()
    manual_update = client.post(
        f"/api/plugin-installations/{install.json()['id']}/update",
        json={"project_id": project["id"]},
    )
    assert manual_update.status_code == 201
    assert manual_update.json()["target_plugin_version"]["version"] == "0.2.0"
    assert manual_update.json()["task"]["kind"] == "plugin_install"


def test_gptq_requires_explicit_calibration_samples(client: TestClient) -> None:
    project = _project(client)
    plugin = next(
        item
        for item in client.get("/api/plugins?kind=quantization").json()
        if item["plugin_id"] == "plugin.llm-compressor.quantization"
    )
    client.put(
        f"/api/projects/{project['id']}/plugin-bindings/{plugin['plugin_id']}",
        json={
            "plugin_version_id": plugin["id"],
            "default_config": {
                "algorithm": "GPTQ",
                "scheme": "W4A16",
                "targets": ["Linear"],
            },
        },
    )
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"placeholder",
            filename="model.tar.gz",
        )
        model = ModelVersion(
            project_id=project["id"], name="GPTQ Model", version=1, artifact_id=artifact.id
        )
        session.add(model)
        session.commit()
        model_id = model.id
    response = client.post(
        f"/api/projects/{project['id']}/quantization-jobs",
        json={
            "name": "Invalid GPTQ",
            "model_version_id": model_id,
            "plugin_version_id": plugin["id"],
            "config": {"algorithm": "GPTQ", "scheme": "W4A16", "targets": ["Linear"]},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "calibration_required"
