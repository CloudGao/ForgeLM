from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import Artifact
from llm_platform.routers.training import _route_training_resource_request
from llm_platform.services.artifacts import artifact_storage, artifact_view

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def create_project(client: TestClient, prefix: str) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix=f"{prefix}-workspace-"))
    response = client.post(
        "/api/projects",
        json={
            "name": f"{prefix}-{uuid4().hex[:8]}",
            "target_model_name": f"{prefix}-LM",
            "project_type": "pretrain",
            "target_parameters": "42M",
            "workspace_uri": str(workspace),
        },
    )
    assert response.status_code == 201
    return response.json()


def test_external_artifact_adds_locations_without_copying(client: TestClient) -> None:
    project = create_project(client, "External-Artifact")
    digest = "a" * 64
    first_uri = f"file:///data/{uuid4().hex}/checkpoint-100"
    first = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": first_uri,
            "content_hash": digest,
            "size_bytes": 1_700_000_000,
            "availability_state": "available",
        },
    )
    assert first.status_code == 201
    artifact = first.json()
    assert artifact["content_hash"] == digest
    assert artifact["locations"][0]["storage_type"] == "external"

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"artifact-worker-{uuid4().hex[:8]}",
            "hostname": "wsl",
            "labels": {"environment": "wsl"},
            "resources": {"cpu_count": 1, "gpus": []},
            "capabilities": [],
        },
    ).json()
    enriched = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": first_uri,
            "content_hash": digest,
            "size_bytes": 1_700_000_000,
            "availability_state": "available",
            "worker_id": worker["id"],
            "metadata": {"source": "history-import"},
        },
    )
    assert enriched.status_code == 201
    enriched_location = next(
        item for item in enriched.json()["locations"] if item["uri"] == first_uri
    )
    assert enriched_location["worker_id"] == worker["id"]
    assert enriched_location["metadata_json"]["source"] == "history-import"

    second = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": f"file:///backup/{uuid4().hex}/checkpoint-100",
            "content_hash": digest,
            "size_bytes": 1_700_000_000,
            "availability_state": "offline",
        },
    )
    assert second.status_code == 201
    assert second.json()["id"] == artifact["id"]
    assert {item["storage_type"] for item in second.json()["locations"]} == {"external"}
    assert len(second.json()["locations"]) == 2

    payload = b"external-content-promoted-to-managed"
    promoted_digest = hashlib.sha256(payload).hexdigest()
    reference = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": f"file:///data/{uuid4().hex}/checkpoint-managed-later",
            "content_hash": promoted_digest,
            "size_bytes": len(payload),
        },
    ).json()
    with SessionLocal() as session:
        managed = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="checkpoint",
            payload=payload,
            filename="checkpoint.bin",
        )
        session.commit()
        promoted = artifact_view(session, managed, project["id"])
    assert managed.id == reference["id"]
    assert {item["storage_type"] for item in promoted["locations"]} == {
        "external",
        "managed",
    }


def test_local_windows_artifact_is_projected_only_to_same_host_wsl_worker(
    client: TestClient,
) -> None:
    project = create_project(client, "Local-WSL-Bridge")
    source = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"native-{uuid4().hex[:8]}",
            "hostname": "same-host",
            "labels": {
                "location": "local",
                "environment": "native",
                "os": "windows",
            },
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [],
        },
    ).json()
    target = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"wsl-{uuid4().hex[:8]}",
            "hostname": "same-host",
            "labels": {
                "location": "local",
                "environment": "wsl",
                "os": "linux",
                "distro": "Ubuntu",
            },
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [],
        },
    ).json()
    remote_wsl = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"remote-wsl-{uuid4().hex[:8]}",
            "hostname": "other-host",
            "labels": {
                "location": "local",
                "environment": "wsl",
                "os": "linux",
                "distro": "Ubuntu",
            },
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [],
        },
    ).json()
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "dataset_snapshot",
            "uri": "file:///F:/datasets/corpus",
            "content_hash": "b" * 64,
            "availability_state": "available",
            "worker_id": source["id"],
        },
    ).json()

    visible = client.get(
        f"/api/agent/artifacts/{artifact['id']}",
        params={"project_id": project["id"], "worker_id": target["id"]},
        headers=AGENT_HEADERS,
    ).json()
    bridge = next(
        location
        for location in visible["locations"]
        if (location.get("metadata_json") or {}).get("access_mode")
        == "local_windows_wsl"
    )
    assert bridge["worker_id"] == target["id"]
    assert bridge["uri"] == "file:///F:/datasets/corpus"

    hidden = client.get(
        f"/api/agent/artifacts/{artifact['id']}",
        params={"project_id": project["id"], "worker_id": remote_wsl["id"]},
        headers=AGENT_HEADERS,
    ).json()
    assert not any(
        (location.get("metadata_json") or {}).get("access_mode")
        == "local_windows_wsl"
        for location in hidden["locations"]
    )

    with SessionLocal() as session:
        access = artifact_storage.inference_access(
            session, session.get(Artifact, artifact["id"]), project["id"]
        )
        routed = _route_training_resource_request(
            session,
            project["id"],
            {"gpus": 1, "worker_selector": {"worker_id": target["id"]}},
            [artifact["id"]],
        )
        with pytest.raises(HTTPException) as mismatch:
            _route_training_resource_request(
                session,
                project["id"],
                {"gpus": 1, "worker_selector": {"worker_id": remote_wsl["id"]}},
                [artifact["id"]],
            )
    assert target["id"] in access["worker_ids"]
    assert remote_wsl["id"] not in access["worker_ids"]
    assert routed["worker_selector"] == {"worker_id": target["id"]}
    assert mismatch.value.detail["code"] == "training_data_worker_location_mismatch"


def test_local_wsl_artifact_is_projected_only_to_same_distro_peer(
    client: TestClient,
) -> None:
    project = create_project(client, "Local-WSL-Peer-Bridge")

    def register_worker(name: str, hostname: str, distro: str) -> dict:
        return client.post(
            "/api/agent/register",
            headers=AGENT_HEADERS,
            json={
                "name": f"{name}-{uuid4().hex[:8]}",
                "hostname": hostname,
                "labels": {
                    "location": "local",
                    "environment": "wsl",
                    "os": "linux",
                    "distro": distro,
                },
                "resources": {"cpu_count": 4, "gpus": []},
                "capabilities": [],
            },
        ).json()

    source = register_worker("data-wsl", "same-host", "MicroLM-Ubuntu")
    target = register_worker("training-wsl", "same-host", "MicroLM-Ubuntu")
    other_distro = register_worker("other-wsl", "same-host", "Ubuntu")
    other_host = register_worker("remote-wsl", "other-host", "MicroLM-Ubuntu")
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "dataset_snapshot",
            "uri": "file:///data/forgelm/data-worker/workspaces/corpus",
            "content_hash": "c" * 64,
            "availability_state": "available",
            "worker_id": source["id"],
        },
    ).json()

    visible = client.get(
        f"/api/agent/artifacts/{artifact['id']}",
        params={"project_id": project["id"], "worker_id": target["id"]},
        headers=AGENT_HEADERS,
    ).json()
    bridge = next(
        location
        for location in visible["locations"]
        if (location.get("metadata_json") or {}).get("access_mode") == "local_wsl_peer"
    )
    assert bridge["worker_id"] == target["id"]
    assert bridge["uri"] == "file:///data/forgelm/data-worker/workspaces/corpus"

    with SessionLocal() as session:
        access = artifact_storage.inference_access(
            session, session.get(Artifact, artifact["id"]), project["id"]
        )
    assert target["id"] in access["worker_ids"]
    assert other_distro["id"] not in access["worker_ids"]
    assert other_host["id"] not in access["worker_ids"]


def test_same_content_has_an_independent_managed_location_per_project(
    client: TestClient, tmp_path: Path
) -> None:
    first = create_project(client, "Scoped-Artifact-A")
    second = create_project(client, "Scoped-Artifact-B")
    payload = b"same immutable content"

    with SessionLocal() as session:
        first_artifact = artifact_storage.put_bytes(
            session,
            project_id=first["id"],
            kind="checkpoint",
            payload=payload,
            filename="model.bin",
        )
        second_artifact = artifact_storage.put_bytes(
            session,
            project_id=second["id"],
            kind="checkpoint",
            payload=payload,
            filename="model.bin",
        )
        session.commit()
        first_path = artifact_storage.managed_payload_path(session, first_artifact, first["id"])
        second_path = artifact_storage.managed_payload_path(session, second_artifact, second["id"])

    assert first_artifact.id == second_artifact.id
    assert first_path is not None and second_path is not None
    assert first_path != second_path
    assert Path(first["workspace_uri"]).resolve() in first_path.parents
    assert Path(second["workspace_uri"]).resolve() in second_path.parents
    assert first_path.read_bytes() == payload
    assert second_path.read_bytes() == payload

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    locked = client.patch(f"/api/projects/{first['id']}", json={"workspace_uri": str(replacement)})
    assert locked.status_code == 409
    assert locked.json()["detail"]["code"] == "project_workspace_locked"


def test_worker_selector_routes_task_to_matching_labels(client: TestClient) -> None:
    project = create_project(client, "Worker-Selector")
    dataset = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Corpus-{uuid4().hex[:8]}",
            "source_type": "local",
            "initial_version": {},
            "start_import": False,
        },
    ).json()
    tokenizer = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": f"Tokenizer-{uuid4().hex[:8]}",
            "algorithm": "BPE",
            "vocab_size": 4096,
            "data_ref_type": "dataset_version",
            "data_ref_id": dataset["current_version"]["id"],
            "resource_request": {
                "gpus": 0,
                "worker_selector": {"labels": {"environment": "wsl", "project": "example-model"}},
            },
            "command": {"executable": sys.executable, "args": ["-c", "pass"]},
            "start": True,
        },
    ).json()
    native_worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"native-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "labels": {"environment": "native", "project": "example-model"},
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.command.tokenizer", "builtin.local-process"],
        },
    ).json()
    native_claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": native_worker["id"]},
    )
    assert native_claim.status_code == 200
    assert native_claim.json() is None

    wsl_worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"wsl-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "labels": {"environment": "wsl", "project": "example-model"},
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.command.tokenizer", "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": wsl_worker["id"]},
    ).json()
    assert claim["task"]["id"] == tokenizer["task_id"]
    assert claim["lease"]["allocation"]["worker_id"] == wsl_worker["id"]


def history_manifest(project_marker: str) -> dict:
    return {
        "source_key": f"legacy-{project_marker}",
        "artifacts": [
            {
                "ref": "tokenizer-files",
                "kind": "tokenizer",
                "uri": f"file:///data/{project_marker}/tokenizer",
                "content_hash": "1" * 64,
                "availability_state": "available",
            },
            {
                "ref": "checkpoint-100",
                "kind": "checkpoint",
                "uri": f"file:///data/{project_marker}/checkpoint-100",
                "content_hash": "2" * 64,
                "availability_state": "available",
            },
            {
                "ref": "checkpoint-200",
                "kind": "checkpoint",
                "uri": f"file:///data/{project_marker}/checkpoint-200",
                "content_hash": "3" * 64,
                "availability_state": "available",
            },
            {
                "ref": "evaluation-report",
                "kind": "evaluation_report",
                "uri": f"file:///data/{project_marker}/evaluation.json",
                "content_hash": "4" * 64,
                "availability_state": "available",
            },
        ],
        "datasets": [
            {
                "external_id": "dataset-main",
                "name": "Legacy Corpus",
                "source_type": "local",
                "current_version_external_id": "dataset-main-v1",
                "versions": [
                    {
                        "external_id": "dataset-main-v1",
                        "version": 1,
                        "tier": "gold",
                        "token_estimate": 1_000_000,
                        "train_ready": True,
                    }
                ],
            }
        ],
        "tokenizers": [
            {
                "external_id": "tokenizer-main",
                "name": "Legacy BPE",
                "version": 1,
                "vocab_size": 32768,
                "artifact_ref": "tokenizer-files",
                "is_default": True,
            }
        ],
        "training_runs": [
            {
                "external_id": "run-base",
                "name": "Base pretraining",
                "status": "succeeded",
                "current_step": 200,
                "current_tokens": 2_000_000,
                "checkpoints": [
                    {
                        "external_id": "cp-100",
                        "step": 100,
                        "token_count": 1_000_000,
                        "artifact_ref": "checkpoint-100",
                        "metrics": {"loss": 2.1},
                    },
                    {
                        "external_id": "cp-200",
                        "step": 200,
                        "token_count": 2_000_000,
                        "artifact_ref": "checkpoint-200",
                        "metrics": {"loss": 1.9},
                    },
                ],
                "models": [
                    {
                        "external_id": "model-base",
                        "name": "Legacy-LM",
                        "version": 1,
                        "kind": "base",
                        "source_checkpoint_external_id": "cp-100",
                        "artifact_ref": "checkpoint-100",
                        "tokenizer_external_id": "tokenizer-main",
                    },
                    {
                        "external_id": "model-sft",
                        "name": "Legacy-LM-Instruct",
                        "version": 1,
                        "kind": "sft",
                        "parent_model_external_id": "model-base",
                        "source_checkpoint_external_id": "cp-200",
                        "artifact_ref": "checkpoint-200",
                        "tokenizer_external_id": "tokenizer-main",
                        "is_official": True,
                    },
                ],
            }
        ],
        "evaluation_suites": [
            {
                "external_id": "suite-standard",
                "name": "Standard",
                "version": 1,
                "scoring_config": {"protocol": "v1"},
            }
        ],
        "evaluations": [
            {
                "external_id": "eval-sft",
                "name": "SFT standard evaluation",
                "model_external_ids": ["model-sft"],
                "suite_external_ids": ["suite-standard"],
                "summary": {"pass_rate": 0.75},
                "artifact_ref": "evaluation-report",
            }
        ],
        "gates": [
            {
                "external_id": "gate-sft",
                "subject_entity_type": "model",
                "subject_external_id": "model-sft",
                "gate_type": "release",
                "status": "passed",
            }
        ],
    }


def test_history_import_is_idempotent_and_preserves_lineage(client: TestClient) -> None:
    project = create_project(client, "History-Import")
    marker = uuid4().hex
    manifest = history_manifest(marker)
    first = client.post(f"/api/projects/{project['id']}/history-imports", json=manifest)
    assert first.status_code == 201, first.text
    assert first.json()["summary"]["created"]["checkpoint"] == 2
    assert first.json()["summary"]["created"]["model"] == 2

    registry = client.get(f"/api/projects/{project['id']}/models").json()
    assert len(registry["checkpoints"]) == 2
    assert {model["kind"] for model in registry["models"]} == {"base", "sft"}
    official = next(model for model in registry["models"] if model["is_official"])
    base = next(model for model in registry["models"] if model["kind"] == "base")
    assert official["parent_model_version_id"] == base["id"]

    replay = client.post(f"/api/projects/{project['id']}/history-imports", json=manifest)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    registry_after = client.get(f"/api/projects/{project['id']}/models").json()
    assert len(registry_after["checkpoints"]) == 2
    assert len(registry_after["models"]) == 2

    incremental = copy.deepcopy(manifest)
    incremental["training_runs"][0]["models"][1]["metadata"] = {
        "inference": {"adapter_id": "test.project-inference.v1"}
    }
    incremental["artifacts"].append(
        {
            "ref": "checkpoint-300",
            "kind": "checkpoint",
            "uri": f"file:///data/{marker}/checkpoint-300",
            "content_hash": "7" * 64,
            "availability_state": "available",
        }
    )
    incremental["training_runs"][0]["checkpoints"].append(
        {
            "external_id": "cp-300",
            "step": 300,
            "token_count": 3_000_000,
            "artifact_ref": "checkpoint-300",
        }
    )
    incremental["training_runs"][0]["models"].append(
        {
            "external_id": "model-dpo",
            "name": "Legacy-LM-DPO",
            "version": 1,
            "kind": "dpo",
            "parent_model_external_id": "model-sft",
            "source_checkpoint_external_id": "cp-300",
            "artifact_ref": "checkpoint-300",
        }
    )
    increment = client.post(f"/api/projects/{project['id']}/history-imports", json=incremental)
    assert increment.status_code == 201, increment.text
    assert increment.json()["summary"]["created"]["checkpoint"] == 1
    assert increment.json()["summary"]["created"]["model"] == 1
    incremented_registry = client.get(f"/api/projects/{project['id']}/models").json()
    assert len(incremented_registry["checkpoints"]) == 3
    assert {model["kind"] for model in incremented_registry["models"]} == {
        "base",
        "sft",
        "dpo",
    }
    updated_official = next(
        model for model in incremented_registry["models"] if model["is_official"]
    )
    assert updated_official["metadata_json"]["inference"] == {
        "adapter_id": "test.project-inference.v1"
    }


def test_training_result_can_register_multiple_external_checkpoints_and_models(
    client: TestClient,
) -> None:
    project = create_project(client, "Multi-Checkpoint")
    adapter_id = f"test.multi-output.{uuid4().hex}"
    plugin = client.post(
        "/api/plugins",
        json={
            "plugin_id": adapter_id,
            "version": "1.0.0",
            "kind": "training_engine",
            "entrypoint": "test:adapter",
        },
    )
    assert plugin.status_code == 201
    dataset = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Corpus-{uuid4().hex[:8]}",
            "source_type": "local",
            "initial_version": {},
            "start_import": False,
        },
    ).json()
    tokenizer = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": f"Tokenizer-{uuid4().hex[:8]}",
            "algorithm": "BPE",
            "vocab_size": 4096,
            "data_ref_type": "dataset_version",
            "data_ref_id": dataset["current_version"]["id"],
            "start": False,
        },
    ).json()
    client.post(f"/api/tokenizers/{tokenizer['id']}/lock")
    plan = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Multi-output",
            "stages": [{"name": "SFT", "type": "sft", "target_steps": 200}],
            "tokenizer_version_id": tokenizer["id"],
            "engine_plugin_id": adapter_id,
            "engine_config": {},
            "resource_request": {"gpus": 0},
        },
    ).json()
    run = client.post(f"/api/training-plans/{plan['id']}/runs", json={}).json()
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"multi-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [adapter_id, "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    task_id = run["task"]["id"]
    attempt_id = claim["attempt"]["id"]
    client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id},
    )
    complete = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "outputs": [
                    {
                        "ref": "weights-100",
                        "kind": "checkpoint",
                        "storage_type": "external",
                        "uri": f"file:///data/{uuid4().hex}/checkpoint-100",
                        "content_hash": "5" * 64,
                    },
                    {
                        "ref": "weights-200",
                        "kind": "checkpoint",
                        "storage_type": "external",
                        "uri": f"file:///data/{uuid4().hex}/checkpoint-200",
                        "content_hash": "6" * 64,
                    },
                ],
                "checkpoints": [
                    {"ref": "cp-100", "step": 100, "artifact_ref": "weights-100"},
                    {"ref": "cp-200", "step": 200, "artifact_ref": "weights-200"},
                ],
                "models": [
                    {
                        "ref": "base",
                        "name": "Task-LM",
                        "version": 1,
                        "kind": "base",
                        "source_checkpoint_ref": "cp-100",
                    },
                    {
                        "name": "Task-LM-Instruct",
                        "version": 1,
                        "kind": "sft",
                        "source_checkpoint_ref": "cp-200",
                        "parent_model_ref": "base",
                    },
                ],
            },
        },
    )
    assert complete.status_code == 200, complete.text
    result = complete.json()["result"]
    assert len(result["checkpoint_ids"]) == 2
    assert len(result["model_version_ids"]) == 2
    registry = client.get(f"/api/projects/{project['id']}/models").json()
    assert {model["kind"] for model in registry["models"]} == {"base", "sft"}
    assert len(registry["checkpoints"]) == 2

    second_plan = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Colliding model version",
            "stages": [{"name": "SFT", "type": "sft", "target_steps": 300}],
            "tokenizer_version_id": tokenizer["id"],
            "engine_plugin_id": adapter_id,
            "engine_config": {},
            "resource_request": {"gpus": 0},
        },
    ).json()
    second_run = client.post(
        f"/api/training-plans/{second_plan['id']}/runs", json={}
    ).json()
    second_claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    second_task_id = second_run["task"]["id"]
    second_attempt_id = second_claim["attempt"]["id"]
    client.post(
        f"/api/agent/tasks/{second_task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": second_attempt_id},
    )
    second_complete = client.post(
        f"/api/agent/tasks/{second_task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": second_attempt_id,
            "status": "succeeded",
            "result": {
                "outputs": [
                    {
                        "ref": "weights-300",
                        "kind": "checkpoint",
                        "storage_type": "external",
                        "uri": f"file:///data/{uuid4().hex}/checkpoint-300",
                        "content_hash": "7" * 64,
                    }
                ],
                "checkpoints": [
                    {"ref": "cp-300", "step": 300, "artifact_ref": "weights-300"}
                ],
                "models": [
                    {
                        "name": "Task-LM",
                        "version": 1,
                        "kind": "base",
                        "source_checkpoint_ref": "cp-300",
                    }
                ],
            },
        },
    )
    assert second_complete.status_code == 200, second_complete.text
    second_result = second_complete.json()["result"]
    assert second_result["models"][0]["version"] == 2
    assert second_result["models"][0]["metadata"] == {
        "requested_version": 1,
        "version_assignment": "next_available",
    }
    second_model_id = second_result["model_version_id"]
    second_model = client.get(f"/api/models/{second_model_id}").json()
    assert second_model["version"] == 2
    assert second_model["source_checkpoint_id"] == second_result["checkpoint_id"]
