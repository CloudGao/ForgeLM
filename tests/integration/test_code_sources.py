from __future__ import annotations

import io
import json
import sys
import tempfile
import zipfile

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Checkpoint,
    DataMixture,
    DataMixtureVersion,
    ModelVersion,
    TokenizerVersion,
)
from llm_platform.services.artifacts import artifact_storage
from llm_platform_core.checkpoint_compatibility import build_checkpoint_compatibility

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def create_project(client: TestClient, name: str) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": name,
            "target_model_name": name,
            "project_type": "pretrain",
            "workspace_uri": tempfile.mkdtemp(prefix=f"{name}-workspace-"),
        },
    )
    assert response.status_code == 201
    return response.json()


def code_archive(
    *,
    model_vocab_size: int | None = None,
    model_config: dict | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("example-model/train.py", "print('train')\n")
        archive.writestr("example-model/requirements.txt", "torch\n")
        if model_config is not None:
            archive.writestr("model-config.json", json.dumps(model_config))
        elif model_vocab_size is not None:
            archive.writestr(
                "model-config.json", f'{{"vocab_size":{model_vocab_size}}}'
            )
    return buffer.getvalue()


def test_checkpoint_compatibility_is_decided_in_preflight_and_frozen(
    client: TestClient,
) -> None:
    project = create_project(client, "Checkpoint-Compatibility-Contract")
    model_config = {
        "model_name": "ContractModel",
        "vocab_size": 256,
        "hidden_size": 960,
        "num_hidden_layers": 12,
        "use_cache": True,
    }
    source = client.post(
        f"/api/projects/{project['id']}/code-sources/upload",
        data={
            "name": "Checkpoint contract model",
            "entrypoints_json": '{"training":"example-model/train.py"}',
        },
        files={
            "file": (
                "checkpoint-contract.zip",
                code_archive(model_config=model_config),
                "application/zip",
            )
        },
    ).json()
    validation = client.post(
        f"/api/projects/{project['id']}/training-validation-sets/upload",
        data={"name": "Validation"},
        files={"file": ("validation.txt", b"sample\n", "text/plain")},
    ).json()
    with SessionLocal() as session:
        tokenizer = TokenizerVersion(
            project_id=project["id"],
            name="Contract Tokenizer",
            version=1,
            algorithm="BPE",
            vocab_size=256,
            data_ref_type="dataset_version",
            data_ref_id="contract-data",
            status="passed",
            locked=True,
        )
        session.add(tokenizer)
        session.commit()
        tokenizer_id = tokenizer.id

    plan = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Checkpoint compatibility plan",
            "stages": [
                {"name": "Gate 0", "target_steps": 1, "target_tokens": 100},
                {"name": "Gate 1", "target_steps": 1, "target_tokens": 100},
            ],
            "input_model_ref": {
                "type": "random_init",
                "architecture": "ContractModel",
                "config_path": "model-config.json",
                "checkpoint_compatibility": {
                    "contract": "ContractModel",
                    "contract_version": 1,
                    "runtime_only_fields": ["use_cache"],
                },
            },
            "tokenizer_version_id": tokenizer_id,
            "validation_set_id": validation["id"],
            "code_version_id": source["current_version"]["id"],
            "engine_config": {"executable": sys.executable, "args": ["-c", "pass"]},
            "resource_request": {"gpus": 0},
        },
    ).json()
    first_run = client.post(
        f"/api/training-plans/{plan['id']}/runs",
        json={"selected_stage_indexes": [0]},
    )
    assert first_run.status_code == 201, first_run.text
    with SessionLocal() as session:
        checkpoint_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="checkpoint",
            payload=b"checkpoint",
            filename="checkpoint.tar.gz",
        )
        checkpoint = Checkpoint(
            project_id=project["id"],
            training_run_id=first_run.json()["id"],
            step=1,
            token_count=100,
            artifact_id=checkpoint_artifact.id,
            metrics={
                "sample_offset": 1,
                "model_compatibility": build_checkpoint_compatibility(
                    {**model_config, "use_cache": False},
                    contract="ContractModel",
                    runtime_only_fields=["use_cache"],
                ),
            },
            status="available",
        )
        session.add(checkpoint)
        session.flush()
        source_model = ModelVersion(
            project_id=project["id"],
            name="ContractModel",
            version=1,
            kind="base",
            source_checkpoint_id=checkpoint.id,
            parameter_count=1,
            metadata_json={"architecture": {**model_config, "use_cache": False}},
        )
        session.add(source_model)
        session.commit()
        checkpoint_id = checkpoint.id

    compatible = client.post(
        f"/api/training-plans/{plan['id']}/preflight",
        json={"selected_stage_indexes": [1], "resume_checkpoint_id": checkpoint_id},
    )
    assert compatible.status_code == 200
    assert compatible.json()["status"] == "passed"
    resume_validation = next(
        item
        for item in compatible.json()["system_validations"]
        if item["id"] == "resume_checkpoint_compatible"
    )
    assert resume_validation["status"] == "passed"
    assert resume_validation["details"]["runtime_differences"] == [
        {
            "field": "use_cache",
            "expected_present": True,
            "actual_present": True,
            "expected": True,
            "actual": False,
        }
    ]
    continued = client.post(
        f"/api/training-plans/{plan['id']}/runs",
        json={"selected_stage_indexes": [1], "resume_checkpoint_id": checkpoint_id},
    )
    assert continued.status_code == 201, continued.text
    task_payload = continued.json()["task"]["payload"]
    assert task_payload["checkpoint_compatibility"]["contract"] == "ContractModel"
    assert task_payload["resume_checkpoint"]["model_compatibility"]["configuration"][
        "use_cache"
    ] is False
    assert task_payload["system_validation_contract"]["items"] == compatible.json()[
        "system_validations"
    ]

    with SessionLocal() as session:
        checkpoint = session.get(Checkpoint, checkpoint_id)
        checkpoint.metrics = {
            **checkpoint.metrics,
            "model_compatibility": build_checkpoint_compatibility(
                {**model_config, "hidden_size": 1024, "use_cache": False},
                contract="ContractModel",
                runtime_only_fields=["use_cache"],
            ),
        }
        session.commit()
    incompatible = client.post(
        f"/api/training-plans/{plan['id']}/preflight",
        json={"selected_stage_indexes": [1], "resume_checkpoint_id": checkpoint_id},
    )
    assert incompatible.json()["status"] == "blocked"
    issue = next(
        item
        for item in incompatible.json()["issues"]
        if item["code"] == "resume_checkpoint_model_incompatible"
    )
    assert issue["differences"][0]["field"] == "hidden_size"


def test_uploaded_code_version_can_be_frozen_into_training_task(client: TestClient) -> None:
    project = create_project(client, "Code-Upload-Training")
    validation_upload = client.post(
        f"/api/projects/{project['id']}/training-validation-sets/upload",
        data={"name": "Stable loss holdout"},
        files={
            "file": (
                "validation.jsonl",
                b'{"text":"first validation document"}\n{"content":"second document"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert validation_upload.status_code == 201, validation_upload.text
    validation_set = validation_upload.json()
    assert validation_set["version"] == 1
    assert validation_set["document_count"] == 2
    upload = client.post(
        f"/api/projects/{project['id']}/code-sources/upload",
        data={
            "name": "Example model source",
            "entrypoints_json": '{"training":"train.py"}',
            "dependency_spec": "requirements.txt",
        },
        files={"file": ("example-model.zip", code_archive(), "application/zip")},
    )
    assert upload.status_code == 201, upload.text
    source = upload.json()
    version = source["current_version"]
    assert version["state"] == "ready"
    assert version["artifact"]["kind"] == "source_code"
    assert version["artifact"]["content_hash"]

    with SessionLocal() as session:
        tokenizer = TokenizerVersion(
            project_id=project["id"],
            name="Code Test Tokenizer",
            version=1,
            algorithm="BPE",
            vocab_size=256,
            data_ref_type="dataset_version",
            data_ref_id="code-test-data",
            config={"special_tokens": ["<unk>", "<s>", "</s>"]},
            status="passed",
            locked=True,
        )
        session.add(tokenizer)
        session.flush()
        tokenizer_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenizer",
            payload=b"tokenizer-archive",
            filename="tokenizer.zip",
        )
        tokenizer.artifact_id = tokenizer_artifact.id
        session.commit()
        tokenizer_id = tokenizer.id
        tokenizer_artifact_id = tokenizer_artifact.id
        tokenizer_artifact_hash = tokenizer_artifact.content_hash

    plan_response = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Code-bound plan",
            "stages": [{"name": "smoke", "type": "pretrain", "target_steps": 1}],
            "tokenizer_version_id": tokenizer_id,
            "validation_set_id": validation_set["id"],
            "code_version_id": version["id"],
            "engine_config": {
                "executable": sys.executable,
                "args": ["train.py", "--validation-file", "__validation_file__"],
            },
            "resource_request": {"gpus": 0},
        },
    )
    assert plan_response.status_code == 201, plan_response.text
    plan = plan_response.json()
    assert plan["current_revision"]["code_version_id"] == version["id"]
    preflight = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["status"] == "passed"
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201, started.text
    started_payload = started.json()
    assert started_payload["task"]["payload"]["training_run_id"] == started_payload["id"]
    binding = started_payload["task"]["payload"]["code"]
    assert binding["version_id"] == version["id"]
    assert binding["artifact_id"] == version["artifact_id"]
    assert binding["content_hash"] == version["artifact"]["content_hash"]
    assert binding["entrypoints"] == {"training": "train.py"}
    tokenizer_binding = started_payload["task"]["payload"]["tokenizer"]
    assert tokenizer_binding["version_id"] == tokenizer_id
    assert tokenizer_binding["artifact_id"] == tokenizer_artifact_id
    assert tokenizer_binding["content_hash"] == tokenizer_artifact_hash
    assert tokenizer_binding["special_tokens"] == ["<unk>", "<s>", "</s>"]
    validation_binding = started_payload["task"]["payload"]["validation_set"]
    assert validation_binding["validation_set_id"] == validation_set["id"]
    assert validation_binding["document_count"] == 2
    assert validation_binding["content_hash"]

    reupload = client.post(
        f"/api/code-sources/{source['id']}/versions/upload",
        files={"file": ("example-model-v2.zip", code_archive(), "application/zip")},
    )
    assert reupload.status_code == 201, reupload.text
    assert reupload.json()["version"] == 2
    assert reupload.json()["entrypoints"] == {"training": "train.py"}

    other_project = create_project(client, "Code-Upload-Other")
    invalid = client.post(
        f"/api/projects/{other_project['id']}/training-plans",
        json={
            "name": "Invalid cross-project plan",
            "stages": [{"name": "smoke"}],
            "code_version_id": version["id"],
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_code_version"


def test_training_plan_template_placeholder_is_reported_without_a_start_gate(
    client: TestClient,
) -> None:
    project = create_project(client, "Code-Template-Guard")
    plan_response = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Validation pending plan",
            "stages": [{"name": "smoke", "type": "pretrain", "target_steps": 1}],
            "engine_config": {
                "executable": sys.executable,
                "args": ["train.py", "--validation-file", "${VALIDATION_FILE}"],
            },
            "resource_request": {"gpus": 0},
        },
    )
    assert plan_response.status_code == 201, plan_response.text
    plan = plan_response.json()

    preflight = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["status"] == "blocked"
    assert any(
        issue["code"] == "missing_training_validation_set"
        for issue in preflight.json()["issues"]
    )

    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201
    assert started.json()["preflight"]["status"] == "blocked"


def test_training_preflight_reports_tokenizer_model_vocab_mismatch_without_gate(
    client: TestClient,
) -> None:
    project = create_project(client, "Vocabulary-Contract")
    validation = client.post(
        f"/api/projects/{project['id']}/training-validation-sets/upload",
        data={"name": "Validation"},
        files={"file": ("validation.txt", b"sample\n", "text/plain")},
    ).json()
    source = client.post(
        f"/api/projects/{project['id']}/code-sources/upload",
        data={
            "name": "Vocabulary-bound model",
            "entrypoints_json": '{"training":"example-model/train.py"}',
        },
        files={
            "file": (
                "vocabulary-model.zip",
                code_archive(model_vocab_size=32_000),
                "application/zip",
            )
        },
    ).json()
    with SessionLocal() as session:
        tokenizer = TokenizerVersion(
            project_id=project["id"],
            name="Mismatched Tokenizer",
            version=1,
            algorithm="BPE",
            vocab_size=32_768,
            data_ref_type="dataset_version",
            data_ref_id="data",
            status="passed",
            locked=True,
        )
        session.add(tokenizer)
        session.flush()
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenizer",
            payload=b"tokenizer",
            filename="tokenizer.zip",
        )
        tokenizer.artifact_id = artifact.id
        session.commit()
        tokenizer_id = tokenizer.id

    plan = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Vocabulary contract plan",
            "stages": [{"name": "smoke", "target_steps": 1}],
            "input_model_ref": {
                "type": "random_init",
                "config_path": "model-config.json",
            },
            "tokenizer_version_id": tokenizer_id,
            "validation_set_id": validation["id"],
            "code_version_id": source["current_version"]["id"],
            "engine_config": {"executable": sys.executable, "args": ["-c", "pass"]},
            "resource_request": {"gpus": 0},
        },
    ).json()

    preflight = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert preflight.json()["status"] == "blocked"
    issue = next(
        item
        for item in preflight.json()["issues"]
        if item["code"] == "tokenizer_model_vocab_mismatch"
    )
    assert "32,000" in issue["message"]
    assert "32,768" in issue["message"]
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201
    assert started.json()["preflight"]["status"] == "blocked"

    with SessionLocal() as session:
        tokenizer = session.get(TokenizerVersion, tokenizer_id)
        tokenizer.vocab_size = 32_000
        session.commit()
    passed = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert passed.json()["status"] == "passed"
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201
    assert started.json()["task"]["payload"]["model_vocab_size"] == 32_000


def test_training_validation_sets_are_versioned_and_project_scoped(
    client: TestClient,
) -> None:
    project = create_project(client, "Validation-Set-Versions")
    other_project = create_project(client, "Validation-Set-Other")
    first = client.post(
        f"/api/projects/{project['id']}/training-validation-sets/upload",
        data={"name": "Loss holdout"},
        files={"file": ("validation.txt", b"alpha\nbeta\n", "text/plain")},
    )
    second = client.post(
        f"/api/projects/{project['id']}/training-validation-sets/upload",
        data={"name": "Loss holdout"},
        files={"file": ("validation.txt", b"alpha\nbeta\ngamma\n", "text/plain")},
    )
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert first.json()["version"] == 1
    assert second.json()["version"] == 2

    listed = client.get(f"/api/projects/{project['id']}/training-validation-sets")
    assert listed.status_code == 200
    assert [item["version"] for item in listed.json()] == [2, 1]

    invalid_plan = client.post(
        f"/api/projects/{other_project['id']}/training-plans",
        json={
            "name": "Cross-project validation",
            "stages": [{"name": "smoke"}],
            "validation_set_id": first.json()["id"],
        },
    )
    assert invalid_plan.status_code == 422
    assert invalid_plan.json()["detail"]["code"] == "invalid_training_validation_set"


def test_ready_data_mixture_can_be_frozen_as_training_validation_input(
    client: TestClient,
) -> None:
    project = create_project(client, "Validation-Mixture-Training")
    with SessionLocal() as session:
        tokenizer = TokenizerVersion(
            project_id=project["id"],
            name="Validation Mixture Tokenizer",
            version=1,
            algorithm="BPE",
            vocab_size=256,
            data_ref_type="dataset_version",
            data_ref_id="validation-mixture-data",
            status="passed",
            locked=True,
        )
        mixture = DataMixture(project_id=project["id"], name="2M validation holdout")
        session.add_all([tokenizer, mixture])
        session.flush()
        manifest = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="data_mixture_manifest",
            payload=b'{"format":"llm-mixture-manifest-v1","sources":[]}\n',
            filename="validation-mixture.jsonl",
        )
        version = DataMixtureVersion(
            mixture_id=mixture.id,
            version=1,
            entries=[],
            target_tokens=2_000_000,
            locked=True,
            status="ready",
            manifest_artifact_id=manifest.id,
            manifest_hash=manifest.content_hash,
            resolved_document_count=1024,
            resolved_token_estimate=2_000_000,
        )
        session.add(version)
        session.flush()
        mixture.current_version_id = version.id
        session.commit()
        tokenizer_id = tokenizer.id
        mixture_version_id = version.id

    plan_response = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Mixture validation plan",
            "stages": [{"name": "smoke", "type": "pretrain", "target_steps": 1}],
            "tokenizer_version_id": tokenizer_id,
            "validation_mixture_version_id": mixture_version_id,
            "engine_config": {"executable": sys.executable, "args": ["-c", "print('ok')"]},
            "resource_request": {"gpus": 0},
        },
    )
    assert plan_response.status_code == 201, plan_response.text
    plan = plan_response.json()
    preflight = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["status"] == "passed"
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201, started.text
    binding = started.json()["task"]["payload"]["validation_data_input"]
    assert binding["mixture_version_id"] == mixture_version_id
    assert binding["target_tokens"] == 2_000_000
    assert binding["usage"] == "training_validation"
    assert binding["validation_sampling"] == {
        "strategy": "weighted_random_materialized_v1",
        "seed": 42,
        "materialization_format": "forgelm-validation-text-v1",
    }


def test_worker_code_import_task_finalizes_version(client: TestClient, tmp_path) -> None:
    project = create_project(client, "Code-Worker-Import")
    source_response = client.post(
        f"/api/projects/{project['id']}/code-sources",
        json={
            "name": "Worker local code",
            "source_type": "local",
            "source_uri": str(tmp_path / "model-code"),
            "entrypoints": {"training": "train.py"},
        },
    )
    assert source_response.status_code == 201, source_response.text
    source = source_response.json()
    assert source["current_version"]["state"] == "queued"

    worker_response = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": "code-import-worker",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.code-import", "builtin.local-process"],
        },
    )
    assert worker_response.status_code == 200
    worker = worker_response.json()
    claimed_response = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    )
    assert claimed_response.status_code == 200
    claimed = claimed_response.json()
    assert claimed["task"]["adapter_id"] == "builtin.code-import"
    task_id = claimed["task"]["id"]
    attempt_id = claimed["attempt"]["id"]
    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id},
    )
    assert started.status_code == 200
    artifact = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "source_code",
            "metadata_json": "{}",
        },
        files={"file": ("snapshot.zip", code_archive(), "application/zip")},
    )
    assert artifact.status_code == 200
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "git_commit": "a" * 40,
                "source_snapshot": {"file_count": 2, "size_bytes": 24},
                "output_artifact_ids": [artifact.json()["id"]],
            },
        },
    )
    assert completed.status_code == 200, completed.text
    refreshed = client.get(f"/api/projects/{project['id']}/code-sources").json()[0]
    assert refreshed["status"] == "ready"
    assert refreshed["current_version"]["state"] == "ready"
    assert refreshed["current_version"]["git_commit"] == "a" * 40
    assert refreshed["current_version"]["artifact_id"] == artifact.json()["id"]

    second = client.post(f"/api/code-sources/{source['id']}/versions", json={})
    assert second.status_code == 201
    assert second.json()["version"] == 2
