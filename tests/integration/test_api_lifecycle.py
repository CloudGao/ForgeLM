from __future__ import annotations

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Checkpoint,
    DataMixtureVersion,
    Dataset,
    DatasetVersion,
    EvalSuiteVersion,
    ModelVersion,
    Project,
    Task,
    TaskAttempt,
    TaskEvent,
    TokenizedCorpusVersion,
    TokenizerVersion,
    TrainingDataViewVersion,
    utcnow,
)
from llm_platform.routers.training import _freeze_training_data_view_inputs
from llm_platform.services.artifacts import artifact_storage
from llm_platform.services.tasks import create_task

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def create_project(client: TestClient, prefix: str, workspace_uri: str = "") -> dict:
    workspace_uri = workspace_uri or tempfile.mkdtemp(prefix=f"{prefix}-workspace-")
    response = client.post(
        "/api/projects",
        json={
            "name": f"{prefix}-{uuid4().hex[:8]}",
            "target_model_name": f"{prefix}-LM",
            "workspace_uri": workspace_uri,
            "project_type": "pretrain",
            "target_parameters": "42M",
            "primary_languages": ["zh"],
            "target_capabilities": ["test"],
        },
    )
    assert response.status_code == 201
    return response.json()


def create_dataset(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/datasets",
        json={
            "name": f"Corpus-{uuid4().hex[:8]}",
            "source_type": "local",
            "license": "MIT",
            "language": "zh",
            "initial_version": {
                "tier": "bronze",
                "document_count": 10,
                "token_estimate": 2_000,
                "train_ready": True,
            },
            "start_import": False,
        },
    )
    assert response.status_code == 201
    dataset = response.json()
    version_id = dataset["current_version"]["id"]
    lines = [
        json.dumps(
            {
                "dataset_version_id": version_id,
                "format": "llm-dataset-records-v1",
                "type": "header",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    for index in range(10):
        lines.append(
            json.dumps(
                {
                    "byte_length": 100,
                    "character_length": 100,
                    "content_hash": f"{index:064d}",
                    "language": "zh",
                    "logical_path": "train/part.jsonl",
                    "record_id": f"record-{index:03d}",
                    "record_index": index,
                    "token_estimate": 200,
                    "type": "record",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    with SessionLocal() as session:
        dataset_artifact = artifact_storage.put_bytes(
            session,
            project_id=project_id,
            kind="dataset_snapshot",
            payload=b"synthetic integration dataset",
            filename=f"{version_id}.txt",
            media_type="text/plain",
        )
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project_id,
            kind="dataset_record_manifest",
            payload=("\n".join(lines) + "\n").encode(),
            filename=f"{version_id}.jsonl",
            media_type="application/x-ndjson",
        )
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        version.artifact_id = dataset_artifact.id
        version.record_manifest_artifact_id = artifact.id
        session.commit()
    return client.get(f"/api/datasets/{dataset['id']}").json()


def create_locked_tokenizer(client: TestClient, project_id: str, data_ref_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/tokenizers",
        json={
            "name": f"Tokenizer-{uuid4().hex[:8]}",
            "algorithm": "BPE",
            "vocab_size": 8_192,
            "data_ref_type": "dataset_version",
            "data_ref_id": data_ref_id,
            "start": False,
        },
    )
    assert response.status_code == 201
    lock_response = client.post(f"/api/tokenizers/{response.json()['id']}/lock")
    assert lock_response.status_code == 200
    return lock_response.json()


def test_task_list_supports_stable_offset_pagination(client: TestClient) -> None:
    project = create_project(client, "task-pagination")
    base_time = utcnow()
    with SessionLocal() as session:
        for index in range(23):
            created_at = base_time + timedelta(seconds=index)
            session.add(
                Task(
                    project_id=project["id"],
                    kind="test",
                    title=f"分页任务 {index}",
                    adapter_id="builtin.command.data",
                    launcher_id="builtin.local-process",
                    desired_state="running",
                    observed_state="succeeded",
                    progress=1,
                    payload={},
                    result={},
                    error={},
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
        session.commit()

    first_page = client.get(
        "/api/tasks",
        params={"project_id": project["id"], "limit": 20, "offset": 0},
    )
    second_page = client.get(
        "/api/tasks",
        params={"project_id": project["id"], "limit": 20, "offset": 20},
    )

    assert first_page.status_code == 200
    assert [item["title"] for item in first_page.json()] == [
        f"分页任务 {index}" for index in range(22, 2, -1)
    ]
    assert second_page.status_code == 200
    assert [item["title"] for item in second_page.json()] == [
        "分页任务 2",
        "分页任务 1",
        "分页任务 0",
    ]

    with SessionLocal() as session:
        session.add(
            Task(
                project_id=project["id"],
                kind="data_query",
                title="交互式数据查询",
                adapter_id="builtin.dataset-viewer",
                launcher_id="builtin.local-process",
                desired_state="running",
                observed_state="succeeded",
                progress=1,
                payload={},
                result={},
                error={},
                created_at=base_time + timedelta(minutes=1),
                updated_at=base_time + timedelta(minutes=1),
            )
        )
        session.commit()

    unfiltered = client.get(
        "/api/tasks",
        params={"project_id": project["id"], "limit": 1},
    )
    task_center_page = client.get(
        "/api/tasks",
        params={
            "project_id": project["id"],
            "exclude_kind": "data_query",
            "limit": 1,
        },
    )
    assert unfiltered.json()[0]["title"] == "交互式数据查询"
    assert task_center_page.json()[0]["title"] == "分页任务 22"
    assert client.get("/api/tasks", params={"offset": -1}).status_code == 422


def test_tokenizer_defaults_to_hf_and_freezes_single_dataset_input(client: TestClient) -> None:
    project = create_project(client, "HF-Tokenizer")
    dataset = create_dataset(client, project["id"])
    version = dataset["current_version"]

    response = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": "Default HF Tokenizer",
            "algorithm": "BPE",
            "vocab_size": 4096,
            "data_ref_type": "dataset_version",
            "data_ref_id": version["id"],
            "config": {"sample_tokens": 1000, "normalization": "NFKC"},
            "resource_request": {"gpus": 1},
            "start": True,
        },
    )

    assert response.status_code == 201
    with SessionLocal() as session:
        task = session.get(Task, response.json()["task_id"])
        assert task is not None
        assert task.adapter_id == "builtin.hf-tokenizer"
        assert task.payload["resource_request"]["gpus"] == 0
        assert task.payload["command"] == {}
        assert task.payload["data_input"] == {
            "format": "llm-dataset-input-v1",
            "dataset_version_id": version["id"],
            "dataset_artifact_id": version["artifact_id"],
            "dataset_artifact_hash": task.payload["data_input"]["dataset_artifact_hash"],
            "record_manifest_artifact_id": version["record_manifest_artifact_id"],
            "record_manifest_hash": task.payload["data_input"]["record_manifest_hash"],
        }


def test_tokenizer_same_name_creates_next_immutable_version(client: TestClient) -> None:
    project = create_project(client, "Tokenizer-Revision")
    dataset = create_dataset(client, project["id"])
    endpoint = f"/api/projects/{project['id']}/tokenizers"
    base_payload = {
        "name": "Editable Tokenizer",
        "algorithm": "BPE",
        "vocab_size": 4096,
        "data_ref_type": "dataset_version",
        "data_ref_id": dataset["current_version"]["id"],
        "start": False,
    }

    first = client.post(
        endpoint,
        json={**base_payload, "config": {"sample_tokens": 1_000, "normalization": "NFKC"}},
    )
    second = client.post(
        endpoint,
        json={**base_payload, "config": {"sample_tokens": 2_000, "normalization": "NFC"}},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert first.json()["version"] == 1
    assert second.json()["version"] == 2
    versions = {
        item["version"]: item
        for item in client.get(endpoint).json()
        if item["name"] == "Editable Tokenizer"
    }
    assert versions[1]["config"]["sample_tokens"] == 1_000
    assert versions[1]["config"]["normalization"] == "NFKC"
    assert versions[2]["config"]["sample_tokens"] == 2_000
    assert versions[2]["config"]["normalization"] == "NFC"
    assert versions[1]["config"]["gate_policy"]["version"] == "tokenizer-quality-gate-v1"


def complete_mixture_resolution(client: TestClient, mixture_version_id: str) -> dict:
    payload = (
        json.dumps(
            {
                "format": "llm-mixture-manifest-v1",
                "mixture_version_id": mixture_version_id,
                "type": "header",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"mixture-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.data-mixture", "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claim["task"]["payload"]["mixture_version_id"] == mixture_version_id
    task_id = claim["task"]["id"]
    attempt_id = claim["attempt"]["id"]
    assert (
        client.post(
            f"/api/agent/tasks/{task_id}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker["id"], "attempt_id": attempt_id},
        ).status_code
        == 200
    )
    artifact = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "data_mixture_manifest",
            "metadata_json": "{}",
        },
        files={"file": ("mixture.jsonl", payload, "application/x-ndjson")},
    ).json()
    complete = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "resolver_version": "stable_hash_selector_v1",
                "sampler_version": "weighted_hash_sampler_v1",
                "resolved_document_count": 5,
                "resolved_token_estimate": 1_000,
                "output_artifacts": [
                    {
                        "ref": "mixture_manifest",
                        "artifact_id": artifact["id"],
                        "kind": "data_mixture_manifest",
                        "storage_type": "managed",
                    }
                ],
            },
        },
    )
    assert complete.status_code == 200
    with SessionLocal() as session:
        version = session.get(DataMixtureVersion, mixture_version_id)
        assert version is not None
        return version.to_dict()


def test_tokenized_corpus_creation_freezes_all_behavior_and_v1_is_read_only(
    client: TestClient,
) -> None:
    project = create_project(client, "Tokenized-Dataset")
    dataset = create_dataset(client, project["id"])
    tokenizer = create_locked_tokenizer(
        client, project["id"], dataset["current_version"]["id"]
    )
    with SessionLocal() as session:
        tokenizer_version = session.get(TokenizerVersion, tokenizer["id"])
        assert tokenizer_version is not None
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenizer",
            payload=b"immutable-tokenizer-archive",
            filename="tokenizer.zip",
        )
        tokenizer_version.artifact_id = artifact.id
        session.commit()

    legacy = client.post(
        f"/api/projects/{project['id']}/tokenized-datasets",
        json={
            "mixture_version_id": "legacy-mixture",
            "tokenizer_version_id": tokenizer["id"],
            "sequence_length": 16,
            "target_tokens": 4_096,
        },
    )
    assert legacy.status_code == 410
    assert legacy.json()["detail"]["code"] == "legacy_tokenized_dataset_read_only"

    request = {
        "dataset_version_id": dataset["current_version"]["id"],
        "tokenizer_version_id": tokenizer["id"],
        "selection": {"mode": "all", "estimated_token_limit": None},
        "encoding": {
            "eos_policy": "append_if_available",
            "empty_document_policy": "skip",
            "token_dtype": "auto",
            "shard_token_limit": 1_048_576,
        },
        "resource_request": {"gpus": 1},
    }
    first = client.post(
        f"/api/projects/{project['id']}/tokenized-corpora", json=request
    )
    repeated = client.post(
        f"/api/projects/{project['id']}/tokenized-corpora", json=request
    )

    assert first.status_code == 201
    assert repeated.status_code == 201
    assert repeated.json()["id"] == first.json()["id"]
    assert repeated.json()["identity_hash"] == first.json()["identity_hash"]
    with SessionLocal() as session:
        task = session.get(Task, first.json()["task_id"])
        assert task is not None
        assert task.kind == "data_tokenize_corpus"
        assert task.adapter_id == "builtin.tokenize-corpus"
        assert task.payload["resource_request"]["gpus"] == 0
        assert task.payload["identity_hash"] == first.json()["identity_hash"]
        assert task.payload["data_input"]["dataset_version_id"] == dataset["current_version"]["id"]
        assert task.payload["tokenizer"]["version_id"] == tokenizer["id"]
        assert task.payload["selection"] == request["selection"]
        assert task.payload["encoding"] == request["encoding"]
        assert "target_tokens" not in task.payload
        assert "sequence_length" not in task.payload

    changed_policy = client.post(
        f"/api/projects/{project['id']}/tokenized-corpora",
        json={
            **request,
            "encoding": {**request["encoding"], "eos_policy": "none"},
        },
    )
    assert changed_policy.status_code == 201
    assert changed_policy.json()["id"] != first.json()["id"]
    assert changed_policy.json()["identity_hash"] != first.json()["identity_hash"]


def test_training_data_view_preflight_uses_actual_corpus_capacity(
    client: TestClient,
) -> None:
    project = create_project(client, "Training-Data-View")
    dataset = create_dataset(client, project["id"])
    tokenizer = create_locked_tokenizer(
        client, project["id"], dataset["current_version"]["id"]
    )
    mixture_response = client.post(
        f"/api/projects/{project['id']}/mixtures",
        json={
            "name": "Full source",
            "target_tokens": 50_000,
            "entries": [
                {
                    "dataset_version_id": dataset["current_version"]["id"],
                    "name": dataset["name"],
                    "weight": 1,
                    "available_tokens": 50_000,
                    "selected_tokens": 50_000,
                    "selection": {"sample_ratio": 1, "seed": 42},
                }
            ],
        },
    )
    assert mixture_response.status_code == 201
    mixture_version_id = mixture_response.json()["current_version"]["id"]
    complete_mixture_resolution(client, mixture_version_id)
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenized_corpus",
            payload=b"corpus-manifest",
            filename="manifest.json",
        )
        corpus = TokenizedCorpusVersion(
            project_id=project["id"],
            dataset_version_id=dataset["current_version"]["id"],
            tokenizer_version_id=tokenizer["id"],
            identity_hash="a" * 64,
            status="ready",
            artifact_id=artifact.id,
            manifest_hash=artifact.content_hash,
            document_count=10,
            token_count=2_000,
            shard_count=1,
            selection_config={"mode": "all", "estimated_token_limit": None},
            encoding_config={
                "eos_policy": "append_if_available",
                "empty_document_policy": "skip",
                "token_dtype": "auto",
                "shard_token_limit": 1_048_576,
            },
        )
        session.add(corpus)
        session.commit()
        corpus_id = corpus.id

    request = {
        "mixture_version_id": mixture_version_id,
        "tokenizer_version_id": tokenizer["id"],
        "corpus_version_ids": [corpus_id],
        "config": {
            "sequence_length": 16,
            "requested_tokens": 4_096,
            "consumption_policy": "unique_only",
            "tail_policy": "drop",
            "padding_token_id": None,
            "packing_policy": "document_stream_v1",
            "sampling_policy": "weighted_deficit_v1",
            "seed": 42,
        },
    }
    invalid_block_shuffle = client.post(
        f"/api/projects/{project['id']}/training-data-views/preflight",
        json={
            **request,
            "config": {
                **request["config"],
                "sequence_order_policy": "block_shuffle_v1",
            },
        },
    )
    assert invalid_block_shuffle.status_code == 422
    unknown_strategy = client.post(
        f"/api/projects/{project['id']}/training-data-views/preflight",
        json={
            **request,
            "config": {
                **request["config"],
                "sequence_order_policy": "unknown_shuffle_v1",
            },
        },
    )
    assert unknown_strategy.status_code == 422
    preflight = client.post(
        f"/api/projects/{project['id']}/training-data-views/preflight",
        json=request,
    )
    assert preflight.status_code == 200
    assert preflight.json()["ready"] is False
    assert preflight.json()["unique_sequence_capacity"] == 125
    assert preflight.json()["requested_sequences"] == 256
    assert preflight.json()["storage_estimate"]["materialized_sequence_bytes"] == 0
    assert all(
        item["category"] == "system_validation"
        for item in preflight.json()["system_validations"]
    )

    materialized = client.post(
        f"/api/projects/{project['id']}/training-data-views/preflight",
        json={
            **request,
            "config": {
                **request["config"],
                "sequence_order_policy": "materialized_global_shuffle_v1",
            },
        },
    )
    assert materialized.status_code == 200
    assert materialized.json()["sequence_order_strategy"]["storage_mode"] == (
        "materialized"
    )
    assert materialized.json()["storage_estimate"] == {
        "materialized_sequence_bytes": 4_750,
        "worker_free_space_required_bytes": 1_073_746_574,
        "token_bytes": 2,
    }
    strategies = client.get("/api/training-data-view-strategies")
    assert strategies.status_code == 200
    assert {item["id"] for item in strategies.json()} == {
        "sequential_v1",
        "block_shuffle_v1",
        "materialized_global_shuffle_v1",
    }

    repeated = client.post(
        f"/api/projects/{project['id']}/training-data-views/preflight",
        json={
            **request,
            "config": {**request["config"], "consumption_policy": "repeat_epochs"},
        },
    )
    assert repeated.status_code == 200
    assert repeated.json()["ready"] is True
    assert repeated.json()["required_epochs"] == 3

    created = client.post(
        f"/api/projects/{project['id']}/training-data-views", json=request
    )
    assert created.status_code == 201
    with SessionLocal() as session:
        task = session.get(Task, created.json()["task_id"])
        assert task is not None
        assert task.adapter_id == "builtin.build-training-index"
        assert task.payload["configuration"] == request["config"]
        assert task.payload["preflight"]["ready"] is False

        materialized_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="training_data_index",
            payload=b"self-contained-materialized-view",
            filename="manifest.json",
        )
        materialized_view = TrainingDataViewVersion(
            project_id=project["id"],
            mixture_version_id=mixture_version_id,
            tokenizer_version_id=tokenizer["id"],
            identity_hash="9" * 64,
            sequence_length=16,
            requested_tokens=4_096,
            token_count=4_096,
            sequence_count=256,
            status="ready",
            artifact_id=materialized_artifact.id,
            manifest_hash=materialized_artifact.content_hash,
            config_snapshot={
                "configuration": {
                    **request["config"],
                    "sequence_order_policy": "materialized_global_shuffle_v1",
                },
                "corpora": [{"version_id": "missing-corpus-version"}],
            },
        )
        session.add(materialized_view)
        session.flush()
        frozen = _freeze_training_data_view_inputs(
            session,
            project["id"],
            [{"training_data_view_id": materialized_view.id}],
            tokenizer["id"],
        )
        assert len(frozen) == 1
        assert frozen[0]["tokenized_corpora"] == []
        assert frozen[0]["source_corpora"] == [
            {"version_id": "missing-corpus-version"}
        ]


@pytest.mark.parametrize("reuse_full", [False, True])
def test_filtered_mixture_automatic_preparation_and_corpus_reuse(client: TestClient, reuse_full: bool) -> None:
    project = create_project(client, "Filtered-Preparation")
    dataset = create_dataset(client, project["id"])
    dataset_id = dataset["current_version"]["id"]
    tokenizer = create_locked_tokenizer(client, project["id"], dataset_id)
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(session, project_id=project["id"], kind="tokenizer", payload=b"tokenizer", filename="tokenizer.zip")
        session.get(TokenizerVersion, tokenizer["id"]).artifact_id = artifact.id
        session.commit()
    mixture = client.post(f"/api/projects/{project['id']}/mixtures", json={"name": "Filtered", "entries": [{"dataset_version_id": dataset_id, "weight": 1, "selection": {"include_paths": ["train/**"], "sample_ratio": 0.5}}]}).json()
    mixture_id = mixture["current_version"]["id"]
    complete_mixture_resolution(client, mixture_id)
    with SessionLocal() as session:
        version = session.get(DataMixtureVersion, mixture_id)
        version.entries = [{**entry, "selected_document_count": 5, "selected_tokens": 1000} for entry in version.entries]
        session.commit()
    encoding = {"eos_policy": "append_required", "empty_document_policy": "skip", "token_dtype": "auto", "shard_token_limit": 1048576}
    full = None
    if reuse_full:
        response = client.post(f"/api/projects/{project['id']}/tokenized-corpora", json={"dataset_version_id": dataset_id, "tokenizer_version_id": tokenizer["id"], "selection": {"mode": "all"}, "encoding": encoding})
        assert response.status_code == 201, response.text
        full = response.json()
        with SessionLocal() as session:
            corpus = session.get(TokenizedCorpusVersion, full["id"])
            artifact = artifact_storage.put_bytes(session, project_id=project["id"], kind="tokenized_corpus", payload=b"corpus", filename="manifest.json")
            corpus.status, corpus.artifact_id, corpus.manifest_hash = "ready", artifact.id, artifact.content_hash
            corpus.document_count, corpus.token_count = 10, 2000
            session.commit()
    response = client.post(f"/api/projects/{project['id']}/training-plans", json={
        "name": "Filtered plan", "stages": [{"name": "Pretrain", "type": "pretrain", "target_tokens": 128, "effective_batch_tokens": 16, "sequence_length": 16, "data_ref_id": mixture_id, "data_preparation": {"corpus": {"selection": {"mode": "all"}, "encoding": encoding}, "view": {"consumption_policy": "unique_only", "tail_policy": "drop", "packing_policy": "document_stream_v1", "sampling_policy": "weighted_deficit_v1", "seed": 42}}}],
        "tokenizer_version_id": tokenizer["id"], "engine_plugin_id": "builtin.command.training", "engine_config": {"executable": sys.executable, "args": ["-c", "pass"]}, "resource_request": {"gpus": 0},
    })
    assert response.status_code == 201, response.text
    plan = response.json()
    first = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    assert first.status_code == 200, first.text
    prepared = first.json()
    corpus = prepared["corpora"][0]
    assert corpus["project_id"] == project["id"]
    if reuse_full:
        assert corpus["id"] == full["id"]
        assert prepared["status"] == "preparing_index"
        with SessionLocal() as session:
            task = session.get(Task, prepared["data_index"]["task_id"])
            assert task.payload["preflight"]["selection_capacity_pending"]
            assert not task.payload["preflight"]["ready"]
    else:
        assert prepared["status"] == "preparing_corpora"
        assert corpus["selection_config"]["mode"] == "mixture"
        with SessionLocal() as session:
            task = session.get(Task, corpus["task_id"])
            assert task.payload["selection"]["mixture_version_id"] == mixture_id
    assert task.payload["training_selection_inputs"][0]["mixture_version_id"] == mixture_id
    assert task.payload["training_selection_inputs"][0]["selected_document_count"] == 5
    repeated = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["corpora"][0]["id"] == corpus["id"]
    if reuse_full:
        assert repeated.json()["data_index"]["id"] == prepared["data_index"]["id"]


def test_training_preflight_blocks_invalid_platform_warmup(client: TestClient) -> None:
    project = create_project(client, "Warmup-Validation")
    created = client.post(f"/api/projects/{project['id']}/training-plans", json={
        "name": "Invalid warmup", "stages": [{"name": "Stage 0", "type": "pretrain",
        "target_tokens": 1000, "effective_batch_tokens": 100, "warmup_ratio": 1.2}],
    })
    assert created.status_code == 201
    checked = client.post(f"/api/training-plans/{created.json()['id']}/preflight", json={})
    assert checked.status_code == 200
    assert checked.json()["status"] == "blocked"
    assert any(issue["code"] == "invalid_warmup_schedule" for issue in checked.json()["issues"])


def test_training_plan_idempotently_prepares_declared_data_intent(
    client: TestClient,
) -> None:
    project = create_project(client, "Managed-Training-Data")
    dataset = create_dataset(client, project["id"])
    tokenizer = create_locked_tokenizer(
        client, project["id"], dataset["current_version"]["id"]
    )
    with SessionLocal() as session:
        tokenizer_version = session.get(TokenizerVersion, tokenizer["id"])
        assert tokenizer_version is not None
        tokenizer_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenizer",
            payload=b"managed-tokenizer",
            filename="tokenizer.zip",
        )
        tokenizer_version.artifact_id = tokenizer_artifact.id
        session.commit()
    mixture_response = client.post(
        f"/api/projects/{project['id']}/mixtures",
        json={
            "name": "Managed full source",
            "target_tokens": 50_000,
            "entries": [
                {
                    "dataset_version_id": dataset["current_version"]["id"],
                    "name": dataset["name"],
                    "weight": 1,
                    "available_tokens": 50_000,
                    "selected_tokens": 50_000,
                    "selection": {"sample_ratio": 1, "seed": 42},
                }
            ],
        },
    )
    assert mixture_response.status_code == 201
    mixture_version_id = mixture_response.json()["current_version"]["id"]
    complete_mixture_resolution(client, mixture_version_id)
    validation_mixture_response = client.post(
        f"/api/projects/{project['id']}/mixtures",
        json={
            "name": "Managed validation holdout",
            "target_tokens": 16,
            "entries": [
                {
                    "dataset_version_id": dataset["current_version"]["id"],
                    "name": dataset["name"],
                    "weight": 1,
                    "available_tokens": 50_000,
                    "selected_tokens": 50_000,
                    "selection": {"sample_ratio": 1, "seed": 42},
                }
            ],
        },
    )
    assert validation_mixture_response.status_code == 201
    validation_mixture_version_id = validation_mixture_response.json()[
        "current_version"
    ]["id"]
    validation_mixture_version = complete_mixture_resolution(
        client, validation_mixture_version_id
    )
    data_preparation = {
        "corpus": {
            "selection": {"mode": "all", "estimated_token_limit": None},
            "encoding": {
                "eos_policy": "append_if_available",
                "empty_document_policy": "skip",
                "token_dtype": "auto",
                "shard_token_limit": 1_048_576,
            },
        },
        "view": {
            "consumption_policy": "unique_only",
            "tail_policy": "drop",
            "padding_token_id": None,
            "packing_policy": "document_stream_v1",
            "sampling_policy": "weighted_deficit_v1",
            "seed": 7,
        },
        "execution": {
            "launcher_id": "builtin.local-process",
            "resource_request": {"gpus": 0, "cpus": 2},
        },
    }
    plan_response = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Managed data plan",
            "stages": [
                {
                    "name": "Stage 0",
                    "type": "pretrain",
                    "target_tokens": 1_000,
                    "effective_batch_tokens": 4_096,
                    "warmup_ratio": 0.015,
                    "sequence_length": 16,
                    "data_ref_id": mixture_version_id,
                    "data_preparation": data_preparation,
                }
            ],
            "tokenizer_version_id": tokenizer["id"],
            "validation_mixture_version_id": validation_mixture_version_id,
            "engine_plugin_id": "builtin.command.training",
            "engine_config": {"executable": sys.executable, "args": ["-c", "pass"]},
            "resource_request": {"gpus": 0},
        },
    )
    assert plan_response.status_code == 201
    plan = plan_response.json()

    first = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    repeated = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    assert first.status_code == 200
    assert first.json()["status"] == "preparing_corpora"
    assert repeated.json()["corpora"][0]["id"] == first.json()["corpora"][0]["id"]
    corpus_id = first.json()["corpora"][0]["id"]
    with SessionLocal() as session:
        corpus = session.get(TokenizedCorpusVersion, corpus_id)
        assert corpus is not None
        corpus_task = session.get(Task, corpus.task_id)
        assert corpus_task is not None
        assert corpus_task.payload["trigger_context"] == {
            "source": "training_plan_prepare_data",
            "action": "准备训练数据与运行提示",
            "training_plan_id": plan["id"],
            "training_plan_name": "Managed data plan",
            "plan_revision_id": plan["current_revision"]["id"],
            "plan_revision": 1,
            "selected_stages": [
                {"index": 0, "name": "Stage 0", "type": "pretrain"}
            ],
        }
        corpus_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="tokenized_corpus",
            payload=b"managed-corpus",
            filename="manifest.json",
        )
        corpus.status = "ready"
        corpus.artifact_id = corpus_artifact.id
        corpus.manifest_hash = corpus_artifact.content_hash
        corpus.document_count = 100
        corpus.token_count = 8_192
        corpus.shard_count = 1
        session.commit()

    indexing = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    assert indexing.status_code == 200
    assert indexing.json()["status"] == "preparing_index"
    view_id = indexing.json()["data_index"]["id"]
    with SessionLocal() as session:
        view = session.get(TrainingDataViewVersion, view_id)
        assert view is not None
        view_task = session.get(Task, view.task_id)
        assert view_task is not None
        assert view_task.payload["trigger_context"] == corpus_task.payload[
            "trigger_context"
        ]
        assert view_task.payload["configuration"]["requested_tokens"] == 4_096
        assert view_task.payload["configuration"]["sequence_order_policy"] == (
            "block_shuffle_v1"
        )
        assert view_task.payload["configuration"]["shuffle_block_sequences"] == 16
        assert view_task.payload["configuration"]["validation_holdout"] == {
            "strategy": "weighted_random_tokenized_document_holdout_v1",
            "target_tokens": 16,
            "seed": 42,
            "validation_mixture_version_id": validation_mixture_version_id,
            "validation_manifest_hash": validation_mixture_version["manifest_hash"],
            "sources": [
                    {
                        "dataset_version_id": dataset["current_version"]["id"],
                        "weight_units": 1_000_000,
                    }
            ],
        }
        assert view_task.payload["preflight"]["requested_sequences"] == 256
        view_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="training_data_index",
            payload=b"managed-index",
            filename="manifest.json",
            metadata={
                "validation_holdout": {
                    "strategy": "weighted_random_tokenized_document_holdout_v1",
                    "validation_mixture_version_id": validation_mixture_version_id,
                    "identity_hash": "b" * 64,
                    "sequence_count": 1,
                    "token_count": 16,
                }
            },
        )
        view.status = "ready"
        view.artifact_id = view_artifact.id
        view.manifest_hash = view_artifact.content_hash
        # A capacity superset remains reusable; the run freezes and reads only
        # the batch-aligned prefix it needs.
        view.token_count = 8_176
        view.sequence_count = 511
        session.commit()

    ready = client.post(f"/api/training-plans/{plan['id']}/prepare-data", json={})
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["data_index"]["id"] == view_id
    assert ready.json()["capacity"]["required_sequence_end"] == 256
    assert ready.json()["capacity"]["available_sequences"] == 511
    preflight = client.post(f"/api/training-plans/{plan['id']}/preflight", json={})
    assert preflight.status_code == 200
    assert preflight.json()["status"] == "passed"
    assert preflight.json()["training_data_range"] == {
        "schema_version": 1,
        "sequence_length": 16,
        "sequence_start": 0,
        "sequence_count": 256,
        "sequence_end": 256,
        "planned_steps": 1,
        "planned_train_tokens": 4_096,
        "stage_ranges": [
            {
                "name": "Stage 0",
                "target_tokens": 1_000,
                "effective_batch_tokens": 4_096,
                "planned_steps": 1,
                "planned_train_tokens": 4_096,
                "sequence_start": 0,
                "sequence_count": 256,
                "sequence_end": 256,
            }
        ],
    }
    assert not any(
        issue["code"] == "training_data_preparation_required"
        for issue in preflight.json()["issues"]
    )
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201
    frozen_range = started.json()["task"]["payload"]["training_data_range"]
    schedule = started.json()["task"]["payload"]["learning_rate_schedules"][0]
    assert schedule == preflight.json()["learning_rate_schedules"][0]
    assert schedule["warmup_ratio"] == 0.015
    assert schedule["total_steps"] == 1
    assert schedule["resolved_warmup_steps"] == 1
    assert started.json()["config_snapshot"]["learning_rate_schedules"] == [schedule]
    assert frozen_range["sequence_start"] == 0
    assert frozen_range["sequence_end"] == 256
    binding = started.json()["task"]["payload"]["training_data_view_inputs"][0]
    assert binding["sequence_count"] == 511
    assert binding["training_data_range"] == frozen_range
    validation_binding = started.json()["task"]["payload"]["validation_data_input"]
    assert validation_binding == {
        "format": "forgelm-validation-token-holdout-v1",
        "usage": "training_validation",
        "target_tokens": 16,
        "seed": 42,
        "mixture_version_id": validation_mixture_version_id,
        "training_data_view_version_id": view_id,
        "training_data_view_identity_hash": binding["identity_hash"],
        "holdout_identity_hash": "b" * 64,
        "sequence_count": 1,
        "token_count": 16,
    }

    with SessionLocal() as session:
        checkpoint_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="checkpoint",
            payload=b"managed checkpoint",
            filename="checkpoint.tar.gz",
        )
        checkpoint = Checkpoint(
            project_id=project["id"],
            training_run_id=started.json()["id"],
            step=1,
            token_count=4_096,
            artifact_id=checkpoint_artifact.id,
            metrics={"sequence_offset": 100, "sample_offset": 10},
            status="available",
        )
        session.add(checkpoint)
        session.commit()
        checkpoint_id = checkpoint.id

    resumed = client.post(
        f"/api/training-plans/{plan['id']}/prepare-data",
        json={"resume_checkpoint_id": checkpoint_id},
    )
    assert resumed.status_code == 200
    assert resumed.json()["ready"] is True
    assert resumed.json()["data_index"]["id"] == view_id
    assert resumed.json()["effective_configuration"]["training_data_range"][
        "sequence_start"
    ] == 100
    assert resumed.json()["capacity"]["required_sequence_end"] == 356

    # A larger legacy View with the same configuration but a freshly shuffled
    # order cannot receive this checkpoint's numeric cursor. Preparation must
    # create a lineage-bound expansion from the source run's frozen View.
    with SessionLocal() as session:
        source_view = session.get(TrainingDataViewVersion, view_id)
        assert source_view is not None
        wrong_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="training_data_index",
            payload=b"same semantics but unrelated block order",
            filename="manifest.json",
        )
        wrong_view = TrainingDataViewVersion(
            project_id=project["id"],
            mixture_version_id=source_view.mixture_version_id,
            tokenizer_version_id=source_view.tokenizer_version_id,
            identity_hash="d" * 64,
            sequence_length=source_view.sequence_length,
            requested_tokens=16_000,
            token_count=16_000,
            sequence_count=1_000,
            status="ready",
            artifact_id=wrong_artifact.id,
            manifest_hash=wrong_artifact.content_hash,
            config_snapshot={
                **source_view.config_snapshot,
                "configuration": {
                    **source_view.config_snapshot["configuration"],
                    "requested_tokens": 16_000,
                },
            },
        )
        session.add(wrong_view)
        expansion_checkpoint = Checkpoint(
            project_id=project["id"],
            training_run_id=started.json()["id"],
            step=2,
            token_count=8_192,
            artifact_id=checkpoint_artifact.id,
            metrics={"sequence_offset": 400, "sample_offset": 20},
            status="available",
        )
        session.add(expansion_checkpoint)
        session.commit()
        wrong_view_id = wrong_view.id
        expansion_checkpoint_id = expansion_checkpoint.id
        source_view_identity = source_view.identity_hash

    expansion = client.post(
        f"/api/training-plans/{plan['id']}/prepare-data",
        json={"resume_checkpoint_id": expansion_checkpoint_id},
    )
    assert expansion.status_code == 200
    assert expansion.json()["status"] == "preparing_index"
    assert expansion.json()["data_index"]["id"] not in {view_id, wrong_view_id}
    with SessionLocal() as session:
        expansion_view = session.get(
            TrainingDataViewVersion, expansion.json()["data_index"]["id"]
        )
        assert expansion_view is not None
        expansion_task = session.get(Task, expansion_view.task_id)
        assert expansion_task is not None
        prefix = expansion_task.payload["prefix_training_data_view"]
        assert prefix["version_id"] == view_id
        assert prefix["sequence_count"] == 511
        lineage = expansion_task.payload["identity_snapshot"]["prefix_lineage"]
        assert lineage["base_view"]["identity_hash"] == source_view_identity
        assert lineage["ancestor_views"][-1]["sequence_count"] == 511

def test_queued_download_can_pause_and_resume_without_resetting_progress(
    client: TestClient,
) -> None:
    project = create_project(client, "Download-Control")
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project["id"],
            kind="data_import",
            title="导入数据：large-corpus",
            adapter_id="builtin.data-import",
            payload={"source_type": "huggingface", "source_uri": "org/large-corpus"},
        )
        task.progress = 0.42
        session.commit()
        task_id = task.id

    paused = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "pause", "reason": "pause before worker claim"},
    )
    assert paused.status_code == 200
    assert paused.json()["observed_state"] == "paused"
    assert paused.json()["desired_state"] == "paused"
    assert paused.json()["progress"] == pytest.approx(0.42)
    assert paused.json()["events"][-1]["event_type"] == "task.paused"

    resumed = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "resume", "reason": "continue with cached files"},
    )
    assert resumed.status_code == 200
    assert resumed.json()["observed_state"] == "queued"
    assert resumed.json()["desired_state"] == "running"
    assert resumed.json()["progress"] == pytest.approx(0.42)
    assert resumed.json()["events"][-1]["event_type"] == "task.resume_requested"
    cancelled = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "cancel", "reason": "test cleanup"},
    )
    assert cancelled.status_code == 200


def test_training_safe_pause_requires_explicit_protocol_and_keeps_immediate_stop(
    client: TestClient,
) -> None:
    project = create_project(client, "Training-Safe-Pause")
    with SessionLocal() as session:
        unsupported = create_task(
            session,
            project_id=project["id"],
            kind="train",
            title="legacy training command",
            adapter_id="builtin.command.training",
            payload={"command": {"executable": "python"}},
        )
        unsupported.observed_state = "running"
        supported = create_task(
            session,
            project_id=project["id"],
            kind="train",
            title="checkpointed training command",
            adapter_id="builtin.command.training",
            payload={
                "training_run_id": "stable-training-run",
                "command": {
                    "executable": "python",
                    "control": {
                        "safe_pause": {
                            "protocol": "forgelm-command-control-file-v1",
                        }
                    },
                }
            },
        )
        supported.observed_state = "running"
        session.commit()
        unsupported_id = unsupported.id
        supported_id = supported.id

    rejected = client.post(
        f"/api/tasks/{unsupported_id}/control",
        json={"action": "pause", "reason": "unsupported safe pause"},
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "safe_training_pause_not_supported"

    pausing = client.post(
        f"/api/tasks/{supported_id}/control",
        json={"action": "pause", "reason": "checkpoint before pause"},
    )
    assert pausing.status_code == 200
    assert pausing.json()["observed_state"] == "pausing"
    assert pausing.json()["desired_state"] == "paused"

    with SessionLocal() as session:
        paused_task = session.get(Task, supported_id)
        assert paused_task is not None
        paused_task.observed_state = "paused"
        paused_task.desired_state = "paused"
        session.commit()

    resumed = client.post(
        f"/api/tasks/{supported_id}/control",
        json={"action": "resume", "reason": "continue the same training run"},
    )
    assert resumed.status_code == 200
    assert resumed.json()["observed_state"] == "queued"
    assert resumed.json()["desired_state"] == "running"
    assert resumed.json()["payload"]["training_run_id"] == "stable-training-run"
    assert (
        resumed.json()["payload"]["command"]["control"]["safe_pause"]["protocol"]
        == "forgelm-command-control-file-v1"
    )

    with SessionLocal() as session:
        running_again = session.get(Task, supported_id)
        assert running_again is not None
        running_again.observed_state = "running"
        session.commit()

    pausing_again = client.post(
        f"/api/tasks/{supported_id}/control",
        json={"action": "pause", "reason": "pause safely after resuming"},
    )
    assert pausing_again.status_code == 200
    assert pausing_again.json()["observed_state"] == "pausing"
    assert pausing_again.json()["desired_state"] == "paused"

    stopping = client.post(
        f"/api/tasks/{supported_id}/control",
        json={"action": "cancel", "reason": "stop immediately instead"},
    )
    assert stopping.status_code == 200
    assert stopping.json()["observed_state"] == "stopping"
    assert stopping.json()["desired_state"] == "cancelled"


def test_task_event_windows_are_bounded_grouped_and_attempt_scoped(
    client: TestClient,
) -> None:
    project = create_project(client, "Task-Event-Window")
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project["id"],
            kind="training",
            title="bounded observability",
            adapter_id="builtin.command.training",
        )
        first_attempt = TaskAttempt(task_id=task.id, attempt_number=1, status="failed")
        second_attempt = TaskAttempt(task_id=task.id, attempt_number=2, status="running")
        session.add_all([first_attempt, second_attempt])
        session.flush()
        sequence = 2
        for index in range(305):
            session.add(TaskEvent(task_id=task.id, attempt_id=first_attempt.id, sequence=sequence, event_type="log", payload={"stream": "stdout", "message": f"log-{index:03d}"}))
            sequence += 1
        for index in range(5):
            session.add(TaskEvent(task_id=task.id, attempt_id=first_attempt.id, sequence=sequence, event_type="resource.sample", payload={"sample": index}))
            sequence += 1
        for index in range(8):
            session.add(TaskEvent(task_id=task.id, attempt_id=first_attempt.id, sequence=sequence, event_type="progress", payload={"step": index}))
            sequence += 1
        session.add(TaskEvent(task_id=task.id, attempt_id=second_attempt.id, sequence=sequence, event_type="log", payload={"stream": "stderr", "message": "second-attempt"}))
        session.commit()
        task_id = task.id
        first_attempt_id = first_attempt.id

    compact = client.get(f"/api/tasks/{task_id}?include_events=false")
    assert compact.status_code == 200
    assert "events" not in compact.json()
    legacy = client.get(f"/api/tasks/{task_id}").json()
    assert len(legacy["events"]) == 300
    assert legacy["events_has_more"] is True
    assert legacy["events"][-1]["payload"]["message"] == "second-attempt"

    logs = client.get(
        f"/api/tasks/{task_id}/event-window",
        params={"event_group": "logs", "attempt_id": first_attempt_id, "limit": 300},
    )
    assert logs.status_code == 200
    assert logs.json()["has_more"] is True
    assert len(logs.json()["items"]) == 300
    assert logs.json()["items"][0]["payload"]["message"] == "log-005"
    assert logs.json()["items"][-1]["payload"]["message"] == "log-304"
    assert [item["sequence"] for item in logs.json()["items"]] == sorted(item["sequence"] for item in logs.json()["items"])
    older = client.get(f"/api/tasks/{task_id}/event-window", params={
        "event_group": "logs", "attempt_id": first_attempt_id, "limit": 300,
        "before_sequence": logs.json()["next_before_sequence"],
    }).json()
    assert [item["payload"]["message"] for item in older["items"]] == [f"log-{i:03d}" for i in range(5)]
    assert older["has_more"] is False

    resources = client.get(
        f"/api/tasks/{task_id}/event-window",
        params={"event_group": "resources", "attempt_id": first_attempt_id, "limit": 20},
    )
    assert resources.status_code == 200
    assert [item["event_type"] for item in resources.json()["items"]] == ["resource.sample"] * 5

    timeline = client.get(
        f"/api/tasks/{task_id}/event-window",
        params={"event_group": "timeline", "attempt_id": first_attempt_id, "limit": 20},
    )
    assert timeline.status_code == 200
    assert [item["event_type"] for item in timeline.json()["items"]] == ["progress"] * 8


def test_hf_tokenizer_retry_releases_legacy_gpu_request(client: TestClient) -> None:
    project = create_project(client, "Tokenizer-Retry-CPU")
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project["id"],
            kind="tokenizer_train",
            title="训练 Tokenizer：legacy",
            adapter_id="builtin.hf-tokenizer",
            payload={
                "config": {"sample_tokens": 100_000_000},
                "resource_request": {"gpus": 1, "gpu_indexes": [0]},
            },
        )
        task.observed_state = "lost"
        session.commit()
        task_id = task.id

    retried = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "retry", "reason": "retry with streaming CPU adapter"},
    )

    assert retried.status_code == 200
    assert retried.json()["observed_state"] == "queued"
    assert retried.json()["payload"]["resource_request"]["gpus"] == 0
    assert "gpu_indexes" not in retried.json()["payload"]["resource_request"]


def test_hf_tokenizer_accepts_billion_scale_streaming_target(client: TestClient) -> None:
    project = create_project(client, "Tokenizer-Billion-Scale")
    dataset = create_dataset(client, project["id"])

    response = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": "Two Billion Tokenizer",
            "algorithm": "BPE",
            "vocab_size": 4096,
            "data_ref_type": "dataset_version",
            "data_ref_id": dataset["current_version"]["id"],
            "config": {"sample_tokens": 2_000_000_000},
            "start": True,
        },
    )

    assert response.status_code == 201
    assert response.json()["config"]["sample_tokens"] == 2_000_000_000
    with SessionLocal() as session:
        task = session.get(Task, response.json()["task_id"])
        assert task is not None
        assert task.payload["config"]["sample_tokens"] == 2_000_000_000


def test_disk_bpe_tokenizer_is_selectable_and_freezes_bounded_cpu_config(
    client: TestClient,
) -> None:
    project = create_project(client, "Tokenizer-Disk-BPE")
    dataset = create_dataset(client, project["id"])
    config = {
        "sample_tokens": 12_000_000_000,
        "normalization": "NFKC",
        "min_word_frequency": 2,
        "max_unique_words": 250_000,
        "max_candidate_characters": 4_000_000,
        "max_word_length": 128,
        "merge_occurrence_budget": 2_000_000,
    }

    response = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": "Disk BPE",
            "algorithm": "BPE",
            "vocab_size": 32_768,
            "data_ref_type": "dataset_version",
            "data_ref_id": dataset["current_version"]["id"],
            "adapter_id": "builtin.disk-bpe-tokenizer",
            "config": config,
            "resource_request": {"gpus": 1, "gpu_indexes": [0]},
            "start": True,
        },
    )

    assert response.status_code == 201
    with SessionLocal() as session:
        task = session.get(Task, response.json()["task_id"])
        assert task is not None
        assert task.adapter_id == "builtin.disk-bpe-tokenizer"
        assert {key: task.payload["config"][key] for key in config} == config
        assert task.payload["config"]["gate_policy"]["version"] == "tokenizer-quality-gate-v1"
        assert task.payload["resource_request"]["gpus"] == 0
        assert "gpu_indexes" not in task.payload["resource_request"]


def test_disk_bpe_tokenizer_rejects_non_bpe_and_unbounded_merge_config(
    client: TestClient,
) -> None:
    project = create_project(client, "Tokenizer-Disk-BPE-Validation")
    dataset = create_dataset(client, project["id"])
    base = {
        "name": "Invalid Disk BPE",
        "vocab_size": 4096,
        "data_ref_type": "dataset_version",
        "data_ref_id": dataset["current_version"]["id"],
        "adapter_id": "builtin.disk-bpe-tokenizer",
        "config": {"sample_tokens": 2_000_000_000},
        "start": True,
    }

    wrong_algorithm = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={**base, "algorithm": "Unigram"},
    )
    unsafe_bounds = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            **base,
            "algorithm": "BPE",
            "config": {
                "sample_tokens": 2_000_000_000,
                "max_unique_words": 500_000,
                "merge_occurrence_budget": 100_000,
            },
        },
    )

    assert wrong_algorithm.status_code == 422
    assert "supports BPE only" in wrong_algorithm.json()["detail"]["message"]
    assert unsafe_bounds.status_code == 422
    assert "merge_occurrence_budget" in unsafe_bounds.json()["detail"]["message"]


def test_download_pause_request_survives_worker_claim_race(client: TestClient) -> None:
    project = create_project(client, "Download-Pause-Race")
    adapter_id = f"test.download-control-{uuid4().hex[:8]}"
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"download-control-worker-{uuid4().hex[:8]}",
            "hostname": "download-host",
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": [adapter_id, "builtin.local-process"],
        },
    ).json()
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project["id"],
            kind="data_import",
            title="导入数据：claimed-corpus",
            adapter_id=adapter_id,
            payload={
                "source_type": "https",
                "source_uri": "https://example.test/corpus.jsonl",
                "resource_request": {
                    "gpus": 0,
                    "worker_selector": {"worker_id": worker["id"]},
                },
            },
        )
        task.progress = 0.42
        session.commit()
        task_id = task.id

    claimed = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claimed["task"]["id"] == task_id

    pause = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "pause", "reason": "pause after claim"},
    )
    assert pause.status_code == 200
    assert pause.json()["observed_state"] == "assigned"
    assert pause.json()["desired_state"] == "paused"

    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": claimed["attempt"]["id"]},
    )
    assert started.status_code == 200
    assert started.json()["observed_state"] == "pausing"
    assert started.json()["desired_state"] == "paused"

    progress = client.post(
        f"/api/agent/tasks/{task_id}/events",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": claimed["attempt"]["id"],
            "event_type": "progress",
            "payload": {"progress": 0.05, "phase": "starting_tool"},
        },
    )
    assert progress.status_code == 200
    assert client.get(f"/api/tasks/{task_id}").json()["progress"] == pytest.approx(0.42)

    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": claimed["attempt"]["id"],
            "status": "paused",
            "result": {"controlled_state": "paused"},
            "exit_code": 0,
        },
    )
    assert completed.status_code == 200
    assert completed.json()["observed_state"] == "paused"


def test_dataset_metadata_edit_rejects_unplanned_hf_source_change(client: TestClient) -> None:
    project = create_project(client, "Dataset-Edit")
    created = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"HF-Corpus-{uuid4().hex[:8]}",
            "source_type": "huggingface",
            "source_uri": "org/old-corpus",
            "source_config": {"revision": "main"},
            "initial_version": {"tier": "bronze"},
            "start_import": False,
        },
    )
    assert created.status_code == 201
    dataset = created.json()
    original_version = dataset["current_version"]

    metadata = client.patch(
        f"/api/datasets/{dataset['id']}",
        json={"name": f"Renamed-{uuid4().hex[:8]}", "language": "zh,en"},
    )
    assert metadata.status_code == 200
    assert metadata.json()["language"] == "zh,en"
    assert metadata.json()["current_version"]["id"] == original_version["id"]

    source = client.patch(
        f"/api/datasets/{dataset['id']}",
        json={
            "source_uri": "org/corpus",
            "source_config": {
                "revision": "main",
                "hf_endpoint": "https://hf-mirror.com",
            },
            "resource_request": {"gpus": 0},
        },
    )
    assert source.status_code == 422
    assert source.json()["detail"]["code"] == "hf_plan_required"

    deleted = client.delete(f"/api/datasets/{dataset['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/datasets/{dataset['id']}").status_code == 404


def test_hf_scan_plan_freezes_commit_selection_and_worker(
    client: TestClient, tmp_path: Path
) -> None:
    project_workspace = tmp_path / "hf-project"
    project_workspace.mkdir()
    project = create_project(client, "HF-Asset-Plan", str(project_workspace))
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"hf-worker-{uuid4().hex[:8]}",
            "hostname": "hf-host",
            "labels": {"location": "local"},
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.data-import", "builtin.local-process", "builtin.hf-asset"],
        },
    ).json()
    scan_response = client.post(
        f"/api/projects/{project['id']}/datasets/huggingface/scan",
        json={
            "repo_id": "org/corpus",
            "revision": "main",
            "config": "default",
            "split": "train",
            "resource_request": {"gpus": 0, "worker_selector": {"worker_id": worker["id"]}},
        },
    )
    assert scan_response.status_code == 202
    scan_task = scan_response.json()
    claimed_scan = client.post(
        "/api/agent/tasks/claim", headers=AGENT_HEADERS, json={"worker_id": worker["id"]}
    ).json()
    assert claimed_scan["task"]["id"] == scan_task["id"]
    scan_attempt = claimed_scan["attempt"]
    assert (
        client.post(
            f"/api/agent/tasks/{scan_task['id']}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker["id"], "attempt_id": scan_attempt["id"]},
        ).status_code
        == 200
    )
    commit_hash = "a" * 40
    assert (
        client.post(
            f"/api/agent/tasks/{scan_task['id']}/complete",
            headers=AGENT_HEADERS,
            json={
                "worker_id": worker["id"],
                "attempt_id": scan_attempt["id"],
                "status": "succeeded",
                "result": {
                    "repo_id": "org/corpus",
                    "requested_revision": "main",
                    "commit_hash": commit_hash,
                    "endpoint": "https://huggingface.co",
                    "files": [
                        {"path": "data/train-0000.parquet", "size_bytes": 120},
                        {"path": "README.md", "size_bytes": 20},
                    ],
                    "file_count": 2,
                    "total_size_bytes": 140,
                    "config": "default",
                    "split": "train",
                },
                "exit_code": 0,
            },
        ).status_code
        == 200
    )

    plan_response = client.post(
        f"/api/projects/{project['id']}/datasets/huggingface/plans",
        json={"scan_task_id": scan_task["id"], "selected_paths": ["data/train-0000.parquet"]},
    )
    assert plan_response.status_code == 202
    plan_task = plan_response.json()
    claimed_plan = client.post(
        "/api/agent/tasks/claim", headers=AGENT_HEADERS, json={"worker_id": worker["id"]}
    ).json()
    assert claimed_plan["task"]["id"] == plan_task["id"]
    assert claimed_plan["task"]["payload"]["project_workspace_uri"] == str(project_workspace)
    assert (
        claimed_plan["task"]["payload"]["resource_request"]["worker_selector"]["worker_id"]
        == worker["id"]
    )
    plan_attempt = claimed_plan["attempt"]
    client.post(
        f"/api/agent/tasks/{plan_task['id']}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": plan_attempt["id"]},
    )
    assert (
        client.post(
            f"/api/agent/tasks/{plan_task['id']}/complete",
            headers=AGENT_HEADERS,
            json={
                "worker_id": worker["id"],
                "attempt_id": plan_attempt["id"],
                "status": "succeeded",
                "result": {
                    "repo_id": "org/corpus",
                    "requested_revision": "main",
                    "commit_hash": commit_hash,
                    "endpoint": "https://huggingface.co",
                    "selected_paths": ["data/train-0000.parquet"],
                    "files": [
                        {
                            "path": "data/train-0000.parquet",
                            "size_bytes": 120,
                            "is_cached": False,
                            "will_download": True,
                        }
                    ],
                    "file_count": 1,
                    "selected_bytes": 120,
                    "cached_bytes": 20,
                    "download_bytes": 100,
                    "disk_free_bytes": 1_000,
                    "disk_space_sufficient": True,
                    "config": "default",
                    "split": "train",
                    "project_workspace_uri": str(project_workspace),
                },
                "exit_code": 0,
            },
        ).status_code
        == 200
    )

    missing_plan = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Missing-Plan-{uuid4().hex[:8]}",
            "source_type": "huggingface",
            "source_uri": "org/corpus",
            "source_config": {},
            "start_import": True,
        },
    )
    assert missing_plan.status_code == 422
    assert missing_plan.json()["detail"]["code"] == "hf_plan_required"

    created = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Planned-HF-{uuid4().hex[:8]}",
            "source_type": "huggingface",
            "source_uri": "org/corpus",
            "source_config": {"hf_plan_task_id": plan_task["id"]},
            "start_import": True,
        },
    )
    assert created.status_code == 201
    dataset = created.json()
    assert dataset["source_config"]["revision"] == commit_hash
    assert dataset["source_config"]["selected_paths"] == ["data/train-0000.parquet"]
    assert dataset["source_config"]["config"] == "default"
    assert dataset["source_config"]["project_workspace_uri"] == str(project_workspace)
    with SessionLocal() as session:
        import_task = session.get(Task, dataset["current_version"]["task_id"])
        assert import_task is not None
        selector = import_task.payload["resource_request"]["worker_selector"]
        assert selector["worker_id"] == worker["id"]
        assert "builtin.hf-asset" in selector["capabilities"]


def test_hf_scan_requires_a_project_workspace(client: TestClient) -> None:
    project = create_project(client, "HF-Workspace-Required")
    with SessionLocal() as session:
        stored = session.get(Project, project["id"])
        assert stored is not None
        stored.workspace_uri = ""
        session.commit()

    response = client.post(
        f"/api/projects/{project['id']}/datasets/huggingface/scan",
        json={"repo_id": "org/corpus", "revision": "main"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "project_workspace_required"


def test_dataset_delete_is_blocked_by_tokenizer_reference(client: TestClient) -> None:
    project = create_project(client, "Dataset-Delete-Reference")
    dataset = create_dataset(client, project["id"])
    create_locked_tokenizer(client, project["id"], dataset["current_version"]["id"])

    response = client.delete(f"/api/datasets/{dataset['id']}")

    assert response.status_code == 409
    assert "Tokenizer" in response.json()["detail"]["message"]


def test_playground_freezes_a_real_transformers_inference_and_persists_result(
    client: TestClient,
) -> None:
    project = create_project(client, "Real-Inference")
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Tiny Chat Model",
            version=1,
            kind="sft",
            metadata_json={"inference": {"pretrained": "hf-internal-testing/tiny-random-gpt2"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "第二问",
            "system_prompt": "只回答事实",
            "adapter_id": "builtin.transformers-inference",
            "resource_request": {"gpus": 0},
            "config": {"device": "cpu", "dtype": "float32", "max_tokens": 16},
        },
    )
    assert generated.status_code == 202
    assert generated.json()["record"]["rendered_prompt"] == ""
    task = generated.json()["task"]
    assert task["adapter_id"] == "builtin.transformers-inference"
    assert task["payload"]["model"]["runtime_source"] == "hf-internal-testing/tiny-random-gpt2"
    assert task["payload"]["config"]["prompt_mode"] == "auto"
    assert task["payload"]["messages"] == [
        {"role": "system", "content": "只回答事实"},
        {"role": "user", "content": "第二问"},
    ]

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"inference-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": ["builtin.transformers-inference", "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claim["task"]["id"] == task["id"]
    attempt_id = claim["attempt"]["id"]
    assert (
        client.post(
            f"/api/agent/tasks/{task['id']}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker["id"], "attempt_id": attempt_id},
        ).status_code
        == 200
    )
    streaming = client.post(f"/api/agent/tasks/{task['id']}/events", headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id, "worker_sequence": 1,
              "event_type": "inference.snapshot", "payload": {"phase": "generating", "output": "部分正文"}})
    assert streaming.status_code == 200
    partial = client.get(f"/api/projects/{project['id']}/playground/sessions").json()[0]["records"][0]
    assert partial["output"] == ""
    assert partial["performance"]["stream"]["output"] == "部分正文"
    busy = client.post(f"/api/projects/{project['id']}/playground/generate", json={
        "session_id": generated.json()["session"]["id"], "model_version_ids": [model_id], "prompt": "下一轮"})
    assert busy.status_code == 409
    completed = client.post(
        f"/api/agent/tasks/{task['id']}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "rendered_prompt": "<real-chat-template>",
                "output": "真实模型输出",
                "token_ids": [101, 102],
                "input_tokens": 7,
                "output_tokens": 2,
                "stop_reason": "eos",
                "performance": {"generation_ms": 20, "tokens_per_second": 100},
            },
        },
    )
    assert completed.status_code == 200
    session_view = client.get(f"/api/projects/{project['id']}/playground/sessions").json()[0]
    record = session_view["records"][0]
    assert record["rendered_prompt"] == "<real-chat-template>"
    assert record["output"] == "真实模型输出"
    assert record["token_ids"] == [101, 102]
    late = client.post(f"/api/agent/tasks/{task['id']}/events", headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id, "worker_sequence": 2,
              "event_type": "inference.snapshot", "payload": {"phase": "generating", "output": "过期正文"}})
    assert late.status_code == 409



def test_playground_freezes_an_explicit_worker_selection(client: TestClient) -> None:
    project = create_project(client, "Explicit-Playground-Worker")
    workers = []
    for suffix in ("one", "two"):
        response = client.post(
            "/api/agent/register",
            headers=AGENT_HEADERS,
            json={
                "name": f"playground-{suffix}-{uuid4().hex[:8]}",
                "hostname": suffix,
                "resources": {"cpu_count": 2, "gpus": []},
                "capabilities": [
                    "builtin.transformers-inference",
                    "builtin.local-process",
                ],
            },
        )
        assert response.status_code == 200
        workers.append(response.json())
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Pinned Tiny Chat",
            version=1,
            kind="sft",
            metadata_json={"inference": {"pretrained": "org/pinned-tiny-chat"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "hello",
            "adapter_id": "builtin.transformers-inference",
            "config": {"device": "cpu", "dtype": "float32"},
            "resource_request": {
                "gpus": 0,
                "worker_selector": {"worker_id": workers[1]["id"]},
            },
        },
    )

    assert generated.status_code == 202
    task = generated.json()["task"]
    assert task["payload"]["resource_request"]["worker_selector"] == {
        "worker_id": workers[1]["id"]
    }
    assert (
        client.post(
            "/api/agent/tasks/claim",
            headers=AGENT_HEADERS,
            json={"worker_id": workers[0]["id"]},
        ).json()
        is None
    )
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": workers[1]["id"]},
    ).json()
    assert claim["task"]["id"] == task["id"]


def test_playground_routes_external_model_to_its_capable_worker(client: TestClient) -> None:
    project = create_project(client, "External-Inference")
    artifact_worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"external-inference-{uuid4().hex[:8]}",
            "hostname": "wsl",
            "labels": {"environment": "wsl"},
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": ["builtin.local-process"],
        },
    ).json()
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "model",
            "uri": "file:///data/models/external-chat",
            "content_hash": uuid4().hex * 2,
            "availability_state": "available",
            "worker_id": artifact_worker["id"],
        },
    ).json()
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="External Chat",
            version=1,
            kind="sft",
            artifact_id=artifact["id"],
        )
        session.add(model)
        session.commit()
        model_id = model.id

    blocked = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "hello",
            "adapter_id": "builtin.transformers-inference",
            "resource_request": {"gpus": 0},
        },
    )
    assert blocked.status_code == 422
    assert blocked.json()["detail"]["code"] == "artifact_worker_unavailable"

    registered = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": artifact_worker["name"],
            "hostname": "wsl",
            "labels": {"environment": "wsl"},
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": ["builtin.transformers-inference", "builtin.local-process"],
        },
    )
    assert registered.status_code == 200
    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "hello",
            "adapter_id": "builtin.transformers-inference",
            "resource_request": {"gpus": 0},
        },
    )
    assert generated.status_code == 202
    assert generated.json()["task"]["payload"]["resource_request"]["worker_selector"] == {
        "worker_id": artifact_worker["id"]
    }
    listed_model = next(
        item
        for item in client.get(f"/api/projects/{project['id']}/models").json()["models"]
        if item["id"] == model_id
    )
    assert listed_model["inference_source"] == {
        "type": "external",
        "artifact_id": artifact["id"],
        "worker_ids": [artifact_worker["id"]],
        "globally_accessible": False,
    }


def test_playground_rejects_unpacked_training_checkpoint(client: TestClient) -> None:
    project = create_project(client, "Checkpoint-Inference")
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": "file:///data/runs/checkpoint-100",
            "content_hash": uuid4().hex * 2,
            "availability_state": "available",
        },
    ).json()
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Training Checkpoint",
            version=1,
            kind="diagnostic",
            artifact_id=artifact["id"],
        )
        session.add(model)
        session.commit()
        model_id = model.id

    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "hello",
            "adapter_id": "builtin.transformers-inference",
        },
    )
    assert generated.status_code == 422
    assert generated.json()["detail"]["code"] == "model_artifact_not_inference_ready"
    listed_model = next(
        item
        for item in client.get(f"/api/projects/{project['id']}/models").json()["models"]
        if item["id"] == model_id
    )
    assert listed_model["inference_source"]["type"] == "unsupported"


def test_playground_automatically_routes_historical_microlm_models_to_native_inference(
    client: TestClient,
) -> None:
    project = create_project(client, "MicroLM-Native-Inference")
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"synthetic microlm model archive",
            filename="microlm.tar.gz",
            media_type="application/x-tar",
        )
        model = ModelVersion(
            project_id=project["id"],
            name="MicroLM-306M-v3",
            version=5,
            kind="base",
            artifact_id=artifact.id,
            metadata_json={
                "inference": {"format": "microlm-hybrid-safetensors"}
            },
        )
        session.add(model)
        session.commit()
        model_id = model.id

    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "你好",
            "resource_request": {"gpus": 0},
            "config": {"device": "auto", "dtype": "auto", "max_tokens": 16},
        },
    )

    assert generated.status_code == 202
    task = generated.json()["task"]
    assert task["adapter_id"] == "builtin.microlm-inference"
    assert task["payload"]["config"]["device"] == "cuda"
    assert task["payload"]["resource_request"] == {
        "gpus": 1,
        "worker_selector": {
            "capabilities": ["builtin.microlm-inference"],
        },
    }
    listed_model = next(
        item
        for item in client.get(f"/api/projects/{project['id']}/models").json()["models"]
        if item["id"] == model_id
    )
    assert listed_model["metadata_json"]["inference"] == {
        "format": "microlm-hybrid-safetensors"
    }
    assert listed_model["inference_source"]["type"] == "native"
    assert (
        listed_model["inference_source"]["adapter_id"]
        == "builtin.microlm-inference"
    )
    assert listed_model["inference_source"]["resource_request"]["gpus"] == 1

    incompatible = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "你好",
            "adapter_id": "builtin.transformers-inference",
        },
    )
    assert incompatible.status_code == 422
    assert incompatible.json()["detail"]["code"] == "incompatible_inference_adapter"


def test_playground_uses_registered_project_inference_profile(client: TestClient) -> None:
    project = create_project(client, "Project-Inference")
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"project-inference-{uuid4().hex[:8]}",
            "hostname": "wsl",
            "labels": {"project": "custom-model"},
            "resources": {"cpu_count": 2, "gpus": [{"index": 0}]},
            "capabilities": ["test.project-inference.v1", "builtin.local-process"],
        },
    ).json()
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "checkpoint",
            "uri": "file:///data/models/checkpoint-100",
            "content_hash": uuid4().hex * 2,
            "availability_state": "available",
            "worker_id": worker["id"],
        },
    ).json()
    command = {
        "executable": "/project/.venv/bin/python",
        "args": ["/platform/examples/project_inference.py"],
        "cwd": "/project",
        "env": {},
        "outputs": [],
    }
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Project Model",
            version=1,
            kind="sft",
            artifact_id=artifact["id"],
            metadata_json={
                "inference": {
                    "adapter_id": "test.project-inference.v1",
                    "display_name": "Project Native Inference",
                    "command": command,
                    "resource_request": {
                        "gpus": 1,
                        "worker_selector": {
                            "labels": {"project": "custom-model"},
                            "capabilities": ["test.project-inference.v1"],
                        },
                    },
                }
            },
        )
        session.add(model)
        session.commit()
        model_id = model.id

    generated = client.post(
        f"/api/projects/{project['id']}/playground/generate",
        json={
            "model_version_ids": [model_id],
            "prompt": "hello",
            "resource_request": {"gpus": 0},
            "config": {"max_tokens": 24},
        },
    )
    assert generated.status_code == 202
    task = generated.json()["task"]
    assert task["adapter_id"] == "test.project-inference.v1"
    assert task["payload"]["command"] == command
    assert task["payload"]["resource_request"] == {
        "gpus": 1,
        "worker_selector": {
            "worker_id": worker["id"],
            "labels": {"project": "custom-model"},
            "capabilities": ["test.project-inference.v1"],
        },
    }
    listed_model = next(
        item
        for item in client.get(f"/api/projects/{project['id']}/models").json()["models"]
        if item["id"] == model_id
    )
    assert listed_model["inference_source"]["type"] == "command"
    assert listed_model["inference_source"]["adapter_id"] == "test.project-inference.v1"


def test_project_and_data_version_lifecycle(client: TestClient) -> None:
    project = create_project(client, "Data-Lifecycle")
    dataset = create_dataset(client, project["id"])
    assert dataset["current_version"]["version"] == 1
    assert dataset["current_version"]["immutable"] is True
    assert dataset["current_version"]["task_id"] is None

    version_response = client.post(
        f"/api/datasets/{dataset['id']}/versions",
        json={
            "tier": "gold",
            "document_count": 8,
            "token_estimate": 1_700,
            "automatic_quality_score": 92,
            "train_ready": True,
        },
    )
    assert version_response.status_code == 201
    assert version_response.json()["version"] == 2
    assert version_response.json()["train_ready"] is True


@pytest.mark.parametrize("use_peer", [False, True])
def test_dataset_reanalysis_routes_to_accessible_replica_or_wsl_peer(
    client: TestClient, use_peer: bool
) -> None:
    project = create_project(client, "Reanalysis-replicas")
    dataset = create_dataset(client, project["id"])

    def register(hostname: str, labels: dict) -> dict:
        response = client.post(
            "/api/agent/register",
            headers=AGENT_HEADERS,
            json={
                "name": f"reanalyze-replica-{uuid4().hex[:8]}",
                "hostname": hostname,
                "labels": labels,
                "resources": {"cpu_count": 2, "gpus": []},
                "capabilities": ["builtin.data-import", "builtin.local-process"],
            },
        )
        assert response.status_code == 200
        return response.json()

    remote = register(f"remote-{uuid4().hex}", {"location": "remote"})
    hostname = f"local-{uuid4().hex}"
    labels = {"location": "local", "environment": "wsl", "distro": "test-distro"}
    local = register(hostname, labels)
    selected = register(hostname, labels) if use_peer else local
    unrelated = register(f"unrelated-{uuid4().hex}", labels)
    content_hash = uuid4().hex * 2
    artifacts = []
    # The older remote location must not override an accessible local replica.
    for worker, uri in [
        (remote, f"file:///remote/corpus-{content_hash}"),
        (local, f"file:///mnt/f/corpus-{content_hash}"),
    ]:
        response = client.post(
            f"/api/projects/{project['id']}/artifacts/references",
            json={
                "kind": "dataset_snapshot",
                "uri": uri,
                "content_hash": content_hash,
                "availability_state": "available",
                "worker_id": worker["id"],
            },
        )
        assert response.status_code == 201
        artifacts.append(response.json())
    assert artifacts[0]["id"] == artifacts[1]["id"]
    artifact = artifacts[0]
    version_id = dataset["current_version"]["id"]
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        version.artifact_id = artifact["id"]
        version.source_snapshot = {
            "files": [{"path": "train.parquet", "size_bytes": 123, "sha256": "a" * 64}]
        }
        session.commit()

    def reanalyze(worker_id: str):
        return client.post(
            f"/api/datasets/{dataset['id']}/reanalyze",
            json={"resource_request": {"worker_selector": {"worker_id": worker_id}}},
        )

    rejected = reanalyze(unrelated["id"])
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "dataset_worker_location_mismatch"
    response = reanalyze(selected["id"])
    assert response.status_code == 202
    analysis = response.json()["reanalyze"]
    assert analysis["worker_id"] == selected["id"]
    assert analysis["dataset_version_id"] == version_id
    task = client.get(f"/api/tasks/{analysis['task_id']}").json()
    assert task["payload"]["resource_request"]["worker_selector"] == {
        "worker_id": selected["id"]
    }
    assert task["payload"]["analysis_only"] is True
    assert task["payload"]["reuse_artifact_id"] == artifact["id"]
    assert task["payload"]["source_uri"] == ""
    claim = client.post(
        "/api/agent/tasks/claim", headers=AGENT_HEADERS, json={"worker_id": selected["id"]}
    ).json()
    assert claim["task"]["id"] == task["id"]


@pytest.mark.parametrize("stored_source_type", ["local_wsl", "huggingface"])
def test_dataset_reanalysis_creates_an_analysis_revision_and_keeps_content_version(
    client: TestClient, stored_source_type: str
) -> None:
    project = create_project(client, f"Reanalysis-{stored_source_type}")
    dataset = create_dataset(client, project["id"])
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"reanalyze-worker-{uuid4().hex[:8]}",
            "hostname": "wsl",
            "labels": {"environment": "wsl"},
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": ["builtin.data-import", "builtin.local-process"],
        },
    ).json()
    artifact_uri = f"file:///data/existing-corpus-{stored_source_type}"
    artifact = client.post(
        f"/api/projects/{project['id']}/artifacts/references",
        json={
            "kind": "dataset_snapshot",
            "uri": artifact_uri,
            "content_hash": uuid4().hex * 2,
            "availability_state": "available",
            "worker_id": worker["id"],
        },
    ).json()
    old_version_id = dataset["current_version"]["id"]
    with SessionLocal() as session:
        stored_dataset = session.get(Dataset, dataset["id"])
        stored_version = session.get(DatasetVersion, old_version_id)
        stored_dataset.source_type = stored_source_type
        stored_dataset.source_uri = (
            "org/corpus" if stored_source_type == "huggingface" else artifact_uri
        )
        stored_version.artifact_id = artifact["id"]
        stored_version.source_snapshot = {
            "files": [{"path": "train.parquet", "size_bytes": 123, "sha256": "a" * 64}]
        }
        session.commit()

    response = client.post(f"/api/datasets/{dataset['id']}/reanalyze", json={})
    assert response.status_code == 202
    created = response.json()
    current_version = created["current_version"]
    assert current_version["version"] == 1
    assert current_version["id"] == old_version_id
    assert current_version["state"] == "ready"
    assert len(created["versions"]) == 1
    active_analysis = current_version["active_analysis"]
    assert active_analysis["revision"] == 1
    assert created["reanalyze"] == {
        "task_id": active_analysis["task_id"],
        "analysis_revision_id": active_analysis["id"],
        "analysis_revision": 1,
        "dataset_version_id": old_version_id,
        "analysis_only": True,
        "worker_id": worker["id"],
    }
    task = client.get(f"/api/tasks/{active_analysis['task_id']}").json()
    assert task["payload"]["source_type"] == "local"
    assert task["payload"]["source_uri"] == ""
    assert task["payload"]["reuse_artifact_id"] == artifact["id"]
    assert task["payload"]["known_files"] == []
    assert task["payload"]["expected_snapshot_files"] == [
        {"path": "train.parquet", "size_bytes": 123, "sha256": "a" * 64}
    ]
    assert task["payload"]["analysis_input"] == {
        "artifact_id": artifact["id"],
        "artifact_hash": artifact["content_hash"],
    }
    assert task["payload"]["resource_request"]["worker_selector"] == {"worker_id": worker["id"]}
    duplicate = client.post(f"/api/datasets/{dataset['id']}/reanalyze", json={})
    assert duplicate.status_code == 409
    source_update = client.patch(
        f"/api/datasets/{dataset['id']}",
        json={"source_uri": "org/replacement"},
    )
    assert source_update.status_code == 409
    assert source_update.json()["detail"]["code"] == "dataset_analysis_active"
    assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 409

    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claim["task"]["id"] == task["id"]
    attempt_id = claim["attempt"]["id"]
    assert (
        client.post(
            f"/api/agent/tasks/{task['id']}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker["id"], "attempt_id": attempt_id},
        ).status_code
        == 200
    )
    with SessionLocal() as session:
        record_manifest = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_record_manifest",
            payload=b'{"type":"header","format":"llm.dataset-records.v1"}\n',
            filename="reanalyzed-records.jsonl",
            media_type="application/x-ndjson",
        )
        record_manifest_id = record_manifest.id
        session.commit()
    completed = client.post(
        f"/api/agent/tasks/{task['id']}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "raw_size_bytes": 42,
                "document_count": 1,
                "token_estimate": 4,
                "automatic_quality_score": 100,
                "quality_report": {"analyzed_document_count": 1},
                "sample_preview": [{"sample_key": "train.jsonl:1", "raw_text": "真实"}],
                "source_snapshot": {"fetched": True, "analysis_only": True},
                "train_ready": True,
                "indexes": [
                    {
                        "ref": "record_manifest",
                        "kind": "record_manifest",
                        "format": "llm.dataset-records.v1",
                    }
                ],
                "output_artifact_ids": [artifact["id"], record_manifest_id],
                "output_artifacts": [
                    {
                        "ref": "record_manifest",
                        "artifact_id": record_manifest_id,
                        "kind": "dataset_record_manifest",
                        "storage_type": "managed",
                    }
                ],
            },
        },
    )
    assert completed.status_code == 200
    refreshed = client.get(f"/api/datasets/{dataset['id']}").json()
    assert refreshed["status"] == "ready"
    assert refreshed["current_version"]["id"] == old_version_id
    assert refreshed["current_version"]["version"] == 1
    assert refreshed["current_version"]["analysis_revision"] == 1
    assert refreshed["current_version"]["document_count"] == 1
    assert refreshed["current_version"]["sample_preview"][0]["raw_text"] == "真实"
    assert refreshed["current_version"]["artifact_id"] == artifact["id"]
    assert refreshed["current_version"]["record_manifest_artifact_id"] == record_manifest_id
    assert refreshed["current_version"]["latest_analysis"]["record_manifest_artifact_id"] == record_manifest_id
    assert refreshed["current_version"]["indexes"][0]["analysis_revision_id"] == refreshed["current_version"]["latest_analysis"]["id"]

    second = client.post(f"/api/datasets/{dataset['id']}/reanalyze", json={})
    assert second.status_code == 202
    second_analysis = second.json()["current_version"]["active_analysis"]
    assert second_analysis["revision"] == 2
    second_claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    second_attempt_id = second_claim["attempt"]["id"]
    assert client.post(
        f"/api/agent/tasks/{second_analysis['task_id']}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": second_attempt_id},
    ).status_code == 200
    failed = client.post(
        f"/api/agent/tasks/{second_analysis['task_id']}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": second_attempt_id,
            "status": "failed",
            "error": {
                "type": "SnapshotDriftError",
                "message": "snapshot_drift: source files changed",
            },
            "exit_code": 1,
        },
    )
    assert failed.status_code == 200
    after_failure = client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]
    assert after_failure["id"] == old_version_id
    assert after_failure["analysis_revision"] == 1
    assert after_failure["document_count"] == 1
    assert after_failure["record_manifest_artifact_id"] == record_manifest_id
    assert after_failure["active_analysis"] is None
    assert after_failure["analysis_revisions"][0]["revision"] == 2
    assert after_failure["analysis_revisions"][0]["status"] == "failed"
    assert after_failure["analysis_revisions"][0]["error"]["type"] == "SnapshotDriftError"


def test_real_dataset_preflight_task_result_and_annotations(
    client: TestClient, tmp_path: Path
) -> None:
    project = create_project(client, "Real-Data")
    source = tmp_path / "real.jsonl"
    source.write_text('{"text":"真实样本"}\n', encoding="utf-8")
    preflight = client.post(
        f"/api/projects/{project['id']}/datasets/preflight",
        json={"source_type": "local", "source_uri": str(source), "source_config": {}},
    )
    assert preflight.status_code == 200
    assert preflight.json()["accessible"] is True
    assert preflight.json()["total_size_bytes"] == source.stat().st_size

    created = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Real-Corpus-{uuid4().hex[:8]}",
            "source_type": "local",
            "source_uri": str(source),
            "source_config": {"copy_mode": "managed"},
            "license": "MIT",
            "language": "zh",
            "start_import": True,
            "adapter_id": "builtin.data-import",
            "resource_request": {"gpus": 0},
            "initial_version": {"document_count": 999, "token_estimate": 999},
        },
    )
    assert created.status_code == 201
    dataset = created.json()
    version = dataset["current_version"]
    assert version["document_count"] == 0
    assert version["token_estimate"] == 0
    assert version["state"] == "queued"
    assert dataset["source_config"] == {"copy_mode": "managed"}

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"data-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 2, "gpus": []},
            "capabilities": ["builtin.data-import", "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claim["task"]["id"] == version["task_id"]
    attempt_id = claim["attempt"]["id"]
    task_id = claim["task"]["id"]
    assert (
        client.post(
            f"/api/agent/tasks/{task_id}/start",
            headers=AGENT_HEADERS,
            json={"worker_id": worker["id"], "attempt_id": attempt_id},
        ).status_code
        == 200
    )
    artifact = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "dataset_snapshot",
            "metadata_json": "{}",
        },
        files={"file": ("dataset.tar.gz", b"real dataset", "application/gzip")},
    ).json()
    sample = {
        "sample_key": "real.jsonl:1",
        "source_file": "real.jsonl",
        "raw_text": "真实样本",
        "cleaned_text": "真实样本",
        "language": "zh",
        "length": 4,
        "flags": [],
    }
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "raw_size_bytes": source.stat().st_size,
                "document_count": 1,
                "token_estimate": 4,
                "automatic_quality_score": 100,
                "quality_report": {
                    "analyzed_document_count": 1,
                    "language_distribution": {"zh": 100},
                    "garbled_rate": 0,
                },
                "sample_preview": [sample],
                "source_snapshot": {
                    "fetched": True,
                    "files": [{"path": "real.jsonl", "sha256": "a" * 64}],
                },
                "train_ready": True,
                "output_artifact_ids": [artifact["id"]],
            },
        },
    )
    assert completed.status_code == 200
    refreshed = client.get(f"/api/datasets/{dataset['id']}").json()
    final_version = refreshed["current_version"]
    assert refreshed["status"] == "ready"
    assert final_version["document_count"] == 1
    assert final_version["quality_report"]["garbled_rate"] == 0
    assert final_version["sample_preview"] == [sample]
    assert final_version["artifact_id"] == artifact["id"]

    annotation = client.post(
        f"/api/dataset-versions/{version['id']}/annotations",
        json={
            "sample_key": sample["sample_key"],
            "decision": "accept",
            "human_score": 95,
            "tags": ["高价值"],
            "sample_snapshot": sample,
        },
    )
    assert annotation.status_code == 201
    annotations = client.get(f"/api/dataset-versions/{version['id']}/annotations").json()
    assert annotations[0]["decision"] == "accept"
    assert annotations[0]["human_score"] == 95
    assert (
        client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]["human_score"] == 95
    )


def test_worker_claim_event_and_artifact_completion(client: TestClient) -> None:
    project = create_project(client, "Worker-Contract")
    dataset = create_dataset(client, project["id"])
    tokenizer_response = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": "Contract-BPE",
            "algorithm": "BPE",
            "vocab_size": 8_192,
            "data_ref_type": "dataset_version",
            "data_ref_id": dataset["current_version"]["id"],
            "config": {
                "gate_policy": {
                    "version": "tokenizer-quality-gate-v1",
                    "rules": {
                        "round_trip": {
                            "enabled": True,
                            "operator": "gte",
                            "threshold": 0.999,
                        },
                        "unk_rate": {
                            "enabled": True,
                            "operator": "lte",
                            "threshold": 0.001,
                        },
                    },
                }
            },
            "adapter_id": "builtin.command.tokenizer",
            "resource_request": {"gpus": 0},
            "command": {"executable": sys.executable, "args": ["-c", "pass"]},
            "start": True,
        },
    )
    assert tokenizer_response.status_code == 201
    tokenizer = tokenizer_response.json()
    assert tokenizer["config"]["gate_policy"]["rules"]["round_trip"]["threshold"] == 0.999
    assert "zh_chars_per_token" in tokenizer["config"]["gate_policy"]["rules"]

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"incompatible-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.command.tokenizer"],
        },
    ).json()
    incompatible_claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    )
    assert incompatible_claim.status_code == 200
    assert incompatible_claim.json() is None

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"compatible-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.command.tokenizer", "builtin.local-process"],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    assert claim["task"]["id"] == tokenizer["task_id"]
    assert claim["lease"]["status"] == "active"
    assert claim["lease"]["allocation"]["gpu_indexes"] == []

    task_id = claim["task"]["id"]
    attempt_id = claim["attempt"]["id"]
    start_response = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id},
    )
    assert start_response.status_code == 200

    event_response = client.post(
        f"/api/agent/tasks/{task_id}/events",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "event_type": "progress",
            "payload": {"progress": 0.5, "phase": "tokenizer-training"},
            "worker_sequence": 1,
        },
    )
    assert event_response.status_code == 200
    repeated_event = client.post(
        f"/api/agent/tasks/{task_id}/events",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "event_type": "progress",
            "payload": {"progress": 0.5, "phase": "tokenizer-training"},
            "worker_sequence": 1,
        },
    )
    assert repeated_event.status_code == 200
    assert repeated_event.json()["sequence"] == event_response.json()["sequence"]
    conflicting_event = client.post(
        f"/api/agent/tasks/{task_id}/events",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "event_type": "progress",
            "payload": {"progress": 0.75, "phase": "tokenizer-training"},
            "worker_sequence": 1,
        },
    )
    assert conflicting_event.status_code == 409
    assert conflicting_event.json()["detail"]["code"] == "worker_sequence_conflict"

    artifact_response = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "tokenizer",
            "metadata_json": '{"contract_test": true}',
        },
        files={"file": ("tokenizer.json", b'{"version": 1}', "application/json")},
    )
    assert artifact_response.status_code == 200
    uploaded_artifact = artifact_response.json()
    metrics_response = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "runtime_metrics",
            "metadata_json": '{"schema_version": "attempt_resources.v1"}',
        },
        files={"file": ("runtime-metrics.jsonl.gz", b"compressed", "application/gzip")},
    )
    assert metrics_response.status_code == 200
    metrics_artifact = metrics_response.json()
    metrics_download = client.get(
        f"/api/projects/{project['id']}/artifacts/{metrics_artifact['id']}/payload"
    )
    assert metrics_download.status_code == 200
    assert metrics_download.content == b"compressed"

    completion_payload = {
        "worker_id": worker["id"],
        "attempt_id": attempt_id,
        "status": "succeeded",
        "result": {
            "gate_passed": True,
            "metrics": {"round_trip": 0.995, "unk_rate": 0},
            "output_artifact_ids": [uploaded_artifact["id"]],
        },
        "resource_summary": {
            "schema_version": "attempt_resources.v1",
            "sample_count": 2,
            "process": {"peak_rss_bytes": 1024},
        },
        "runtime_metrics_artifact_id": metrics_artifact["id"],
        "log_summary": {"line_count": 3, "stdout_lines": 2, "stderr_lines": 1},
    }
    complete_response = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json=completion_payload,
    )
    assert complete_response.status_code == 200
    repeated_completion = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json=completion_payload,
    )
    assert repeated_completion.status_code == 200
    refreshed = client.get(f"/api/projects/{project['id']}/tokenizers").json()
    completed = next(item for item in refreshed if item["id"] == tokenizer["id"])
    assert completed["status"] == "failed_gate"
    assert completed["artifact_id"] == uploaded_artifact["id"]
    task_detail = client.get(f"/api/tasks/{task_id}").json()
    assert task_detail["result"]["gate_passed"] is False
    round_trip_gate = next(
        item for item in task_detail["result"]["gate_results"] if item["metric"] == "round_trip"
    )
    assert round_trip_gate["actual"] == 0.995
    assert round_trip_gate["threshold"] == 0.999
    assert round_trip_gate["passed"] is False
    completed_attempt = task_detail["attempts"][0]
    assert completed_attempt["resource_summary"]["sample_count"] == 2
    assert completed_attempt["runtime_metrics_artifact_id"] == metrics_artifact["id"]
    assert completed_attempt["log_summary"]["stderr_lines"] == 1


def test_stopping_worker_can_upload_final_runtime_metrics(client: TestClient) -> None:
    project = create_project(client, "Stopping-Metrics")
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project["id"],
            kind="data_import",
            title="cancelled data import",
            adapter_id="builtin.data-import",
            payload={"source_type": "huggingface", "source_uri": "org/repo"},
        )
        session.commit()
        task_id = task.id

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"stopping-metrics-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": [
                "builtin.data-import",
                "builtin.local-process",
                "builtin.hf-asset",
            ],
        },
    ).json()
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"]},
    ).json()
    attempt_id = claim["attempt"]["id"]
    started = client.post(
        f"/api/agent/tasks/{task_id}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker["id"], "attempt_id": attempt_id},
    )
    assert started.status_code == 200

    cancelled = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "cancel", "reason": "test final telemetry"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["observed_state"] == "stopping"

    metrics_response = client.post(
        f"/api/agent/tasks/{task_id}/artifacts",
        headers=AGENT_HEADERS,
        data={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "kind": "runtime_metrics",
            "metadata_json": '{"schema_version": "attempt_resources.v1"}',
        },
        files={"file": ("runtime-metrics.jsonl.gz", b"compressed", "application/gzip")},
    )
    assert metrics_response.status_code == 200

    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker["id"],
            "attempt_id": attempt_id,
            "status": "cancelled",
            "result": {},
            "runtime_metrics_artifact_id": metrics_response.json()["id"],
        },
    )
    assert completed.status_code == 200
    assert completed.json()["observed_state"] == "cancelled"
    assert (
        completed.json()["attempts"][0]["runtime_metrics_artifact_id"]
        == metrics_response.json()["id"]
    )


def test_training_preflight_requires_a_real_command(client: TestClient) -> None:
    project = create_project(client, "Training-Preflight")
    dataset = create_dataset(client, project["id"])
    tokenizer = create_locked_tokenizer(client, project["id"], dataset["current_version"]["id"])
    mixture_response = client.post(
        f"/api/projects/{project['id']}/mixtures",
        json={
            "name": "Single-source recipe",
            "target_tokens": 1_000,
            "entries": [
                {
                    "dataset_version_id": dataset["current_version"]["id"],
                    "name": dataset["name"],
                    "weight": 1,
                    "available_tokens": dataset["current_version"]["token_estimate"],
                    "selected_tokens": 500,
                    "selection": {
                        "sample_ratio": 0.5,
                        "max_tokens": 500,
                        "min_length": 32,
                        "max_length": 4096,
                        "include_paths": ["train/**"],
                        "exclude_paths": ["**/test/**"],
                        "seed": 42,
                    },
                }
            ],
        },
    )
    assert mixture_response.status_code == 201
    mixture = mixture_response.json()
    completed_mixture = complete_mixture_resolution(client, mixture["current_version"]["id"])
    assert completed_mixture["status"] == "ready"
    assert completed_mixture["manifest_hash"]
    plan_response = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Integration Plan",
            "stages": [
                {
                    "name": "Stage 0",
                    "type": "pretrain",
                    "target_steps": 10,
                    "target_tokens": 1_000,
                    "checkpoint_interval": 5,
                    "data_ref_id": mixture["current_version"]["id"],
                },
                {
                    "name": "Stage 1",
                    "type": "pretrain",
                    "target_steps": 20,
                    "target_tokens": 2_000,
                    "checkpoint_interval": 10,
                    "data_ref_id": mixture["current_version"]["id"],
                },
            ],
            "tokenizer_version_id": tokenizer["id"],
            "engine_plugin_id": "builtin.command.training",
            "engine_config": {"executable": sys.executable, "args": ["-c", "pass"]},
            "resource_request": {"gpus": 0},
        },
    )
    assert plan_response.status_code == 201
    plan = plan_response.json()
    preflight = client.get(f"/api/training-plans/{plan['id']}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["status"] == "passed"
    started = client.post(f"/api/training-plans/{plan['id']}/runs", json={})
    assert started.status_code == 201
    assert started.json()["task"]["observed_state"] == "queued"
    assert started.json()["task"]["adapter_id"] == "builtin.command.training"
    data_input = started.json()["task"]["payload"]["data_inputs"][0]
    assert data_input["mixture_version_id"] == mixture["current_version"]["id"]
    assert data_input["format"] == "llm-mixture-manifest-v1"
    assert data_input["manifest_hash"]
    assert "entries" not in data_input

    skipped_without_checkpoint = client.post(
        f"/api/training-plans/{plan['id']}/preflight",
        json={"selected_stage_indexes": [1]},
    )
    assert skipped_without_checkpoint.status_code == 200
    assert skipped_without_checkpoint.json()["status"] == "blocked"
    assert any(
        issue["code"] == "resume_checkpoint_required"
        for issue in skipped_without_checkpoint.json()["issues"]
    )

    with SessionLocal() as session:
        checkpoint_artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="checkpoint",
            payload=b"checkpoint archive",
            filename="checkpoint.tar.gz",
        )
        checkpoint = Checkpoint(
            project_id=project["id"],
            training_run_id=started.json()["id"],
            step=10,
            token_count=1_000,
            artifact_id=checkpoint_artifact.id,
            metrics={"sample_offset": 12},
            status="available",
        )
        session.add(checkpoint)
        session.commit()
        checkpoint_id = checkpoint.id

    selected_preflight = client.post(
        f"/api/training-plans/{plan['id']}/preflight",
        json={"selected_stage_indexes": [1], "resume_checkpoint_id": checkpoint_id},
    )
    assert selected_preflight.status_code == 200
    assert selected_preflight.json()["status"] == "passed"
    assert selected_preflight.json()["estimates"]["total_tokens"] == 2_000
    selected_run = client.post(
        f"/api/training-plans/{plan['id']}/runs",
        json={"selected_stage_indexes": [1], "resume_checkpoint_id": checkpoint_id},
    )
    assert selected_run.status_code == 201
    assert [stage["name"] for stage in selected_run.json()["task"]["payload"]["stages"]] == [
        "Stage 1"
    ]
    assert selected_run.json()["task"]["payload"]["selected_stage_indexes"] == [1]
    assert selected_run.json()["task"]["payload"]["resume_checkpoint"][
        "checkpoint_id"
    ] == checkpoint_id
    assert selected_run.json()["config_snapshot"]["selected_stage_indexes"] == [1]


def test_standard_benchmark_catalog_suite_and_frozen_run(client: TestClient) -> None:
    project = create_project(client, "Benchmark-Catalog")
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Benchmark-Test-Model",
            version=1,
            kind="imported",
            status="candidate",
            parameter_count=42_000_000,
            metadata_json={"inference": {"pretrained": "local/test-model"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    catalog_response = client.get("/api/benchmarks/catalog")
    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    assert any(item["id"] == "lm-eval:mmlu" for item in catalog)
    assert any(item["id"] == "lm-eval:wikitext" for item in catalog)
    assert any(item["id"] == "lm-eval:lambada_openai" for item in catalog)
    assert all(item["catalog_revision"] for item in catalog)

    preset_response = client.get("/api/benchmarks/suite-presets")
    assert preset_response.status_code == 200
    base_preset = next(
        item
        for item in preset_response.json()
        if item["id"] == "pretrained-base-language-general-v1"
    )
    assert len(base_preset["benchmark_ids"]) == 7
    assert base_preset["overrides"] == {
        "lm-eval:hellaswag": {"num_fewshot": 0},
        "lm-eval:winogrande": {"num_fewshot": 0},
    }
    assert base_preset["recommended_inference_config"] == {
        "apply_chat_template": False,
        "sample_percentage": 10,
        "sample_seed": 42,
    }
    zh_base_preset = next(
        item
        for item in preset_response.json()
        if item["id"] == "pretrained-base-zh-language-education-v1"
    )
    assert len(zh_base_preset["benchmark_ids"]) == 15
    assert zh_base_preset["overrides"] == {}
    assert {
        "lm-eval:xstorycloze_zh",
        "lm-eval:xwinograd_zh",
        "lm-eval:belebele_zho_Hans",
        "lm-eval:cmmlu_elementary_chinese",
        "lm-eval:ceval-valid_high_school_chinese",
        "lm-eval:zhoblimp_classifier_noun_agreement",
    }.issubset(zh_base_preset["benchmark_ids"])
    assert zh_base_preset["recommended_inference_config"] == {
        "apply_chat_template": False,
        "sample_percentage": 25,
        "sample_seed": 42,
    }
    zh_diagnostic_preset = next(
        item
        for item in preset_response.json()
        if item["id"] == "pretrained-base-zh-diagnostic-v2"
    )
    assert len(zh_diagnostic_preset["benchmark_ids"]) == 12
    assert zh_diagnostic_preset["recommended_inference_config"] == {
        "apply_chat_template": False,
        "sample_percentage": 10,
        "sample_seed": 42,
    }
    assert {
        "lm-eval:forgelm_zh_chid_cloze",
        "lm-eval:forgelm_zh_cmrc2019_cloze",
        "lm-eval:forgelm_zh_c3_answer_text",
        "lm-eval:forgelm_zh_cmrc2018_generation",
    }.issubset(zh_diagnostic_preset["benchmark_ids"])
    catalog_by_id = {item["id"]: item for item in catalog}
    for benchmark_id in zh_base_preset["benchmark_ids"]:
        assert benchmark_id in catalog_by_id
        assert catalog_by_id[benchmark_id]["languages"][0].startswith("zh")

    suite_response = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={
            "name": "Standard smoke suite",
            "benchmark_ids": ["lm-eval:mmlu", "lm-eval:gsm8k"],
            "overrides": {"lm-eval:mmlu": {"num_fewshot": 1}},
            "release_gate": {"overall": 0.5},
        },
    )
    assert suite_response.status_code == 201
    suite = suite_response.json()
    assert suite["scoring_config"]["asset_type"] == "benchmark"
    assert len(suite["datasets"]) == 2
    first_dataset_ids = {item["id"] for item in suite["imported_datasets"]}
    assert all(item["benchmark_config"]["config_hash"] for item in suite["imported_datasets"])

    repeated = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={
            "name": "Standard smoke suite copy",
            "benchmark_ids": ["lm-eval:mmlu", "lm-eval:gsm8k"],
            "overrides": {"lm-eval:mmlu": {"num_fewshot": 1}},
        },
    )
    assert repeated.status_code == 201
    assert {item["id"] for item in repeated.json()["imported_datasets"]} == first_dataset_ids

    start_response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "models": {
                    model_id: {
                        "backend": "hf",
                        "model_args": "pretrained=local/test-model,dtype=float32",
                    }
                },
                "device": "cpu",
                "sample_percentage": 10,
                "sample_seed": 42,
            },
            "scoring_config": {"save_samples": True},
            "resource_request": {"gpus": 0},
        },
    )
    assert start_response.status_code == 201
    run = start_response.json()
    assert run["task"]["adapter_id"] == "builtin.lm-eval"
    assert run["task"]["payload"]["command"] == {}
    assert len(run["config_snapshot"]["benchmarks"]) == 2
    assert run["inference_config"]["batch_size"] == "auto"
    assert run["inference_config"]["sampling"] == {
        "schema_version": "forgelm.evaluation-sampling.v1",
        "method": "stable_hash_v1",
        "percentage": 10.0,
        "seed": 42,
    }
    assert run["compatibility_hash"]
    assert run["benchmark_signature"]["status"] == "pending"

    manual_batch_response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "models": {
                    model_id: {
                        "backend": "hf",
                        "model_args": "pretrained=local/test-model,dtype=float32",
                    }
                },
                "batch_size": 8,
                "device": "cuda",
                "sample_percentage": 10,
                "sample_seed": 42,
            },
            "scoring_config": {"save_samples": True},
            "resource_request": {"gpus": 0},
        },
    )
    assert manual_batch_response.status_code == 201
    assert manual_batch_response.json()["compatibility_hash"] == run["compatibility_hash"]

    another_seed_response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "device": "cpu",
                "sample_percentage": 10,
                "sample_seed": 43,
            },
            "scoring_config": {"save_samples": True},
            "resource_request": {"gpus": 0},
        },
    )
    assert another_seed_response.status_code == 201
    assert another_seed_response.json()["compatibility_hash"] != run["compatibility_hash"]

    legacy_limit_response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {"limit": 100},
            "resource_request": {"gpus": 0},
        },
    )
    assert legacy_limit_response.status_code == 422
    assert (
        legacy_limit_response.json()["detail"]["code"]
        == "legacy_evaluation_limit_not_supported"
    )

    invalid_batch_response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "models": {
                    model_id: {
                        "backend": "hf",
                        "model_args": "pretrained=local/test-model,dtype=float32",
                    }
                },
                "batch_size": 0,
            },
            "resource_request": {"gpus": 0},
        },
    )
    assert invalid_batch_response.status_code == 422
    assert invalid_batch_response.json()["detail"]["code"] == "invalid_evaluation_batch_size"
    for task_id in (
        run["task"]["id"],
        manual_batch_response.json()["task"]["id"],
        another_seed_response.json()["task"]["id"],
    ):
        cleanup = client.post(
            f"/api/tasks/{task_id}/control",
            json={"action": "cancel", "reason": "integration test cleanup"},
        )
        assert cleanup.status_code == 200, cleanup.text


def test_lm_eval_rejects_missing_model_runtime(client: TestClient) -> None:
    project = create_project(client, "Benchmark-Validation")
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Validation-Model",
            version=1,
            kind="imported",
            status="candidate",
        )
        session.add(model)
        session.commit()
        model_id = model.id

    suite = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={"name": "Validation suite", "benchmark_ids": ["lm-eval:hellaswag"]},
    ).json()
    response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "adapter_id": "builtin.lm-eval",
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "model_inference_profile_missing"


def test_lm_eval_supports_mixed_local_and_openai_compatible_model_targets(
    client: TestClient,
) -> None:
    project = create_project(client, "Benchmark-Mixed-Targets")
    with SessionLocal() as session:
        primary = ModelVersion(
            project_id=project["id"],
            name="Primary Local Model",
            version=1,
            kind="base",
            status="candidate",
            metadata_json={"inference": {"pretrained": "/models/primary"}},
        )
        session.add(primary)
        session.commit()
        primary_id = primary.id

    endpoint_response = client.post(
        f"/api/projects/{project['id']}/models",
        json={
            "name": "Ollama Baseline",
            "source_type": "openai_compatible",
            "endpoint_url": "http://localhost:11434/v1",
            "endpoint_model": "qwen3.5:0.8b",
            "endpoint_mode": "completions",
            "endpoint_tokenizer": "Qwen/Qwen3.5-0.8B",
        },
    )
    assert endpoint_response.status_code == 201, endpoint_response.text
    endpoint = endpoint_response.json()
    suite = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={"name": "Mixed target suite", "benchmark_ids": ["lm-eval:mmlu"]},
    ).json()

    response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [primary_id, endpoint["id"]],
            "primary_model_version_id": primary_id,
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "models": {
                    primary_id: {
                        "backend": "hf",
                        "model_args": "pretrained=/models/primary",
                    },
                    endpoint["id"]: {
                        "backend": "hf",
                        "model_args": "pretrained=untrusted/client-override",
                    },
                },
                "batch_size": 1,
                "device": "cpu",
            },
            "resource_request": {"gpus": 0},
        },
    )

    assert response.status_code == 201, response.text
    run = response.json()
    assert run["config_snapshot"]["primary_model_version_id"] == primary_id
    assert run["config_snapshot"]["baseline_model_version_ids"] == [endpoint["id"]]
    runtimes = run["task"]["payload"]["inference_config"]["models"]
    assert runtimes[primary_id]["backend"] == "hf"
    assert runtimes[endpoint["id"]]["backend"] == "local-completions"
    assert "untrusted/client-override" not in runtimes[endpoint["id"]]["model_args"]
    cleanup = client.post(
        f"/api/tasks/{run['task']['id']}/control",
        json={"action": "cancel", "reason": "integration test cleanup"},
    )
    assert cleanup.status_code == 200, cleanup.text


def test_lm_eval_derives_microlm_native_runtime_from_playground_profile(
    client: TestClient,
) -> None:
    project = create_project(client, "Benchmark-MicroLM-Native")
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="model",
            payload=b"synthetic microlm model archive",
            filename="microlm.tar.gz",
        )
        model = ModelVersion(
            project_id=project["id"],
            name="MicroLM",
            version=8,
            kind="base",
            status="candidate",
            artifact_id=artifact.id,
            metadata_json={"inference": {"format": "microlm-hybrid-safetensors"}},
        )
        session.add(model)
        session.commit()
        model_id = model.id

    suite = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={"name": "Native suite", "benchmark_ids": ["lm-eval:mmlu"]},
    ).json()
    response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite["id"]],
            "inference_config": {
                "device": "cuda",
                "sample_percentage": 10,
                "sample_seed": 42,
            },
            "resource_request": {"gpus": 0},
        },
    )

    assert response.status_code == 201, response.text
    task = response.json()["task"]
    runtime = task["payload"]["inference_config"]["models"][model_id]
    assert runtime["backend"] == "microlm-native"
    assert runtime["model_args"] == "pretrained=__FORGELM_MODEL_ARTIFACT__"
    assert runtime["inference_adapter_id"] == "builtin.microlm-inference"
    assert task["payload"]["evaluation_model_inputs"][0]["model"]["id"] == model_id
    assert task["payload"]["resource_request"]["gpus"] == 1
    assert "builtin.microlm-inference" in task["payload"]["resource_request"][
        "worker_selector"
    ]["capabilities"]
    cleanup = client.post(
        f"/api/tasks/{task['id']}/control",
        json={"action": "cancel", "reason": "integration test cleanup"},
    )
    assert cleanup.status_code == 200, cleanup.text


def test_lm_eval_rejects_chat_endpoint_for_loglikelihood_benchmark(
    client: TestClient,
) -> None:
    project = create_project(client, "Benchmark-Chat-Capability")
    endpoint = client.post(
        f"/api/projects/{project['id']}/models",
        json={
            "name": "Cloud Chat Baseline",
            "source_type": "openai_compatible",
            "endpoint_url": "https://models.example.test/v1",
            "endpoint_model": "vendor-chat",
            "endpoint_mode": "chat_completions",
        },
    ).json()
    suite = client.post(
        f"/api/projects/{project['id']}/benchmark-suites",
        json={"name": "Logprob suite", "benchmark_ids": ["lm-eval:hellaswag"]},
    ).json()

    response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [endpoint["id"]],
            "suite_version_ids": [suite["id"]],
            "resource_request": {"gpus": 0},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "endpoint_capability_mismatch"
    assert response.json()["detail"]["task_ids"] == ["hellaswag"]


def test_evaluation_rejects_an_empty_suite(client: TestClient) -> None:
    project = create_project(client, "Empty-Evaluation-Suite")
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Empty-Suite-Test-Model",
            version=1,
            kind="imported",
            status="candidate",
        )
        suite = EvalSuiteVersion(
            project_id=project["id"],
            name="Historical empty suite",
            version=1,
            datasets=[],
        )
        session.add_all([model, suite])
        session.commit()
        model_id = model.id
        suite_id = suite.id

    response = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite_id],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "empty_evaluation_suite",
        "message": "请先添加评测集或从 Benchmark 目录创建套件",
        "ids": [suite_id],
    }


def test_unknown_api_route_never_falls_back_to_spa_html(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "charset=utf-8" in response.headers["content-type"].lower()
    assert response.json()["detail"]["code"] == "api_route_not_found"


def test_json_responses_declare_utf8_charset(client: TestClient) -> None:
    response = client.get("/api")
    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "application/json; charset=utf-8"
    assert response.json()["name"] == "Local LLM Platform"


def test_local_release_creates_immutable_manifest_artifact(client: TestClient) -> None:
    project = create_project(client, "Release-Manifest")
    with SessionLocal() as session:
        model = ModelVersion(
            project_id=project["id"],
            name="Imported-Test-Model",
            version=1,
            kind="imported",
            status="candidate",
            parameter_count=42_000_000,
        )
        session.add(model)
        session.commit()
        model_id = model.id

    response = client.post(
        f"/api/projects/{project['id']}/releases",
        json={
            "name": "Contract Release",
            "version": "v-contract-1",
            "model_version_id": model_id,
            "target": "local",
            "model_card": "# Contract model card",
            "license_manifest": {"status": "reviewed"},
        },
    )
    assert response.status_code == 201
    release = response.json()
    assert release["status"] == "published"
    assert release["artifact_id"] is not None
    artifacts = client.get("/api/artifacts").json()
    manifest = next(item for item in artifacts if item["id"] == release["artifact_id"])
    assert manifest["kind"] == "release_manifest"
