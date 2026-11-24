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
            "name": f"Data-Processing-{uuid4().hex[:8]}",
            "target_model_name": "Processing-LM",
            "workspace_uri": tempfile.mkdtemp(prefix="data-processing-workspace-"),
            "project_type": "pretrain",
            "target_parameters": "42M",
        },
    )
    assert response.status_code == 201
    return response.json()


def _dataset(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/datasets",
        json={
            "name": f"Bronze-{uuid4().hex[:8]}",
            "source_type": "local",
            "license": "MIT",
            "language": "zh",
            "initial_version": {
                "tier": "bronze",
                "raw_size_bytes": 24,
                "document_count": 2,
                "token_estimate": 12,
                "train_ready": True,
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
            payload=b'{"text":"first"}\n{"text":"second"}\n',
            filename="bronze.jsonl",
            media_type="application/x-ndjson",
        )
        version = session.get(DatasetVersion, dataset["current_version"]["id"])
        assert version is not None
        version.artifact_id = artifact.id
        session.commit()
    return client.get(f"/api/datasets/{dataset['id']}").json()


def _recipe(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/data-processing-recipes",
        json={
            "name": "基础文本清洗",
            "description": "规范空白并过滤异常长度",
            "operators": [
                {"whitespace_normalization_mapper": {}},
                {"text_length_filter": {"min_len": 2, "max_len": 1000}},
            ],
            "execution_config": {"np": 2},
        },
    )
    assert response.status_code == 201
    return response.json()


def test_recipe_versions_and_processing_success_are_immutable(client: TestClient) -> None:
    project = _project(client)
    dataset = _dataset(client, project["id"])
    recipe = _recipe(client, project["id"])
    original_recipe_version = recipe["current_version"]

    revised = client.post(
        f"/api/data-processing-recipes/{recipe['id']}/versions",
        json={
            "operators": [{"text_length_filter": {"min_len": 5, "max_len": 500}}],
            "execution_config": {"np": 1},
        },
    )
    assert revised.status_code == 201
    assert revised.json()["current_version"]["version"] == 2
    assert revised.json()["versions"][1]["operators"] == original_recipe_version["operators"]

    created = client.post(
        f"/api/projects/{project['id']}/data-processing-runs",
        json={
            "recipe_version_id": original_recipe_version["id"],
            "input_dataset_version_id": dataset["current_version"]["id"],
            "output_tier": "silver",
            "resource_request": {"gpus": 0},
        },
    )
    assert created.status_code == 202
    run = created.json()
    assert run["status"] == "queued"
    assert run["output_version"]["tier"] == "silver"
    assert run["output_version"]["parent_version_id"] == dataset["current_version"]["id"]
    # A queued output never displaces the last usable version.
    assert client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]["id"] == dataset["current_version"]["id"]
    with SessionLocal() as session:
        task = session.get(Task, run["task_id"])
        assert task is not None
        assert task.kind == "data_process"
        assert task.adapter_id == "builtin.data-juicer"
        assert task.payload["resource_request"]["worker_selector"]["capabilities"] == [
            "builtin.data-juicer"
        ]
        assert task.payload["recipe"]["operators"] == original_recipe_version["operators"]

    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"data-juicer-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "labels": {"os": "linux"},
            "capabilities": ["builtin.data-juicer", "builtin.local-process"],
            "resources": {"cpu_count": 8, "gpus": []},
            "runtime": {},
        },
    )
    assert worker.status_code == 200
    worker_id = worker.json()["id"]
    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker_id},
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
        files={"file": ("silver.jsonl", b'{"text":"cleaned"}\n', "application/x-ndjson")},
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
    assert snapshot.status_code == 200
    assert manifest.status_code == 200
    completed = client.post(
        f"/api/agent/tasks/{task_id}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "status": "succeeded",
            "exit_code": 0,
            "result": {
                "engine_version": "1.5.3",
                "raw_size_bytes": 19,
                "document_count": 1,
                "token_estimate": 5,
                "automatic_quality_score": 99,
                "quality_report": {"analyzed_document_count": 1},
                "sample_preview": [{"sample_key": "silver:0", "raw_text": "cleaned"}],
                "source_snapshot": {"source_type": "data_juicer", "files": []},
                "train_ready": True,
                "processing_summary": {
                    "input_document_count": 2,
                    "output_document_count": 1,
                    "document_delta": -1,
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
                ],
            },
        },
    )
    assert completed.status_code == 200
    final_run = client.get(
        f"/api/projects/{project['id']}/data-processing-runs"
    ).json()[0]
    assert final_run["status"] == "succeeded"
    assert final_run["engine_version"] == "1.5.3"
    assert final_run["output_version"]["record_manifest_artifact_id"] == manifest.json()["id"]
    current = client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]
    assert current["id"] == final_run["output_version"]["id"]
    assert current["tier"] == "silver"
    assert current["document_count"] == 1


def test_processing_rejects_non_advancing_tier_and_unsafe_recipe_globals(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset = _dataset(client, project["id"])
    invalid_recipe = client.post(
        f"/api/projects/{project['id']}/data-processing-recipes",
        json={
            "name": "Unsafe",
            "operators": [{"text_length_filter": {}}],
            "execution_config": {"np": 1, "export_path": "C:/outside.jsonl"},
        },
    )
    assert invalid_recipe.status_code == 422
    recipe = _recipe(client, project["id"])
    non_advancing = client.post(
        f"/api/projects/{project['id']}/data-processing-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": dataset["current_version"]["id"],
            "output_tier": "silver",
        },
    )
    assert non_advancing.status_code == 202
    output_id = non_advancing.json()["output_version"]["id"]
    with SessionLocal() as session:
        source = session.get(DatasetVersion, dataset["current_version"]["id"])
        output = session.get(DatasetVersion, output_id)
        assert source is not None and output is not None
        output.state = "ready"
        output.train_ready = True
        output.artifact_id = source.artifact_id
        session.commit()
    same_tier = client.post(
        f"/api/projects/{project['id']}/data-processing-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": output_id,
            "output_tier": "silver",
        },
    )
    assert same_tier.status_code == 422
    assert same_tier.json()["detail"]["code"] == "invalid_processing_output_tier"

    other_project = _project(client)
    cross_project = client.post(
        f"/api/projects/{other_project['id']}/data-processing-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": dataset["current_version"]["id"],
            "output_tier": "silver",
        },
    )
    assert cross_project.status_code == 422
    assert cross_project.json()["detail"]["code"] == "invalid_processing_input"
    cleanup = client.post(
        f"/api/tasks/{non_advancing.json()['task_id']}/control",
        json={"action": "cancel", "reason": "test cleanup"},
    )
    assert cleanup.status_code == 200


def test_failed_processing_keeps_the_last_ready_dataset_version(client: TestClient) -> None:
    project = _project(client)
    dataset = _dataset(client, project["id"])
    recipe = _recipe(client, project["id"])
    source_version_id = dataset["current_version"]["id"]
    worker = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"failed-data-juicer-worker-{uuid4().hex[:8]}",
            "hostname": "localhost",
            "labels": {"os": "linux"},
            "capabilities": ["builtin.data-juicer", "builtin.local-process"],
            "resources": {"cpu_count": 8, "gpus": []},
            "runtime": {},
        },
    )
    assert worker.status_code == 200
    worker_id = worker.json()["id"]
    created = client.post(
        f"/api/projects/{project['id']}/data-processing-runs",
        json={
            "recipe_version_id": recipe["current_version"]["id"],
            "input_dataset_version_id": source_version_id,
            "output_tier": "silver",
            "resource_request": {"worker_selector": {"worker_id": worker_id}},
        },
    )
    assert created.status_code == 202
    run = created.json()

    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": worker_id},
    )
    assert claim.status_code == 200
    attempt = claim.json()["attempt"]
    started = client.post(
        f"/api/agent/tasks/{run['task_id']}/start",
        headers=AGENT_HEADERS,
        json={"worker_id": worker_id, "attempt_id": attempt["id"]},
    )
    assert started.status_code == 200
    failed = client.post(
        f"/api/agent/tasks/{run['task_id']}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": worker_id,
            "attempt_id": attempt["id"],
            "status": "failed",
            "exit_code": 1,
            "error": {"code": "data_juicer_failed", "message": "operator failed"},
        },
    )
    assert failed.status_code == 200

    final_run = client.get(
        f"/api/projects/{project['id']}/data-processing-runs"
    ).json()[0]
    assert final_run["status"] == "failed"
    assert final_run["failure_reason"] == "operator failed"
    assert final_run["output_version"]["state"] == "failed"
    current = client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]
    assert current["id"] == source_version_id


def test_bound_file_selection_global_processing_preserves_source_and_binds_output(client):
    from llm_platform.models import (
        Artifact,
        DataMixture,
        DataMixtureVersion,
        Dataset,
        ProjectDataBinding,
    )
    from llm_platform.services.global_data import ensure_global_data_project
    from sqlalchemy import select

    source_project, consumer = _project(client), _project(client)
    source = _dataset(client, source_project["id"])
    source_id = source["current_version"]["id"]
    with SessionLocal() as session:
        global_id = ensure_global_data_project(session).id
        version = session.get(DatasetVersion, source_id)
        version.source_snapshot = {"files": [
            {"path": "keep.parquet", "size_bytes": 20, "sha256": "a" * 64},
            {"path": "drop.parquet", "size_bytes": 30, "sha256": "b" * 64}]}
        source_artifact = session.get(Artifact, version.artifact_id)
        manifest = artifact_storage.put_bytes(session, project_id=global_id, kind="data_mixture_manifest",
                                             payload=b"frozen-selection", filename="selection.jsonl")
        mixture = DataMixture(project_id=global_id, name="selected-shards")
        session.add(mixture)
        session.flush()
        selected = DataMixtureVersion(mixture_id=mixture.id, version=1, status="ready", locked=True,
            manifest_artifact_id=manifest.id, manifest_hash=manifest.content_hash,
            entries=[{"dataset_version_id": source_id, "dataset_artifact_id": source_artifact.id,
                      "dataset_artifact_hash": source_artifact.content_hash, "weight": 1,
                      "selection": {"include_paths": ["keep.parquet"]}}])
        session.add(selected)
        session.flush()
        selected_id = selected.id
        session.commit()
    recipe = _recipe(client, global_id)
    request = {"recipe_version_id": recipe["current_version"]["id"], "input_dataset_version_id": source_id,
               "input_mixture_version_id": selected_id, "asset_scope": "global", "output_tier": "silver"}
    endpoint = f"/api/projects/{consumer['id']}/data-processing-runs"
    assert client.post(endpoint, json=request).status_code == 422
    for kind, identity in [("dataset_version", source_id), ("mixture_version", selected_id)]:
        assert client.post(f"/api/projects/{consumer['id']}/data-bindings", json={
            "asset_type": kind, "asset_version_id": identity, "purpose": "training"}).status_code == 201
    result = client.post(endpoint, json=request)
    assert result.status_code == 202, result.text
    run = result.json()
    with SessionLocal() as session:
        task = session.get(Task, run["task_id"])
        assert task.project_id == global_id
        assert task.payload["consumer_project_id"] == consumer["id"]
        assert task.payload["input_mixture_version_id"] == selected_id
        from llm_platform.services.storage_lifecycle import estimated_write_bytes
        assert estimated_write_bytes(session, task) == 120
        assert [x["path"] for x in task.payload["data_processing_input"]["file_selection"]["files"]] == ["keep.parquet"]
        output = session.get(DatasetVersion, run["output_dataset_version_id"])
        assert output.parent_version_id == source_id
        assert output.dataset_id != source["id"]
        assert session.get(Dataset, output.dataset_id).project_id == global_id
        assert session.get(Dataset, source["id"]).current_version_id == source_id
        assert session.scalar(select(ProjectDataBinding.id).where(
            ProjectDataBinding.project_id == consumer["id"], ProjectDataBinding.asset_version_id == output.id))
    # The normal instance entry must preserve global ownership and project node lineage.
    base = f"/api/projects/{consumer['id']}/training-instances"
    instance = client.post(base, json={"name": "selected processing", "nodes": [
        {"id": "source", "type": "data", "label": "Source", "config": {"asset_type": "dataset_version", "asset_version_id": source_id}},
        {"id": "selection", "type": "data", "label": "Selection", "config": {"asset_type": "mixture_version", "asset_version_id": selected_id}},
        {"id": "clean", "type": "clean", "label": "Clean registered selection",
         "config": {"execution_mode": "native", "native_operation": "data.processing", "parameters": request,
                    "input_bindings": {"input_dataset_version_id": {"node_id": "source", "output_path": "asset_version_id"},
                                       "input_mixture_version_id": {"node_id": "selection", "output_path": "asset_version_id"}}}}],
        "edges": [{"id": "source-clean", "source": "source", "target": "clean", "source_handle": "out", "target_handle": "in"},
                  {"id": "selection-clean", "source": "selection", "target": "clean", "source_handle": "out", "target_handle": "selection"}]})
    assert instance.status_code == 201, instance.text
    attempt = client.post(f"{base}/{instance.json()['id']}/attempts", json={})
    assert attempt.status_code == 201, attempt.text
    detail = client.get(f"{base}/{instance.json()['id']}/attempts/{attempt.json()['id']}").json()
    node = next(n for n in detail["node_runs"] if n["node_id"] == "clean")
    assert node["status"] in {"queued", "running"}, node
    assert {x["asset_type"] for x in node["input_snapshot"]["consumed_assets"]} == {"dataset_version", "mixture_version"}
    with SessionLocal() as session:
        task = session.get(Task, node["task_id"])
        assert task.project_id == global_id
        assert task.payload["pipeline_instance_id"] == instance.json()["id"]
        selected = session.get(DataMixtureVersion, selected_id)
        selected.entries = [{**selected.entries[0], "selection": {"include_paths": ["keep.parquet"], "sample_ratio": 0.5}}]
        session.commit()
    assert client.post(endpoint, json=request).status_code == 422
