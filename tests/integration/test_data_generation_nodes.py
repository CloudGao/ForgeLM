from uuid import uuid4

import pytest
from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.routers.data_generation import _resources
from llm_platform.services.artifacts import artifact_storage
from llm_platform.services.global_data import (
    ensure_global_data_project,
    project_can_access_artifact,
)
from llm_platform.services.tasks import finalize_task_domain, sync_task_domain_status
from sqlalchemy import select


def interleaved_fixture(client, tmp_path):
    pid, source, weight, _, _ = setup(client, tmp_path)
    with SessionLocal() as session:
        session.get(m.DatasetVersion, source).document_count = 7
        session.commit()
    common = {"name": "test", "model_weight_version_id": weight, "engine": "vllm",
              "prompt_version": "cpt-generation-v8", "document_mode": "whole", "styles": ["facts"],
              "proper_name_policy": "source_exact", "validation_retry_limit": 0}
    graph = {"name": "two-stage", "nodes": [
        {"id": "t", "type": "fact_translation", "label": "翻译", "config": {
            "execution_mode": "native", "native_operation": "data.fact_translation",
            "parameters": {**common, "input_dataset_version_id": source, "temperature": 0.1}}},
        {"id": "g", "type": "knowledge", "label": "生成", "config": {
            "execution_mode": "native", "native_operation": "data.knowledge", "parameters": common,
            "input_bindings": {"input_dataset_version_id": {"node_id": "t", "output_path": "dataset_version_id"}}}},
    ], "edges": [{"id": "edge", "source": "t", "target": "g"}]}
    response = client.post(f"/api/projects/{pid}/training-instances", json=graph)
    assert response.status_code == 201, response.text
    return pid, response.json()["id"], source


def test_interleaved_compiler_and_native_dag_first_task(client, tmp_path):
    pid, instance, source = interleaved_fixture(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    body = {"name": "batched", "batch_records": 3}
    preview = client.post(f"{base}/{instance}/interleaved-preview", json=body)
    assert preview.status_code == 200, preview.text
    graph = preview.json()["graph"]
    batches = [n for n in graph["nodes"] if n["type"] == "fact_translation"]
    assert [n["config"]["parameters"]["record_range"] for n in batches] == [
        {"logical_file": "data.jsonl", "start": a, "stop": b, "total": 7} for a, b in [(0, 3), (3, 6), (6, 7)]
    ]
    assert len(graph["nodes"]) == 8
    created = client.post(f"{base}/{instance}/interleaved", json=body)
    assert created.status_code == 201, created.text
    started = client.post(f"{base}/{created.json()['id']}/attempts", json={})
    assert started.status_code == 201, started.text
    with SessionLocal() as session:
        rows = session.scalars(select(m.ProjectPipelineNodeRun).where(m.ProjectPipelineNodeRun.pipeline_run_id == started.json()["id"])).all()
        active = [row for row in rows if row.task_id]
        assert len(active) == 1 and active[0].node_id == "translate-0000"
        task = session.get(m.Task, active[0].task_id)
        assert task.payload["input_dataset_version_id"] == source
        assert task.payload["generation"]["record_range"]["stop"] == 3
    assert client.post(f"{base}/{instance}/interleaved", json=body).status_code == 409


def test_interleaved_respects_existing_dag_size_limit(client, tmp_path):
    pid, instance, source = interleaved_fixture(client, tmp_path)
    with SessionLocal() as session:
        session.get(m.DatasetVersion, source).document_count = 6642359
        session.commit()
    url = f"/api/projects/{pid}/training-instances/{instance}/interleaved"
    for suffix in ["-preview", ""]:
        response = client.post(url + suffix, json={"name": "too-many", "batch_records": 50000})
        assert response.status_code == 422, response.text
        assert "99" in response.text
    response = client.post(url, json={"name": "full-scale", "batch_records": 100000})
    assert response.status_code == 201, response.text
    assert len(response.json()["nodes"]) == 136


def test_interleaved_continuation_requires_same_instance_paused_task(client, tmp_path):
    pid, instance, _ = interleaved_fixture(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    started = client.post(f"{base}/{instance}/attempts", json={})
    assert started.status_code == 201, started.text
    with SessionLocal() as session:
        row = session.scalar(select(m.ProjectPipelineNodeRun).where(m.ProjectPipelineNodeRun.pipeline_run_id == started.json()["id"], m.ProjectPipelineNodeRun.node_id == "t"))
        task_id = row.task_id
    body = {"name": "continuation", "batch_records": 3, "checkpoint_source_task_id": task_id}
    assert client.post(f"{base}/{instance}/interleaved", json=body).status_code == 409
    with SessionLocal() as session:
        task = session.get(m.Task, task_id)
        task.observed_state = task.desired_state = "paused"
        worker = m.WorkerNode(name="continuation-worker", hostname="test", labels={}, capabilities=[], resources={})
        session.add(worker)
        session.flush()
        attempt = m.TaskAttempt(task_id=task_id, attempt_number=1, status="paused", worker_id=worker.id)
        session.add(attempt)
        session.commit()
    response = client.post(f"{base}/{instance}/interleaved", json=body)
    assert response.status_code == 201, response.text
    assert response.json()["variables"]["generation_interleave"]["order"] == [0, 2, 1]
    started = client.post(f"{base}/{response.json()['id']}/attempts", json={})
    assert started.status_code == 201, started.text
    with SessionLocal() as session:
        node = session.scalar(select(m.ProjectPipelineNodeRun).where(m.ProjectPipelineNodeRun.pipeline_run_id == started.json()["id"], m.ProjectPipelineNodeRun.node_id == "translate-0000"))
        assert node.task_id, node.error
        task = session.get(m.Task, node.task_id)
        assert task.payload["generation_continuation"]["task"]["id"] == task_id
        assert task.payload["resource_request"]["worker_selector"]["worker_id"]


def test_empty_batches_advance_in_order_without_fake_train_ready_or_duplicate_tasks(client, tmp_path):
    from llm_platform.services.pipeline_dags import advance_pipeline_run, sync_pipeline_task_status

    pid, instance, _ = interleaved_fixture(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    created = client.post(f"{base}/{instance}/interleaved", json={"name": "empty-batches", "batch_records": 3})
    assert created.status_code == 201, created.text
    started = client.post(f"{base}/{created.json()['id']}/attempts", json={})
    assert started.status_code == 201, started.text
    run_id = started.json()["id"]
    for expected in ["translate-0000", "generate-0000", "translate-0001", "generate-0001", "translate-0002", "generate-0002"]:
        with SessionLocal() as session:
            nodes = session.scalars(select(m.ProjectPipelineNodeRun).where(m.ProjectPipelineNodeRun.pipeline_run_id == run_id)).all()
            active = [node for node in nodes if node.status == "queued"]
            assert len(active) == 1 and active[0].node_id == expected
            task = session.get(m.Task, active[0].task_id)
            attempt = m.TaskAttempt(task_id=task.id, attempt_number=1, status="succeeded")
            session.add(attempt)
            session.flush()
            outputs = []
            for ref, kind in [("dataset", "dataset_snapshot"), ("record_manifest", "dataset_record_manifest")]:
                artifact = artifact_storage.put_bytes(session, project_id=pid, kind=kind,
                    payload=f"empty-{task.id}-{ref}".encode(), filename=ref, created_by_task_attempt_id=attempt.id)
                outputs.append({"ref": ref, "kind": kind, "artifact_id": artifact.id, "storage_type": "managed"})
            result = {"train_ready": False, "document_count": 0, "token_estimate": 0,
                      "source_snapshot": {"generation_empty": True, "files": [{"path": "data.jsonl"}]}, "output_artifacts": outputs}
            finalize_task_domain(session, task, attempt, result)
            task.observed_state, task.result = "succeeded", result
            sync_pipeline_task_status(session, task)
            run = session.get(m.ProjectPipelineRun, run_id)
            advance_pipeline_run(session, run)
            advance_pipeline_run(session, run)
            session.commit()
    with SessionLocal() as session:
        run = session.get(m.ProjectPipelineRun, run_id)
        assert run.status == "succeeded", run.error
        nodes = session.scalars(select(m.ProjectPipelineNodeRun).where(m.ProjectPipelineNodeRun.pipeline_run_id == run_id)).all()
        assert len([node for node in nodes if node.task_id]) == 6
        assert all(node.output_snapshot["empty"] for node in nodes if node.node_type == "mixture")
        versions = session.scalars(select(m.DatasetVersion).where(m.DatasetVersion.task_id.in_([node.task_id for node in nodes if node.task_id]))).all()
        assert len(versions) == 6 and all(v.state == "ready" and not v.train_ready for v in versions)


def setup(client, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    response = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "test", "workspace_uri": str(tmp_path)},
    )
    assert response.status_code == 201, response.text
    pid = response.json()["id"]
    with SessionLocal() as session:
        dataset = m.Dataset(project_id=pid, name="input", source_type="local", license="CC-BY-SA")
        session.add(dataset)
        session.flush()
        artifact = artifact_storage.put_bytes(
            session,
            project_id=pid,
            kind="dataset_snapshot",
            payload=b'{"text":"Physics"}\n',
            filename="data.jsonl",
            media_type="application/x-ndjson",
        )
        source = m.DatasetVersion(
            dataset_id=dataset.id,
            version=1,
            state="ready",
            train_ready=True,
            artifact_id=artifact.id,
            source_snapshot={"files": [{"path": "data.jsonl"}]},
        )
        session.add(source)
        session.flush()
        dataset.current_version_id = source.id
        global_project = ensure_global_data_project(session)
        weights = artifact_storage.put_bytes(
            session,
            project_id=global_project.id,
            kind="model_weights",
            payload=b"weight-test",
            filename="weights",
            media_type="application/octet-stream",
        )
        asset = m.ModelWeightAsset(name="HY", repository_id=uuid4().hex)
        session.add(asset)
        session.flush()
        weight = m.ModelWeightVersion(
            asset_id=asset.id,
            version=1,
            identity_hash=uuid4().hex,
            resolved_commit="abc",
            state="ready",
            artifact_id=weights.id,
        )
        session.add(weight)
        session.commit()
        return pid, source.id, weight.id, artifact.content_hash, weights.id


@pytest.mark.parametrize("version", ["cpt-generation-v5", "cpt-generation-v6"])
def test_fact_styles_create_separate_versions_with_shared_frozen_input(client, tmp_path, version):
    pid, source_id, weight_id, digest, _ = setup(client, tmp_path)
    outputs = []
    for style in ("retrospective", "thematic"):
        response = client.post(f"/api/projects/{pid}/knowledge-generation-runs", json={
            "name": "facts-" + style, "input_dataset_version_id": source_id,
            "model_weight_version_id": weight_id, "prompt_version": version,
            "styles": [style], "document_mode": "whole",
            "engine": "vllm", "fact_input_selection": "chinese_ready", "validation_retry_limit": 0,
        })
        assert response.status_code == 202, response.text
        result = response.json()
        with SessionLocal() as session:
            task = session.get(m.Task, result["task_id"])
            assert task.payload["generation"]["styles"] == [style]
            assert task.payload["generation"]["prompt_version"] == version
            assert task.payload["generation"]["fact_input_selection"] == "chinese_ready"
            assert task.payload["generation"]["validation_retry_limit"] == 0
            assert task.payload["generation_input"]["artifact_hash"] == digest
            output = session.get(m.DatasetVersion, result["output_dataset_version_id"])
            assert output.parent_version_id == source_id
            result["dataset_id"] = output.dataset_id
        outputs.append(result)
    assert outputs[0]["dataset_id"] != outputs[1]["dataset_id"]
    assert outputs[0]["output_dataset_version_id"] != outputs[1]["output_dataset_version_id"]


@pytest.mark.parametrize(
    "kind,operation", [("translation", "data.translation"), ("knowledge", "data.knowledge"),
                       ("fact_translation", "data.fact_translation")]
)
def test_independent_dag_node_freezes_inputs_and_registers_output(
    client, tmp_path, kind, operation
):
    pid, source_id, weight_id, digest, weight_artifact_id = setup(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    body = {
        "name": "generate",
        "nodes": [
            {
                "id": "generate",
                "type": kind,
                "label": "generate",
                "config": {
                    "native_operation": operation,
                    "execution_mode": "native",
                    "parameters": {
                        "name": "output",
                        "input_dataset_version_id": source_id,
                        "model_weight_version_id": weight_id,
                        "document_mode": "whole",
                        "max_source_characters": 3000,
                        "prompt_version": "cpt-generation-v8" if kind == "fact_translation" else "cpt-generation-v2",
                    },
                },
            }
        ],
        "edges": [],
    }
    response = client.post(base, json=body)
    assert response.status_code == 201, response.text
    attempt_response = client.post(f"{base}/{response.json()['id']}/attempts", json={})
    assert attempt_response.status_code == 201, attempt_response.text
    with SessionLocal() as session:
        node = session.scalar(
            select(m.ProjectPipelineNodeRun).where(
                m.ProjectPipelineNodeRun.pipeline_run_id == attempt_response.json()["id"]
            )
        )
        assert node.task_id, node.error
        task = session.get(m.Task, node.task_id)
        assert task.kind == "data_generate"
        assert task.payload["generation_input"]["artifact_hash"] == digest
        assert task.payload["generation"]["enable_thinking"] is False
        assert task.payload["generation"]["batch_size"] == "auto"
        assert task.payload["generation"]["document_mode"] == "whole"
        assert task.payload["generation"]["max_source_characters"] == 3000
        assert task.payload["generation"]["prompt_version"] == ("cpt-generation-v8" if kind == "fact_translation" else "cpt-generation-v2")
        if kind == "fact_translation":
            assert task.payload["generation"]["translation_protocol"] == "fact-fields-zh-v1"
            assert task.payload["generation"]["validation_retry_limit"] == 0
            assert task.payload["generation"]["mode"] == "fact_translation"
        assert task.payload["resource_request"]["gpus"] == 1
        assert project_can_access_artifact(session, pid, weight_artifact_id)
        assert {r["asset_type"] for r in node.input_snapshot["consumed_assets"]} == {
            "dataset_version",
            "model_weight_version",
        }
        output = session.get(m.DatasetVersion, node.output_snapshot["dataset_version_id"])
        assert output.state == "queued" and not output.train_ready
        assert output.parent_version_id == source_id
        assert "replay_payload" not in output.materialization_spec
        task.observed_state = "failed"
        sync_task_domain_status(session, task)
        assert output.state == "failed" and not output.train_ready
        assert session.get(m.DatasetVersion, source_id).state == "ready"


def test_reject_thinking_and_unbound_data(client, tmp_path):
    pid, source, weight, _, _ = setup(client, tmp_path)
    url = f"/api/projects/{pid}/data-translation-runs"
    body = {"name": "output", "input_dataset_version_id": source, "model_weight_version_id": weight}
    assert client.post(url, json={**body, "enable_thinking": True}).status_code == 422
    foreign, _, _, _, _ = setup(client, tmp_path / "foreign")
    assert (
        client.post(f"/api/projects/{foreign}/data-translation-runs", json=body).status_code == 422
    )


def test_resource_intersection(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        artifact_storage,
        "inference_access",
        lambda session, artifact, pid: {
            "type": "external",
            "globally_accessible": False,
            "worker_ids": artifact.workers,
        },
    )
    from types import SimpleNamespace

    data = SimpleNamespace(kind="dataset_snapshot", workers=["a", "b"])
    other = SimpleNamespace(kind="dataset_snapshot", workers=["b", "c"])
    with SessionLocal() as session:
        request = _resources(session, "project", [data, other], {}, "vllm")
        assert request["worker_selector"]["worker_ids"] == ["b"]
        with pytest.raises(HTTPException):
            _resources(
                session, "project", [data, other], {"worker_selector": {"worker_id": "a"}}, "vllm"
            )


def test_success_and_rerun_preserve_previous_ready_output(client, tmp_path):
    pid, source, weight, _, _ = setup(client, tmp_path)
    url = f"/api/projects/{pid}/data-translation-runs"
    body = {"name": "output", "input_dataset_version_id": source, "model_weight_version_id": weight}
    created = client.post(url, json=body)
    assert created.status_code == 202, created.text
    with SessionLocal() as session:
        task = session.get(m.Task, created.json()["task_id"])
        attempt = m.TaskAttempt(task_id=task.id, attempt_number=1, status="running")
        session.add(attempt)
        session.flush()
        outputs = []
        for ref, kind in [
            ("dataset", "dataset_snapshot"),
            ("record_manifest", "dataset_record_manifest"),
        ]:
            artifact = artifact_storage.put_bytes(
                session,
                project_id=pid,
                kind=kind,
                payload=f"{ref}-{task.id}".encode(),
                filename=f"{ref}.jsonl",
                created_by_task_attempt_id=attempt.id,
            )
            outputs.append(
                {"ref": ref, "kind": kind, "artifact_id": artifact.id, "storage_type": "managed"}
            )
        finalize_task_domain(
            session,
            task,
            attempt,
            {
                "train_ready": True,
                "document_count": 2,
                "token_estimate": 20,
                "source_snapshot": {"generation": task.payload["generation"]},
                "output_artifacts": outputs,
            },
        )
        task.observed_state = "succeeded"
        session.commit()
        version = session.get(m.DatasetVersion, created.json()["output_dataset_version_id"])
        assert version.state == "ready" and version.train_ready
        assert version.record_manifest_artifact_id and version.artifact_id
    second = client.post(url, json=body)
    assert second.status_code == 202, second.text
    with SessionLocal() as session:
        task = session.get(m.Task, second.json()["task_id"])
        version = session.get(m.DatasetVersion, second.json()["output_dataset_version_id"])
        assert version.version == 2
        dataset = session.get(m.Dataset, version.dataset_id)
        assert dataset.current_version_id == created.json()["output_dataset_version_id"]
        task.observed_state = "failed"
        sync_task_domain_status(session, task)
        assert dataset.status == "ready"
        assert version.state == "failed"


def test_success_requires_manifest(client, tmp_path):
    pid, source, weight, _, _ = setup(client, tmp_path)
    created = client.post(
        f"/api/projects/{pid}/data-translation-runs",
        json={
            "name": "output",
            "input_dataset_version_id": source,
            "model_weight_version_id": weight,
        },
    )
    with SessionLocal() as session:
        task = session.get(m.Task, created.json()["task_id"])
        attempt = m.TaskAttempt(task_id=task.id, attempt_number=1, status="running")
        session.add(attempt)
        session.flush()
        with pytest.raises(ValueError, match="manifest"):
            finalize_task_domain(session, task, attempt, {"train_ready": True})


def test_native_generation_freezes_source_name_policy_and_lineage(client, tmp_path):
    pid, source_id, weight_id, digest, _ = setup(client, tmp_path)
    response = client.post(f"/api/projects/{pid}/knowledge-generation-runs", json={
        "name": "source names", "input_dataset_version_id": source_id,
        "model_weight_version_id": weight_id, "prompt_version": "cpt-generation-v8",
        "styles": ["facts", "textbook"], "document_mode": "whole", "engine": "vllm",
        "proper_name_policy": "source_exact", "validation_retry_limit": 0,
    })
    assert response.status_code == 202, response.text
    result = response.json()
    with SessionLocal() as session:
        task = session.get(m.Task, result["task_id"])
        assert task.project_id == pid
        assert task.payload["generation"]["proper_name_policy"] == "source_exact"
        assert task.payload["generation"]["model_weight_version_id"] == weight_id
        assert task.payload["generation_input"]["artifact_hash"] == digest
        output = session.get(m.DatasetVersion, result["output_dataset_version_id"])
        assert output.parent_version_id == source_id
