from __future__ import annotations

import hashlib
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException
from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.services import model_deletion as deletion
from sqlalchemy import select


@pytest.fixture
def registered(client, tmp_path, monkeypatch):
    from llm_platform.services import storage_management

    monkeypatch.setattr(storage_management._executor, "submit", lambda fn, *args: fn(*args))
    project = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "DeleteLM", "workspace_uri": str(tmp_path)},
    ).json()
    with SessionLocal() as session:
        task = m.Task(
            project_id=project["id"],
            title="finished",
            kind="train",
            desired_state="succeeded",
            observed_state="succeeded",
        )
        plan = m.TrainingPlan(project_id=project["id"], name="plan")
        session.add_all([task, plan])
        session.flush()
        revision = m.PlanRevision(plan_id=plan.id, revision=1, stages=[])
        session.add(revision)
        session.flush()
        run = m.TrainingRun(
            project_id=project["id"],
            task_id=task.id,
            plan_revision_id=revision.id,
            status="succeeded",
        )
        session.add(run)
        session.flush()
        paths = {}
        artifacts = {}
        for kind in ("model", "checkpoint", "dataset_snapshot"):
            path = tmp_path / ".llm-platform" / "artifacts" / kind / "payload.bin"
            path.parent.mkdir(parents=True)
            data = (kind + uuid4().hex).encode()
            path.write_bytes(data)
            artifact = m.Artifact(
                kind=kind,
                uri=path.as_uri(),
                size_bytes=len(data),
                content_hash=hashlib.sha256(data).hexdigest(),
            )
            session.add(artifact)
            session.flush()
            session.add(
                m.ArtifactLocation(
                    project_id=project["id"],
                    artifact_id=artifact.id,
                    uri=path.as_uri(),
                    storage_type="managed",
                    availability_state="available",
                )
            )
            artifacts[kind] = artifact.id
            paths[kind] = path
        cp = m.Checkpoint(
            project_id=project["id"],
            training_run_id=run.id,
            artifact_id=artifacts["checkpoint"],
            step=10,
            token_count=100,
        )
        session.add(cp)
        session.flush()
        model = m.ModelVersion(
            project_id=project["id"],
            name="DeleteLM",
            version=1,
            artifact_id=artifacts["model"],
            source_checkpoint_id=cp.id,
        )
        session.add(model)
        session.commit()
        root = tmp_path / ".llm-platform" / "training-runs" / run.id
        (root / "trainer" / "checkpoint-10").mkdir(parents=True)
        (root / "trainer" / "checkpoint-10" / "optimizer.pt").write_bytes(b"state")
        (root / "checkpoint-final").mkdir()
        (root / "checkpoint-final" / "training-state.pt").write_bytes(b"final state")
        (root / "metrics.json").write_text('{"loss": 1}')
        yield {
            "project": project["id"],
            "model": model.id,
            "run": run.id,
            "task": task.id,
            "checkpoint": cp.id,
            "paths": paths,
            "artifacts": artifacts,
            "root": root,
        }


def preview(client, data, mode="normal"):
    response = client.post(f"/api/models/{data['model']}/deletion-plans", json={"mode": mode})
    assert response.status_code == 201, response.text
    return response.json()


def execute(client, data, plan):
    url = f"/api/models/{data['model']}/deletion-plans/{plan['id']}"
    result = client.post(url + "/execute", json={"confirmation": plan["confirmation_phrase"]})
    assert result.status_code == 202, result.text
    return client.get(url).json()


@pytest.mark.parametrize("mode", ["normal", "cascade"])
def test_deletion_scope_and_history(client, registered, mode):
    data = registered
    plan = preview(client, data, mode)
    assert not plan["summary"]["blockers"]
    result = execute(client, data, plan)
    assert result["status"] == "succeeded", result
    assert not data["paths"]["model"].exists()
    assert data["paths"]["checkpoint"].exists() == (mode == "normal")
    assert (data["root"] / "checkpoint-final").exists() == (mode == "normal")
    assert data["paths"]["dataset_snapshot"].exists()
    assert (data["root"] / "metrics.json").exists()
    assert result["summary"]["actual_reclaimed_bytes"] > 0
    detail = client.get(f"/api/models/{data['model']}").json()
    assert detail["status"] == "deleted"
    assert detail["inference_source"]["type"] == "unavailable"
    assert detail["source_checkpoint"]["id"] == data["checkpoint"]
    listing = client.get(f"/api/projects/{data['project']}/models?include_archived=true").json()
    assert not listing["models"]
    assert client.post(f"/api/models/{data['model']}/mark-best").status_code == 410
    assert client.post(f"/api/models/{data['model']}/restore").status_code == 410
    repeated = execute(client, data, plan)
    assert (
        repeated["summary"]["actual_reclaimed_bytes"] == result["summary"]["actual_reclaimed_bytes"]
    )
    with SessionLocal() as session:
        assert session.get(m.TrainingRun, data["run"])
        assert session.scalar(
            select(m.AuditEvent.id).where(
                m.AuditEvent.action == "model.deletion.completed",
                m.AuditEvent.subject_id == data["model"],
            )
        )
        with pytest.raises(HTTPException):
            deletion.guard_task_inputs(session, {"nested": {"model_id": data["model"]}})


def test_shared_model_and_sibling_run_are_protected(client, registered):
    data = registered
    with SessionLocal() as session:
        session.add(
            m.ModelVersion(
                project_id=data["project"],
                name="Sibling",
                version=2,
                artifact_id=data["artifacts"]["model"],
                source_checkpoint_id=data["checkpoint"],
            )
        )
        session.commit()
    plan = preview(client, data, "cascade")
    assert all(row["status"] == "protected" for row in plan["items"])
    result = execute(client, data, plan)
    assert result["summary"]["actual_reclaimed_bytes"] == 0
    assert data["paths"]["model"].exists()
    assert (data["root"] / "checkpoint-final").exists()


def test_parent_child_history_does_not_block_independent_model(client, registered):
    with SessionLocal() as session:
        session.add(
            m.ModelVersion(
                project_id=registered["project"],
                name="IndependentChild",
                version=2,
                parent_model_version_id=registered["model"],
            )
        )
        session.commit()
    execute(client, registered, preview(client, registered))
    with SessionLocal() as session:
        child = session.scalar(
            select(m.ModelVersion).where(
                m.ModelVersion.parent_model_version_id == registered["model"]
            )
        )
        assert child.status != "deleted"


@pytest.mark.parametrize("protection", ["official", "active", "paused"])
def test_active_and_official_models_block_deletion(client, registered, protection):
    with SessionLocal() as session:
        if protection == "official":
            session.get(m.ModelVersion, registered["model"]).is_official = True
        elif protection == "paused":
            session.get(m.TrainingRun, registered["run"]).status = "paused"
        else:
            task = session.get(m.Task, registered["task"])
            task.observed_state = "running"
            task.payload = {"model_id": registered["model"]}
        session.commit()
    plan = preview(client, registered)
    assert plan["summary"]["blockers"]
    response = client.post(
        f"/api/models/{registered['model']}/deletion-plans/{plan['id']}/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    assert response.status_code == 409
    assert registered["paths"]["model"].exists()


@pytest.mark.parametrize("change", ["file", "reference", "expiry", "confirmation"])
def test_plan_must_be_current_and_confirmed(client, registered, change):
    plan = preview(client, registered)
    if change == "file":
        registered["paths"]["model"].write_bytes(b"changed")
    else:
        with SessionLocal() as session:
            if change == "reference":
                session.add(
                    m.ModelVersion(
                        project_id=registered["project"],
                        name="NewReader",
                        version=2,
                        artifact_id=registered["artifacts"]["model"],
                    )
                )
            if change == "expiry":
                session.get(m.StorageCleanupPlan, plan["id"]).expires_at = m.utcnow() - timedelta(
                    seconds=1
                )
            session.commit()
    result = client.post(
        f"/api/models/{registered['model']}/deletion-plans/{plan['id']}/execute",
        json={"confirmation": "wrong" if change == "confirmation" else plan["confirmation_phrase"]},
    )
    assert result.status_code in {409, 422}, result.text
    assert registered["paths"]["model"].exists()
    assert client.get(f"/api/models/{registered['model']}").json()["status"] != "deleted"


def test_nested_registered_data_protects_training_directory(client, registered):
    nested = registered["root"] / "checkpoint-final" / "shared-data.bin"
    nested.write_bytes(b"raw")
    with SessionLocal() as session:
        session.add(
            m.Artifact(
                kind="dataset_snapshot",
                uri=nested.as_uri(),
                content_hash=uuid4().hex * 2,
                size_bytes=3,
            )
        )
        session.commit()
    plan = preview(client, registered, "cascade")
    assert any(
        row["status"] == "protected" and row["path"] == str(nested.parent) for row in plan["items"]
    )
    execute(client, registered, plan)
    assert nested.exists()


def test_external_path_is_not_owned_and_failure_is_reported(client, registered, monkeypatch):
    plan = preview(client, registered)

    def error(_):
        raise PermissionError("locked file")

    monkeypatch.setattr(deletion, "remove_target", error)
    result = execute(client, registered, plan)
    assert result["status"] == "partial"
    assert result["summary"]["actual_reclaimed_bytes"] == 0
    assert registered["paths"]["model"].exists()
    assert "locked file" in str(result["items"])


def test_generic_storage_execute_cannot_bypass_model_deletion(client, registered):
    plan = preview(client, registered)
    response = client.post(
        f"/api/storage/cleanup-plans/{plan['id']}/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    assert response.status_code == 409
    assert registered["paths"]["model"].exists()


def test_remote_worker_scan_and_cleanup_are_audited(client, registered):
    with SessionLocal() as session:
        worker = m.WorkerNode(
            name="remote-delete",
            hostname="separate-host",
            status="online",
            capabilities=["builtin.storage-maintenance", deletion.WORKER_CAPABILITY],
            resources={"workspace": {"root": "/remote/workspaces"}},
        )
        session.add(worker)
        session.flush()
        artifact = session.get(m.Artifact, registered["artifacts"]["model"])
        artifact.uri = f"file:///remote/workspaces/projects/{registered['project']}/.llm-platform/artifacts/model/hash/model.bin"
        loc = session.scalar(
            select(m.ArtifactLocation).where(m.ArtifactLocation.artifact_id == artifact.id)
        )
        loc.uri = artifact.uri
        loc.worker_id = worker.id
        worker_id = worker.id
        session.commit()
    plan = preview(client, registered)
    assert plan["status"] == "scanning", plan
    item = plan["items"][0]
    with SessionLocal() as session:
        job = session.get(m.Task, item["metadata_json"]["maintenance_task_id"])
        deletion.refresh_worker_payload(session, job, worker_id)
        assert not job.payload["deletion_items"][0]["blocked"]
        job.observed_state = "succeeded"
        job.result = {
            "items": [
                {
                    "item_id": item["id"],
                    "path": item["path"],
                    "status": "inspected",
                    "size_bytes": 123,
                    "file_count": 1,
                    "fingerprint": "f" * 64,
                }
            ]
        }
        session.commit()
    url = f"/api/models/{registered['model']}/deletion-plans/{plan['id']}"
    plan = client.get(url).json()
    assert plan["status"] == "planned"
    assert plan["summary"]["estimated_reclaimable_bytes"] == 123
    result = execute(client, registered, plan)
    assert result["status"] == "executing"
    with SessionLocal() as session:
        job = session.get(m.Task, result["items"][0]["metadata_json"]["maintenance_task_id"])
        deletion.refresh_worker_payload(session, job, worker_id)
        job.observed_state = "succeeded"
        job.result = {"items": [{"item_id": item["id"], "path": item["path"], "status": "deleted"}]}
        session.commit()
        deletion.reconcile_plans(session)
    result = client.get(url).json()
    assert result["status"] == "succeeded", result
    assert result["summary"]["actual_reclaimed_bytes"] == 123
    with SessionLocal() as session:
        assert (
            session.get(m.Artifact, registered["artifacts"]["model"]).availability_state
            == "missing"
        )


def test_cascade_reclaims_exclusive_derived_but_not_other_runs_data(client, registered):
    ids = []
    paths = []
    with SessionLocal() as session:
        for kind in ("tokenized_corpus", "supervision_corpus"):
            path = (
                registered["root"].parents[2]
                / ".llm-platform"
                / "data-assets"
                / kind
                / "tokens.bin"
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(kind.encode())
            artifact = m.Artifact(
                kind=kind, uri=path.as_uri(), content_hash=uuid4().hex * 2, size_bytes=len(kind)
            )
            session.add(artifact)
            session.flush()
            ids.append(artifact.id)
            paths.append(path)
        run = session.get(m.TrainingRun, registered["run"])
        run.config_snapshot = {
            "data_inputs": ids,
            "raw_input": registered["artifacts"]["dataset_snapshot"],
        }
        task = m.Task(
            project_id=registered["project"],
            kind="train",
            title="other finished run",
            observed_state="succeeded",
            desired_state="succeeded",
        )
        session.add(task)
        session.flush()
        session.add(
            m.TrainingRun(
                project_id=registered["project"],
                task_id=task.id,
                plan_revision_id=run.plan_revision_id,
                status="succeeded",
                config_snapshot={"data_inputs": [ids[1]]},
            )
        )
        session.commit()
    result = execute(client, registered, preview(client, registered, "cascade"))
    assert result["status"] == "succeeded", result
    assert not paths[0].exists()
    assert paths[1].exists()
    assert registered["paths"]["dataset_snapshot"].exists()


def test_restart_retains_tombstone_and_releases_interrupted_plan(client, registered):
    plan = preview(client, registered)
    with SessionLocal() as session:
        stored = session.get(m.StorageCleanupPlan, plan["id"])
        stored.status = "executing"
        session.get(m.ModelVersion, registered["model"]).status = "deleted"
        session.commit()
    deletion.recover_interrupted_plans()
    with SessionLocal() as session:
        assert session.get(m.StorageCleanupPlan, plan["id"]).status == "partial"
        assert session.get(m.ModelVersion, registered["model"]).status == "deleted"
    assert registered["paths"]["model"].exists()
    assert preview(client, registered)["status"] == "planned"


@pytest.mark.parametrize("shared_view", [False, True, "plan"])
def test_typed_training_data_retains_shared_corpus_and_rejects_deleted_view(
    client, registered, shared_view
):
    from test_training_layered_data_lineage import supervision_chain

    paths = []
    with SessionLocal() as session:
        _, _, _, corpus, view = supervision_chain(session, registered["project"])
        for record, kind in ((corpus, "supervision_corpus"), (view, "supervision_training_index")):
            path = (
                registered["root"].parents[2] / ".llm-platform" / "data-assets" / kind / "data.bin"
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"encoded")
            artifact = m.Artifact(
                kind=kind, uri=path.as_uri(), content_hash=uuid4().hex * 2, size_bytes=7
            )
            session.add(artifact)
            session.flush()
            record.artifact_id = artifact.id
            paths.append(path)
        if shared_view is True:
            session.add(
                m.SupervisionTrainingViewVersion(
                    project_id=registered["project"],
                    corpus_version_id=corpus.id,
                    objective_kind="sft",
                    identity_hash=uuid4().hex,
                    format=view.format,
                    budget_unit=view.budget_unit,
                    budget_value=50,
                    status="ready",
                )
            )
        elif shared_view == "plan":
            plan = m.TrainingPlan(project_id=registered["project"], name="reusable")
            session.add(plan)
            session.flush()
            session.add(
                m.PlanRevision(plan_id=plan.id, revision=1, stages=[{"data_view_id": view.id}])
            )
        session.get(m.TrainingRun, registered["run"]).config_snapshot = {"view_id": view.id}
        view_id, corpus_id = view.id, corpus.id
        session.commit()
    result = execute(client, registered, preview(client, registered, "cascade"))
    assert result["status"] == "succeeded", result
    assert paths[0].exists() == bool(shared_view)
    assert paths[1].exists() == (shared_view == "plan")
    with SessionLocal() as session:
        assert session.get(m.SupervisionTrainingViewVersion, view_id).status == (
            "ready" if shared_view == "plan" else "missing"
        )
        assert session.get(m.SupervisionCorpusVersion, corpus_id).status == (
            "ready" if shared_view else "missing"
        )
        if shared_view != "plan":
            with pytest.raises(HTTPException) as failure:
                deletion.guard_task_inputs(session, {"supervision_view_id": view_id})
            assert failure.value.detail["code"] == "training_data_deleted"
        else:
            deletion.guard_task_inputs(session, {"supervision_view_id": view_id})


def remote_model(session, registered, *, online=True):
    worker = m.WorkerNode(
        name="remote-" + uuid4().hex,
        hostname="different-physical-host",
        status="online" if online else "offline",
        capabilities=["builtin.storage-maintenance", deletion.WORKER_CAPABILITY],
        resources={"workspace": {"root": "/remote/workspaces"}},
    )
    session.add(worker)
    session.flush()
    artifact = session.get(m.Artifact, registered["artifacts"]["model"])
    artifact.uri = f"file:///remote/workspaces/projects/{registered['project']}/.llm-platform/artifacts/model/weights"
    loc = session.scalar(
        select(m.ArtifactLocation).where(m.ArtifactLocation.artifact_id == artifact.id)
    )
    loc.uri, loc.worker_id = artifact.uri, worker.id
    return worker, artifact


def test_completion_reloads_stale_items_and_reconciles_without_new_worker_result(
    client, registered
):
    plan = preview(client, registered)
    with SessionLocal() as session:
        session.get(m.StorageCleanupPlan, plan["id"]).status = "executing"
        session.get(m.ModelVersion, registered["model"]).status = "deleted"
        session.commit()
    with SessionLocal() as stale_session:
        stored = stale_session.get(m.StorageCleanupPlan, plan["id"])
        stale_item = stale_session.get(m.StorageCleanupPlanItem, plan["items"][0]["id"])
        assert stale_item.status == "planned"
        with SessionLocal() as other_executor:
            item = other_executor.get(m.StorageCleanupPlanItem, stale_item.id)
            registered["paths"]["model"].unlink()
            item.status = "deleted"
            item.actual_reclaimed_bytes = item.size_bytes
            other_executor.commit()
        deletion.reconcile_worker_results(stale_session, stored)
        assert stored.status == "succeeded"
        assert stale_item.status == "deleted"
        assert stored.summary["actual_reclaimed_bytes"] == stale_item.size_bytes
        deletion.reconcile_worker_results(stale_session, stored)
        audits = list(
            stale_session.scalars(
                select(m.AuditEvent).where(
                    m.AuditEvent.subject_id == registered["model"],
                    m.AuditEvent.action == "model.deletion.completed",
                )
            )
        )
        assert len(audits) == 1


def test_restart_finalizes_already_completed_items(client, registered):
    result = execute(client, registered, preview(client, registered))
    with SessionLocal() as session:
        stored = session.get(m.StorageCleanupPlan, result["id"])
        stored.status = "executing"
        session.commit()
    deletion.recover_interrupted_plans()
    with SessionLocal() as session:
        assert session.get(m.StorageCleanupPlan, result["id"]).status == "succeeded"


def test_remote_nested_raw_location_protects_entire_directory(client, registered):
    with SessionLocal() as session:
        worker, artifact = remote_model(session, registered)
        session.add(
            m.ArtifactLocation(
                project_id=registered["project"],
                worker_id=worker.id,
                artifact_id=registered["artifacts"]["dataset_snapshot"],
                uri=artifact.uri + "/raw.jsonl",
                availability_state="available",
            )
        )
        session.commit()
    plan = preview(client, registered)
    assert plan["status"] == "planned", plan
    assert plan["summary"]["estimated_reclaimable_bytes"] == 0
    assert all(item["status"] == "protected" for item in plan["items"])


def test_offline_worker_is_retained_and_unclaimed_scan_times_out(client, registered):
    with SessionLocal() as session:
        worker, _ = remote_model(session, registered, online=False)
        worker_id = worker.id
        session.commit()
    assert preview(client, registered)["items"][0]["status"] == "protected"
    with SessionLocal() as session:
        session.get(m.WorkerNode, worker_id).status = "online"
        session.commit()
    plan = preview(client, registered)
    assert plan["status"] == "scanning"
    job_id = plan["items"][0]["metadata_json"]["maintenance_task_id"]
    with SessionLocal() as session:
        session.get(m.Task, job_id).created_at = m.utcnow() - timedelta(minutes=11)
        session.commit()
    plan = client.get(f"/api/models/{registered['model']}/deletion-plans/{plan['id']}").json()
    assert plan["status"] == "planned", plan
    assert plan["items"][0]["status"] == "protected"
    assert "十分钟" in plan["items"][0]["reason"]
    with SessionLocal() as session:
        job = session.get(m.Task, job_id)
        assert job.observed_state == "cancelled"
        with pytest.raises(HTTPException):
            deletion.refresh_worker_payload(session, job, worker_id)
