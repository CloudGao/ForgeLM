from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.services import data_cleanup, storage_management


@pytest.fixture
def cleanup_data(client, tmp_path, monkeypatch):
    monkeypatch.setattr(storage_management._executor, "submit", lambda fn, *args: fn(*args))
    project = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "KeepLM", "workspace_uri": str(tmp_path)},
    ).json()
    with SessionLocal() as session:
        task = m.Task(
            project_id=project["id"],
            title="completed training",
            kind="train",
            observed_state="succeeded",
        )
        plan = m.TrainingPlan(project_id=project["id"], name="training")
        session.add_all([task, plan])
        session.flush()
        attempt = m.TaskAttempt(task_id=task.id, attempt_number=1, status="succeeded")
        revision = m.PlanRevision(plan_id=plan.id, revision=1, stages=[])
        session.add_all([attempt, revision])
        session.flush()
        run = m.TrainingRun(
            project_id=project["id"],
            task_id=task.id,
            plan_revision_id=revision.id,
            status="succeeded",
        )
        session.add(run)
        session.flush()
        data = ("final optimizer and model state " + uuid4().hex).encode()
        digest = hashlib.sha256(data).hexdigest()
        archive = (
            tmp_path / ".llm-platform/artifacts/checkpoint" / digest[:2] / digest / "final.tar.gz"
        )
        archive.parent.mkdir(parents=True)
        archive.write_bytes(data)
        artifact = m.Artifact(
            kind="checkpoint",
            uri=archive.as_uri(),
            size_bytes=len(data),
            content_hash=digest,
            created_by_task_attempt_id=attempt.id,
            metadata_json={
                "archived_directory": True,
                "source_path": f"/worker/projects/{project['id']}/.llm-platform/training-runs/{run.id}/checkpoint-final",
            },
        )
        session.add(artifact)
        session.flush()
        location = m.ArtifactLocation(
            artifact_id=artifact.id,
            project_id=project["id"],
            uri=archive.as_uri(),
            storage_type="managed",
        )
        cp = m.Checkpoint(
            project_id=project["id"], training_run_id=run.id, artifact_id=artifact.id, step=20
        )
        session.add_all([location, cp])
        session.flush()
        model = m.ModelVersion(
            project_id=project["id"], name="KeepLM", version=1, source_checkpoint_id=cp.id
        )
        session.add(model)
        session.flush()
        task.result = {
            "model_version_id": model.id,
            "checkpoint_id": cp.id,
            "artifact_id": artifact.id,
        }
        session.commit()
        historical = (
            tmp_path / ".llm-platform/worker-workspaces/projects" / project["id"] / ".llm-platform"
        )
        middle = historical / "training-runs" / run.id / "trainer/checkpoint-10"
        middle.mkdir(parents=True)
        (middle / "optimizer.pt").write_bytes(b"intermediate state")
        final = middle.parent.parent / "checkpoint-final"
        final.mkdir()
        (final / "training-state.pt").write_bytes(b"preserve final state")
        logs = middle.parent / "trainer_state.json"
        logs.write_text("{}")
        scratch = historical / "runs" / task.id / attempt.id / "inputs"
        scratch.mkdir(parents=True)
        (scratch / "cached-model.bin").write_bytes(b"cache")
        return dict(
            project=project["id"],
            task=task.id,
            attempt=attempt.id,
            run=run.id,
            model=model.id,
            cp=cp.id,
            artifact=artifact.id,
            location=location.id,
            middle=middle,
            final=final,
            archive=archive,
            scratch=scratch,
            logs=logs,
            root=tmp_path,
            revision=revision.id,
        )


def scan(client, data):
    response = client.post("/api/storage/scans", json={"project_ids": [data["project"]]})
    assert response.status_code == 202, response.text
    result = client.get(f"/api/storage/scans/{response.json()['id']}").json()
    assert result["status"] == "succeeded", result
    return result


def selected_plan(client, snapshot, paths):
    ids = [i["id"] for i in snapshot["items"] if i["path"] in {str(p) for p in paths}]
    response = client.post(
        "/api/storage/cleanup-plans", json={"scan_id": snapshot["id"], "item_ids": ids}
    )
    assert response.status_code == 201, response.text
    return response.json()


def execute(client, plan):
    response = client.post(
        f"/api/storage/cleanup-plans/{plan['id']}/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    assert response.status_code == 202, response.text
    return client.get(f"/api/storage/cleanup-plans/{plan['id']}").json()


@pytest.mark.parametrize("blocker", [None, "changed_copy", "registered", "running", "remote", "changed_source"])
def test_historical_manifest_copy_cleanup(client, cleanup_data, tmp_path, blocker):
    import platform

    d = cleanup_data
    with SessionLocal() as session:
        worker = m.WorkerNode(name=uuid4().hex, hostname=platform.node(),
                              labels={"location": "remote" if blocker == "remote" else "local"})
        session.add(worker)
        session.flush()
        attempt = session.get(m.TaskAttempt, d["attempt"])
        attempt.worker_id = worker.id
        base = tmp_path / "old-worker" / "projects" / d["project"] / ".llm-platform" / "runs" / d["task"] / d["attempt"]
        attempt.runtime_handle = {"run_dir": str(base)}
        data = ("immutable dataset manifest " + uuid4().hex).encode()
        source = tmp_path / "canonical-manifest.jsonl"
        source.write_bytes(data)
        artifact = m.Artifact(kind="dataset_record_manifest", uri=source.as_uri(),
                              size_bytes=len(data), content_hash=hashlib.sha256(data).hexdigest())
        session.add(artifact)
        session.flush()
        target = base / "inputs" / "data" / f"000-{artifact.id}" / "dataset-records.jsonl"
        target.parent.mkdir(parents=True)
        target.write_bytes(data if blocker != "changed_copy" else b"X" * len(data))
        if blocker == "registered":
            session.add(m.ArtifactLocation(artifact_id=artifact.id, uri=target.as_uri(), storage_type="external"))
        if blocker == "running":
            session.get(m.Task, d["task"]).observed_state = "running"
        session.commit()
    snapshot = scan(client, d)
    found = [i for i in snapshot["items"] if i["path"] == str(target)]
    if blocker not in {None, "changed_source"}:
        assert not found
        assert target.exists()
        return
    assert len(found) == 1 and found[0]["reclaimable"]
    plan = selected_plan(client, snapshot, [target])
    if blocker == "changed_source":
        source.write_bytes(b"Y" * len(data))
    result = execute(client, plan)
    assert target.exists() == (blocker == "changed_source"), result
    assert source.exists()


def test_selective_historical_cleanup_preserves_final_and_model(client, cleanup_data):
    d = cleanup_data
    # A new task reading the final archive must not prevent retiring other steps.
    with SessionLocal() as session:
        session.add(
            m.Task(
                project_id=d["project"],
                title="SFT from final",
                kind="train",
                observed_state="running",
                payload={
                    "resume_checkpoint": {"training_run_id": d["run"], "artifact_id": d["artifact"]}
                },
            )
        )
        session.commit()
    snapshot = scan(client, d)
    items = {i["path"]: i for i in snapshot["items"]}
    assert items[str(d["middle"])]["reclaimable"]
    assert "不能恢复" in items[str(d["middle"])]["metadata_json"]["impact"]
    plan = selected_plan(client, snapshot, [d["middle"]])
    assert plan["summary"]["kind"] == "data_cleanup"
    result = execute(client, plan)
    assert result["status"] == "succeeded", result
    assert not d["middle"].exists()
    assert all(d[k].exists() for k in ("final", "archive", "scratch", "logs"))
    with SessionLocal() as session:
        assert session.get(m.ModelVersion, d["model"]).status != "deleted"
        assert session.get(m.Checkpoint, d["cp"]).status == "available"


@pytest.mark.parametrize(
    "change",
    [
        "missing_final",
        "corrupt_final",
        "new_path_reader",
        "registered_path",
        "owner_active",
        "lease",
    ],
)
def test_changed_dependencies_skip_deletion(client, cleanup_data, change):
    d = cleanup_data
    snapshot = scan(client, d)
    plan = selected_plan(client, snapshot, [d["middle"]])
    if change == "missing_final":
        d["archive"].unlink()
    elif change == "corrupt_final":
        d["archive"].write_bytes(b"bad final archive")
    else:
        with SessionLocal() as session:
            if change == "new_path_reader":
                session.add(
                    m.Task(
                        project_id=d["project"],
                        title="new consumer",
                        kind="train",
                        observed_state="queued",
                        payload={"command": f"train --resume {d['middle']}"},
                    )
                )
            elif change == "registered_path":
                session.add(
                    m.ArtifactLocation(
                        artifact_id=d["artifact"],
                        project_id=d["project"],
                        uri=d["middle"].as_uri(),
                        storage_type="external",
                    )
                )
            elif change == "owner_active":
                session.get(m.Task, d["task"]).observed_state = "running"
            elif change == "lease":
                worker = m.WorkerNode(name="worker", hostname="host")
                session.add(worker)
                session.flush()
                session.add(
                    m.ResourceLease(
                        task_attempt_id=d["attempt"],
                        worker_id=worker.id,
                        status="active",
                        expires_at=m.utcnow(),
                    )
                )
            session.commit()
    result = execute(client, plan)
    assert result["status"] == "partial", result
    assert result["items"][0]["status"] == "skipped"
    assert d["middle"].exists()


@pytest.mark.parametrize("ids_kind", ["empty", "unknown", "duplicate", "protected", "other_scan"])
def test_invalid_selection_is_rejected(client, cleanup_data, ids_kind):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.Task, d["task"]).observed_state = "running"
        session.commit()
    snapshot = scan(client, d)
    protected = next(i for i in snapshot["items"] if i["item_type"] == "cleanup_candidate")
    ids = {
        "empty": [],
        "unknown": [uuid4().hex],
        "duplicate": [protected["id"]] * 2,
        "protected": [protected["id"]],
        "other_scan": [protected["id"]],
    }[ids_kind]
    if ids_kind == "other_scan":
        snapshot = scan(client, d)
    result = client.post(
        "/api/storage/cleanup-plans", json={"scan_id": snapshot["id"], "item_ids": ids}
    )
    assert result.status_code == 409, result.text


def test_cancelled_paused_attempt_without_lease_is_eligible(client, cleanup_data):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.TrainingRun, d["run"]).status = "cancelled"
        session.get(m.Task, d["task"]).observed_state = "cancelled"
        attempt = session.get(m.TaskAttempt, d["attempt"])
        attempt.status = "paused"
        attempt.ended_at = m.utcnow()
        session.commit()
    d["archive"].unlink()
    snapshot = scan(client, d)
    middle = next(i for i in snapshot["items"] if i["path"] == str(d["middle"]))
    assert middle["reclaimable"] and middle["category"] == "cancelled_checkpoint"


def test_orphan_cleanup_updates_availability_without_erasing_history(client, cleanup_data):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        session.commit()
    snapshot = scan(client, d)
    plan = selected_plan(client, snapshot, [d["archive"]])
    result = execute(client, plan)
    assert result["status"] == "succeeded", result
    assert not d["archive"].exists() and d["final"].exists()
    with SessionLocal() as session:
        assert session.get(m.Artifact, d["artifact"]).availability_state == "missing"
        assert session.get(m.ArtifactLocation, d["location"]).availability_state == "missing"
        assert session.get(m.Checkpoint, d["cp"]).status == "missing"
        assert session.get(m.Task, d["task"]).result["model_version_id"] == d["model"]


def test_orphan_plan_reference_is_protected(client, cleanup_data):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        session.get(m.PlanRevision, d["revision"]).input_model_ref = {"checkpoint_id": d["cp"]}
        session.commit()
    snapshot = scan(client, d)
    item = next(i for i in snapshot["items"] if i["path"] == str(d["archive"]))
    assert not item["reclaimable"] and "引用" in item["reason"]


def test_cleanup_fences_new_tasks_and_recovers_interrupted_plan(client, cleanup_data, monkeypatch):
    d = cleanup_data
    snapshot = scan(client, d)
    plan = selected_plan(client, snapshot, [d["middle"]])
    monkeypatch.setattr(storage_management._executor, "submit", lambda *args: None)
    result = execute(client, plan)
    assert result["status"] == "executing"
    from llm_platform.services.model_deletion import guard_task_inputs

    with SessionLocal() as session, pytest.raises(HTTPException):
        guard_task_inputs(session, {})
    other = selected_plan(client, snapshot, [d["scratch"]])
    assert (
        client.post(
            f"/api/storage/cleanup-plans/{other['id']}/execute",
            json={"confirmation": other["confirmation_phrase"]},
        ).status_code
        == 409
    )
    data_cleanup.recover_interrupted()
    result = client.get(f"/api/storage/cleanup-plans/{plan['id']}").json()
    assert result["status"] == "partial" and d["middle"].exists()


def test_model_preview_does_not_replace_storage_inventory(client, cleanup_data):
    snapshot = scan(client, cleanup_data)
    with SessionLocal() as session:
        session.add(
            m.StorageScan(status="succeeded", scope={"kind": "model_deletion"}, ended_at=m.utcnow())
        )
        session.commit()
    overview = client.get("/api/storage/overview").json()
    assert overview["latest_successful_scan"]["id"] == snapshot["id"]


def test_final_recovery_proof_cannot_be_selected_for_deletion(client, cleanup_data):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        session.commit()
    snapshot = scan(client, d)
    ids = [i["id"] for i in snapshot["items"] if i["path"] in {str(d["middle"]), str(d["archive"])}]
    assert len(ids) == 2
    response = client.post(
        "/api/storage/cleanup-plans", json={"scan_id": snapshot["id"], "item_ids": ids}
    )
    assert response.status_code == 409
    assert "最终恢复副本" in response.text


def test_legacy_task_summary_is_not_an_orphan_payload(client, cleanup_data):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        artifact = session.get(m.Artifact, d["artifact"])
        artifact.metadata_json = {"task_id": d["task"], "summary": True}
        session.commit()
    snapshot = scan(client, d)
    assert not any(i["path"] == str(d["archive"]) for i in snapshot["items"])


def test_reparse_checkpoint_is_visible_but_not_reclaimable(client, cleanup_data, monkeypatch):
    from types import SimpleNamespace

    d = cleanup_data
    original = Path.lstat

    def marked(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == d["middle"]:
            return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=0x400)
        return value

    monkeypatch.setattr(Path, "lstat", marked)
    snapshot = scan(client, d)
    item = next(i for i in snapshot["items"] if i["path"] == str(d["middle"]))
    assert not item["reclaimable"]
    assert "重解析" in item["reason"]
    assert (d["middle"] / "optimizer.pt").exists()


def test_artifact_location_registration_is_fenced(client, cleanup_data, monkeypatch):
    from llm_platform.services.artifacts import LocalArtifactStorage

    d = cleanup_data
    plan = selected_plan(client, scan(client, d), [d["middle"]])
    monkeypatch.setattr(storage_management._executor, "submit", lambda *args: None)
    execute(client, plan)
    with SessionLocal() as session, pytest.raises(ValueError, match="重叠"):
        LocalArtifactStorage._register_location(
            session,
            session.get(m.Artifact, d["artifact"]),
            storage_type="external",
            uri=d["middle"].as_uri(),
            availability_state="available",
            project_id=d["project"],
        )
    data_cleanup.recover_interrupted()


def test_interrupted_orphan_removal_repairs_missing_metadata(client, cleanup_data, monkeypatch):
    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        session.commit()
    plan = selected_plan(client, scan(client, d), [d["archive"]])
    monkeypatch.setattr(storage_management._executor, "submit", lambda *args: None)
    execute(client, plan)
    # Emulate a process ending after unlink but before its database commit.
    d["archive"].unlink()
    data_cleanup.recover_interrupted()
    with SessionLocal() as session:
        assert session.get(m.Artifact, d["artifact"]).availability_state == "missing"
        assert session.get(m.Checkpoint, d["cp"]).status == "missing"
        assert session.get(m.StorageCleanupPlan, plan["id"]).status == "succeeded"


def test_interrupted_existing_archive_is_unknown_and_cannot_be_consumed(
    client, cleanup_data, monkeypatch
):
    from llm_platform.services.model_deletion import guard_task_inputs

    d = cleanup_data
    with SessionLocal() as session:
        session.get(m.ModelVersion, d["model"]).status = "deleted"
        session.commit()
    plan = selected_plan(client, scan(client, d), [d["archive"]])
    monkeypatch.setattr(storage_management._executor, "submit", lambda *args: None)
    execute(client, plan)
    data_cleanup.recover_interrupted()
    with SessionLocal() as session:
        assert session.get(m.Checkpoint, d["cp"]).status == "unknown"
        with pytest.raises(HTTPException) as error:
            guard_task_inputs(session, {"checkpoint_id": d["cp"]})
        assert error.value.detail["code"] == "checkpoint_deleted"


def test_new_cache_retention_policy_prevents_execution(client, cleanup_data):
    d = cleanup_data
    plan = selected_plan(client, scan(client, d), [d["scratch"]])
    with SessionLocal() as session:
        session.get(m.Project, d["project"]).storage_policy = {"cache_cleanup": "retain"}
        session.commit()
    result = execute(client, plan)
    assert result["status"] == "partial"
    assert "长期保留" in result["items"][0]["error"]["message"]
    assert d["scratch"].exists()


@pytest.mark.parametrize("case", ["success", "cancelled_stale_run", "still_paused", "open_attempt", "lease", "referenced", "missing_final", "unproven_cancel"])
def test_closed_pause_history_cleanup_keeps_live_protections(client, cleanup_data, case):
    d = cleanup_data
    with SessionLocal() as session:
        task = session.get(m.Task, d["task"])
        run = session.get(m.TrainingRun, d["run"])
        task.ended_at = m.utcnow()
        paused = m.TaskAttempt(task_id=task.id, attempt_number=2, status="paused", ended_at=m.utcnow())
        session.add(paused)
        session.flush()
        if case in {"cancelled_stale_run", "unproven_cancel"}:
            task.observed_state = "cancelled"
            run.status = "paused"
            if case == "unproven_cancel":
                task.ended_at = None
        elif case == "still_paused":
            task.observed_state = run.status = "paused"
        elif case == "open_attempt":
            paused.ended_at = None
        elif case == "lease":
            worker = m.WorkerNode(name="pause-lease", hostname="host")
            session.add(worker)
            session.flush()
            session.add(m.ResourceLease(task_attempt_id=paused.id, worker_id=worker.id, expires_at=m.utcnow()))
        elif case == "referenced":
            session.add(m.Task(project_id=d["project"], title="reader", kind="train", observed_state="queued",
                               payload={"resume_path": str(d["middle"])}))
        session.commit()
        paused_id = paused.id
    if case == "missing_final":
        d["archive"].unlink()
    snapshot = scan(client, d)
    middle = next(i for i in snapshot["items"] if i["path"] == str(d["middle"]))
    expected = case in {"success", "cancelled_stale_run"}
    assert middle["reclaimable"] == expected, middle
    if expected:
        result = execute(client, selected_plan(client, snapshot, [d["middle"]]))
        assert result["status"] == "succeeded", result
        assert not d["middle"].exists()
        assert d["final"].exists() and d["archive"].exists()
    with SessionLocal() as session:
        assert session.get(m.TaskAttempt, paused_id).status == "paused"
        if case == "cancelled_stale_run":
            assert session.get(m.TrainingRun, d["run"]).status == "paused"
