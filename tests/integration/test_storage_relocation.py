from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from llm_platform import models as m
from llm_platform.config import get_settings
from llm_platform.database import SessionLocal
from llm_platform.services import storage_management
from llm_platform.services.artifacts import artifact_storage


@pytest.fixture
def relocation(client, tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    project = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "Test", "workspace_uri": str(old)},
    ).json()
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            payload=uuid4().hex.encode(),
            filename="data.jsonl",
        )
        version = m.Dataset(project_id=project["id"], name="source", source_type="local")
        session.add(version)
        session.flush()
        dv = m.DatasetVersion(
            dataset_id=version.id, version=1, artifact_id=artifact.id, state="ready"
        )
        session.add(dv)
        session.flush()
        loc = artifact_storage._managed_location(session, artifact, project["id"])
        dv.primary_location_id = loc.id
        session.commit()
        data = {
            "project": project["id"],
            "artifact": artifact.id,
            "location": loc.id,
            "version": dv.id,
            "source": Path(artifact.uri.removeprefix("file:///"))
            if __import__("os").name == "nt"
            else Path(artifact.uri.removeprefix("file://")),
        }
    root = tmp_path / "unified"
    monkeypatch.setattr(get_settings(), "storage_root", str(root))
    monkeypatch.setenv("LLM_PLATFORM_STORAGE_ROOT", str(root))
    monkeypatch.setattr(storage_management._executor, "submit", lambda fn, *args: fn(*args))
    return {**data, "root": root, "old": old}


def _preview(client, data, scope="assets"):
    r = client.post(
        "/api/storage/relocations/preview", json={"scope": scope, "project_ids": [data["project"]]}
    )
    assert r.status_code == 201, r.text
    return r.json()


def _execute(client, plan):
    r = client.post(
        "/api/storage/relocations/" + plan["id"] + "/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    assert r.status_code == 202, r.text
    return client.get("/api/storage/cleanup-plans/" + plan["id"]).json()


def test_relocates_shared_data_preserving_identity_primary_and_history(client, relocation):
    d = relocation
    before = d["source"].read_bytes()
    plan = _preview(client, d)
    assert len(plan["items"]) == 1
    target = Path(plan["items"][0]["metadata_json"]["destination"])
    assert target.is_relative_to(d["root"] / "dataset")
    result = _execute(client, plan)
    assert result["status"] == "succeeded", result
    assert not d["source"].exists() and target.read_bytes() == before
    with SessionLocal() as session:
        version = session.get(m.DatasetVersion, d["version"])
        loc = session.get(m.ArtifactLocation, d["location"])
        artifact = session.get(m.Artifact, d["artifact"])
        assert version.primary_location_id == loc.id and version.artifact_id == artifact.id
        assert artifact.content_hash == hashlib.sha256(before).hexdigest()
        assert artifact_storage.managed_payload_path(session, artifact, d["project"]) == target
    from llm_platform_core.storage_layout import resolve_relocated_path

    assert Path(resolve_relocated_path(str(d["source"]))) == target


@pytest.mark.parametrize("change", ["source_changed", "target_exists", "lease"])
def test_relocation_refuses_changed_or_in_use_files(client, relocation, change):
    d = relocation
    plan = _preview(client, d)
    target = Path(plan["items"][0]["metadata_json"]["destination"])
    lease_id = None
    if change == "source_changed":
        d["source"].write_bytes(b"changed")
    elif change == "target_exists":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"other")
    else:
        with SessionLocal() as session:
            task = m.Task(
                project_id=d["project"], kind="test", title="lease", observed_state="paused"
            )
            worker = m.WorkerNode(name=uuid4().hex, hostname="host")
            session.add_all([task, worker])
            session.flush()
            attempt = m.TaskAttempt(
                task_id=task.id, attempt_number=1, status="paused", ended_at=m.utcnow()
            )
            session.add(attempt)
            session.flush()
            lease = m.ResourceLease(
                task_attempt_id=attempt.id, worker_id=worker.id, expires_at=m.utcnow()
            )
            session.add(lease)
            session.commit()
            lease_id = lease.id
        r = client.post(
            "/api/storage/relocations/" + plan["id"] + "/execute",
            json={"confirmation": plan["confirmation_phrase"]},
        )
        assert r.status_code == 409
        with SessionLocal() as session:
            session.get(m.ResourceLease, lease_id).released_at = m.utcnow()
            session.commit()
        return
    result = _execute(client, plan)
    assert result["status"] == "partial" and d["source"].exists()


def test_new_project_and_public_upload_use_unified_roots(client, relocation):
    d = relocation
    response = client.post("/api/projects", json={"name": uuid4().hex, "target_model_name": "Test"})
    assert response.status_code == 201, response.text
    p = response.json()
    assert Path(p["workspace_uri"]).is_relative_to(d["root"] / "projects")
    with SessionLocal() as session:
        payload = uuid4().hex.encode()
        shared = artifact_storage.put_bytes(
            session,
            project_id=d["project"],
            kind="dataset_record_manifest",
            payload=payload,
            filename="records.jsonl",
        )
        again = artifact_storage.put_bytes(
            session,
            project_id=p["id"],
            kind="dataset_record_manifest",
            payload=payload,
            filename="records.jsonl",
        )
        assert shared.id == again.id
        from sqlalchemy import select

        assert (
            len(
                list(
                    session.scalars(
                        select(m.ArtifactLocation).where(
                            m.ArtifactLocation.artifact_id == shared.id
                        )
                    )
                )
            )
            == 1
        )
        model = artifact_storage.put_bytes(
            session, project_id=p["id"], kind="model", payload=payload, filename="model.bin"
        )
        assert artifact_storage.managed_payload_path(session, model, p["id"]).is_relative_to(
            Path(p["workspace_uri"]) / "artifacts"
        )
        session.commit()


@pytest.mark.parametrize("cross_device", [False, True])
def test_recovery_after_location_switch_keeps_ids_and_completes_copy(
    client, relocation, monkeypatch, cross_device
):
    from llm_platform.services import storage_relocation as service

    d = relocation
    monkeypatch.setattr(service, "_same_device", lambda *_: not cross_device)
    plan = _preview(client, d)
    original = service._journal
    monkeypatch.setattr(
        service, "_journal", lambda *_: (_ for _ in ()).throw(OSError("injected interruption"))
    )
    result = _execute(client, plan)
    assert result["status"] == "interrupted"
    assert result["items"][0]["status"] == "moved"
    assert d["source"].exists() == cross_device
    monkeypatch.setattr(service, "_journal", original)
    result = _execute(client, result)
    assert result["status"] == "succeeded"
    assert not d["source"].exists()
    assert result["items"][0]["metadata_json"]["phase"] == "complete"


def test_recovery_after_rename_before_registration(client, relocation):
    from llm_platform.services import storage_relocation as service

    d = relocation
    plan = _preview(client, d)
    target = Path(plan["items"][0]["metadata_json"]["destination"])
    target.parent.mkdir(parents=True)
    with SessionLocal() as session:
        item = session.get(m.StorageCleanupPlanItem, plan["items"][0]["id"])
        item.metadata_json = {**item.metadata_json, "phase": "moving", "same_filesystem": True}
        session.get(m.StorageCleanupPlan, plan["id"]).status = "executing"
        session.commit()
    d["source"].rename(target)
    service.recover_interrupted()
    result = _execute(client, plan)
    assert result["status"] == "succeeded"
    with SessionLocal() as session:
        assert session.get(m.DatasetVersion, d["version"]).primary_location_id == d["location"]


def test_nested_asset_locations_move_as_one_tree(client, relocation):
    d = relocation
    parent = d["old"] / "checkpoint"
    parent.mkdir()
    child = parent / "model.bin"
    child.write_bytes(b"immutable model")
    with SessionLocal() as session:
        a = m.Artifact(
            kind="checkpoint", uri=parent.as_uri(), content_hash=uuid4().hex * 2, size_bytes=15
        )
        b = m.Artifact(
            kind="model",
            uri=child.as_uri(),
            content_hash=hashlib.sha256(child.read_bytes()).hexdigest(),
            size_bytes=15,
        )
        session.add_all([a, b])
        session.flush()
        locations = [
            m.ArtifactLocation(
                artifact_id=a.id,
                project_id=d["project"],
                storage_type="external",
                availability_state="available",
                uri=parent.as_uri(),
            ),
            m.ArtifactLocation(
                artifact_id=b.id,
                project_id=d["project"],
                storage_type="external",
                availability_state="available",
                uri=child.as_uri(),
            ),
        ]
        session.add_all(locations)
        session.commit()
        ids = [location.id for location in locations]
    plan = _preview(client, d)
    group = next(i for i in plan["items"] if i["path"] == str(parent))
    assert set(group["metadata_json"]["location_ids"]) == set(ids)
    result = _execute(client, plan)
    assert result["status"] == "succeeded"
    target = Path(group["metadata_json"]["destination"])
    assert (target / "model.bin").read_bytes() == b"immutable model"
    with SessionLocal() as session:
        assert session.get(m.ArtifactLocation, ids[1]).uri == (target / "model.bin").as_uri()


def test_project_runtime_migration_preserves_paused_paths(client, relocation):
    d = relocation
    with SessionLocal() as session:
        paused = m.Task(
            project_id=d["project"], kind="train", title="paused", observed_state="paused"
        )
        done = m.Task(
            project_id=d["project"], kind="test", title="done", observed_state="succeeded"
        )
        session.add_all([paused, done])
        session.flush()
        attempts = [
            m.TaskAttempt(
                task_id=t.id, attempt_number=1, status=t.observed_state, ended_at=m.utcnow()
            )
            for t in (paused, done)
        ]
        session.add_all(attempts)
        session.commit()
        paths = [
            d["old"] / ".llm-platform" / "runs" / t.id / a.id
            for t, a in zip((paused, done), attempts, strict=True)
        ]
    for path in paths:
        path.mkdir(parents=True)
        (path / "log.txt").write_text("saved")
    plan = _preview(client, d, "projects")
    assert any("暂停" in item["reason"] for item in plan["summary"]["blocked"])
    assert _execute(client, plan)["status"] == "succeeded"
    assert paths[0].exists() and not paths[1].exists()
    with SessionLocal() as session:
        from llm_platform.services.project_storage import workspace_history

        project = session.get(m.Project, d["project"])
        assert Path(project.workspace_uri).is_relative_to(d["root"] / "projects")
        assert str(d["old"]) in workspace_history(session, project)

    selected = client.post("/api/storage/relocations/preview", json={
        "scope": "projects", "project_ids": [d["project"]],
        "paused_training_task_ids": [paused.id],
    })
    assert selected.status_code == 201, selected.text
    selected_plan = selected.json()
    assert selected_plan["summary"]["paused_training_task_ids"] == [paused.id]
    assert _execute(client, selected_plan)["status"] == "succeeded"
    from llm_platform_core.storage_layout import resolve_relocated_path
    target = Path(resolve_relocated_path(str(paths[0])))
    assert not paths[0].exists() and (target / "log.txt").read_text() == "saved"
    with SessionLocal() as session:
        assert session.get(m.Task, paused.id).observed_state == "paused"


@pytest.mark.parametrize("invalid", ["kind", "project", "active", "unended", "scope"])
def test_paused_training_selection_rejects_unsafe_tasks(client, relocation, invalid):
    d = relocation
    with SessionLocal() as session:
        task = m.Task(project_id=d["project"], kind="test" if invalid == "kind" else "train",
                      title="pause selection", observed_state="running" if invalid == "active" else "paused")
        session.add(task)
        session.flush()
        session.add(m.TaskAttempt(task_id=task.id, attempt_number=1, status="paused",
                                  ended_at=None if invalid == "unended" else m.utcnow()))
        session.commit()
        task_id = task.id
    response = client.post("/api/storage/relocations/preview", json={
        "scope": "assets" if invalid == "scope" else "projects",
        "project_ids": [str(uuid4())] if invalid == "project" else [d["project"]],
        "paused_training_task_ids": [task_id],
    })
    assert response.status_code == 409, response.text
    with SessionLocal() as session:
        session.get(m.Task, task_id).observed_state = "paused"
        session.commit()


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows verbatim path semantics")
def test_cross_filesystem_copy_preserves_wsl_trailing_dot_names(client, relocation, monkeypatch):
    from llm_platform.services import storage_relocation as service
    from llm_platform_core.storage_cleanup import filesystem_path

    d = relocation
    directory = d["old"] / "dataset"
    physical = filesystem_path(directory) / "export_.."
    physical.mkdir(parents=True)
    (physical / "log.txt").write_bytes(b"original")
    with SessionLocal() as session:
        artifact = m.Artifact(
            kind="dataset_snapshot",
            uri=directory.as_uri(),
            content_hash=uuid4().hex * 2,
            size_bytes=8,
        )
        session.add(artifact)
        session.flush()
        session.add(
            m.ArtifactLocation(
                artifact_id=artifact.id,
                project_id=d["project"],
                storage_type="external",
                availability_state="available",
                uri=directory.as_uri(),
            )
        )
        session.commit()
    monkeypatch.setattr(service, "_same_device", lambda *_: False)
    plan = _preview(client, d)
    item = next(i for i in plan["items"] if i["path"] == str(directory))
    result = _execute(client, plan)
    assert result["status"] == "succeeded", result
    target = filesystem_path(item["metadata_json"]["destination"])
    assert (target / "export_.." / "log.txt").read_bytes() == b"original"
    assert not directory.exists()


def test_root_migration_preserves_registered_and_unregistered_files_and_waits_for_cutover(
    client, relocation, monkeypatch
):
    import json

    from llm_platform.services import storage_relocation as service
    from llm_platform_core.storage_layout import resolve_relocated_path

    d = relocation
    assert _execute(client, _preview(client, d))["status"] == "succeeded"
    assert _execute(client, _preview(client, d, "projects"))["status"] == "succeeded"
    source_input = d["root"] / "dataset" / "sources" / "retained.txt"
    source_input.parent.mkdir(parents=True)
    source_input.write_text("unregistered provenance")
    database = d["root"] / "platform" / "db" / "platform.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"live database placeholder")
    destination = d["root"].parent / "second-drive"
    r = client.post(
        "/api/storage/relocations/preview", json={"scope": "root", "target_root": str(destination)}
    )
    assert r.status_code == 201, r.text
    plan = r.json()
    monkeypatch.setattr(service, "_same_device", lambda a, b: False)
    result = _execute(client, plan)
    assert result["status"] == "awaiting_cutover", result
    assert database.exists()
    assert (destination / "dataset/sources/retained.txt").read_text() == "unregistered provenance"
    with SessionLocal() as session:
        assert service.active_plan(session) == plan["id"]
        version = session.get(m.DatasetVersion, d["version"])
        assert version.primary_location_id == d["location"]
        assert service.location_path(
            session, session.get(m.ArtifactLocation, d["location"])
        ).is_relative_to(destination)
        assert Path(session.get(m.Project, d["project"]).workspace_uri).is_relative_to(destination)
    mapped = Path(resolve_relocated_path(str(d["source"])))
    assert mapped.exists() and mapped.is_relative_to(destination)
    journal = d["root"] / "platform/config/path-relocations.json"
    new_journal = destination / "platform/config/path-relocations.json"
    new_journal.parent.mkdir(parents=True)
    new_journal.write_text(journal.read_text(), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "storage_root", str(destination))
    monkeypatch.setenv("LLM_PLATFORM_STORAGE_ROOT", str(destination))
    service.recover_interrupted()
    with SessionLocal() as session:
        assert service.active_plan(session) is None
        assert session.get(m.StorageCleanupPlan, plan["id"]).status == "succeeded"
    assert Path(resolve_relocated_path(str(d["source"]))) == mapped
    assert all(
        not Path(row["target"]).is_relative_to(d["root"] / "dataset")
        for row in json.loads(new_journal.read_text())
    )


def test_root_migration_rejects_nested_or_existing_destination(client, relocation):
    d = relocation
    for destination in (d["root"], d["root"] / "nested", d["root"].parent, d["old"]):
        response = client.post(
            "/api/storage/relocations/preview",
            json={"scope": "root", "target_root": str(destination)},
        )
        assert response.status_code == 409, response.text


@pytest.mark.parametrize("changed", ["source", "target"])
def test_copy_verified_state_protects_source_if_either_copy_changes(
    client, relocation, monkeypatch, changed
):
    from llm_platform.services import storage_relocation as service

    d = relocation
    original = service._journal
    monkeypatch.setattr(service, "_same_device", lambda a, b: False)
    monkeypatch.setattr(
        service, "_journal", lambda *a: (_ for _ in ()).throw(RuntimeError("cutover interrupted"))
    )
    plan = _execute(client, _preview(client, d))
    assert plan["status"] == "interrupted"
    target = Path(plan["items"][0]["metadata_json"]["destination"])
    changed_path = d["source"] if changed == "source" else target
    original_bytes = changed_path.read_bytes()
    changed_path.write_bytes(b"changed-copy")
    monkeypatch.setattr(service, "_journal", original)
    result = _execute(client, plan)
    assert result["status"] == "interrupted"
    assert d["source"].exists()
    # Restore metadata from the verified copy so the fixture can release its lock.
    import shutil

    if changed == "source":
        shutil.copy2(target, d["source"])
    else:
        shutil.copy2(d["source"], target)
    with SessionLocal() as session:
        item = session.get(m.StorageCleanupPlanItem, plan["items"][0]["id"])
        item.metadata_json = {k: v for k, v in item.metadata_json.items() if k != "verified_target"}
        session.commit()
    assert changed_path.read_bytes() == original_bytes
    assert _execute(client, result)["status"] == "succeeded"


def test_relocation_resumes_partial_source_removal_and_windows_readonly(client, relocation, monkeypatch):
    import os
    import stat
    from llm_platform.services import storage_relocation as service

    d = relocation
    source = d["old"] / "readonly-tool"
    source.mkdir()
    (source / "first.txt").write_bytes(b"first")
    retained = source / "pack.idx"
    retained.write_bytes(b"immutable pack")
    if os.name == "nt":
        retained.chmod(stat.S_IREAD)
    with SessionLocal() as session:
        artifact = m.Artifact(kind="model_weights", uri=source.as_uri(), content_hash=uuid4().hex * 2, size_bytes=19)
        session.add(artifact)
        session.flush()
        session.add(m.ArtifactLocation(artifact_id=artifact.id, project_id=d["project"],
                                      storage_type="external", availability_state="available", uri=source.as_uri()))
        session.commit()
    real_remove = service.remove_target
    interrupted = False

    def partial_remove(item):
        nonlocal interrupted
        if Path(item["path"]) == source and not interrupted:
            interrupted = True
            (source / "first.txt").unlink()
            raise PermissionError("interrupted after first file")
        return real_remove(item)

    monkeypatch.setattr(service, "_same_device", lambda *_: False)
    monkeypatch.setattr(service, "remove_target", partial_remove)
    plan = _execute(client, _preview(client, d))
    assert plan["status"] == "interrupted" and retained.exists()
    result = _execute(client, plan)
    assert result["status"] == "succeeded", result
    target = Path(next(i for i in result["items"] if i["path"] == str(source))["metadata_json"]["destination"])
    assert not source.exists()
    assert (target / "first.txt").read_bytes() == b"first"
    assert (target / "pack.idx").read_bytes() == b"immutable pack"
