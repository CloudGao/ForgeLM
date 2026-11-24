from uuid import uuid4

import pytest
from llm_platform.database import SessionLocal
from llm_platform.models import Artifact, Dataset, DatasetVersion, Task, TaskAttempt


def project(client, tmp_path):
    root = tmp_path / uuid4().hex
    root.mkdir()
    r = client.post(
        "/api/projects",
        json={
            "name": "Delete-" + uuid4().hex,
            "target_model_name": "test",
            "workspace_uri": str(root),
        },
    )
    assert r.status_code == 201
    return r.json(), root


def preview(client, p):
    r = client.get(f"/api/projects/{p['id']}/deletion-preview")
    assert r.status_code == 200, r.text
    return r.json()


def execute(client, p, plan):
    return client.post(
        f"/api/projects/{p['id']}/delete",
        json={"confirmation": p["name"], "fingerprint": plan["fingerprint"]},
    )


def test_project_deletion_removes_records_files_and_preserves_shared_source(client, tmp_path):
    p, root = project(client, tmp_path)
    other, shared_root = project(client, tmp_path)
    (root / "output.txt").write_text("only this project")
    (shared_root / "source.txt").write_text("shared source")
    with SessionLocal() as session:
        shared = Artifact(
            kind="dataset_snapshot",
            uri=(shared_root / "source.txt").as_uri(),
            content_hash=uuid4().hex * 2,
            metadata_json={"project_id": other["id"]},
        )
        session.add(shared)
        session.flush()
        dataset = Dataset(project_id=other["id"], name="shared", source_type="local")
        session.add(dataset)
        session.flush()
        version = DatasetVersion(dataset_id=dataset.id, version=1, artifact_id=shared.id)
        session.add(version)
        session.flush()
        task = Task(
            project_id=p["id"],
            kind="data_import",
            title="old custom task",
            desired_state="cancelled",
            observed_state="cancelled",
            payload={"source_id": version.id},
        )
        session.add(task)
        session.commit()
        artifact_id, version_id = shared.id, version.id
    client.post(
        f"/api/projects/{p['id']}/data-bindings",
        json={"asset_type": "dataset_version", "asset_version_id": version_id},
    )
    plan = preview(client, p)
    assert plan["can_delete"], plan["blockers"]
    r = execute(client, p, plan)
    assert r.status_code == 200, r.text
    assert not root.exists()
    assert (shared_root / "source.txt").read_text() == "shared source"
    assert client.get(f"/api/projects/{p['id']}").status_code == 404
    with SessionLocal() as session:
        assert session.get(Artifact, artifact_id) is not None
        assert session.get(DatasetVersion, version_id) is not None


@pytest.mark.parametrize("reference_kind", ["binding", "json"])
def test_project_deletion_blocks_cross_project_consumers(client, tmp_path, reference_kind):
    p, root = project(client, tmp_path)
    other, _ = project(client, tmp_path)
    with SessionLocal() as session:
        d = Dataset(project_id=p["id"], name=uuid4().hex, source_type="local")
        session.add(d)
        session.flush()
        v = DatasetVersion(dataset_id=d.id, version=1)
        session.add(v)
        session.flush()
        version_id = v.id
        if reference_kind == "json":
            session.add(
                Task(
                    project_id=other["id"],
                    kind="train",
                    title="consumer",
                    observed_state="succeeded",
                    payload={"nested": [{"dataset_version_id": version_id}]},
                )
            )
        session.commit()
    if reference_kind == "binding":
        client.post(
            f"/api/projects/{other['id']}/data-bindings",
            json={"asset_type": "dataset_version", "asset_version_id": version_id},
        )
    plan = preview(client, p)
    assert not plan["can_delete"]
    assert execute(client, p, plan).status_code == 409
    assert root.exists()


def test_project_deletion_rechecks_activity_and_changed_files(client, tmp_path):
    p, root = project(client, tmp_path)
    plan = preview(client, p)
    (root / "new.txt").write_text("changed")
    assert execute(client, p, plan).status_code == 409
    plan = preview(client, p)
    with SessionLocal() as session:
        session.add(
            Task(project_id=p["id"], kind="train", title="active", observed_state="running")
        )
        session.commit()
    assert execute(client, p, plan).status_code == 409
    assert root.exists()


def test_project_deletion_blocks_overlapping_workspaces(client, tmp_path):
    p, root = project(client, tmp_path)
    other, _ = project(client, root)
    plan = preview(client, p)
    assert any(b["kind"] == "overlapping_workspace" for b in plan["blockers"])
    assert execute(client, p, plan).status_code == 409
    assert client.get(f"/api/projects/{other['id']}").status_code == 200


def test_project_deletion_rejects_wrong_confirmation(client, tmp_path):
    p, root = project(client, tmp_path)
    plan = preview(client, p)
    r = client.post(
        f"/api/projects/{p['id']}/delete",
        json={"confirmation": "wrong", "fingerprint": plan["fingerprint"]},
    )
    assert r.status_code == 422
    assert root.exists()


def test_project_deletion_accepts_cancelled_task_with_old_paused_attempt(client, tmp_path):
    p, root = project(client, tmp_path)
    with SessionLocal() as session:
        task = Task(project_id=p['id'], kind='train', title='cancelled', observed_state='cancelled')
        session.add(task)
        session.flush()
        session.add(TaskAttempt(task_id=task.id, attempt_number=1, status='paused'))
        session.commit()
    plan = preview(client, p)
    assert plan['can_delete'], plan['blockers']
    assert execute(client, p, plan).status_code == 200
    assert not root.exists()
