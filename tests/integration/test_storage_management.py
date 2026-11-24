from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import Task, TaskAttempt


def _wait_for(client: TestClient, path: str, terminal: set[str]) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] in terminal:
            return value
        time.sleep(0.02)
    raise AssertionError(f"operation did not finish: {path}")


def test_storage_scan_dry_run_and_guarded_cleanup(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/storage/overview").json()["cleanup_selection_supported"] is True
    workspace = tmp_path / "storage-project"
    workspace.mkdir()
    created = client.post(
        "/api/projects",
        json={
            "name": f"Storage-{uuid4().hex[:8]}",
            "target_model_name": "StorageLM",
            "workspace_uri": str(workspace),
        },
    )
    assert created.status_code == 201, created.text
    project = created.json()

    with SessionLocal() as session:
        finished_task = Task(
            project_id=project["id"],
            kind="data",
            title="finished scratch owner",
            observed_state="succeeded",
            desired_state="running",
        )
        active_task = Task(
            project_id=project["id"],
            kind="data",
            title="active scratch owner",
            observed_state="running",
            desired_state="running",
        )
        session.add_all([finished_task, active_task])
        session.flush()
        finished_attempt = TaskAttempt(
            task_id=finished_task.id, attempt_number=1, status="succeeded"
        )
        active_attempt = TaskAttempt(
            task_id=active_task.id, attempt_number=1, status="running"
        )
        session.add_all([finished_attempt, active_attempt])
        session.commit()

    root = workspace / ".llm-platform"
    protected = root / "artifacts" / "model" / "payload.bin"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"model")
    finished_scratch = root / "runs" / finished_task.id / finished_attempt.id / "inputs"
    finished_scratch.mkdir(parents=True)
    (finished_scratch / "manifest.bin").write_bytes(b"rebuildable")
    active_scratch = root / "runs" / active_task.id / active_attempt.id / "inputs"
    active_scratch.mkdir(parents=True)
    (active_scratch / "active.bin").write_bytes(b"active")

    submitted = client.post(
        "/api/storage/scans", json={"project_ids": [project["id"]]}
    )
    assert submitted.status_code == 202, submitted.text
    scan = _wait_for(
        client,
        f"/api/storage/scans/{submitted.json()['id']}",
        {"succeeded", "failed"},
    )
    assert scan["status"] == "succeeded", scan
    assert scan["summary"]["candidate_count"] == 1
    candidates = [item for item in scan["items"] if item["reclaimable"]]
    assert [Path(item["path"]) for item in candidates] == [finished_scratch.resolve()]
    assert scan["summary"]["reclaimable_bytes"] == len(b"rebuildable")

    planned = client.post(
        "/api/storage/cleanup-plans",
        json={"scan_id": scan["id"], "project_ids": [project["id"]]},
    )
    assert planned.status_code == 201, planned.text
    plan = planned.json()
    assert plan["summary"]["item_count"] == 1
    rejected = client.post(
        f"/api/storage/cleanup-plans/{plan['id']}/execute",
        json={"confirmation": "错误确认"},
    )
    assert rejected.status_code == 422

    executing = client.post(
        f"/api/storage/cleanup-plans/{plan['id']}/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    assert executing.status_code == 202, executing.text
    completed = _wait_for(
        client,
        f"/api/storage/cleanup-plans/{plan['id']}",
        {"succeeded", "partial", "failed"},
    )
    assert completed["status"] == "succeeded", completed
    assert completed["summary"]["actual_reclaimed_bytes"] == len(b"rebuildable")
    assert not finished_scratch.exists()
    assert active_scratch.is_dir()
    assert protected.is_file()


def test_cleanup_revalidates_changed_candidate(client: TestClient, tmp_path: Path) -> None:
    workspace = tmp_path / "changed-storage-project"
    workspace.mkdir()
    project = client.post(
        "/api/projects",
        json={
            "name": f"Changed-Storage-{uuid4().hex[:8]}",
            "target_model_name": "StorageLM",
            "workspace_uri": str(workspace),
        },
    ).json()
    with SessionLocal() as session:
        task = Task(
            project_id=project["id"],
            kind="data",
            title="changed scratch owner",
            observed_state="failed",
            desired_state="running",
        )
        session.add(task)
        session.flush()
        attempt = TaskAttempt(task_id=task.id, attempt_number=1, status="failed")
        session.add(attempt)
        session.commit()
    scratch = workspace / ".llm-platform" / "runs" / task.id / attempt.id / "inputs"
    scratch.mkdir(parents=True)
    payload = scratch / "partial.bin"
    payload.write_bytes(b"before")
    submitted = client.post(
        "/api/storage/scans", json={"project_ids": [project["id"]]}
    ).json()
    scan = _wait_for(
        client, f"/api/storage/scans/{submitted['id']}", {"succeeded", "failed"}
    )
    plan = client.post(
        "/api/storage/cleanup-plans", json={"scan_id": scan["id"]}
    ).json()
    payload.write_bytes(b"changed after scan")
    client.post(
        f"/api/storage/cleanup-plans/{plan['id']}/execute",
        json={"confirmation": plan["confirmation_phrase"]},
    )
    completed = _wait_for(
        client,
        f"/api/storage/cleanup-plans/{plan['id']}",
        {"succeeded", "partial", "failed"},
    )
    assert completed["status"] == "partial"
    assert completed["summary"]["skipped_count"] == 1
    assert completed["items"][0]["error"]["code"] == "candidate_changed"
    assert payload.is_file()
