from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import TaskAttempt, TaskEvent, utcnow
from llm_platform.services.tasks import create_task


def test_task_timing_keeps_a_window_anchor_beyond_the_recent_event_limit(
    client: TestClient, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_response = client.post(
        "/api/projects",
        json={
            "name": f"Timing-{uuid4().hex[:8]}",
            "target_model_name": "Timing-LM",
            "workspace_uri": str(workspace),
            "project_type": "pretrain",
            "target_parameters": "42M",
            "primary_languages": ["zh"],
            "target_capabilities": ["test"],
        },
    )
    assert project_response.status_code == 201

    sample_end = utcnow()
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project_response.json()["id"],
            kind="train",
            title="dense progress timing",
            adapter_id="builtin.command.training",
        )
        task.observed_state = "running"
        task.progress = 0.8
        task.started_at = sample_end - timedelta(seconds=60)
        attempt = TaskAttempt(
            task_id=task.id,
            attempt_number=1,
            status="running",
            started_at=task.started_at,
        )
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        for index in range(301):
            session.add(
                TaskEvent(
                    task_id=task.id,
                    attempt_id=attempt.id,
                    sequence=index + 2,
                    event_type="progress",
                    payload={"progress": 0.2 + 0.6 * index / 300},
                    created_at=sample_end - timedelta(seconds=60 - index * 0.2),
                )
            )
        session.commit()
        task_id = task.id

    response = client.get(f"/api/tasks/{task_id}?include_events=false")
    assert response.status_code == 200
    timing = response.json()["timing_estimate"]
    assert timing["status"] == "available"
    assert timing["sample_count"] >= 121
    assert timing["remaining_seconds"] is not None
    assert timing["estimated_completion_at"] is not None
