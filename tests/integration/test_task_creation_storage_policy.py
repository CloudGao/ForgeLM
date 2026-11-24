from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.services.tasks import create_task


def test_task_creation_does_not_apply_a_global_disk_reserve(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_response = client.post(
        "/api/projects",
        json={
            "name": f"No-global-disk-gate-{uuid4().hex[:8]}",
            "target_model_name": "Storage-policy-test",
            "workspace_uri": str(workspace),
            "project_type": "pretrain",
            "target_parameters": "1M",
            "primary_languages": ["zh"],
            "target_capabilities": ["test"],
        },
    )
    assert project_response.status_code == 201

    def unexpected_disk_usage(_path):
        raise AssertionError("generic task creation must not inspect a disk reserve")

    monkeypatch.setattr(shutil, "disk_usage", unexpected_disk_usage)
    with SessionLocal() as session:
        task = create_task(
            session,
            project_id=project_response.json()["id"],
            kind="inference",
            title="low-write inference",
            adapter_id="builtin.transformers-inference",
        )
        session.commit()
        assert task.observed_state == "queued"
