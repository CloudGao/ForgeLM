from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import DatasetVersion, Task
from llm_platform.services.artifacts import artifact_storage
from llm_platform.services.dataset_analyses import create_analysis_revision


def create_project(client: TestClient, tmp_path: Path) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"Modalities-{uuid4().hex[:8]}",
            "target_model_name": "Modal-LM",
            "workspace_uri": str(tmp_path),
            "project_type": "pretrain",
            "target_parameters": "42M",
            "primary_languages": ["zh"],
            "target_capabilities": ["text"],
        },
    )
    assert response.status_code == 201
    return response.json()


def test_dataset_modality_and_parser_contract_are_frozen_in_import_task(
    client: TestClient, tmp_path: Path
) -> None:
    project = create_project(client, tmp_path)
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")

    response = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Text-{uuid4().hex[:8]}",
            "modality": "text",
            "source_type": "local",
            "source_uri": str(source),
            "start_import": True,
        },
    )

    assert response.status_code == 201
    dataset = response.json()
    version = dataset["current_version"]
    assert dataset["modality"] == "text"
    assert version["modality"] == "text"
    assert version["parser_id"] == "builtin.text-parser"
    assert version["parser_version"] == "1.0.0"
    with SessionLocal() as session:
        task = session.get(Task, version["task_id"])
        assert task is not None
        assert task.payload["modality"] == "text"
        assert task.payload["parser"] == {
            "id": "builtin.text-parser",
            "version": "1.0.0",
        }
    cancelled = client.post(
        f"/api/tasks/{version['task_id']}/control", json={"action": "cancel"}
    )
    assert cancelled.status_code == 200


def test_non_text_dataset_is_registered_without_a_fake_semantic_parser(
    client: TestClient, tmp_path: Path
) -> None:
    project = create_project(client, tmp_path)
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video-source")

    response = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Video-{uuid4().hex[:8]}",
            "modality": "video",
            "source_type": "local",
            "source_uri": str(source),
            "start_import": True,
        },
    )

    assert response.status_code == 201
    version = response.json()["current_version"]
    assert version["modality"] == "video"
    assert version["parser_id"] is None
    with SessionLocal() as session:
        task = session.get(Task, version["task_id"])
        assert task is not None
        assert task.payload["parser"] is None
    cancelled = client.post(
        f"/api/tasks/{version['task_id']}/control", json={"action": "cancel"}
    )
    assert cancelled.status_code == 200


def test_unknown_dataset_modality_is_rejected(client: TestClient, tmp_path: Path) -> None:
    project = create_project(client, tmp_path)
    response = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Unknown-{uuid4().hex[:8]}",
            "modality": "binary-ish",
            "source_type": "local",
            "start_import": False,
        },
    )
    assert response.status_code == 422


def test_reanalysis_creates_analysis_revision_without_changing_content_version(
    client: TestClient, tmp_path: Path
) -> None:
    project = create_project(client, tmp_path)
    response = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"Stable-{uuid4().hex[:8]}",
            "modality": "text",
            "source_type": "local",
            "source_uri": str(tmp_path / "stable.jsonl"),
            "start_import": False,
        },
    )
    assert response.status_code == 201
    dataset = response.json()
    version_id = dataset["current_version"]["id"]
    file_record = {"path": "stable.jsonl", "size_bytes": 10, "sha256": "a" * 64}
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session, project_id=project["id"], kind="dataset_snapshot",
            payload=b'{"text":"stable"}\n', filename="stable.jsonl",
            media_type="application/x-ndjson",
        )
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        version.artifact_id = artifact.id
        version.state = "ready"
        version.source_snapshot = {"files": [file_record]}
        analysis = create_analysis_revision(
            session, version, task_id=None, status="succeeded"
        )
        analysis.source_snapshot = {"files": [file_record]}
        analysis.snapshot_artifact_id = artifact.id
        analysis.snapshot_content_hash = artifact.content_hash
        session.commit()

    reanalyze = client.post(
        f"/api/datasets/{dataset['id']}/reanalyze",
        json={"resource_request": {"gpus": 0}},
    )

    assert reanalyze.status_code == 202, reanalyze.text
    body = reanalyze.json()
    assert body["current_version"]["id"] == version_id
    assert body["current_version"]["version"] == 1
    assert body["reanalyze"]["analysis_revision"] == 2
    assert body["current_version"]["active_analysis"]["id"] == body["reanalyze"][
        "analysis_revision_id"
    ]
    with SessionLocal() as session:
        task = session.get(Task, body["reanalyze"]["task_id"])
        assert task is not None
        assert task.payload["dataset_version_id"] == version_id
        assert task.payload["expected_snapshot_files"] == [file_record]
    cancelled = client.post(
        f"/api/tasks/{body['reanalyze']['task_id']}/control",
        json={"action": "cancel"},
    )
    assert cancelled.status_code == 200
