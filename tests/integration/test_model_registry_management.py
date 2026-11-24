from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Artifact,
    AuditEvent,
    Checkpoint,
    PlanRevision,
    Task,
    TrainingPlan,
    TrainingRun,
)
from sqlalchemy import select


def _project(client: TestClient, tmp_path, suffix: str) -> dict:
    workspace = tmp_path / f"models-{suffix}-{uuid4().hex[:8]}"
    workspace.mkdir()
    response = client.post(
        "/api/projects",
        json={
            "name": f"Models-{suffix}-{uuid4().hex[:8]}",
            "target_model_name": "RegistryLM",
            "workspace_uri": str(workspace),
        },
    )
    assert response.status_code == 201
    return response.json()


def _register_huggingface(
    client: TestClient,
    project_id: str,
    *,
    name: str = "RegistryLM",
    revision: str = "0123456789abcdef0123456789abcdef01234567",
    parent_model_version_id: str | None = None,
) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": name,
            "kind": "base",
            "source_type": "huggingface",
            "huggingface_id": "example/registry-lm",
            "huggingface_revision": revision,
            "parent_model_version_id": parent_model_version_id,
            "parameter_count": 306_000_000,
            "metadata": {"description": "Imported model"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_model_registry_registers_real_sources_and_assigns_immutable_versions(
    client: TestClient, tmp_path
) -> None:
    project = _project(client, tmp_path, "sources")
    project_id = project["id"]

    first = _register_huggingface(client, project_id)
    assert first["version"] == 1
    assert first["status"] == "candidate"
    assert first["inference_source"] == {
        "type": "pretrained",
        "value": "example/registry-lm",
        "worker_ids": [],
        "globally_accessible": True,
    }
    assert first["metadata_json"]["inference"]["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert first["metadata_json"]["registry_source"]["type"] == "huggingface"

    second = _register_huggingface(
        client,
        project_id,
        revision="fedcba9876543210fedcba9876543210fedcba98",
        parent_model_version_id=first["id"],
    )
    assert second["version"] == 2
    assert second["parent_model"] == {
        "id": first["id"],
        "name": "RegistryLM",
        "version": 1,
        "kind": "base",
        "status": "candidate",
    }

    collision = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "RegistryLM",
            "version": 1,
            "source_type": "huggingface",
            "huggingface_id": "example/registry-lm",
            "huggingface_revision": "1" * 40,
        },
    )
    assert collision.status_code == 409
    assert collision.json()["detail"]["code"] == "model_version_exists"

    moving_revision = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "MovingRefLM",
            "source_type": "huggingface",
            "huggingface_id": "example/moving-ref-lm",
            "huggingface_revision": "main",
        },
    )
    assert moving_revision.status_code == 422

    external = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "ExternalLM",
            "kind": "sft",
            "source_type": "external_artifact",
            "artifact": {
                "kind": "model",
                "uri": f"file:///models/{uuid4().hex}/external-lm",
                "content_hash": "a" * 64,
                "size_bytes": 987654321,
                "media_type": "application/x-directory",
                "availability_state": "available",
            },
            "is_candidate": False,
        },
    )
    assert external.status_code == 201, external.text
    external_model = external.json()
    assert external_model["status"] == "registered"
    assert external_model["artifact"]["kind"] == "model"
    assert external_model["artifact"]["content_hash"] == "a" * 64
    assert external_model["inference_source"]["type"] == "external"
    assert external_model["inference_source"]["globally_accessible"] is True

    with SessionLocal() as session:
        plan = TrainingPlan(project_id=project_id, name=f"Imported-{uuid4().hex[:8]}")
        session.add(plan)
        session.flush()
        revision = PlanRevision(plan_id=plan.id, revision=1, stages=[])
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        task = Task(
            project_id=project_id,
            kind="train",
            title="Imported checkpoint",
            desired_state="succeeded",
            observed_state="succeeded",
        )
        session.add(task)
        session.flush()
        run = TrainingRun(
            project_id=project_id,
            plan_revision_id=revision.id,
            task_id=task.id,
            status="succeeded",
            current_stage="Stage 1",
            current_step=120,
            current_tokens=24_000_000,
        )
        artifact = Artifact(
            kind="checkpoint",
            uri=f"file:///checkpoints/{uuid4().hex}",
            content_hash="b" * 64,
            size_bytes=123456,
            availability_state="available",
        )
        session.add_all([run, artifact])
        session.flush()
        checkpoint = Checkpoint(
            project_id=project_id,
            training_run_id=run.id,
            step=120,
            token_count=24_000_000,
            artifact_id=artifact.id,
            metrics={"loss": 2.4},
            status="available",
        )
        session.add(checkpoint)
        session.commit()
        checkpoint_id = checkpoint.id

    checkpoint_model = client.post(
        f"/api/projects/{project_id}/models",
        json={
            "name": "CheckpointLM",
            "kind": "base",
            "source_type": "checkpoint",
            "source_checkpoint_id": checkpoint_id,
        },
    )
    assert checkpoint_model.status_code == 201, checkpoint_model.text
    checkpoint_body = checkpoint_model.json()
    assert checkpoint_body["source_checkpoint"]["id"] == checkpoint_id
    assert checkpoint_body["source_training_run"]["current_step"] == 120
    assert checkpoint_body["inference_source"]["type"] == "unsupported"


def test_model_registry_lifecycle_actions_are_audited_and_hide_archived_models(
    client: TestClient, tmp_path
) -> None:
    project = _project(client, tmp_path, "lifecycle")
    project_id = project["id"]
    first = _register_huggingface(client, project_id, name="LifecycleLM")
    second = _register_huggingface(
        client, project_id, name="LifecycleLM", revision="2" * 40
    )

    best = client.post(f"/api/models/{first['id']}/mark-best", json={"reason": "best loss"})
    assert best.status_code == 200
    assert best.json()["is_best"] is True

    unmarked = client.post(
        f"/api/models/{first['id']}/unmark-best", json={"reason": "selected by mistake"}
    )
    assert unmarked.status_code == 200
    assert unmarked.json()["is_best"] is False
    assert unmarked.json()["is_candidate"] is True
    assert unmarked.json()["status"] == "candidate"
    repeated_unmark = client.post(f"/api/models/{first['id']}/unmark-best")
    assert repeated_unmark.status_code == 409
    assert repeated_unmark.json()["detail"]["code"] == "model_not_best"
    best = client.post(f"/api/models/{first['id']}/mark-best")
    assert best.status_code == 200

    promoted_first = client.post(
        f"/api/models/{first['id']}/promote", json={"reason": "release gate passed"}
    )
    assert promoted_first.status_code == 200
    assert promoted_first.json()["is_official"] is True
    assert promoted_first.json()["is_candidate"] is False

    promoted_second = client.post(f"/api/models/{second['id']}/promote", json={})
    assert promoted_second.status_code == 200
    registry = client.get(f"/api/projects/{project_id}/models").json()["models"]
    first_after = next(model for model in registry if model["id"] == first["id"])
    second_after = next(model for model in registry if model["id"] == second["id"])
    assert first_after["status"] == "registered"
    assert first_after["is_official"] is False
    assert second_after["status"] == "official"
    assert second_after["is_official"] is True

    archived = client.post(
        f"/api/models/{first['id']}/archive", json={"reason": "superseded"}
    )
    assert archived.status_code == 200
    assert archived.json()["status"] == "archived"
    assert archived.json()["is_best"] is False

    visible_ids = {
        model["id"]
        for model in client.get(f"/api/projects/{project_id}/models").json()["models"]
    }
    assert first["id"] not in visible_ids
    all_models = client.get(
        f"/api/projects/{project_id}/models?include_archived=true"
    ).json()["models"]
    assert any(model["id"] == first["id"] for model in all_models)

    restored = client.post(f"/api/models/{first['id']}/restore")
    assert restored.status_code == 200
    assert restored.json()["status"] == "registered"
    candidate = client.post(f"/api/models/{first['id']}/mark-candidate")
    assert candidate.status_code == 200
    assert candidate.json()["status"] == "candidate"
    assert candidate.json()["is_candidate"] is True

    cannot_archive_official = client.post(
        f"/api/models/{second['id']}/archive", json={}
    )
    assert cannot_archive_official.status_code == 409
    assert cannot_archive_official.json()["detail"]["code"] == "official_model"

    with SessionLocal() as session:
        actions = session.scalars(
            select(AuditEvent.action).where(AuditEvent.project_id == project_id)
        ).all()
    assert {
        "model.register",
        "model.mark_best",
        "model.unmark_best",
        "model.promote",
        "model.archive",
        "model.restore",
        "model.mark_candidate",
    }.issubset(set(actions))


def test_model_registry_renames_display_name_without_changing_identity(
    client: TestClient, tmp_path
) -> None:
    project = _project(client, tmp_path, "rename")
    model = _register_huggingface(client, project["id"], name="RegistryLM")

    renamed = client.patch(
        f"/api/models/{model['id']}/name",
        json={"name": "RegistryLM · FineWeb中文4B", "reason": "描述训练阶段"},
    )

    assert renamed.status_code == 200, renamed.text
    body = renamed.json()
    assert body["id"] == model["id"]
    assert body["version"] == model["version"]
    assert body["name"] == "RegistryLM · FineWeb中文4B"
    with SessionLocal() as session:
        audit = session.scalar(
            select(AuditEvent).where(
                AuditEvent.subject_id == model["id"], AuditEvent.action == "model.rename"
            )
        )
        assert audit is not None
        assert audit.payload == {
            "previous_name": "RegistryLM",
            "name": "RegistryLM · FineWeb中文4B",
            "version": 1,
        }


def test_model_registry_rejects_cross_project_parent(client: TestClient, tmp_path) -> None:
    first_project = _project(client, tmp_path, "parent-a")
    second_project = _project(client, tmp_path, "parent-b")
    parent = _register_huggingface(client, first_project["id"], name="ParentLM")

    response = client.post(
        f"/api/projects/{second_project['id']}/models",
        json={
            "name": "ChildLM",
            "source_type": "huggingface",
            "huggingface_id": "example/child-lm",
            "huggingface_revision": "3" * 40,
            "parent_model_version_id": parent["id"],
        },
    )
    assert response.status_code == 404
    registry = client.get(f"/api/projects/{second_project['id']}/models").json()
    assert not any(model["name"] == "ChildLM" for model in registry["models"])


def test_model_registry_registers_openai_compatible_evaluation_endpoint(
    client: TestClient, tmp_path
) -> None:
    project = _project(client, tmp_path, "endpoint")
    response = client.post(
        f"/api/projects/{project['id']}/models",
        json={
            "name": "Ollama Qwen",
            "kind": "sft",
            "source_type": "openai_compatible",
            "endpoint_url": "http://localhost:11434/v1",
            "endpoint_model": "qwen3.5:0.8b",
            "endpoint_mode": "completions",
            "endpoint_tokenizer": "Qwen/Qwen3.5-0.8B",
            "endpoint_auth_env_var": "OLLAMA_EVAL_TOKEN",
            "endpoint_num_concurrent": 2,
            "endpoint_timeout_seconds": 600,
        },
    )

    assert response.status_code == 201, response.text
    model = response.json()
    assert model["artifact_id"] is None
    assert model["inference_source"] == {
        "type": "api",
        "display_name": "OpenAI 兼容评测端点",
        "endpoint_url": "http://localhost:11434/v1/completions",
        "endpoint_model": "qwen3.5:0.8b",
        "endpoint_mode": "completions",
        "capabilities": ["generate", "loglikelihood", "loglikelihood_rolling"],
        "reproducibility": "provider_managed",
    }
    runtime = model["metadata_json"]["lm_eval"]
    assert runtime["backend"] == "local-completions"
    assert runtime["auth_env_var"] == "OLLAMA_EVAL_TOKEN"
    assert "base_url=http://localhost:11434/v1/completions" in runtime["model_args"]
    assert "tokenizer=Qwen/Qwen3.5-0.8B" in runtime["model_args"]
    assert "OLLAMA_EVAL_TOKEN" not in runtime["model_args"]

    chat_response = client.post(
        f"/api/projects/{project['id']}/models",
        json={
            "name": "Cloud chat model",
            "source_type": "openai_compatible",
            "endpoint_url": "https://api.example.com/v1/chat/completions",
            "endpoint_model": "provider-chat-v1",
            "endpoint_mode": "chat_completions",
        },
    )
    assert chat_response.status_code == 201, chat_response.text
    assert chat_response.json()["metadata_json"]["lm_eval"]["backend"] == (
        "local-chat-completions"
    )

    embedded_credentials = client.post(
        f"/api/projects/{project['id']}/models",
        json={
            "name": "Unsafe endpoint",
            "source_type": "openai_compatible",
            "endpoint_url": "https://secret@example.com/v1",
            "endpoint_model": "cloud-model",
            "endpoint_mode": "chat_completions",
        },
    )
    assert embedded_credentials.status_code == 422
