from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def create_project(client: TestClient, tmp_path) -> dict:
    workspace = tmp_path / "model-spec-project"
    workspace.mkdir()
    response = client.post(
        "/api/projects",
        json={
            "name": f"ModelSpec-{uuid4().hex[:8]}",
            "target_model_name": "VisualLM",
            "workspace_uri": str(workspace),
        },
    )
    assert response.status_code == 201
    return response.json()


def test_visual_model_specification_lifecycle_and_training_binding(
    client: TestClient, tmp_path
) -> None:
    project = create_project(client, tmp_path)
    catalog = client.get("/api/model-architectures")
    assert catalog.status_code == 200
    assert {item["id"] for item in catalog.json()["architectures"]} >= {
        "llama_dense",
        "mistral",
        "qwen2",
        "gemma2",
        "phi3",
        "gpt_neox",
        "mixtral_moe",
        "qwen2_moe",
        "deepseek_v3",
        "custom_code",
    }

    payload = {
        "name": "Visual Dense 1B",
        "description": "Created from typed controls",
        "source_type": "standard",
        "architecture_family_id": "llama_dense",
        "preset_id": None,
        "config": {},
        "initialization": {"type": "random"},
    }
    preview = client.post(
        f"/api/projects/{project['id']}/model-specifications/resolve", json=payload
    )
    assert preview.status_code == 200
    assert preview.json()["status"] == "ready"
    assert "LlamaConfig" in preview.json()["generated_code"]

    created = client.post(f"/api/projects/{project['id']}/model-specifications", json=payload)
    assert created.status_code == 201
    specification = created.json()
    revision = specification["current_revision"]
    assert revision["revision"] == 1
    assert revision["generated_json"]["config_signature"] == revision["config_signature"]

    revised = client.post(
        f"/api/model-specifications/{specification['id']}/revisions",
        json={
            "source_type": "standard",
            "architecture_family_id": "llama_dense",
            "preset_id": None,
            "config": {"max_position_embeddings": 8192},
            "initialization": {"type": "random"},
        },
    )
    assert revised.status_code == 201
    specification = revised.json()
    revision = specification["current_revision"]
    assert revision["revision"] == 2
    assert len(specification["revisions"]) == 2
    assert revision["resolved_config"]["max_position_embeddings"] == 8192

    plan = client.post(
        f"/api/projects/{project['id']}/training-plans",
        json={
            "name": "Visual model training",
            "stages": [{"name": "Stage 0", "type": "pretrain", "target_tokens": 1000}],
            "model_specification_revision_id": revision["id"],
            "engine_config": {"executable": "python", "args": ["train.py"]},
        },
    )
    assert plan.status_code == 201
    plan_revision = plan.json()["current_revision"]
    assert plan_revision["model_specification_revision_id"] == revision["id"]
    assert plan_revision["input_model_ref"]["type"] == "model_specification"
    assert plan_revision["input_model_ref"]["vocab_size"] == 32000

    preflight = client.get(f"/api/training-plans/{plan.json()['id']}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["estimates"]["parameters"]["method"] == "architecture_formula_v2"
