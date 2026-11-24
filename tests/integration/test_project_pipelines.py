from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Artifact,
    Checkpoint,
    CodeSource,
    CodeVersion,
    DataMixture,
    DataMixtureVersion,
    ModelVersion,
    PlanRevision,
    Task,
    TokenizerVersion,
    TrainingPlan,
    TrainingRun,
)


def test_explicit_pipeline_freezes_run_versions_and_uses_observed_progress(
    client: TestClient, tmp_path
) -> None:
    workspace = tmp_path / f"pipeline-{uuid4().hex[:8]}"
    workspace.mkdir()
    created = client.post(
        "/api/projects",
        json={
            "name": f"Pipeline-{uuid4().hex[:8]}",
            "target_model_name": "MicroLM-306M-v3",
            "workspace_uri": str(workspace),
        },
    )
    assert created.status_code == 201
    project_id = created.json()["id"]

    with SessionLocal() as session:
        mixture = DataMixture(
            project_id=project_id,
            name="tokenizer专用",
            description="Frozen pretraining mixture",
        )
        session.add(mixture)
        session.flush()
        mixture_version = DataMixtureVersion(
            mixture_id=mixture.id,
            version=3,
            status="ready",
            locked=True,
            resolved_document_count=2_929_792,
            resolved_token_estimate=157_903_449_905,
        )
        session.add(mixture_version)
        session.flush()
        mixture.current_version_id = mixture_version.id

        artifact = Artifact(
            kind="code_snapshot",
            uri=f"memory://pipeline-code/{uuid4().hex}",
            content_hash=uuid4().hex + uuid4().hex,
            size_bytes=1_024,
        )
        source = CodeSource(
            project_id=project_id,
            name="MicroLM-306M-v3",
            source_type="git",
            source_uri="https://example.invalid/microlm.git",
            status="ready",
        )
        session.add_all([artifact, source])
        session.flush()
        code_version = CodeVersion(
            source_id=source.id,
            version=22,
            state="ready",
            artifact_id=artifact.id,
            git_commit="8d9f0a1b2c3d4e5f",
        )
        session.add(code_version)
        session.flush()
        source.current_version_id = code_version.id

        tokenizer = TokenizerVersion(
            project_id=project_id,
            name="实验1",
            version=4,
            algorithm="BPE",
            vocab_size=32_000,
            data_ref_type="mixture_version",
            data_ref_id=mixture_version.id,
            status="passed",
            locked=True,
        )
        plan = TrainingPlan(
            project_id=project_id,
            name="MicroLM-306M-v3 基座预训练",
            status="running",
        )
        session.add_all([tokenizer, plan])
        session.flush()
        stages = [
            {
                "name": "Gate 2",
                "type": "pretrain",
                "data_ref_id": mixture_version.id,
                "cumulative_target_tokens": 6_000_000_000,
                "target_steps": 22_507,
            }
        ]
        revision = PlanRevision(
            plan_id=plan.id,
            revision=25,
            stages=stages,
            tokenizer_version_id=tokenizer.id,
            code_version_id=code_version.id,
            locked=True,
        )
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        task = Task(
            project_id=project_id,
            kind="train",
            title="MicroLM Gate 2",
            desired_state="cancelled",
            observed_state="cancelled",
            progress=0.5116251118644067,
            current_phase="Gate 2",
        )
        session.add(task)
        session.flush()
        run = TrainingRun(
            project_id=project_id,
            plan_revision_id=revision.id,
            task_id=task.id,
            status="running",
            current_stage="Gate 2",
            current_step=11_515,
            current_tokens=3_118_588_160,
            config_snapshot={"selected_stages": stages},
        )
        session.add(run)
        session.flush()
        checkpoint = Checkpoint(
            project_id=project_id,
            training_run_id=run.id,
            step=11_515,
            token_count=3_118_588_160,
            status="available",
        )
        session.add(checkpoint)
        session.flush()
        model = ModelVersion(
            project_id=project_id,
            name="MicroLM-306M-v3",
            version=5,
            kind="base",
            source_checkpoint_id=checkpoint.id,
            status="candidate",
            is_candidate=True,
        )
        session.add(model)
        session.flush()
        session.commit()
        task_id = task.id
        run_id = run.id
        revision_id = revision.id
        tokenizer_id = tokenizer.id
        code_version_id = code_version.id
        mixture_version_id = mixture_version.id
        model_id = model.id

    response = client.post(
        f"/api/projects/{project_id}/pipelines",
        json={
            "name": "MicroLM-306M-v3 基座预训练",
            "objective": "pretraining",
            "training_run_id": run_id,
        },
    )
    assert response.status_code == 201, response.text
    pipeline = response.json()
    assert pipeline["revision"]["revision"] == 1
    assert pipeline["revision"]["training_run_id"] == run_id
    assert pipeline["revision"]["plan_revision_id"] == revision_id
    assert pipeline["revision"]["tokenizer_version_id"] == tokenizer_id
    assert pipeline["revision"]["code_version_id"] == code_version_id
    assert pipeline["revision"]["data_refs"] == [
        {
            "asset_type": "mixture_version",
            "asset_version_id": mixture_version_id,
            "purpose": "training",
        }
    ]
    assert pipeline["status"] == "stopped"
    assert pipeline["current_node"] == "training"
    assert pipeline["progress"] == 51.16

    definitions_response = client.get(
        f"/api/projects/{project_id}/pipeline-definitions"
    )
    assert definitions_response.status_code == 200
    compatibility_definition = next(
        item
        for item in definitions_response.json()
        if item["id"] == pipeline["id"]
    )
    assert compatibility_definition["definition_kind"] == "legacy_compatibility"
    assert compatibility_definition["current_revision"]["validation_report"] == {
        "valid": True,
        "errors": [],
        "warnings": [],
        "node_count": 8,
        "edge_count": 7,
    }
    expected_dependency_edges = {
        ("data", "training"),
        ("tokenizer", "training"),
        ("code", "training"),
        ("training", "model"),
        ("model", "evaluation"),
        ("model", "quantization"),
        ("quantization", "release"),
    }
    assert {
        (edge["source"], edge["target"])
        for edge in compatibility_definition["current_revision"]["edges"]
    } == expected_dependency_edges

    nodes = {node["key"]: node for node in pipeline["nodes"]}
    assert nodes["data"]["status"] == "ready"
    assert nodes["data"]["version"]["label"] == "tokenizer专用 · V3"
    assert nodes["tokenizer"]["status"] == "ready"
    assert nodes["tokenizer"]["version"]["label"] == "实验1 · V4"
    assert nodes["code"]["status"] == "ready"
    assert nodes["code"]["version"]["label"] == "MicroLM-306M-v3 · V22"
    assert nodes["training"]["raw_status"] == "cancelled"
    assert nodes["training"]["metrics"]["current_step"] == 11_515
    assert nodes["training"]["metrics"]["target_tokens"] == 6_000_000_000
    assert nodes["model"]["status"] == "ready"
    assert [item["id"] for item in nodes["model"]["versions"]] == [model_id]

    instances_response = client.get(
        f"/api/projects/{project_id}/pipeline-instances"
    )
    assert instances_response.status_code == 200
    instances = instances_response.json()
    assert len(instances) == 1
    imported_instance = instances[0]
    assert imported_instance["pipeline_id"] == pipeline["id"]
    assert imported_instance["instance_kind"] == "imported"
    assert imported_instance["execution_owner"] == "domain_task"
    assert imported_instance["can_run"] is False
    assert imported_instance["source_refs"] == {
        "training_run_id": run_id,
        "task_id": task_id,
    }
    assert {
        (edge["source"], edge["target"]) for edge in imported_instance["edges"]
    } == expected_dependency_edges
    assert imported_instance["latest_run"]["run_kind"] == "imported"
    assert imported_instance["latest_run"]["status"] == "cancelled"

    imported_runs_response = client.get(
        f"/api/projects/{project_id}/pipeline-instances/{imported_instance['id']}/runs"
    )
    assert imported_runs_response.status_code == 200
    imported_runs = imported_runs_response.json()
    assert len(imported_runs) == 1
    imported_run = imported_runs[0]
    assert imported_run["run_kind"] == "imported"
    assert imported_run["execution_owner"] == "domain_task"
    assert imported_run["status"] == "cancelled"
    assert imported_run["node_summary"]["source_progress"] == 0.5116
    assert {
        (edge["source"], edge["target"])
        for edge in imported_run["graph_snapshot"]["edges"]
    } == expected_dependency_edges
    imported_nodes = {item["node_id"]: item for item in imported_run["node_runs"]}
    assert imported_nodes["data"]["status"] == "succeeded"
    assert imported_nodes["training"]["status"] == "cancelled"
    assert imported_nodes["training"]["task_id"] == task_id
    assert imported_nodes["model"]["status"] == "succeeded"
    assert imported_nodes["evaluation"]["status"] == "skipped"
    assert imported_run["lineage"]["stored"] is True
    assert imported_run["lineage"]["execution_count"] == 8
    training_lineage = imported_nodes["training"]["lineage"]
    assert training_lineage["execution_identity"]["training_run_id"] == run_id
    assert training_lineage["execution_identity"]["task_id"] == task_id
    assert {
        (item["asset_type"], item["asset_id"])
        for item in training_lineage["consumed_assets"]
    } >= {
        ("mixture_version", mixture_version_id),
        ("tokenizer_version", tokenizer_id),
        ("code_version", code_version_id),
        ("plan_revision", revision_id),
    }

    lineage_response = client.get(
        f"/api/projects/{project_id}/pipeline-runs/{imported_run['id']}/lineage"
    )
    assert lineage_response.status_code == 200
    assert lineage_response.json()["execution_count"] == 8
    mixture_lineage_response = client.get(
        f"/api/projects/{project_id}/lineage/assets/mixture_version/{mixture_version_id}"
    )
    assert mixture_lineage_response.status_code == 200
    assert {
        item["node_id"]
        for item in mixture_lineage_response.json()["consumers"]
    } >= {"data", "training"}
    model_lineage_response = client.get(
        f"/api/projects/{project_id}/lineage/assets/model_version/{model_id}"
    )
    assert model_lineage_response.status_code == 200
    assert [
        item["node_id"] for item in model_lineage_response.json()["producers"]
    ] == ["model"]

    rerun = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{imported_instance['id']}/runs",
        json={"inputs": {}},
    )
    assert rerun.status_code == 409
    assert rerun.json()["detail"]["code"] == "imported_pipeline_read_only"
    cancel_imported = client.post(
        f"/api/projects/{project_id}/pipeline-instances/{imported_instance['id']}/runs/{imported_run['id']}/cancel"
    )
    assert cancel_imported.status_code == 409
    assert cancel_imported.json()["detail"]["code"] == "imported_pipeline_read_only"

    duplicate = client.post(
        f"/api/projects/{project_id}/pipelines",
        json={
            "name": "another pipeline",
            "objective": "pretraining",
            "training_run_id": run_id,
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "training_run_already_mapped"

    with SessionLocal() as session:
        unrelated_task = Task(
            project_id=project_id,
            kind="train",
            title="Unrelated completed run",
            desired_state="succeeded",
            observed_state="succeeded",
            progress=1,
        )
        session.add(unrelated_task)
        session.flush()
        unrelated_run = TrainingRun(
            project_id=project_id,
            plan_revision_id=revision_id,
            task_id=unrelated_task.id,
            status="succeeded",
            current_stage="Gate 3",
            current_step=22_507,
            current_tokens=6_000_000_000,
            config_snapshot={"selected_stages": stages},
        )
        session.add(unrelated_run)
        session.flush()
        unrelated_checkpoint = Checkpoint(
            project_id=project_id,
            training_run_id=unrelated_run.id,
            step=22_507,
            token_count=6_000_000_000,
            status="available",
        )
        session.add(unrelated_checkpoint)
        session.flush()
        session.add(
            ModelVersion(
                project_id=project_id,
                name="Unrelated MicroLM",
                version=1,
                kind="base",
                source_checkpoint_id=unrelated_checkpoint.id,
                status="candidate",
                is_candidate=True,
            )
        )
        session.commit()

    compatibility_pipelines = client.get(f"/api/projects/{project_id}/pipelines")
    assert compatibility_pipelines.status_code == 200
    refreshed_pipeline = next(
        item
        for item in compatibility_pipelines.json()
        if item["id"] == pipeline["id"]
    )
    refreshed_nodes = {node["key"]: node for node in refreshed_pipeline["nodes"]}
    assert refreshed_nodes["training"]["raw_status"] == "cancelled"
    assert refreshed_nodes["model"]["status"] == "ready"
    assert [item["id"] for item in refreshed_nodes["model"]["versions"]] == [model_id]

    options = client.get(f"/api/projects/{project_id}/pipeline-options")
    assert options.status_code == 200
    run_option = next(item for item in options.json()["training_runs"] if item["id"] == run_id)
    assert run_option["progress"] == 51.16
    assert run_option["data_refs"][0]["asset_version_id"] == mixture_version_id
    assert run_option["mapped_pipeline_id"] == pipeline["id"]
    assert run_option["mapped_pipeline_name"] == "MicroLM-306M-v3 基座预训练"
