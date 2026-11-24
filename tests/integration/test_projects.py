from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    EvalSuiteVersion,
    EvaluationRun,
    ModelVersion,
    PlanRevision,
    Project,
    ProjectDataBinding,
    QuantizationRun,
    Release,
    Task,
    TokenizerVersion,
    TrainingPlan,
    TrainingRun,
)
from sqlalchemy import select


def test_project_workspace_uri_lifecycle(client: TestClient, tmp_path) -> None:
    first_workspace = tmp_path / "example-lm"
    first_workspace.mkdir()
    workspace_uri = str(first_workspace)
    created_response = client.post(
        "/api/projects",
        json={
            "name": f"Workspace-{uuid4().hex[:8]}",
            "target_model_name": "ExampleLM",
            "workspace_uri": f"  {workspace_uri}  ",
        },
    )
    assert created_response.status_code == 201
    project = created_response.json()
    assert project["workspace_uri"] == workspace_uri

    dashboard = client.get(f"/api/projects/{project['id']}/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["project"]["workspace_uri"] == workspace_uri
    assert dashboard.json()["derived_lifecycle_stage"] == "data_preparation"
    assert "pipelines" not in dashboard.json()
    assert all(
        node["status"] == "not_started"
        for node in dashboard.json()["lifecycle_nodes"]
    )

    second_workspace = tmp_path / "example-lm-moved"
    second_workspace.mkdir()
    updated_uri = second_workspace.as_uri()
    updated = client.patch(f"/api/projects/{project['id']}", json={"workspace_uri": updated_uri})
    assert updated.status_code == 200
    assert updated.json()["workspace_uri"] == updated_uri

    copied = client.post(f"/api/projects/{project['id']}/copy")
    assert copied.status_code == 201
    assert copied.json()["workspace_uri"] == ""

    cleared = client.patch(f"/api/projects/{project['id']}", json={"workspace_uri": ""})
    assert cleared.status_code == 422


def test_project_dashboard_derives_each_lifecycle_node_from_real_versions(
    client: TestClient, tmp_path
) -> None:
    workspace = tmp_path / "derived-lifecycle"
    workspace.mkdir()
    created = client.post(
        "/api/projects",
        json={
            "name": f"Derived-Lifecycle-{uuid4().hex[:8]}",
            "target_model_name": "MicroLM",
            "workspace_uri": str(workspace),
        },
    )
    assert created.status_code == 201
    project_id = created.json()["id"]

    with SessionLocal() as session:
        global_project = session.scalar(
            select(Project).where(Project.system_kind == "global_data")
        )
        if global_project is None:
            global_workspace = tmp_path / "global-data"
            global_workspace.mkdir()
            global_project = Project(
                name=f"Global-Data-{uuid4().hex[:8]}",
                target_model_name="Global Data",
                workspace_uri=str(global_workspace),
                system_kind="global_data",
            )
            session.add(global_project)
            session.flush()
        dataset = Dataset(
            project_id=global_project.id,
            name=f"Global-Corpus-{uuid4().hex[:8]}",
            source_type="local",
            source_uri="file:///global/corpus.jsonl",
        )
        session.add(dataset)
        session.flush()
        data_version = DatasetVersion(
            dataset_id=dataset.id,
            version=3,
            state="ready",
            train_ready=True,
            document_count=1_024,
            token_estimate=8_000_000,
        )
        session.add(data_version)
        session.flush()
        dataset.current_version_id = data_version.id
        session.add(
            ProjectDataBinding(
                project_id=project_id,
                asset_type="dataset_version",
                asset_version_id=data_version.id,
                purpose="training",
            )
        )

        tokenizer = TokenizerVersion(
            project_id=project_id,
            name="MicroLM Tokenizer",
            version=4,
            algorithm="BPE",
            vocab_size=32_000,
            data_ref_type="dataset_version",
            data_ref_id=data_version.id,
            status="passed",
            locked=True,
            is_default=False,
        )
        session.add(tokenizer)

        plan = TrainingPlan(project_id=project_id, name="Pretraining")
        session.add(plan)
        session.flush()
        revision = PlanRevision(
            plan_id=plan.id,
            revision=2,
            stages=[{"name": "Gate 2", "type": "pretrain"}],
        )
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        training_task = Task(
            project_id=project_id,
            kind="train",
            title="Pretraining run",
            desired_state="cancelled",
            observed_state="cancelled",
        )
        session.add(training_task)
        session.flush()
        training_run = TrainingRun(
            project_id=project_id,
            plan_revision_id=revision.id,
            task_id=training_task.id,
            status="cancelled",
            current_stage="Gate 2",
            current_step=382,
            current_tokens=100_000_000,
            config_snapshot={"selected_stages": [{"name": "Gate 2", "type": "pretrain"}]},
        )
        session.add(training_run)
        session.flush()
        checkpoint = Checkpoint(
            project_id=project_id,
            training_run_id=training_run.id,
            step=382,
            token_count=100_000_000,
            status="available",
        )
        session.add(checkpoint)
        session.flush()
        model = ModelVersion(
            project_id=project_id,
            name="MicroLM",
            version=4,
            kind="base",
            source_checkpoint_id=checkpoint.id,
            tokenizer_version_id=tokenizer.id,
            status="candidate",
            is_candidate=True,
        )
        session.add(model)
        session.flush()

        suite = EvalSuiteVersion(
            project_id=project_id,
            name="Release Gate",
            version=2,
            datasets=[{"id": "benchmark-version"}],
        )
        evaluation_task = Task(
            project_id=project_id,
            kind="evaluate",
            title="Release evaluation",
            desired_state="succeeded",
            observed_state="succeeded",
        )
        session.add_all([suite, evaluation_task])
        session.flush()
        evaluation = EvaluationRun(
            project_id=project_id,
            task_id=evaluation_task.id,
            model_version_ids=[model.id],
            suite_version_ids=[suite.id],
            status="succeeded",
            summary={"overall": 0.71},
        )
        session.add(evaluation)

        quantization_task = Task(
            project_id=project_id,
            kind="quantize",
            title="Q4 quantization",
            desired_state="succeeded",
            observed_state="succeeded",
        )
        session.add(quantization_task)
        session.flush()
        quantization = QuantizationRun(
            project_id=project_id,
            model_version_id=model.id,
            task_id=quantization_task.id,
            method="GPTQ",
            format="Q4",
            status="succeeded",
        )
        session.add(quantization)
        session.flush()
        session.add(
            Release(
                project_id=project_id,
                name="MicroLM Local",
                version="2026.08",
                model_version_id=model.id,
                quantization_run_id=quantization.id,
                target="local",
                status="published",
            )
        )
        session.commit()

    response = client.get(f"/api/projects/{project_id}/dashboard")
    assert response.status_code == 200
    dashboard = response.json()
    nodes = {node["key"]: node for node in dashboard["lifecycle_nodes"]}

    assert dashboard["derived_lifecycle_stage"] == "pretraining"
    assert dashboard["project"]["lifecycle_stage"] == "pretraining"
    assert nodes["data_preparation"]["status"] == "ready"
    assert nodes["data_preparation"]["version"]["label"].endswith("· V3")
    assert nodes["tokenizer"]["status"] == "ready"
    assert nodes["tokenizer"]["version"]["label"] == "MicroLM Tokenizer · V4"
    assert nodes["tokenizer"]["version"]["metadata"]["is_default"] is False
    assert nodes["pretraining"]["status"] == "stopped"
    assert nodes["pretraining"]["raw_status"] == "cancelled"
    assert nodes["pretraining"]["version"]["label"] == "MicroLM · V4"
    assert any(item["kind"] == "checkpoint" for item in nodes["pretraining"]["versions"])
    assert nodes["post_training"]["status"] == "not_started"
    assert nodes["evaluation"]["status"] == "ready"
    assert nodes["evaluation"]["version"]["label"] == "Release Gate · V2"
    assert nodes["quantization"]["status"] == "ready"
    assert nodes["quantization"]["version"]["label"] == "GPTQ · Q4"
    assert nodes["release"]["status"] == "ready"
    assert nodes["release"]["version"]["label"] == "MicroLM Local · 2026.08"
    assert dashboard["current"]["data_asset"] == {
        "id": data_version.id,
        "asset_type": "dataset_version",
        "name": dataset.name,
        "version": 3,
        "status": "ready",
        "purposes": ["training"],
        "token_estimate": 8_000_000,
        "document_count": 1_024,
    }

    with SessionLocal() as session:
        stored_project = session.get(Project, project_id)
        assert stored_project is not None
        assert stored_project.lifecycle_stage == "data_preparation"


def test_project_dashboard_classifies_sft_and_dpo_as_post_training(
    client: TestClient, tmp_path
) -> None:
    workspace = tmp_path / "post-training-lifecycle"
    workspace.mkdir()
    created = client.post(
        "/api/projects",
        json={
            "name": f"Post-Training-Lifecycle-{uuid4().hex[:8]}",
            "target_model_name": "MicroLM",
            "workspace_uri": str(workspace),
        },
    )
    assert created.status_code == 201
    project_id = created.json()["id"]

    with SessionLocal() as session:
        plan = TrainingPlan(project_id=project_id, name="Alignment")
        session.add(plan)
        session.flush()
        stages = [
            {"name": "Instruction tuning", "objective": {"kind": "sft"}},
            {"name": "Preference tuning", "objective": {"kind": "dpo"}},
        ]
        revision = PlanRevision(plan_id=plan.id, revision=1, stages=stages)
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        task = Task(
            project_id=project_id,
            kind="train",
            title="Alignment run",
            desired_state="running",
            observed_state="running",
        )
        session.add(task)
        session.flush()
        session.add(
            TrainingRun(
                project_id=project_id,
                plan_revision_id=revision.id,
                task_id=task.id,
                status="running",
                current_stage="Preference tuning",
                config_snapshot={"selected_stages": stages},
            )
        )
        session.commit()

    response = client.get(f"/api/projects/{project_id}/dashboard")
    assert response.status_code == 200
    dashboard = response.json()
    nodes = {node["key"]: node for node in dashboard["lifecycle_nodes"]}
    assert dashboard["derived_lifecycle_stage"] == "post_training"
    assert nodes["pretraining"]["status"] == "not_started"
    assert nodes["post_training"]["status"] == "in_progress"
    assert nodes["post_training"]["version"]["kind"] == "training_run"
    assert nodes["post_training"]["version"]["metadata"]["objectives"] == ["dpo", "sft"]
    assert "目标：DPO / SFT" in nodes["post_training"]["detail"]
