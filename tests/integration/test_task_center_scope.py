from datetime import timedelta
from uuid import uuid4

import pytest
from llm_platform.database import SessionLocal
from llm_platform.models import (
    PlanRevision,
    ProjectPipelineInstance,
    ProjectPipelineNodeRun,
    ProjectPipelineRun,
    Task,
    TrainingPlan,
    TrainingRun,
    utcnow,
)


def project(client, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    response = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "TestLM", "workspace_uri": str(tmp_path)},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def task(session, pid, kind, index=0, state="succeeded"):
    created = utcnow() + timedelta(seconds=index)
    row = Task(
        project_id=pid, kind=kind, title=f"{kind}-{index}",
        adapter_id="builtin.command", launcher_id="builtin.local-process",
        observed_state=state, desired_state="running", created_at=created,
        started_at=created, ended_at=created + timedelta(seconds=1),
    )
    session.add(row)
    session.flush()
    return row


def link_node(session, pid, tid):
    instance = ProjectPipelineInstance(project_id=pid, name=uuid4().hex, status="ready")
    session.add(instance)
    session.flush()
    run = ProjectPipelineRun(project_id=pid, instance_id=instance.id, status="succeeded")
    session.add(run)
    session.flush()
    session.add(ProjectPipelineNodeRun(
        pipeline_run_id=run.id, node_id="node", node_type="custom",
        label="Production node", task_id=tid, status="succeeded",
    ))


def test_pipeline_scope_filters_before_pagination_and_preserves_detail(client, tmp_path):
    pid = project(client, tmp_path)
    other_pid = project(client, tmp_path / "other")
    with SessionLocal() as session:
        production_ids = []
        for index, kind in enumerate(("train", "suite_node", "data_generate", "future_node_kind")):
            row = task(session, pid, kind, index, "failed" if index == 0 else "succeeded")
            production_ids.append(row.id)
            link_node(session, pid, row.id)
        # A resumed task may belong to multiple attempts but occupies one row.
        link_node(session, pid, production_ids[0])
        hidden_ids = []
        for index, kind in enumerate((
            "inference", "data_query", "hf_repository_scan", "hf_download_plan",
            "plugin_install", "suite_node", "train", "future_background_kind",
        ), start=20):
            row = task(session, pid, kind, index)
            # Payload hints alone must never establish execution ownership.
            row.payload = {"pipeline_node_run_id": "fake", "pipeline_instance_id": "fake"}
            hidden_ids.append(row.id)
        # A mismatched project relationship cannot make an unrelated task visible.
        link_node(session, other_pid, hidden_ids[-1])
        other = task(session, other_pid, "suite_node", 100)
        link_node(session, other_pid, other.id)
        session.commit()

    params = {"project_id": pid, "scope": "pipeline", "limit": 2}
    first = client.get("/api/tasks", params=params)
    second = client.get("/api/tasks", params={**params, "offset": 2})
    assert first.status_code == second.status_code == 200
    assert [row["id"] for row in first.json() + second.json()] == production_ids[::-1]
    assert client.get("/api/tasks", params={**params, "offset": 4}).json() == []
    assert [row["id"] for row in client.get(
        "/api/tasks", params={**params, "kind": "train", "status": "failed"},
    ).json()] == [production_ids[0]]
    assert client.get("/api/tasks", params={
        **params, "kind": "train", "exclude_kind": "train",
    }).json() == []
    all_rows = client.get("/api/tasks", params={"project_id": pid}).json()
    assert len(all_rows) == len(production_ids) + len(hidden_ids)
    assert client.get(f"/api/tasks/{hidden_ids[0]}").json()["kind"] == "inference"
    assert client.get("/api/tasks", params={"scope": "typo"}).status_code == 422


@pytest.mark.parametrize("metadata_key,run_key", [
    ("__training_history", "training_run_id"),
    ("__forgelm", "source_training_run_id"),
])
def test_pipeline_scope_retains_imported_training_instances(client, tmp_path, metadata_key, run_key):
    pid = project(client, tmp_path)
    other_pid = project(client, tmp_path / "other")
    with SessionLocal() as session:
        plan = TrainingPlan(project_id=pid, name="Historical training")
        session.add(plan)
        session.flush()
        revision = PlanRevision(plan_id=plan.id, revision=1)
        session.add(revision)
        session.flush()
        visible_id = None
        for index, owner_pid in enumerate((pid, other_pid)):
            row = task(session, pid, "train", index)
            run = TrainingRun(project_id=pid, plan_revision_id=revision.id, task_id=row.id)
            session.add(run)
            session.flush()
            session.add(ProjectPipelineInstance(
                project_id=owner_pid, name=uuid4().hex, status="archived",
                variables={metadata_key: {run_key: run.id}},
            ))
            if index == 0:
                visible_id = row.id
        session.commit()
    result = client.get("/api/tasks", params={"project_id": pid, "scope": "pipeline"})
    assert result.status_code == 200
    assert [row["id"] for row in result.json()] == [visible_id]
