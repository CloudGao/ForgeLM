from copy import deepcopy
from datetime import timedelta
from uuid import uuid4

from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.services.training_history import ensure_history_instances, source_run_id
from sqlalchemy import func, select
from test_training_upstream_lineage import attempt, dataset, project, save


def task(session, pid, name, kind="train", **kwargs):
    return save(
        session,
        m.Task(
            project_id=pid,
            kind=kind,
            title=name,
            adapter_id="builtin.command.training",
            launcher_id="builtin.local-process",
            observed_state="succeeded",
            **kwargs,
        ),
    )


def legacy_training(session, pid, data):
    plan = save(session, m.TrainingPlan(project_id=pid, name="历史 SFT"))
    revision = save(session, m.PlanRevision(plan_id=plan.id, revision=1))
    execution = task(session, pid, "历史 SFT", payload={"dataset_version_id": data.id})
    run = save(
        session,
        m.TrainingRun(
            project_id=pid,
            task_id=execution.id,
            plan_revision_id=revision.id,
            current_stage="SFT",
            status="succeeded",
            config_snapshot={
                "revision": {
                    "stages": [
                        {
                            "type": "pretrain",
                            "data_ref_id": data.id,
                            "data_ref_type": "dataset_version",
                        }
                    ]
                }
            },
        ),
    )
    execution.payload = {**execution.payload, "training_run_id": run.id}
    attempts = [
        save(
            session,
            m.TaskAttempt(
                task_id=execution.id, attempt_number=n, status="failed" if n == 1 else "succeeded"
            ),
        )
        for n in (1, 2)
    ]
    checkpoints = []
    for number, execution_attempt in enumerate(attempts, 1):
        artifact = save(
            session,
            m.Artifact(
                kind="checkpoint",
                uri=f"file:///{uuid4()}",
                content_hash=uuid4().hex,
                created_by_task_attempt_id=execution_attempt.id,
            ),
        )
        checkpoints.append(
            save(
                session,
                m.Checkpoint(
                    project_id=pid,
                    training_run_id=run.id,
                    artifact_id=artifact.id,
                    step=number * 10,
                ),
            )
        )
    return run, attempts, checkpoints


def test_legacy_training_has_one_stable_instance_real_attempts_and_version_graph(client):
    with SessionLocal() as session:
        pid = project(session).id
        first, cleaned = dataset(session, pid, 2)
        run, attempts, checkpoints = legacy_training(session, pid, cleaned)
        snapshot = deepcopy(run.config_snapshot)
        ensure_history_instances(session, pid)
        instance = next(
            i
            for i in session.scalars(select(m.ProjectPipelineInstance)).all()
            if source_run_id(i) == run.id
        )
        ensure_history_instances(session, pid)
        assert (
            session.scalar(
                select(func.count(m.ProjectPipelineInstance.id)).where(
                    m.ProjectPipelineInstance.project_id == pid
                )
            )
            == 1
        )
        ids = [a.id for a in attempts]
        cpids = [c.id for c in checkpoints]
        base = f"/api/projects/{pid}/training-instances"
        route = f"{base}/{instance.id}"
        run_id, instance_id, original_hash = run.id, instance.id, instance.content_hash
        session.commit()
    assert client.get(f"{base}/resolve-run/{run_id}").json() == {"instance_id": instance_id}
    listing = client.get(base).json()
    assert [i["id"] for i in listing] == [instance_id]
    assert listing[0]["run_count"] == 2
    assert [a["id"] for a in client.get(route + "/attempts").json()] == ids[::-1]
    for attempt_id, checkpoint_id in zip(ids, cpids, strict=True):
        response = client.get(route + f"/execution-graph?attempt_id={attempt_id}")
        assert response.status_code == 200, response.text
        graph = response.json()
        current = {
            a["asset_id"]: a
            for a in graph["artifacts"]
            if a["kind"] == "asset" and not a["context_only"]
        }
        assert checkpoint_id in current and cleaned.id in current
        assert not set(cpids) - {checkpoint_id} & set(current)
        assert any(a.get("asset_id") == first.id and a["context_only"] for a in graph["artifacts"])
        assert any(e["target"] == "training" and not e["context_only"] for e in graph["bindings"])
        assert any(e["source"] == "training" and not e["context_only"] for e in graph["bindings"])
        assert graph["execution_node_ids"] == []
    assert client.post(route + "/attempts", json={}).status_code == 409
    assert client.patch(route + "/archive", json={"archived": True}).status_code == 200
    assert client.get(base).json() == []
    with SessionLocal() as session:
        assert session.get(m.ProjectPipelineInstance, instance_id).content_hash == original_hash
        assert session.get(m.TrainingRun, run_id).config_snapshot == snapshot
        assert (
            session.scalar(
                select(func.count(m.TaskAttempt.id)).where(m.TaskAttempt.task_id == run.task_id)
            )
            == 2
        )


def test_recorded_workflow_expands_steps_with_real_versions_and_no_foreign_or_command_tasks(client):
    with SessionLocal() as session:
        pid = project(session).id
        source, cleaned = dataset(session, pid, 2)
        root = task(session, pid, "编排", "custom", created_at=m.utcnow())
        earlier = task(
            session,
            pid,
            "历史数据清洗",
            "data_process",
            created_at=root.created_at - timedelta(days=1),
            payload={
                "input_dataset_version_id": source.id,
                "output_dataset_version_id": cleaned.id,
            },
        )
        cleaned.task_id = earlier.id
        train, train_attempts, checkpoints = legacy_training(session, pid, cleaned)
        foreign = task(session, project(session).id, "foreign-secret")
        ignored = task(session, pid, "command-only")
        run, row = attempt(session, pid, [], node_id="workflow")
        row.task_id = root.id
        row.output_snapshot = {
            "workflow": {
                "clean": {"task_id": earlier.id},
                "train": {"task_id": train.task_id},
                "foreign": {"task_id": foreign.id},
            },
            "command": {"task_id": ignored.id},
        }
        root.result = deepcopy(row.output_snapshot)
        frozen = deepcopy(run.graph_snapshot)
        route = f"/api/projects/{pid}/training-instances/{run.instance_id}/execution-graph?attempt_id={run.id}&include_upstream=false"
        session.commit()
    response = client.get(route)
    assert response.status_code == 200, response.text
    graph = response.json()
    assert {r["task_id"] for r in graph["node_runs"]} == {root.id, earlier.id, train.task_id}
    assert graph["execution_node_ids"] == ["workflow"]
    assert len(graph["graph"]["nodes"]) == 3
    assert next(r for r in graph["node_runs"] if r["task_id"] == earlier.id)["config_snapshot"][
        "_context_only"
    ]
    clean_id = f"version:dataset_version:{cleaned.id}"
    assert any(
        e["source"] == f"task:{earlier.id}" and e["target"] == clean_id for e in graph["bindings"]
    )
    assert any(
        e["source"] == clean_id and e["target"] == f"task:{train.task_id}"
        for e in graph["bindings"]
    )
    assert "foreign-secret" not in response.text and "command-only" not in response.text
    with SessionLocal() as session:
        assert session.get(m.ProjectPipelineRun, run.id).graph_snapshot == frozen
        assert (
            session.scalar(
                select(func.count(m.ProjectPipelineNodeRun.id)).where(
                    m.ProjectPipelineNodeRun.pipeline_run_id == run.id
                )
            )
            == 1
        )


def test_missing_and_foreign_attempts_cannot_be_mixed_into_an_instance(client):
    with SessionLocal() as session:
        pid = project(session).id
        version = dataset(session, pid)[0]
        run, _, _ = legacy_training(session, pid, version)
        ensure_history_instances(session, pid)
        instance = next(
            i
            for i in session.scalars(select(m.ProjectPipelineInstance)).all()
            if source_run_id(i) == run.id
        )
        other = project(session).id
        session.commit()
    base = f"/api/projects/{pid}/training-instances/{instance.id}"
    assert client.get(base + f"/execution-graph?attempt_id={uuid4()}").status_code == 404
    assert client.get(base.replace(pid, other) + "/execution-graph").status_code == 404
    assert (
        client.get(f"/api/projects/{other}/training-instances/resolve-run/{run.id}").status_code
        == 404
    )


def test_default_expands_to_both_ends_without_shared_input_siblings(client):
    with SessionLocal() as session:
        pid = project(session).id
        versions = dataset(session, pid, 16)
        run, attempts, checkpoints = legacy_training(session, pid, versions[-1])
        sibling, _, _ = legacy_training(session, pid, versions[-1])
        child, _, child_checkpoints = legacy_training(session, pid, versions[-1])
        child.config_snapshot = {
            **child.config_snapshot,
            "resume_checkpoint": {"checkpoint_id": checkpoints[-1].id},
        }
        foreign, _, _ = legacy_training(session, project(session).id, versions[-1])
        foreign.config_snapshot = {
            **foreign.config_snapshot,
            "resume_checkpoint": {"checkpoint_id": checkpoints[-1].id},
        }
        model = save(
            session,
            m.ModelVersion(
                project_id=pid,
                name="child model",
                version=1,
                source_checkpoint_id=child_checkpoints[-1].id,
            ),
        )
        evaluation_task = task(session, pid, "descendant evaluation", "evaluate")
        evaluation = save(
            session,
            m.EvaluationRun(
                project_id=pid, task_id=evaluation_task.id, model_version_ids=[model.id]
            ),
        )
        ensure_history_instances(session, pid)
        instance = next(
            i
            for i in session.scalars(select(m.ProjectPipelineInstance))
            if source_run_id(i) == run.id
        )
        route = f"/api/projects/{pid}/training-instances/{instance.id}/execution-graph?attempt_id={attempts[-1].id}"
        frozen = deepcopy(instance.nodes)
        session.commit()
    response = client.get(route)
    assert response.status_code == 200, response.text
    graph = response.json()
    asset_ids = {a.get("asset_id") for a in graph["artifacts"]}
    assert {v.id for v in versions} <= asset_ids
    assert model.id in asset_ids and evaluation.id in asset_ids
    sources = {a["source"].get("training_run_id") for a in graph["artifacts"]}
    assert child.id in sources
    assert sibling.id not in sources and foreign.id not in sources
    assert not graph["truncated"]
    assert graph["center_node_ids"] == ["training"]
    with SessionLocal() as session:
        assert session.get(m.ProjectPipelineInstance, instance.id).nodes == frozen


def test_single_center_excludes_a_sibling_inside_the_same_frozen_graph(client):
    with SessionLocal() as session:
        pid = project(session).id
        shared = dataset(session, pid)[0]
        run, row = attempt(session, pid, [("dataset_version", shared.id)], node_id="center")
        sibling = {
            **deepcopy(run.graph_snapshot["nodes"][0]),
            "id": "sibling",
            "label": "unrelated sibling",
        }
        frozen = {**run.graph_snapshot, "nodes": [*run.graph_snapshot["nodes"], sibling]}
        run.graph_snapshot = frozen
        instance = session.get(m.ProjectPipelineInstance, run.instance_id)
        instance.nodes = frozen["nodes"]
        save(
            session,
            m.ProjectPipelineNodeRun(
                pipeline_run_id=run.id,
                node_id="sibling",
                node_type="custom",
                label="unrelated sibling",
                status="succeeded",
                config_snapshot=deepcopy(row.config_snapshot),
            ),
        )
        route = f"/api/projects/{pid}/training-instances/{run.instance_id}/execution-graph?attempt_id={run.id}&center_node_id=center"
        session.commit()
    response = client.get(route)
    assert response.status_code == 200, response.text
    assert [n["id"] for n in response.json()["graph"]["nodes"]] == ["center"]
    assert (
        client.get(route.replace("center_node_id=center", "center_node_id=missing")).status_code
        == 404
    )
