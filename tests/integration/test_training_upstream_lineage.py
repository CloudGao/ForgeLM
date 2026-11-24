from copy import deepcopy
from uuid import uuid4

from llm_platform import models as m
from llm_platform.database import SessionLocal
from llm_platform.services.training_upstream_lineage import version_refs
from sqlalchemy import func, select


def save(session, record):
    session.add(record)
    session.flush()
    return record


def project(session, name="project"):
    return save(session, m.Project(name=f"{name}-{uuid4().hex}", target_model_name="TestLM"))


def dataset(session, pid, count=1):
    data = save(session, m.Dataset(project_id=pid, name=uuid4().hex, source_type="local"))
    versions = []
    for number in range(count):
        versions.append(
            save(
                session,
                m.DatasetVersion(
                    dataset_id=data.id,
                    version=number + 1,
                    parent_version_id=versions[-1].id if versions else None,
                ),
            )
        )
    return versions


def attempt(session, pid, refs, *, node_id="train"):
    config = {
        "data_refs": [
            {"asset_type": kind, "asset_version_id": record_id} for kind, record_id in refs
        ]
    }
    nodes = [
        {
            "id": node_id,
            "type": "custom",
            "label": "当前训练",
            "position": {"x": 0, "y": 0},
            "config": config,
        }
    ]
    instance = save(
        session,
        m.ProjectPipelineInstance(
            project_id=pid,
            name=uuid4().hex,
            nodes=nodes,
            edges=[],
            locked=True,
            status="ready",
            content_hash=uuid4().hex,
        ),
    )
    run = save(
        session,
        m.ProjectPipelineRun(
            project_id=pid,
            instance_id=instance.id,
            status="succeeded",
            graph_snapshot={"nodes": deepcopy(nodes), "edges": [], "variables": {}, "viewport": {}},
        ),
    )
    row = save(
        session,
        m.ProjectPipelineNodeRun(
            pipeline_run_id=run.id,
            node_id=node_id,
            node_type="custom",
            label="当前训练",
            status="succeeded",
            config_snapshot=config,
        ),
    )
    return run, row


def endpoint(run):
    return f"/api/projects/{run.project_id}/training-instances/{run.instance_id}/attempts/{run.id}/upstream-lineage"


def assets(view):
    return {n["asset_id"]: n for n in view["nodes"] if n["kind"] == "asset"}


def test_multihop_mixture_lineage_is_read_only_and_shared_ancestors_are_unique(client):
    with SessionLocal() as session:
        pid = project(session).id
        first, second, third = dataset(session, pid, 3)
        mix = save(session, m.DataMixture(project_id=pid, name="mix"))
        version = save(
            session,
            m.DataMixtureVersion(
                mixture_id=mix.id,
                version=1,
                entries=[{"dataset_version_id": third.id}, {"dataset_version_id": first.id}],
                status="ready",
            ),
        )
        run, row = attempt(
            session, pid, [("mixture_version", version.id), ("dataset_version", third.id)]
        )
        route, run_id = endpoint(run), run.id
        snapshot, config, instance_hash = (
            deepcopy(run.graph_snapshot),
            deepcopy(row.config_snapshot),
            session.get(m.ProjectPipelineInstance, run.instance_id).content_hash,
        )
        expected = {first.id, second.id, third.id, version.id}
        counts = [
            session.scalar(select(func.count(model.id)))
            for model in (m.Task, m.ProjectPipelineRun, m.LineageExecution)
        ]
        session.commit()
    response = client.get(route)
    assert response.status_code == 200, response.text
    view = response.json()
    assert set(assets(view)) == expected
    assert len(view["nodes"]) == len({n["id"] for n in view["nodes"]})
    assert all(n["context_only"] and n["read_only"] for n in view["nodes"])
    assert all(e["context_only"] for e in view["edges"])
    assert any(e["target"] == "train" for e in view["edges"])
    assert not view["truncated"]
    with SessionLocal() as session:
        current = session.get(m.ProjectPipelineRun, run_id)
        assert current.graph_snapshot == snapshot
        assert (
            session.scalar(
                select(m.ProjectPipelineNodeRun).where(
                    m.ProjectPipelineNodeRun.pipeline_run_id == run_id
                )
            ).config_snapshot
            == config
        )
        assert (
            session.get(m.ProjectPipelineInstance, current.instance_id).content_hash
            == instance_hash
        )
        assert [
            session.scalar(select(func.count(model.id)))
            for model in (m.Task, m.ProjectPipelineRun, m.LineageExecution)
        ] == counts


def test_model_checkpoint_training_history_reaches_its_frozen_dataset(client):
    with SessionLocal() as session:
        pid = project(session).id
        data = dataset(session, pid)[0]
        task = save(
            session,
            m.Task(
                project_id=pid,
                kind="train",
                title="历史训练",
                adapter_id="builtin.command.training",
                launcher_id="builtin.local-process",
                observed_state="succeeded",
            ),
        )
        plan = save(session, m.TrainingPlan(project_id=pid, name="historical"))
        revision = save(session, m.PlanRevision(plan_id=plan.id, revision=1))
        train = save(
            session,
            m.TrainingRun(
                project_id=pid,
                task_id=task.id,
                plan_revision_id=revision.id,
                status="succeeded",
                config_snapshot={
                    "input_data": {"asset_type": "dataset_version", "asset_id": data.id}
                },
            ),
        )
        checkpoint = save(session, m.Checkpoint(project_id=pid, training_run_id=train.id, step=30))
        model = save(
            session,
            m.ModelVersion(
                project_id=pid, name="Model", version=7, source_checkpoint_id=checkpoint.id
            ),
        )
        run, _ = attempt(session, pid, [("model_version", model.id)])
        route, expected, train_id = endpoint(run), {model.id, checkpoint.id, data.id}, train.id
        session.commit()
    view = client.get(route).json()
    assert set(assets(view)) == expected
    assert {n["label"] for n in view["nodes"] if n["kind"] == "process"} == {"模型登记", "模型训练"}
    history = next(n for n in view["nodes"] if n["source"].get("training_run_id") == train_id)
    assert history["source"]["training_name"] == "历史训练"
    assert "attempt_id" not in history["source"]


def test_actual_producer_ledger_takes_precedence_over_domain_fallback(client):
    with SessionLocal() as session:
        pid = project(session).id
        first, output = dataset(session, pid, 2)
        previous, old_node = attempt(session, pid, [], node_id="old-clean")
        fact = save(
            session,
            m.LineageExecution(
                project_id=pid,
                pipeline_run_id=previous.id,
                node_run_id=old_node.id,
                node_id=old_node.node_id,
                node_type="clean",
                status="succeeded",
                execution_owner="pipeline_scheduler",
                content_hash=uuid4().hex,
            ),
        )
        for kind, record in [("consumed", first), ("produced", output)]:
            save(
                session,
                m.LineageAssetRelation(
                    project_id=pid,
                    execution_id=fact.id,
                    relation_kind=kind,
                    asset_type="dataset_version",
                    asset_id=record.id,
                    role=kind,
                    label="registered version",
                ),
            )
        run, _ = attempt(session, pid, [("dataset_version", output.id)])
        route, old_id = endpoint(run), previous.id
        old_instance_id = previous.instance_id
        old_instance_name = session.get(m.ProjectPipelineInstance, previous.instance_id).name
        session.commit()
    view = client.get(route).json()
    processes = [n for n in view["nodes"] if n["kind"] == "process"]
    assert len(processes) == 1
    assert processes[0]["source"]["attempt_id"] == old_id
    assert processes[0]["source"]["instance_id"] == old_instance_id
    assert processes[0]["source"]["instance_name"] == old_instance_name
    assert all(n["label"] != "数据派生" for n in processes)


def test_global_bound_version_ancestors_are_visible_but_foreign_models_are_not(client):
    with SessionLocal() as session:
        owner, consumer = project(session, "owner"), project(session, "consumer")
        first, derived = dataset(session, owner.id, 2)
        save(
            session,
            m.ProjectDataBinding(
                project_id=consumer.id,
                asset_type="dataset_version",
                asset_version_id=derived.id,
                purpose="training",
            ),
        )
        secret = save(
            session, m.ModelVersion(project_id=owner.id, name="private-model-name", version=1)
        )
        hidden_data = dataset(session, owner.id)[0]
        run, _ = attempt(
            session,
            consumer.id,
            [
                ("dataset_version", derived.id),
                ("dataset_version", hidden_data.id),
                ("model_version", secret.id),
            ],
        )
        route, expected, hidden_ids = (
            endpoint(run),
            {first.id, derived.id},
            {secret.id, hidden_data.id},
        )
        session.commit()
    response = client.get(route)
    assert set(assets(response.json())) == expected
    assert not any(record_id in response.text for record_id in hidden_ids)
    assert "private-model-name" not in response.text
    assert response.json()["warnings"]
    assert client.get(route.replace(run.project_id, owner.id)).status_code == 404


def test_missing_cycle_and_depth_limits_are_explicit(client):
    with SessionLocal() as session:
        pid = project(session).id
        first, second, third = dataset(session, pid, 3)
        first.parent_version_id = third.id
        missing_id = str(uuid4())
        run, _ = attempt(
            session, pid, [("dataset_version", third.id), ("model_version", missing_id)]
        )
        route = endpoint(run)
        session.commit()
    view = client.get(route).json()
    assert "缺失" in assets(view)[missing_id]["note"]
    assert any("循环" in warning for warning in view["warnings"])
    limited = client.get(route + "?max_depth=1").json()
    assert limited["truncated"] and len(limited["nodes"]) < len(view["nodes"])
    assert client.get(route + "?max_depth=100").status_code == 422


def test_current_attempt_outputs_are_not_added_as_historical_roots(client):
    with SessionLocal() as session:
        pid = project(session).id
        data = dataset(session, pid)[0]
        run, row = attempt(session, pid, [("dataset_version", data.id)])
        fact = save(
            session,
            m.LineageExecution(
                project_id=pid,
                pipeline_run_id=run.id,
                node_run_id=row.id,
                node_id=row.node_id,
                node_type="custom",
                status="succeeded",
                execution_owner="pipeline_scheduler",
                content_hash=uuid4().hex,
            ),
        )
        save(
            session,
            m.LineageAssetRelation(
                project_id=pid,
                execution_id=fact.id,
                relation_kind="produced",
                asset_type="dataset_version",
                asset_id=data.id,
                role="output",
                label="current output",
            ),
        )
        route = endpoint(run)
        session.commit()
    assert client.get(route).json()["nodes"] == []


def test_supervision_mapping_and_mixture_follow_exact_versions(client):
    with SessionLocal() as session:
        pid = project(session).id
        raw = dataset(session, pid)[0]
        supervision = save(session, m.SupervisionDataset(project_id=pid, name="supervision"))
        mapped = save(
            session,
            m.SupervisionDatasetVersion(
                supervision_dataset_id=supervision.id,
                version=1,
                dataset_version_id=raw.id,
                kind="sft",
                identity_hash=uuid4().hex,
            ),
        )
        mixture = save(session, m.SupervisionMixture(project_id=pid, name="sft mixture"))
        mixed = save(
            session,
            m.SupervisionMixtureVersion(
                mixture_id=mixture.id,
                version=1,
                kind="sft",
                identity_hash=uuid4().hex,
                entries=[{"supervision_dataset_version_id": mapped.id}],
            ),
        )
        run, row = attempt(session, pid, [])
        row.config_snapshot = {
            "training_spec": {
                "stages": [{"type": "sft", "objective": {"mixture_version_id": mixed.id}}]
            }
        }
        route, expected = endpoint(run), {raw.id, mapped.id, mixed.id}
        session.commit()
    assert set(assets(client.get(route).json())) == expected


def test_explicit_version_parser_ignores_task_results_and_preserves_sft_type():
    value = {
        "stages": [{"type": "sft", "objective": {"mixture_version_id": "sft"}}],
        "task_snapshot": {"result": {"model_version_id": "output"}},
    }
    assert version_refs(value) == [("supervision_mixture_version", "sft")]


def test_processing_run_exposes_recipe_identity_and_exact_input(client):
    with SessionLocal() as session:
        pid = project(session).id
        first, second = dataset(session, pid, 2)
        recipe = save(session, m.DataProcessingRecipe(project_id=pid, name="cleanup"))
        revision = save(session, m.DataProcessingRecipeVersion(recipe_id=recipe.id, version=1))
        processing = save(
            session,
            m.DataProcessingRun(
                project_id=pid,
                recipe_version_id=revision.id,
                input_dataset_version_id=first.id,
                output_dataset_version_id=second.id,
                output_tier="silver",
                status="succeeded",
            ),
        )
        second.processing_run_id = processing.id
        run, _ = attempt(session, pid, [("dataset_version", second.id)])
        route, recipe_id = endpoint(run), revision.id
        session.commit()
    processes = [node for node in client.get(route).json()["nodes"] if node["kind"] == "process"]
    assert len(processes) == 1
    assert processes[0]["label"] == "数据清洗"
    assert processes[0]["source"]["recipe_version_id"] == recipe_id


def test_multiple_roots_do_not_hide_cycles_and_node_budget_is_enforced(client):
    with SessionLocal() as session:
        pid = project(session).id
        versions = dataset(session, pid, 15)
        versions[0].parent_version_id = versions[-1].id
        run, _ = attempt(session, pid, [("dataset_version", v.id) for v in versions])
        route = endpoint(run)
        session.commit()
    view = client.get(route + "?max_nodes=10").json()
    assert view["truncated"] and len(view["nodes"]) <= 10
    complete = client.get(route).json()
    assert any("循环" in message for message in complete["warnings"])
    targets = {node["id"]: [] for node in complete["nodes"]}
    for edge in complete["edges"]:
        if edge["target"] in targets:
            targets[edge["source"]].append(edge["target"])

    def walk(node, path):
        assert node not in path
        for child in targets[node]:
            walk(child, path | {node})

    for node in targets:
        walk(node, set())
