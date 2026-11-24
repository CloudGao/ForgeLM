from uuid import uuid4

from llm_platform.database import SessionLocal
from llm_platform.models import ChatTemplate, ProjectPipelineNodeRun, Task, utcnow
from llm_platform.services.native_pipeline_nodes import resolve_parameters
from sqlalchemy import func, select


def setup_project(client, tmp_path):
    response = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "test", "workspace_uri": str(tmp_path)},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def template_node():
    return {
        "id": "template",
        "type": "preparation",
        "label": "对话模板",
        "config": {
            "execution_mode": "native",
            "native_operation": "template.create",
            "parameters": {
                "name": "chat",
                "template_body": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            },
        },
    }


def test_native_metadata_node_and_output_binding(client, tmp_path):
    pid = setup_project(client, tmp_path)
    base = f"/api/projects/{pid}/training-instances"
    body = {
        "name": "domain",
        "nodes": [
            template_node(),
            {
                "id": "suite",
                "type": "preparation",
                "label": "评测配置",
                "config": {
                    "execution_mode": "native",
                    "native_operation": "template.create",
                    "parameters": {"template_body": "{{ messages[0]['content'] }}"},
                    "input_bindings": {
                        "name": {"node_id": "template", "output_path": "chat_template_version_id"}
                    },
                },
            },
        ],
        "edges": [
            {
                "id": "edge",
                "source": "template",
                "target": "suite",
                "source_handle": "out",
                "target_handle": "in",
            }
        ],
    }
    response = client.post(base, json=body)
    assert response.status_code == 201, response.text
    iid = response.json()["id"]
    attempt = client.post(f"{base}/{iid}/attempts", json={})
    assert attempt.status_code == 201, attempt.text
    with SessionLocal() as session:
        rows = session.scalars(
            select(ProjectPipelineNodeRun).where(
                ProjectPipelineNodeRun.pipeline_run_id == attempt.json()["id"]
            )
        ).all()
        assert all(row.status == "succeeded" for row in rows), [(r.status, r.error) for r in rows]
        template, suite = sorted(rows, key=lambda row: row.node_id, reverse=True)
        assert (
            suite.input_snapshot["domain_parameters"]["name"]
            == template.output_snapshot["chat_template_version_id"]
        )
        assert suite.output_snapshot["chat_template_version_id"]
        assert session.scalar(select(func.count(Task.id)).where(Task.project_id == pid)) == 0


def test_invalid_binding_does_not_create_domain_objects(client, tmp_path):
    pid = setup_project(client, tmp_path)
    node = template_node()
    node["config"]["input_bindings"] = {"name": {"node_id": "missing", "output_path": "id"}}
    response = client.post(
        f"/api/projects/{pid}/training-instances",
        json={"name": "bad", "nodes": [node], "edges": []},
    )
    assert response.status_code == 422
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(func.count(ChatTemplate.id)).where(ChatTemplate.project_id == pid)
            )
            == 0
        )


def test_binding_preserves_numeric_epoch_budget():
    config = {
        "parameters": {"stages": [{"tokens": 1}]},
        "input_bindings": {
            "stages.0.tokens": {"node_id": "corpus", "output_path": "tokens", "multiplier": 2}
        },
    }
    assert (
        resolve_parameters(config, {"upstream": {"corpus": {"tokens": 321}}})["stages"][0]["tokens"]
        == 642
    )
    assert config["parameters"]["stages"][0]["tokens"] == 1


def test_domain_schema_rejects_custom_execution_only_for_native_node(client, tmp_path):
    pid = setup_project(client, tmp_path)
    node = {
        "id": "tokenizer",
        "type": "tokenizer",
        "label": "bad",
        "config": {
            "native_operation": "tokenizer.train",
            "parameters": {
                "name": "bad",
                "data_ref_type": "dataset_version",
                "data_ref_id": "missing",
                "adapter_id": "builtin.command.tokenizer",
            },
        },
    }
    response = client.post(
        f"/api/projects/{pid}/training-instances/validate",
        json={"name": "bad", "nodes": [node], "edges": []},
    )
    assert response.status_code == 200
    assert not response.json()["valid"]


def test_task_output_keeps_domain_version_identity(client, tmp_path):
    # Completion must not replace the domain IDs needed by downstream bindings.
    from llm_platform.services.training_instances import node_output

    pid = setup_project(client, tmp_path)
    with SessionLocal() as session:
        task = Task(
            project_id=pid,
            kind="data_process",
            title="clean",
            adapter_id="builtin.data-juicer",
            result={"record_count": 42},
        )
        session.add(task)
        session.flush()
        row = ProjectPipelineNodeRun(
            node_id="clean",
            node_type="clean",
            label="clean",
            created_at=utcnow(),
            output_snapshot={"dataset_version_id": "frozen-id"},
            config_snapshot={"native_operation": "data.processing"},
        )
        result = node_output(session, task, row)
        assert result["dataset_version_id"] == "frozen-id" and result["record_count"] == 42
