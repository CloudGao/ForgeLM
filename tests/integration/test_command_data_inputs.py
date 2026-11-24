from uuid import uuid4

from llm_platform.database import SessionLocal
from llm_platform.models import Artifact, CodeSource, CodeVersion, Dataset, DatasetVersion, Task
from llm_platform.services.training_upstream_lineage import _domain_source


def fixture_assets(client, tmp_path):
    root = tmp_path / uuid4().hex
    root.mkdir()
    response = client.post(
        "/api/projects",
        json={"name": uuid4().hex, "target_model_name": "test", "workspace_uri": str(root)},
    )
    assert response.status_code == 201, response.text
    pid = response.json()["id"]
    with SessionLocal() as session:
        source = CodeSource(project_id=pid, name="code", source_type="local")
        snapshot = Artifact(
            kind="dataset_snapshot", uri=f"test://{uuid4()}", content_hash=uuid4().hex * 2
        )
        manifest = Artifact(
            kind="dataset_record_manifest", uri=f"test://{uuid4()}", content_hash=uuid4().hex * 2
        )
        code_art = Artifact(
            kind="source_code", uri=f"test://{uuid4()}", content_hash=uuid4().hex * 2
        )
        dataset = Dataset(project_id=pid, name="input", source_type="local")
        session.add_all([source, snapshot, manifest, code_art, dataset])
        session.flush()
        code = CodeVersion(source_id=source.id, version=1, state="ready", artifact_id=code_art.id)
        version = DatasetVersion(
            dataset_id=dataset.id,
            version=1,
            artifact_id=snapshot.id,
            record_manifest_artifact_id=manifest.id,
            train_ready=True,
            state="ready",
            document_count=12,
        )
        session.add_all([code, version])
        session.commit()
        return pid, code.id, version.id


def body(code, parent):
    return {
        "name": uuid4().hex,
        "source_type": "command",
        "start_import": True,
        "adapter_id": "builtin.command.data",
        "command": {"executable": "__worker_python__", "args": ["clean.py"]},
        "code_version_id": code,
        "input_dataset_version_ids": [parent, parent],
    }


def test_freezes_verified_inputs_and_exposes_derived_lineage(client, tmp_path):
    pid, code, parent = fixture_assets(client, tmp_path)
    result = client.post(f"/api/projects/{pid}/datasets", json=body(code, parent))
    assert result.status_code == 201, result.text
    identity = result.json()["current_version_id"]
    with SessionLocal() as session:
        version = session.get(DatasetVersion, identity)
        task = session.get(Task, version.task_id)
        assert (
            task.payload["code"]["content_hash"]
            == session.get(Artifact, session.get(CodeVersion, code).artifact_id).content_hash
        )
        assert len(task.payload["data_inputs"]) == 1
        assert (
            task.payload["data_inputs"][0]["dataset_artifact_hash"]
            == session.get(Artifact, session.get(DatasetVersion, parent).artifact_id).content_hash
        )
        source = _domain_source(session, "dataset_version", version)
        assert source["inputs"] == [("dataset_version", parent)]


def test_rejects_foreign_code_missing_parent_and_unready_input(client, tmp_path):
    pid, code, parent = fixture_assets(client, tmp_path)
    other, foreign, _ = fixture_assets(client, tmp_path)
    for data in [body(foreign, parent), body(code, str(uuid4()))]:
        assert client.post(f"/api/projects/{pid}/datasets", json=data).status_code == 422
    with SessionLocal() as session:
        session.get(DatasetVersion, parent).state = "importing"
        session.commit()
    assert client.post(f"/api/projects/{pid}/datasets", json=body(code, parent)).status_code == 409
    data = body(code, parent)
    data["start_import"] = False
    assert client.post(f"/api/projects/{pid}/datasets", json=data).status_code == 422


def test_command_can_parse_a_verified_raw_archive_without_text_records(client, tmp_path):
    pid, code, parent = fixture_assets(client, tmp_path)
    with SessionLocal() as session:
        version = session.get(DatasetVersion, parent)
        version.train_ready = False
        version.document_count = 0
        session.commit()
    response = client.post(f"/api/projects/{pid}/datasets", json=body(code, parent))
    assert response.status_code == 201, response.text


def test_command_evaluation_freezes_code_and_native_model_input(client, tmp_path):
    from llm_platform.models import ModelVersion
    from llm_platform.services.artifacts import artifact_storage

    pid, code, _ = fixture_assets(client, tmp_path)
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session, project_id=pid, kind="model", payload=uuid4().bytes, filename="model.tar.gz"
        )
        model = ModelVersion(
            project_id=pid,
            name="native",
            version=1,
            kind="base",
            status="candidate",
            artifact_id=artifact.id,
            metadata_json={"inference": {"format": "microlm-hybrid-safetensors"}},
        )
        session.add(model)
        session.commit()
        mid = model.id
    suite = client.post(
        f"/api/projects/{pid}/evaluation-suites",
        json={"name": "fixture", "benchmark_ids": ["lm-eval:piqa"]},
    )
    assert suite.status_code == 201, suite.text
    result = client.post(
        f"/api/projects/{pid}/evaluations",
        json={
            "model_version_ids": [mid],
            "suite_version_ids": [suite.json()["id"]],
            "code_version_id": code,
            "adapter_id": "builtin.command.evaluation",
            "command": {"executable": "__worker_python__", "args": ["eval.py"]},
        },
    )
    assert result.status_code == 201, result.text
    frozen = result.json()["task"]["payload"]
    assert frozen["code"]["version_id"] == code
    assert frozen["evaluation_model_inputs"][0]["model_version_id"] == mid
    assert frozen["config_snapshot"]["code"]["version_id"] == code
