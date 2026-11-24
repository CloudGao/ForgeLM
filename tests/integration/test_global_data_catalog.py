from __future__ import annotations

import io
import json
import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Artifact,
    DatasetLabelOperation,
    DatasetLabelSnapshot,
    DatasetVersion,
    Task,
    TaskAttempt,
)
from llm_platform.services.artifacts import artifact_storage
from llm_platform.services.tasks import finalize_task_domain


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"binding-project-{uuid4().hex[:8]}",
            "target_model_name": "BindingLM",
            "workspace_uri": tempfile.mkdtemp(prefix="binding-project-"),
            "project_type": "pretrain",
        },
    )
    assert response.status_code == 201
    return response.json()


def _label_scheme_payload(code: str) -> dict:
    return {
        "code": code,
        "name": "文本质量",
        "modality": "text",
        "description": "行级质量标签",
        "created_by": "tester",
        "groups": [
            {
                "code": "quality",
                "name": "质量",
                "selection_mode": "single",
                "required": False,
                "labels": [
                    {"code": "good", "name": "可用", "color": "#22c55e"},
                    {"code": "bad", "name": "不可用", "color": "#ef4444"},
                ],
            }
        ],
    }


def _global_dataset(client: TestClient) -> tuple[dict, str]:
    context = client.get("/api/catalog/context").json()
    response = client.post(
        "/api/catalog/datasets",
        json={
            "name": f"global-corpus-{uuid4().hex[:8]}",
            "source_type": "local",
            "source_uri": "C:/fixtures/global-corpus",
            "license": "MIT",
            "language": "zh",
            "tags": ["pretrain", "code"],
            "initial_version": {
                "tier": "gold",
                "document_count": 3,
                "token_estimate": 50,
                "train_ready": True,
            },
            "start_import": False,
        },
    )
    assert response.status_code == 201
    dataset = response.json()
    version_id = dataset["current_version"]["id"]
    lines = [
        json.dumps(
            {
                "type": "header",
                "format": "llm-dataset-records-v1",
                "dataset_version_id": version_id,
            },
            separators=(",", ":"),
        )
    ]
    for index, (path, length, tokens) in enumerate(
        [
            ("train/a.parquet", 40, 10),
            ("train/b.parquet", 100, 20),
            ("validation/c.parquet", 120, 20),
        ]
    ):
        lines.append(
            json.dumps(
                {
                    "type": "record",
                    "record_id": f"record-{index}",
                    "logical_path": path,
                    "record_index": index,
                    "content_hash": f"{index + 1:064x}",
                    "character_length": length,
                    "byte_length": length,
                    "token_estimate": tokens,
                    "language": "zh",
                },
                separators=(",", ":"),
            )
        )
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        snapshot = artifact_storage.put_bytes(
            session,
            project_id=context["catalog_project_id"],
            kind="dataset_snapshot",
            payload=b"global parquet snapshot",
            filename="dataset.parquet",
            media_type="application/vnd.apache.parquet",
        )
        manifest = artifact_storage.put_bytes(
            session,
            project_id=context["catalog_project_id"],
            kind="dataset_record_manifest",
            payload=("\n".join(lines) + "\n").encode(),
            filename="records.jsonl",
            media_type="application/x-ndjson",
        )
        version.artifact_id = snapshot.id
        version.record_manifest_artifact_id = manifest.id
        session.commit()
    return dataset, manifest.id


def _project_dataset(client: TestClient, project: dict, suffix: str) -> dict:
    response = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": f"legacy-{suffix}-{uuid4().hex[:8]}",
            "source_type": "local",
            "source_uri": f"C:/fixtures/{suffix}",
            "license": "MIT",
            "language": "zh",
            "initial_version": {
                "tier": "gold",
                "document_count": 1,
                "token_estimate": 10,
                "train_ready": True,
            },
            "start_import": False,
        },
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    version_id = dataset["current_version"]["id"]
    manifest_payload = "\n".join(
        [
            json.dumps(
                {
                    "type": "header",
                    "format": "llm-dataset-records-v1",
                    "dataset_version_id": version_id,
                }
            ),
            json.dumps(
                {
                    "type": "record",
                    "record_id": f"record-{suffix}",
                    "logical_path": f"train/{suffix}.parquet",
                    "record_index": 0,
                    "content_hash": f"{1:064x}",
                    "character_length": 20,
                    "byte_length": 20,
                    "token_estimate": 10,
                    "language": "zh",
                }
            ),
        ]
    )
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        snapshot = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            payload=f"snapshot-{suffix}".encode(),
            filename=f"{suffix}.parquet",
            media_type="application/vnd.apache.parquet",
        )
        manifest = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_record_manifest",
            payload=(manifest_payload + "\n").encode(),
            filename=f"{suffix}.jsonl",
            media_type="application/x-ndjson",
        )
        version.artifact_id = snapshot.id
        version.record_manifest_artifact_id = manifest.id
        session.commit()
    return client.get(f"/api/datasets/{dataset['id']}").json()


def test_global_catalog_duckdb_query_mixture_and_project_binding(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset, manifest_id = _global_dataset(client)
    version_id = dataset["current_version"]["id"]

    projects = client.get("/api/projects").json()
    assert all(item.get("system_kind", "") == "" for item in projects)
    catalog_project_id = client.get("/api/catalog/context").json()["catalog_project_id"]
    assert client.get(f"/api/projects/{catalog_project_id}").status_code == 404

    query_payload = {
        "dataset_version_ids": [version_id],
        "include_paths": ["train/**"],
        "exclude_paths": [],
        "languages": ["zh"],
        "min_length": 50,
        "max_length": 0,
        "sample_ratio": 1,
        "seed": 17,
        "offset": 0,
        "limit": 100,
    }
    response = client.post("/api/catalog/explorer/query", json=query_payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["engine"] == "duckdb_record_manifest_v1"
    assert result["total_rows"] == 1
    assert result["total_tokens"] == 20
    assert result["rows"][0]["logical_path"] == "train/b.parquet"
    assert result["materializable_as_mixture"] is False

    unsupported = client.post(
        "/api/catalog/explorer/mixtures",
        json={
            **query_payload,
            "name": f"language-mixture-{uuid4().hex[:8]}",
            "description": "language query result",
            "target_tokens": result["total_tokens"],
            "weights": {version_id: 1},
        },
    )
    assert unsupported.status_code == 409
    assert unsupported.json()["detail"]["code"] == "query_not_materializable"

    response = client.post(
        "/api/catalog/explorer/mixtures",
        json={
            **query_payload,
            "languages": [],
            "name": f"query-mixture-{uuid4().hex[:8]}",
            "description": "DuckDB query result",
            "target_tokens": result["total_tokens"],
            "weights": {version_id: 1},
        },
    )
    assert response.status_code == 201, response.text
    mixture = response.json()
    mixture_version = mixture["current_version"]
    assert mixture_version["entries"][0]["selection"]["include_paths"] == ["train/**"]
    assert mixture_version["entries"][0]["selection"]["min_length"] == 50

    response = client.post(
        f"/api/projects/{project['id']}/data-bindings",
        json={
            "asset_type": "mixture_version",
            "asset_version_id": mixture_version["id"],
            "purpose": "training",
            "alias": "训练组合",
            "metadata": {"reason": "integration-test"},
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["asset"]["name"] == mixture["name"]

    bound_mixtures = client.get(
        f"/api/projects/{project['id']}/mixtures"
    ).json()
    assert [item["id"] for item in bound_mixtures] == [mixture["id"]]
    assert bound_mixtures[0]["binding_scope"] == "bound"

    with SessionLocal() as session:
        manifest = session.get(Artifact, manifest_id)
        assert manifest is not None
        assert artifact_storage.managed_payload_path(
            session, manifest, project["id"]
        ) is not None

    task_id = mixture_version.get("task_id")
    assert task_id
    cancel = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "cancel", "reason": "integration test cleanup"},
    )
    assert cancel.status_code == 200, cancel.text


def test_dataset_viewer_query_creates_bounded_read_only_worker_task(
    client: TestClient,
) -> None:
    dataset, _manifest_id = _global_dataset(client)
    version_id = dataset["current_version"]["id"]
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        version.source_snapshot = {
            "files": [
                {"path": "data/train.parquet", "size_bytes": 100},
                {"path": "README.md", "size_bytes": 20},
            ]
        }
        session.commit()

    response = client.post(
        f"/api/catalog/dataset-versions/{version_id}/viewer/query",
        json={
            "sql": "SELECT source, messages FROM dataset WHERE source = 'alpha'",
            "configuration": "default",
            "split": "train",
            "offset": 100,
            "limit": 50,
            "timeout_seconds": 12,
        },
    )

    assert response.status_code == 202, response.text
    task = response.json()
    assert task["kind"] == "data_query"
    assert task["adapter_id"] == "builtin.dataset-viewer"
    assert task["payload"]["dataset_version_id"] == version_id
    assert task["payload"]["sql"].startswith("SELECT source")
    assert task["payload"]["offset"] == 100
    assert task["payload"]["limit"] == 50
    assert task["payload"]["timeout_seconds"] == 12
    assert task["payload"]["dataset_viewer_input"]["logical_files"] == [
        "data/train.parquet",
        "README.md",
    ]
    assert task["payload"]["resource_request"] == {
        "gpus": 0,
        "cpus": 1,
        "worker_selector": {"capabilities": ["builtin.dataset-viewer"]},
    }

    with SessionLocal() as session:
        stored = session.get(Task, task["id"])
        assert stored is not None
        assert stored.payload["dataset_viewer_input"]["artifact_hash"]

    cancel = client.post(
        f"/api/tasks/{task['id']}/control",
        json={"action": "cancel", "reason": "integration test cleanup"},
    )
    assert cancel.status_code == 200, cancel.text


def test_global_benchmark_center_registers_and_uploads_assets(
    client: TestClient,
) -> None:
    benchmark = client.get("/api/benchmarks/catalog").json()[0]
    response = client.post(
        f"/api/catalog/evaluation-datasets/from-benchmark/{benchmark['id']}",
        json={"overrides": {}},
    )
    assert response.status_code == 201, response.text
    registered = response.json()
    assert registered["source_type"] == "benchmark"
    assert registered["benchmark_config"]["id"] == benchmark["id"]

    repeated = client.post(
        f"/api/catalog/evaluation-datasets/from-benchmark/{benchmark['id']}",
        json={"overrides": {}},
    )
    assert repeated.status_code == 201
    assert repeated.json()["id"] == registered["id"]

    inspection = client.post(
        "/api/catalog/evaluation-datasets/inspect-upload",
        data={"schema_config": '{"prompt_field":"prompt","answer_field":"answer"}'},
        files={
            "file": (
                "eval.jsonl",
                b'{"prompt":"hello","answer":"world","topic":"smoke"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert inspection.status_code == 200, inspection.text
    inspected = inspection.json()
    assert inspected["fields"] == ["answer", "prompt", "topic"]
    assert inspected["valid_sample_count"] == 1
    assert inspected["invalid_sample_count"] == 0
    assert inspected["preview"][0]["prompt"] == "hello"

    response = client.post(
        "/api/catalog/evaluation-datasets/upload",
        data={
            "name": "global-private-eval",
            "task_type": "generation",
            "license": "internal",
            "dimensions": '["instruction-following"]',
            "contamination_tags": '["private-eval"]',
            "schema_config": '{"prompt_field":"prompt","answer_field":"answer"}',
            "scoring_config": '{"metric":"exact_match"}',
        },
        files={
            "file": (
                "eval.jsonl",
                b'{"prompt":"hello","answer":"world"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert uploaded["source_type"] == "upload"
    assert uploaded["sample_count"] == 1
    assert uploaded["source_config"]["fields"] == ["answer", "prompt"]

    invalid_mapping = client.post(
        "/api/catalog/evaluation-datasets/upload",
        data={
            "name": "invalid-global-eval",
            "task_type": "generation",
            "license": "internal",
            "dimensions": "[]",
            "contamination_tags": "[]",
            "schema_config": '{"prompt_field":"question","answer_field":"answer"}',
            "scoring_config": '{"metric":"exact_match"}',
        },
        files={
            "file": (
                "eval.jsonl",
                b'{"prompt":"hello","answer":"world"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert invalid_mapping.status_code == 422
    assert invalid_mapping.json()["detail"]["code"] == "invalid_evaluation_field_mapping"

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_buffer = io.BytesIO()
    pq.write_table(pa.table({"prompt": ["p1", "p2"], "answer": ["a1", "a2"]}), parquet_buffer)
    parquet_inspection = client.post(
        "/api/catalog/evaluation-datasets/inspect-upload",
        data={"schema_config": '{"prompt_field":"prompt","answer_field":"answer"}'},
        files={"file": ("eval.parquet", parquet_buffer.getvalue(), "application/vnd.apache.parquet")},
    )
    assert parquet_inspection.status_code == 200, parquet_inspection.text
    assert parquet_inspection.json()["sample_count"] == 2
    assert parquet_inspection.json()["format"] == "parquet"

    assets = client.get("/api/catalog/evaluation-datasets").json()
    assert {registered["id"], uploaded["id"]}.issubset({item["id"] for item in assets})


def test_global_evaluation_asset_creates_immutable_version_and_deletes_unreferenced(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/catalog/evaluation-datasets/upload",
        data={
            "name": "crud-evaluation-asset",
            "task_type": "generation",
            "license": "internal",
            "dimensions": '["reasoning"]',
            "contamination_tags": '["private"]',
            "schema_config": '{"prompt_field":"prompt","answer_field":"answer"}',
            "scoring_config": '{"metric":"exact_match"}',
        },
        files={
            "file": (
                "crud.jsonl",
                b'{"prompt":"2+2?","answer":"4"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert created.status_code == 201, created.text
    first = created.json()

    next_response = client.post(
        f"/api/catalog/evaluation-datasets/{first['id']}/versions",
        json={
            "task_type": "generation",
            "license": "internal-v2",
            "dimensions": ["reasoning", "math"],
            "contamination_tags": ["private", "reviewed"],
            "schema_config": {"prompt_field": "prompt", "answer_field": "answer"},
            "scoring_config": {"metric": "accuracy"},
        },
    )
    assert next_response.status_code == 201, next_response.text
    detail = next_response.json()
    second = detail["version"]
    assert second["version"] == first["version"] + 1
    assert second["artifact_id"] == first["artifact_id"]
    assert second["license"] == "internal-v2"
    assert [item["version"] for item in detail["versions"]] == [2, 1]
    assert detail["artifact"]["content_hash"] == first["source_config"]["content_hash"]
    assert detail["can_delete"] is True

    original = client.get(
        f"/api/catalog/evaluation-datasets/{first['id']}"
    ).json()["version"]
    assert original["license"] == "internal"
    assert original["scoring_config"]["metric"] == "exact_match"

    deleted = client.delete(
        f"/api/catalog/evaluation-datasets/{second['id']}"
    )
    assert deleted.status_code == 204, deleted.text
    assert (
        client.get(f"/api/catalog/evaluation-datasets/{second['id']}").status_code
        == 404
    )


def test_global_evaluation_asset_delete_reports_project_and_suite_blockers(
    client: TestClient,
) -> None:
    project = _project(client)
    benchmark = client.get("/api/benchmarks/catalog").json()[0]
    registered = client.post(
        f"/api/catalog/evaluation-datasets/from-benchmark/{benchmark['id']}",
        json={"overrides": {}},
    ).json()
    binding = client.post(
        f"/api/projects/{project['id']}/data-bindings",
        json={
            "asset_type": "eval_dataset_version",
            "asset_version_id": registered["id"],
            "purpose": "evaluation",
        },
    )
    assert binding.status_code == 201, binding.text
    suite = client.post(
        f"/api/projects/{project['id']}/evaluation-suites",
        json={
            "name": "bound-evaluation-suite",
            "datasets": [{"dataset_version_id": registered["id"], "weight": 1}],
            "scoring_config": {},
            "release_gate": {},
        },
    )
    assert suite.status_code == 201, suite.text

    detail = client.get(
        f"/api/catalog/evaluation-datasets/{registered['id']}"
    ).json()
    assert detail["can_delete"] is False
    assert {item["kind"] for item in detail["references"]} == {
        "project_binding",
        "eval_suite_version",
    }

    blocked = client.delete(
        f"/api/catalog/evaluation-datasets/{registered['id']}"
    )
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["code"] == "evaluation_dataset_version_in_use"
    assert {item["kind"] for item in blocked.json()["detail"]["blockers"]} == {
        "project_binding",
        "eval_suite_version",
    }


def test_binding_is_consume_only_and_cannot_be_removed_while_referenced(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset, _ = _global_dataset(client)
    version_id = dataset["current_version"]["id"]
    binding_response = client.post(
        f"/api/projects/{project['id']}/data-bindings",
        json={
            "asset_type": "dataset_version",
            "asset_version_id": version_id,
            "purpose": "training",
        },
    )
    assert binding_response.status_code == 201, binding_response.text
    binding = binding_response.json()

    mutation = client.post(
        f"/api/projects/{project['id']}/mixtures",
        json={
            "name": f"forbidden-{uuid4().hex[:8]}",
            "entries": [{"dataset_version_id": version_id, "weight": 1}],
            "target_tokens": 10,
        },
    )
    assert mutation.status_code == 422
    assert mutation.json()["detail"]["code"] == "invalid_dataset_version"

    tokenizer = client.post(
        f"/api/projects/{project['id']}/tokenizers",
        json={
            "name": f"binding-tokenizer-{uuid4().hex[:8]}",
            "algorithm": "BPE",
            "vocab_size": 4096,
            "data_ref_type": "dataset_version",
            "data_ref_id": version_id,
            "start": False,
        },
    )
    assert tokenizer.status_code == 201, tokenizer.text

    blocked = client.delete(
        f"/api/projects/{project['id']}/data-bindings/{binding['id']}"
    )
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert detail["code"] == "data_binding_in_use"
    assert any(item["kind"] == "tokenizer_version" for item in detail["blockers"])


def test_project_data_binding_updates_only_project_configuration(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset, _ = _global_dataset(client)
    version_id = dataset["current_version"]["id"]
    binding = client.post(
        f"/api/projects/{project['id']}/data-bindings",
        json={
            "asset_type": "dataset_version",
            "asset_version_id": version_id,
            "purpose": "training",
        },
    ).json()

    updated_response = client.patch(
        f"/api/projects/{project['id']}/data-bindings/{binding['id']}",
        json={"purpose": "validation", "alias": "回归验证集"},
    )

    assert updated_response.status_code == 200, updated_response.text
    updated = updated_response.json()
    assert updated["purpose"] == "validation"
    assert updated["alias"] == "回归验证集"
    assert updated["asset_type"] == "dataset_version"
    assert updated["asset_version_id"] == version_id
    assert updated["asset"]["name"] == dataset["name"]

    listed = client.get(
        f"/api/projects/{project['id']}/data-bindings",
        params={"purpose": "validation"},
    ).json()
    assert [item["id"] for item in listed] == [binding["id"]]


def test_catalog_can_version_legacy_mixture_with_cross_project_sources(
    client: TestClient,
) -> None:
    left_project = _project(client)
    right_project = _project(client)
    left_dataset = _project_dataset(client, left_project, "left")
    right_dataset = _project_dataset(client, right_project, "right")
    left_version_id = left_dataset["current_version"]["id"]
    right_version_id = right_dataset["current_version"]["id"]

    initial = client.post(
        f"/api/projects/{left_project['id']}/mixtures",
        json={
            "name": f"legacy-mixture-{uuid4().hex[:8]}",
            "entries": [{"dataset_version_id": left_version_id, "weight": 1}],
            "target_tokens": 10,
            "seed": 7,
        },
    )
    assert initial.status_code == 201, initial.text
    mixture = initial.json()
    initial_version = mixture["current_version"]

    updated = client.post(
        f"/api/catalog/mixtures/{mixture['id']}/versions",
        json={
            "entries": [
                {"dataset_version_id": left_version_id, "weight": 1},
                {"dataset_version_id": right_version_id, "weight": 2},
            ],
            "target_tokens": 30,
            "seed": 11,
        },
    )
    assert updated.status_code == 201, updated.text
    new_version = updated.json()
    assert new_version["version"] == 2
    assert {entry["dataset_version_id"] for entry in new_version["entries"]} == {
        left_version_id,
        right_version_id,
    }

    catalog_project_id = client.get("/api/catalog/context").json()["catalog_project_id"]
    with SessionLocal() as session:
        task = session.get(Task, new_version["task_id"])
        assert task is not None
        assert task.project_id == catalog_project_id

    for task_id in (initial_version["task_id"], new_version["task_id"]):
        response = client.post(
            f"/api/tasks/{task_id}/control",
            json={"action": "cancel", "reason": "integration test cleanup"},
        )
        assert response.status_code == 200, response.text
def test_label_scheme_versions_and_dataset_label_operation(client: TestClient) -> None:
    code = f"quality_{uuid4().hex[:8]}"
    created = client.post("/api/catalog/label-schemes", json=_label_scheme_payload(code))
    assert created.status_code == 201, created.text
    scheme = created.json()
    assert scheme["current_version"]["version"] == 1
    assert scheme["current_version"]["groups"][0]["selection_mode"] == "single"

    version_payload = _label_scheme_payload(code)
    version_payload.pop("code")
    version_payload.pop("name")
    version_payload.pop("modality")
    version_payload["description"] = "V2"
    versioned = client.post(
        f"/api/catalog/label-schemes/{scheme['id']}/versions", json=version_payload
    )
    assert versioned.status_code == 201, versioned.text
    scheme = versioned.json()
    assert scheme["current_version"]["version"] == 2
    assert scheme["version_count"] == 2

    dataset, _manifest_id = _global_dataset(client)
    dataset_version_id = dataset["current_version"]["id"]
    context = client.get("/api/catalog/context").json()
    record_hash = f"{1:064x}"
    with SessionLocal() as session:
        source_task = Task(
            project_id=context["catalog_project_id"],
            kind="data_query",
            title="frozen query",
            adapter_id="builtin.dataset-viewer",
            desired_state="running",
            observed_state="succeeded",
            payload={
                "dataset_version_id": dataset_version_id,
                "dataset_viewer_input": {"logical_files": ["train/a.parquet"]},
                "sql": "SELECT * FROM dataset",
                "split": "train",
                "label_snapshot_id": None,
            },
            result={
                "labelable": True,
                "sql": "SELECT * FROM dataset",
                "split": "train",
                "row_identities": [
                    {"record_id": "record-0", "record_hash": record_hash}
                ],
            },
        )
        session.add(source_task)
        session.commit()
        source_task_id = source_task.id

    label_id = scheme["current_version"]["groups"][0]["labels"][0]["id"]
    response = client.post(
        f"/api/catalog/dataset-versions/{dataset_version_id}/label-operations",
        json={
            "source_query_task_id": source_task_id,
            "scope": "selected",
            "action": "apply",
            "scheme_version_id": scheme["current_version"]["id"],
            "label_ids": [label_id],
            "records": [{"record_id": "record-0", "record_hash": record_hash}],
            "annotator": "tester",
            "note": "integration",
        },
    )
    assert response.status_code == 202, response.text
    operation = response.json()
    assert operation["scope"] == "selected"
    assert operation["task"]["adapter_id"] == "builtin.dataset-labeler"
    with SessionLocal() as session:
        task = session.get(Task, operation["task_id"])
        assert task is not None
        assert task.payload["operation_id"] == operation["id"]
        assert task.payload["labels"][0]["label_code"] == "good"
        attempt = TaskAttempt(task_id=task.id, attempt_number=1, status="running")
        session.add(attempt)
        session.flush()
        output_artifacts = []
        for ref, kind in (
            ("label_selection", "data_label_selection"),
            ("label_delta", "data_label_delta"),
            ("label_snapshot", "data_label_snapshot"),
        ):
            artifact = artifact_storage.put_bytes(
                session,
                project_id=context["catalog_project_id"],
                kind=kind,
                payload=f"{kind}-{operation['id']}".encode(),
                filename=f"{kind}.parquet",
                created_by_task_attempt_id=attempt.id,
            )
            output_artifacts.append(
                {"ref": ref, "artifact_id": artifact.id, "kind": kind, "storage_type": "managed"}
            )
        finalize_task_domain(
            session,
            task,
            attempt,
            {
                "engine": "duckdb_data_labeler_v1",
                "scope": "selected",
                "action": "apply",
                "selection_count": 1,
                "assignment_count": 1,
                "labeled_record_count": 1,
                "output_artifacts": output_artifacts,
            },
        )
        task.observed_state = "succeeded"
        task.desired_state = "succeeded"
        session.commit()
        finalized = session.get(DatasetLabelOperation, operation["id"])
        assert finalized is not None and finalized.status == "succeeded"
        snapshot = session.get(DatasetLabelSnapshot, finalized.output_snapshot_id)
        assert snapshot is not None
        assert snapshot.version == 1
        assert snapshot.assignment_count == 1


def test_project_global_mixture_requires_binding_and_binds_output(client: TestClient):
    project = _project(client)
    dataset, _ = _global_dataset(client)
    version_id = dataset["current_version"]["id"]
    context = client.get("/api/catalog/context").json()
    request = {"name": "shared selection", "asset_scope": "global", "entries": [
        {"dataset_version_id": version_id, "weight": 1, "selection": {"include_paths": ["train/a.parquet"]}}
    ]}
    url = f"/api/projects/{project['id']}/mixtures"
    assert client.post(url, json=request).status_code == 422
    bound = client.post(f"/api/projects/{project['id']}/data-bindings", json={
        "asset_type": "dataset_version", "asset_version_id": version_id, "purpose": "training"
    })
    assert bound.status_code == 201, bound.text
    response = client.post(url, json=request)
    assert response.status_code == 201, response.text
    mixture = response.json()
    assert mixture["project_id"] == context["catalog_project_id"]
    version = mixture["current_version"]
    with SessionLocal() as session:
        task = session.get(Task, version["task_id"])
        assert task.project_id == context["catalog_project_id"]
    bindings = client.get(f"/api/projects/{project['id']}/data-bindings").json()
    assert any(x["asset_version_id"] == version["id"] for x in bindings)
