from __future__ import annotations

import json
import sys
import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import DatasetVersion, ModelVersion
from llm_platform.services.artifacts import artifact_storage


def create_project(client: TestClient, prefix: str) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"{prefix}-{uuid4().hex[:8]}",
            "target_model_name": f"{prefix}-LM",
            "project_type": "pretrain",
            "workspace_uri": tempfile.mkdtemp(prefix=f"{prefix}-workspace-"),
        },
    )
    assert response.status_code == 201
    return response.json()


def test_catalog_includes_safe_modern_benchmarks(client: TestClient) -> None:
    response = client.get("/api/benchmarks/catalog")

    assert response.status_code == 200
    catalog = {item["id"]: item for item in response.json()}
    assert {
        "lm-eval:mmlu_pro",
        "lm-eval:ifeval",
        "lm-eval:longbench2",
    }.issubset(catalog)
    assert catalog["lm-eval:longbench2"]["runtime_requirements"]["extra"] == "longbench"
    assert "lm-eval:humaneval_plus" not in catalog
    assert "lm-eval:mbpp_plus" not in catalog


def test_unified_suite_preserves_frozen_versions_and_resolves_new_catalog_items(
    client: TestClient,
) -> None:
    project = create_project(client, "Unified-Eval")
    base = f"/api/projects/{project['id']}"
    original = client.post(f"{base}/benchmark-suites", json={
        "name": "Frozen custom fewshot", "benchmark_ids": ["lm-eval:piqa"],
        "overrides": {"lm-eval:piqa": {"num_fewshot": 3}},
    })
    assert original.status_code == 201, original.text
    frozen = original.json()["imported_datasets"][0]
    created = client.post(f"{base}/evaluation-suites", json={
        "name": "Unified suite", "datasets": [{"dataset_version_id": frozen["id"], "weight": 2}],
        "benchmark_ids": ["lm-eval:hellaswag"], "scoring_config": {"aggregator": "weighted_mean"},
    })
    assert created.status_code == 201, created.text
    suite = created.json()
    assert suite["scoring_config"]["asset_type"] == "benchmark"
    assert suite["datasets"][0] == {
        "dataset_version_id": frozen["id"], "weight": 2,
        "benchmark_id": "lm-eval:piqa", "task_id": "piqa", "config_hash": frozen["compatibility_hash"],
    }
    assert suite["datasets"][1]["benchmark_id"] == "lm-eval:hellaswag"
    current = client.get(f"{base}/evaluation-datasets").json()
    assert next(item for item in current if item["id"] == frozen["id"])["benchmark_config"] == frozen["benchmark_config"]
    # Subsequent submissions reuse the frozen catalog version, without duplicate imports.
    repeat = client.post(f"{base}/evaluation-suites", json={"name": "Repeat", "benchmark_ids": ["lm-eval:hellaswag"]})
    assert repeat.status_code == 201, repeat.text
    assert repeat.json()["datasets"][0]["dataset_version_id"] == suite["datasets"][1]["dataset_version_id"]


def test_unified_suite_rolls_back_imports_and_checks_project_access(client: TestClient) -> None:
    project = create_project(client, "Unified-Atomic")
    other = create_project(client, "Unified-Other")
    base = f"/api/projects/{project['id']}"
    bad = client.post(f"{base}/evaluation-suites", json={
        "name": "Invalid catalog selection", "benchmark_ids": ["lm-eval:piqa", "missing:task"],
    })
    assert bad.status_code == 404, bad.text
    assert client.get(f"{base}/evaluation-datasets").json() == []
    assert client.get(f"{base}/evaluation-suites").json() == []
    valid = client.post(f"{base}/evaluation-suites", json={"name": "Owned", "benchmark_ids": ["lm-eval:piqa"]})
    assert valid.status_code == 201, valid.text
    version_id = valid.json()["datasets"][0]["dataset_version_id"]
    foreign = client.post(f"/api/projects/{other['id']}/evaluation-suites", json={
        "name": "Foreign", "datasets": [{"dataset_version_id": version_id}],
        "benchmark_ids": ["lm-eval:hellaswag"],
    })
    assert foreign.status_code == 404, foreign.text
    assert client.get(f"/api/projects/{other['id']}/evaluation-datasets").json() == []
    duplicate = client.post(f"{base}/evaluation-suites", json={
        "name": "Duplicate", "datasets": [{"dataset_version_id": version_id}], "benchmark_ids": ["lm-eval:piqa"],
    })
    assert duplicate.status_code == 422, duplicate.text
    assert duplicate.json()["detail"]["code"] == "duplicate_dataset"
    empty = client.post(f"{base}/evaluation-suites", json={"name": "Empty"})
    assert empty.status_code == 422, empty.text


def test_unified_suite_rejects_incompatible_new_catalog_and_uploaded_inputs(client: TestClient) -> None:
    project = create_project(client, "Unified-Engine")
    base = f"/api/projects/{project['id']}"
    upload = client.post(f"{base}/evaluation-datasets/upload", data={
        "name": "Private questions", "task_type": "generation", "license": "internal",
        "dimensions": "[]", "contamination_tags": "[]", "scoring_config": "{}", "schema_config": json.dumps({"prompt_field": "prompt", "answer_field": "answer"}),
    }, files={"file": ("questions.jsonl", b'{"prompt":"Q","answer":"A"}\n', "application/x-ndjson")})
    assert upload.status_code == 201, upload.text
    mixed = client.post(f"{base}/evaluation-suites", json={
        "name": "Mixed engines", "datasets": [{"dataset_version_id": upload.json()["id"]}], "benchmark_ids": ["lm-eval:piqa"],
    })
    assert mixed.status_code == 422, mixed.text
    assert mixed.json()["detail"]["code"] == "incompatible_evaluation_engines"
    assert len(client.get(f"{base}/evaluation-datasets").json()) == 1


def test_catalog_includes_curated_chinese_base_model_benchmarks(
    client: TestClient,
) -> None:
    response = client.get("/api/benchmarks/catalog")

    assert response.status_code == 200
    catalog = {item["id"]: item for item in response.json()}
    expected = {
        "lm-eval:xstorycloze_zh",
        "lm-eval:xwinograd_zh",
        "lm-eval:belebele_zho_Hans",
        "lm-eval:zhoblimp_classifier_noun_agreement",
        "lm-eval:zhoblimp_anaphor_number_agreement",
        "lm-eval:zhoblimp_passive_agent_deletion_short",
        "lm-eval:zhoblimp_topicalization_SOV",
        "lm-eval:cmmlu_elementary_chinese",
        "lm-eval:cmmlu_elementary_commonsense",
        "lm-eval:cmmlu_chinese_literature",
        "lm-eval:cmmlu_chinese_history",
        "lm-eval:ceval-valid_high_school_chinese",
        "lm-eval:ceval-valid_chinese_language_and_literature",
        "lm-eval:ceval-valid_high_school_history",
        "lm-eval:ceval-valid_education_science",
        "lm-eval:forgelm_zh_chid_cloze",
        "lm-eval:forgelm_zh_cmrc2019_cloze",
        "lm-eval:forgelm_zh_c3_answer_text",
        "lm-eval:forgelm_zh_cmrc2018_generation",
        "lm-eval:forgelm_zh_cmmlu_elementary_chinese_answer_text",
        "lm-eval:forgelm_zh_ceval_high_school_chinese_answer_text",
    }
    assert expected.issubset(catalog)
    assert len(catalog) == len(response.json())
    assert catalog["lm-eval:belebele_zho_Hans"]["primary_metric"] == "acc"
    assert (
        catalog["lm-eval:zhoblimp_classifier_noun_agreement"]["primary_metric"]
        == "acc_norm"
    )
    assert catalog["lm-eval:forgelm_zh_cmrc2018_generation"]["primary_metric"] == "char_f1"
    assert catalog["lm-eval:forgelm_zh_cmrc2018_generation"]["include_in_overall"] is False
    assert catalog["lm-eval:forgelm_zh_cmrc2019_cloze"]["include_in_overall"] is False
    assert (
        catalog["lm-eval:forgelm_zh_ceval_high_school_chinese_answer_text"][
            "include_in_overall"
        ]
        is False
    )
    assert catalog["lm-eval:forgelm_zh_cmmlu_elementary_chinese_answer_text"].get(
        "include_in_overall", True
    )


def test_upload_creates_versioned_evaluation_dataset_with_artifact(
    client: TestClient,
) -> None:
    project = create_project(client, "Eval-Upload")
    data = {
        "name": "Private regression",
        "task_type": "generation",
        "license": "internal",
        "dimensions": json.dumps(["领域知识"]),
        "contamination_tags": json.dumps(["private-eval"]),
        "schema_config": json.dumps(
            {"prompt_field": "question", "answer_field": "answer", "id_field": "id"}
        ),
        "scoring_config": json.dumps({"metric": "exact_match"}),
    }
    files = {
        "file": (
            "private.jsonl",
            b'{"id":"q1","question":"2+2?","answer":"4"}\n'
            b'{"id":"q2","question":"3+3?","answer":"6"}\n',
            "application/x-ndjson",
        )
    }

    first = client.post(
        f"/api/projects/{project['id']}/evaluation-datasets/upload",
        data=data,
        files=files,
    )
    second = client.post(
        f"/api/projects/{project['id']}/evaluation-datasets/upload",
        data=data,
        files=files,
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    created = first.json()
    assert created["version"] == 1
    assert second.json()["version"] == 2
    assert created["source_type"] == "upload"
    assert created["sample_count"] == 2
    assert created["artifact_id"]
    assert created["source_config"]["format"] == "jsonl"
    assert created["schema_config"]["prompt_field"] == "question"
    assert len(created["compatibility_hash"]) == 64


def test_data_center_version_becomes_runnable_custom_evaluation_input(
    client: TestClient,
) -> None:
    project = create_project(client, "Eval-HF-Reference")
    created_data = client.post(
        f"/api/projects/{project['id']}/datasets",
        json={
            "name": "Imported benchmark",
            "source_type": "huggingface",
            "source_uri": "example/benchmark",
            "license": "Apache-2.0",
            "language": "en",
            "initial_version": {
                "tier": "bronze",
                "document_count": 2,
                "token_estimate": 30,
            },
            "start_import": False,
        },
    )
    assert created_data.status_code == 201, created_data.text
    data_version_id = created_data.json()["current_version"]["id"]
    with SessionLocal() as session:
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            payload=b'{"prompt":"A?","answer":"A"}\n',
            filename="benchmark.jsonl",
            media_type="application/x-ndjson",
        )
        data_version = session.get(DatasetVersion, data_version_id)
        assert data_version is not None
        data_version.artifact_id = artifact.id
        session.commit()

    imported = client.post(
        f"/api/projects/{project['id']}/evaluation-datasets/from-data-version",
        json={
            "name": "HF benchmark",
            "dataset_version_id": data_version_id,
            "task_type": "generation",
            "dimensions": ["领域知识"],
            "contamination_tags": ["public-source"],
            "schema_config": {"prompt_field": "prompt", "answer_field": "answer"},
            "scoring_config": {"metric": "exact_match"},
        },
    )
    assert imported.status_code == 201, imported.text
    eval_dataset = imported.json()
    assert eval_dataset["source_type"] == "huggingface"
    assert eval_dataset["source_config"]["dataset_version_id"] == data_version_id
    assert eval_dataset["artifact_id"] == artifact.id

    suite = client.post(
        f"/api/projects/{project['id']}/evaluation-suites",
        json={
            "name": "Custom suite",
            "datasets": [{"dataset_version_id": eval_dataset["id"], "weight": 1}],
        },
    )
    assert suite.status_code == 201, suite.text
    with SessionLocal() as session:
        model = ModelVersion(project_id=project["id"], name="Example", version=1)
        session.add(model)
        session.commit()
        model_id = model.id

    started = client.post(
        f"/api/projects/{project['id']}/evaluations",
        json={
            "model_version_ids": [model_id],
            "suite_version_ids": [suite.json()["id"]],
            "adapter_id": "builtin.command.evaluation",
            "command": {"executable": sys.executable, "args": ["evaluate.py"]},
        },
    )

    assert started.status_code == 201, started.text
    task = started.json()["task"]
    frozen = task["payload"]["evaluation_inputs"]
    assert len(frozen) == 1
    assert frozen[0]["artifact_id"] == artifact.id
    assert frozen[0]["artifact_hash"] == artifact.content_hash
    assert frozen[0]["schema_config"]["answer_field"] == "answer"
    assert started.json()["config_snapshot"]["evaluation_inputs"] == frozen
