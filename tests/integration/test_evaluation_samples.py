from __future__ import annotations

import io
import json
import tarfile
import tempfile
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import EvaluationRun, Task
from llm_platform.services.artifacts import artifact_storage


def _report_archive(model_id: str, task_id: str, member_template: str = "lm-eval-report/{model_id}/{task_id}/samples.jsonl") -> bytes:
    rows = [
        {
            "doc_id": index,
            "doc": {"context": f"Context {index}", "question": f"Question {index}", "choices": ["No", "Yes"], "answer": 1, "subject": "logic"},
            "target": "1",
            "filtered_resps": [["-2.0", "False"], ["-0.2", "True"]],
            "acc": 1.0,
            "acc_norm": 1.0,
        }
        for index in range(3)
    ]
    payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as archive:
        info = tarfile.TarInfo(member_template.format(model_id=model_id, task_id=task_id))
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return target.getvalue()


@pytest.mark.parametrize(
    "member_template",
    [
        "lm-eval-report/{model_id}/{task_id}/samples.jsonl",
        "lm-eval-report/{model_id}/{task_id}/attempt-1-batch-16/samples.jsonl",
        "lm-eval-report/{model_id}/{task_id}/attempt-1-batch-auto/model/path/samples_{task_id}_2026-08-27.jsonl",
    ],
)
def test_evaluation_samples_reads_a_bounded_report_page(client: TestClient, member_template: str) -> None:
    project_response = client.post(
        "/api/projects",
        json={
            "name": f"Evaluation samples {uuid4().hex[:8]}",
            "target_model_name": "SampleLM",
            "workspace_uri": tempfile.mkdtemp(prefix="evaluation-samples-"),
            "project_type": "pretrain",
            "target_parameters": "1B",
            "primary_languages": ["en"],
            "target_capabilities": ["test"],
        },
    )
    assert project_response.status_code == 201
    project_id = project_response.json()["id"]
    model_id = "model-version-1"
    benchmark_task = "mmlu"

    with SessionLocal() as session:
        task = Task(project_id=project_id, kind="evaluate", title="Evaluation", observed_state="succeeded")
        session.add(task)
        session.flush()
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project_id,
            kind="evaluation_report",
            payload=_report_archive(model_id, benchmark_task, member_template),
            filename="report.tar.gz",
            media_type="application/x-tar",
        )
        task.result = {"artifact_id": artifact.id}
        run = EvaluationRun(
            project_id=project_id,
            task_id=task.id,
            model_version_ids=[model_id],
            suite_version_ids=["suite-1"],
            status="succeeded",
            summary={"models": {model_id: {"tasks": {benchmark_task: {"metric": "acc_norm", "score": 1.0}}}}},
        )
        session.add(run)
        session.commit()
        run_id = run.id

    response = client.get(
        f"/api/projects/{project_id}/evaluations/{run_id}/samples",
        params={"model_version_id": model_id, "task_id": benchmark_task, "offset": 0, "limit": 2},
    )
    assert response.status_code == 200
    page = response.json()
    assert page["has_more"] is True
    assert [item["question"] for item in page["items"]] == [
        "材料：Context 0\n\n问题：Question 0",
        "材料：Context 1\n\n问题：Question 1",
    ]
    assert page["items"][0]["model_answer"] == {"index": 1, "label": "B", "value": "Yes"}
    assert page["items"][0]["reference_answer"] == {"index": 1, "label": "B", "value": "Yes"}
    assert page["items"][0]["score"] == 1.0

    missing = client.get(
        f"/api/projects/{project_id}/evaluations/{run_id}/samples",
        params={"model_version_id": model_id, "task_id": "not-a-task", "limit": 2},
    )
    assert missing.status_code == 404
