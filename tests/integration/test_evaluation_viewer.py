import json

from llm_platform.database import SessionLocal
from llm_platform.models import Artifact, Task


def test_uploaded_evaluation_query_freezes_full_artifact_and_protects_active_query(client):
    uploaded = client.post("/api/catalog/evaluation-datasets/upload", data={
        "name": "viewer-complete", "task_type": "generation", "license": "internal", "dimensions": "[]", "contamination_tags": "[]",
        "schema_config": '{"prompt_field":"prompt","answer_field":"answer"}', "scoring_config": '{"metric":"exact_match"}',
    }, files={"file": ("eval.jsonl", "\n".join(json.dumps({"prompt": f"question-{i}", "answer": str(i)}) for i in range(120)).encode(), "application/x-ndjson")})
    assert uploaded.status_code == 201, uploaded.text
    version = uploaded.json()
    endpoint = f"/api/catalog/evaluation-datasets/{version['id']}/viewer/query"
    response = client.post(endpoint, json={"search": "question-119", "offset": 110, "limit": 10})
    assert response.status_code == 202, response.text
    task = response.json()
    assert task["payload"]["dataset_viewer_input"]["artifact_id"] == version["artifact_id"]
    assert task["payload"]["search"] == "question-119"
    assert task["payload"]["offset"] == 110
    assert task["payload"]["resource_request"]["worker_selector"]["capabilities"] == ["builtin.evaluation-viewer"]
    assert client.delete(f"/api/catalog/evaluation-datasets/{version['id']}").status_code == 409
    client.post(f"/api/tasks/{task['id']}/control", json={"action": "cancel"})
    assert client.delete(f"/api/catalog/evaluation-datasets/{version['id']}").status_code == 204


def test_builtin_query_requires_lm_eval_and_rejects_wrong_snapshot(client, monkeypatch):
    benchmark = client.get("/api/benchmarks/catalog").json()[0]
    endpoint = f"/api/catalog/benchmarks/{benchmark['id']}/viewer/query"
    response = client.post(endpoint, json={})
    assert response.status_code == 202, response.text
    task = response.json()
    assert task["payload"]["benchmark_viewer_input"]["task_id"] == benchmark["task_id"]
    assert task["payload"]["resource_request"]["worker_selector"]["capabilities"] == ["builtin.evaluation-viewer", "builtin.lm-eval"]
    assert client.post(endpoint, json={"source_task_id": task["id"]}).status_code == 409
    with SessionLocal() as session:
        stored = session.get(Task, task["id"])
        stored.observed_state = "succeeded"
        session.commit()
    assert client.post(endpoint, json={"source_task_id": task["id"]}).json()["detail"]["code"] == "evaluation_query_snapshot_missing"
    with SessionLocal() as session:
        artifact = Artifact(kind="dataset_snapshot", uri="file:///missing-eval-snapshot", content_hash="b" * 64, availability_state="available")
        session.add(artifact)
        session.flush()
        stored = session.get(Task, task["id"])
        stored.result = {"output_artifacts": [{"ref": "evaluation_viewer_snapshot", "artifact_id": artifact.id}]}
        session.commit()
    # A valid root reaches location validation; Artifact identities are global, not project-owned.
    unavailable = client.post(endpoint, json={"source_task_id": task["id"]})
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"]["code"] == "snapshot_location_unavailable"
    from llm_platform.services.evaluation_viewer import artifact_storage
    monkeypatch.setattr(artifact_storage, "inference_access", lambda *args: {"type": "managed"})
    reused = client.post(endpoint, json={"source_task_id": task["id"], "offset": 110, "search": "answer"})
    assert reused.status_code == 202, reused.text
    payload = reused.json()["payload"]
    assert payload["evaluation_snapshot"] is True
    assert "benchmark_viewer_input" not in payload
    assert payload["source_task_id"] == task["id"]
    assert payload["resource_request"]["worker_selector"]["capabilities"] == ["builtin.evaluation-viewer"]
    registered = client.post(f"/api/catalog/evaluation-datasets/from-benchmark/{benchmark['id']}", json={"overrides": {}}).json()
    mismatch = client.post(f"/api/catalog/evaluation-datasets/{registered['id']}/viewer/query", json={"source_task_id": task["id"]})
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "evaluation_query_source_mismatch"
    assert client.post("/api/catalog/benchmarks/unknown/viewer/query", json={}).status_code == 404
