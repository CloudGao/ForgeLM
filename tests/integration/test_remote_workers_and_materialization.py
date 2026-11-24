from __future__ import annotations

import tempfile
from uuid import uuid4

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    Artifact,
    ArtifactLocation,
    DatasetVersion,
    RemoteWorkerEnvironment,
    RemoteWorkerOperation,
)
from llm_platform.routers import execution_environments
from llm_platform.services.artifacts import artifact_storage
from llm_platform.services.dataset_materialization import ensure_version_primary_location
from llm_platform.services.remote_workers import issue_enrollment_token

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def test_autodl_create_uses_tunnel_and_does_not_persist_password(client, monkeypatch):
    monkeypatch.setattr(execution_environments, "capture_host_key", lambda *args, **kwargs: ("host ssh-ed25519 AAAA", "SHA256:test"))
    monkeypatch.setattr(execution_environments, "prepare_credential", lambda *args: "managed_" + "a" * 32)
    monkeypatch.setattr(execution_environments, "start_remote_worker_operation", lambda *_: None)
    response = client.post("/api/execution-environments/remote", json={
        "name": uuid4().hex, "ssh_command": "ssh -p 30221 root@autodl.test",
        "ssh_password": "never-persist-this", "runtime": "inference", "transport": "ssh_tunnel",
    })
    assert response.status_code == 201, response.text
    assert response.json()["transport"] == "ssh_tunnel"
    assert response.json()["ssh_user"] == "root"
    assert response.json()["runtime"] == "inference"
    assert "never-persist-this" not in response.text
    with SessionLocal() as session:
        env = session.get(RemoteWorkerEnvironment, response.json()["id"])
        assert "never-persist-this" not in str(env.to_dict())


def test_remote_validation_never_echoes_password(client):
    response = client.post("/api/execution-environments/remote", json={
        "name": "bad", "ssh_command": "ssh root@host command", "ssh_password": "do-not-echo",
    })
    assert response.status_code == 422
    assert "do-not-echo" not in response.text


def test_upgrade_blocks_task_claim_and_rejects_active_lease(client, tmp_path, monkeypatch):
    from llm_platform.models import Project, WorkerNode
    from llm_platform.services.tasks import claim_task, create_task
    monkeypatch.setattr(execution_environments, "start_remote_worker_operation", lambda *_: None)
    with SessionLocal() as session:
        project = Project(name=uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        worker = WorkerNode(name=uuid4().hex, hostname="remote", status="online",
                            resources={"cpu_count": 2, "gpus": []}, capabilities=["test.remote-bootstrap", "builtin.local-process"])
        session.add_all([project, worker])
        session.flush()
        environment = RemoteWorkerEnvironment(name=uuid4().hex, ssh_host="busy.remote.test", ssh_user="root",
                                              credential_ref="KEY", host_key="test", host_key_fingerprint="test",
                                              control_plane_url="http://localhost", worker_id=worker.id)
        session.add(environment)
        session.flush()
        task = create_task(session, project_id=project.id, kind="export", title="test",
                           adapter_id="test.remote-bootstrap", payload={"resource_request": {"worker_ids": [worker.id]}})
        operation = RemoteWorkerOperation(environment_id=environment.id, kind="upgrade", status="queued", phase="queued")
        session.add(operation)
        session.commit()
        assert claim_task(session, worker.id) is None
        operation.status = "failed"
        session.commit()
        claimed = claim_task(session, worker.id)
        assert claimed is not None
        session.commit()
        env_id = environment.id
    response = client.post(f"/api/execution-environments/remote/{env_id}/operations", json={"kind": "repair"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "remote_worker_busy"


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": f"Materialization-{uuid4().hex[:8]}",
            "target_model_name": "Materialization-LM",
            "workspace_uri": tempfile.mkdtemp(prefix="materialization-workspace-"),
            "project_type": "pretrain",
            "target_parameters": "42M",
        },
    )
    assert response.status_code == 201
    return response.json()


def _worker(client: TestClient, *, capabilities: list[str]) -> dict:
    response = client.post(
        "/api/agent/register",
        headers=AGENT_HEADERS,
        json={
            "name": f"materialization-worker-{uuid4().hex[:8]}",
            "hostname": "worker.example.test",
            "labels": {"os": "linux"},
            "resources": {"cpu_count": 8, "gpus": []},
            "capabilities": capabilities,
        },
    )
    assert response.status_code == 200
    return response.json()


def _imported_dataset_with_managed_primary(client: TestClient, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/datasets",
        json={
            "name": f"Imported-{uuid4().hex[:8]}",
            "source_type": "local",
            "source_uri": "file:///manual/import.jsonl",
            "initial_version": {
                "tier": "bronze",
                "raw_size_bytes": 17,
                "document_count": 1,
                "token_estimate": 4,
                "train_ready": True,
            },
        },
    )
    assert response.status_code == 201
    dataset = response.json()
    with SessionLocal() as session:
        version = session.get(DatasetVersion, dataset["current_version"]["id"])
        assert version is not None
        artifact = artifact_storage.put_bytes(
            session,
            project_id=project_id,
            kind="dataset_snapshot",
            payload=b'{"text":"hello"}\n',
            filename="import.jsonl",
            media_type="application/x-ndjson",
        )
        version.artifact_id = artifact.id
        ensure_version_primary_location(session, version, artifact)
        session.commit()
    return client.get(f"/api/datasets/{dataset['id']}").json()


def test_remote_environment_capture_and_operation_creation(client: TestClient, monkeypatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(execution_environments, "resolve_private_key", lambda _reference: None)
    monkeypatch.setattr(
        execution_environments,
        "capture_host_key",
        lambda host, port, expected_fingerprint=None: (
            f"[{host}]:{port} ssh-ed25519 AAAATEST",
            expected_fingerprint or "SHA256:test-host-key",
        ),
    )
    monkeypatch.setattr(
        execution_environments,
        "start_remote_worker_operation",
        lambda operation_id: started.append(operation_id),
    )
    name = f"ssh-worker-{uuid4().hex[:8]}"
    remote_payload = {
        "name": name,
        "ssh_host": "compute.example.test",
        "ssh_port": 2222,
        "ssh_user": "forgelm",
        "credential_ref": "FORGELM_TEST_SSH_KEY",
        "expected_host_key_fingerprint": "SHA256:test-host-key",
        "control_plane_url": "https://control.example.test",
        "workspace_root": "/srv/forgelm/workspaces",
        "install_root": "/opt/forgelm-worker",
        "profile": "data",
        "labels": {"zone": "lab-a"},
    }
    created = client.post("/api/execution-environments/remote", json=remote_payload)
    assert created.status_code == 201
    environment = created.json()
    assert environment["host_key_fingerprint"] == "SHA256:test-host-key"
    assert "enrollment_token_hash" not in environment
    assert environment["latest_operation"]["kind"] == "provision"
    assert environment["latest_operation"]["status"] == "queued"
    assert started == [environment["latest_operation"]["id"]]

    duplicate_endpoint = client.post(
        "/api/execution-environments/remote",
        json={**remote_payload, "name": f"duplicate-endpoint-{uuid4().hex[:8]}"},
    )
    assert duplicate_endpoint.status_code == 409
    assert duplicate_endpoint.json()["detail"]["code"] == "remote_endpoint_in_use"

    overlapping_roots = client.post(
        "/api/execution-environments/remote",
        json={
            **remote_payload,
            "name": f"unsafe-roots-{uuid4().hex[:8]}",
            "workspace_root": "/opt/forgelm-worker/workspaces",
        },
    )
    assert overlapping_roots.status_code == 422

    credentialed_control_url = client.post(
        "/api/execution-environments/remote",
        json={
            **remote_payload,
            "name": f"unsafe-url-{uuid4().hex[:8]}",
            "control_plane_url": "https://user:secret@control.example.test/api",
        },
    )
    assert credentialed_control_url.status_code == 422


def test_enrollment_is_single_use_and_node_token_is_worker_scoped(
    client: TestClient,
) -> None:
    environment_name = f"enrolled-{uuid4().hex[:8]}"
    with SessionLocal() as session:
        environment = RemoteWorkerEnvironment(
            name=environment_name,
            ssh_host="worker.example.test",
            ssh_port=22,
            ssh_user="forgelm",
            credential_ref="FORGELM_TEST_SSH_KEY",
            host_key="worker.example.test ssh-ed25519 AAAATEST",
            host_key_fingerprint="SHA256:test",
            control_plane_url="https://control.example.test",
            workspace_root="/srv/forgelm/workspaces",
            install_root="/opt/forgelm-worker",
            profile="general",
            labels={"zone": "trusted-zone"},
        )
        session.add(environment)
        session.flush()
        operation = RemoteWorkerOperation(
            environment_id=environment.id,
            kind="provision",
            status="succeeded",
            phase="waiting_for_agent",
            result={"waiting_for_agent": True},
        )
        session.add(operation)
        session.flush()
        enrollment_token = issue_enrollment_token(environment)
        environment_id = environment.id
        operation_id = operation.id
        session.commit()

    payload = {
        "environment_id": environment_id,
        "name": environment_name,
        "hostname": "worker.example.test",
        "labels": {"environment": "untrusted", "zone": "untrusted-zone"},
        "resources": {"cpu_count": 4, "gpus": []},
        "capabilities": ["builtin.local-process"],
    }
    enrolled = client.post(
        "/api/agent/enroll",
        headers={"X-Agent-Token": enrollment_token},
        json=payload,
    )
    assert enrolled.status_code == 200
    node_token = enrolled.json()["agent_token"]
    worker_id = enrolled.json()["worker"]["id"]
    assert node_token != enrollment_token
    assert "token_hash" not in enrolled.json()["worker"]
    assert enrolled.json()["worker"]["labels"] == {
        "environment": "remote-ssh",
        "zone": "trusted-zone",
        "remote_environment_id": environment_id,
    }
    with SessionLocal() as session:
        completed_operation = session.get(RemoteWorkerOperation, operation_id)
        assert completed_operation is not None
        assert completed_operation.phase == "completed"
        assert completed_operation.result["waiting_for_agent"] is False

    reused = client.post(
        "/api/agent/enroll",
        headers={"X-Agent-Token": enrollment_token},
        json=payload,
    )
    assert reused.status_code == 401
    assert reused.json()["detail"]["code"] == "invalid_enrollment_token"

    heartbeat = client.post(
        f"/api/agent/workers/{worker_id}/heartbeat",
        headers={"X-Agent-Token": node_token, "X-Worker-ID": worker_id},
        json={"resources": {"cpu_count": 4}, "running_attempt_ids": []},
    )
    assert heartbeat.status_code == 200
    assert "token_hash" not in heartbeat.json()["worker"]
    listed_worker = next(
        item for item in client.get("/api/workers").json() if item["id"] == worker_id
    )
    assert "token_hash" not in listed_worker

    registered = client.post(
        "/api/agent/register",
        headers={"X-Agent-Token": node_token, "X-Worker-ID": worker_id},
        json={
            "name": environment_name,
            "hostname": "worker.example.test",
            "labels": {"environment": "native", "zone": "untrusted-zone"},
            "resources": {"cpu_count": 4, "gpus": []},
            "capabilities": ["builtin.local-process"],
        },
    )
    assert registered.status_code == 200
    assert registered.json()["labels"] == {
        "environment": "remote-ssh",
        "zone": "trusted-zone",
        "remote_environment_id": environment_id,
    }

    other = _worker(client, capabilities=["builtin.local-process"])
    forbidden = client.post(
        f"/api/agent/workers/{other['id']}/heartbeat",
        headers={"X-Agent-Token": node_token, "X-Worker-ID": worker_id},
        json={"resources": {}, "running_attempt_ids": []},
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "agent_worker_scope_mismatch"


def test_imported_dataset_auto_copy_registers_verified_replica(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset = _imported_dataset_with_managed_primary(client, project["id"])
    version = dataset["current_version"]
    primary_id = version["materialization"]["primary_location"]["id"]
    assert version["materialization"]["kind"] == "imported"
    target = _worker(
        client,
        capabilities=["builtin.dataset-materializer", "builtin.local-process"],
    )

    preview = client.get(
        f"/api/dataset-versions/{version['id']}/materialization-routes",
        params={"target_worker_id": target["id"]},
    )
    assert preview.status_code == 200
    feasible = [route for route in preview.json()["routes"] if route["feasible"]]
    assert feasible[0]["strategy"] == "managed_copy"
    assert all(route["strategy"] not in {"download", "rebuild"} for route in feasible)

    created = client.post(
        f"/api/dataset-versions/{version['id']}/materializations",
        json={"target_worker_id": target["id"], "selection_mode": "auto"},
    )
    assert created.status_code == 202
    operation = created.json()
    assert operation["chosen_route"]["strategy"] == "managed_copy"
    assert operation["task_id"]

    claim = client.post(
        "/api/agent/tasks/claim",
        headers=AGENT_HEADERS,
        json={"worker_id": target["id"]},
    )
    assert claim.status_code == 200
    claimed = claim.json()
    assert claimed["task"]["id"] == operation["task_id"]
    assert claimed["task"]["adapter_id"] == "builtin.dataset-materializer"
    attempt_id = claimed["attempt"]["id"]
    started = client.post(
        f"/api/agent/tasks/{operation['task_id']}/start",
        headers=AGENT_HEADERS,
        json={
            "worker_id": target["id"],
            "attempt_id": attempt_id,
            "runtime_handle": {},
            "environment_snapshot": {},
        },
    )
    assert started.status_code == 200
    replica_uri = f"file:///srv/forgelm/replicas/{version['id']}"
    completed = client.post(
        f"/api/agent/tasks/{operation['task_id']}/complete",
        headers=AGENT_HEADERS,
        json={
            "worker_id": target["id"],
            "attempt_id": attempt_id,
            "status": "succeeded",
            "result": {
                "materialized_uri": replica_uri,
                "outputs": [
                    {
                        "ref": "dataset",
                        "kind": "dataset_snapshot",
                        "storage_type": "external",
                        "uri": replica_uri,
                        "content_hash": preview.json()["expected_content_hash"],
                        "size_bytes": 17,
                        "media_type": "application/x-ndjson",
                        "availability_state": "available",
                        "worker_id": target["id"],
                    }
                ],
            },
            "error": {},
            "exit_code": 0,
            "resource_summary": {},
            "log_summary": {},
        },
    )
    assert completed.status_code == 200

    materialization = client.get(f"/api/dataset-materializations/{operation['id']}").json()
    assert materialization["status"] == "succeeded"
    assert materialization["result_location_id"]
    refreshed = client.get(f"/api/datasets/{dataset['id']}").json()["current_version"]
    assert refreshed["materialization"]["primary_location"]["id"] == primary_id
    assert any(
        replica["worker_id"] == target["id"] and replica["last_verified_at"] is not None
        for replica in refreshed["materialization"]["replicas"]
    )


def test_dataset_primary_location_does_not_silently_fail_over(client: TestClient) -> None:
    project = _project(client)
    dataset = _imported_dataset_with_managed_primary(client, project["id"])
    version_id = dataset["current_version"]["id"]
    primary_id = dataset["current_version"]["materialization"]["primary_location"]["id"]

    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None and version.artifact_id
        artifact = session.get(Artifact, version.artifact_id)
        primary = session.get(ArtifactLocation, primary_id)
        assert artifact is not None and primary is not None
        primary.availability_state = "missing"
        _, replacement = artifact_storage.register_external(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            uri=f"file:///srv/forgelm/recovered/{uuid4().hex}.jsonl",
            content_hash=artifact.content_hash,
            size_bytes=artifact.size_bytes,
            media_type=artifact.media_type,
            availability_state="available",
        )

        selected = ensure_version_primary_location(session, version, artifact)

        assert selected is not None and selected.id == primary_id
        assert replacement.id != primary_id
        assert version.primary_location_id == primary_id
        assert version.materialization_state == "degraded"


def test_dataset_locations_stay_scoped_when_artifact_content_is_deduplicated(
    client: TestClient,
) -> None:
    first_project = _project(client)
    second_project = _project(client)
    first = _imported_dataset_with_managed_primary(client, first_project["id"])
    second = _imported_dataset_with_managed_primary(client, second_project["id"])
    first_version = first["current_version"]
    second_version = second["current_version"]
    first_primary = first_version["materialization"]["primary_location"]
    second_primary = second_version["materialization"]["primary_location"]

    assert first_version["artifact_id"] == second_version["artifact_id"]
    assert first_primary["id"] != second_primary["id"]
    assert first_primary["project_id"] == first_project["id"]
    assert second_primary["project_id"] == second_project["id"]
    assert all(
        replica["project_id"] in {second_project["id"], None}
        for replica in second_version["materialization"]["replicas"]
    )

    worker = _worker(
        client,
        capabilities=["builtin.dataset-materializer", "builtin.local-process"],
    )
    preview = client.get(
        f"/api/dataset-versions/{second_version['id']}/materialization-routes",
        params={"target_worker_id": worker["id"]},
    )
    assert preview.status_code == 200
    source_location_ids = {
        route["source_location_id"]
        for route in preview.json()["routes"]
        if route["source_location_id"]
    }
    assert second_primary["id"] in source_location_ids
    assert first_primary["id"] not in source_location_ids


def test_download_route_can_be_selected_manually_and_requires_capability(
    client: TestClient,
) -> None:
    project = _project(client)
    dataset = _imported_dataset_with_managed_primary(client, project["id"])
    version_id = dataset["current_version"]["id"]
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None
        version.materialization_kind = "download"
        version.materialization_spec = {
            "schema_version": 1,
            "kind": "download",
            "replay_adapter_id": "builtin.data-import",
            "replay_payload": {
                "dataset_id": dataset["id"],
                "dataset_version_id": version_id,
                "source_type": "https",
                "source_uri": "https://example.test/frozen.jsonl",
                "source_config": {},
                "resource_request": {"gpus": 0},
                "command": {},
            },
        }
        session.commit()

    capable = _worker(
        client,
        capabilities=["builtin.data-import", "builtin.local-process"],
    )
    preview = client.get(
        f"/api/dataset-versions/{version_id}/materialization-routes",
        params={"target_worker_id": capable["id"]},
    )
    assert preview.status_code == 200
    download = next(route for route in preview.json()["routes"] if route["strategy"] == "download")
    assert download["feasible"] is True

    created = client.post(
        f"/api/dataset-versions/{version_id}/materializations",
        json={
            "target_worker_id": capable["id"],
            "selection_mode": "manual",
            "route_id": download["route_id"],
        },
    )
    assert created.status_code == 202
    assert created.json()["chosen_route"]["strategy"] == "download"

    incapable = _worker(client, capabilities=["builtin.local-process"])
    blocked = client.post(
        f"/api/dataset-versions/{version_id}/materializations",
        json={
            "target_worker_id": incapable["id"],
            "selection_mode": "manual",
            "route_id": "download:frozen-source",
        },
    )
    assert blocked.status_code == 422


def test_derived_rebuild_route_checks_frozen_input_visibility(client: TestClient) -> None:
    project = _project(client)
    dataset = _imported_dataset_with_managed_primary(client, project["id"])
    version_id = dataset["current_version"]["id"]
    target = _worker(
        client,
        capabilities=["builtin.data-conversion", "builtin.local-process"],
    )
    with SessionLocal() as session:
        version = session.get(DatasetVersion, version_id)
        assert version is not None and version.artifact_id
        version.materialization_kind = "derived"
        version.materialization_spec = {
            "schema_version": 1,
            "kind": "derived",
            "replay_adapter_id": "builtin.data-conversion",
            "replay_payload": {
                "data_conversion_input": {
                    "artifact_id": version.artifact_id,
                    "artifact_hash": "frozen",
                },
                "recipe": {"operators": []},
                "resource_request": {"gpus": 0},
                "command": {},
            },
        }
        session.commit()

    visible = client.get(
        f"/api/dataset-versions/{version_id}/materialization-routes",
        params={"target_worker_id": target["id"]},
    )
    rebuild = next(route for route in visible.json()["routes"] if route["strategy"] == "rebuild")
    assert rebuild["feasible"] is True

    other = _worker(client, capabilities=["builtin.local-process"])
    with SessionLocal() as session:
        inaccessible_artifact, _ = artifact_storage.register_external(
            session,
            project_id=project["id"],
            kind="dataset_snapshot",
            uri="file:///private/other-worker/input.jsonl",
            content_hash="7" * 64,
            size_bytes=7,
            media_type="application/x-ndjson",
            availability_state="available",
            worker_id=other["id"],
        )
        version = session.get(DatasetVersion, version_id)
        spec = dict(version.materialization_spec or {})
        replay = dict(spec["replay_payload"])
        replay["data_conversion_input"] = {
            "artifact_id": inaccessible_artifact.id,
            "artifact_hash": inaccessible_artifact.content_hash,
        }
        spec["replay_payload"] = replay
        version.materialization_spec = spec
        session.commit()

    blocked = client.get(
        f"/api/dataset-versions/{version_id}/materialization-routes",
        params={"target_worker_id": target["id"]},
    )
    rebuild = next(route for route in blocked.json()["routes"] if route["strategy"] == "rebuild")
    assert rebuild["feasible"] is False
    assert "输入快照" in rebuild["reason"]
