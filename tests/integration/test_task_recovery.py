from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from llm_platform.database import SessionLocal
from llm_platform.models import (
    PlanRevision,
    Project,
    RemoteWorkerEnvironment,
    ResourceLease,
    Task,
    TaskAttempt,
    TrainingPlan,
    TrainingRun,
    WorkerNode,
    utcnow,
)
from llm_platform.services.tasks import (
    claim_task,
    create_task,
    mark_stale_workers_offline,
    reap_expired_leases,
    transition_task,
)
from sqlalchemy import select

AGENT_HEADERS = {"X-Agent-Token": "local-development-token"}


def test_expired_attempt_lease_marks_task_lost(
    client: TestClient, tmp_path: Path
) -> None:
    del client
    with SessionLocal() as session:
        project = Project(
            name="lease-recovery-project",
            target_model_name="lease-recovery-model",
            workspace_uri=str(tmp_path),
        )
        worker = WorkerNode(
            name="lease-recovery-worker",
            hostname="localhost",
            status="online",
            resources={"cpu_count": 2, "gpus": []},
            capabilities=["test.adapter", "builtin.local-process"],
        )
        session.add_all([project, worker])
        session.flush()
        task = create_task(
            session,
            project_id=project.id,
            kind="export",
            title="Lease recovery contract",
            adapter_id="test.adapter",
            payload={"resource_request": {}},
        )
        session.commit()

        claimed = claim_task(session, worker.id)
        assert claimed is not None
        task, attempt, lease = claimed
        transition_task(session, task, "starting", attempt_id=attempt.id)
        transition_task(session, task, "running", attempt_id=attempt.id)
        lease.expires_at = utcnow() - timedelta(seconds=1)
        session.commit()

        assert reap_expired_leases(session) == 1
        session.commit()
        assert task.observed_state == "lost"
        assert attempt.status == "lost"
        assert lease.status == "expired"


def test_stale_worker_is_marked_offline_and_heartbeat_can_restore_it(
    client: TestClient,
) -> None:
    del client
    with SessionLocal() as session:
        worker = WorkerNode(
            name="stale-worker",
            hostname="localhost",
            status="online",
            labels={"environment": "wsl"},
            resources={"cpu_count": 2, "gpus": []},
            capabilities=[],
            last_heartbeat_at=utcnow() - timedelta(seconds=31),
        )
        session.add(worker)
        session.flush()
        environment = RemoteWorkerEnvironment(
            name="stale-remote-environment",
            ssh_host="stale.example.test",
            ssh_port=22,
            ssh_user="forgelm",
            credential_ref="FORGELM_TEST_SSH_KEY",
            host_key="stale.example.test ssh-ed25519 AAAATEST",
            host_key_fingerprint="SHA256:test",
            control_plane_url="https://control.example.test",
            workspace_root="/srv/forgelm/workspaces",
            install_root="/opt/forgelm-worker",
            profile="general",
            labels={},
            worker_id=worker.id,
            status="online",
        )
        session.add(environment)
        session.commit()

        assert mark_stale_workers_offline(session, timeout_seconds=30) == 1
        session.commit()
        assert worker.status == "offline"
        assert environment.status == "offline"

        worker.status = "online"
        worker.last_heartbeat_at = utcnow()
        session.commit()
        assert mark_stale_workers_offline(session, timeout_seconds=30) == 0
        assert worker.status == "online"


def test_live_worker_heartbeat_recovers_attempt_lost_only_by_expired_lease(
    client: TestClient, tmp_path: Path
) -> None:
    with SessionLocal() as session:
        project = Project(
            name="live-attempt-recovery-project",
            target_model_name="live-attempt-recovery-model",
            workspace_uri=str(tmp_path),
        )
        worker = WorkerNode(
            name="live-attempt-recovery-worker",
            hostname="localhost",
            status="online",
            resources={"cpu_count": 2, "gpus": []},
            capabilities=["test.adapter", "builtin.local-process"],
        )
        session.add_all([project, worker])
        session.flush()
        task = create_task(
            session,
            project_id=project.id,
            kind="export",
            title="Live attempt recovery contract",
            adapter_id="test.adapter",
            payload={"resource_request": {}},
        )
        session.commit()
        claimed = claim_task(session, worker.id)
        assert claimed is not None
        task, attempt, lease = claimed
        transition_task(session, task, "starting", attempt_id=attempt.id)
        transition_task(session, task, "running", attempt_id=attempt.id)
        attempt.status = "running"
        lease.expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
        assert reap_expired_leases(session) == 1
        session.commit()
        task_id = task.id
        attempt_id = attempt.id
        worker_id = worker.id

    response = client.post(
        f"/api/agent/workers/{worker_id}/heartbeat",
        headers=AGENT_HEADERS,
        json={"resources": {"cpu_count": 2, "gpus": []}, "running_attempt_ids": [attempt_id]},
    )
    assert response.status_code == 200
    assert response.json()["recovered_attempt_ids"] == [attempt_id]

    with SessionLocal() as session:
        task = session.get(Task, task_id)
        attempt = session.get(TaskAttempt, attempt_id)
        lease = session.scalar(
            select(ResourceLease).where(ResourceLease.task_attempt_id == attempt_id)
        )
        assert task is not None and task.observed_state == "running"
        assert task.ended_at is None and task.error == {}
        assert attempt is not None and attempt.status == "running"
        assert attempt.ended_at is None and attempt.error == {}
        assert lease is not None and lease.status == "active"
        assert lease.released_at is None


def test_live_worker_heartbeat_does_not_recover_cancelled_intent(
    client: TestClient, tmp_path: Path
) -> None:
    with SessionLocal() as session:
        project = Project(
            name="cancelled-attempt-recovery-project",
            target_model_name="cancelled-attempt-recovery-model",
            workspace_uri=str(tmp_path),
        )
        worker = WorkerNode(
            name="cancelled-attempt-recovery-worker",
            hostname="localhost",
            status="online",
            resources={"cpu_count": 2, "gpus": []},
            capabilities=["test.adapter", "builtin.local-process"],
        )
        session.add_all([project, worker])
        session.flush()
        task = create_task(
            session,
            project_id=project.id,
            kind="export",
            title="Cancelled intent recovery contract",
            adapter_id="test.adapter",
            payload={"resource_request": {}},
        )
        session.commit()
        claimed = claim_task(session, worker.id)
        assert claimed is not None
        task, attempt, lease = claimed
        transition_task(session, task, "starting", attempt_id=attempt.id)
        transition_task(session, task, "running", attempt_id=attempt.id)
        attempt.status = "running"
        lease.expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
        assert reap_expired_leases(session) == 1
        task.desired_state = "cancelled"
        session.commit()
        task_id = task.id
        attempt_id = attempt.id
        worker_id = worker.id

    response = client.post(
        f"/api/agent/workers/{worker_id}/heartbeat",
        headers=AGENT_HEADERS,
        json={"running_attempt_ids": [attempt_id]},
    )
    assert response.status_code == 200
    assert response.json()["recovered_attempt_ids"] == []
    with SessionLocal() as session:
        assert session.get(Task, task_id).observed_state == "lost"
        assert session.get(TaskAttempt, attempt_id).status == "lost"


def test_training_retry_rejects_frozen_vocab_mismatch(
    client: TestClient, tmp_path: Path
) -> None:
    with SessionLocal() as session:
        project = Project(
            name="retry-vocab-project",
            target_model_name="retry-vocab-model",
            workspace_uri=str(tmp_path),
        )
        session.add(project)
        session.flush()
        task = create_task(
            session,
            project_id=project.id,
            kind="train",
            title="Mismatched frozen vocabulary",
            adapter_id="builtin.command.training",
            payload={
                "model_vocab_size": 32_000,
                "tokenizer": {"vocab_size": 32_768},
                "resource_request": {},
            },
        )
        task.observed_state = "failed"
        task.desired_state = "running"
        session.commit()
        task_id = task.id

    response = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "retry", "reason": "must not bypass vocabulary contract"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "tokenizer_model_vocab_mismatch"


def test_cancelled_training_can_resume_as_a_new_attempt(
    client: TestClient, tmp_path: Path
) -> None:
    with SessionLocal() as session:
        project = Project(
            name="cancelled-training-resume-project",
            target_model_name="cancelled-training-resume-model",
            workspace_uri=str(tmp_path),
        )
        session.add(project)
        session.flush()
        plan = TrainingPlan(
            project_id=project.id,
            name="Resume plan",
            description="",
            status="cancelled",
        )
        session.add(plan)
        session.flush()
        revision = PlanRevision(plan_id=plan.id, revision=1, stages=[])
        session.add(revision)
        session.flush()
        plan.current_revision_id = revision.id
        task = create_task(
            session,
            project_id=project.id,
            kind="train",
            title="Resume cancelled training",
            adapter_id="test.cancelled-training-resume",
            payload={"resource_request": {}},
        )
        run = TrainingRun(
            project_id=project.id,
            plan_revision_id=revision.id,
            task_id=task.id,
            status="cancelled",
        )
        worker = WorkerNode(
            name="cancelled-training-resume-worker",
            hostname="localhost",
            status="online",
            resources={"cpu_count": 2, "gpus": []},
            capabilities=["test.cancelled-training-resume", "builtin.local-process"],
        )
        previous_attempt = TaskAttempt(
            task_id=task.id,
            attempt_number=1,
            status="cancelled",
            ended_at=utcnow(),
        )
        session.add_all([run, worker, previous_attempt])
        session.flush()
        task.payload = {**(task.payload or {}), "training_run_id": run.id}
        task.active_attempt_id = previous_attempt.id
        task.observed_state = "cancelled"
        task.desired_state = "cancelled"
        task.progress = 0.36
        task.error = {"code": "cancelled", "message": "stopped by user"}
        task.ended_at = utcnow()
        run.ended_at = task.ended_at
        session.commit()
        task_id = task.id
        run_id = run.id
        worker_id = worker.id

    response = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "resume", "reason": "continue training"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["observed_state"] == "queued"
    assert body["desired_state"] == "running"
    assert body["progress"] == 0.36
    assert body["ended_at"] is None
    assert body["error"] == {}
    assert body["events"][-1]["event_type"] == "task.resume_requested"
    assert body["events"][-1]["payload"]["from_state"] == "cancelled"
    with SessionLocal() as session:
        resumed_run = session.get(TrainingRun, run_id)
        assert resumed_run is not None
        assert resumed_run.status == "queued"
        assert resumed_run.ended_at is None
        claimed = claim_task(session, worker_id)
        assert claimed is not None
        resumed_task, resumed_attempt, _lease = claimed
        assert resumed_task.id == task_id
        assert resumed_task.payload["training_run_id"] == run_id
        assert resumed_attempt.attempt_number == 2


def test_cancelled_non_training_task_cannot_resume(client: TestClient, tmp_path: Path) -> None:
    with SessionLocal() as session:
        project = Project(
            name="cancelled-export-project",
            target_model_name="cancelled-export-model",
            workspace_uri=str(tmp_path),
        )
        session.add(project)
        session.flush()
        task = create_task(
            session,
            project_id=project.id,
            kind="export",
            title="Cancelled export",
            adapter_id="test.adapter",
        )
        task.observed_state = "cancelled"
        task.desired_state = "cancelled"
        session.commit()
        task_id = task.id

    response = client.post(
        f"/api/tasks/{task_id}/control",
        json={"action": "resume", "reason": "must remain stopped"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "not_resumable"
