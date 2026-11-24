from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import select


def test_native_receipts_and_step_intervals_are_verified_and_ordered(client):
    from llm_platform.database import SessionLocal
    from llm_platform.models import Project, Task, TaskAttempt
    from llm_platform.services.execution_provenance import record_interval, record_receipt
    with SessionLocal() as session:
        project = Project(name="native-receipt-" + uuid4().hex, target_model_name="test")
        session.add(project)
        session.flush()
        task = Task(project_id=project.id, kind="train", title="native", payload={"stages": [{"type": "pretrain"}], "code": {"version_id": "new-code"}})
        session.add(task)
        session.flush()
        attempt = TaskAttempt(task_id=task.id, attempt_number=1)
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        receipt = {"schema": "forgelm.microlm-optimizations.v1", "code": task.payload["code"], "resolved_config": {"stages": task.payload["stages"]}, "start_step": 7}
        with pytest.raises(ValueError, match="code identity"):
            record_receipt(session, task, attempt, {**receipt, "code": {"version_id": "wrong"}})
        frozen = record_receipt(session, task, attempt, receipt)
        interval = {"receipt_digest": frozen.digest, "start_step": 7, "end_step": 9, "dispatch": {"partial_rope": 10}}
        first = record_interval(session, task, attempt, interval)
        assert record_interval(session, task, attempt, interval).id == first.id
        with pytest.raises(ValueError, match="ordered"):
            record_interval(session, task, attempt, {**interval, "end_step": 10})
        record_interval(session, task, attempt, {**interval, "start_step": 9, "end_step": 10})


def test_attempt_snapshots_are_append_only_and_queryable(client):
    from llm_platform.database import SessionLocal
    from llm_platform.models import ExecutionSnapshot, Project, Task, TaskAttempt
    from llm_platform.services.execution_provenance import append_snapshot, freeze_intent
    with SessionLocal() as session:
        project = Project(name="lineage-" + uuid4().hex, target_model_name="test")
        session.add(project)
        session.flush()
        task = Task(project_id=project.id, kind="train", title="lineage", payload={"config": {"lr": 0.001}, "code": {"digest": "abc"}})
        session.add(task)
        session.flush()
        attempts = []
        for index in range(2):
            attempt = TaskAttempt(task_id=task.id, attempt_number=index + 1, status="paused" if index == 0 else "running")
            session.add(attempt)
            session.flush()
            freeze_intent(session, task, attempt, {})
            attempts.append(attempt)
        row = append_snapshot(session, attempts[0], "completed", {"status": "paused"})
        assert append_snapshot(session, attempts[0], "completed", {"status": "paused"}).id == row.id
        with pytest.raises(ValueError, match="Immutable"):
            append_snapshot(session, attempts[0], "completed", {"status": "succeeded"})
        task.payload = {"config": {"lr": 0.1}}
        session.commit()
        rows = list(session.scalars(select(ExecutionSnapshot).where(ExecutionSnapshot.task_id == task.id)))
        assert len(rows) == 3
        assert all(r.snapshot["payload"]["config"]["lr"] == 0.001 for r in rows if r.phase == "intent")
        task_id = task.id
    response = client.get(f"/api/tasks/{task_id}/execution-snapshots")
    assert response.status_code == 200 and len(response.json()) == 3


def test_checkpoint_continuation_requires_project_state_and_input_compatibility(client, tmp_path):
    from llm_platform.database import SessionLocal
    from llm_platform.models import Project, Task, TaskAttempt, TrainingStateCheckpoint
    from llm_platform.services.artifacts import artifact_storage
    from llm_platform.services.training_continuation import prepare
    contract = {"checkpoint_format": "model-state-v1", "state_compatibility": "adamw-v1", "modes": ["full", "configured", "weights"]}
    config = {"lr": 0.001}
    inputs = {"data": {"digest": "source-data"}}
    with SessionLocal() as session:
        project = Project(name="continuation-" + uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        session.add(project)
        session.flush()
        task = Task(project_id=project.id, kind="suite_node", title="source", payload={"suite_execution": {"inputs": inputs}})
        session.add(task)
        session.flush()
        attempt = TaskAttempt(task_id=task.id, attempt_number=1)
        session.add(attempt)
        session.flush()
        artifact = artifact_storage.put_bytes(session, project_id=project.id, kind="training_state_checkpoint", payload=b"checkpoint", filename="state.bin")
        checkpoint = TrainingStateCheckpoint(project_id=project.id, task_id=task.id, attempt_id=attempt.id, artifact_id=artifact.id, step=7, manifest={"contract": contract, "config": config})
        session.add(checkpoint)
        session.flush()
        source = deepcopy(task.payload)
        handoff = prepare(session, project.id, {"training": contract}, config, inputs, checkpoint.id, "full")
        assert set(handoff["state_policy"].values()) == {"keep"}
        assert handoff["source_attempt_id"] == attempt.id
        changed = {"data": {"digest": "new-data"}}
        with pytest.raises(ValueError, match="完整状态"):
            prepare(session, project.id, {"training": contract}, config, changed, checkpoint.id, "full")
        handoff = prepare(session, project.id, {"training": contract}, {"lr": 0.01}, changed, checkpoint.id, "configured", True)
        assert handoff["state_policy"]["optimizer"] == "keep" and handoff["state_policy"]["data"] == "reset"
        incompatible = {**contract, "state_compatibility": "other-layout"}
        with pytest.raises(ValueError, match="布局"):
            prepare(session, project.id, {"training": incompatible}, config, inputs, checkpoint.id, "full")
        assert prepare(session, project.id, {"training": incompatible}, config, inputs, checkpoint.id, "weights")["state_policy"]["optimizer"] == "reset"
        with pytest.raises(ValueError, match="项目"):
            prepare(session, "other-project", {"training": contract}, config, inputs, checkpoint.id, "weights")
        assert task.payload == source


def test_runtime_receipts_and_checkpoints_reject_replacement_and_foreign_artifacts(client, tmp_path):
    from llm_platform.database import SessionLocal
    from llm_platform.models import Project, Task, TaskAttempt, WorkerNode
    from llm_platform.services.artifacts import artifact_storage
    from llm_platform.services.execution_provenance import record_receipt, register_checkpoint
    from llm_platform.services.tasks import append_task_event
    from llm_platform.suite_models import (
        Suite,
        SuiteEnvironmentLease,
        SuiteInstallation,
        SuiteVersion,
        WorkerSuiteDeployment,
    )
    contract = {"checkpoint_format": "state-v1", "state_compatibility": "adam-v1", "modes": ["full", "configured", "weights"]}
    with SessionLocal() as session:
        project = Project(name="receipts-" + uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        suite = Suite(suite_id="test." + uuid4().hex, name="test", publisher="test")
        worker = WorkerNode(name="receipts-" + uuid4().hex, hostname="test")
        session.add_all([project, suite, worker])
        session.flush()
        artifact = artifact_storage.put_bytes(session, project_id=project.id, kind="suite_package", payload=b"package", filename="package.bin")
        component = {"id": "train", "implementationVersion": "1.1.0"}
        version = SuiteVersion(suite_id=suite.id, version="1.1.0", artifact_id=artifact.id, digest="a" * 64, package={"manifest": {"components": [component]}, "nodes": [{"id": "train", "training": contract}]})
        session.add(version)
        session.flush()
        deployment = WorkerSuiteDeployment(worker_id=worker.id, suite_id=suite.id)
        session.add(deployment)
        session.flush()
        env = {"generation": "one", "target_id": "cpu", "groups": {"default": {"lock_digest": "locked"}}, "digest": version.digest}
        installation = SuiteInstallation(deployment_id=deployment.id, suite_version_id=version.id, generation="one", receipt=env, state="ready")
        task = Task(project_id=project.id, kind="suite_node", title="train", payload={"suite_execution": {"version_id": version.id, "node_id": "train", "component_id": "train", "config": {"lr": 0.001}}})
        session.add_all([installation, task])
        session.flush()
        attempt = TaskAttempt(task_id=task.id, attempt_number=1, worker_id=worker.id)
        session.add(attempt)
        session.flush()
        task.active_attempt_id = attempt.id
        session.add(SuiteEnvironmentLease(installation_id=installation.id, task_id=task.id))
        session.flush()
        receipt = {"suite_execution": task.payload["suite_execution"], "component": component, "resolved_config": {"lr": 0.001}, "environment": env}
        with pytest.raises(ValueError, match="environment"):
            record_receipt(session, task, attempt, {**receipt, "environment": {**env, "generation": "wrong"}})
        row = record_receipt(session, task, attempt, receipt)
        assert record_receipt(session, task, attempt, receipt).id == row.id
        with pytest.raises(ValueError, match="Immutable"):
            record_receipt(session, task, attempt, {**receipt, "backends": [{"id": "changed"}]})
        metadata = {"contract": contract, "config": {"lr": 0.001}, "step": 2,
                    "execution": {"schema": "forgelm.native-training-execution.v1", "start_step": 0, "end_step": 2, "dispatch": {"partial_rope": 4}}}
        state = artifact_storage.put_bytes(session, project_id=project.id, kind="training_state_checkpoint", payload=b"state", filename="state.bin", metadata=metadata)
        data = {"artifact_id": state.id, "metadata": metadata}
        with pytest.raises(ValueError, match="ownership"):
            register_checkpoint(session, task, attempt, data)
        append_task_event(session, task, "artifact.created", {"artifact_id": state.id}, attempt.id)
        session.flush()
        checkpoint = register_checkpoint(session, task, attempt, data)
        from llm_platform.models import ExecutionSnapshot
        interval = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == attempt.id, ExecutionSnapshot.phase == "interval:2"))
        assert interval.snapshot["dispatch"] == {"partial_rope": 4}
        assert interval.snapshot["checkpoint_id"] == checkpoint.id
        assert register_checkpoint(session, task, attempt, data).id == checkpoint.id
        with pytest.raises(ValueError, match="metadata"):
            register_checkpoint(session, task, attempt, {**data, "metadata": {**metadata, "step": 3}})
