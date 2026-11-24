"""Append-only attempt provenance. Telemetry is not the source of configuration truth."""
from __future__ import annotations

from copy import deepcopy

from forgelm_plugin_sdk.training import digest
from sqlalchemy import select

from ..models import Artifact, ExecutionSnapshot, TaskEvent


def append_snapshot(session, attempt, phase, value):
    frozen = deepcopy(value)
    sha = digest(frozen)
    existing = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == attempt.id, ExecutionSnapshot.phase == phase))
    if existing:
        if existing.digest != sha:
            raise ValueError(f"Immutable {phase} execution snapshot already exists")
        return existing
    row = ExecutionSnapshot(task_id=attempt.task_id, attempt_id=attempt.id, phase=phase, digest=sha, snapshot=frozen)
    session.add(row)
    session.flush()
    return row


def freeze_intent(session, task, attempt, allocation):
    # Includes native code/config bindings as well as complete suite identities.
    value = {"schema": "forgelm.execution-intent.v1", "task_id": task.id, "attempt_id": attempt.id,
             "adapter_id": task.adapter_id, "launcher_id": task.launcher_id,
             "payload": deepcopy(task.payload), "worker_id": attempt.worker_id, "allocation": allocation}
    spec = task.payload.get("suite_execution")
    if spec:
        from ..suite_models import SuiteVersion
        version = session.get(SuiteVersion, spec["version_id"])
        value["component"] = next(c for c in version.package["manifest"]["components"] if c["id"] == spec["component_id"])
        value["package_targets"] = version.package["targets"] if "targets" in version.package else version.package["manifest"]["targets"]
        value["dependency_locks"] = version.package.get("locks", {})
    return append_snapshot(session, attempt, "intent", value)


def record_receipt(session, task, attempt, receipt):
    if attempt.ended_at or task.active_attempt_id != attempt.id:
        raise ValueError("Execution receipt requires the active attempt")
    spec = task.payload.get("suite_execution")
    if not spec:
        if task.kind not in {"train", "training_performance"} or receipt.get("schema") != "forgelm.microlm-optimizations.v1":
            raise ValueError("Unsupported native execution receipt")
        if any((receipt.get("code") or {}).get(k) != v for k, v in (task.payload.get("code") or {}).items()):
            raise ValueError("Native receipt code identity differs from frozen task")
        if receipt.get("resolved_config") != {"stages": task.payload.get("stages")}:
            raise ValueError("Native receipt configuration differs from frozen task")
        return append_snapshot(session, attempt, "resolved", receipt)
    if not spec or receipt.get("suite_execution") != spec:
        raise ValueError("Execution receipt does not match frozen suite identity")
    from ..suite_models import SuiteEnvironmentLease, SuiteInstallation, SuiteVersion
    lease = session.scalar(select(SuiteEnvironmentLease).where(SuiteEnvironmentLease.task_id == task.id, SuiteEnvironmentLease.state == "active"))
    installation = session.get(SuiteInstallation, lease.installation_id) if lease else None
    if not installation or any(receipt.get("environment", {}).get(k) != installation.receipt.get(k) for k in ("generation", "target_id", "groups", "digest")):
        raise ValueError("Execution receipt does not match leased environment")
    version = session.get(SuiteVersion, spec["version_id"])
    component = next(c for c in version.package["manifest"]["components"] if c["id"] == spec["component_id"])
    if receipt.get("component") != component or receipt.get("resolved_config") != spec["config"]:
        raise ValueError("Execution receipt component/config mismatch")
    return append_snapshot(session, attempt, "resolved", receipt)


def register_checkpoint(session, task, attempt, data):
    from ..models import TrainingStateCheckpoint
    from ..suite_models import SuiteVersion
    if attempt.ended_at or task.active_attempt_id != attempt.id:
        raise ValueError("Checkpoint requires the active attempt")
    spec = task.payload.get("suite_execution", {})
    version = session.get(SuiteVersion, spec.get("version_id"))
    node = next((n for n in version.package["nodes"] if n["id"] == spec["node_id"]), {}) if version else {}
    contract = node.get("training")
    artifact = session.get(Artifact, data.get("artifact_id"))
    uploaded = session.scalar(select(TaskEvent.id).where(TaskEvent.task_id == task.id, TaskEvent.attempt_id == attempt.id, TaskEvent.event_type == "artifact.created", TaskEvent.payload["artifact_id"].as_string() == data.get("artifact_id")))
    if not contract or not artifact or artifact.kind != "training_state_checkpoint" or not uploaded:
        raise ValueError("Invalid checkpoint artifact ownership or capability")
    metadata = data.get("metadata", {})
    if any(artifact.metadata_json.get(k) != v for k, v in metadata.items()):
        raise ValueError("Checkpoint announcement differs from uploaded artifact metadata")
    if metadata.get("contract") != contract or metadata.get("config") != spec["config"] or type(metadata.get("step")) is not int or metadata["step"] < 0:
        raise ValueError("Checkpoint does not match frozen training contract/config")
    resolved = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == attempt.id, ExecutionSnapshot.phase == "resolved"))
    if not resolved:
        raise ValueError("Checkpoint requires a persisted execution receipt")
    existing = session.scalar(select(TrainingStateCheckpoint).where(TrainingStateCheckpoint.attempt_id == attempt.id, TrainingStateCheckpoint.step == metadata["step"]))
    if existing:
        if existing.artifact_id != artifact.id or existing.manifest != metadata:
            raise ValueError("Checkpoint step is immutable")
        return existing
    row = TrainingStateCheckpoint(project_id=task.project_id, task_id=task.id, attempt_id=attempt.id,
                                  artifact_id=artifact.id, step=metadata["step"], manifest=metadata, execution_digest=resolved.digest)
    session.add(row)
    session.flush()
    execution = metadata.get("execution")
    if execution and execution.get("schema") == "forgelm.native-training-execution.v1":
        record_interval(session, task, attempt, {
            "receipt_digest": resolved.digest,
            "start_step": execution["start_step"], "end_step": execution["end_step"],
            "dispatch": execution["dispatch"], "checkpoint_id": row.id,
        })
    return row


def snapshots_for_task(session, task_id):
    return [r.to_dict() for r in session.scalars(select(ExecutionSnapshot).where(ExecutionSnapshot.task_id == task_id).order_by(ExecutionSnapshot.created_at))]


def record_interval(session, task, attempt, value):
    if attempt.ended_at or task.active_attempt_id != attempt.id:
        raise ValueError("Execution interval requires the active attempt")
    resolved = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == attempt.id, ExecutionSnapshot.phase == "resolved"))
    if not resolved or value.get("receipt_digest") != resolved.digest:
        raise ValueError("Interval does not reference this attempt's persisted receipt")
    start, end = value.get("start_step"), value.get("end_step")
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("Invalid execution step interval")
    previous = list(session.scalars(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == attempt.id, ExecutionSnapshot.phase.like("interval:%"))))
    replay = next((row for row in previous if row.phase == f"interval:{end}"), None)
    if replay is None:
        expected_start = max((row.snapshot["end_step"] for row in previous), default=resolved.snapshot.get("start_step", resolved.snapshot.get("native_engine", {}).get("start_step", 0)))
        if start != expected_start:
            raise ValueError("Execution intervals must be contiguous and ordered")
    if any(type(count) is not int or count < 0 for count in value.get("dispatch", {}).values()):
        raise ValueError("Invalid backend dispatch count")
    return append_snapshot(session, attempt, f"interval:{end}", value)
