from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from forgelm_plugin_sdk.validation import defaults, validate_schema_instance
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models import Artifact, Project, Task, WorkerNode
from ..suite_models import (
    Suite,
    SuiteConfigRevision,
    SuiteEnvironmentLease,
    SuiteInstallation,
    SuiteOperation,
    SuiteOperationItem,
    SuiteOperationPlan,
    SuiteServiceRequest,
    SuiteVersion,
    WorkerSuiteDeployment,
)
from .artifacts import artifact_storage

TERMINAL = {"succeeded", "failed", "cancelled", "lost"}


def fail(code: str, message: str, status: int = 409):
    raise HTTPException(status, {"code": code, "message": message})


def get(session: Session, cls, key):
    value = session.get(cls, key)
    if not value:
        fail("not_found", f"{cls.__name__} 不存在", 404)
    return value


def system_project(session: Session) -> Project:
    existing = session.scalar(select(Project).where(Project.system_kind == "suite_packages"))
    if existing:
        return existing
    from .global_data import ensure_global_data_project
    global_project = ensure_global_data_project(session)
    workspace = Path(global_project.workspace_uri).parent / "suite-packages"
    workspace.mkdir(parents=True, exist_ok=True)
    project = Project(name="__forgelm_suite_packages__", target_model_name="suite-packages", workspace_uri=str(workspace), system_kind="suite_packages")
    session.add(project)
    session.flush()
    from ..config import get_settings
    if get_settings().storage_root:
        from .storage_relocation import project_target
        workspace = project_target(project)
        workspace.mkdir(parents=True, exist_ok=True)
        project.workspace_uri = str(workspace)
    return project


def version_view(session: Session, version: SuiteVersion) -> dict:
    suite = get(session, Suite, version.suite_id)
    return {**version.to_dict(), "suite": suite.to_dict(), "manifest": version.package["manifest"], "nodes": version.package["nodes"]}


def deployments(session: Session) -> list[dict]:
    result = []
    for deployment in session.scalars(select(WorkerSuiteDeployment)):
        installed = session.scalars(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id)).all()
        worker = session.get(WorkerNode, deployment.worker_id)
        result.append({**deployment.to_dict(), "worker_name": worker.name if worker else "", "worker_status": worker.status if worker else "offline", "installations": [r.to_dict() for r in installed]})
    return result


def target_for(version: SuiteVersion, worker: WorkerNode) -> dict:
    runtime = (worker.resources or {}).get("suite_runtime", {})
    from forgelm_plugin_sdk.package import accepts_range
    if not accepts_range(runtime.get("plugin_host", "1.0.0"), version.package["manifest"]["compatibility"]["pluginHost"]):
        fail("incompatible_host", "目标套件要求更新的 Worker 插件宿主")
    for target in version.package["manifest"]["targets"]:
        if all(target[k] == runtime.get(k) for k in ("os", "arch", "python")):
            if target["accelerator"] == "cpu" or (worker.resources or {}).get("gpus"):
                return target
    fail("incompatible_worker", f"{worker.name} 未上报兼容的套件运行时，需更新 Worker 或选择支持的平台")


def get_deployment(session: Session, worker_id: str, suite_id: str, create=False):
    row = session.scalar(select(WorkerSuiteDeployment).where(WorkerSuiteDeployment.worker_id == worker_id, WorkerSuiteDeployment.suite_id == suite_id))
    if row is None and create:
        row = WorkerSuiteDeployment(worker_id=worker_id, suite_id=suite_id)
        session.add(row)
        session.flush()
    return row


def tasks_using(session: Session, worker_id: str, version_ids: list[str]) -> list[Task]:
    result = []
    for task in session.scalars(select(Task).where(Task.adapter_id == "builtin.suite-runner", Task.observed_state.not_in(TERMINAL))):
        execution = task.payload.get("suite_execution", {})
        if execution.get("version_id") not in version_ids:
            continue
        requested = task.payload.get("resource_request", {}).get("worker_selector", {}).get("worker_id")
        if not requested or requested == worker_id:
            result.append(task)
    return result


def make_plan(session: Session, action: str, version_id: str, worker_ids: list[str], *, version_ids=None, drain=False) -> SuiteOperationPlan:
    if action not in {"install", "update", "rollback", "repair", "uninstall"}:
        fail("invalid_action", "不支持的套件操作", 422)
    version = get(session, SuiteVersion, version_id)
    suite = get(session, Suite, version.suite_id)
    if not worker_ids or len(worker_ids) > 100 or len(set(worker_ids)) != len(worker_ids):
        fail("invalid_workers", "请选择不重复的目标 Worker", 422)
    if action != "uninstall" and (not version.enabled or suite.archived):
        fail("suite_disabled", "该套件版本已停用或归档")
    items = []
    for worker_id in worker_ids:
        worker = get(session, WorkerNode, worker_id)
        if worker.status != "online" or "builtin.suite-manager" not in (worker.capabilities or []):
            fail("worker_unavailable", f"{worker.name} 不在线或尚未更新套件安装器")
        deployment = get_deployment(session, worker_id, suite.id, create=True)
        if deployment.active_operation_id:
            fail("suite_busy", "该 Worker 上有未完成的套件操作")
        installed = session.scalars(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.state == "ready")).all()
        selected = list(version_ids) if version_ids is not None else [i.suite_version_id for i in installed]
        if action == "uninstall":
            if not selected or set(selected) - {i.suite_version_id for i in installed}:
                fail("invalid_uninstall_versions", "卸载必须选择此 Worker 上已安装的版本", 422)
        else:
            selected = [version.id]
        in_use = tasks_using(session, worker_id, selected) if action in {"uninstall", "repair"} else []
        queued = [t for t in in_use if t.observed_state == "queued"]
        references = list(session.scalars(select(SuiteConfigRevision.id).where(SuiteConfigRevision.suite_version_id.in_(selected)))) if action == "uninstall" else []
        old = get(session, SuiteVersion, deployment.default_version_id) if deployment.default_version_id else None
        old_nodes = {n["id"] for n in old.package["nodes"]} if old else set()
        new_nodes = {n["id"] for n in version.package["nodes"]}
        items.append({"worker_id": worker_id, "worker_name": worker.name, "deployment_id": deployment.id, "expected_generation": deployment.generation,
                      "target_id": target_for(version, worker)["id"] if action != "uninstall" else None, "version_ids": selected,
                      "in_use": [{"id": t.id, "title": t.title, "status": t.observed_state} for t in in_use], "config_references": references,
                      "blocked": bool(queued or in_use and not drain), "download_bytes": version.package["size_bytes"] if action != "uninstall" else 0,
                      "expanded_bytes": version.package["expanded_bytes"], "disk_bytes": version.package["size_bytes"] + version.package["expanded_bytes"] + version.package["installed_bytes"] + 128 * 1024**2, "previous_version": old.version if old else None,
                      "diff": {"added_nodes": sorted(new_nodes - old_nodes), "removed_nodes": sorted(old_nodes - new_nodes), "changed_nodes": [n["id"] for n in version.package["nodes"] if old and any(p["id"] == n["id"] and p != n for p in old.package["nodes"])], "dependencies_changed": bool(old and old.package["locks"] != version.package["locks"]), "previous_requirements": old.package["manifest"]["executionRequirements"] if old else {}, "requirements": version.package["manifest"]["executionRequirements"]}})
    plan = SuiteOperationPlan(action=action, spec={"suite_id": suite.id, "suite_name": suite.suite_id, "version_id": version.id, "digest": version.digest, "trusted": version.trusted, "drain": drain, "items": items})
    session.add(plan)
    session.flush()
    return plan


def start_operation(session: Session, plan: SuiteOperationPlan, key: str, trust_digest: str | None, accept_impact: bool) -> SuiteOperation:
    existing = session.scalar(select(SuiteOperation).where(SuiteOperation.idempotency_key == key))
    if existing:
        if existing.plan_id != plan.id:
            fail("idempotency_conflict", "该幂等键属于另一个操作计划")
        return existing
    created = plan.created_at.replace(tzinfo=UTC) if plan.created_at.tzinfo is None else plan.created_at
    if datetime.now(UTC) - created > timedelta(minutes=15):
        fail("plan_expired", "操作计划已过期，请重新检查")
    spec = plan.spec
    version = get(session, SuiteVersion, spec["version_id"])
    if plan.action != "uninstall":
        if not version.enabled or get(session, Suite, version.suite_id).archived:
            fail("suite_disabled", "套件不可用")
        if not version.trusted and trust_digest != version.digest:
            fail("trust_required", "首次执行需明确信任此包摘要")
    if any(i["config_references"] for i in spec["items"]) and not accept_impact:
        fail("references_require_acknowledgement", "卸载影响已有配置，请确认影响计划")
    if any(i["blocked"] for i in spec["items"]):
        fail("suite_in_use", "存在占用任务；请先处理排队任务或选择排空后卸载")
    operation = SuiteOperation(plan_id=plan.id, idempotency_key=key, action=plan.action, spec=spec)
    session.add(operation)
    session.flush()
    from .tasks import create_task
    for item in spec["items"]:
        worker = get(session, WorkerNode, item["worker_id"])
        if worker.status != "online":
            fail("worker_offline", "目标 Worker 已离线")
        changed = session.execute(update(WorkerSuiteDeployment).where(WorkerSuiteDeployment.id == item["deployment_id"], WorkerSuiteDeployment.generation == item["expected_generation"], WorkerSuiteDeployment.active_operation_id.is_(None)).values(active_operation_id=operation.id, generation=item["expected_generation"] + 1, blocked_versions=item["version_ids"] if plan.action in {"uninstall", "repair"} else []))
        if changed.rowcount != 1:
            fail("stale_plan", "部署状态已变化，请刷新计划")
        active = tasks_using(session, worker.id, item["version_ids"]) if plan.action in {"uninstall", "repair"} else []
        if any(t.observed_state == "queued" for t in active) or active and not spec["drain"]:
            fail("suite_in_use", "计划生成后新增了占用任务")
        artifact = get(session, Artifact, version.artifact_id)
        payload = {"action": plan.action, "suite_id": spec["suite_name"], "operation_id": operation.id, "generation": item["expected_generation"] + 1, "version_id": version.id, "version_ids": item["version_ids"], "target_id": item["target_id"], "trusted_digest": version.digest,
                   "package": {"artifact_id": artifact.id, "digest": artifact.content_hash, "size_bytes": artifact.size_bytes}, "package_project_id": system_project(session).id}
        task = create_task(session, project_id=system_project(session).id, kind="suite_operation", title=f"{plan.action} {spec['suite_name']} {version.version}", adapter_id="builtin.suite-manager", payload={"suite_operation": payload, "resource_request": {"cpus": 1, "worker_selector": {"worker_id": worker.id, "capabilities": ["builtin.suite-manager"]}}, "command": {}})
        session.add(SuiteOperationItem(operation_id=operation.id, deployment_id=item["deployment_id"], task_id=task.id, generation=payload["generation"]))
    if plan.action != "uninstall" and trust_digest == version.digest:
        version.trusted = True
    return operation


def operation_view(session: Session, operation: SuiteOperation) -> dict:
    items = session.scalars(select(SuiteOperationItem).where(SuiteOperationItem.operation_id == operation.id)).all()
    return {**operation.to_dict(), "items": [i.to_dict() for i in items]}


def retry_operation(session: Session, operation: SuiteOperation):
    from sqlalchemy import or_

    from .tasks import create_task
    retried = False
    items = session.scalars(select(SuiteOperationItem).where(SuiteOperationItem.operation_id == operation.id)).all()
    for item in items:
        if item.status not in {"failed", "cancelled", "lost"}:
            continue
        old = get(session, Task, item.task_id)
        deployment = get(session, WorkerSuiteDeployment, item.deployment_id)
        worker = get(session, WorkerNode, deployment.worker_id)
        if worker.status != "online":
            fail("worker_offline", "Worker 离线，暂不能重试或对账")
        if deployment.generation != item.generation or deployment.active_operation_id not in {None, operation.id}:
            fail("stale_operation", "已有更新的部署操作，不能重放旧操作")
        previous_status = item.status
        acquired = session.execute(update(SuiteOperationItem).where(SuiteOperationItem.id == item.id, SuiteOperationItem.task_id == old.id, SuiteOperationItem.status.in_({"failed", "cancelled", "lost"})).values(status="queued"))
        if acquired.rowcount != 1:
            continue
        fenced = session.execute(update(WorkerSuiteDeployment).where(WorkerSuiteDeployment.id == deployment.id, WorkerSuiteDeployment.generation == item.generation, or_(WorkerSuiteDeployment.active_operation_id.is_(None), WorkerSuiteDeployment.active_operation_id == operation.id)).values(active_operation_id=operation.id))
        if fenced.rowcount != 1:
            fail("stale_operation", "已有更新的部署操作，不能重放旧操作")
        deployment.blocked_versions = old.payload["suite_operation"]["version_ids"] if operation.action in {"uninstall", "repair"} else []
        task = create_task(session, project_id=old.project_id, kind=old.kind, title=old.title + " · 重试/对账", adapter_id=old.adapter_id, payload=old.payload)
        item.result = {"retry_history": [*(item.result.get("retry_history", [])), {"task_id": old.id, "status": previous_status, "error": old.error}]}
        item.task_id, item.status = task.id, "queued"
        retried = True
    if retried:
        operation.status = "running"
    session.flush()
    return operation


def finalize_operation(session: Session, task: Task, result: dict):
    item = session.scalar(select(SuiteOperationItem).where(SuiteOperationItem.task_id == task.id))
    if not item:
        raise ValueError("Suite operation item missing")
    deployment = get(session, WorkerSuiteDeployment, item.deployment_id)
    spec = task.payload["suite_operation"]
    receipt = result.get("suite_receipt", {})
    if deployment.active_operation_id != item.operation_id or deployment.generation != item.generation:
        raise ValueError("Stale suite operation result")
    if any(receipt.get(k) != spec[k] for k in ("operation_id", "generation", "suite_id", "action")):
        raise ValueError("Suite operation receipt identity mismatch")
    if not isinstance(receipt.get("installations"), list):
        raise ValueError("Incomplete suite installation receipt")
    previous = set(session.scalars(select(SuiteInstallation.suite_version_id).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.state == "ready")))
    expected = previous - set(spec["version_ids"]) if spec["action"] == "uninstall" else previous | {spec["version_id"]}
    if {i.get("version_id") for i in receipt["installations"]} != expected:
        raise ValueError("Receipt must account for every retained and removed version")
    seen = set()
    for installed in receipt["installations"]:
        version = get(session, SuiteVersion, installed["version_id"])
        if version.suite_id != deployment.suite_id or installed["digest"] != version.digest or version.id in seen:
            raise ValueError("Suite installation content mismatch")
        seen.add(version.id)
        target = next((t for t in version.package["manifest"]["targets"] if t["id"] == installed.get("target_id")), None)
        if not target or set(installed.get("groups", {})) != set(target["runtimeGroups"]):
            raise ValueError("Invalid installed runtime groups")
        if any(installed["groups"][group].get("lock_digest") != version.package["locks"][target["id"] + "/" + group]["digest"] for group in target["runtimeGroups"]):
            raise ValueError("Installed dependency lock does not match package")
        component_ids = {c["id"] for c in version.package["manifest"]["components"]}
        if set(installed.get("probes", {})) != component_ids or not all(p.get("ready") is True for p in installed["probes"].values()):
            raise ValueError("Suite components have not all passed validation")
        row = session.scalar(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.suite_version_id == version.id))
        if row is None:
            row = SuiteInstallation(deployment_id=deployment.id, suite_version_id=version.id, generation=installed["generation"])
            session.add(row)
        row.generation, row.receipt, row.state = installed["generation"], installed, "ready"
    if spec["action"] == "uninstall":
        if seen.intersection(spec["version_ids"]):
            raise ValueError("Removed suite version remains in receipt")
        for row in session.scalars(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.suite_version_id.in_(spec["version_ids"]))):
            row.state = "removed"
    elif spec["version_id"] not in seen:
        raise ValueError("Target suite version is not installed")
    default = receipt.get("default_version_id")
    if default and default not in seen:
        raise ValueError("Invalid suite default version")
    deployment.default_version_id = default
    deployment.active_operation_id = None
    deployment.blocked_versions = []
    item.status, item.result = "succeeded", {**receipt, "retry_history": item.result.get("retry_history", [])}


def finalize_run(session: Session, task: Task):
    spec = task.payload["suite_execution"]
    version = get(session, SuiteVersion, spec["version_id"])
    node = next(n for n in version.package["nodes"] if n["id"] == spec["node_id"])
    environment = task.result.get("environment", {})
    lease = session.scalar(select(SuiteEnvironmentLease).where(SuiteEnvironmentLease.task_id == task.id, SuiteEnvironmentLease.state == "active"))
    if not lease:
        raise ValueError("Suite result has no claimed environment lease")
    installation = get(session, SuiteInstallation, lease.installation_id)
    if task.result.get("suite_execution") != spec or any(environment.get(k) != installation.receipt.get(k) for k in ("generation", "target_id", "groups", "digest")):
        raise ValueError("Suite result environment identity mismatch")
    if node.get("training"):
        from ..models import ExecutionSnapshot, TrainingStateCheckpoint
        resolved = session.scalar(select(ExecutionSnapshot).where(ExecutionSnapshot.attempt_id == task.active_attempt_id, ExecutionSnapshot.phase == "resolved"))
        checkpoint = session.scalar(select(TrainingStateCheckpoint.id).where(TrainingStateCheckpoint.attempt_id == task.active_attempt_id))
        if not resolved or resolved.snapshot != task.result.get("execution_receipt") or not checkpoint:
            raise ValueError("Training result requires the persisted execution receipt and checkpoint")
    outputs = [r for r in task.result.get("output_artifacts", []) if r["kind"] == "suite_output"]
    declared = {p["name"]: p for p in node["outputs"]}
    ports = [r["ref"] for r in outputs]
    if len(ports) != len(set(ports)) or set(ports) - declared.keys() or any(p["required"] and name not in ports for name, p in declared.items()):
        raise ValueError("Suite result output ports do not match NodeSpec")
    for output in outputs:
        artifact = get(session, Artifact, output["artifact_id"])
        if artifact.metadata_json.get("format") != declared[output["ref"]]["format"]:
            raise ValueError("Suite result artifact format mismatch")


def sync_status(session: Session, task: Task):
    if task.adapter_id == "builtin.suite-manager":
        item = session.scalar(select(SuiteOperationItem).where(SuiteOperationItem.task_id == task.id))
        if item:
            item.status = task.observed_state
            if task.observed_state in {"failed", "cancelled", "lost"}:
                item.result = {"error": task.error}
                # Keep lost operations fenced until retried/reconciled; a lost
                # Worker may have committed locally without delivering a receipt.
                if task.observed_state != "lost":
                    row = get(session, WorkerSuiteDeployment, item.deployment_id)
                    if row.active_operation_id == item.operation_id:
                        row.active_operation_id = None
                        row.blocked_versions = []
            session.flush()
            operation = get(session, SuiteOperation, item.operation_id)
            states = list(session.scalars(select(SuiteOperationItem.status).where(SuiteOperationItem.operation_id == operation.id)))
            operation.status = "succeeded" if all(s == "succeeded" for s in states) else "failed" if all(s in TERMINAL for s in states) else "running"
    elif task.adapter_id == "builtin.suite-runner" and task.observed_state in (TERMINAL - {"lost"}) | {"paused"}:
        session.execute(update(SuiteEnvironmentLease).where(SuiteEnvironmentLease.task_id == task.id).values(state="released"))
    if task.kind == "suite_service" and task.observed_state in TERMINAL:
        for request in session.scalars(select(SuiteServiceRequest).where(SuiteServiceRequest.task_id == task.id, SuiteServiceRequest.status.in_({"queued", "running"}))):
            request.status, request.error = "failed", {"message": "推理服务已结束"}


def can_allocate(session: Session, worker: WorkerNode, task: Task) -> bool:
    if task.adapter_id == "builtin.suite-manager":
        spec = task.payload["suite_operation"]
        if spec["action"] in {"uninstall", "repair"}:
            using = tasks_using(session, worker.id, spec["version_ids"])
            if any(t.observed_state != "queued" for t in using):
                return False
        return True
    if task.adapter_id != "builtin.suite-runner":
        return True
    spec = task.payload["suite_execution"]
    version = session.get(SuiteVersion, spec["version_id"])
    if not version or not version.enabled or not version.trusted:
        return False
    from forgelm_plugin_sdk.package import accepts_range
    if not accepts_range((worker.resources or {}).get("suite_runtime", {}).get("plugin_host", "1.0.0"), version.package["manifest"]["compatibility"]["pluginHost"]):
        return False
    node = next(n for n in version.package["nodes"] if n["id"] == spec["node_id"])
    available = int((worker.resources or {}).get("memory_available_bytes", 0))
    if node["resources"]["memoryMiB"] * 1024**2 > available:
        return False
    deployment = get_deployment(session, worker.id, version.suite_id)
    if not deployment or version.id in deployment.blocked_versions:
        return False
    row = session.scalar(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.suite_version_id == version.id, SuiteInstallation.state == "ready"))
    return row is not None


def claim_environment(session: Session, worker: WorkerNode, task: Task):
    if task.adapter_id != "builtin.suite-runner":
        return
    version = get(session, SuiteVersion, task.payload["suite_execution"]["version_id"])
    deployment = get_deployment(session, worker.id, version.suite_id)
    # SQLite claim already holds a write transaction; this write serializes the
    # admission decision with operation creation on other supported databases.
    session.execute(update(WorkerSuiteDeployment).where(WorkerSuiteDeployment.id == deployment.id).values(generation=WorkerSuiteDeployment.generation))
    session.refresh(deployment)
    if not can_allocate(session, worker, task):
        raise ValueError("Suite admission changed during claim")
    row = session.scalar(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.suite_version_id == version.id))
    lease = session.scalar(select(SuiteEnvironmentLease).where(SuiteEnvironmentLease.installation_id == row.id, SuiteEnvironmentLease.task_id == task.id))
    if lease:
        lease.state = "active"
    else:
        session.add(SuiteEnvironmentLease(installation_id=row.id, task_id=task.id))


def create_revision(session: Session, project_id: str, version_id: str, node_id: str, config: dict, previous_id=None):
    get(session, Project, project_id)
    version = get(session, SuiteVersion, version_id)
    suite = get(session, Suite, version.suite_id)
    if not version.enabled or not version.trusted or suite.archived:
        fail("suite_unavailable", "请选择已信任、已启用的套件版本")
    node = next((n for n in version.package["nodes"] if n["id"] == node_id), None)
    if not node:
        fail("node_not_found", "套件中不存在该节点", 422)
    value = defaults(config, node["config_schema"])
    try:
        validate_schema_instance(value, node["config_schema"])
    except ValueError as exc:
        fail("invalid_config", str(exc), 422)
    if previous_id:
        previous = get(session, SuiteConfigRevision, previous_id)
        if previous.project_id != project_id:
            fail("revision_project_mismatch", "原配置不属于此项目", 422)
    revision = SuiteConfigRevision(project_id=project_id, suite_version_id=version.id, node_id=node_id, config=value, requested_config=config, previous_revision_id=previous_id)
    session.add(revision)
    session.flush()
    return revision


def run_revision(session: Session, revision: SuiteConfigRevision, inputs: dict, worker_id: str, timeout_seconds=86400, continuation=None):
    from .tasks import create_task
    version = get(session, SuiteVersion, revision.suite_version_id)
    worker = get(session, WorkerNode, worker_id)
    if not version.enabled or not version.trusted:
        fail("suite_unavailable", "套件版本未启用或未信任")
    deployment = get_deployment(session, worker.id, version.suite_id)
    if not deployment or version.id in deployment.blocked_versions:
        fail("suite_not_installed", "套件未安装或正在排空")
    installed = session.scalar(select(SuiteInstallation).where(SuiteInstallation.deployment_id == deployment.id, SuiteInstallation.suite_version_id == version.id, SuiteInstallation.state == "ready"))
    if not installed or worker.status != "online":
        fail("suite_not_ready", "Worker 上没有可用的精确套件版本")
    target_for(version, worker)
    node = next(n for n in version.package["nodes"] if n["id"] == revision.node_id)
    declarations = {p["name"]: p for p in node["inputs"]}
    if inputs.keys() - declarations.keys() or any(p["required"] and name not in inputs for name, p in declarations.items()):
        fail("invalid_inputs", "输入端口与节点声明不匹配", 422)
    frozen = {}
    for name, artifact_id in inputs.items():
        artifact = get(session, Artifact, artifact_id)
        if not artifact_storage.managed_payload_path(session, artifact, revision.project_id):
            fail("input_unavailable", "输入不是项目可访问的托管制品", 422)
        metadata = artifact.metadata_json or {}
        formats = {metadata.get("format"), (metadata.get("inference") or {}).get("format"), artifact.media_type}
        format_name = declarations[name]["format"]
        if format_name not in formats:
            fail("input_format_mismatch", f"{name} 输入格式不匹配", 422)
        frozen[name] = {"artifact_id": artifact.id, "digest": artifact.content_hash, "size_bytes": artifact.size_bytes, "format": format_name}
    suite = get(session, Suite, version.suite_id)
    execution = {"suite_id": suite.suite_id, "version_id": version.id, "version": version.version, "digest": version.digest, "component_id": node["component"], "node_id": node["id"], "node_digest": node["digest"], "config_revision_id": revision.id, "config": revision.config, "inputs": frozen, "timeout_seconds": timeout_seconds}
    if node.get("training"):
        execution["training"] = node["training"]
    execution["requested_config"] = revision.requested_config
    if continuation:
        from .training_continuation import prepare
        try:
            execution["continuation"] = prepare(session, revision.project_id, node, revision.config, frozen, **continuation)
        except ValueError as exc:
            fail("invalid_continuation", str(exc), 422)
    return create_task(session, project_id=revision.project_id, kind="suite_service" if node["execution"]["kind"] == "service" else "suite_node", title=node["displayName"], adapter_id="builtin.suite-runner", payload={"suite_execution": execution, "resource_request": {"cpus": node["resources"]["cpu"], "gpus": node["resources"]["gpu"], "worker_selector": {"worker_id": worker.id, "capabilities": ["builtin.suite-runner", "suite@" + version.digest]}}, "command": {}})


def validate_canvas_node(session: Session, node: dict, edges: list) -> list[dict]:
    config = node.get("config", {})
    try:
        version = get(session, SuiteVersion, config.get("suite_version_id"))
        if not version.enabled or not version.trusted:
            raise ValueError("套件版本未启用或未信任")
        spec = next((n for n in version.package["nodes"] if n["id"] == config.get("suite_node_id")), None)
        if not spec or spec["execution"]["kind"] != "task":
            raise ValueError("工作流节点必须选择套件中的批任务组件")
        validate_schema_instance(defaults(config.get("suite_config", {}), spec["config_schema"]), spec["config_schema"])
        get(session, WorkerNode, config.get("suite_worker_id"))
        inputs = config.get("suite_inputs", {})
        ports = {p["name"]: p for p in spec["inputs"]}
        if inputs.keys() - ports.keys():
            raise ValueError("存在未声明的输入端口")
        for name, port in ports.items():
            binding = inputs.get(name)
            if not binding and port["required"]:
                raise ValueError(f"缺少输入 {name}")
            if binding and "source_node_id" in binding and not any(e["source"] == binding["source_node_id"] and e["target"] == node["id"] for e in edges):
                raise ValueError(f"输入 {name} 引用的上游节点未连线")
        return []
    except (ValueError, HTTPException, TypeError, KeyError) as exc:
        return [{"severity": "error", "code": "invalid_suite_node", "node_id": node["id"], "message": str(exc)}]


def canvas_task(session: Session, run, node, row, context):
    config = node["config"]
    revision = create_revision(session, run.project_id, config["suite_version_id"], config["suite_node_id"], config.get("suite_config", {}))
    inputs = {}
    for name, binding in config.get("suite_inputs", {}).items():
        if binding.get("artifact_id"):
            inputs[name] = binding["artifact_id"]
        elif binding.get("source_node_id"):
            output = context.get("upstream", {}).get(binding["source_node_id"], {})
            candidates = output.get("output_artifacts", [])
            found = next((o for o in candidates if o.get("ref") == binding.get("output_ref", "result")), None)
            if not found:
                fail("upstream_output_missing", f"上游未产生输入 {name} 所需制品", 422)
            inputs[name] = found["artifact_id"]
    task = run_revision(session, revision, inputs, config["suite_worker_id"])
    task.payload = {**task.payload, "pipeline_run_id": run.id, "pipeline_instance_id": run.instance_id, "pipeline_node_run_id": row.id, "pipeline_node_id": row.node_id, "pipeline_context": context}
    row.task_id, row.status = task.id, "queued"
    row.input_snapshot = {**context, "suite_execution": task.payload["suite_execution"], "consumed_assets": [{"asset_type": "artifact", "asset_id": ref["artifact_id"], "role": name, "metadata": {"content_hash": ref["digest"]}} for name, ref in task.payload["suite_execution"]["inputs"].items()]}
    return task
