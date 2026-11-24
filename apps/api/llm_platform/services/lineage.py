from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Checkpoint,
    CodeSource,
    CodeVersion,
    EvaluationRun,
    LineageAssetRelation,
    LineageExecution,
    ModelVersion,
    ModelWeightAsset,
    ModelWeightVersion,
    PlanRevision,
    ProjectPipelineNodeRun,
    ProjectPipelineRun,
    QuantizationRun,
    Release,
    Task,
    TokenizerVersion,
    TrainingPlan,
)
from .global_data import asset_descriptor

TERMINAL_LINEAGE_STATUSES = {"succeeded", "failed", "cancelled", "skipped"}
RELATION_KINDS = {"consumed", "produced"}
EXECUTION_ONLY_ASSET_TYPES = {"training_run"}
OUTPUT_AS_INPUT_TYPES = {"plan_revision"}
IDENTITY_FIELDS = {
    "task_attempt_id",
    "adapter_id",
    "launcher_id",
    "engine",
    "engine_version",
    "code_version_id",
    "recipe_version_id",
    "plan_revision_id",
    "training_run_id",
    "task_id",
    "worker_id",
    "image",
    "image_digest",
    "config_hash",
    "command_digest",
    "executable",
    "source",
}
CONFIG_ASSET_FIELDS = {
    "tokenizer_version_id": "tokenizer_version",
    "code_version_id": "code_version",
    "plan_revision_id": "plan_revision",
    "model_version_id": "model_version",
    "evaluation_run_id": "evaluation_run",
    "quantization_run_id": "quantization_run",
    "release_id": "release",
    "recipe_version_id": "recipe_version",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _asset_type(value: Any) -> str:
    return str(value or "").strip()


def _is_asset_type(value: str) -> bool:
    return bool(
        value
        and (
            value.endswith("_version")
            or value.endswith("_revision")
            or value
            in {
                "artifact",
                "checkpoint",
                "evaluation_run",
                "quantization_run",
                "release",
                "recipe_version",
                "training_run",
            }
        )
    )


def _normalize_asset_ref(
    value: Any,
    *,
    role: str = "",
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    kind = _asset_type(value.get("asset_type") or value.get("kind"))
    asset_id = str(
        value.get("asset_version_id")
        or value.get("asset_id")
        or value.get("id")
        or ""
    ).strip()
    if not asset_id or not _is_asset_type(kind) or kind in EXECUTION_ONLY_ASSET_TYPES:
        return None
    version = value.get("version")
    if version is None and kind == "plan_revision":
        version = value.get("revision")
    label = str(value.get("label") or value.get("name") or "").strip()
    if not label:
        suffix = f" · V{version}" if version not in (None, "") else ""
        label = f"{kind}{suffix}" if suffix else f"{kind} · {asset_id[:8]}"
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    return {
        "asset_type": kind,
        "asset_id": asset_id,
        "role": str(value.get("role") or role or ""),
        "label": label,
        "version": str(version) if version not in (None, "") else None,
        "status": str(value.get("status") or ""),
        "metadata": deepcopy(metadata),
    }


def _collect_refs(value: Any, *, role: str = "") -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            refs.extend(_collect_refs(item, role=role))
        return refs
    if not isinstance(value, dict):
        return refs
    direct = _normalize_asset_ref(value, role=role)
    if direct is not None:
        refs.append(direct)
    for key, item in value.items():
        if key in {"metadata", "asset"}:
            continue
        child_role = role
        if key in {"versions", "consumed_assets", "produced_assets", "data_refs"}:
            child_role = key.removesuffix("_assets")
        refs.extend(_collect_refs(item, role=child_role))
    return refs


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for ref in refs:
        key = (ref["asset_type"], ref["asset_id"], ref.get("role") or "")
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def _database_asset_ref(
    session: Session,
    asset_type: str,
    asset_id: str,
    *,
    role: str = "",
) -> dict[str, Any]:
    descriptor = asset_descriptor(session, asset_type, asset_id)
    if descriptor is not None:
        return _normalize_asset_ref(descriptor, role=role) or {
            "asset_type": asset_type,
            "asset_id": asset_id,
            "role": role,
            "label": f"{asset_type} · {asset_id[:8]}",
            "version": None,
            "status": "",
            "metadata": {},
        }

    label = ""
    version: Any = None
    status = ""
    if asset_type == "tokenizer_version":
        record = session.get(TokenizerVersion, asset_id)
        if record is not None:
            label, version, status = record.name, record.version, record.status
    elif asset_type == "code_version":
        record = session.get(CodeVersion, asset_id)
        source = session.get(CodeSource, record.source_id) if record is not None else None
        if record is not None:
            label = source.name if source is not None else "代码版本"
            version, status = record.version, record.state
    elif asset_type == "plan_revision":
        record = session.get(PlanRevision, asset_id)
        plan = session.get(TrainingPlan, record.plan_id) if record is not None else None
        if record is not None:
            label = plan.name if plan is not None else "训练计划"
            version, status = record.revision, "locked" if record.locked else "draft"
    elif asset_type == "checkpoint":
        record = session.get(Checkpoint, asset_id)
        if record is not None:
            label, version, status = "Checkpoint", record.step, record.status
    elif asset_type == "model_version":
        record = session.get(ModelVersion, asset_id)
        if record is not None:
            label, version, status = record.name, record.version, record.status
    elif asset_type == "model_weight_version":
        record = session.get(ModelWeightVersion, asset_id)
        if record is not None:
            asset = session.get(ModelWeightAsset, record.asset_id)
            label, version, status = asset.name if asset else "本地权重", record.version, record.state
    elif asset_type == "evaluation_run":
        record = session.get(EvaluationRun, asset_id)
        if record is not None:
            label, status = "评测运行", record.status
    elif asset_type == "quantization_run":
        record = session.get(QuantizationRun, asset_id)
        if record is not None:
            label, status = "量化运行", record.status
    elif asset_type == "release":
        record = session.get(Release, asset_id)
        if record is not None:
            label, status = "发布版本", record.status
    suffix = ""
    if version not in (None, ""):
        suffix = f" · {'R' if asset_type == 'plan_revision' else 'V'}{version}"
    return {
        "asset_type": asset_type,
        "asset_id": asset_id,
        "role": role,
        "label": f"{label}{suffix}" if label else f"{asset_type} · {asset_id[:8]}",
        "version": str(version) if version not in (None, "") else None,
        "status": str(status or ""),
        "metadata": {},
    }


def _config_refs(
    session: Session,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    asset_type = _asset_type(config.get("asset_type"))
    asset_id = str(config.get("asset_version_id") or "").strip()
    if asset_type and asset_id:
        refs.append(_database_asset_ref(session, asset_type, asset_id, role="binding"))
    for field, kind in CONFIG_ASSET_FIELDS.items():
        record_id = str(config.get(field) or "").strip()
        if record_id:
            refs.append(_database_asset_ref(session, kind, record_id, role=field))
    for item in config.get("data_refs") or []:
        if not isinstance(item, dict):
            continue
        kind = _asset_type(item.get("asset_type"))
        record_id = str(item.get("asset_version_id") or "").strip()
        if kind and record_id:
            refs.append(
                _database_asset_ref(
                    session,
                    kind,
                    record_id,
                    role=str(item.get("purpose") or "data"),
                )
            )
    return refs


def _legacy_mapping_refs(
    session: Session,
    mapping: dict[str, Any],
    node_id: str,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if node_id in {"data", "training"}:
        for item in mapping.get("data") or []:
            if not isinstance(item, dict):
                continue
            kind = _asset_type(item.get("asset_type"))
            asset_id = str(item.get("asset_version_id") or item.get("id") or "")
            if kind and asset_id:
                refs.append(_normalize_asset_ref({**item, "id": asset_id}, role="training_data") or _database_asset_ref(session, kind, asset_id, role="training_data"))
    for key, kind, nodes in (
        ("tokenizer", "tokenizer_version", {"tokenizer", "training"}),
        ("code", "code_version", {"code", "training"}),
        ("plan", "plan_revision", {"training"}),
    ):
        item = mapping.get(key)
        if node_id not in nodes or not isinstance(item, dict) or not item.get("id"):
            continue
        ref = _normalize_asset_ref(
            {**item, "kind": kind, "label": item.get("label") or item.get("name")},
            role=key,
        )
        refs.append(ref or _database_asset_ref(session, kind, str(item["id"]), role=key))
    return refs


def _safe_execution_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        field: deepcopy(value[field])
        for field in IDENTITY_FIELDS
        if field in value and value[field] not in (None, "", [], {})
    }


def _execution_identity(
    session: Session,
    run: ProjectPipelineRun,
    node_run: ProjectPipelineNodeRun,
    config: dict[str, Any],
    output: dict[str, Any],
    *,
    execution_owner: str,
) -> dict[str, Any]:
    identity = _safe_execution_identity(output.get("execution_identity"))
    identity["training_instance_id"] = run.instance_id
    identity["execution_attempt_id"] = run.id
    if (node_run.input_snapshot or {}).get("task_attempt_id"):
        identity["task_attempt_id"] = node_run.input_snapshot["task_attempt_id"]
    task = session.get(Task, node_run.task_id) if node_run.task_id else None
    if task is not None:
        suite = (task.payload or {}).get("suite_execution")
        if suite:
            identity.update({"suite_id": suite["suite_id"], "suite_version_id": suite["version_id"], "suite_version": suite["version"], "package_digest": suite["digest"], "component_id": suite["component_id"], "node_spec_digest": suite["node_digest"], "suite_config_revision_id": suite["config_revision_id"]})
        identity.update(
            {
                "task_id": task.id,
                "adapter_id": task.adapter_id,
                "launcher_id": task.launcher_id,
            }
        )
        from ..models import ExecutionSnapshot
        identity["execution_snapshots"] = [{"id": row.id, "attempt_id": row.attempt_id, "phase": row.phase, "digest": row.digest}
                                           for row in session.scalars(select(ExecutionSnapshot).where(ExecutionSnapshot.task_id == task.id))]
    identity.setdefault("adapter_id", config.get("adapter_id"))
    identity.setdefault("launcher_id", config.get("launcher_id"))
    for field in (
        "code_version_id",
        "recipe_version_id",
        "plan_revision_id",
        "training_run_id",
    ):
        if config.get(field):
            identity.setdefault(field, config[field])
    metadata = (run.context_snapshot or {}).get("__forgelm")
    if isinstance(metadata, dict):
        if metadata.get("source_training_run_id"):
            identity.setdefault("training_run_id", metadata["source_training_run_id"])
        if metadata.get("source_task_id"):
            identity.setdefault("task_id", metadata["source_task_id"])
    command = config.get("command") if isinstance(config.get("command"), dict) else {}
    if command.get("executable"):
        identity.setdefault("executable", command["executable"])
        identity.setdefault("command_digest", _content_hash(command))
    identity.setdefault("config_hash", _content_hash(config))
    identity["execution_owner"] = execution_owner
    return {key: value for key, value in identity.items() if value not in (None, "")}


def record_terminal_node_lineage(
    session: Session,
    run: ProjectPipelineRun,
    node_run: ProjectPipelineNodeRun,
    *,
    input_snapshot: dict[str, Any] | None = None,
    output_snapshot: dict[str, Any] | None = None,
    execution_owner: str = "pipeline_scheduler",
    status: str | None = None,
) -> LineageExecution | None:
    terminal_status = status or node_run.status
    if terminal_status not in TERMINAL_LINEAGE_STATUSES:
        return None
    existing = session.scalar(
        select(LineageExecution).where(LineageExecution.node_run_id == node_run.id)
    )
    if existing is not None:
        return existing

    config = deepcopy(node_run.config_snapshot or {})
    frozen_input = deepcopy(input_snapshot if input_snapshot is not None else node_run.input_snapshot or {})
    frozen_output = deepcopy(output_snapshot if output_snapshot is not None else node_run.output_snapshot or {})
    consumed = _config_refs(session, config)
    consumed.extend(_collect_refs({key: value for key, value in frozen_input.items() if key != "task_snapshot"}, role="input"))
    explicit_consumed = frozen_output.get("consumed_assets")
    if explicit_consumed is not None:
        consumed.extend(_collect_refs(explicit_consumed, role="declared_input"))

    legacy_mapping = frozen_input.get("legacy_mapping")
    if isinstance(legacy_mapping, dict):
        consumed.extend(_legacy_mapping_refs(session, legacy_mapping, node_run.node_id))

    output_versions = _collect_refs(
        frozen_output.get("produced_assets")
        if frozen_output.get("produced_assets") is not None
        else frozen_output.get("versions") or [],
        role="output",
    )
    produced: list[dict[str, Any]] = []
    legacy_source = bool(config.get("legacy_mapping")) and node_run.node_id in {
        "data",
        "tokenizer",
        "code",
    }
    for ref in output_versions:
        if legacy_source or ref["asset_type"] in OUTPUT_AS_INPUT_TYPES:
            consumed.append({**ref, "role": ref.get("role") or "binding"})
        else:
            produced.append(ref)

    if node_run.node_type == "data":
        descriptor = frozen_output.get("descriptor")
        ref = _normalize_asset_ref(descriptor, role="binding")
        if ref is not None:
            consumed.append(ref)

    consumed = _dedupe_refs(consumed)
    if frozen_input.get("reused_from_node_run_id"):
        # Reuse consumes a prior result; it does not claim to have produced it again.
        consumed.extend({**ref, "role": "reused_output"} for ref in produced)
        produced = []
    produced = _dedupe_refs(produced)
    identity = _execution_identity(
        session,
        run,
        node_run,
        config,
        frozen_output,
        execution_owner=execution_owner,
    )
    snapshot = {
        "project_id": run.project_id,
        "pipeline_run_id": run.id,
        "node_run_id": node_run.id,
        "node_id": node_run.node_id,
        "node_type": node_run.node_type,
        "status": terminal_status,
        "execution_owner": execution_owner,
        "execution_identity": identity,
        "consumed_assets": consumed,
        "produced_assets": produced,
        "started_at": _timestamp(node_run.started_at),
        "ended_at": _timestamp(node_run.ended_at),
    }
    execution = LineageExecution(
        project_id=run.project_id,
        pipeline_run_id=run.id,
        node_run_id=node_run.id,
        task_id=node_run.task_id,
        node_id=node_run.node_id,
        node_type=node_run.node_type,
        status=terminal_status,
        execution_owner=execution_owner,
        execution_identity=identity,
        content_hash=_content_hash(snapshot),
        started_at=node_run.started_at,
        ended_at=node_run.ended_at,
    )
    session.add(execution)
    session.flush()
    for relation_kind, refs in (("consumed", consumed), ("produced", produced)):
        for ref in refs:
            session.add(
                LineageAssetRelation(
                    project_id=run.project_id,
                    execution_id=execution.id,
                    relation_kind=relation_kind,
                    asset_type=ref["asset_type"],
                    asset_id=ref["asset_id"],
                    role=ref.get("role") or "",
                    label=ref["label"],
                    version=ref.get("version"),
                    status=ref.get("status") or "",
                    metadata_json=deepcopy(ref.get("metadata") or {}),
                )
            )
    session.flush()
    return execution


def record_terminal_run_lineage(
    session: Session,
    run: ProjectPipelineRun,
    node_runs: list[ProjectPipelineNodeRun],
    *,
    projected_nodes: list[dict[str, Any]] | None = None,
    execution_owner: str = "pipeline_scheduler",
    finalize_lost: bool = False,
) -> None:
    provisional_loss = bool((run.context_snapshot or {}).get("__training_instance")) and not finalize_lost and any((row.error or {}).get("code") == "lease_expired" for row in node_runs)
    projected_by_id = {
        str(item.get("id") or ""): item for item in (projected_nodes or [])
    }
    for node_run in node_runs:
        if (node_run.input_snapshot or {}).get("excluded_from_execution"):
            continue
        if provisional_loss and (node_run.error or {}).get("code") in {"lease_expired", "upstream_failed"}:
            continue
        projected = projected_by_id.get(node_run.id, {})
        projected_status = str(projected.get("status") or node_run.status)
        if projected_status not in TERMINAL_LINEAGE_STATUSES:
            continue
        record_terminal_node_lineage(
            session,
            run,
            node_run,
            input_snapshot=projected.get("input_snapshot"),
            output_snapshot=projected.get("output_snapshot"),
            execution_owner=execution_owner,
            status=projected_status,
        )


def _relation_view(relation: LineageAssetRelation) -> dict[str, Any]:
    return {
        "id": relation.id,
        "relation_kind": relation.relation_kind,
        "asset_type": relation.asset_type,
        "asset_id": relation.asset_id,
        "role": relation.role,
        "label": relation.label,
        "version": relation.version,
        "status": relation.status,
        "metadata": deepcopy(relation.metadata_json or {}),
    }


def lineage_execution_view(
    session: Session,
    execution: LineageExecution,
    *,
    relations: list[LineageAssetRelation] | None = None,
) -> dict[str, Any]:
    rows = relations
    if rows is None:
        rows = list(
            session.scalars(
                select(LineageAssetRelation)
                .where(LineageAssetRelation.execution_id == execution.id)
                .order_by(LineageAssetRelation.created_at.asc())
            ).all()
        )
    node_run = session.get(ProjectPipelineNodeRun, execution.node_run_id)
    return {
        **execution.to_dict(),
        "node_label": node_run.label if node_run is not None else execution.node_id,
        "consumed_assets": [
            _relation_view(item) for item in rows if item.relation_kind == "consumed"
        ],
        "produced_assets": [
            _relation_view(item) for item in rows if item.relation_kind == "produced"
        ],
    }


def run_lineage_view(session: Session, run: ProjectPipelineRun) -> dict[str, Any]:
    executions = list(
        session.scalars(
            select(LineageExecution)
            .where(LineageExecution.pipeline_run_id == run.id)
            .order_by(LineageExecution.created_at.asc())
        ).all()
    )
    execution_ids = [item.id for item in executions]
    relations = (
        list(
            session.scalars(
                select(LineageAssetRelation)
                .where(LineageAssetRelation.execution_id.in_(execution_ids))
                .order_by(LineageAssetRelation.created_at.asc())
            ).all()
        )
        if execution_ids
        else []
    )
    by_execution: dict[str, list[LineageAssetRelation]] = {}
    for relation in relations:
        by_execution.setdefault(relation.execution_id, []).append(relation)
    return {
        "pipeline_run_id": run.id,
        "project_id": run.project_id,
        "stored": bool(executions),
        "execution_count": len(executions),
        "relation_count": len(relations),
        "executions": [
            lineage_execution_view(
                session,
                execution,
                relations=by_execution.get(execution.id, []),
            )
            for execution in executions
        ],
    }


def node_lineage_views(
    session: Session,
    run: ProjectPipelineRun,
) -> dict[str, dict[str, Any]]:
    ledger = run_lineage_view(session, run)
    return {
        str(item["node_run_id"]): item for item in ledger["executions"]
    }


def asset_lineage_view(
    session: Session,
    project_id: str,
    asset_type: str,
    asset_id: str,
    *,
    limit: int = 100,
) -> dict[str, Any]:
    matches = list(
        session.scalars(
            select(LineageAssetRelation)
            .where(
                LineageAssetRelation.project_id == project_id,
                LineageAssetRelation.asset_type == asset_type,
                LineageAssetRelation.asset_id == asset_id,
            )
            .order_by(LineageAssetRelation.created_at.desc())
            .limit(limit + 1)
        ).all()
    )
    truncated = len(matches) > limit
    matches = matches[:limit]
    execution_ids = list(dict.fromkeys(item.execution_id for item in matches))
    executions = {
        item.id: item
        for item in session.scalars(
            select(LineageExecution).where(
                LineageExecution.project_id == project_id,
                LineageExecution.id.in_(execution_ids),
            )
        ).all()
    } if execution_ids else {}
    all_relations = (
        list(
            session.scalars(
                select(LineageAssetRelation)
                .where(LineageAssetRelation.execution_id.in_(execution_ids))
                .order_by(LineageAssetRelation.created_at.asc())
            ).all()
        )
        if execution_ids
        else []
    )
    by_execution: dict[str, list[LineageAssetRelation]] = {}
    for relation in all_relations:
        by_execution.setdefault(relation.execution_id, []).append(relation)

    def views(kind: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in matches:
            if match.relation_kind != kind or match.execution_id in seen:
                continue
            execution = executions.get(match.execution_id)
            if execution is None:
                continue
            result.append(
                lineage_execution_view(
                    session,
                    execution,
                    relations=by_execution.get(execution.id, []),
                )
            )
            seen.add(match.execution_id)
        return result

    asset = _relation_view(matches[0]) if matches else {
        "asset_type": asset_type,
        "asset_id": asset_id,
        "label": f"{asset_type} · {asset_id[:8]}",
        "version": None,
        "status": "",
        "metadata": {},
    }
    return {
        "project_id": project_id,
        "asset": asset,
        "producers": views("produced"),
        "consumers": views("consumed"),
        "truncated": truncated,
        "limit": limit,
    }
