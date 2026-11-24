from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Checkpoint,
    DataMixture,
    DataMixtureVersion,
    Dataset,
    DatasetVersion,
    EvalSuiteVersion,
    EvaluationRun,
    ModelVersion,
    ProjectDataBinding,
    QuantizationRun,
    Release,
    SupervisionDataset,
    SupervisionDatasetVersion,
    SupervisionMixture,
    SupervisionMixtureVersion,
    TokenizerVersion,
    TrainingRun,
)
from .dataset_analyses import effective_train_ready
from .global_data import asset_descriptor, asset_record

LIFECYCLE_STAGES = (
    ("data_preparation", "数据准备", "data"),
    ("tokenizer", "Tokenizer", "tokenizers"),
    ("pretraining", "预训练", "training"),
    ("post_training", "后训练", "training"),
    ("evaluation", "评测", "evaluations"),
    ("quantization", "量化", "releases"),
    ("release", "发布", "releases"),
)

ACTIVE_STATUSES = {
    "queued",
    "assigned",
    "starting",
    "running",
    "pausing",
    "stopping",
    "resolving",
}
READY_STATUSES = {"succeeded", "passed", "ready", "available", "published", "official"}
FAILED_STATUSES = {"failed", "lost"}
POST_TRAINING_OBJECTIVES = {"sft", "dpo"}
STATUS_LABELS = {
    "queued": "排队中",
    "assigned": "已分配",
    "starting": "启动中",
    "running": "运行中",
    "pausing": "暂停中",
    "paused": "已暂停",
    "stopping": "停止中",
    "succeeded": "已完成",
    "passed": "已通过",
    "ready": "就绪",
    "failed": "失败",
    "failed_gate": "门禁未通过",
    "cancelled": "已取消",
    "lost": "Worker 已失联",
    "published": "已发布",
}
PURPOSE_LABELS = {
    "training": "训练",
    "validation": "验证",
    "reference": "参考",
    "evaluation": "评测",
}


def _timestamp(record: Any | None) -> str | None:
    if record is None:
        return None
    value = getattr(record, "updated_at", None) or getattr(record, "created_at", None)
    return value.isoformat() if value is not None else None


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def _version_ref(
    kind: str,
    record: Any,
    label: str,
    *,
    raw_status: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": kind,
        "id": record.id,
        "label": label,
    }
    if hasattr(record, "version"):
        result["version"] = record.version
    if raw_status:
        result["status"] = raw_status
    if metadata:
        result["metadata"] = metadata
    return result


def _normalized_status(
    raw_status: str | None,
    *,
    ready: bool = False,
    blocked: bool = False,
) -> str:
    raw = str(raw_status or "").lower()
    if raw == "paused":
        return "paused"
    if raw in ACTIVE_STATUSES:
        return "in_progress"
    if raw in FAILED_STATUSES:
        return "failed"
    if raw == "cancelled":
        return "stopped"
    if blocked or raw == "failed_gate":
        return "blocked"
    if ready or raw in READY_STATUSES:
        return "ready"
    return "waiting" if raw else "not_started"


def _node(
    key: str,
    *,
    status: str,
    raw_status: str = "",
    version: dict[str, Any] | None = None,
    versions: list[dict[str, Any]] | None = None,
    detail: str,
    updated_at: str | None = None,
) -> dict[str, Any]:
    label, target = next(
        (label, target) for stage, label, target in LIFECYCLE_STAGES if stage == key
    )
    return {
        "key": key,
        "label": label,
        "status": status,
        "raw_status": raw_status,
        "version": version,
        "versions": versions or ([version] if version else []),
        "detail": detail,
        "updated_at": updated_at,
        "target": target,
    }


def _data_entry(
    session: Session,
    asset_type: str,
    record: Any,
    *,
    purposes: Iterable[str],
) -> dict[str, Any]:
    descriptor = asset_descriptor(session, asset_type, record.id) or {}
    name = str(descriptor.get("name") or asset_type)
    raw_status = str(getattr(record, "status", getattr(record, "state", "ready")))
    if asset_type == "dataset_version":
        ready = raw_status == "ready" and effective_train_ready(session, record)
        blocked = raw_status == "ready" and not ready
        token_count = int(record.token_estimate or 0)
        document_count = int(record.document_count or 0)
    elif asset_type == "mixture_version":
        ready = raw_status == "ready"
        blocked = raw_status not in ACTIVE_STATUSES and not ready
        token_count = int(record.resolved_token_estimate or record.target_tokens or 0)
        document_count = int(record.resolved_document_count or 0)
    elif asset_type == "supervision_dataset_version":
        ready = raw_status == "ready"
        blocked = raw_status not in ACTIVE_STATUSES and not ready
        token_count = 0
        document_count = int(record.accepted_record_count or 0)
    else:
        ready = raw_status == "ready"
        blocked = raw_status not in ACTIVE_STATUSES and not ready
        token_count = 0
        document_count = 0
    purpose_list = sorted(set(purposes))
    reference = _version_ref(
        asset_type,
        record,
        f"{name} · V{record.version}",
        raw_status=raw_status,
        metadata={"purposes": purpose_list},
    )
    return {
        "asset_type": asset_type,
        "asset": record,
        "name": name,
        "version": record.version,
        "purposes": purpose_list,
        "raw_status": raw_status,
        "status": _normalized_status(raw_status, ready=ready, blocked=blocked),
        "ready": ready,
        "token_count": token_count,
        "document_count": document_count,
        "updated_at": _timestamp(record),
        "reference": reference,
    }


def _project_data_entries(session: Session, project_id: str) -> list[dict[str, Any]]:
    collected: dict[tuple[str, str], dict[str, Any]] = {}
    bindings = session.scalars(
        select(ProjectDataBinding)
        .where(
            ProjectDataBinding.project_id == project_id,
            ProjectDataBinding.asset_type != "eval_dataset_version",
            ProjectDataBinding.purpose != "evaluation",
        )
        .order_by(ProjectDataBinding.created_at.desc())
    ).all()
    for binding in bindings:
        record = asset_record(session, binding.asset_type, binding.asset_version_id)
        if record is None:
            continue
        key = (binding.asset_type, binding.asset_version_id)
        current = collected.get(key)
        purposes = set(current["purposes"]) if current else set()
        purposes.add(binding.purpose)
        collected[key] = _data_entry(
            session,
            binding.asset_type,
            record,
            purposes=purposes,
        )

    legacy_sources: tuple[tuple[str, type[Any], type[Any]], ...] = (
        ("dataset_version", Dataset, DatasetVersion),
        ("mixture_version", DataMixture, DataMixtureVersion),
        ("supervision_dataset_version", SupervisionDataset, SupervisionDatasetVersion),
        ("supervision_mixture_version", SupervisionMixture, SupervisionMixtureVersion),
    )
    for asset_type, container_type, version_type in legacy_sources:
        containers = session.scalars(
            select(container_type).where(container_type.project_id == project_id)
        ).all()
        for container in containers:
            if not container.current_version_id:
                continue
            record = session.get(version_type, container.current_version_id)
            if record is None:
                continue
            key = (asset_type, record.id)
            if key not in collected:
                collected[key] = _data_entry(
                    session,
                    asset_type,
                    record,
                    purposes=("training",),
                )

    purpose_priority = {"training": 0, "validation": 1, "reference": 2, "evaluation": 3}

    def entry_order(item: dict[str, Any]) -> tuple[int, str]:
        purposes = item["purposes"] or ["evaluation"]
        priority = min(purpose_priority.get(purpose, 9) for purpose in purposes)
        return priority, str(item["updated_at"] or "")

    return sorted(collected.values(), key=entry_order, reverse=False)


def _data_node(session: Session, project_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    entries = _project_data_entries(session, project_id)
    if not entries:
        return (
            _node(
                "data_preparation",
                status="not_started",
                detail="尚未绑定训练、验证或监督数据版本",
            ),
            None,
        )
    training_entries = [entry for entry in entries if "training" in entry["purposes"]]
    candidates = training_entries or entries
    primary = max(candidates, key=lambda item: str(item["updated_at"] or ""))
    ready_count = sum(1 for entry in entries if entry["ready"])
    detail_parts = [f"{ready_count}/{len(entries)} 个精确版本可用于下游"]
    if primary["token_count"]:
        detail_parts.append(f"{primary['token_count']:,} Tokens")
    elif primary["document_count"]:
        detail_parts.append(f"{primary['document_count']:,} 条记录")
    detail_parts.append(
        "用途：" + "、".join(PURPOSE_LABELS.get(item, item) for item in primary["purposes"])
    )
    current_asset = {
        "id": primary["asset"].id,
        "asset_type": primary["asset_type"],
        "name": primary["name"],
        "version": primary["version"],
        "status": primary["raw_status"],
        "purposes": primary["purposes"],
        "token_estimate": primary["token_count"],
        "document_count": primary["document_count"],
    }
    return (
        _node(
            "data_preparation",
            status=primary["status"],
            raw_status=primary["raw_status"],
            version=primary["reference"],
            versions=[entry["reference"] for entry in entries],
            detail=" · ".join(detail_parts),
            updated_at=primary["updated_at"],
        ),
        current_asset,
    )


def _tokenizer_node(
    session: Session, project_id: str
) -> tuple[dict[str, Any], TokenizerVersion | None]:
    tokenizers = session.scalars(
        select(TokenizerVersion)
        .where(TokenizerVersion.project_id == project_id)
        .order_by(TokenizerVersion.created_at.desc())
    ).all()
    if not tokenizers:
        return (
            _node(
                "tokenizer",
                status="not_started",
                detail="尚未创建 Tokenizer 版本",
            ),
            None,
        )
    selected = next((item for item in tokenizers if item.is_default), None)
    selected = selected or next(
        (item for item in tokenizers if item.locked and item.status == "passed"),
        tokenizers[0],
    )
    ready = selected.status == "passed" and selected.locked
    blocked = selected.status == "failed_gate"
    status = _normalized_status(selected.status, ready=ready, blocked=blocked)
    reference = _version_ref(
        "tokenizer_version",
        selected,
        f"{selected.name} · V{selected.version}",
        raw_status=selected.status,
        metadata={
            "algorithm": selected.algorithm,
            "vocab_size": selected.vocab_size,
            "locked": selected.locked,
            "is_default": selected.is_default,
        },
    )
    if ready and selected.is_default:
        detail = f"默认版本已锁定 · {selected.algorithm} · {selected.vocab_size:,} 词表"
    elif ready:
        detail = f"已通过并锁定，尚未设为默认 · {selected.vocab_size:,} 词表"
    elif selected.status == "passed":
        detail = "质量门禁已通过，尚未锁定"
    elif blocked:
        detail = "最新候选未通过 Tokenizer 质量门禁"
    else:
        detail = f"当前版本状态：{_status_label(selected.status)}"
    return (
        _node(
            "tokenizer",
            status=status,
            raw_status=selected.status,
            version=reference,
            versions=[
                _version_ref(
                    "tokenizer_version",
                    item,
                    f"{item.name} · V{item.version}",
                    raw_status=item.status,
                )
                for item in tokenizers[:5]
            ],
            detail=detail,
            updated_at=_timestamp(selected),
        ),
        selected,
    )


def _training_objectives(run: TrainingRun) -> set[str]:
    snapshot = run.config_snapshot or {}
    stages = snapshot.get("selected_stages") or snapshot.get("stages") or []
    objectives: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        objective = stage.get("objective") if isinstance(stage.get("objective"), dict) else {}
        value = objective.get("kind") or stage.get("type") or stage.get("workflow")
        if value:
            objectives.add(str(value).lower())
    return objectives or {"pretrain"}


def _is_post_training_model(model: ModelVersion) -> bool:
    return str(model.objective_kind or model.kind or "").lower() in POST_TRAINING_OBJECTIVES


def _training_node(
    key: str,
    runs: list[TrainingRun],
    checkpoints: list[Checkpoint],
    models: list[ModelVersion],
) -> tuple[dict[str, Any], TrainingRun | None]:
    post_training = key == "post_training"
    relevant_runs = []
    for run in runs:
        objectives = _training_objectives(run)
        belongs = (
            bool(objectives & POST_TRAINING_OBJECTIVES)
            if post_training
            else bool(objectives - POST_TRAINING_OBJECTIVES)
        )
        if belongs:
            relevant_runs.append(run)
    relevant_models = [
        model for model in models if _is_post_training_model(model) == post_training
    ]
    relevant_run_ids = {run.id for run in relevant_runs}
    relevant_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.training_run_id in relevant_run_ids
    ]
    latest_run = relevant_runs[0] if relevant_runs else None
    latest_model = relevant_models[0] if relevant_models else None
    latest_checkpoint = relevant_checkpoints[0] if relevant_checkpoints else None
    if latest_run is None and latest_model is None and latest_checkpoint is None:
        return (
            _node(
                key,
                status="not_started",
                detail="尚无后训练运行或模型" if post_training else "尚无预训练运行或模型",
            ),
            None,
        )
    raw_status = str(
        latest_run.status
        if latest_run is not None
        else getattr(latest_model or latest_checkpoint, "status", "ready")
    )
    status = _normalized_status(
        raw_status,
        ready=latest_run is None and (latest_model is not None or latest_checkpoint is not None),
    )
    versions: list[dict[str, Any]] = []
    if latest_model is not None:
        versions.append(
            _version_ref(
                "model_version",
                latest_model,
                f"{latest_model.name} · V{latest_model.version}",
                raw_status=latest_model.status,
                metadata={
                    "kind": latest_model.kind,
                    "objective_kind": latest_model.objective_kind,
                },
            )
        )
    if latest_checkpoint is not None:
        versions.append(
            _version_ref(
                "checkpoint",
                latest_checkpoint,
                f"Checkpoint · step {latest_checkpoint.step:,}",
                raw_status=latest_checkpoint.status,
                metadata={"token_count": latest_checkpoint.token_count},
            )
        )
    if latest_run is not None:
        versions.append(
            _version_ref(
                "training_run",
                latest_run,
                f"运行 · {latest_run.id[:8]}",
                raw_status=latest_run.status,
                metadata={"objectives": sorted(_training_objectives(latest_run))},
            )
        )
    primary = versions[0] if versions else None
    detail_parts = []
    if latest_run is not None:
        detail_parts.append(
            f"最近运行 {_status_label(latest_run.status)}"
            + (f" · {latest_run.current_stage}" if latest_run.current_stage else "")
        )
    if latest_model is not None:
        detail_parts.append(f"可用模型 {latest_model.name} V{latest_model.version}")
    if latest_checkpoint is not None:
        detail_parts.append(f"Checkpoint step {latest_checkpoint.step:,}")
    if post_training:
        objectives = set()
        for run in relevant_runs:
            objectives.update(_training_objectives(run) & POST_TRAINING_OBJECTIVES)
        for model in relevant_models:
            objective = str(model.objective_kind or model.kind or "").lower()
            if objective in POST_TRAINING_OBJECTIVES:
                objectives.add(objective)
        if objectives:
            detail_parts.append("目标：" + " / ".join(item.upper() for item in sorted(objectives)))
    updated_record = latest_run or latest_model or latest_checkpoint
    return (
        _node(
            key,
            status=status,
            raw_status=raw_status,
            version=primary,
            versions=versions,
            detail="；".join(detail_parts),
            updated_at=_timestamp(updated_record),
        ),
        latest_run,
    )


def _evaluation_node(
    session: Session, run: EvaluationRun | None
) -> dict[str, Any]:
    if run is None:
        return _node("evaluation", status="not_started", detail="尚无项目评测运行")
    suites = [
        suite
        for suite_id in (run.suite_version_ids or [])
        if (suite := session.get(EvalSuiteVersion, suite_id)) is not None
    ]
    versions = [
        _version_ref(
            "eval_suite_version",
            suite,
            f"{suite.name} · V{suite.version}",
            raw_status="ready",
        )
        for suite in suites
    ]
    versions.append(
        _version_ref(
            "evaluation_run",
            run,
            f"评测运行 · {run.id[:8]}",
            raw_status=run.status,
        )
    )
    detail = f"{len(run.model_version_ids or [])} 个模型 · {len(run.suite_version_ids or [])} 个套件"
    return _node(
        "evaluation",
        status=_normalized_status(run.status),
        raw_status=run.status,
        version=versions[0] if versions else None,
        versions=versions,
        detail=detail,
        updated_at=_timestamp(run),
    )


def _quantization_node(
    session: Session, run: QuantizationRun | None
) -> dict[str, Any]:
    if run is None:
        return _node("quantization", status="not_started", detail="尚无量化运行")
    source_model = session.get(ModelVersion, run.model_version_id)
    versions = [
        _version_ref(
            "quantization_run",
            run,
            f"{run.method} · {run.format}",
            raw_status=run.status,
        )
    ]
    if source_model is not None:
        versions.append(
            _version_ref(
                "model_version",
                source_model,
                f"{source_model.name} · V{source_model.version}",
                raw_status=source_model.status,
            )
        )
    return _node(
        "quantization",
        status=_normalized_status(run.status),
        raw_status=run.status,
        version=versions[0],
        versions=versions,
        detail=(
            f"源模型 {source_model.name} V{source_model.version}"
            if source_model is not None
            else f"源模型 {run.model_version_id[:8]}"
        ),
        updated_at=_timestamp(run),
    )


def _release_node(session: Session, release: Release | None) -> dict[str, Any]:
    if release is None:
        return _node("release", status="not_started", detail="尚无正式发布记录")
    model = session.get(ModelVersion, release.model_version_id)
    versions = [
        _version_ref(
            "release",
            release,
            f"{release.name} · {release.version}",
            raw_status=release.status,
        )
    ]
    if model is not None:
        versions.append(
            _version_ref(
                "model_version",
                model,
                f"{model.name} · V{model.version}",
                raw_status=model.status,
            )
        )
    detail = f"目标：{release.target}"
    if model is not None:
        detail += f" · 模型 {model.name} V{model.version}"
    if release.quantization_run_id:
        detail += " · 引用量化结果"
    return _node(
        "release",
        status=_normalized_status(release.status),
        raw_status=release.status,
        version=versions[0],
        versions=versions,
        detail=detail,
        updated_at=_timestamp(release),
    )


def _derived_current_stage(nodes: list[dict[str, Any]]) -> str:
    attention = {
        "in_progress",
        "paused",
        "failed",
        "blocked",
        "stopped",
    }
    candidates = [
        node for node in nodes if node["status"] in attention and node.get("updated_at")
    ]
    if candidates:
        return max(candidates, key=lambda node: str(node["updated_at"]))["key"]
    for index, node in enumerate(nodes):
        if node["status"] != "ready":
            later_ready = [
                candidate
                for candidate in nodes[index + 1 :]
                if candidate["status"] == "ready"
            ]
            if later_ready:
                return later_ready[-1]["key"]
            return node["key"]
    return "release"


def project_lifecycle_snapshot(session: Session, project_id: str) -> dict[str, Any]:
    data_node, current_data = _data_node(session, project_id)
    tokenizer_node, tokenizer = _tokenizer_node(session, project_id)
    runs = session.scalars(
        select(TrainingRun)
        .where(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.created_at.desc())
    ).all()
    checkpoints = session.scalars(
        select(Checkpoint)
        .where(Checkpoint.project_id == project_id)
        .order_by(Checkpoint.created_at.desc())
    ).all()
    models = session.scalars(
        select(ModelVersion)
        .where(ModelVersion.project_id == project_id)
        .order_by(ModelVersion.created_at.desc())
    ).all()
    pretraining_node, pretraining_run = _training_node(
        "pretraining", runs, checkpoints, models
    )
    post_training_node, post_training_run = _training_node(
        "post_training", runs, checkpoints, models
    )
    evaluation = session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.project_id == project_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(1)
    )
    quantization = session.scalar(
        select(QuantizationRun)
        .where(QuantizationRun.project_id == project_id)
        .order_by(QuantizationRun.created_at.desc())
        .limit(1)
    )
    release = session.scalar(
        select(Release)
        .where(Release.project_id == project_id)
        .order_by(Release.created_at.desc())
        .limit(1)
    )
    nodes = [
        data_node,
        tokenizer_node,
        pretraining_node,
        post_training_node,
        _evaluation_node(session, evaluation),
        _quantization_node(session, quantization),
        _release_node(session, release),
    ]
    derived_stage = _derived_current_stage(nodes)
    for node in nodes:
        node["is_current"] = node["key"] == derived_stage

    candidate = next((model for model in models if model.is_candidate), None)
    latest_run = runs[0] if runs else None
    latest_checkpoint = checkpoints[0] if checkpoints else None
    current_dataset = None
    if current_data and current_data["asset_type"] == "dataset_version":
        current_dataset = session.get(DatasetVersion, current_data["id"])
    return {
        "derived_lifecycle_stage": derived_stage,
        "lifecycle_nodes": nodes,
        "current": {
            "data_asset": current_data,
            "dataset_version": current_dataset.to_dict() if current_dataset else None,
            "tokenizer": tokenizer.to_dict() if tokenizer else None,
            "training_run": latest_run.to_dict() if latest_run else None,
            "pretraining_run": pretraining_run.to_dict() if pretraining_run else None,
            "post_training_run": post_training_run.to_dict() if post_training_run else None,
            "checkpoint": latest_checkpoint.to_dict() if latest_checkpoint else None,
            "candidate_model": candidate.to_dict() if candidate else None,
            "evaluation": evaluation.to_dict() if evaluation else None,
            "quantization": quantization.to_dict() if quantization else None,
            "release": release.to_dict() if release else None,
        },
    }
