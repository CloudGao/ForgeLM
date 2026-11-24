from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Checkpoint,
    CodeSource,
    CodeVersion,
    DataMixture,
    DataMixtureVersion,
    Dataset,
    DatasetVersion,
    EvaluationRun,
    ModelVersion,
    PlanRevision,
    ProjectDataBinding,
    ProjectPipeline,
    ProjectPipelineRevision,
    QuantizationRun,
    Release,
    SupervisionDataset,
    SupervisionDatasetVersion,
    SupervisionMixture,
    SupervisionMixtureVersion,
    Task,
    TokenizerVersion,
    TrainingDataViewVersion,
    TrainingPlan,
    TrainingRun,
)
from ..schemas import ProjectPipelineCreate
from .dataset_analyses import effective_train_ready
from .global_data import (
    asset_descriptor,
    asset_record,
    project_has_asset_access,
)

PIPELINE_NODES = (
    ("data", "数据", "data"),
    ("tokenizer", "Tokenizer", "tokenizers"),
    ("code", "代码", "overview"),
    ("training", "训练", "training"),
    ("model", "模型", "models"),
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


class ProjectPipelineError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _timestamp(record: Any | None) -> str | None:
    if record is None:
        return None
    value = getattr(record, "updated_at", None) or getattr(record, "created_at", None)
    return value.isoformat() if value is not None else None


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


def _ref(
    kind: str,
    record: Any,
    label: str,
    *,
    status: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"kind": kind, "id": record.id, "label": label}
    if hasattr(record, "version"):
        value["version"] = record.version
    if status:
        value["status"] = status
    if metadata:
        value["metadata"] = metadata
    return value


def _node(
    key: str,
    *,
    status: str,
    raw_status: str = "",
    versions: list[dict[str, Any]] | None = None,
    detail: str,
    updated_at: str | None = None,
    progress: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label, target = next(
        (label, target) for node_key, label, target in PIPELINE_NODES if node_key == key
    )
    version_list = versions or []
    return {
        "key": key,
        "label": label,
        "status": status,
        "raw_status": raw_status,
        "version": version_list[0] if version_list else None,
        "versions": version_list,
        "detail": detail,
        "updated_at": updated_at,
        "progress": progress,
        "metrics": metrics or {},
        "target": target,
    }


def _asset_type_for_id(session: Session, asset_version_id: str) -> str | None:
    for asset_type in (
        "dataset_version",
        "mixture_version",
        "supervision_dataset_version",
        "supervision_mixture_version",
    ):
        if asset_record(session, asset_type, asset_version_id) is not None:
            return asset_type
    return None


def _objective_from_stages(stages: list[dict[str, Any]]) -> str:
    objectives = {
        str(
            (
                stage.get("objective", {}).get("kind")
                if isinstance(stage.get("objective"), dict)
                else None
            )
            or stage.get("type")
            or "pretrain"
        ).lower()
        for stage in stages
        if isinstance(stage, dict)
    }
    return "post_training" if objectives and objectives <= POST_TRAINING_OBJECTIVES else "pretraining"


def _inferred_data_refs(
    session: Session,
    stages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        asset_version_id = str(stage.get("data_ref_id") or "").strip()
        if not asset_version_id:
            continue
        asset_type = _asset_type_for_id(session, asset_version_id)
        if asset_type is None or (asset_type, asset_version_id) in seen:
            continue
        results.append(
            {
                "asset_type": asset_type,
                "asset_version_id": asset_version_id,
                "purpose": "training",
            }
        )
        seen.add((asset_type, asset_version_id))
    return results


def _project_plan_revision(
    session: Session, project_id: str, revision_id: str
) -> tuple[PlanRevision, TrainingPlan]:
    revision = session.get(PlanRevision, revision_id)
    plan = session.get(TrainingPlan, revision.plan_id) if revision is not None else None
    if revision is None or plan is None or plan.project_id != project_id:
        raise ProjectPipelineError(
            "invalid_pipeline_plan_revision", "训练计划修订版不属于当前项目"
        )
    return revision, plan


def _project_code_version(
    session: Session, project_id: str, version_id: str
) -> tuple[CodeVersion, CodeSource]:
    version = session.get(CodeVersion, version_id)
    source = session.get(CodeSource, version.source_id) if version is not None else None
    if version is None or source is None or source.project_id != project_id:
        raise ProjectPipelineError("invalid_pipeline_code_version", "代码版本不属于当前项目")
    return version, source


def _project_record(
    session: Session,
    model: type[Any],
    record_id: str,
    project_id: str,
    *,
    code: str,
    label: str,
) -> Any:
    record = session.get(model, record_id)
    if record is None or getattr(record, "project_id", None) != project_id:
        raise ProjectPipelineError(code, f"{label}不属于当前项目")
    return record


def _choose_exact(
    explicit: str | None,
    inferred: str | None,
    *,
    label: str,
) -> str | None:
    if explicit and inferred and explicit != inferred:
        raise ProjectPipelineError(
            "pipeline_binding_mismatch",
            f"显式选择的{label}与训练运行冻结的版本不一致",
            status_code=409,
        )
    return explicit or inferred


def resolve_pipeline_mapping(
    session: Session,
    project_id: str,
    payload: ProjectPipelineCreate,
) -> dict[str, Any]:
    run: TrainingRun | None = None
    frozen_stages: list[dict[str, Any]] = []
    inferred_revision_id: str | None = None
    if payload.training_run_id:
        run = _project_record(
            session,
            TrainingRun,
            payload.training_run_id,
            project_id,
            code="invalid_pipeline_training_run",
            label="训练运行",
        )
        inferred_revision_id = run.plan_revision_id
        frozen_stages = list((run.config_snapshot or {}).get("selected_stages") or [])

    plan_revision_id = _choose_exact(
        payload.plan_revision_id,
        inferred_revision_id,
        label="训练计划修订版",
    )
    revision: PlanRevision | None = None
    plan: TrainingPlan | None = None
    if plan_revision_id:
        revision, plan = _project_plan_revision(session, project_id, plan_revision_id)
        frozen_stages = frozen_stages or list(revision.stages or [])

    if frozen_stages and _objective_from_stages(frozen_stages) != payload.objective:
        raise ProjectPipelineError(
            "pipeline_objective_mismatch",
            "产线用途与训练运行冻结的阶段类型不一致",
            status_code=409,
        )

    tokenizer_version_id = _choose_exact(
        payload.tokenizer_version_id,
        revision.tokenizer_version_id if revision else None,
        label="Tokenizer",
    )
    code_version_id = _choose_exact(
        payload.code_version_id,
        revision.code_version_id if revision else None,
        label="代码",
    )

    inferred_data_refs = _inferred_data_refs(session, frozen_stages)
    explicit_data_refs = [item.model_dump() for item in payload.data_refs]
    if explicit_data_refs and inferred_data_refs:
        explicit_training_ids = {
            item["asset_version_id"]
            for item in explicit_data_refs
            if item["purpose"] == "training"
        }
        missing = {
            item["asset_version_id"] for item in inferred_data_refs
        } - explicit_training_ids
        if missing:
            raise ProjectPipelineError(
                "pipeline_binding_mismatch",
                "显式训练数据与训练运行冻结的数据版本不一致",
                status_code=409,
            )
    data_refs = explicit_data_refs or inferred_data_refs

    unique_refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str, str]] = set()
    for item in data_refs:
        key = (item["asset_type"], item["asset_version_id"], item["purpose"])
        if key in seen_refs:
            continue
        if not project_has_asset_access(
            session, project_id, item["asset_type"], item["asset_version_id"]
        ):
            raise ProjectPipelineError(
                "invalid_pipeline_data_version",
                "产线数据版本不存在或项目没有消费权限",
            )
        unique_refs.append(item)
        seen_refs.add(key)

    tokenizer: TokenizerVersion | None = None
    if tokenizer_version_id:
        tokenizer = _project_record(
            session,
            TokenizerVersion,
            tokenizer_version_id,
            project_id,
            code="invalid_pipeline_tokenizer_version",
            label="Tokenizer 版本",
        )
    code_version: CodeVersion | None = None
    code_source: CodeSource | None = None
    if code_version_id:
        code_version, code_source = _project_code_version(
            session, project_id, code_version_id
        )

    downstream: dict[str, Any] = {}
    for field_name, model, error_code, label in (
        ("model_version_id", ModelVersion, "invalid_pipeline_model", "模型版本"),
        ("evaluation_run_id", EvaluationRun, "invalid_pipeline_evaluation", "评测运行"),
        (
            "quantization_run_id",
            QuantizationRun,
            "invalid_pipeline_quantization",
            "量化运行",
        ),
        ("release_id", Release, "invalid_pipeline_release", "发布记录"),
    ):
        record_id = getattr(payload, field_name)
        downstream[field_name] = (
            _project_record(
                session,
                model,
                record_id,
                project_id,
                code=error_code,
                label=label,
            )
            if record_id
            else None
        )

    model = downstream["model_version_id"]
    evaluation = downstream["evaluation_run_id"]
    quantization = downstream["quantization_run_id"]
    release = downstream["release_id"]
    if model is not None and evaluation is not None and model.id not in (evaluation.model_version_ids or []):
        raise ProjectPipelineError(
            "pipeline_binding_mismatch", "评测运行没有引用所选模型版本", status_code=409
        )
    if model is not None and quantization is not None and quantization.model_version_id != model.id:
        raise ProjectPipelineError(
            "pipeline_binding_mismatch", "量化运行没有引用所选模型版本", status_code=409
        )
    if model is not None and release is not None and release.model_version_id != model.id:
        raise ProjectPipelineError(
            "pipeline_binding_mismatch", "发布记录没有引用所选模型版本", status_code=409
        )
    if (
        quantization is not None
        and release is not None
        and release.quantization_run_id
        and release.quantization_run_id != quantization.id
    ):
        raise ProjectPipelineError(
            "pipeline_binding_mismatch", "发布记录没有引用所选量化运行", status_code=409
        )

    snapshot_data = []
    for item in unique_refs:
        descriptor = asset_descriptor(
            session, item["asset_type"], item["asset_version_id"]
        )
        snapshot_data.append(
            {
                **item,
                "name": descriptor.get("name") if descriptor else "",
                "version": descriptor.get("version") if descriptor else None,
            }
        )
    return {
        "data_refs": unique_refs,
        "tokenizer_version_id": tokenizer_version_id,
        "code_version_id": code_version_id,
        "plan_revision_id": plan_revision_id,
        "training_run_id": run.id if run else None,
        "model_version_id": model.id if model else None,
        "evaluation_run_id": evaluation.id if evaluation else None,
        "quantization_run_id": quantization.id if quantization else None,
        "release_id": release.id if release else None,
        "mapping_snapshot": {
            "data": snapshot_data,
            "tokenizer": (
                {"id": tokenizer.id, "name": tokenizer.name, "version": tokenizer.version}
                if tokenizer
                else None
            ),
            "code": (
                {
                    "id": code_version.id,
                    "name": code_source.name,
                    "version": code_version.version,
                    "commit": code_version.git_commit,
                }
                if code_version and code_source
                else None
            ),
            "plan": (
                {"id": revision.id, "name": plan.name, "revision": revision.revision}
                if revision and plan
                else None
            ),
            "training_run_id": run.id if run else None,
        },
    }


def create_project_pipeline(
    session: Session,
    project_id: str,
    payload: ProjectPipelineCreate,
) -> ProjectPipeline:
    duplicate = session.scalar(
        select(ProjectPipeline.id).where(
            ProjectPipeline.project_id == project_id,
            ProjectPipeline.name == payload.name,
        )
    )
    if duplicate is not None:
        raise ProjectPipelineError(
            "duplicate_project_pipeline", "项目中已经存在同名产线", status_code=409
        )
    mapping = resolve_pipeline_mapping(session, project_id, payload)
    training_run_id = mapping.get("training_run_id")
    if training_run_id:
        existing = session.scalar(
            select(ProjectPipeline.id)
            .join(
                ProjectPipelineRevision,
                ProjectPipelineRevision.id == ProjectPipeline.current_revision_id,
            )
            .where(
                ProjectPipeline.project_id == project_id,
                ProjectPipelineRevision.training_run_id == training_run_id,
            )
        )
        if existing is not None:
            raise ProjectPipelineError(
                "training_run_already_mapped",
                "该训练运行已经属于另一条产线",
                status_code=409,
            )
    pipeline = ProjectPipeline(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description.strip(),
        objective=payload.objective,
    )
    session.add(pipeline)
    session.flush()
    revision = ProjectPipelineRevision(
        pipeline_id=pipeline.id,
        revision=1,
        **mapping,
    )
    session.add(revision)
    session.flush()
    pipeline.current_revision_id = revision.id
    return pipeline


def _data_node(
    session: Session,
    revision: ProjectPipelineRevision,
    run: TrainingRun | None,
) -> dict[str, Any]:
    versions: list[dict[str, Any]] = []
    statuses: list[str] = []
    updated_at: str | None = None
    total_tokens = 0
    for item in revision.data_refs or []:
        descriptor = asset_descriptor(
            session, item.get("asset_type", ""), item.get("asset_version_id", "")
        )
        record = asset_record(
            session, item.get("asset_type", ""), item.get("asset_version_id", "")
        )
        if descriptor is None or record is None:
            statuses.append("blocked")
            continue
        raw_status = str(descriptor.get("status") or "")
        ready = raw_status == "ready"
        if item.get("asset_type") == "dataset_version":
            ready = ready and effective_train_ready(session, record)
            total_tokens += int(record.token_estimate or 0)
        elif item.get("asset_type") == "mixture_version":
            total_tokens += int(record.resolved_token_estimate or record.target_tokens or 0)
        status = _normalized_status(raw_status, ready=ready, blocked=raw_status == "ready" and not ready)
        statuses.append(status)
        versions.append(
            _ref(
                item["asset_type"],
                record,
                f"{descriptor['name']} · V{descriptor['version']}",
                status=raw_status,
                metadata={"purpose": item.get("purpose", "training")},
            )
        )
        timestamp = _timestamp(record)
        updated_at = max(updated_at or "", timestamp or "") or None

    prepared_count = 0
    if run is not None:
        for item in (run.config_snapshot or {}).get("training_data_view_inputs") or []:
            version_id = str(item.get("version_id") or "")
            view = session.get(TrainingDataViewVersion, version_id) if version_id else None
            if view is None:
                continue
            prepared_count += int(view.status == "ready")
            statuses.append(_normalized_status(view.status))
            versions.append(
                _ref(
                    "training_data_view_version",
                    view,
                    f"Training View · {view.id[:8]}",
                    status=view.status,
                    metadata={
                        "sequence_count": view.sequence_count,
                        "token_count": view.token_count,
                    },
                )
            )

    if not revision.data_refs:
        return _node("data", status="waiting", detail="尚未映射训练数据版本")
    if any(status in {"failed", "blocked", "stopped"} for status in statuses):
        status = "blocked"
    elif any(status in {"in_progress", "paused", "waiting"} for status in statuses):
        status = "in_progress" if "in_progress" in statuses else "waiting"
    else:
        status = "ready"
    detail = f"{sum(item == 'ready' for item in statuses)}/{len(statuses)} 个映射/制品就绪"
    if total_tokens:
        detail += f" · {total_tokens:,} Tokens"
    if run is not None and prepared_count:
        detail += f" · {prepared_count} 个 Training View"
    return _node(
        "data",
        status=status,
        versions=versions,
        detail=detail,
        updated_at=updated_at,
    )


def _tokenizer_node(
    session: Session, revision: ProjectPipelineRevision
) -> dict[str, Any]:
    tokenizer = (
        session.get(TokenizerVersion, revision.tokenizer_version_id)
        if revision.tokenizer_version_id
        else None
    )
    if tokenizer is None:
        return _node("tokenizer", status="waiting", detail="尚未映射 Tokenizer 版本")
    ready = tokenizer.status == "passed" and tokenizer.locked
    blocked = tokenizer.status == "failed_gate"
    detail = f"{tokenizer.algorithm} · {tokenizer.vocab_size:,} 词表"
    if ready:
        detail += " · 已通过并锁定"
    elif tokenizer.status == "passed":
        detail += " · 尚未锁定"
    return _node(
        "tokenizer",
        status=_normalized_status(tokenizer.status, ready=ready, blocked=blocked),
        raw_status=tokenizer.status,
        versions=[
            _ref(
                "tokenizer_version",
                tokenizer,
                f"{tokenizer.name} · V{tokenizer.version}",
                status=tokenizer.status,
                metadata={"locked": tokenizer.locked, "is_default": tokenizer.is_default},
            )
        ],
        detail=detail,
        updated_at=_timestamp(tokenizer),
    )


def _code_node(session: Session, revision: ProjectPipelineRevision) -> dict[str, Any]:
    if not revision.code_version_id:
        return _node("code", status="waiting", detail="尚未映射训练代码版本")
    version = session.get(CodeVersion, revision.code_version_id)
    source = session.get(CodeSource, version.source_id) if version is not None else None
    if version is None or source is None:
        return _node("code", status="blocked", detail="映射的代码版本已不存在")
    ready = version.state == "ready" and bool(version.artifact_id)
    blocked = version.state == "ready" and not version.artifact_id
    detail = "不可变代码制品已就绪" if ready else "代码制品尚未就绪"
    if version.git_commit:
        detail += f" · Commit {version.git_commit[:10]}"
    return _node(
        "code",
        status=_normalized_status(version.state, ready=ready, blocked=blocked),
        raw_status=version.state,
        versions=[
            _ref(
                "code_version",
                version,
                f"{source.name} · V{version.version}",
                status=version.state,
                metadata={"git_commit": version.git_commit},
            )
        ],
        detail=detail,
        updated_at=_timestamp(version),
    )


def _training_node(
    session: Session,
    revision: ProjectPipelineRevision,
    prerequisites: list[dict[str, Any]],
) -> tuple[dict[str, Any], TrainingRun | None]:
    plan_revision: PlanRevision | None = None
    plan: TrainingPlan | None = None
    if revision.plan_revision_id:
        plan_revision = session.get(PlanRevision, revision.plan_revision_id)
        plan = session.get(TrainingPlan, plan_revision.plan_id) if plan_revision else None
    run = session.get(TrainingRun, revision.training_run_id) if revision.training_run_id else None
    if run is None:
        if plan_revision is None or plan is None:
            return _node("training", status="waiting", detail="尚未映射训练计划修订版"), None
        ready = all(node["status"] == "ready" for node in prerequisites)
        return (
            _node(
                "training",
                status="ready" if ready else "waiting",
                versions=[
                    _ref(
                        "plan_revision",
                        plan_revision,
                        f"{plan.name} · R{plan_revision.revision}",
                        status="ready" if ready else "waiting",
                    )
                ],
                detail="输入映射就绪，可以启动训练" if ready else "等待数据、Tokenizer 或代码就绪",
                updated_at=_timestamp(plan_revision),
            ),
            None,
        )
    task = session.get(Task, run.task_id)
    observed_status = task.observed_state if task is not None else run.status
    progress = round(float(task.progress if task is not None else 0) * 100, 2)
    stages = list((run.config_snapshot or {}).get("selected_stages") or [])
    target_tokens = max(
        (int(stage.get("cumulative_target_tokens") or stage.get("target_tokens") or 0) for stage in stages),
        default=0,
    )
    planned_steps = sum(int(stage.get("target_steps") or 0) for stage in stages)
    versions = []
    if plan_revision is not None and plan is not None:
        versions.append(
            _ref(
                "plan_revision",
                plan_revision,
                f"{plan.name} · R{plan_revision.revision}",
                status="locked" if plan_revision.locked else "draft",
            )
        )
    versions.append(
        _ref(
            "training_run",
            run,
            f"训练运行 · {run.id[:8]}",
            status=run.status,
        )
    )
    checkpoints = session.scalars(
        select(Checkpoint)
        .where(Checkpoint.training_run_id == run.id)
        .order_by(Checkpoint.step.desc())
        .limit(3)
    ).all()
    versions.extend(
        _ref(
            "checkpoint",
            checkpoint,
            f"Checkpoint · step {checkpoint.step:,}",
            status=checkpoint.status,
        )
        for checkpoint in checkpoints
    )
    detail = f"{run.current_stage or (task.current_phase if task else '等待运行')} · Step {run.current_step:,}"
    if target_tokens:
        detail += f" · {run.current_tokens:,}/{target_tokens:,} Tokens"
    return (
        _node(
            "training",
            status=_normalized_status(observed_status),
            raw_status=observed_status,
            versions=versions,
            detail=detail,
            updated_at=_timestamp(run),
            progress=progress,
            metrics={
                "progress": progress,
                "current_stage": run.current_stage,
                "current_step": run.current_step,
                "current_tokens": run.current_tokens,
                "target_tokens": target_tokens,
                "planned_steps": planned_steps,
                "loss": (run.latest_metrics or {}).get("loss"),
                "task_id": run.task_id,
            },
        ),
        run,
    )


def _model_node(
    session: Session,
    revision: ProjectPipelineRevision,
    run: TrainingRun | None,
) -> dict[str, Any]:
    models: list[ModelVersion] = []
    if revision.model_version_id:
        model = session.get(ModelVersion, revision.model_version_id)
        if model is not None:
            models.append(model)
    elif run is not None:
        checkpoints = session.scalars(
            select(Checkpoint.id).where(Checkpoint.training_run_id == run.id)
        ).all()
        if checkpoints:
            models = session.scalars(
                select(ModelVersion)
                .where(ModelVersion.source_checkpoint_id.in_(checkpoints))
                .order_by(ModelVersion.created_at.desc())
            ).all()
    if not models:
        return _node("model", status="not_started", detail="该产线尚未产出模型版本")
    model = models[0]
    return _node(
        "model",
        status=_normalized_status(model.status, ready=True),
        raw_status=model.status,
        versions=[
            _ref(
                "model_version",
                item,
                f"{item.name} · V{item.version}",
                status=item.status,
                metadata={"kind": item.kind, "objective_kind": item.objective_kind},
            )
            for item in models
        ],
        detail=f"{model.kind} · {model.parameter_count:,} 参数",
        updated_at=_timestamp(model),
    )


def _evaluation_node(
    session: Session, revision: ProjectPipelineRevision
) -> dict[str, Any]:
    run = session.get(EvaluationRun, revision.evaluation_run_id) if revision.evaluation_run_id else None
    if run is None:
        return _node("evaluation", status="not_started", detail="尚未映射评测运行")
    return _node(
        "evaluation",
        status=_normalized_status(run.status),
        raw_status=run.status,
        versions=[_ref("evaluation_run", run, f"评测运行 · {run.id[:8]}", status=run.status)],
        detail=f"{len(run.model_version_ids or [])} 个模型 · {len(run.suite_version_ids or [])} 个套件",
        updated_at=_timestamp(run),
    )


def _quantization_node(
    session: Session, revision: ProjectPipelineRevision
) -> dict[str, Any]:
    run = (
        session.get(QuantizationRun, revision.quantization_run_id)
        if revision.quantization_run_id
        else None
    )
    if run is None:
        return _node("quantization", status="not_started", detail="尚未映射量化运行")
    return _node(
        "quantization",
        status=_normalized_status(run.status),
        raw_status=run.status,
        versions=[
            _ref(
                "quantization_run",
                run,
                f"{run.method} · {run.format}",
                status=run.status,
            )
        ],
        detail=f"源模型 {run.model_version_id[:8]}",
        updated_at=_timestamp(run),
    )


def _release_node(session: Session, revision: ProjectPipelineRevision) -> dict[str, Any]:
    release = session.get(Release, revision.release_id) if revision.release_id else None
    if release is None:
        return _node("release", status="not_started", detail="尚未映射发布记录")
    return _node(
        "release",
        status=_normalized_status(release.status),
        raw_status=release.status,
        versions=[
            _ref(
                "release",
                release,
                f"{release.name} · {release.version}",
                status=release.status,
            )
        ],
        detail=f"发布目标：{release.target}",
        updated_at=_timestamp(release),
    )


def project_pipeline_view(
    session: Session, pipeline: ProjectPipeline
) -> dict[str, Any]:
    revision = (
        session.get(ProjectPipelineRevision, pipeline.current_revision_id)
        if pipeline.current_revision_id
        else None
    )
    if revision is None:
        return {
            **pipeline.to_dict(),
            "revision": None,
            "status": "blocked",
            "progress": None,
            "nodes": [],
        }
    run = session.get(TrainingRun, revision.training_run_id) if revision.training_run_id else None
    data_node = _data_node(session, revision, run)
    tokenizer_node = _tokenizer_node(session, revision)
    code_node = _code_node(session, revision)
    training_node, run = _training_node(
        session, revision, [data_node, tokenizer_node, code_node]
    )
    nodes = [
        data_node,
        tokenizer_node,
        code_node,
        training_node,
        _model_node(session, revision, run),
        _evaluation_node(session, revision),
        _quantization_node(session, revision),
        _release_node(session, revision),
    ]
    attention = next(
        (
            node
            for node in reversed(nodes)
            if node["status"] in {"in_progress", "paused", "failed", "blocked", "stopped"}
        ),
        None,
    )
    if attention is None:
        attention = next((node for node in nodes if node["status"] != "ready"), nodes[-1])
    return {
        **pipeline.to_dict(),
        "revision": revision.to_dict(),
        "status": attention["status"],
        "current_node": attention["key"],
        "progress": training_node.get("progress"),
        "nodes": nodes,
    }


def list_project_pipeline_views(
    session: Session, project_id: str
) -> list[dict[str, Any]]:
    pipelines = session.scalars(
        select(ProjectPipeline)
        .where(
            ProjectPipeline.project_id == project_id,
            ProjectPipeline.archived.is_(False),
        )
        .order_by(ProjectPipeline.updated_at.desc(), ProjectPipeline.created_at.desc())
    ).all()
    return [project_pipeline_view(session, pipeline) for pipeline in pipelines]


def _all_project_data_options(session: Session, project_id: str) -> list[dict[str, Any]]:
    references: set[tuple[str, str]] = {
        (binding.asset_type, binding.asset_version_id)
        for binding in session.scalars(
            select(ProjectDataBinding).where(ProjectDataBinding.project_id == project_id)
        ).all()
        if binding.asset_type != "eval_dataset_version"
    }
    for revision in session.scalars(
        select(PlanRevision)
        .join(TrainingPlan, TrainingPlan.id == PlanRevision.plan_id)
        .where(TrainingPlan.project_id == project_id)
    ).all():
        for item in _inferred_data_refs(session, list(revision.stages or [])):
            references.add((item["asset_type"], item["asset_version_id"]))
    legacy_queries = (
        (
            "dataset_version",
            select(DatasetVersion.id)
            .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
            .where(Dataset.project_id == project_id),
        ),
        (
            "mixture_version",
            select(DataMixtureVersion.id)
            .join(DataMixture, DataMixture.id == DataMixtureVersion.mixture_id)
            .where(DataMixture.project_id == project_id),
        ),
        (
            "supervision_dataset_version",
            select(SupervisionDatasetVersion.id)
            .join(
                SupervisionDataset,
                SupervisionDataset.id == SupervisionDatasetVersion.supervision_dataset_id,
            )
            .where(SupervisionDataset.project_id == project_id),
        ),
        (
            "supervision_mixture_version",
            select(SupervisionMixtureVersion.id)
            .join(
                SupervisionMixture,
                SupervisionMixture.id == SupervisionMixtureVersion.mixture_id,
            )
            .where(SupervisionMixture.project_id == project_id),
        ),
    )
    for asset_type, query in legacy_queries:
        references.update((asset_type, item) for item in session.scalars(query).all())
    options = []
    for asset_type, asset_version_id in references:
        descriptor = asset_descriptor(session, asset_type, asset_version_id)
        if descriptor is None:
            continue
        options.append(
            {
                "asset_type": asset_type,
                "asset_version_id": asset_version_id,
                "label": f"{descriptor['name']} · V{descriptor['version']}",
                "status": descriptor["status"],
            }
        )
    return sorted(options, key=lambda item: item["label"].lower())


def project_pipeline_options(session: Session, project_id: str) -> dict[str, Any]:
    tokenizer_versions = session.scalars(
        select(TokenizerVersion)
        .where(TokenizerVersion.project_id == project_id)
        .order_by(TokenizerVersion.created_at.desc())
    ).all()
    code_rows = session.execute(
        select(CodeVersion, CodeSource)
        .join(CodeSource, CodeSource.id == CodeVersion.source_id)
        .where(CodeSource.project_id == project_id)
        .order_by(CodeVersion.created_at.desc())
    ).all()
    plan_rows = session.execute(
        select(PlanRevision, TrainingPlan)
        .join(TrainingPlan, TrainingPlan.id == PlanRevision.plan_id)
        .where(TrainingPlan.project_id == project_id)
        .order_by(PlanRevision.created_at.desc())
    ).all()
    plan_map = {revision.id: (revision, plan) for revision, plan in plan_rows}
    runs = session.scalars(
        select(TrainingRun)
        .where(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.created_at.desc())
    ).all()
    mapped_runs = {
        training_run_id: {"id": pipeline_id, "name": pipeline_name}
        for training_run_id, pipeline_id, pipeline_name in session.execute(
            select(
                ProjectPipelineRevision.training_run_id,
                ProjectPipeline.id,
                ProjectPipeline.name,
            )
            .join(
                ProjectPipeline,
                ProjectPipeline.current_revision_id == ProjectPipelineRevision.id,
            )
            .where(
                ProjectPipeline.project_id == project_id,
                ProjectPipeline.archived.is_(False),
                ProjectPipelineRevision.training_run_id.is_not(None),
            )
        ).all()
    }
    run_options = []
    for run in runs:
        revision, plan = plan_map.get(run.plan_revision_id, (None, None))
        task = session.get(Task, run.task_id)
        stages = list((run.config_snapshot or {}).get("selected_stages") or [])
        inferred_stages = stages or (list(revision.stages or []) if revision else [])
        run_options.append(
            {
                "id": run.id,
                "label": f"{plan.name if plan else '训练运行'} · {run.current_stage or run.id[:8]}",
                "status": run.status,
                "progress": round(float(task.progress if task else 0) * 100, 2),
                "objective": _objective_from_stages(stages),
                "plan_revision_id": run.plan_revision_id,
                "tokenizer_version_id": revision.tokenizer_version_id if revision else None,
                "code_version_id": revision.code_version_id if revision else None,
                "data_refs": _inferred_data_refs(session, inferred_stages),
                "mapped_pipeline_id": (
                    mapped_runs[run.id]["id"] if run.id in mapped_runs else None
                ),
                "mapped_pipeline_name": (
                    mapped_runs[run.id]["name"] if run.id in mapped_runs else None
                ),
            }
        )
    return {
        "data_versions": _all_project_data_options(session, project_id),
        "tokenizer_versions": [
            {
                "id": item.id,
                "label": f"{item.name} · V{item.version}",
                "status": item.status,
                "locked": item.locked,
            }
            for item in tokenizer_versions
        ],
        "code_versions": [
            {
                "id": version.id,
                "label": f"{source.name} · V{version.version}",
                "status": version.state,
                "git_commit": version.git_commit,
            }
            for version, source in code_rows
        ],
        "plan_revisions": [
            {
                "id": revision.id,
                "label": f"{plan.name} · R{revision.revision}",
                "plan_id": plan.id,
                "revision": revision.revision,
                "tokenizer_version_id": revision.tokenizer_version_id,
                "code_version_id": revision.code_version_id,
                "objective": _objective_from_stages(list(revision.stages or [])),
                "data_refs": _inferred_data_refs(session, list(revision.stages or [])),
            }
            for revision, plan in plan_rows
        ],
        "training_runs": run_options,
        "model_versions": [
            {
                "id": model.id,
                "label": f"{model.name} · V{model.version}",
                "status": model.status,
            }
            for model in session.scalars(
                select(ModelVersion)
                .where(ModelVersion.project_id == project_id)
                .order_by(ModelVersion.created_at.desc())
            ).all()
        ],
        "evaluation_runs": [
            {"id": run.id, "label": f"评测运行 · {run.id[:8]}", "status": run.status}
            for run in session.scalars(
                select(EvaluationRun)
                .where(EvaluationRun.project_id == project_id)
                .order_by(EvaluationRun.created_at.desc())
            ).all()
        ],
        "quantization_runs": [
            {
                "id": run.id,
                "label": f"{run.method} · {run.format}",
                "status": run.status,
            }
            for run in session.scalars(
                select(QuantizationRun)
                .where(QuantizationRun.project_id == project_id)
                .order_by(QuantizationRun.created_at.desc())
            ).all()
        ],
        "releases": [
            {
                "id": release.id,
                "label": f"{release.name} · {release.version}",
                "status": release.status,
            }
            for release in session.scalars(
                select(Release)
                .where(Release.project_id == project_id)
                .order_by(Release.created_at.desc())
            ).all()
        ],
    }
