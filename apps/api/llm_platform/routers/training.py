from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from llm_platform_core.checkpoint_compatibility import (
    build_checkpoint_compatibility,
    compare_checkpoint_compatibility,
)
from llm_platform_core.mixtures import MIXTURE_MANIFEST_FORMAT, is_identity_selection
from llm_platform_core.training_data import sequence_order_strategy
from llm_platform_core.training_resources import freeze_resource_batch_policy
from llm_platform_core.training_schedule import freeze_learning_rate_schedules
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    ChatTemplateVersion,
    Checkpoint,
    CodeSource,
    CodeVersion,
    DataMixture,
    DataMixtureVersion,
    GateResult,
    ModelSpecification,
    ModelSpecificationRevision,
    ModelVersion,
    PlanRevision,
    PluginVersion,
    Project,
    SupervisionCorpusVersion,
    SupervisionMixture,
    SupervisionMixtureVersion,
    SupervisionTrainingViewVersion,
    TokenizedCorpusVersion,
    TokenizedDatasetVersion,
    TokenizerVersion,
    TrainingDataViewVersion,
    TrainingPlan,
    TrainingRun,
    TrainingValidationSet,
    WorkerNode,
    new_id,
)
from ..schemas import (
    PlanRevisionCreate,
    SupervisionCorpusCreate,
    SupervisionTrainingViewCreate,
    TokenizedCorpusCreate,
    TrainingDataViewCreate,
    TrainingPerformanceCreate,
    TrainingPlanCreate,
    TrainingStart,
)
from ..services.artifacts import artifact_storage
from ..services.global_data import project_has_asset_access
from ..services.tasks import create_task, worker_matches_selector
from ..services.transactions import commit_or_flush
from ..stage_gates import data_consumption_mode, normalize_stages
from ..training_system_validations import (
    annotate_system_validation_issues,
    build_training_system_validations,
)
from .common import add_audit, get_or_404

router = APIRouter(tags=["training"])


@router.post("/projects/{project_id}/training-performance")
def create_training_performance(project_id: str, payload: TrainingPerformanceCreate,
                                session: Session = Depends(get_session)) -> dict:
    from ..services.training_performance import create_comparison

    version = _get_project_code_version(session, project_id, payload.engine_version_id)
    return create_comparison(session, project_id, payload, version)


@router.get("/projects/{project_id}/training-performance/{task_id}/draft")
def training_performance_draft(project_id: str, task_id: str, session: Session = Depends(get_session)):
    from copy import deepcopy

    from ..models import Task
    task = get_or_404(session, Task, task_id, "训练比较")
    if task.project_id != project_id or task.kind != "training_performance" or task.observed_state != "succeeded":
        raise HTTPException(409, "请选择当前项目已完成的训练比较")
    report = (task.result or {}).get("performance_report") or {}
    if not report.get("recommendation"):
        raise HTTPException(409, "比较没有通过验证的推荐组合")
    source_id = (task.payload.get("performance_probe") or {}).get("source_training_run_id")
    source = get_or_404(session, TrainingRun, source_id, "来源训练")
    if source.project_id != project_id:
        raise HTTPException(404, "来源训练不属于此项目")
    revision = deepcopy(source.config_snapshot["revision"])
    allowed = {"model_specification_revision_id", "tokenizer_version_id", "validation_set_id", "validation_mixture_version_id", "engine_plugin_id", "launcher_plugin_id", "resource_request"}
    spec = {k: v for k, v in revision.items() if k in allowed}
    spec.update({"stages": [{**task.payload["performance_probe"]["source_stage"], **report["recommendation"]}],
                 "code_version_id": task.payload["code"]["version_id"],
                 "engine_config": (revision.get("effective_config") or {}).get("engine_config", {}),
                 "input_model_ref": {**revision.get("input_model_ref", {}), "type": "full_state_continuation", "performance_task_id": task.id}})
    return spec


VALIDATION_TEMPLATE_NAMES = {"VALIDATION_FILE", "LLM_PLATFORM_VALIDATION_FILE"}
MAX_VALIDATION_UPLOAD_BYTES = 256 * 1024 * 1024
TEMPLATE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_CHECKPOINT_RUNTIME_ONLY_FIELDS = (
    "use_cache",
    "attention_backend",
    "deltanet_backend",
    "_attn_implementation",
)


def _template_names(value: object) -> set[str]:
    if isinstance(value, str):
        return set(TEMPLATE_PATTERN.findall(value))
    if isinstance(value, dict):
        return set().union(*(_template_names(item) for item in value.values()), set())
    if isinstance(value, list | tuple):
        return set().union(*(_template_names(item) for item in value), set())
    return set()


def _contains_validation_file_token(value: object) -> bool:
    if isinstance(value, str):
        return "__validation_file__" in value or bool(
            _template_names(value).intersection(VALIDATION_TEMPLATE_NAMES)
        )
    if isinstance(value, dict):
        return any(_contains_validation_file_token(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_validation_file_token(item) for item in value)
    return False


def _get_project_code_version(
    session: Session, project_id: str, code_version_id: str | None
) -> CodeVersion | None:
    if not code_version_id:
        return None
    version = session.get(CodeVersion, code_version_id)
    source = session.get(CodeSource, version.source_id) if version else None
    if version is None or source is None or source.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_code_version",
                "message": "代码版本不存在或不属于当前项目",
            },
        )
    return version


def _get_project_model_specification_revision(
    session: Session, project_id: str, revision_id: str | None
) -> ModelSpecificationRevision | None:
    if not revision_id:
        return None
    revision = session.get(ModelSpecificationRevision, revision_id)
    specification = (
        session.get(ModelSpecification, revision.specification_id) if revision else None
    )
    if revision is None or specification is None or specification.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_model_specification_revision",
                "message": "模型规格修订版不存在或不属于当前项目",
            },
        )
    if specification.status != "ready":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_specification_not_ready",
                "message": "模型规格尚未通过解析校验",
            },
        )
    return revision


def _plan_model_binding(
    session: Session,
    project_id: str,
    model_specification_revision_id: str | None,
    code_version_id: str | None,
    input_model_ref: dict,
) -> tuple[ModelSpecificationRevision | None, str | None, dict]:
    model_revision = _get_project_model_specification_revision(
        session, project_id, model_specification_revision_id
    )
    effective_code_version_id = code_version_id
    effective_input_model_ref = input_model_ref
    if model_revision is not None:
        effective_input_model_ref = {
            "type": "model_specification",
            "model_specification_revision_id": model_revision.id,
            "architecture_family_id": model_revision.architecture_family_id,
            "vocab_size": (model_revision.resolved_config or {}).get("vocab_size"),
            "parameter_count": (model_revision.parameter_summary or {}).get(
                "total_parameters"
            ),
            "config_signature": model_revision.config_signature,
            "initialization": model_revision.initialization or {},
        }
    _get_project_code_version(session, project_id, effective_code_version_id)
    return model_revision, effective_code_version_id, effective_input_model_ref


def _freeze_model_specification(
    session: Session, project_id: str, revision_id: str | None
) -> dict | None:
    revision = _get_project_model_specification_revision(session, project_id, revision_id)
    if revision is None:
        return None
    specification = session.get(ModelSpecification, revision.specification_id)
    runtime_requirements = revision.runtime_requirements or {}
    schema_version = (
        "forgelm.model-spec.v2"
        if runtime_requirements.get("contract_version") == 2
        else "forgelm.model-spec.v1"
    )
    frozen = {
        "schema_version": schema_version,
        "source_type": revision.source_type,
        "architecture": {
            "family_id": revision.architecture_family_id,
            "family_version": revision.architecture_version,
            **(
                {
                    "category": runtime_requirements.get("category"),
                    "implementation": runtime_requirements,
                }
                if schema_version == "forgelm.model-spec.v2"
                else {}
            ),
        },
        "config": revision.resolved_config or {},
        "initialization": revision.initialization or {},
        "parameters": revision.parameter_summary or {},
        "runtime_requirements": runtime_requirements,
        "config_signature": revision.config_signature,
    }
    return {
        "specification_id": specification.id,
        "specification_name": specification.name,
        "revision_id": revision.id,
        "revision": revision.revision,
        "source_type": revision.source_type,
        "architecture_family_id": revision.architecture_family_id,
        "architecture_version": revision.architecture_version,
        "preset_id": revision.preset_id,
        "code_version_id": revision.code_version_id,
        "resolved_config": revision.resolved_config or {},
        "initialization": revision.initialization or {},
        "parameter_summary": revision.parameter_summary or {},
        "runtime_requirements": runtime_requirements,
        "schema_version": schema_version,
        "generated_json": frozen,
        "generated_code": revision.generated_code,
        "config_signature": revision.config_signature,
    }


def _get_project_validation_set(
    session: Session, project_id: str, validation_set_id: str | None
) -> TrainingValidationSet | None:
    if not validation_set_id:
        return None
    validation_set = session.get(TrainingValidationSet, validation_set_id)
    if validation_set is None or validation_set.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_training_validation_set",
                "message": "训练验证集不存在或不属于当前项目",
            },
        )
    return validation_set


def _get_project_validation_mixture(
    session: Session, project_id: str, mixture_version_id: str | None
) -> DataMixtureVersion | None:
    if not mixture_version_id:
        return None
    version = session.get(DataMixtureVersion, mixture_version_id)
    mixture = session.get(DataMixture, version.mixture_id) if version else None
    if version is None or mixture is None or not project_has_asset_access(
        session, project_id, "mixture_version", mixture_version_id
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_validation_mixture",
                "message": "验证数据混合不存在或不属于当前项目",
            },
        )
    return version


def _validate_validation_reference(
    session: Session,
    project_id: str,
    validation_set_id: str | None,
    validation_mixture_version_id: str | None,
) -> None:
    if validation_set_id and validation_mixture_version_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "multiple_validation_inputs",
                "message": "训练验证输入只能选择文件版本或数据混合版本之一",
            },
        )
    _get_project_validation_set(session, project_id, validation_set_id)
    _get_project_validation_mixture(session, project_id, validation_mixture_version_id)


def _validation_holdout_configuration(
    training_mixture: DataMixtureVersion,
    validation_mixture: DataMixtureVersion | None,
) -> dict | None:
    """Freeze a document-level holdout when validation reuses training sources."""

    if validation_mixture is None:
        return None
    training_entries = list(training_mixture.entries or [])
    validation_entries = list(validation_mixture.entries or [])
    training_by_dataset = {
        str(entry.get("dataset_version_id") or ""): entry
        for entry in training_entries
    }
    validation_by_dataset = {
        str(entry.get("dataset_version_id") or ""): entry
        for entry in validation_entries
    }
    if (
        not training_by_dataset
        or not validation_by_dataset
        or "" in training_by_dataset
        or "" in validation_by_dataset
        or len(training_by_dataset) != len(training_entries)
        or len(validation_by_dataset) != len(validation_entries)
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_holdout_sources_invalid",
                "message": "训练或验证混合包含无法唯一映射的数据来源",
            },
        )
    overlap = set(training_by_dataset) & set(validation_by_dataset)
    if not overlap:
        return None
    if set(training_by_dataset) != set(validation_by_dataset):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_holdout_partial_overlap",
                "message": "验证混合与训练混合仅部分重叠，无法证明验证集独立；请使用完全同源混合或完全独立的数据来源",
                "overlapping_dataset_version_ids": sorted(overlap),
            },
        )
    mismatched = [
        dataset_version_id
        for dataset_version_id, training_entry in training_by_dataset.items()
        if (training_entry.get("selection") or {})
        != (validation_by_dataset[dataset_version_id].get("selection") or {})
    ]
    if mismatched:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_holdout_selection_mismatch",
                "message": "同源训练与验证混合使用了不同选择语义，无法按 Tokenized Corpus 文档身份安全留出",
                "dataset_version_ids": sorted(mismatched),
            },
        )
    if not validation_mixture.manifest_hash or int(validation_mixture.target_tokens or 0) <= 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_holdout_identity_missing",
                "message": "验证混合缺少 Manifest 身份或正数 Token 目标",
            },
        )
    sources = []
    for training_entry in training_entries:
        dataset_version_id = str(training_entry.get("dataset_version_id") or "")
        validation_entry = validation_by_dataset[dataset_version_id]
        sources.append(
            {
                "dataset_version_id": dataset_version_id,
                "weight_units": max(int(validation_entry.get("weight_units") or 0), 0),
            }
        )
    if not any(source["weight_units"] for source in sources):
        for source in sources:
            source["weight_units"] = 1
    return {
        "strategy": "weighted_random_tokenized_document_holdout_v1",
        "target_tokens": int(validation_mixture.target_tokens),
        "seed": int(validation_mixture.seed or 0),
        "validation_mixture_version_id": validation_mixture.id,
        "validation_manifest_hash": validation_mixture.manifest_hash,
        "sources": sources,
    }


def _validation_holdout_for_stages(
    session: Session,
    revision: PlanRevision,
    stages: list[dict],
) -> dict | None:
    if not revision.validation_mixture_version_id:
        return None
    mixture_ids = {
        str(stage.get("data_ref_id") or "")
        for stage in stages
        if isinstance(stage.get("data_preparation"), dict)
    }
    if len(mixture_ids) != 1 or "" in mixture_ids:
        return None
    training_mixture = session.get(DataMixtureVersion, next(iter(mixture_ids)))
    validation_mixture = session.get(
        DataMixtureVersion, revision.validation_mixture_version_id
    )
    if training_mixture is None or validation_mixture is None:
        return None
    return _validation_holdout_configuration(training_mixture, validation_mixture)


def revision_view(revision: PlanRevision) -> dict:
    view = revision.to_dict()
    stages = normalize_stages(list(revision.stages or []))
    view["stages"] = stages
    effective_config = dict(view.get("effective_config") or {})
    effective_config["schema_version"] = max(
        2, int(effective_config.get("schema_version") or 1)
    )
    effective_config["stages"] = stages
    view["effective_config"] = effective_config
    return view


def plan_view(session: Session, plan: TrainingPlan) -> dict:
    revisions = session.scalars(
        select(PlanRevision)
        .where(PlanRevision.plan_id == plan.id)
        .order_by(PlanRevision.revision.desc())
    ).all()
    current = (
        session.get(PlanRevision, plan.current_revision_id) if plan.current_revision_id else None
    )
    return {
        **plan.to_dict(),
        "current_revision": revision_view(current) if current else None,
        "revisions": [revision_view(revision) for revision in revisions],
    }


def _freeze_mixture_sources(
    session: Session, mixture_version: DataMixtureVersion
) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()
    for entry in mixture_version.entries or []:
        artifact_id = str(entry.get("dataset_artifact_id") or "")
        if not artifact_id or artifact_id in seen:
            continue
        artifact = session.get(Artifact, artifact_id)
        if artifact is None or artifact.kind != "dataset_snapshot":
            continue
        sources.append(
            {
                "dataset_version_id": entry.get("dataset_version_id"),
                "artifact_id": artifact.id,
                "content_hash": str(
                    entry.get("dataset_artifact_hash") or artifact.content_hash
                ),
            }
        )
        seen.add(artifact_id)
    return sources


def _freeze_data_inputs(session: Session, project_id: str, stages: list[dict]) -> list[dict]:
    """Freeze resolved mixture manifests without passing recipes to downstream commands."""
    inputs: list[dict] = []
    seen: set[str] = set()
    for stage in stages:
        reference_id = str(stage.get("data_ref_id") or "")
        if not reference_id or reference_id in seen:
            continue
        mixture_version = session.get(DataMixtureVersion, reference_id)
        if mixture_version is None:
            continue
        mixture = session.get(DataMixture, mixture_version.mixture_id)
        if (
            mixture is None
            or not project_has_asset_access(
                session, project_id, "mixture_version", reference_id
            )
            or mixture_version.status != "ready"
            or not mixture_version.manifest_artifact_id
            or not mixture_version.manifest_hash
        ):
            continue
        manifest = session.get(Artifact, mixture_version.manifest_artifact_id)
        inputs.append(
            {
                "format": str(
                    (manifest.metadata_json or {}).get("format")
                    or MIXTURE_MANIFEST_FORMAT
                ),
                "mixture_id": mixture.id,
                "mixture_name": mixture.name,
                "mixture_version_id": mixture_version.id,
                "version": mixture_version.version,
                "target_tokens": mixture_version.target_tokens,
                "seed": mixture_version.seed,
                "resolver_version": mixture_version.resolver_version,
                "sampler_version": mixture_version.sampler_version,
                "manifest_artifact_id": mixture_version.manifest_artifact_id,
                "manifest_hash": mixture_version.manifest_hash,
                "resolved_document_count": mixture_version.resolved_document_count,
                "resolved_token_estimate": mixture_version.resolved_token_estimate,
                "source_artifacts": _freeze_mixture_sources(session, mixture_version),
            }
        )
        seen.add(reference_id)
    return inputs


def _requires_tokenized_data(revision: PlanRevision) -> bool:
    engine_config = (revision.effective_config or {}).get("engine_config") or {}
    return engine_config.get("data_input_mode") == "tokenized_required"


def _objective_kind(stage: dict) -> str:
    objective = stage.get("objective") if isinstance(stage.get("objective"), dict) else {}
    return str(objective.get("kind") or stage.get("type") or "pretrain").lower()


def _post_training_configuration(stages: list[dict]) -> dict | None:
    if not stages or not all(
        _objective_kind(stage) in {"sft", "dpo"}
        and isinstance(stage.get("objective"), dict)
        and int(stage["objective"].get("schema_version") or 0) == 1
        for stage in stages
    ):
        return None
    configurations: list[dict] = []
    for stage in stages:
        budget = stage.get("budget") if isinstance(stage.get("budget"), dict) else {}
        objective = _objective_kind(stage)
        default_unit = "supervised_tokens" if objective == "sft" else "preference_pairs"
        configurations.append(
            {
                "objective_kind": objective,
                "objective": stage.get("objective") or {},
                "mixture_version_id": str(stage.get("data_ref_id") or ""),
                "chat_template_version_id": str(stage.get("chat_template_version_id") or ""),
                "budget_unit": str(budget.get("unit") or default_unit),
                "budget_value": int(budget.get("value") or stage.get("target_tokens") or 0),
                "preparation": stage.get("supervision_preparation") or {},
            }
        )
    if len(configurations) > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "native_post_training_requires_single_stage",
                "message": "原生 SFT/DPO 一次运行只接受一个阶段；请分别启动并用 Checkpoint 串联。",
            },
        )
    return configurations[0]


def _freeze_dpo_reference_checkpoint(
    session: Session,
    project_id: str,
    stages: list[dict],
    resume_checkpoint: dict | None,
) -> dict | None:
    if not stages or _objective_kind(stages[0]) != "dpo":
        return None
    references = []
    for stage in stages:
        objective = stage.get("objective") if isinstance(stage.get("objective"), dict) else {}
        reference = objective.get("reference") if isinstance(objective.get("reference"), dict) else {}
        if reference.get("id"):
            references.append((str(reference.get("type") or "checkpoint"), str(reference["id"])))
    references = list(dict.fromkeys(references))
    if len(references) > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "multiple_dpo_references",
                "message": "A native DPO run can use only one frozen reference checkpoint.",
            },
        )
    if not references:
        return dict(resume_checkpoint) if resume_checkpoint else None
    reference_type, checkpoint_id = references[0]
    if reference_type != "checkpoint":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_dpo_reference",
                "message": "Native DPO currently accepts a checkpoint reference.",
            },
        )
    checkpoint = session.get(Checkpoint, checkpoint_id)
    artifact = (
        session.get(Artifact, checkpoint.artifact_id)
        if checkpoint is not None and checkpoint.artifact_id
        else None
    )
    if (
        checkpoint is None
        or checkpoint.project_id != project_id
        or checkpoint.status != "available"
        or artifact is None
        or artifact.kind != "checkpoint"
        or artifact_storage.inference_access(session, artifact, project_id)["type"]
        == "unavailable"
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "dpo_reference_unavailable",
                "message": "The selected DPO reference checkpoint is not available to a Worker.",
            },
        )
    source_model = session.scalar(
        select(ModelVersion).where(ModelVersion.source_checkpoint_id == checkpoint.id)
    )
    return {
        "checkpoint_id": checkpoint.id,
        "training_run_id": checkpoint.training_run_id,
        "source_model_version_id": source_model.id if source_model else None,
        "step": checkpoint.step,
        "token_count": checkpoint.token_count,
        "artifact_id": artifact.id,
        "content_hash": artifact.content_hash,
        "mode": "frozen_reference",
    }


def _freeze_supervision_training_view_inputs(
    session: Session, project_id: str, stages: list[dict]
) -> list[dict]:
    ids = list(
        dict.fromkeys(
            str(stage.get("supervision_training_view_id") or "")
            for stage in stages
            if stage.get("supervision_training_view_id")
        )
    )
    bindings: list[dict] = []
    for view_id in ids:
        view = session.get(SupervisionTrainingViewVersion, view_id)
        view_artifact = session.get(Artifact, view.artifact_id) if view and view.artifact_id else None
        corpus = session.get(SupervisionCorpusVersion, view.corpus_version_id) if view else None
        corpus_artifact = (
            session.get(Artifact, corpus.artifact_id) if corpus and corpus.artifact_id else None
        )
        template_version = (
            session.get(ChatTemplateVersion, corpus.chat_template_version_id) if corpus else None
        )
        if (
            view is None
            or view.project_id != project_id
            or view.status != "ready"
            or view_artifact is None
            or view_artifact.kind != "supervision_training_index"
            or view_artifact.content_hash != view.manifest_hash
            or corpus is None
            or corpus.status != "ready"
            or corpus_artifact is None
            or corpus_artifact.kind != "supervision_corpus"
            or corpus_artifact.content_hash != corpus.manifest_hash
            or template_version is None
        ):
            continue
        bindings.append(
            {
                "version_id": view.id,
                "identity_hash": view.identity_hash,
                "format": view.format,
                "objective_kind": view.objective_kind,
                "budget_unit": view.budget_unit,
                "budget_value": view.budget_value,
                "example_count": view.example_count,
                "pair_count": view.pair_count,
                "raw_token_count": view.raw_token_count,
                "supervised_token_count": view.supervised_token_count,
                "artifact_id": view_artifact.id,
                "content_hash": view_artifact.content_hash,
                "configuration": (view.config_snapshot or {}).get("configuration") or {},
                "supervision_corpus": {
                    "version_id": corpus.id,
                    "identity_hash": corpus.identity_hash,
                    "format": corpus.format,
                    "objective_kind": corpus.objective_kind,
                    "tokenizer_version_id": corpus.tokenizer_version_id,
                    "chat_template_version_id": corpus.chat_template_version_id,
                    "chat_template": {
                        "version_id": template_version.id,
                        "content_hash": template_version.content_hash,
                        "template_body": template_version.template_body,
                        "required_special_tokens": template_version.required_special_tokens,
                        "assistant_mask_mode": template_version.assistant_mask_mode,
                        "renderer_version": template_version.renderer_version,
                    },
                    "artifact_id": corpus_artifact.id,
                    "content_hash": corpus_artifact.content_hash,
                },
            }
        )
    return bindings


def _find_supervision_training_view_inputs(
    session: Session,
    project_id: str,
    stages: list[dict],
    tokenizer_version_id: str | None,
) -> list[dict]:
    explicit = _freeze_supervision_training_view_inputs(session, project_id, stages)
    if explicit:
        return explicit
    config = _post_training_configuration(stages)
    if config is None or config.get("custom_execution") or not tokenizer_version_id:
        return []
    corpora = session.scalars(
        select(SupervisionCorpusVersion).where(
            SupervisionCorpusVersion.project_id == project_id,
            SupervisionCorpusVersion.mixture_version_id == config["mixture_version_id"],
            SupervisionCorpusVersion.tokenizer_version_id == tokenizer_version_id,
            SupervisionCorpusVersion.chat_template_version_id
            == config["chat_template_version_id"],
            SupervisionCorpusVersion.objective_kind == config["objective_kind"],
            SupervisionCorpusVersion.status == "ready",
        )
    ).all()
    if not corpora:
        return []
    views = session.scalars(
        select(SupervisionTrainingViewVersion)
        .where(
            SupervisionTrainingViewVersion.project_id == project_id,
            SupervisionTrainingViewVersion.corpus_version_id.in_(
                [corpus.id for corpus in corpora]
            ),
            SupervisionTrainingViewVersion.objective_kind == config["objective_kind"],
            SupervisionTrainingViewVersion.budget_unit == config["budget_unit"],
            SupervisionTrainingViewVersion.budget_value == config["budget_value"],
            SupervisionTrainingViewVersion.status == "ready",
        )
        .order_by(SupervisionTrainingViewVersion.created_at.desc())
    ).all()
    for view in views:
        binding = _freeze_supervision_training_view_inputs(
            session, project_id, [{"supervision_training_view_id": view.id}]
        )
        if binding:
            return binding
    return []


def _tokenized_requirements(stages: list[dict]) -> list[dict]:
    requirements: dict[tuple[str, int], dict] = {}
    running_tokens = 0
    running_operational_tokens = 0
    for stage in stages:
        target_tokens = int(stage.get("target_tokens") or 0)
        running_tokens += target_tokens
        effective_batch_tokens = int(stage.get("effective_batch_tokens") or 0)
        operational_tokens = (
            math.ceil(target_tokens / effective_batch_tokens) * effective_batch_tokens
            if target_tokens > 0 and effective_batch_tokens > 0
            else target_tokens
        )
        running_operational_tokens += operational_tokens
        mixture_version_id = str(stage.get("data_ref_id") or "")
        sequence_length = int(stage.get("sequence_length") or 0)
        if not mixture_version_id or sequence_length <= 0:
            continue
        coverage_mode = data_consumption_mode(stage)
        coverage = (
            max(
                int(stage.get("cumulative_target_tokens") or running_tokens),
                running_operational_tokens,
            )
            if coverage_mode == "unique_only"
            else sequence_length
        )
        key = (mixture_version_id, sequence_length)
        current = requirements.get(key)
        if current is None:
            requirements[key] = {
                "mixture_version_id": mixture_version_id,
                "sequence_length": sequence_length,
                "required_tokens": coverage,
                "coverage_mode": coverage_mode,
            }
        elif coverage_mode == "unique_only":
            current["coverage_mode"] = "unique_only"
            current["required_tokens"] = max(
                coverage, int(current["required_tokens"])
            )
    return list(requirements.values())


def _training_data_range(
    stages: list[dict], sequence_length: int, *, sequence_start: int = 0
) -> dict:
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    if sequence_start < 0:
        raise ValueError("training data sequence start cannot be negative")
    required = 0
    planned_train_tokens = 0
    planned_steps = 0
    step_count_available = True
    stage_ranges: list[dict] = []
    stage_start = sequence_start
    for stage in stages:
        target_tokens = int(stage.get("target_tokens") or 0)
        effective_batch_tokens = int(stage.get("effective_batch_tokens") or 0)
        if target_tokens <= 0:
            continue
        if effective_batch_tokens <= 0:
            operational_tokens = math.ceil(target_tokens / sequence_length) * sequence_length
            stage_steps = None
            step_count_available = False
        else:
            stage_steps = math.ceil(target_tokens / effective_batch_tokens)
            operational_tokens = stage_steps * effective_batch_tokens
        if operational_tokens % sequence_length:
            raise ValueError("effective batch tokens must align to Tokenized Dataset rows")
        stage_sequences = operational_tokens // sequence_length
        stage_end = stage_start + stage_sequences
        stage_ranges.append(
            {
                "name": str(stage.get("name") or ""),
                "target_tokens": target_tokens,
                "effective_batch_tokens": effective_batch_tokens or None,
                "planned_steps": stage_steps,
                "planned_train_tokens": operational_tokens,
                "sequence_start": stage_start,
                "sequence_count": stage_sequences,
                "sequence_end": stage_end,
            }
        )
        required += stage_sequences
        planned_train_tokens += operational_tokens
        if stage_steps is not None:
            planned_steps += stage_steps
        stage_start = stage_end
    return {
        "schema_version": 1,
        "sequence_length": sequence_length,
        "sequence_start": sequence_start,
        "sequence_count": required,
        "sequence_end": sequence_start + required,
        "planned_steps": planned_steps if step_count_available else None,
        "planned_train_tokens": planned_train_tokens,
        "stage_ranges": stage_ranges,
    }


def _required_training_sequences(stages: list[dict], sequence_length: int) -> int:
    return int(_training_data_range(stages, sequence_length)["sequence_count"])


def _estimated_stage_checkpoint_count(stage: dict) -> int:
    target_tokens = int(stage.get("target_tokens") or 0)
    interval_tokens = int(stage.get("checkpoint_interval_tokens") or 0)
    if interval_tokens <= 0:
        interval_steps = int(stage.get("checkpoint_interval") or 0)
        effective_batch_tokens = int(stage.get("effective_batch_tokens") or 0)
        interval_tokens = interval_steps * effective_batch_tokens
    periodic = max(0, (target_tokens - 1) // interval_tokens) if interval_tokens > 0 else 0
    return periodic + int(bool(stage.get("checkpoint_at_end", interval_tokens > 0)))


def _freeze_tokenized_data_inputs(
    session: Session,
    project_id: str,
    stages: list[dict],
    tokenizer_version_id: str | None,
) -> list[dict]:
    if not tokenizer_version_id:
        return []
    bindings: list[dict] = []
    for requirement in _tokenized_requirements(stages):
        candidates = session.scalars(
            select(TokenizedDatasetVersion)
            .where(
                TokenizedDatasetVersion.project_id == project_id,
                TokenizedDatasetVersion.mixture_version_id
                == requirement["mixture_version_id"],
                TokenizedDatasetVersion.tokenizer_version_id == tokenizer_version_id,
                TokenizedDatasetVersion.sequence_length
                == requirement["sequence_length"],
                TokenizedDatasetVersion.token_count
                >= requirement["required_tokens"],
                TokenizedDatasetVersion.status == "ready",
                TokenizedDatasetVersion.format == "forgelm-packed-tokens-v1",
                TokenizedDatasetVersion.packing_version == "document-pack-v1",
            )
            .order_by(TokenizedDatasetVersion.token_count.asc())
        ).all()
        selected: tuple[TokenizedDatasetVersion, Artifact] | None = None
        for candidate in candidates:
            artifact = (
                session.get(Artifact, candidate.artifact_id)
                if candidate.artifact_id
                else None
            )
            if (
                artifact is not None
                and artifact.kind == "tokenized_dataset"
                and candidate.manifest_hash == artifact.content_hash
                and artifact_storage.inference_access(session, artifact, project_id)["type"]
                != "unavailable"
            ):
                selected = candidate, artifact
                break
        if selected is None:
            continue
        version, artifact = selected
        bindings.append(
            {
                "version_id": version.id,
                "identity_hash": version.identity_hash,
                "format": version.format,
                "packing_version": version.packing_version,
                "mixture_version_id": version.mixture_version_id,
                "tokenizer_version_id": version.tokenizer_version_id,
                "sequence_length": version.sequence_length,
                "target_tokens": version.target_tokens,
                "token_count": version.token_count,
                "sequence_count": version.sequence_count,
                "shard_count": version.shard_count,
                "artifact_id": artifact.id,
                "content_hash": artifact.content_hash,
                "required_tokens": requirement["required_tokens"],
                "coverage_mode": requirement["coverage_mode"],
            }
        )
    return bindings


def _freeze_training_data_view_inputs(
    session: Session,
    project_id: str,
    stages: list[dict],
    tokenizer_version_id: str | None,
) -> list[dict]:
    view_ids = list(
        dict.fromkeys(
            str(stage.get("training_data_view_id") or "")
            for stage in stages
            if stage.get("training_data_view_id")
        )
    )
    bindings: list[dict] = []
    for view_id in view_ids:
        version = session.get(TrainingDataViewVersion, view_id)
        artifact = (
            session.get(Artifact, version.artifact_id)
            if version is not None and version.artifact_id
            else None
        )
        if (
            version is None
            or version.project_id != project_id
            or version.tokenizer_version_id != tokenizer_version_id
            or version.status != "ready"
            or version.format != "forgelm-training-index-v1"
            or artifact is None
            or artifact.kind != "training_data_index"
            or artifact.content_hash != version.manifest_hash
            or artifact_storage.inference_access(session, artifact, project_id)["type"]
            == "unavailable"
        ):
            continue
        corpus_bindings: list[dict] = []
        valid_corpora = True
        configuration = (version.config_snapshot or {}).get("configuration") or {}
        frozen_corpora = list((version.config_snapshot or {}).get("corpora") or [])
        strategy = sequence_order_strategy(
            str(configuration.get("sequence_order_policy") or "sequential_v1")
        )
        if not strategy["materializes_sequences"]:
            for frozen in frozen_corpora:
                corpus = session.get(
                    TokenizedCorpusVersion, str(frozen.get("version_id") or "")
                )
                corpus_artifact = (
                    session.get(Artifact, corpus.artifact_id)
                    if corpus is not None and corpus.artifact_id
                    else None
                )
                if (
                    corpus is None
                    or corpus.status != "ready"
                    or corpus_artifact is None
                    or corpus_artifact.kind != "tokenized_corpus"
                    or corpus_artifact.content_hash != corpus.manifest_hash
                    or corpus_artifact.content_hash != frozen.get("content_hash")
                ):
                    valid_corpora = False
                    break
                corpus_bindings.append({**frozen, "artifact_id": corpus_artifact.id})
        if not valid_corpora or (
            not strategy["materializes_sequences"] and not corpus_bindings
        ):
            continue
        bindings.append(
            {
                "version_id": version.id,
                "identity_hash": version.identity_hash,
                "format": version.format,
                "mixture_version_id": version.mixture_version_id,
                "tokenizer_version_id": version.tokenizer_version_id,
                "sequence_length": version.sequence_length,
                "requested_tokens": version.requested_tokens,
                "token_count": version.token_count,
                "sequence_count": version.sequence_count,
                "configuration": configuration,
                "validation_holdout": (artifact.metadata_json or {}).get(
                    "validation_holdout"
                ),
                "warnings": version.warnings or [],
                "artifact_id": artifact.id,
                "content_hash": artifact.content_hash,
                "prefix_lineage": (version.config_snapshot or {}).get(
                    "prefix_lineage"
                ),
                "source_corpora": frozen_corpora,
                "tokenized_corpora": corpus_bindings,
            }
        )
    return bindings


DEFAULT_CORPUS_SELECTION = {"mode": "all", "estimated_token_limit": None}
DEFAULT_CORPUS_ENCODING = {
    "eos_policy": "append_if_available",
    "empty_document_policy": "skip",
    "token_dtype": "auto",
    "shard_token_limit": 268_435_456,
}
DEFAULT_TRAINING_DATA_VIEW = {
    "consumption_policy": "unique_only",
    "tail_policy": "drop",
    "padding_token_id": None,
    "packing_policy": "document_stream_v1",
    "sampling_policy": "weighted_deficit_v1",
    "sequence_order_policy": "block_shuffle_v1",
    "shuffle_block_sequences": 16,
    "seed": 42,
}


def _managed_data_configuration(
    stages: list[dict],
    *,
    sequence_start: int = 0,
    validation_holdout: dict | None = None,
) -> dict | None:
    """Resolve one explicit training-data intent for the selected run.

    A run currently freezes one index. Multiple stages are therefore accepted only
    when their mixture, sequence length and preparation semantics are identical.
    The requested View size is a capacity lower bound that covers the frozen run
    range, including full-batch rounding and an optional checkpoint cursor.
    """
    managed = [stage for stage in stages if isinstance(stage.get("data_preparation"), dict)]
    if not managed:
        return None
    if len(managed) != len(stages):
        raise ValueError("所选阶段不能混用自动准备与历史 View 绑定")
    configurations: list[dict] = []
    for stage in managed:
        preparation = stage.get("data_preparation") or {}
        corpus = preparation.get("corpus") or {}
        view = preparation.get("view") or {}
        configurations.append(
            {
                "mixture_version_id": str(stage.get("data_ref_id") or ""),
                "sequence_length": int(stage.get("sequence_length") or 0),
                "selection": {**DEFAULT_CORPUS_SELECTION, **(corpus.get("selection") or {})},
                "encoding": {**DEFAULT_CORPUS_ENCODING, **(corpus.get("encoding") or {})},
                "view": {**DEFAULT_TRAINING_DATA_VIEW, **view},
                "execution": preparation.get("execution") or {},
            }
        )
    comparable = configurations
    if any(item != comparable[0] for item in comparable[1:]):
        raise ValueError("一次运行所选阶段的数据混合、序列长度和准备策略必须一致；请分阶段运行")
    resolved = configurations[0]
    if validation_holdout is not None:
        resolved["view"] = {
            **resolved["view"],
            "validation_holdout": validation_holdout,
        }
    training_range = _training_data_range(
        stages,
        int(resolved["sequence_length"]),
        sequence_start=sequence_start,
    )
    resolved["training_data_range"] = training_range
    resolved["requested_tokens"] = (
        int(training_range["sequence_end"]) * int(resolved["sequence_length"])
    )
    return resolved


def _find_managed_training_data_view(
    session: Session,
    project_id: str,
    stages: list[dict],
    tokenizer_version_id: str | None,
    *,
    sequence_start: int = 0,
    validation_holdout: dict | None = None,
    required_prefix_view: dict | None = None,
) -> tuple[list[dict], dict | None]:
    config = _managed_data_configuration(
        stages,
        sequence_start=sequence_start,
        validation_holdout=validation_holdout,
    )
    if config is None or not tokenizer_version_id:
        return [], config
    if sequence_start > 0 and required_prefix_view is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "checkpoint_training_data_view_unknown",
                "message": (
                    "Checkpoint 没有冻结产生该数据游标的 Training Data View；"
                    "不能把数值游标应用到另一个读取顺序"
                ),
            },
        )
    expected_semantics = {
        "sequence_length": config["sequence_length"],
        **config["view"],
    }
    required_sequence_end = int(config["training_data_range"]["sequence_end"])
    candidates = session.scalars(
        select(TrainingDataViewVersion)
        .where(
            TrainingDataViewVersion.project_id == project_id,
            TrainingDataViewVersion.mixture_version_id == config["mixture_version_id"],
            TrainingDataViewVersion.tokenizer_version_id == tokenizer_version_id,
            TrainingDataViewVersion.status == "ready",
        )
        .order_by(
            TrainingDataViewVersion.sequence_count.asc(),
            TrainingDataViewVersion.created_at.desc(),
        )
    ).all()
    for candidate in candidates:
        if required_prefix_view is not None:
            required_identity = str(
                required_prefix_view.get("identity_hash") or ""
            )
            required_content_hash = str(
                required_prefix_view.get("content_hash") or ""
            )
            preserves_prefix = (
                candidate.identity_hash == required_identity
                and candidate.manifest_hash == required_content_hash
            )
            if not preserves_prefix:
                lineage = (candidate.config_snapshot or {}).get(
                    "prefix_lineage"
                ) or {}
                preserves_prefix = any(
                    str(ancestor.get("identity_hash") or "")
                    == required_identity
                    and str(ancestor.get("content_hash") or "")
                    == required_content_hash
                    and int(ancestor.get("sequence_count") or 0)
                    >= sequence_start
                    for ancestor in (lineage.get("ancestor_views") or [])
                )
            if not preserves_prefix:
                continue
        candidate_config = (
            (candidate.config_snapshot or {}).get("configuration") or {}
        )
        candidate_semantics = {
            key: candidate_config.get(key) for key in expected_semantics
        }
        if candidate_semantics != expected_semantics:
            continue
        if int(candidate.sequence_count or 0) < required_sequence_end:
            continue
        bindings = _freeze_training_data_view_inputs(
            session,
            project_id,
            [{"training_data_view_id": candidate.id}],
            tokenizer_version_id,
        )
        if not bindings:
            continue
        corpora_match = all(
            (
                (corpus := session.get(
                    TokenizedCorpusVersion, str(frozen.get("version_id") or "")
                ))
                is not None
                and (
                    (corpus.selection_config or {}) == config["selection"]
                    or (frozen.get("selection_input") and (
                        (corpus.selection_config or {}).get("mode") == "all"
                        or ((corpus.selection_config or {}).get("mode") == "mixture"
                            and (corpus.selection_config or {}).get("mixture_version_id") == config["mixture_version_id"])
                    ))
                )
                and (corpus.encoding_config or {}) == config["encoding"]
            )
            for frozen in (
                bindings[0].get("source_corpora")
                or bindings[0].get("tokenized_corpora")
                or []
            )
        )
        if corpora_match:
            return bindings, config
    return [], config


def _freeze_tokenizer_binding(
    session: Session, project_id: str, tokenizer_version_id: str | None
) -> dict | None:
    """Freeze a usable tokenizer artifact for command-based training engines.

    Historical plans can contain locked metadata-only tokenizer versions, so the
    binding remains optional for compatibility. New model sources can require the
    runtime binding in their own entrypoint.
    """

    if not tokenizer_version_id:
        return None
    tokenizer = session.get(TokenizerVersion, tokenizer_version_id)
    artifact = (
        session.get(Artifact, tokenizer.artifact_id)
        if tokenizer is not None and tokenizer.artifact_id
        else None
    )
    if (
        tokenizer is None
        or tokenizer.project_id != project_id
        or not tokenizer.locked
        or artifact is None
        or artifact.kind != "tokenizer"
        or artifact_storage.managed_payload_path(session, artifact, project_id) is None
    ):
        return None
    return {
        "version_id": tokenizer.id,
        "name": tokenizer.name,
        "version": tokenizer.version,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": list((tokenizer.config or {}).get("special_tokens") or []),
        "artifact_id": artifact.id,
        "content_hash": artifact.content_hash,
    }


def _plan_model_configuration(
    session: Session, plan: TrainingPlan | None, revision: PlanRevision
) -> tuple[dict | None, str | None]:
    if revision.model_specification_revision_id:
        model_revision = session.get(
            ModelSpecificationRevision, revision.model_specification_revision_id
        )
        if model_revision is None:
            return None, "模型规格修订版不存在"
        return dict(model_revision.resolved_config or {}), None
    reference = revision.input_model_ref or {}
    config_path = str(reference.get("config_path") or "").strip()
    if not config_path or plan is None or not revision.code_version_id:
        return None, None
    code_version = session.get(CodeVersion, revision.code_version_id)
    artifact = (
        session.get(Artifact, code_version.artifact_id)
        if code_version is not None and code_version.artifact_id
        else None
    )
    if artifact is None:
        return None, None
    try:
        config = artifact_storage.read_managed_archive_json(
            session, artifact, plan.project_id, config_path
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"无法读取代码版本中的模型配置 {config_path}：{exc}"
    if config is None:
        return None, None
    if not isinstance(config, dict):
        return None, f"模型配置 {config_path} 必须是 JSON 对象"
    return config, None


def _model_vocab_contract(
    session: Session, plan: TrainingPlan | None, revision: PlanRevision
) -> tuple[int | None, str | None]:
    reference = revision.input_model_ref or {}
    declared = reference.get("vocab_size")
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            return None, "模型输入引用的 vocab_size 必须是正整数"
        return declared, None
    config, error = _plan_model_configuration(session, plan, revision)
    if error:
        return None, error
    if config is None or "vocab_size" not in config:
        return None, None
    vocab_size = config.get("vocab_size")
    if revision.model_specification_revision_id:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            return None, "模型规格解析结果中的 vocab_size 必须是正整数"
        return vocab_size, None
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        config_path = str(reference.get("config_path") or "").strip()
        return None, f"模型配置 {config_path} 的 vocab_size 必须是正整数"
    return vocab_size, None


def _plan_checkpoint_compatibility(
    session: Session, plan: TrainingPlan | None, revision: PlanRevision
) -> tuple[dict | None, str | None]:
    configuration, error = _plan_model_configuration(session, plan, revision)
    if error or configuration is None:
        return None, error
    reference = revision.input_model_ref or {}
    declared = reference.get("checkpoint_compatibility") or {}
    if not isinstance(declared, dict):
        return None, "checkpoint_compatibility 必须是对象"
    if revision.model_specification_revision_id:
        model_revision = session.get(
            ModelSpecificationRevision, revision.model_specification_revision_id
        )
        if model_revision is None:
            return None, "模型规格修订版不存在"
        runtime_requirements = model_revision.runtime_requirements or {}
        contract = str(
            declared.get("contract")
            or f"{model_revision.architecture_family_id}.checkpoint"
        )
        contract_version = declared.get("contract_version") or 1
        runtime_only_fields = declared.get("runtime_only_fields")
        if runtime_only_fields is None:
            runtime_only_fields = runtime_requirements.get("checkpoint_runtime_only_fields") or []
    else:
        contract = str(
            declared.get("contract")
            or reference.get("architecture")
            or configuration.get("model_name")
            or "legacy-model.checkpoint"
        )
        contract_version = declared.get("contract_version") or 1
        runtime_only_fields = declared.get("runtime_only_fields")
        if runtime_only_fields is None:
            runtime_only_fields = DEFAULT_CHECKPOINT_RUNTIME_ONLY_FIELDS
    if not isinstance(runtime_only_fields, list | tuple):
        return None, "checkpoint_compatibility.runtime_only_fields 必须是字段路径列表"
    try:
        return (
            build_checkpoint_compatibility(
                configuration,
                contract=contract,
                contract_version=int(contract_version),
                runtime_only_fields=runtime_only_fields,
            ),
            None,
        )
    except (TypeError, ValueError) as exc:
        return None, f"Checkpoint 兼容性契约无效：{exc}"


def _checkpoint_model_compatibility(
    checkpoint: Checkpoint,
    source_model: ModelVersion | None,
    expected: dict | None,
) -> dict | None:
    metrics = checkpoint.metrics or {}
    metadata = source_model.metadata_json or {} if source_model is not None else {}
    metric_descriptor = metrics.get("model_compatibility")
    if isinstance(metric_descriptor, dict):
        return {**metric_descriptor, "provenance": "checkpoint_metrics"}
    metadata_descriptor = metadata.get("model_compatibility")
    if isinstance(metadata_descriptor, dict):
        return {**metadata_descriptor, "provenance": "model_metadata"}
    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict) or expected is None:
        return None
    try:
        return {
            **build_checkpoint_compatibility(
            architecture,
            contract=str(expected.get("contract") or ""),
            contract_version=int(expected.get("contract_version") or 1),
            runtime_only_fields=expected.get("runtime_only_fields") or [],
            ),
            "provenance": "model_metadata_fallback",
        }
    except (TypeError, ValueError):
        return None


def _checkpoint_compatibility_message(report: dict) -> str:
    differences = report.get("differences") or []
    if not differences:
        return str(report.get("message") or "Checkpoint 模型兼容性校验失败")
    rendered = "；".join(
        f"{item.get('field')}: Checkpoint={item.get('actual')!r}, 计划={item.get('expected')!r}"
        for item in differences[:8]
    )
    suffix = f"；另有 {len(differences) - 8} 项差异" if len(differences) > 8 else ""
    return f"Checkpoint 模型结构或语义配置不兼容：{rendered}{suffix}"


def _freeze_validation_binding(
    session: Session, project_id: str, validation_set_id: str | None
) -> dict | None:
    if not validation_set_id:
        return None
    validation_set = session.get(TrainingValidationSet, validation_set_id)
    artifact = (
        session.get(Artifact, validation_set.artifact_id) if validation_set is not None else None
    )
    if (
        validation_set is None
        or validation_set.project_id != project_id
        or validation_set.status != "ready"
        or artifact is None
        or artifact.kind != "training_validation"
        or artifact_storage.managed_payload_path(session, artifact, project_id) is None
    ):
        return None
    return {
        "validation_set_id": validation_set.id,
        "name": validation_set.name,
        "version": validation_set.version,
        "format": validation_set.format,
        "filename": validation_set.filename,
        "document_count": validation_set.document_count,
        "estimated_tokens": validation_set.estimated_tokens,
        "size_bytes": validation_set.size_bytes,
        "artifact_id": artifact.id,
        "content_hash": artifact.content_hash,
    }


def _freeze_validation_mixture_binding(
    session: Session, project_id: str, mixture_version_id: str | None
) -> dict | None:
    if not mixture_version_id:
        return None
    inputs = _freeze_data_inputs(
        session, project_id, [{"data_ref_id": mixture_version_id}]
    )
    if not inputs:
        return None
    frozen = inputs[0]
    return {
        **frozen,
        "usage": "training_validation",
        "validation_sampling": {
            "strategy": "weighted_random_materialized_v1",
            "seed": int(frozen.get("seed") or 0),
            "materialization_format": "forgelm-validation-text-v1",
        },
    }


def _effective_validation_data_input(
    validation_mixture_binding: dict | None,
    training_data_view_inputs: list[dict],
) -> dict | None:
    """Prefer a proven same-view holdout over raw same-source materialization."""

    if validation_mixture_binding is None:
        return None
    configured = [
        item
        for item in training_data_view_inputs
        if isinstance((item.get("configuration") or {}).get("validation_holdout"), dict)
    ]
    if not configured:
        return validation_mixture_binding
    if len(configured) != 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "validation_holdout_view_ambiguous",
                "message": "训练运行必须冻结唯一的验证留出 Training Data View",
            },
        )
    view = configured[0]
    request = (view.get("configuration") or {}).get("validation_holdout") or {}
    actual = view.get("validation_holdout") or {}
    if (
        request.get("validation_mixture_version_id")
        != validation_mixture_binding.get("mixture_version_id")
        or actual.get("strategy") != request.get("strategy")
        or actual.get("validation_mixture_version_id")
        != request.get("validation_mixture_version_id")
        or not actual.get("identity_hash")
        or int(actual.get("sequence_count") or 0) <= 0
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "validation_holdout_not_materialized",
                "message": "Training Data View 没有发布与验证混合一致的文档级留出制品",
            },
        )
    return {
        "format": "forgelm-validation-token-holdout-v1",
        "usage": "training_validation",
        "target_tokens": int(request.get("target_tokens") or 0),
        "seed": int(request.get("seed") or 0),
        "mixture_version_id": request.get("validation_mixture_version_id"),
        "training_data_view_version_id": view.get("version_id"),
        "training_data_view_identity_hash": view.get("identity_hash"),
        "holdout_identity_hash": actual.get("identity_hash"),
        "sequence_count": int(actual.get("sequence_count") or 0),
        "token_count": int(actual.get("token_count") or 0),
    }


def _training_source_artifact_ids(*bindings: dict | list[dict] | None) -> list[str]:
    artifact_ids: list[str] = []
    for binding in bindings:
        items = binding if isinstance(binding, list) else [binding] if binding else []
        for item in items:
            for source in item.get("source_artifacts") or []:
                artifact_id = str(source.get("artifact_id") or "")
                if artifact_id and artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)
    return artifact_ids


def _selected_stages(
    revision: PlanRevision, selected_stage_indexes: list[int] | None
) -> tuple[list[dict], list[int]]:
    if selected_stage_indexes is None:
        indexes = list(range(len(revision.stages or [])))
    else:
        indexes = list(selected_stage_indexes)
        if not indexes:
            raise HTTPException(
                status_code=422,
                detail={"code": "no_selected_stages", "message": "至少选择一个训练阶段"},
            )
        if len(set(indexes)) != len(indexes):
            raise HTTPException(
                status_code=422,
                detail={"code": "duplicate_stage_indexes", "message": "训练阶段不能重复选择"},
            )
        invalid = [index for index in indexes if index < 0 or index >= len(revision.stages or [])]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_stage_indexes",
                    "message": f"训练阶段索引超出范围：{invalid}",
                },
            )
        indexes.sort()
    return normalize_stages([revision.stages[index] for index in indexes]), indexes


def _freeze_resume_checkpoint(
    session: Session,
    *,
    plan: TrainingPlan,
    revision: PlanRevision,
    selected_stage_indexes: list[int],
    checkpoint_id: str | None,
) -> dict | None:
    first_index = selected_stage_indexes[0] if selected_stage_indexes else 0
    selected_stages = normalize_stages(
        [revision.stages[index] for index in selected_stage_indexes]
    )
    post_training = _post_training_configuration(selected_stages)
    native_post_training = post_training is not None and not post_training.get(
        "custom_execution"
    )
    input_model_ref = revision.input_model_ref or {}
    if input_model_ref.get("type") == "checkpoint_continuation":
        from ..services.training_continuation import freeze_native
        if first_index != 0 or checkpoint_id:
            raise HTTPException(409, "Checkpoint 分段接续必须从目标首阶段开始")
        try:
            return freeze_native(session, plan.project_id, revision, selected_stages)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    if input_model_ref.get("type") == "full_state_continuation":
        from ..services.training_performance import freeze_continuation

        if first_index != 0 or checkpoint_id:
            raise HTTPException(409, "完整状态接续必须从同一阶段开始且不能覆盖 Checkpoint")
        return freeze_continuation(session, plan.project_id, revision, selected_stages)
    if (
        not checkpoint_id
        and first_index == 0
        and str(input_model_ref.get("type") or "") == "checkpoint_finetune"
    ):
        checkpoint_id = str(input_model_ref.get("checkpoint_id") or "").strip() or None
        if checkpoint_id is None:
            source_model_id = str(
                input_model_ref.get("source_model_version_id")
                or input_model_ref.get("model_version_id")
                or ""
            ).strip()
            source_model = session.get(ModelVersion, source_model_id) if source_model_id else None
            checkpoint_id = (
                str(source_model.source_checkpoint_id).strip()
                if source_model is not None and source_model.source_checkpoint_id
                else None
            )
        if checkpoint_id is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "input_model_checkpoint_required",
                    "message": (
                        "训练计划声明从 Checkpoint 微调，但起始模型引用没有可用 Checkpoint；"
                        "为避免静默随机初始化，运行已被阻止"
                    ),
                },
            )
    if not checkpoint_id:
        if first_index > 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "resume_checkpoint_required",
                    "message": "跳过计划前置阶段时必须选择一个可用 Checkpoint 接续",
                },
            )
        return None
    checkpoint = session.get(Checkpoint, checkpoint_id)
    source_run = (
        session.get(TrainingRun, checkpoint.training_run_id) if checkpoint is not None else None
    )
    source_revision = (
        session.get(PlanRevision, source_run.plan_revision_id) if source_run is not None else None
    )
    # A new corpus has its own cursor. Only an explicit first-stage weight fork
    # may discard the source plan's data position; ordinary resumes keep it.
    pretrain_weight_fork = bool(
        first_index == 0
        and selected_stages
        and all(stage.get("type") == "pretrain" for stage in selected_stages)
        and _requires_tokenized_data(revision)
        and input_model_ref.get("type") == "checkpoint_finetune"
        and input_model_ref.get("data_cursor_policy") == "reset"
        and source_revision is not None
        and source_revision.plan_id != plan.id
    )
    artifact = (
        session.get(Artifact, checkpoint.artifact_id)
        if checkpoint is not None and checkpoint.artifact_id
        else None
    )
    if (
        checkpoint is None
        or checkpoint.project_id != plan.project_id
        or source_run is None
        or source_revision is None
        or (
            not native_post_training
            and not pretrain_weight_fork
            and source_revision.plan_id != plan.id
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "invalid_resume_checkpoint",
                "message": (
                    "接续 Checkpoint 不存在、属于其他项目，或当前训练类型不允许跨计划分叉"
                ),
            },
        )
    if checkpoint.status != "available" or artifact is None or artifact.kind != "checkpoint":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "resume_checkpoint_unavailable",
                "message": "接续 Checkpoint 当前不可用或没有有效 checkpoint 制品",
            },
        )
    access = artifact_storage.inference_access(session, artifact, plan.project_id)
    if access["type"] == "unavailable":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "resume_checkpoint_artifact_unavailable",
                "message": "接续 Checkpoint 没有托管副本或可用 Worker 位置",
            },
        )
    expected_prior_tokens = sum(
        int(stage.get("target_tokens") or 0) for stage in (revision.stages or [])[:first_index]
    )
    metrics = checkpoint.metrics or {}
    source_model = session.scalar(
        select(ModelVersion).where(ModelVersion.source_checkpoint_id == checkpoint.id)
    )
    expected_model_compatibility, _compatibility_error = _plan_checkpoint_compatibility(
        session, plan, revision
    )
    checkpoint_model_compatibility = _checkpoint_model_compatibility(
        checkpoint, source_model, expected_model_compatibility
    )
    tokenized_required = _requires_tokenized_data(revision) and not native_post_training
    sequence_offset = (
        0 if pretrain_weight_fork
        else None if native_post_training
        else metrics.get("sequence_offset")
    )
    if tokenized_required:
        if sequence_offset is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_sequence_offset_unknown",
                    "message": (
                        "Checkpoint 没有 Tokenized Dataset 序列游标，不能在预分词制品上安全续训"
                    ),
                },
            )
        try:
            sequence_offset = int(sequence_offset)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_checkpoint_sequence_offset",
                    "message": "Checkpoint 的 Tokenized Dataset 序列游标无效",
                },
            ) from exc
        if sequence_offset < 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_checkpoint_sequence_offset",
                    "message": "Checkpoint 的 Tokenized Dataset 序列游标不能为负数",
                },
            )

    sample_offset = (
        0 if pretrain_weight_fork
        else None if native_post_training
        else metrics.get("sample_offset")
    )
    if sample_offset is not None:
        try:
            sample_offset = int(sample_offset)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_checkpoint_sample_offset",
                    "message": "Checkpoint 的数据采样游标无效，不能安全接续",
                },
            ) from exc
        if sample_offset < 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_checkpoint_sample_offset",
                    "message": "Checkpoint 的数据采样游标不能为负数",
                },
            )
    elif first_index > 0 and checkpoint.token_count != expected_prior_tokens:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "checkpoint_sample_offset_unknown",
                "message": (
                    "历史 Checkpoint 没有数据采样游标，且累计 Tokens 与所选阶段的前置门禁不一致"
                ),
            },
        )
    source_view_inputs = (
        []
        if native_post_training or pretrain_weight_fork
        else list((source_run.config_snapshot or {}).get("training_data_view_inputs") or [])
    )
    source_training_data_view = None
    if source_view_inputs:
        if len(source_view_inputs) != 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_training_data_view_ambiguous",
                    "message": "Checkpoint 来源运行没有冻结唯一的 Training Data View",
                },
            )
        source_frozen = source_view_inputs[0]
        source_bindings = _freeze_training_data_view_inputs(
            session,
            plan.project_id,
            [
                {
                    "training_data_view_id": str(
                        source_frozen.get("version_id") or ""
                    )
                }
            ],
            source_run.config_snapshot.get("revision", {}).get(
                "tokenizer_version_id"
            )
            or source_frozen.get("tokenizer_version_id"),
        )
        if (
            len(source_bindings) != 1
            or source_bindings[0]["identity_hash"]
            != source_frozen.get("identity_hash")
            or source_bindings[0]["content_hash"]
            != source_frozen.get("content_hash")
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_training_data_view_unavailable",
                    "message": "Checkpoint 来源运行冻结的 Training Data View 当前不可验证",
                },
            )
        source_training_data_view = source_bindings[0]
        if int(source_training_data_view.get("sequence_count") or 0) < int(
            sequence_offset or 0
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "checkpoint_training_data_cursor_out_of_range",
                    "message": "Checkpoint 游标已超出来源 Training Data View 的冻结容量",
                },
            )
    return {
        "checkpoint_id": checkpoint.id,
        "training_run_id": checkpoint.training_run_id,
        "source_model_version_id": source_model.id if source_model is not None else None,
        "step": checkpoint.step,
        "token_count": checkpoint.token_count,
        "artifact_id": artifact.id,
        "content_hash": artifact.content_hash,
        "sequence_offset": sequence_offset,
        "sample_offset": sample_offset,
        "replay_sample_offset_tokens": (
            expected_prior_tokens if sample_offset is None and first_index > 0 else 0
        ),
        **(
            {"training_data_view": source_training_data_view}
            if source_training_data_view is not None
            else {}
        ),
        **(
            {"model_compatibility": checkpoint_model_compatibility}
            if checkpoint_model_compatibility is not None
            else {}
        ),
        "mode": (
            "weights_only_new_data" if pretrain_weight_fork
            else "checkpoint_finetune" if native_post_training
            else "weights_and_data_cursor"
        ),
    }


def _route_training_resource_request(
    session: Session,
    project_id: str,
    resource_request: dict,
    artifact_ids: list[str],
    model_specification: dict | None = None,
) -> dict:
    request = {**resource_request}
    runtime_requirements = (model_specification or {}).get("runtime_requirements") or {}
    if (model_specification or {}).get("schema_version") == "forgelm.model-spec.v2":
        capability = (
            "builtin.model-runtime.custom"
            if (model_specification or {}).get("source_type") == "custom_code"
            else str(
                runtime_requirements.get("worker_capability")
                or "builtin.model-runtime.transformers"
            )
        )
        selector = dict(request.get("worker_selector") or {})
        selector["capabilities"] = list(
            dict.fromkeys([*(selector.get("capabilities") or []), capability])
        )
        request["worker_selector"] = selector
    eligible_worker_ids: set[str] | None = None
    for artifact_id in artifact_ids:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "training_dataset_snapshot_missing",
                    "message": "训练输入引用的数据快照不存在",
                },
            )
        access = artifact_storage.inference_access(session, artifact, project_id)
        if access["type"] == "unavailable":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "training_dataset_snapshot_unavailable",
                    "message": "训练输入数据快照没有托管副本或可用 Worker 位置",
                },
            )
        if access["type"] == "external" and not access["globally_accessible"]:
            source_workers = {str(value) for value in access["worker_ids"]}
            eligible_worker_ids = (
                source_workers
                if eligible_worker_ids is None
                else eligible_worker_ids & source_workers
            )
    if eligible_worker_ids is None:
        return request
    selector = dict(request.get("worker_selector") or {})
    selected_worker_id = selector.get("worker_id")
    if selected_worker_id:
        eligible_worker_ids &= {str(selected_worker_id)}
    selected_worker_ids = {str(value) for value in selector.get("worker_ids") or []}
    if selected_worker_ids:
        eligible_worker_ids &= selected_worker_ids
    if not eligible_worker_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "training_data_worker_location_mismatch",
                "message": "没有一个 Worker 能同时访问全部训练与验证数据快照",
            },
        )
    if len(eligible_worker_ids) == 1:
        selector.pop("worker_ids", None)
        selector["worker_id"] = next(iter(eligible_worker_ids))
    else:
        selector.pop("worker_id", None)
        selector["worker_ids"] = sorted(eligible_worker_ids)
    request["worker_selector"] = selector
    return request


@router.get("/projects/{project_id}/training-validation-sets")
def list_training_validation_sets(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    validation_sets = session.scalars(
        select(TrainingValidationSet)
        .where(TrainingValidationSet.project_id == project_id)
        .order_by(TrainingValidationSet.name, TrainingValidationSet.version.desc())
    ).all()
    return [validation_set.to_dict() for validation_set in validation_sets]


@router.post("/projects/{project_id}/training-validation-sets/upload", status_code=201)
def upload_training_validation_set(
    project_id: str,
    name: Annotated[str, Form(min_length=1, max_length=180)],
    file: Annotated[UploadFile, File()],
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    clean_name = name.strip()
    filename = Path(file.filename or "validation.jsonl").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jsonl", ".ndjson", ".txt"}:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_validation_format",
                "message": "验证集仅支持 JSONL、NDJSON 或 TXT",
            },
        )
    payload = file.file.read(MAX_VALIDATION_UPLOAD_BYTES + 1)
    if len(payload) > MAX_VALIDATION_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "validation_file_too_large", "message": "验证集不能超过 256 MiB"},
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_not_utf8", "message": "验证集必须使用 UTF-8 编码"},
        ) from exc

    non_empty_lines = [(index, line) for index, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not non_empty_lines:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_validation_set", "message": "验证集没有非空样本"},
        )
    file_format = "jsonl" if suffix in {".jsonl", ".ndjson"} else "txt"
    if file_format == "jsonl":
        for line_number, line in non_empty_lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_validation_jsonl",
                        "message": f"验证集第 {line_number} 行不是有效 JSON",
                    },
                ) from exc
            if isinstance(item, str):
                sample_text = item
            elif isinstance(item, dict):
                sample_text = next(
                    (
                        item[key]
                        for key in ("text", "content", "document")
                        if isinstance(item.get(key), str)
                    ),
                    None,
                )
            else:
                sample_text = None
            if not sample_text or not sample_text.strip():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "missing_validation_text",
                        "message": (
                            f"验证集第 {line_number} 行必须是字符串，或包含 text、content、document 文本字段"
                        ),
                    },
                )

    version = int(
        session.scalar(
            select(func.coalesce(func.max(TrainingValidationSet.version), 0)).where(
                TrainingValidationSet.project_id == project_id,
                TrainingValidationSet.name == clean_name,
            )
        )
        or 0
    ) + 1
    estimated_tokens = max(1, (len(payload) + 3) // 4)
    artifact = artifact_storage.put_bytes(
        session,
        project_id=project_id,
        kind="training_validation",
        payload=payload,
        filename=filename,
        media_type="application/x-ndjson" if file_format == "jsonl" else "text/plain",
        metadata={
            "name": clean_name,
            "format": file_format,
            "document_count": len(non_empty_lines),
            "estimated_tokens": estimated_tokens,
        },
    )
    validation_set = TrainingValidationSet(
        project_id=project_id,
        name=clean_name,
        version=version,
        format=file_format,
        filename=filename,
        document_count=len(non_empty_lines),
        estimated_tokens=estimated_tokens,
        size_bytes=len(payload),
        status="ready",
        artifact_id=artifact.id,
        metadata_json={"token_estimate_method": "utf8_bytes_div_4"},
    )
    session.add(validation_set)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="training_validation_set.upload",
        subject_type="training_validation_set",
        subject_id=validation_set.id,
    )
    session.commit()
    return validation_set.to_dict()


@router.get("/projects/{project_id}/training-plans")
def list_training_plans(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    plans = session.scalars(
        select(TrainingPlan)
        .where(TrainingPlan.project_id == project_id)
        .order_by(TrainingPlan.updated_at.desc())
    ).all()
    return [plan_view(session, plan) for plan in plans]


@router.post("/projects/{project_id}/training-plans", status_code=201)
def create_training_plan(
    project_id: str, payload: TrainingPlanCreate, session: Session = Depends(get_session)
) -> dict:
    if not payload.stages:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_stages", "message": "训练计划至少需要一个阶段"},
    )
    _model_revision, code_version_id, input_model_ref = _plan_model_binding(
        session,
        project_id,
        payload.model_specification_revision_id,
        payload.code_version_id,
        payload.input_model_ref,
    )
    _validate_validation_reference(
        session,
        project_id,
        payload.validation_set_id,
        payload.validation_mixture_version_id,
    )
    plan = TrainingPlan(project_id=project_id, name=payload.name, description=payload.description)
    session.add(plan)
    session.flush()
    revision = PlanRevision(
        plan_id=plan.id,
        revision=1,
        stages=payload.stages,
        input_model_ref=input_model_ref,
        model_specification_revision_id=payload.model_specification_revision_id,
        tokenizer_version_id=payload.tokenizer_version_id,
        validation_set_id=payload.validation_set_id,
        validation_mixture_version_id=payload.validation_mixture_version_id,
        code_version_id=code_version_id,
        engine_plugin_id=payload.engine_plugin_id,
        launcher_plugin_id=payload.launcher_plugin_id,
        resource_request=payload.resource_request,
        effective_config={
            "schema_version": 2,
            "stages": payload.stages,
            "engine_config": payload.engine_config,
        },
    )
    session.add(revision)
    session.flush()
    plan.current_revision_id = revision.id
    add_audit(
        session,
        project_id=project_id,
        action="training_plan.create",
        subject_type="training_plan",
        subject_id=plan.id,
    )
    commit_or_flush(session)
    return plan_view(session, plan)


@router.post("/training-plans/{plan_id}/revisions", status_code=201)
def create_plan_revision(
    plan_id: str, payload: PlanRevisionCreate, session: Session = Depends(get_session)
) -> dict:
    plan = get_or_404(session, TrainingPlan, plan_id, "训练计划")
    current_revision = session.get(PlanRevision, plan.current_revision_id)
    if current_revision and (current_revision.effective_config or {}).get("training_instance_id"):
        raise HTTPException(status_code=409, detail={"code": "training_instance_immutable", "message": "该配置属于不可变训练实例，请复制实例后修改"})
    _model_revision, code_version_id, input_model_ref = _plan_model_binding(
        session,
        plan.project_id,
        payload.model_specification_revision_id,
        payload.code_version_id,
        payload.input_model_ref,
    )
    _validate_validation_reference(
        session,
        plan.project_id,
        payload.validation_set_id,
        payload.validation_mixture_version_id,
    )
    latest = int(
        session.scalar(
            select(func.coalesce(func.max(PlanRevision.revision), 0)).where(
                PlanRevision.plan_id == plan_id
            )
        )
        or 0
    )
    revision = PlanRevision(
        plan_id=plan_id,
        revision=latest + 1,
        stages=payload.stages,
        input_model_ref=input_model_ref,
        model_specification_revision_id=payload.model_specification_revision_id,
        tokenizer_version_id=payload.tokenizer_version_id,
        validation_set_id=payload.validation_set_id,
        validation_mixture_version_id=payload.validation_mixture_version_id,
        code_version_id=code_version_id,
        engine_plugin_id=payload.engine_plugin_id,
        launcher_plugin_id=payload.launcher_plugin_id,
        resource_request=payload.resource_request,
        effective_config={
            "schema_version": 2,
            "stages": payload.stages,
            "engine_config": payload.engine_config,
        },
    )
    session.add(revision)
    session.flush()
    plan.current_revision_id = revision.id
    session.commit()
    return revision_view(revision)


def run_preflight(
    session: Session,
    revision: PlanRevision,
    *,
    selected_stages: list[dict] | None = None,
    selected_stage_indexes: list[int] | None = None,
    resume_checkpoint_id: str | None = None,
) -> dict:
    issues: list[dict] = []
    executed_validations = {
        "training_plan_valid",
        "data_mixture_ready",
        "tokenizer_locked",
        "validation_input_ready",
        "engine_configuration_valid",
    }
    not_applicable_validations: set[str] = set()
    validation_evidence: dict[str, dict] = {}
    plan = session.get(TrainingPlan, revision.plan_id)
    effective_stages = normalize_stages(
        list(revision.stages if selected_stages is None else selected_stages)
    )
    effective_indexes = (
        list(range(len(revision.stages or [])))
        if selected_stage_indexes is None
        else selected_stage_indexes
    )
    if not effective_stages:
        issues.append({"severity": "blocker", "code": "no_stages", "message": "没有训练阶段"})
    learning_rate_schedules = []
    try:
        learning_rate_schedules = freeze_learning_rate_schedules(effective_stages or [])
    except (ValueError, TypeError, OverflowError) as exc:
        issues.append({
            "severity": "blocker",
            "code": "invalid_warmup_schedule",
            "message": f"预热配置无效：{exc}",
        })
    checked_data_refs: set[str] = set()
    for stage in effective_stages or []:
        reference_id = str(stage.get("data_ref_id") or "")
        if not reference_id:
            continue
        if reference_id in checked_data_refs:
            continue
        checked_data_refs.add(reference_id)
        if _post_training_configuration([stage]) is not None:
            supervision_version = session.get(SupervisionMixtureVersion, reference_id)
            supervision_mixture = (
                session.get(SupervisionMixture, supervision_version.mixture_id)
                if supervision_version is not None
                else None
            )
            if (
                plan is None
                or supervision_mixture is None
                or not project_has_asset_access(
                    session,
                    plan.project_id,
                    "supervision_mixture_version",
                    reference_id,
                )
            ):
                issues.append(
                    {
                        "severity": "blocker",
                        "code": "invalid_supervision_mixture",
                        "message": "后训练阶段引用的数据混合不存在或不属于当前项目。",
                    }
                )
            elif supervision_version.status != "ready":
                issues.append(
                    {
                        "severity": "warning",
                        "code": "supervision_mixture_not_ready",
                        "message": "后训练数据混合尚未就绪；自定义命令仍可自行处理数据。",
                    }
                )
            continue
        mixture_version = session.get(DataMixtureVersion, reference_id)
        mixture = (
            session.get(DataMixture, mixture_version.mixture_id)
            if mixture_version is not None
            else None
        )
        manifest = (
            session.get(Artifact, mixture_version.manifest_artifact_id)
            if mixture_version is not None and mixture_version.manifest_artifact_id
            else None
        )
        if (
            not reference_id
            or plan is None
            or mixture is None
            or not project_has_asset_access(
                session, plan.project_id, "mixture_version", reference_id
            )
        ):
            issues.append(
                {
                    "severity": "blocker",
                    "code": "invalid_data_mixture",
                    "message": "训练阶段引用的数据混合不存在或不属于当前项目",
                }
            )
        elif (
            mixture_version is None
            or mixture_version.status != "ready"
            or not mixture_version.locked
            or manifest is None
            or manifest.kind != "data_mixture_manifest"
            or manifest.content_hash != mixture_version.manifest_hash
            or artifact_storage.managed_payload_path(session, manifest, plan.project_id) is None
        ):
            issues.append(
                {
                    "severity": "blocker",
                    "code": "data_mixture_not_ready",
                    "message": f"数据混合 {mixture.name} 的解析 Manifest 尚未就绪或不可用",
                }
            )
    tokenizer = (
        session.get(TokenizerVersion, revision.tokenizer_version_id)
        if revision.tokenizer_version_id
        else None
    )
    if tokenizer is None:
        issues.append(
            {"severity": "blocker", "code": "missing_tokenizer", "message": "没有选择 Tokenizer"}
        )
    elif not tokenizer.locked:
        issues.append(
            {"severity": "blocker", "code": "unlocked_tokenizer", "message": "Tokenizer 尚未锁定"}
        )
    model_vocab_size, model_vocab_error = _model_vocab_contract(session, plan, revision)
    if model_vocab_error or model_vocab_size is not None:
        executed_validations.add("tokenizer_model_compatible")
    _expected_checkpoint_compatibility, checkpoint_contract_error = (
        _plan_checkpoint_compatibility(session, plan, revision)
    )
    if checkpoint_contract_error:
        issues.append(
            {
                "severity": "blocker",
                "code": "model_checkpoint_contract_invalid",
                "message": checkpoint_contract_error,
            }
        )
    model_specification_binding = None
    if revision.model_specification_revision_id:
        try:
            model_specification_binding = _freeze_model_specification(
                session,
                plan.project_id if plan is not None else "",
                revision.model_specification_revision_id,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            issues.append(
                {
                    "severity": "blocker",
                    "code": detail.get("code", "invalid_model_specification_revision"),
                    "message": detail.get("message", "模型规格修订版不可用"),
                }
            )
    if model_vocab_error:
        issues.append(
            {
                "severity": "blocker",
                "code": "model_vocab_contract_invalid",
                "message": model_vocab_error,
            }
        )
    elif (
        tokenizer is not None
        and model_vocab_size is not None
        and tokenizer.vocab_size != model_vocab_size
    ):
        issues.append(
            {
                "severity": "blocker",
                "code": "tokenizer_model_vocab_mismatch",
                "message": (
                    f"模型配置要求 {model_vocab_size:,} 词表，但所选 Tokenizer 为 "
                    f"{tokenizer.vocab_size:,}；请更换 Tokenizer 或创建匹配的新模型代码版本"
                ),
            }
        )
    validation_binding = _freeze_validation_binding(
        session,
        plan.project_id if plan is not None else "",
        revision.validation_set_id,
    )
    validation_mixture_binding = _freeze_validation_mixture_binding(
        session,
        plan.project_id if plan is not None else "",
        revision.validation_mixture_version_id,
    )
    if revision.validation_set_id and validation_binding is None:
        issues.append(
            {
                "severity": "blocker",
                "code": "training_validation_set_unavailable",
                "message": "绑定的训练验证集不存在、未就绪或制品不可用",
            }
        )
    if revision.validation_mixture_version_id and validation_mixture_binding is None:
        issues.append(
            {
                "severity": "blocker",
                "code": "validation_mixture_unavailable",
                "message": "绑定的验证数据混合尚未解析、未锁定或 Manifest 不可用",
            }
        )
    routed_resource_request = revision.resource_request or {}
    training_data_range = None
    if plan is not None:
        post_training = _post_training_configuration(effective_stages or [])
        native_post_training = post_training is not None and not post_training.get(
            "custom_execution"
        )
        supervision_training_view_inputs = (
            _find_supervision_training_view_inputs(
                session,
                plan.project_id,
                effective_stages or [],
                revision.tokenizer_version_id,
            )
            if native_post_training
            else []
        )
        training_inputs = (
            []
            if native_post_training
            else _freeze_data_inputs(session, plan.project_id, effective_stages or [])
        )
        uses_training_data_view = not native_post_training and any(
            stage.get("training_data_view_id") for stage in effective_stages or []
        )
        uses_managed_data = not native_post_training and any(
            isinstance(stage.get("data_preparation"), dict)
            for stage in effective_stages or []
        )
        tokenized_required = not native_post_training and (
            _requires_tokenized_data(revision)
            or uses_training_data_view
            or uses_managed_data
        )
        if native_post_training:
            not_applicable_validations.add("tokenized_dataset_integrity")
            if len(supervision_training_view_inputs) == 1:
                executed_validations.update(
                    {
                        "training_data_view_integrity",
                        "training_data_view_selection",
                        "training_data_view_capacity",
                    }
                )
                supervision_binding = supervision_training_view_inputs[0]
                validation_evidence["training_data_view_integrity"] = {
                    "summary": "原生后训练使用的监督 Corpus 与训练视图均已冻结并通过内容校验。",
                    "details": {
                        "supervision_training_view_id": supervision_binding[
                            "version_id"
                        ],
                        "supervision_corpus_id": supervision_binding[
                            "supervision_corpus"
                        ]["version_id"],
                        "objective_kind": supervision_binding["objective_kind"],
                    },
                }
            else:
                issues.append(
                    {
                        "severity": "blocker",
                        "code": "supervision_data_preparation_required",
                        "message": (
                            "原生后训练所需的 Supervision Corpus 与 Training View 尚未准备完成；"
                            "请先执行训练数据准备"
                        ),
                        "effective_configuration": post_training,
                    }
                )
        elif tokenized_required and not uses_training_data_view and not uses_managed_data:
            executed_validations.add("tokenized_dataset_integrity")
        else:
            not_applicable_validations.add("tokenized_dataset_integrity")
        if not native_post_training and (uses_training_data_view or uses_managed_data):
            executed_validations.update(
                {"training_data_view_integrity", "training_data_view_selection"}
            )
        elif not native_post_training:
            not_applicable_validations.update(
                {
                    "training_data_view_integrity",
                    "training_data_view_selection",
                    "training_data_view_capacity",
                }
            )
        resume_checkpoint = None
        try:
            resume_checkpoint = _freeze_resume_checkpoint(
                session,
                plan=plan,
                revision=revision,
                selected_stage_indexes=effective_indexes,
                checkpoint_id=resume_checkpoint_id,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            issues.append(
                {
                    "severity": "blocker",
                    "code": detail.get("code", "invalid_resume_checkpoint"),
                    "message": detail.get("message", "接续 Checkpoint 不可用"),
                }
            )
        if resume_checkpoint is None:
            if not resume_checkpoint_id and (
                not effective_indexes or effective_indexes[0] == 0
            ):
                not_applicable_validations.add("resume_checkpoint_compatible")
        else:
            expected_compatibility, compatibility_error = _plan_checkpoint_compatibility(
                session, plan, revision
            )
            actual_compatibility = resume_checkpoint.get("model_compatibility")
            if compatibility_error:
                issues.append(
                    {
                        "severity": "blocker",
                        "code": "resume_checkpoint_model_incompatible",
                        "message": compatibility_error,
                    }
                )
            elif expected_compatibility is None:
                validation_evidence["resume_checkpoint_compatible"] = {
                    "summary": (
                        "该历史计划没有声明模型兼容性契约；Checkpoint 身份和游标已检查，"
                        "模型结构仍由训练入口在运行时复核。"
                    )
                }
            elif not isinstance(actual_compatibility, dict):
                issues.append(
                    {
                        "severity": "warning",
                        "code": "resume_checkpoint_model_compatibility_unverified",
                        "message": (
                            "当前计划或历史 Checkpoint 没有可比较的模型兼容性描述；"
                            "系统不会把该项伪装成已通过，Worker 启动模型前仍会复核"
                        ),
                    }
                )
                validation_evidence["resume_checkpoint_compatible"] = {
                    "summary": "模型结构兼容性尚无可比较元数据，将在 Worker 阶段复核。"
                }
            else:
                compatibility_report = compare_checkpoint_compatibility(
                    expected_compatibility, actual_compatibility
                )
                compatibility_provenance = str(
                    actual_compatibility.get("provenance") or "unknown"
                )
                validation_evidence["resume_checkpoint_compatible"] = {
                    "summary": (
                        "Checkpoint 自报兼容性与计划一致。"
                        if compatibility_report["compatible"]
                        and compatibility_provenance == "checkpoint_metrics"
                        else "历史模型元数据与计划一致，实际制品配置将在 Worker 复核。"
                        if compatibility_report["compatible"]
                        else "Checkpoint 权重结构与计划不兼容。"
                    ),
                    "details": {
                        "reason": compatibility_report.get("reason"),
                        "provenance": compatibility_provenance,
                        "differences": compatibility_report.get("differences") or [],
                        "runtime_differences": compatibility_report.get(
                            "runtime_differences"
                        )
                        or [],
                    },
                }
                if compatibility_report["compatible"]:
                    if compatibility_provenance == "checkpoint_metrics":
                        executed_validations.add("resume_checkpoint_compatible")
                    else:
                        issues.append(
                            {
                                "severity": "warning",
                                "code": "resume_checkpoint_model_compatibility_unverified",
                                "message": (
                                    "历史 Checkpoint 只有模型登记元数据，没有由 Checkpoint "
                                    "结果写入的兼容性描述；结构元数据初步一致，Worker 将读取"
                                    "实际制品配置后再放行训练入口"
                                ),
                            }
                        )
                else:
                    issues.append(
                        {
                            "severity": "blocker",
                            "code": "resume_checkpoint_model_incompatible",
                            "message": _checkpoint_compatibility_message(
                                compatibility_report
                            ),
                            "differences": compatibility_report.get("differences")
                            or [],
                            "runtime_differences": compatibility_report.get(
                                "runtime_differences"
                            )
                            or [],
                        }
                    )
        sequence_start = int((resume_checkpoint or {}).get("sequence_offset") or 0)
        managed_config = None
        try:
            if native_post_training:
                training_data_view_inputs = []
            elif uses_managed_data:
                validation_holdout = _validation_holdout_for_stages(
                    session, revision, effective_stages or []
                )
                training_data_view_inputs, managed_config = _find_managed_training_data_view(
                    session,
                    plan.project_id,
                    effective_stages or [],
                    revision.tokenizer_version_id,
                    sequence_start=sequence_start,
                    validation_holdout=validation_holdout,
                    required_prefix_view=(resume_checkpoint or {}).get(
                        "training_data_view"
                    ),
                )
                training_data_range = managed_config["training_data_range"]
            else:
                training_data_view_inputs = (
                    _freeze_training_data_view_inputs(
                        session,
                        plan.project_id,
                        effective_stages or [],
                        revision.tokenizer_version_id,
                    )
                    if uses_training_data_view
                    else []
                )
        except (ValueError, HTTPException) as exc:
            training_data_view_inputs = []
            detail = exc.detail if isinstance(exc, HTTPException) and isinstance(exc.detail, dict) else {}
            issues.append(
                {
                    "severity": "blocker",
                    "code": detail.get(
                        "code", "training_data_configuration_mismatch"
                    ),
                    "message": detail.get("message", str(exc)),
                }
            )
        tokenized_inputs = (
            _freeze_tokenized_data_inputs(
                session,
                plan.project_id,
                effective_stages or [],
                revision.tokenizer_version_id,
            )
            if tokenized_required and not uses_training_data_view and not uses_managed_data
            else []
        )
        tokenized_requirements = _tokenized_requirements(effective_stages or [])
        if uses_training_data_view and len(training_data_view_inputs) != 1:
            issues.append(
                {
                    "severity": "blocker",
                    "code": (
                        "multiple_training_data_views"
                        if len(training_data_view_inputs) > 1
                        else "training_data_view_unavailable"
                    ),
                    "message": (
                        "一次训练运行必须明确绑定一个就绪且完整的 Training Data View；"
                        "所选阶段不能混用多个视图"
                    ),
                }
            )
        if uses_managed_data and len(training_data_view_inputs) != 1 and managed_config is not None:
            issues.append(
                {
                    "severity": "blocker",
                    "code": "training_data_preparation_required",
                    "message": "训练数据尚未准备完成；请先构建或等待系统复用所需制品",
                    "effective_configuration": managed_config,
                }
            )
        if tokenized_required and not uses_training_data_view and not uses_managed_data and len(tokenized_inputs) != len(tokenized_requirements):
            missing = [
                requirement
                for requirement in tokenized_requirements
                if not any(
                    binding["mixture_version_id"]
                    == requirement["mixture_version_id"]
                    and binding["sequence_length"] == requirement["sequence_length"]
                    and binding["token_count"] >= requirement["required_tokens"]
                    for binding in tokenized_inputs
                )
            ]
            issues.append(
                {
                    "severity": "blocker",
                    "code": "tokenized_dataset_unavailable",
                    "message": (
                        "训练要求冻结 Tokenized Dataset，但没有 ready 且覆盖所选累计 "
                        "Token 范围的兼容版本；请先在数据中心创建预分词制品"
                    ),
                    "requirements": missing,
                }
            )
        effective_tokenized_inputs = training_data_view_inputs or tokenized_inputs
        if tokenized_required and len(effective_tokenized_inputs) == 1:
            binding = effective_tokenized_inputs[0]
            sequence_length = int(binding["sequence_length"])
            try:
                training_data_range = _training_data_range(
                    effective_stages or [],
                    sequence_length,
                    sequence_start=sequence_start,
                )
            except ValueError as exc:
                issues.append(
                    {
                        "severity": "blocker",
                        "code": "tokenized_dataset_batch_alignment_invalid",
                        "message": str(exc),
                    }
                )
            else:
                if uses_training_data_view or uses_managed_data:
                    executed_validations.add("training_data_view_capacity")
                required_end = int(training_data_range["sequence_end"])
                consumption_policy = (
                    (binding.get("configuration") or {}).get("consumption_policy")
                    or binding.get("coverage_mode")
                )
                physical_range_required = uses_training_data_view or uses_managed_data
                if (
                    physical_range_required or consumption_policy == "unique_only"
                ) and int(binding["sequence_count"]) < required_end:
                    issues.append(
                        {
                            "severity": "blocker",
                            "code": (
                                "training_data_view_range_insufficient"
                                if uses_training_data_view or uses_managed_data
                                else "tokenized_dataset_range_insufficient"
                            ),
                            "message": (
                                "唯一数据的实际容量不足以覆盖续训游标之后的完整训练范围；"
                                "请降低阶段目标 Tokens、显式改为允许重复，或取消本次启动"
                            ),
                            "required_sequence_end": required_end,
                            "available_sequences": int(binding["sequence_count"]),
                            "decision_options": [
                                "reduce_target_tokens",
                                "repeat_epochs",
                                "cancel",
                            ],
                        }
                    )
        if native_post_training:
            effective_validation_input = None
        else:
            try:
                effective_validation_input = _effective_validation_data_input(
                    validation_mixture_binding, training_data_view_inputs
                )
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                effective_validation_input = None
                issues.append(
                    {
                        "severity": "blocker",
                        "code": detail.get("code", "validation_holdout_unavailable"),
                        "message": detail.get("message", "验证留出制品不可用"),
                    }
                )
        if native_post_training:
            source_artifact_ids = [
                str(artifact_id)
                for binding in supervision_training_view_inputs
                for artifact_id in [
                    binding["artifact_id"],
                    binding["supervision_corpus"]["artifact_id"],
                ]
            ]
        elif tokenized_required:
            source_artifact_ids = [
                str(artifact_id)
                for binding in effective_tokenized_inputs
                for artifact_id in [
                    binding["artifact_id"],
                    *[
                        corpus["artifact_id"]
                        for corpus in binding.get("tokenized_corpora") or []
                    ],
                ]
            ] + _training_source_artifact_ids([], effective_validation_input)
        else:
            source_artifact_ids = _training_source_artifact_ids(
                training_inputs, effective_validation_input
            )
        if resume_checkpoint:
            source_artifact_ids.append(str(resume_checkpoint["artifact_id"]))
        try:
            routed_resource_request = _route_training_resource_request(
                session,
                plan.project_id,
                routed_resource_request,
                source_artifact_ids,
                model_specification_binding,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            issues.append(
                {
                    "severity": "blocker",
                    "code": detail.get("code", "training_data_unavailable"),
                    "message": detail.get(
                        "message", "训练与验证数据对所选 Worker 不可用"
                    ),
                }
            )
    if revision.code_version_id:
        executed_validations.add("code_version_ready")
        code_version = session.get(CodeVersion, revision.code_version_id)
        code_source = (
            session.get(CodeSource, code_version.source_id) if code_version is not None else None
        )
        if (
            code_version is None
            or code_source is None
            or plan is None
            or code_source.project_id != plan.project_id
        ):
            issues.append(
                {
                    "severity": "blocker",
                    "code": "invalid_code_version",
                    "message": "绑定的代码版本不存在或不属于当前项目",
                }
            )
        elif code_version.state != "ready" or not code_version.artifact_id:
            issues.append(
                {
                    "severity": "blocker",
                    "code": "code_version_not_ready",
                    "message": "绑定的代码版本尚未导入完成",
                }
            )
        else:
            artifact = session.get(Artifact, code_version.artifact_id)
            if (
                artifact is None
                or artifact.kind != "source_code"
                or artifact_storage.managed_payload_path(session, artifact, plan.project_id) is None
            ):
                issues.append(
                    {
                        "severity": "blocker",
                        "code": "code_artifact_unavailable",
                        "message": "绑定的代码版本没有可用 source_code 制品",
                    }
                )
            configured_cwd = str(
                ((revision.effective_config or {}).get("engine_config") or {}).get("cwd") or ""
            )
            if configured_cwd and Path(configured_cwd).is_absolute():
                issues.append(
                    {
                        "severity": "warning",
                        "code": "absolute_cwd_with_code_version",
                        "message": "训练已绑定代码版本，但绝对工作目录仍指向 Worker 外部路径",
                    }
                )
    else:
        not_applicable_validations.add("code_version_ready")
    gpus = int(routed_resource_request.get("gpus", 0))
    if gpus < 0:
        issues.append(
            {"severity": "blocker", "code": "invalid_gpu_request", "message": "GPU 数量不能为负数"}
        )
    selector = routed_resource_request.get("worker_selector")
    if selector is not None and not isinstance(selector, dict):
        issues.append(
            {
                "severity": "blocker",
                "code": "invalid_worker_selector",
                "message": "Worker 选择器必须是对象",
            }
        )
    elif selector:
        online_workers = session.scalars(
            select(WorkerNode).where(WorkerNode.status == "online")
        ).all()
        matching_workers = [
            worker
            for worker in online_workers
            if worker_matches_selector(worker, routed_resource_request)
            and (
                "*" in set(worker.capabilities or [])
                or {
                    revision.engine_plugin_id,
                    revision.launcher_plugin_id,
                }.issubset(set(worker.capabilities or []))
            )
        ]
        if not matching_workers:
            issues.append(
                {
                    "severity": "warning",
                    "code": "worker_selector_unavailable",
                    "message": "当前没有在线 Worker 同时匹配选择器和执行能力，任务将保持排队",
                }
            )
    engine = session.scalar(
        select(PluginVersion).where(
            PluginVersion.plugin_id == revision.engine_plugin_id,
            PluginVersion.enabled.is_(True),
        )
    )
    if engine is None:
        issues.append(
            {
                "severity": "blocker",
                "code": "engine_unavailable",
                "message": "训练引擎未登记或未启用",
            }
        )
    engine_config = (revision.effective_config or {}).get("engine_config", {})
    if revision.engine_plugin_id.startswith("builtin.command.") and not engine_config.get(
        "executable"
    ):
        issues.append(
            {
                "severity": "blocker",
                "code": "missing_executable",
                "message": "用户命令训练引擎没有配置可执行文件",
            }
        )
    template_names = _template_names(engine_config)
    if _contains_validation_file_token(engine_config) and validation_binding is None:
        issues.append(
            {
                "severity": "blocker",
                "code": "missing_training_validation_set",
                "message": "训练命令需要验证文件，但计划修订尚未选择训练验证集",
            }
        )
    unresolved_names = template_names - (
        VALIDATION_TEMPLATE_NAMES if validation_binding is not None else set()
    )
    if unresolved_names:
        issues.append(
            {
                "severity": "blocker",
                "code": "unresolved_engine_config_template",
                "message": (
                    "训练引擎配置仍包含未填写的模板变量："
                    + "、".join(f"${{{name}}}" for name in sorted(unresolved_names))
                ),
            }
        )
    if plan is not None:
        executed_validations.add("worker_capability_available")
    annotate_system_validation_issues(issues)
    blockers = [issue for issue in issues if issue["severity"] == "blocker"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    system_validations = build_training_system_validations(
        issues,
        executed=executed_validations,
        not_applicable=not_applicable_validations,
        evidence=validation_evidence,
    )
    estimates = {
        "parameters": (
            (model_specification_binding or {}).get("parameter_summary")
            or revision.effective_config.get("parameters", "由模型配置决定")
        ),
        "total_steps": sum(int(stage.get("target_steps", 0)) for stage in effective_stages),
        "total_tokens": sum(int(stage.get("target_tokens", 0)) for stage in effective_stages),
        "checkpoint_count": max(
            1, sum(_estimated_stage_checkpoint_count(stage) for stage in effective_stages)
        ),
        "resource_request": routed_resource_request,
    }
    return {
        "status": "blocked" if blockers else "warning" if warnings else "passed",
        "issues": issues,
        "system_validations": system_validations,
        "estimates": estimates,
        "learning_rate_schedules": learning_rate_schedules,
        "revision_id": revision.id,
        "selected_stage_indexes": effective_indexes,
        "resume_checkpoint_id": resume_checkpoint_id,
        "effective_stages": effective_stages,
        "training_data_range": training_data_range,
    }


@router.get("/training-plans/{plan_id}/preflight")
def preflight_training_plan(plan_id: str, session: Session = Depends(get_session)) -> dict:
    plan = get_or_404(session, TrainingPlan, plan_id, "训练计划")
    revision = get_or_404(session, PlanRevision, plan.current_revision_id, "训练计划修订版")
    result = run_preflight(session, revision)
    session.add(
        GateResult(
            project_id=plan.project_id,
            subject_type="plan_revision",
            subject_id=revision.id,
            gate_type="training_preflight",
            status=result["status"],
            issues=result["issues"],
            input_snapshot={"revision": revision_view(revision)},
        )
    )
    session.commit()
    return result


@router.post("/training-plans/{plan_id}/preflight")
def preflight_training_selection(
    plan_id: str, payload: TrainingStart, session: Session = Depends(get_session)
) -> dict:
    plan = get_or_404(session, TrainingPlan, plan_id, "训练计划")
    revision_id = payload.revision_id or plan.current_revision_id
    revision = get_or_404(session, PlanRevision, revision_id, "训练计划修订版")
    if revision.plan_id != plan.id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_plan_revision", "message": "修订版不属于当前训练计划"},
        )
    stages, indexes = _selected_stages(revision, payload.selected_stage_indexes)
    result = run_preflight(
        session,
        revision,
        selected_stages=stages,
        selected_stage_indexes=indexes,
        resume_checkpoint_id=payload.resume_checkpoint_id,
    )
    session.add(
        GateResult(
            project_id=plan.project_id,
            subject_type="plan_revision",
            subject_id=revision.id,
            gate_type="training_preflight",
            status=result["status"],
            issues=result["issues"],
            input_snapshot={
                "revision": revision_view(revision),
                "selected_stage_indexes": indexes,
                "resume_checkpoint_id": payload.resume_checkpoint_id,
            },
        )
    )
    session.commit()
    return result


@router.post("/training-plans/{plan_id}/prepare-data")
def prepare_training_data(
    plan_id: str, payload: TrainingStart, session: Session = Depends(get_session)
) -> dict:
    """Idempotently advance the physical data dependencies for a training intent."""
    from .tokenized_corpora import _create_tokenized_corpus
    from .training_data_views import _create_training_data_view
    from .training_data_views import _view as training_data_view

    plan = get_or_404(session, TrainingPlan, plan_id, "训练计划")
    revision_id = payload.revision_id or plan.current_revision_id
    revision = get_or_404(session, PlanRevision, revision_id, "训练计划修订版")
    if revision.plan_id != plan.id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_plan_revision", "message": "修订版不属于当前训练计划"},
        )
    stages, indexes = _selected_stages(revision, payload.selected_stage_indexes)
    trigger_context = {
        "source": "training_plan_prepare_data",
        "action": "准备训练数据与运行提示",
        "training_plan_id": plan.id,
        "training_plan_name": plan.name,
        "plan_revision_id": revision.id,
        "plan_revision": revision.revision,
        "selected_stages": [
            {
                "index": index,
                "name": str(stage.get("name") or f"Stage {index}"),
                "type": str(stage.get("type") or stage.get("workflow") or "training"),
            }
            for stage, index in zip(stages, indexes, strict=True)
        ],
    }
    post_training = _post_training_configuration(stages)
    if post_training is not None:
        if all(stage.get("supervision_training_view_id") for stage in stages):
            bindings = _freeze_supervision_training_view_inputs(session, plan.project_id, stages)
            return {"status": "ready", "ready": bool(bindings), "selected_stage_indexes": indexes,
                "message": "使用上游节点已构建的监督读取索引", "supervision_training_view_inputs": bindings}
        if post_training.get("custom_execution"):
            return {
                "status": "custom",
                "ready": True,
                "message": "所选阶段使用不同后训练配置；数据和切换逻辑交给自定义训练命令。",
                "selected_stage_indexes": indexes,
            }
        if not revision.tokenizer_version_id:
            raise HTTPException(status_code=422, detail={"code": "missing_tokenizer"})
        if not post_training["mixture_version_id"] or not post_training[
            "chat_template_version_id"
        ]:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_supervision_binding",
                    "message": "原生后训练数据准备需要数据混合和 Chat Template。",
                },
            )
        from .post_training import create_supervision_corpus, create_supervision_training_view

        preparation = post_training.get("preparation") or {}
        corpus_config = preparation.get("corpus") or {}
        execution = preparation.get("execution") or {}
        corpus = create_supervision_corpus(
            plan.project_id,
            SupervisionCorpusCreate(
                mixture_version_id=post_training["mixture_version_id"],
                tokenizer_version_id=revision.tokenizer_version_id,
                chat_template_version_id=post_training["chat_template_version_id"],
                objective_kind=post_training["objective_kind"],
                deduplicate_questions=bool(corpus_config.get("deduplicate_questions", False)),
                max_length=int(corpus_config.get("max_length") or stages[0].get("sequence_length") or 2048),
                overflow_policy=str(corpus_config.get("overflow_policy") or "drop_overflow_v1"),
                min_completion_tokens=int(corpus_config.get("min_completion_tokens") or 1),
                token_dtype=str(corpus_config.get("token_dtype") or "auto"),
                launcher_id=str(execution.get("launcher_id") or "builtin.local-process"),
                resource_request={**(execution.get("resource_request") or {}), "gpus": 0},
            ),
            session,
        )
        if corpus.get("status") != "ready":
            return {
                "status": "preparing_corpus",
                "ready": False,
                "message": "正在构建后训练 Corpus。",
                "selected_stage_indexes": indexes,
                "corpus": corpus,
                "data_index": None,
            }
        view_config = preparation.get("view") or {}
        view = create_supervision_training_view(
            plan.project_id,
            SupervisionTrainingViewCreate(
                corpus_version_id=corpus["id"],
                config={
                    "budget_unit": post_training["budget_unit"],
                    "budget_value": post_training["budget_value"],
                    "consumption_policy": str(view_config.get("consumption_policy") or "unique_only"),
                    "packing_policy": str(
                        view_config.get("packing_policy")
                        or ("no_packing_v1" if post_training["objective_kind"] == "sft" else "none_v1")
                    ),
                    "sample_order_policy": str(
                        view_config.get("sample_order_policy") or "sequential_v1"
                    ),
                    "sample_offset": int(view_config.get("sample_offset") or 0),
                    "seed": int(view_config.get("seed") or 42),
                    "micro_batch_size": int(stages[0].get("micro_batch_size") or 1),
                    "gradient_accumulation_steps": int(
                        stages[0].get("gradient_accumulation_steps") or 1
                    ),
                },
                launcher_id=str(execution.get("launcher_id") or "builtin.local-process"),
                resource_request={**(execution.get("resource_request") or {}), "gpus": 0},
            ),
            session,
        )
        if view.get("status") == "ready":
            for stage in stages:
                stage["supervision_training_view_id"] = view["id"]
        return {
            "status": "ready" if view.get("status") == "ready" else "preparing_index",
            "ready": view.get("status") == "ready",
            "message": "后训练数据已准备完成。" if view.get("status") == "ready" else "正在构建后训练读取索引。",
            "selected_stage_indexes": indexes,
            "corpus": corpus,
            "data_index": view,
        }
    try:
        config = _managed_data_configuration(stages)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "training_data_configuration_mismatch", "message": str(exc)},
        ) from exc
    if config is None:
        return {
            "status": "legacy",
            "ready": True,
            "message": "历史计划继续使用已绑定的数据制品",
            "selected_stage_indexes": indexes,
        }
    if not config["mixture_version_id"]:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_data_mixture", "message": "训练阶段尚未选择数据混合"},
        )
    mixture_version = session.get(DataMixtureVersion, config["mixture_version_id"])
    mixture = (
        session.get(DataMixture, mixture_version.mixture_id)
        if mixture_version is not None
        else None
    )
    if (
        mixture is None
        or not project_has_asset_access(
            session,
            plan.project_id,
            "mixture_version",
            str(config["mixture_version_id"]),
        )
        or mixture_version is None
        or mixture_version.status != "ready"
        or not mixture_version.locked
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "mixture_not_ready", "message": "数据混合版本尚未解析并锁定"},
        )
    validation_mixture = _get_project_validation_mixture(
        session, plan.project_id, revision.validation_mixture_version_id
    )
    validation_holdout = _validation_holdout_configuration(
        mixture_version, validation_mixture
    )
    if not revision.tokenizer_version_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_tokenizer", "message": "训练计划尚未选择 Tokenizer"},
        )

    resume_checkpoint = _freeze_resume_checkpoint(
        session,
        plan=plan,
        revision=revision,
        selected_stage_indexes=indexes,
        checkpoint_id=payload.resume_checkpoint_id,
    )
    sequence_start = int((resume_checkpoint or {}).get("sequence_offset") or 0)
    try:
        config = _managed_data_configuration(
            stages,
            sequence_start=sequence_start,
            validation_holdout=validation_holdout,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "training_data_configuration_mismatch", "message": str(exc)},
        ) from exc
    assert config is not None

    execution = config.get("execution") or {}
    resource_request = {
        **(execution.get("resource_request") or {}),
        "gpus": 0,
    }
    corpus_versions: list[dict] = []
    for entry in mixture_version.entries or []:
        corpus_versions.append(
            _create_tokenized_corpus(
                plan.project_id,
                TokenizedCorpusCreate(
                    dataset_version_id=str(entry.get("dataset_version_id") or ""),
                    tokenizer_version_id=revision.tokenizer_version_id,
                    selection=(config["selection"] if is_identity_selection(entry.get("selection")) else {"mode": "mixture", "mixture_version_id": mixture_version.id}),
                    encoding=config["encoding"],
                    launcher_id=str(execution.get("launcher_id") or "builtin.local-process"),
                    resource_request=resource_request,
                ),
                session,
                trigger_context=trigger_context,
            )
        )
    pending_corpora = [item for item in corpus_versions if item.get("status") != "ready"]
    effective = {
        "mixture_version_id": mixture_version.id,
        "tokenizer_version_id": revision.tokenizer_version_id,
        "sequence_length": config["sequence_length"],
        "requested_tokens": config["requested_tokens"],
        "training_data_range": config["training_data_range"],
        "corpus": {"selection": config["selection"], "encoding": config["encoding"]},
        "view": config["view"],
        "execution": execution,
    }
    if pending_corpora:
        return {
            "status": "preparing_corpora",
            "ready": False,
            "message": "正在编码缺失的数据来源",
            "selected_stage_indexes": indexes,
            "effective_configuration": effective,
            "corpora": corpus_versions,
            "data_index": None,
        }

    reusable_views, _resolved_config = _find_managed_training_data_view(
        session,
        plan.project_id,
        stages,
        revision.tokenizer_version_id,
        sequence_start=sequence_start,
        validation_holdout=validation_holdout,
        required_prefix_view=(resume_checkpoint or {}).get("training_data_view"),
    )
    if reusable_views:
        reusable = reusable_views[0]
        reusable_version = session.get(
            TrainingDataViewVersion, str(reusable["version_id"])
        )
        assert reusable_version is not None
        return {
            "status": "ready",
            "ready": True,
            "message": "训练数据已准备完成（复用兼容容量）",
            "selected_stage_indexes": indexes,
            "effective_configuration": effective,
            "corpora": corpus_versions,
            "data_index": training_data_view(session, reusable_version),
            "capacity": {
                "requested_tokens": config["requested_tokens"],
                "required_sequence_end": config["training_data_range"]["sequence_end"],
                "available_tokens": int(reusable.get("token_count") or 0),
                "available_sequences": int(reusable.get("sequence_count") or 0),
                "consumption_policy": config["view"]["consumption_policy"],
                "warnings": reusable.get("warnings") or [],
            },
        }

    view = _create_training_data_view(
        plan.project_id,
        TrainingDataViewCreate(
            mixture_version_id=mixture_version.id,
            tokenizer_version_id=revision.tokenizer_version_id,
            corpus_version_ids=[str(item["id"]) for item in corpus_versions],
            config={
                "sequence_length": config["sequence_length"],
                "requested_tokens": config["requested_tokens"],
                **config["view"],
            },
            launcher_id=str(execution.get("launcher_id") or "builtin.local-process"),
            resource_request=resource_request,
        ),
        session,
        trigger_context=trigger_context,
        prefix_training_data_view=(resume_checkpoint or {}).get(
            "training_data_view"
        ),
    )
    ready = view.get("status") == "ready"
    return {
        "status": "ready" if ready else "preparing_index",
        "ready": ready,
        "message": "训练数据已准备完成" if ready else "正在建立训练读取索引",
        "selected_stage_indexes": indexes,
        "effective_configuration": effective,
        "corpora": corpus_versions,
        "data_index": view,
        "capacity": {
            "requested_tokens": config["requested_tokens"],
            "required_sequence_end": config["training_data_range"]["sequence_end"],
            "available_tokens": int(view.get("token_count") or 0),
            "available_sequences": int(view.get("sequence_count") or 0),
            "consumption_policy": config["view"]["consumption_policy"],
            "warnings": view.get("warnings") or [],
        },
    }


@router.post("/training-plans/{plan_id}/runs", status_code=201)
def start_training_run(
    plan_id: str, payload: TrainingStart, session: Session = Depends(get_session)
) -> dict:
    plan = get_or_404(session, TrainingPlan, plan_id, "训练计划")
    revision_id = payload.revision_id or plan.current_revision_id
    revision = get_or_404(session, PlanRevision, revision_id, "训练计划修订版")
    if (revision.effective_config or {}).get("training_instance_id") and not session.info.get("domain_transaction_depth"):
        raise HTTPException(status_code=409, detail={"code": "training_instance_required", "message": "请从所属训练实例创建执行记录"})
    if revision.plan_id != plan.id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_plan_revision", "message": "修订版不属于当前训练计划"},
        )
    selected_stages, selected_stage_indexes = _selected_stages(
        revision, payload.selected_stage_indexes
    )
    check = run_preflight(
        session,
        revision,
        selected_stages=selected_stages,
        selected_stage_indexes=selected_stage_indexes,
        resume_checkpoint_id=payload.resume_checkpoint_id,
    )
    # 预检只用于说明风险和自动配置，不再形成启动门禁或警告豁免流程。
    # 真正无法执行的输入会在冻结、物化或训练命令中返回普通错误。

    resume_checkpoint = _freeze_resume_checkpoint(
        session,
        plan=plan,
        revision=revision,
        selected_stage_indexes=selected_stage_indexes,
        checkpoint_id=payload.resume_checkpoint_id,
    )
    reference_checkpoint = _freeze_dpo_reference_checkpoint(
        session, plan.project_id, selected_stages, resume_checkpoint
    )
    code_version = (
        session.get(CodeVersion, revision.code_version_id) if revision.code_version_id else None
    )
    code_artifact = (
        session.get(Artifact, code_version.artifact_id)
        if code_version is not None and code_version.artifact_id
        else None
    )
    code_binding = (
        {
            "version_id": code_version.id,
            "version": code_version.version,
            "artifact_id": code_version.artifact_id,
            "content_hash": code_artifact.content_hash if code_artifact else None,
            "git_commit": code_version.git_commit,
            "entrypoints": code_version.entrypoints or {},
            "dependency_spec": code_version.dependency_spec,
            "training_contract": code_version.training_contract or {},
        }
        if code_version is not None
        else None
    )
    engine_config = (revision.effective_config or {}).get("engine_config") or {}
    if any("optimizations" in stage for stage in selected_stages) and (
        code_version is None or (code_version.training_contract or {}).get("optimization_profiles") != "forgelm.microlm-optimizations.v1"
    ):
        raise HTTPException(422, "所选引擎未声明算子配置协议，请导入支持该协议的代码版本")
    try:
        resource_batch_policy = freeze_resource_batch_policy(
            selected_stages,
            {"resource_batch_planning": engine_config["resource_batch_planning"]}
            if engine_config.get("resource_batch_planning") is not None
            else code_version.training_contract if code_version is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_resource_batch_policy", "message": str(exc)},
        ) from exc
    supervision_training_view_inputs = _find_supervision_training_view_inputs(
        session,
        plan.project_id,
        selected_stages,
        revision.tokenizer_version_id,
    )
    uses_supervision_data = bool(supervision_training_view_inputs)
    data_inputs = (
        []
        if uses_supervision_data
        else _freeze_data_inputs(session, plan.project_id, selected_stages)
    )
    tokenizer_binding = _freeze_tokenizer_binding(
        session, plan.project_id, revision.tokenizer_version_id
    )
    uses_training_data_view = any(
        stage.get("training_data_view_id") for stage in selected_stages
    )
    uses_managed_data = any(
        isinstance(stage.get("data_preparation"), dict) for stage in selected_stages
    )
    tokenized_required = not uses_supervision_data and (
        _requires_tokenized_data(revision)
        or uses_training_data_view
        or uses_managed_data
    )
    if uses_managed_data:
        validation_holdout = _validation_holdout_for_stages(
            session, revision, selected_stages
        )
        training_data_view_inputs, _managed_config = _find_managed_training_data_view(
            session,
            plan.project_id,
            selected_stages,
            revision.tokenizer_version_id,
            sequence_start=int((resume_checkpoint or {}).get("sequence_offset") or 0),
            validation_holdout=validation_holdout,
            required_prefix_view=(resume_checkpoint or {}).get(
                "training_data_view"
            ),
        )
    else:
        training_data_view_inputs = (
            _freeze_training_data_view_inputs(
                session,
                plan.project_id,
                selected_stages,
                revision.tokenizer_version_id,
            )
            if uses_training_data_view
            else []
        )
    tokenized_data_inputs = (
        _freeze_tokenized_data_inputs(
            session,
            plan.project_id,
            selected_stages,
            revision.tokenizer_version_id,
        )
        if tokenized_required and not uses_training_data_view and not uses_managed_data
        else []
    )
    effective_tokenized_inputs = training_data_view_inputs or tokenized_data_inputs
    training_data_range = None
    if tokenized_required and len(effective_tokenized_inputs) == 1:
        training_data_range = _training_data_range(
            selected_stages,
            int(effective_tokenized_inputs[0]["sequence_length"]),
            sequence_start=int((resume_checkpoint or {}).get("sequence_offset") or 0),
        )
        if uses_training_data_view or uses_managed_data:
            training_data_view_inputs = [
                {**binding, "training_data_range": training_data_range}
                for binding in training_data_view_inputs
            ]
        else:
            tokenized_data_inputs = [
                {**binding, "training_data_range": training_data_range}
                for binding in tokenized_data_inputs
            ]
        effective_tokenized_inputs = training_data_view_inputs or tokenized_data_inputs
    model_vocab_size, _model_vocab_error = _model_vocab_contract(session, plan, revision)
    model_specification_binding = _freeze_model_specification(
        session, plan.project_id, revision.model_specification_revision_id
    )
    checkpoint_compatibility, _checkpoint_compatibility_error = (
        _plan_checkpoint_compatibility(session, plan, revision)
    )
    system_validation_contract = {
        "schema_version": "forgelm.training-system-validations.v1",
        "items": check["system_validations"],
    }
    validation_binding = _freeze_validation_binding(
        session, plan.project_id, revision.validation_set_id
    )
    validation_mixture_binding = _freeze_validation_mixture_binding(
        session, plan.project_id, revision.validation_mixture_version_id
    )
    effective_validation_input = _effective_validation_data_input(
        validation_mixture_binding, training_data_view_inputs
    )
    routed_resource_request = _route_training_resource_request(
        session,
        plan.project_id,
        revision.resource_request or {},
        [
            *(
                [
                    str(artifact_id)
                    for binding in supervision_training_view_inputs
                    for artifact_id in [
                        binding["artifact_id"],
                        binding["supervision_corpus"]["artifact_id"],
                    ]
                ]
                if uses_supervision_data
                else []
            ),
            *(
                [
                    str(artifact_id)
                    for binding in effective_tokenized_inputs
                    for artifact_id in [
                        binding["artifact_id"],
                        *[
                            corpus["artifact_id"]
                            for corpus in binding.get("tokenized_corpora") or []
                        ],
                    ]
                ]
                + _training_source_artifact_ids([], effective_validation_input)
                if tokenized_required
                else _training_source_artifact_ids(
                    data_inputs, effective_validation_input
                )
            ),
            *(
                [str(resume_checkpoint["artifact_id"])]
                if resume_checkpoint is not None
                else []
            ),
            *(
                [str(reference_checkpoint["artifact_id"])]
                if reference_checkpoint is not None
                else []
            ),
        ],
        model_specification_binding,
    )
    if resource_batch_policy is not None:
        selector = dict(routed_resource_request.get("worker_selector") or {})
        selector["capabilities"] = list(
            dict.fromkeys(
                [
                    *(selector.get("capabilities") or []),
                    "builtin.training-resource-planner.v1",
                ]
            )
        )
        routed_resource_request["worker_selector"] = selector
    training_run_id = new_id()
    task = create_task(
        session,
        project_id=plan.project_id,
        kind="train",
        title=f"训练：{plan.name} r{revision.revision}",
        adapter_id=revision.engine_plugin_id,
        launcher_id=revision.launcher_plugin_id,
        payload={
            "training_run_id": training_run_id,
            "plan_id": plan.id,
            "revision_id": revision.id,
            "selected_stage_indexes": selected_stage_indexes,
            "stages": selected_stages,
            "learning_rate_schedules": check["learning_rate_schedules"],
            "resource_request": routed_resource_request,
            "effective_config": revision.effective_config,
            "input_model_ref": revision.input_model_ref or {},
            "system_validation_contract": system_validation_contract,
            **(
                {"resource_batch_policy": resource_batch_policy}
                if resource_batch_policy is not None
                else {}
            ),
            **(
                {"training_data_range": training_data_range}
                if training_data_range is not None
                else {}
            ),
            **(
                {
                    "supervision_training_view_inputs": supervision_training_view_inputs,
                    "objective_kind": supervision_training_view_inputs[0]["objective_kind"],
                }
                if uses_supervision_data
                else
                {
                    "training_data_view_inputs": training_data_view_inputs,
                    "tokenized_corpora": training_data_view_inputs[0]["tokenized_corpora"],
                }
                if uses_training_data_view or uses_managed_data
                else {"tokenized_data_inputs": tokenized_data_inputs}
                if tokenized_required
                else {"data_inputs": data_inputs}
            ),
            "command": (revision.effective_config or {}).get("engine_config", {}),
            **({"code": code_binding} if code_binding else {}),
            **({"tokenizer": tokenizer_binding} if tokenizer_binding else {}),
            **({"model_vocab_size": model_vocab_size} if model_vocab_size else {}),
            **(
                {"checkpoint_compatibility": checkpoint_compatibility}
                if checkpoint_compatibility is not None
                else {}
            ),
            **(
                {"model_specification": model_specification_binding}
                if model_specification_binding
                else {}
            ),
            **(
                {"resume_checkpoint": resume_checkpoint}
                if resume_checkpoint is not None
                else {}
            ),
            **(
                {"reference_checkpoint": reference_checkpoint}
                if reference_checkpoint is not None
                else {}
            ),
            **({"validation_set": validation_binding} if validation_binding else {}),
            **(
                {"validation_data_input": effective_validation_input}
                if effective_validation_input
                else {}
            ),
        },
    )
    revision.locked = True
    if revision.model_specification_revision_id:
        model_specification_revision = session.get(
            ModelSpecificationRevision, revision.model_specification_revision_id
        )
        if model_specification_revision is not None:
            model_specification_revision.locked = True
    plan.status = "running"
    run = TrainingRun(
        id=training_run_id,
        project_id=plan.project_id,
        plan_revision_id=revision.id,
        task_id=task.id,
        status="queued",
        current_stage=selected_stages[0].get("name") if selected_stages else None,
        config_snapshot={
            "revision": revision.to_dict(),
            "selected_stage_indexes": selected_stage_indexes,
            "selected_stages": selected_stages,
            "learning_rate_schedules": check["learning_rate_schedules"],
            "data_inputs": data_inputs,
            "system_validation_contract": system_validation_contract,
            **(
                {"resource_batch_policy": resource_batch_policy}
                if resource_batch_policy is not None
                else {}
            ),
            **(
                {"training_data_range": training_data_range}
                if training_data_range is not None
                else {}
            ),
            **(
                {"supervision_training_view_inputs": supervision_training_view_inputs}
                if uses_supervision_data
                else
                {
                    "training_data_view_inputs": training_data_view_inputs,
                    "tokenized_corpora": training_data_view_inputs[0]["tokenized_corpora"],
                }
                if uses_training_data_view or uses_managed_data
                else {"tokenized_data_inputs": tokenized_data_inputs}
                if tokenized_required
                else {}
            ),
            **({"tokenizer": tokenizer_binding} if tokenizer_binding else {}),
            **({"model_vocab_size": model_vocab_size} if model_vocab_size else {}),
            **(
                {"checkpoint_compatibility": checkpoint_compatibility}
                if checkpoint_compatibility is not None
                else {}
            ),
            **(
                {"model_specification": model_specification_binding}
                if model_specification_binding
                else {}
            ),
            **(
                {"resume_checkpoint": resume_checkpoint}
                if resume_checkpoint is not None
                else {}
            ),
            **(
                {"reference_checkpoint": reference_checkpoint}
                if reference_checkpoint is not None
                else {}
            ),
            **({"validation_set": validation_binding} if validation_binding else {}),
            **(
                {"validation_data_input": effective_validation_input}
                if effective_validation_input
                else {}
            ),
        },
    )
    session.add(run)
    session.flush()
    from ..services.training_history import ensure_history_instances

    ensure_history_instances(session, run.project_id)
    add_audit(
        session,
        project_id=plan.project_id,
        action="training_run.start",
        subject_type="training_run",
        subject_id=run.id,
        reason=None,
        payload={
            "selected_stage_indexes": selected_stage_indexes,
            "resume_checkpoint_id": payload.resume_checkpoint_id,
        },
    )
    commit_or_flush(session)
    return {**run.to_dict(), "task": task.to_dict(), "preflight": check}


@router.get("/projects/{project_id}/training-runs")
def list_training_runs(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    runs = session.scalars(
        select(TrainingRun)
        .where(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.created_at.desc())
    ).all()
    return [run.to_dict() for run in runs]


@router.get("/training-runs/{run_id}")
def get_training_run(run_id: str, session: Session = Depends(get_session)) -> dict:
    from ..services.training_metrics import training_run_metrics_view

    return training_run_metrics_view(
        session, get_or_404(session, TrainingRun, run_id, "训练运行").to_dict()
    )
