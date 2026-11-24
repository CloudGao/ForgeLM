from __future__ import annotations

import math
import re
import shlex
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from llm_platform_core.conversion import validate_conversion_operators
from llm_platform_core.training_data import sequence_order_strategy
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from .stage_gates import normalize_stages

WorkspaceUri = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2048)]


def validate_training_stage_names(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, stage in enumerate(stages):
        name = stage.get("name")
        if isinstance(name, str) and any("\x80" <= character <= "\x9f" for character in name):
            raise ValueError(
                f"stage {index + 1} name contains invalid control characters; "
                "verify that the JSON client uses UTF-8"
            )
    normalized = normalize_stages(stages)
    for index, stage in enumerate(normalized):
        if "engineering" in stage:
            raise ValueError("专项工程优化已撤回，请通过模型扩展点配置自定义实现")
        if "optimizations" in stage:
            profile = stage["optimizations"]
            choices = {"partial_rope": {"reference", "flash_attn"}, "linear_cross_entropy": {"fla", "torch_chunked"}}
            if not isinstance(profile, dict) or set(profile) - set(choices) or any(v not in choices[k] for k, v in profile.items()):
                raise ValueError("Unknown training optimization extension or provider")
            if stage.get("type") == "dpo":
                raise ValueError("DPO completion-logps does not support CE optimization profiles")
        objective_value = stage.get("objective")
        kind = str(
            (objective_value.get("kind") if isinstance(objective_value, dict) else None)
            or stage.get("type")
            or "pretrain"
        ).lower()
        if kind not in {"sft", "dpo"}:
            continue
        objective = stage.get("objective")
        if objective is None:
            # Historical type-only stages remain custom-command stages. Native
            # post-training is explicit through the versioned objective contract.
            continue
        if not isinstance(objective, dict) or str(objective.get("kind") or "").lower() != kind:
            raise ValueError(f"stage {index + 1} objective kind does not match its type")
        if int(objective.get("schema_version") or 0) != 1:
            raise ValueError(f"stage {index + 1} objective schema_version must be 1")
        if not str(stage.get("data_ref_id") or "").strip():
            raise ValueError(f"stage {index + 1} native {kind} requires a supervision mixture")
        if not str(stage.get("chat_template_version_id") or "").strip():
            raise ValueError(f"stage {index + 1} native {kind} requires a Chat Template version")
        budget = stage.get("budget")
        if not isinstance(budget, dict) or int(budget.get("value") or 0) <= 0:
            raise ValueError(f"stage {index + 1} budget must contain a positive value")
        supported = (
            {"supervised_tokens", "examples", "steps"}
            if kind == "sft"
            else {"preference_pairs", "steps"}
        )
        if str(budget.get("unit") or "") not in supported:
            raise ValueError(f"stage {index + 1} budget unit is invalid for {kind}")
        if kind == "sft":
            if str(objective.get("loss_scope") or "assistant_only") != "assistant_only":
                raise ValueError(f"stage {index + 1} SFT v1 only supports assistant-only loss")
            if str(objective.get("mask_policy") or "template_generation_spans_v1") != (
                "template_generation_spans_v1"
            ):
                raise ValueError(f"stage {index + 1} SFT v1 requires template generation spans")
        else:
            if str(objective.get("loss_type") or "sigmoid") != "sigmoid":
                raise ValueError(f"stage {index + 1} DPO v1 only supports sigmoid loss")
            beta = float(objective.get("beta") or 0)
            smoothing = float(objective.get("label_smoothing") or 0)
            if beta <= 0:
                raise ValueError(f"stage {index + 1} DPO beta must be positive")
            if smoothing < 0 or smoothing >= 0.5:
                raise ValueError(f"stage {index + 1} DPO label_smoothing must be in [0, 0.5)")
    return normalized


class ExecutionOptions(BaseModel):
    adapter_id: str | None = None
    launcher_id: str = "builtin.local-process"
    resource_request: dict[str, Any] = Field(default_factory=dict)
    command: dict[str, Any] = Field(default_factory=dict)


class TrainingPerformanceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    micro_batch_size: int = Field(ge=1, le=64)
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = Field(default=1, ge=1, le=32)
    optimizations: dict[str, str] | None = None

    @field_validator("optimizations")
    @classmethod
    def validate_optimizations(cls, value):
        if value is not None:
            validate_training_stage_names([{"name": "probe", "type": "pretrain", "optimizations": value}])
        return value


class TrainingPerformanceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_task_id: str = Field(min_length=1)
    engine_version_id: str = Field(min_length=1)
    candidates: list[TrainingPerformanceCandidate] = Field(default_factory=list, max_length=8)
    search_mode: Literal["auto", "manual"] | None = None
    max_candidates: int = Field(default=12, ge=2, le=24)
    warmup_steps: int = Field(default=2, ge=1, le=10)
    measurement_steps: int = Field(default=6, ge=3, le=30)
    memory_headroom_mb: int | None = Field(default=None, ge=1024, le=16384)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    target_model_name: str = Field(min_length=1, max_length=160)
    workspace_uri: WorkspaceUri = ""
    project_type: Literal["pretrain", "continue_training"] = "pretrain"
    target_parameters: str = "300M"
    primary_languages: list[str] = Field(default_factory=list)
    target_capabilities: list[str] = Field(default_factory=list)
    default_hardware: dict[str, Any] = Field(default_factory=dict)
    training_budget: dict[str, Any] = Field(default_factory=dict)
    acceptance_goal: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: str | None = None
    target_model_name: str | None = None
    workspace_uri: WorkspaceUri | None = None
    target_parameters: str | None = None
    primary_languages: list[str] | None = None
    target_capabilities: list[str] | None = None
    default_hardware: dict[str, Any] | None = None
    training_budget: dict[str, Any] | None = None
    acceptance_goal: dict[str, Any] | None = None
    lifecycle_stage: str | None = None


class ProjectDataBindingCreate(BaseModel):
    asset_type: Literal[
        "dataset_version",
        "mixture_version",
        "supervision_dataset_version",
        "supervision_mixture_version",
        "eval_dataset_version",
    ]
    asset_version_id: str = Field(min_length=1, max_length=36)
    purpose: Literal["training", "validation", "evaluation", "reference"] = "training"
    alias: str = Field(default="", max_length=180)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectDataBindingUpdate(BaseModel):
    purpose: Literal["training", "validation", "evaluation", "reference"] | None = None
    alias: str | None = Field(default=None, max_length=180)


class ProjectPipelineDataRef(BaseModel):
    asset_type: Literal[
        "dataset_version",
        "mixture_version",
        "supervision_dataset_version",
        "supervision_mixture_version",
    ]
    asset_version_id: str = Field(min_length=1, max_length=36)
    purpose: Literal["training", "validation", "reference"] = "training"


class ProjectPipelineCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    objective: Literal["pretraining", "post_training", "evaluation", "release"] = (
        "pretraining"
    )
    data_refs: list[ProjectPipelineDataRef] = Field(default_factory=list)
    tokenizer_version_id: str | None = None
    code_version_id: str | None = None
    plan_revision_id: str | None = None
    training_run_id: str | None = None
    model_version_id: str | None = None
    evaluation_run_id: str | None = None
    quantization_run_id: str | None = None
    release_id: str | None = None


PipelineNodeKind = Literal[
    "translation", "fact_translation", "knowledge",
    "selection",
    "preparation",
    "data",
    "transform",
    "clean",
    "mixture",
    "tokenizer",
    "training",
    "evaluation",
    "check",
    "custom",
    "branch",
    "join",
    "quantization",
    "release",
]


class ProjectPipelineDagNode(BaseModel):
    id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    type: PipelineNodeKind
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    label: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=1000)
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def freeze_generation_defaults(self):
        if self.type in {"translation", "fact_translation", "knowledge"} and self.config.get("native_operation"):
            from llm_platform_core.data_generation import (
                DataGenerationCreate,
                FactTranslationCreate,
            )

            parameters = self.config.get("parameters") or {}
            if isinstance(parameters, dict):
                parameters = dict(parameters)
                schema = FactTranslationCreate if self.type == "fact_translation" else DataGenerationCreate
                for key, field in schema.model_fields.items():
                    if not field.is_required():
                        parameters.setdefault(key, field.get_default(call_default_factory=True))
                if "top_p" not in (self.config.get("parameters") or {}) and self.type == "translation":
                    parameters["top_p"] = 0.6
                self.config = {**self.config, "parameters": parameters}
        return self

    @field_validator("position")
    @classmethod
    def validate_position(cls, position: dict[str, float]) -> dict[str, float]:
        if set(position) != {"x", "y"}:
            raise ValueError("position must contain exactly x and y")
        normalized = {key: float(value) for key, value in position.items()}
        if not all(math.isfinite(value) and abs(value) <= 1_000_000 for value in normalized.values()):
            raise ValueError("position coordinates must be finite and bounded")
        return normalized


class ProjectPipelineDagEdge(BaseModel):
    id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    source_handle: str | None = Field(default=None, max_length=120)
    target_handle: str | None = Field(default=None, max_length=120)
    label: str = Field(default="", max_length=180)


class ProjectPipelineDefinitionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    objective: Literal["pretraining", "post_training", "evaluation", "release"] = (
        "pretraining"
    )
    nodes: list[ProjectPipelineDagNode] = Field(default_factory=list, max_length=200)
    edges: list[ProjectPipelineDagEdge] = Field(default_factory=list, max_length=500)
    variables: dict[str, Any] = Field(default_factory=dict)
    viewport: dict[str, Any] = Field(default_factory=dict)


class ProjectPipelineDraftUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    objective: Literal["pretraining", "post_training", "evaluation", "release"] = (
        "pretraining"
    )
    nodes: list[ProjectPipelineDagNode] = Field(default_factory=list, max_length=200)
    edges: list[ProjectPipelineDagEdge] = Field(default_factory=list, max_length=500)
    variables: dict[str, Any] = Field(default_factory=dict)
    viewport: dict[str, Any] = Field(default_factory=dict)


class ProjectPipelineRunCreate(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


class ProjectPipelineInstanceCreate(BaseModel):
    definition_revision_id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)


class ProjectPipelineInstanceDraftUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    nodes: list[ProjectPipelineDagNode] = Field(default_factory=list, max_length=200)
    edges: list[ProjectPipelineDagEdge] = Field(default_factory=list, max_length=500)
    variables: dict[str, Any] = Field(default_factory=dict)
    viewport: dict[str, Any] = Field(default_factory=dict)


class CodeSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    source_type: Literal["git", "local", "archive"]
    source_uri: str = Field(min_length=1, max_length=4096)
    source_config: dict[str, Any] = Field(default_factory=dict)
    entrypoints: dict[str, str] = Field(default_factory=dict)
    config_schema: dict[str, Any] = Field(default_factory=dict)
    training_contract: dict[str, Any] = Field(default_factory=dict)
    dependency_spec: str = Field(default="", max_length=4096)
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class CodeVersionCreate(BaseModel):
    source_config: dict[str, Any] | None = None
    entrypoints: dict[str, str] | None = None
    config_schema: dict[str, Any] | None = None
    training_contract: dict[str, Any] | None = None
    dependency_spec: str | None = Field(default=None, max_length=4096)
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class DatasetCreate(ExecutionOptions):
    name: str
    source_type: str
    modality: Literal["text", "image", "audio", "video", "multimodal", "other"] = "text"
    source_uri: str | None = None
    license: str = "unknown"
    language: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    source_config: dict[str, Any] = Field(default_factory=dict)
    initial_version: dict[str, Any] | None = None
    start_import: bool = False
    code_version_id: str | None = None
    input_dataset_version_ids: list[str] = Field(default_factory=list, max_length=100)


class DatasetPreflightRequest(BaseModel):
    source_type: Literal["http", "https", "huggingface", "git", "local"]
    source_uri: str = Field(min_length=1)
    source_config: dict[str, Any] = Field(default_factory=dict)


class HuggingFaceRepositoryScanRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=512)
    revision: str = Field(default="main", min_length=1, max_length=240)
    hf_endpoint: str | None = Field(default=None, max_length=2048)
    config: str | None = Field(default=None, max_length=240)
    split: str | None = Field(default=None, max_length=240)
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class HuggingFaceDownloadPlanRequest(BaseModel):
    scan_task_id: str = Field(min_length=1, max_length=80)
    selected_paths: list[str] = Field(min_length=1, max_length=20_000)


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    source_uri: str | None = Field(default=None, min_length=1)
    source_config: dict[str, Any] | None = None
    license: str | None = Field(default=None, max_length=120)
    language: str | None = Field(default=None, max_length=80)
    tags: list[str] | None = None
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class DatasetReanalyze(BaseModel):
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class DatasetVersionCreate(BaseModel):
    tier: Literal["bronze", "silver", "gold"] = "bronze"
    raw_size_bytes: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    record_count: int = Field(default=0, ge=0)
    token_estimate: int = Field(default=0, ge=0)
    automatic_quality_score: float | None = Field(default=None, ge=0, le=100)
    human_score: float | None = Field(default=None, ge=0, le=100)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    train_ready: bool = False


class DatasetMaterializationCreate(BaseModel):
    target_worker_id: str = Field(min_length=1, max_length=36)
    selection_mode: Literal["auto", "manual"] = "auto"
    route_id: str | None = Field(default=None, min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_manual_route(self):
        if self.selection_mode == "manual" and not self.route_id:
            raise ValueError("manual materialization requires route_id")
        if self.selection_mode == "auto" and self.route_id:
            raise ValueError("route_id is only valid for manual materialization")
        return self


class DataConversionRecipeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    engine_adapter_id: Literal["builtin.data-conversion"] = "builtin.data-conversion"
    operators: list[dict[str, Any]] = Field(min_length=1, max_length=32)
    execution_config: dict[str, Any] = Field(default_factory=dict)

    _validate_operators = field_validator("operators")(validate_conversion_operators)


class DataConversionRecipeVersionCreate(BaseModel):
    engine_adapter_id: Literal["builtin.data-conversion"] = "builtin.data-conversion"
    operators: list[dict[str, Any]] = Field(min_length=1, max_length=32)
    execution_config: dict[str, Any] = Field(default_factory=dict)

    _validate_operators = field_validator("operators")(validate_conversion_operators)


class DataConversionRunCreate(BaseModel):
    recipe_version_id: str = Field(min_length=1, max_length=36)
    input_dataset_version_id: str = Field(min_length=1, max_length=36)
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


def validate_data_processing_operators(
    operators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not operators:
        raise ValueError("data processing recipe must contain at least one operator")
    if len(operators) > 200:
        raise ValueError("data processing recipe cannot contain more than 200 operators")
    for index, operator in enumerate(operators):
        if not isinstance(operator, dict) or len(operator) != 1:
            raise ValueError(f"operator {index + 1} must contain exactly one operator name")
        name, parameters = next(iter(operator.items()))
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"operator {index + 1} has an invalid name")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(f"operator {index + 1} parameters must be an object")
    return operators


class DataProcessingRecipeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    engine_adapter_id: Literal["builtin.data-juicer"] = "builtin.data-juicer"
    operators: list[dict[str, Any]] = Field(min_length=1, max_length=200)
    execution_config: dict[str, Any] = Field(default_factory=lambda: {"np": 1})

    _validate_operators = field_validator("operators")(validate_data_processing_operators)


class DataProcessingRecipeVersionCreate(BaseModel):
    engine_adapter_id: Literal["builtin.data-juicer"] = "builtin.data-juicer"
    operators: list[dict[str, Any]] = Field(min_length=1, max_length=200)
    execution_config: dict[str, Any] = Field(default_factory=lambda: {"np": 1})

    _validate_operators = field_validator("operators")(validate_data_processing_operators)


class DataProcessingRunCreate(BaseModel):
    recipe_version_id: str = Field(min_length=1, max_length=36)
    input_dataset_version_id: str = Field(min_length=1, max_length=36)
    input_mixture_version_id: str | None = Field(default=None, min_length=1, max_length=36)
    asset_scope: Literal["project", "global"] = "project"
    output_tier: Literal["silver", "gold"]
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 0})


class AnnotationCreate(BaseModel):
    sample_key: str
    decision: Literal["accept", "reject"] | None = None
    human_score: float | None = Field(default=None, ge=0, le=100)
    tags: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    note: str | None = None
    sample_snapshot: dict[str, Any] = Field(default_factory=dict)


class MixtureCreate(BaseModel):
    name: str
    asset_scope: Literal["project", "global"] = "project"
    description: str = ""
    entries: list[dict[str, Any]] = Field(default_factory=list)
    target_tokens: int = Field(default=0, ge=0)
    seed: int = 42


class MixtureVersionCreate(BaseModel):
    entries: list[dict[str, Any]]
    target_tokens: int = Field(default=0, ge=0)
    estimated_steps: int = Field(default=0, ge=0)
    distribution: dict[str, Any] = Field(default_factory=dict)
    gate_result: dict[str, Any] = Field(default_factory=dict)
    locked: bool = False
    seed: int = 42


class DatasetExplorerQuery(BaseModel):
    dataset_version_ids: list[str] = Field(min_length=1, max_length=50)
    include_paths: list[str] = Field(default_factory=list, max_length=100)
    exclude_paths: list[str] = Field(default_factory=list, max_length=100)
    languages: list[str] = Field(default_factory=list, max_length=50)
    min_length: int = Field(default=0, ge=0)
    max_length: int = Field(default=0, ge=0)
    sample_ratio: float = Field(default=1.0, ge=0, le=1)
    seed: int = 42
    offset: int = Field(default=0, ge=0, le=10_000_000)
    limit: int = Field(default=100, ge=1, le=500)

    @field_validator("dataset_version_ids")
    @classmethod
    def validate_dataset_version_ids(cls, version_ids: list[str]) -> list[str]:
        result = [str(version_id).strip() for version_id in version_ids]
        if any(not version_id or len(version_id) > 36 for version_id in result):
            raise ValueError("dataset version ID must be between 1 and 36 characters")
        if len(result) != len(set(result)):
            raise ValueError("dataset version IDs must be unique")
        return result

    @field_validator("include_paths", "exclude_paths")
    @classmethod
    def validate_path_patterns(cls, patterns: list[str]) -> list[str]:
        result = [str(pattern).strip() for pattern in patterns if str(pattern).strip()]
        if any(len(pattern) > 500 for pattern in result):
            raise ValueError("path pattern cannot exceed 500 characters")
        return list(dict.fromkeys(result))

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, languages: list[str]) -> list[str]:
        result = [str(language).strip().casefold() for language in languages if str(language).strip()]
        if any(len(language) > 80 for language in result):
            raise ValueError("language cannot exceed 80 characters")
        return list(dict.fromkeys(result))

    @model_validator(mode="after")
    def validate_length_range(self):
        if self.max_length and self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        return self

    def mixture_selection(self) -> dict[str, Any]:
        return {
            "include_paths": self.include_paths,
            "exclude_paths": self.exclude_paths,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "sample_ratio": self.sample_ratio,
            "seed": self.seed,
            "max_tokens": 0,
        }


class DatasetViewerQuery(BaseModel):
    sql: str = Field(default="SELECT * FROM dataset", max_length=20_000)
    search: str = Field(default="", max_length=2000)
    configuration: str = Field(default="default", max_length=120)
    split: str = Field(default="", max_length=120)
    offset: int = Field(default=0, ge=0, le=10_000_000)
    limit: int = Field(default=100, ge=1, le=500)
    timeout_seconds: int = Field(default=30, ge=1, le=60)

    @field_validator("sql")
    @classmethod
    def validate_sql(cls, value: str) -> str:
        return value.strip() or "SELECT * FROM dataset"

    @field_validator("configuration", "split")
    @classmethod
    def validate_selector(cls, value: str) -> str:
        value = value.strip()
        if any(character in value for character in ("/", "\\", "\x00")):
            raise ValueError("dataset viewer selector cannot contain path separators")
        return value


class LabelDefinitionInput(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=2000)
    color: str = Field(default="", max_length=32)


class LabelGroupInput(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=2000)
    selection_mode: Literal["single", "multiple"] = "single"
    required: bool = False
    labels: list[LabelDefinitionInput] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_unique_labels(self):
        codes = [label.code for label in self.labels]
        if len(codes) != len(set(codes)):
            raise ValueError("label codes must be unique within a group")
        return self


class LabelSchemeCreate(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=180)
    modality: Literal["text", "image", "audio", "video", "multimodal", "other"]
    description: str = Field(default="", max_length=4000)
    created_by: str = Field(default="local-user", min_length=1, max_length=180)
    groups: list[LabelGroupInput] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_groups(self):
        codes = [group.code for group in self.groups]
        if len(codes) != len(set(codes)):
            raise ValueError("group codes must be unique within a scheme version")
        return self


class LabelSchemeVersionCreate(BaseModel):
    description: str = Field(default="", max_length=4000)
    created_by: str = Field(default="local-user", min_length=1, max_length=180)
    groups: list[LabelGroupInput] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_groups(self):
        codes = [group.code for group in self.groups]
        if len(codes) != len(set(codes)):
            raise ValueError("group codes must be unique within a scheme version")
        return self


class LabelSchemeUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    description: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "archived"] | None = None


class LabelRecordReference(BaseModel):
    record_id: str = Field(min_length=1, max_length=128)
    record_hash: str = Field(min_length=1, max_length=128)


class DatasetLabelOperationCreate(BaseModel):
    source_query_task_id: str = Field(min_length=1, max_length=36)
    scope: Literal["selected", "page", "query"]
    action: Literal["apply", "remove"] = "apply"
    scheme_version_id: str = Field(min_length=1, max_length=36)
    label_ids: list[str] = Field(min_length=1, max_length=500)
    records: list[LabelRecordReference] = Field(default_factory=list, max_length=500)
    annotator: str = Field(default="local-user", min_length=1, max_length=180)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def validate_scope_records(self):
        if self.scope in {"selected", "page"} and not self.records:
            raise ValueError("selected and page scopes require record identities")
        if self.scope == "query" and self.records:
            raise ValueError("query scope freezes identities on the Worker")
        if len(self.label_ids) != len(set(self.label_ids)):
            raise ValueError("label IDs must be unique")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record IDs must be unique")
        return self


class DatasetExplorerMixtureCreate(DatasetExplorerQuery):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    target_tokens: int = Field(default=0, ge=0)
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weights(self):
        unknown = set(self.weights) - set(self.dataset_version_ids)
        if unknown:
            raise ValueError("weights contain dataset versions outside the query")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights.values()):
            raise ValueError("weights must be positive finite numbers")
        return self


class TokenizerCreate(ExecutionOptions):
    name: str
    algorithm: str = "BPE"
    vocab_size: int = Field(default=32000, ge=256)
    data_ref_type: Literal["dataset_version", "mixture_version"]
    data_ref_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    start: bool = True


class ChatTemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    template_body: str = Field(min_length=1)
    role_schema: dict[str, Any] = Field(
        default_factory=lambda: {
            "roles": ["system", "user", "assistant"],
            "system_position": "first_only",
        }
    )
    required_special_tokens: list[str] = Field(default_factory=list)
    assistant_mask_mode: Literal["generation_spans_v1", "unsupported"] = (
        "generation_spans_v1"
    )


class ChatTemplateVersionCreate(BaseModel):
    template_body: str = Field(min_length=1)
    role_schema: dict[str, Any] = Field(default_factory=dict)
    required_special_tokens: list[str] = Field(default_factory=list)
    assistant_mask_mode: Literal["generation_spans_v1", "unsupported"] = (
        "generation_spans_v1"
    )


class ChatTemplateValidate(BaseModel):
    tokenizer_version_id: str | None = None
    messages: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )


class SupervisionLookupCreate(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    dataset_version_id: str = Field(min_length=1, max_length=36)
    source_key: str = Field(min_length=1, max_length=240)
    lookup_key: str = Field(min_length=1, max_length=240)
    fields: dict[str, str] = Field(min_length=1, max_length=64)
    required: bool = True


class SupervisionDatasetCreate(ExecutionOptions):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    dataset_version_id: str = Field(min_length=1, max_length=36)
    kind: Literal["conversation", "preference"]
    mapping: dict[str, Any]
    lookups: list[SupervisionLookupCreate] = Field(default_factory=list, max_length=8)
    filters: dict[str, Any] = Field(default_factory=dict)
    start_scan: bool = True


class SupervisionDatasetVersionCreate(ExecutionOptions):
    dataset_version_id: str = Field(min_length=1, max_length=36)
    kind: Literal["conversation", "preference"]
    mapping: dict[str, Any]
    lookups: list[SupervisionLookupCreate] = Field(default_factory=list, max_length=8)
    filters: dict[str, Any] = Field(default_factory=dict)
    start_scan: bool = True


class SupervisionMixtureEntry(BaseModel):
    supervision_dataset_version_id: str = Field(min_length=1, max_length=36)
    weight: int = Field(default=1, ge=1, le=1_000_000)


class SupervisionMixtureCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    entries: list[SupervisionMixtureEntry] = Field(min_length=1)
    seed: int = 42


class SupervisionMixtureVersionCreate(BaseModel):
    entries: list[SupervisionMixtureEntry] = Field(min_length=1)
    seed: int = 42


class SupervisionMixtureUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    description: str | None = Field(default=None, max_length=4000)


class SupervisionCorpusCreate(ExecutionOptions):
    deduplicate_questions: bool = False
    mixture_version_id: str = Field(min_length=1, max_length=36)
    tokenizer_version_id: str = Field(min_length=1, max_length=36)
    chat_template_version_id: str = Field(min_length=1, max_length=36)
    objective_kind: Literal["sft", "dpo"]
    max_length: int = Field(default=2048, ge=16, le=65535)
    overflow_policy: Literal[
        "drop_overflow_v1", "conversation_tail_v1", "prompt_completion_v1", "sliding_window_v1"
    ] = "drop_overflow_v1"
    min_completion_tokens: int = Field(default=1, ge=1, le=4096)
    token_dtype: Literal["auto", "uint16", "uint32"] = "auto"


class SupervisionTrainingViewConfig(BaseModel):
    budget_unit: Literal["supervised_tokens", "examples", "preference_pairs", "steps"]
    budget_value: int = Field(gt=0)
    consumption_policy: Literal["unique_only", "repeat_epochs"] = "unique_only"
    packing_policy: Literal["no_packing_v1", "isolated_example_packing_v1", "none_v1"] = (
        "no_packing_v1"
    )
    sample_order_policy: Literal["sequential_v1", "global_shuffle_v1"] = "sequential_v1"
    reshuffle_each_epoch: bool = False
    sample_offset: int = Field(default=0, ge=0)
    seed: int = 42
    micro_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)


class SupervisionTrainingViewCreate(ExecutionOptions):
    corpus_version_id: str = Field(min_length=1, max_length=36)
    config: SupervisionTrainingViewConfig


class TokenizedDatasetCreate(ExecutionOptions):
    mixture_version_id: str
    tokenizer_version_id: str
    sequence_length: int = Field(default=2048, ge=16, le=65535)
    target_tokens: int = Field(gt=0)
    shard_sequences: int = Field(default=8192, ge=128, le=65536)
    packing_version: Literal["document-pack-v1"] = "document-pack-v1"


class TokenizedCorpusSelectionConfig(BaseModel):
    mode: Literal["all", "estimated_token_limit", "mixture"] = "all"
    estimated_token_limit: int | None = Field(default=None, gt=0)
    mixture_version_id: str | None = None

    @model_validator(mode="after")
    def validate_limit(self):
        if self.mode == "estimated_token_limit" and self.estimated_token_limit is None:
            raise ValueError("estimated_token_limit is required for estimated_token_limit mode")
        if self.mode == "mixture" and not self.mixture_version_id:
            raise ValueError("mixture_version_id is required for mixture selection")
        if self.mode == "mixture" and self.estimated_token_limit is not None:
            raise ValueError("mixture selection uses its frozen records, not an estimated token limit")
        if self.mode != "mixture" and self.mixture_version_id is not None:
            raise ValueError("mixture_version_id requires mixture selection")
        return self


class TokenizedCorpusEncodingConfig(BaseModel):
    eos_policy: Literal["append_required", "append_if_available", "none"] = (
        "append_if_available"
    )
    empty_document_policy: Literal["skip", "eos_only"] = "skip"
    token_dtype: Literal["auto", "uint16", "uint32"] = "auto"
    shard_token_limit: int = Field(default=268_435_456, ge=1_048_576)


class TokenizedCorpusCreate(ExecutionOptions):
    dataset_version_id: str
    tokenizer_version_id: str
    selection: TokenizedCorpusSelectionConfig = Field(
        default_factory=TokenizedCorpusSelectionConfig
    )
    encoding: TokenizedCorpusEncodingConfig = Field(
        default_factory=TokenizedCorpusEncodingConfig
    )


class TrainingValidationHoldoutSource(BaseModel):
    dataset_version_id: str = Field(min_length=1)
    weight_units: int = Field(ge=0)


class TrainingValidationHoldoutConfig(BaseModel):
    strategy: Literal["weighted_random_tokenized_document_holdout_v1"]
    target_tokens: int = Field(gt=0)
    seed: int = 42
    validation_mixture_version_id: str = Field(min_length=1)
    validation_manifest_hash: str = Field(min_length=64, max_length=64)
    sources: list[TrainingValidationHoldoutSource] = Field(min_length=1)


class TrainingDataViewConfig(BaseModel):
    sequence_length: int = Field(default=2048, ge=16, le=65535)
    requested_tokens: int = Field(gt=0)
    consumption_policy: Literal["unique_only", "repeat_epochs"] = "unique_only"
    tail_policy: Literal["drop", "pad"] = "drop"
    padding_token_id: int | None = Field(default=None, ge=0)
    packing_policy: Literal["document_stream_v1"] = "document_stream_v1"
    sampling_policy: Literal["weighted_deficit_v1"] = "weighted_deficit_v1"
    sequence_order_policy: str = Field(default="sequential_v1", min_length=1, max_length=120)
    shuffle_block_sequences: int | None = Field(default=None, ge=1, le=4096)
    repeat_sequence_count: int | None = Field(default=None, gt=0)
    validation_holdout: TrainingValidationHoldoutConfig | None = None
    seed: int = 42

    @model_validator(mode="after")
    def validate_padding(self):
        if self.tail_policy == "pad" and self.padding_token_id is None:
            raise ValueError("padding_token_id is required when tail_policy is pad")
        strategy = sequence_order_strategy(self.sequence_order_policy)
        if self.repeat_sequence_count is not None:
            if self.consumption_policy != "repeat_epochs" or self.sequence_order_policy != "block_shuffle_v1":
                raise ValueError("repeat_sequence_count requires repeat_epochs and block_shuffle_v1")
            block = self.shuffle_block_sequences or 1
            if self.repeat_sequence_count % block or self.requested_tokens % (self.repeat_sequence_count * self.sequence_length):
                raise ValueError("repeated epochs must be complete and block aligned")
        if strategy["requires_block_size"] and self.shuffle_block_sequences is None:
            raise ValueError(
                "shuffle_block_sequences is required for block_shuffle_v1"
            )
        return self


class TrainingDataViewCreate(ExecutionOptions):
    mixture_version_id: str
    tokenizer_version_id: str
    corpus_version_ids: list[str] = Field(min_length=1)
    config: TrainingDataViewConfig


class ModelSpecificationResolve(BaseModel):
    source_type: Literal["standard", "custom_code"] = "standard"
    architecture_family_id: str = Field(min_length=1, max_length=120)
    preset_id: str | None = Field(default=None, max_length=120)
    code_version_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    initialization: dict[str, Any] = Field(default_factory=lambda: {"type": "random"})


class ModelSpecificationCreate(ModelSpecificationResolve):
    name: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)


class ModelSpecificationRevisionCreate(ModelSpecificationResolve):
    pass


class TrainingPlanCreate(BaseModel):
    name: str
    description: str = ""
    stages: list[dict[str, Any]]
    input_model_ref: dict[str, Any] = Field(default_factory=dict)
    model_specification_revision_id: str | None = None
    tokenizer_version_id: str | None = None
    validation_set_id: str | None = None
    validation_mixture_version_id: str | None = None
    code_version_id: str | None = None
    engine_plugin_id: str = "builtin.command.training"
    launcher_plugin_id: str = "builtin.local-process"
    resource_request: dict[str, Any] = Field(default_factory=lambda: {"gpus": 1})
    engine_config: dict[str, Any] = Field(default_factory=dict)

    _validate_stage_names = field_validator("stages")(validate_training_stage_names)


class PlanRevisionCreate(BaseModel):
    stages: list[dict[str, Any]]
    input_model_ref: dict[str, Any] = Field(default_factory=dict)
    model_specification_revision_id: str | None = None
    tokenizer_version_id: str | None = None
    validation_set_id: str | None = None
    validation_mixture_version_id: str | None = None
    code_version_id: str | None = None
    engine_plugin_id: str = "builtin.command.training"
    launcher_plugin_id: str = "builtin.local-process"
    resource_request: dict[str, Any] = Field(default_factory=dict)
    engine_config: dict[str, Any] = Field(default_factory=dict)

    _validate_stage_names = field_validator("stages")(validate_training_stage_names)


class TrainingStart(BaseModel):
    revision_id: str | None = None
    selected_stage_indexes: list[int] | None = None
    resume_checkpoint_id: str | None = None


class EvalDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    source_type: str = "custom"
    task_type: str = "generation"
    license: str = "unknown"
    dimensions: list[str] = Field(default_factory=list)
    sample_count: int = Field(default=0, ge=0)
    contamination_tags: list[str] = Field(default_factory=list)
    source_config: dict[str, Any] = Field(default_factory=dict)
    schema_config: dict[str, Any] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=dict)


class EvalDatasetFromDataVersion(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    dataset_version_id: str = Field(min_length=1)
    task_type: str = "generation"
    license: str | None = Field(default=None, max_length=120)
    dimensions: list[str] = Field(default_factory=list)
    contamination_tags: list[str] = Field(default_factory=list)
    schema_config: dict[str, Any] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=dict)


class EvalDatasetVersionCreate(BaseModel):
    task_type: str = Field(min_length=1, max_length=80)
    license: str = Field(min_length=1, max_length=120)
    dimensions: list[str] = Field(default_factory=list)
    contamination_tags: list[str] = Field(default_factory=list)
    schema_config: dict[str, Any] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=dict)


class BenchmarkAssetCreate(BaseModel):
    overrides: dict[str, Any] = Field(default_factory=dict)


class EvaluationDatasetViewerQuery(DatasetViewerQuery):
    source_task_id: str | None = Field(default=None, max_length=64)


class EvalSuiteCreate(BaseModel):
    name: str
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    benchmark_ids: list[str] = Field(default_factory=list)
    scoring_config: dict[str, Any] = Field(default_factory=dict)
    release_gate: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuiteCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    benchmark_ids: list[str] = Field(min_length=1)
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=lambda: {"aggregator": "weighted_mean"})
    release_gate: dict[str, Any] = Field(default_factory=dict)


class EvaluationStart(ExecutionOptions):
    model_version_ids: list[str]
    primary_model_version_id: str | None = None
    suite_version_ids: list[str]
    inference_config: dict[str, Any] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=dict)
    code_version_id: str | None = None


class QuantizationStart(ExecutionOptions):
    model_version_id: str
    method: str = "Q4"
    format: str = "GGUF"
    calibration_data_ref: dict[str, Any] = Field(default_factory=dict)


class PluginInstallCreate(BaseModel):
    worker_id: str
    plugin_version_id: str


class PluginSuiteInstallCreate(BaseModel):
    worker_id: str
    component_keys: list[str] | None = Field(default=None, min_length=1, max_length=100)


class PluginUpdateCreate(BaseModel):
    project_id: str
    target_plugin_version_id: str | None = None


class ProjectPluginBindingUpsert(BaseModel):
    plugin_version_id: str
    enabled: bool = True
    default_config: dict[str, Any] = Field(default_factory=dict)


class QuantizationJobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = ""
    model_version_id: str
    plugin_version_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    calibration_data_ref: dict[str, Any] = Field(default_factory=dict)
    resource_request: dict[str, Any] = Field(default_factory=dict)
    launcher_id: str = "builtin.local-process"
    enabled: bool = True


class QuantizationJobPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    description: str | None = None
    model_version_id: str | None = None
    plugin_version_id: str | None = None
    config: dict[str, Any] | None = None
    calibration_data_ref: dict[str, Any] | None = None
    resource_request: dict[str, Any] | None = None
    launcher_id: str | None = None
    enabled: bool | None = None


class ReleaseCreate(BaseModel):
    name: str
    version: str
    model_version_id: str
    quantization_run_id: str | None = None
    target: str = "local"
    model_card: str = ""
    license_manifest: dict[str, Any] = Field(default_factory=dict)


class ReleaseJobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    description: str = ""
    source_kind: Literal["model_version", "quantization_run"]
    source_id: str
    exporter_plugin_version_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    gate_policy: dict[str, Any] = Field(default_factory=dict)
    model_card_template: str = ""
    license_policy: dict[str, Any] = Field(default_factory=dict)
    resource_request: dict[str, Any] = Field(default_factory=dict)
    launcher_id: str = "builtin.local-process"
    enabled: bool = True


class ReleaseJobPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    description: str | None = None
    source_kind: Literal["model_version", "quantization_run"] | None = None
    source_id: str | None = None
    exporter_plugin_version_id: str | None = None
    config: dict[str, Any] | None = None
    gate_policy: dict[str, Any] | None = None
    model_card_template: str | None = None
    license_policy: dict[str, Any] | None = None
    resource_request: dict[str, Any] | None = None
    launcher_id: str | None = None
    enabled: bool | None = None


class ReleaseBuildRunCreate(BaseModel):
    release_name: str = Field(min_length=1, max_length=180)
    release_version: str = Field(min_length=1, max_length=80)


class ReleaseDeliveryJobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    publisher_plugin_version_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    resource_request: dict[str, Any] = Field(default_factory=dict)
    launcher_id: str = "builtin.local-process"
    enabled: bool = True


class ReleaseDeliveryJobPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=180)
    publisher_plugin_version_id: str | None = None
    config: dict[str, Any] | None = None
    resource_request: dict[str, Any] | None = None
    launcher_id: str | None = None
    enabled: bool | None = None


class PlaygroundGenerate(ExecutionOptions):
    session_id: str | None = None
    mode: Literal["single"] = "single"
    model_version_ids: list[str] = Field(min_length=1, max_length=1)
    prompt: str = Field(min_length=1)
    system_prompt: str = "You are a helpful assistant."
    config: dict[str, Any] = Field(default_factory=dict)


class FeedbackCreate(BaseModel):
    inference_record_id: str
    kind: Literal["eval_failure", "sft_candidate", "ideal_answer", "dpo_pair"]
    preferred_output: str | None = None
    rejected_output: str | None = None
    labels: list[str] = Field(default_factory=list)


class TaskControl(BaseModel):
    action: Literal["pause", "resume", "safe_stop", "cancel", "retry"]
    reason: str | None = None


class PluginCreate(BaseModel):
    plugin_id: str
    version: str
    kind: str
    entrypoint: str
    protocol_version: int = 1
    config_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)


class ExternalArtifactReferenceCreate(BaseModel):
    kind: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    uri: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    size_bytes: int = Field(default=0, ge=0)
    media_type: str = "application/octet-stream"
    availability_state: Literal["available", "missing", "unknown", "offline"] = "unknown"
    worker_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelVersionRename(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip()


class ModelVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    version: int | None = Field(default=None, ge=1)
    kind: str = Field(default="base", min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    source_type: Literal[
        "huggingface", "checkpoint", "external_artifact", "openai_compatible"
    ]
    source_checkpoint_id: str | None = Field(default=None, max_length=36)
    parent_model_version_id: str | None = Field(default=None, max_length=36)
    tokenizer_version_id: str | None = Field(default=None, max_length=36)
    parameter_count: int = Field(default=0, ge=0)
    huggingface_id: str | None = Field(default=None, max_length=500)
    huggingface_revision: str | None = Field(
        default=None, max_length=40, pattern=r"^[a-fA-F0-9]{40}$"
    )
    artifact: ExternalArtifactReferenceCreate | None = None
    endpoint_url: str | None = Field(default=None, max_length=2048)
    endpoint_model: str | None = Field(default=None, max_length=500)
    endpoint_mode: Literal["completions", "chat_completions"] | None = None
    endpoint_tokenizer: str | None = Field(default=None, max_length=500)
    endpoint_auth_env_var: str | None = Field(
        default=None, max_length=160, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    endpoint_num_concurrent: int = Field(default=1, ge=1, le=128)
    endpoint_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    endpoint_verify_tls: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_candidate: bool = True

    @field_validator(
        "name",
        "kind",
        "source_checkpoint_id",
        "parent_model_version_id",
        "tokenizer_version_id",
        "huggingface_id",
        "huggingface_revision",
        "endpoint_url",
        "endpoint_model",
        "endpoint_tokenizer",
        "endpoint_auth_env_var",
        mode="before",
    )
    @classmethod
    def strip_model_version_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @model_validator(mode="after")
    def validate_model_source(self):
        if self.source_type == "huggingface":
            if not self.huggingface_id or not self.huggingface_revision:
                raise ValueError("Hugging Face model registration requires repository ID and revision")
            if self.source_checkpoint_id or self.artifact or self.endpoint_url:
                raise ValueError("Hugging Face model registration cannot also bind a Checkpoint or artifact")
        elif self.source_type == "checkpoint":
            if not self.source_checkpoint_id:
                raise ValueError("Checkpoint model registration requires source_checkpoint_id")
            if self.huggingface_id or self.huggingface_revision or self.artifact or self.endpoint_url:
                raise ValueError("Checkpoint model registration cannot also bind another source")
        elif self.source_type == "external_artifact":
            if self.artifact is None:
                raise ValueError("External model registration requires artifact metadata")
            if (
                self.source_checkpoint_id
                or self.huggingface_id
                or self.huggingface_revision
                or self.endpoint_url
            ):
                raise ValueError("External model registration cannot also bind another source")
        elif self.source_type == "openai_compatible":
            if not self.endpoint_url or not self.endpoint_model or not self.endpoint_mode:
                raise ValueError(
                    "OpenAI-compatible model registration requires URL, model, and mode"
                )
            if self.source_checkpoint_id or self.huggingface_id or self.artifact:
                raise ValueError(
                    "OpenAI-compatible model registration cannot also bind another source"
                )
            parsed = urlsplit(self.endpoint_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("OpenAI-compatible endpoint URL must use http or https")
            if parsed.username or parsed.password:
                raise ValueError("Endpoint credentials must use a Worker environment variable")
            endpoint_values = [
                self.endpoint_url,
                self.endpoint_model,
                self.endpoint_tokenizer or "",
            ]
            if any("," in value or any(ord(character) < 32 for character in value) for value in endpoint_values):
                raise ValueError("Endpoint fields cannot contain commas or control characters")
            if self.endpoint_mode == "completions" and not self.endpoint_tokenizer:
                raise ValueError(
                    "Completions endpoints require a tokenizer for Log-likelihood evaluation"
                )
        return self


class ModelLifecycleAction(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class HistoryArtifactSpec(ExternalArtifactReferenceCreate):
    ref: str = Field(min_length=1, max_length=240)


class HistoryDatasetVersionSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    version: int = Field(ge=1)
    tier: Literal["bronze", "silver", "gold"] = "bronze"
    state: str = "ready"
    raw_size_bytes: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    token_estimate: int = Field(default=0, ge=0)
    automatic_quality_score: float | None = Field(default=None, ge=0, le=100)
    human_score: float | None = Field(default=None, ge=0, le=100)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: str | None = None
    train_ready: bool = False


class HistoryDatasetSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(min_length=1, max_length=180)
    source_type: str = "local"
    source_uri: str | None = None
    license: str = "unknown"
    language: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    status: str = "ready"
    current_version_external_id: str | None = None
    versions: list[HistoryDatasetVersionSpec] = Field(default_factory=list)


class HistoryTokenizerSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(min_length=1, max_length=180)
    version: int = Field(default=1, ge=1)
    algorithm: str = "BPE"
    vocab_size: int = Field(default=32000, ge=256)
    data_ref_type: str = "external"
    data_ref_id: str = "external"
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str = "passed"
    locked: bool = True
    is_default: bool = False
    artifact_ref: str | None = None


class HistoryCheckpointSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    step: int = Field(ge=0)
    token_count: int = Field(default=0, ge=0)
    artifact_ref: str | None = None
    parent_model_external_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str = "available"


class HistoryModelSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(min_length=1, max_length=180)
    version: int = Field(default=1, ge=1)
    kind: str = "base"
    parent_model_external_id: str | None = None
    source_checkpoint_external_id: str | None = None
    artifact_ref: str | None = None
    tokenizer_external_id: str | None = None
    parameter_count: int = Field(default=0, ge=0)
    status: str = "candidate"
    is_best: bool = False
    is_candidate: bool = True
    is_official: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryTrainingRunSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(min_length=1, max_length=180)
    status: str = "succeeded"
    stages: list[dict[str, Any]] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    engine_plugin_id: str = "imported.history"
    launcher_plugin_id: str = "imported.history"
    resource_request: dict[str, Any] = Field(default_factory=dict)
    current_stage: str | None = None
    current_step: int = Field(default=0, ge=0)
    current_tokens: int = Field(default=0, ge=0)
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    checkpoints: list[HistoryCheckpointSpec] = Field(default_factory=list)
    models: list[HistoryModelSpec] = Field(default_factory=list)


class HistoryEvaluationSuiteSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(min_length=1, max_length=180)
    version: int = Field(default=1, ge=1)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    scoring_config: dict[str, Any] = Field(default_factory=dict)
    release_gate: dict[str, Any] = Field(default_factory=dict)


class HistoryEvaluationSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    name: str = Field(default="历史评测", min_length=1, max_length=180)
    status: str = "succeeded"
    model_external_ids: list[str] = Field(default_factory=list)
    suite_external_ids: list[str] = Field(default_factory=list)
    inference_config: dict[str, Any] = Field(default_factory=dict)
    scoring_config: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    sample_results: list[dict[str, Any]] = Field(default_factory=list)
    artifact_ref: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class HistoryGateSpec(BaseModel):
    external_id: str = Field(min_length=1, max_length=240)
    subject_entity_type: str = Field(min_length=1, max_length=60)
    subject_external_id: str = Field(min_length=1, max_length=240)
    gate_type: str = Field(min_length=1, max_length=80)
    rule_version: str = "1"
    status: str
    issues: list[dict[str, Any]] = Field(default_factory=list)
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    waiver: dict[str, Any] = Field(default_factory=dict)


class HistoryImportCreate(BaseModel):
    source_key: str = Field(min_length=1, max_length=180)
    artifacts: list[HistoryArtifactSpec] = Field(default_factory=list)
    datasets: list[HistoryDatasetSpec] = Field(default_factory=list)
    tokenizers: list[HistoryTokenizerSpec] = Field(default_factory=list)
    training_runs: list[HistoryTrainingRunSpec] = Field(default_factory=list)
    evaluation_suites: list[HistoryEvaluationSuiteSpec] = Field(default_factory=list)
    evaluations: list[HistoryEvaluationSpec] = Field(default_factory=list)
    gates: list[HistoryGateSpec] = Field(default_factory=list)


class WorkerRegister(BaseModel):
    name: str
    hostname: str
    labels: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


class WorkerEnroll(WorkerRegister):
    environment_id: str = Field(min_length=1, max_length=36)


class RemoteWorkerEnvironmentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    ssh_host: str = Field(
        default="", min_length=1, max_length=240, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,239}$"
    )
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_user: str = Field(
        default="root", min_length=1, max_length=120, pattern=r"^[a-z_][a-z0-9_-]{0,118}\$?$"
    )
    ssh_command: str | None = Field(default=None, max_length=600)
    credential_ref: str = Field(default="", pattern=r"^$|^[A-Za-z_][A-Za-z0-9_]{0,179}$")
    private_key_path: str | None = Field(default=None, max_length=1024)
    ssh_password: SecretStr | None = None
    transport: Literal["direct", "ssh_tunnel"] = "direct"
    runtime: Literal["base", "data", "inference"] = "base"
    service_mode: Literal["auto", "systemd", "process"] = "auto"
    expected_host_key_fingerprint: str | None = Field(default=None, max_length=160)
    control_plane_url: str = Field(default="http://127.0.0.1:8000", min_length=1, max_length=2048)
    workspace_root: str = Field(
        default="/var/lib/forgelm/workspaces", min_length=1, max_length=500
    )
    install_root: str = Field(default="/opt/forgelm-worker", min_length=1, max_length=500)
    profile: Literal["general", "data"] = "general"
    labels: dict[str, str] = Field(default_factory=dict)
    desired_release: str = Field(default="source", min_length=1, max_length=120)
    auto_provision: bool = True

    @field_validator("control_plane_url")
    @classmethod
    def validate_control_plane_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("control_plane_url must be an HTTP(S) origin without credentials")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def parse_ssh_command(cls, values):
        if not isinstance(values, dict) or not values.get("ssh_command"):
            return values
        values = dict(values)
        parts = shlex.split(values["ssh_command"])
        if not parts or parts.pop(0) != "ssh":
            raise ValueError("请输入 ssh [-p 端口] 用户@主机")
        port = 22
        if parts and parts[0] == "-p" and len(parts) >= 3:
            port = int(parts[1])
            parts = parts[2:]
        if len(parts) != 1 or parts[0].count("@") != 1:
            raise ValueError("SSH 指令只支持 -p 端口和用户@主机，不支持附加命令或选项")
        user, host = parts[0].split("@")
        values.update(ssh_host=host.strip("[]"), ssh_user=user, ssh_port=port)
        return values

    @field_validator("workspace_root", "install_root")
    @classmethod
    def validate_remote_absolute_path(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized.startswith("/")
            or normalized == "/"
            or any(part in {".", ".."} for part in normalized.split("/"))
            or any(character in normalized for character in ("\x00", "\n", "\r"))
            or re.fullmatch(r"(?:/[A-Za-z0-9._-]+)+", normalized.rstrip("/")) is None
            or len([part for part in normalized.split("/") if part]) < 2
        ):
            raise ValueError("remote paths must use safe absolute Linux path components")
        return normalized.rstrip("/")

    @model_validator(mode="after")
    def validate_remote_roots_are_disjoint(self):
        if not self.ssh_host:
            raise ValueError("需要 SSH 主机或登录指令")
        if sum(bool(value) for value in (self.credential_ref, self.private_key_path, self.ssh_password)) != 1:
            raise ValueError("请选择一种 SSH 凭据：私钥路径、私钥环境变量或首次登录密码")
        if self.profile == "data" and self.runtime == "inference":
            raise ValueError("数据 Worker 不能选择 GPU 运行配置")
        install_prefix = f"{self.install_root}/"
        workspace_prefix = f"{self.workspace_root}/"
        if (
            self.install_root == self.workspace_root
            or self.install_root.startswith(workspace_prefix)
            or self.workspace_root.startswith(install_prefix)
        ):
            raise ValueError("install_root and workspace_root must be disjoint")
        return self


class RemoteWorkerOperationCreate(BaseModel):
    wait_until_idle: bool = False
    kind: Literal["probe", "provision", "upgrade", "repair"]


class RemoteDownloadPolicyUpdate(BaseModel):
    download_policy: Literal["official", "domestic"]


class LocalEnvironmentInject(BaseModel):
    environment_id: str = Field(min_length=1, max_length=240)
    profile: Literal["base"] = "base"
    workspace_root: str = Field(default="/data/forgelm/workspaces", min_length=1, max_length=500)


class LocalEnvironmentStart(BaseModel):
    environment_id: str = Field(min_length=1, max_length=240)


class ControlPlaneRestart(BaseModel):
    reason: str = Field(default="用户从产品界面重启控制面", min_length=1, max_length=200)


class WorkerHeartbeat(BaseModel):
    resources: dict[str, Any] = Field(default_factory=dict)
    running_attempt_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] | None = None


class StorageScanCreate(BaseModel):
    project_ids: list[str] = Field(default_factory=list)


class WorkerStorageJobCreate(BaseModel):
    worker_id: str
    action: Literal["scan", "cleanup"] = "scan"
    scan_task_id: str | None = None
    item_ids: list[str] = Field(default_factory=list)


class StoragePolicyUpdate(BaseModel):
    cache_cleanup: Literal["auto", "retain"] = "auto"
    intermediate_retention: Literal["after_consumers", "retain"] = "after_consumers"
    grace_hours: int = Field(default=24, ge=0, le=8760)
    checkpoint_retention: Literal["retain"] = "retain"
    original_retention: Literal["retain"] = "retain"


class StorageCleanupPlanCreate(BaseModel):
    scan_id: str
    project_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] | None = None


class StorageCleanupExecute(BaseModel):
    confirmation: str = Field(min_length=1, max_length=120)


class AgentClaim(BaseModel):
    worker_id: str


class AgentStart(BaseModel):
    worker_id: str
    attempt_id: str
    runtime_handle: dict[str, Any] = Field(default_factory=dict)
    environment_snapshot: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(BaseModel):
    worker_id: str
    attempt_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    worker_sequence: int | None = None


class AgentComplete(BaseModel):
    worker_id: str
    attempt_id: str
    status: Literal["succeeded", "failed", "cancelled", "paused"]
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)
    exit_code: int = 0
    resource_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_metrics_artifact_id: str | None = None
    log_summary: dict[str, Any] = Field(default_factory=dict)
