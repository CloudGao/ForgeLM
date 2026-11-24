"""Typed DAG nodes backed by the platform's existing domain services.

No worker orchestration program is involved: domain records and their tasks are
created atomically by the control plane, then handled by the usual task protocol.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from llm_platform_core.data_generation import DataGenerationCreate, FactTranslationCreate
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .. import models as m
from .. import schemas as s
from .lineage import _database_asset_ref
from .transactions import domain_transaction


class TokenizerLock(BaseModel):
    tokenizer_version_id: str = Field(min_length=1)


class ModelSelection(BaseModel):
    evaluation_run_id: str = Field(min_length=1)
    comparison_evaluation_run_ids: list[str] = Field(default_factory=list)
    metric_path: str = "overall"
    direction: str = Field(default="max", pattern="^(min|max)$")


class OutputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    multiplier: int = Field(default=1, ge=1, le=100)
    round_up_to: int = Field(default=1, ge=1, le=16777216)


@dataclass(frozen=True)
class Operation:
    kind: str
    label: str
    schema: type[BaseModel]
    module: str
    function: str
    output_key: str
    asset_type: str | None
    result_path: str = "id"


OPERATIONS = {
    "data.generation_collection": Operation("mixture", "汇集生成批次", s.MixtureCreate,
        "data_generation", "create_generation_collection", "mixture_version_id", "mixture_version", "current_version_id"),
    "data.fact_translation": Operation("fact_translation", "三元组翻译", FactTranslationCreate, "data_generation", "create_fact_translation", "dataset_version_id", "dataset_version", "output_dataset_version_id"),
    "data.translation": Operation("translation", "数据翻译", DataGenerationCreate, "data_generation", "create_data_translation", "dataset_version_id", "dataset_version", "output_dataset_version_id"),
    "data.knowledge": Operation("knowledge", "知识数据生成", DataGenerationCreate, "data_generation", "create_knowledge_generation", "dataset_version_id", "dataset_version", "output_dataset_version_id"),
    "training.performance": Operation(
        "preparation", "训练性能比较（保留完整训练状态）", s.TrainingPerformanceCreate,
        "training", "create_training_performance", "performance_task_id", None,
    ),
    "storage.maintenance": Operation(
        "preparation", "Worker 存储盘点与回收", s.WorkerStorageJobCreate,
        "storage", "create_worker_storage_job", "maintenance_task_id", None,
    ),
    "data.recipe": Operation(
        "preparation",
        "创建数据清洗配方",
        s.DataProcessingRecipeCreate,
        "data_processing",
        "create_data_processing_recipe",
        "recipe_version_id",
        None,
        "current_version_id",
    ),
    "training.view": Operation(
        "preparation",
        "预训练读取索引",
        s.TrainingDataViewCreate,
        "training_data_views",
        "create_training_data_view",
        "training_view_version_id",
        "training_data_view_version",
    ),
    "model.select": Operation(
        "selection",
        "按验证评测选择模型",
        ModelSelection,
        "",
        "",
        "model_version_id",
        "model_version",
    ),
    "data.processing": Operation(
        "clean",
        "Data-Juicer 清洗",
        s.DataProcessingRunCreate,
        "data_processing",
        "create_data_processing_run",
        "dataset_version_id",
        "dataset_version",
        "output_dataset_version_id",
    ),
    "data.conversion": Operation(
        "transform",
        "数据格式转换",
        s.DataConversionRunCreate,
        "data_conversion",
        "create_data_conversion_run",
        "dataset_version_id",
        "dataset_version",
        "output_dataset_version_id",
    ),
    "data.mixture": Operation(
        "mixture",
        "预训练数据混合",
        s.MixtureCreate,
        "data",
        "create_mixture",
        "mixture_version_id",
        "mixture_version",
        "current_version_id",
    ),
    "tokenizer.train": Operation(
        "tokenizer",
        "训练 Tokenizer",
        s.TokenizerCreate,
        "tokenizers",
        "create_tokenizer",
        "tokenizer_version_id",
        "tokenizer_version",
    ),
    "tokenizer.lock": Operation(
        "preparation",
        "锁定 Tokenizer",
        TokenizerLock,
        "tokenizers",
        "lock_tokenizer",
        "tokenizer_version_id",
        "tokenizer_version",
    ),
    "corpus.encode": Operation(
        "preparation",
        "编码预训练 Corpus",
        s.TokenizedCorpusCreate,
        "tokenized_corpora",
        "create_tokenized_corpus",
        "corpus_version_id",
        "tokenized_corpus_version",
    ),
    "template.create": Operation(
        "preparation",
        "创建对话模板",
        s.ChatTemplateCreate,
        "post_training",
        "create_chat_template",
        "chat_template_version_id",
        "chat_template_version",
        "current_version_id",
    ),
    "supervision.map": Operation(
        "preparation",
        "监督数据映射与扫描",
        s.SupervisionDatasetCreate,
        "post_training",
        "create_supervision_dataset",
        "supervision_dataset_version_id",
        "supervision_dataset_version",
    ),
    "supervision.mixture": Operation(
        "mixture",
        "监督数据混合",
        s.SupervisionMixtureCreate,
        "post_training",
        "create_supervision_mixture",
        "mixture_version_id",
        "supervision_mixture_version",
    ),
    "supervision.encode": Operation(
        "preparation",
        "编码监督 Corpus",
        s.SupervisionCorpusCreate,
        "post_training",
        "create_supervision_corpus",
        "corpus_version_id",
        "supervision_corpus_version",
    ),
    "supervision.view": Operation(
        "preparation",
        "监督训练读取索引",
        s.SupervisionTrainingViewCreate,
        "post_training",
        "create_supervision_training_view",
        "training_view_version_id",
        "supervision_training_view_version",
    ),
    "evaluation.dataset": Operation(
        "preparation",
        "从数据版本构建评测集",
        s.EvalDatasetFromDataVersion,
        "evaluations",
        "create_eval_dataset_from_data_version",
        "eval_dataset_version_id",
        "eval_dataset_version",
    ),
    "evaluation.suite": Operation(
        "preparation",
        "创建评测套件",
        s.EvalSuiteCreate,
        "evaluations",
        "create_eval_suite",
        "suite_version_id",
        "eval_suite_version",
    ),
    "evaluation.run": Operation(
        "evaluation",
        "标准模型评测",
        s.EvaluationStart,
        "evaluations",
        "start_evaluation",
        "evaluation_run_id",
        None,
    ),
}


def is_domain_node(node: dict) -> bool:
    return bool((node.get("config") or {}).get("native_operation"))


def operation_catalog(kind: str) -> list[dict]:
    return [
        {
            "id": key,
            "label": op.label,
            "input_schema": op.schema.model_json_schema(),
            "output_key": op.output_key,
        }
        for key, op in OPERATIONS.items()
        if op.kind == kind
    ]


def get_path(value: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError(f"输出路径不存在：{path}")
    if value is None:
        raise ValueError(f"输出路径为空：{path}")
    return deepcopy(value)


def set_path(value: dict, path: str, replacement: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current.setdefault(part, {})
        else:
            raise ValueError(f"参数绑定路径不存在：{path}")
    if isinstance(current, dict):
        current[parts[-1]] = replacement
    elif isinstance(current, list) and parts[-1].isdigit() and int(parts[-1]) < len(current):
        current[int(parts[-1])] = replacement
    else:
        raise ValueError(f"参数绑定路径不存在：{path}")


def resolve_parameters(config: dict, context: dict, *, key="parameters") -> dict:
    result = deepcopy(config.get(key) or {})
    for path, raw in (config.get("input_bindings") or {}).items():
        binding = OutputBinding.model_validate(raw)
        upstream = (context.get("upstream") or {}).get(binding.node_id)
        if upstream is None:
            raise ValueError(f"上游节点未完成或没有输入连线：{binding.node_id}")
        value = get_path(upstream, binding.output_path)
        if binding.multiplier != 1 or binding.round_up_to != 1:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("epoch 倍数只能应用于数值输出")
            value *= binding.multiplier
            value = math.ceil(value / binding.round_up_to) * binding.round_up_to
        set_path(result, path, value)
    return result


def validate_node(node: dict, edges: list) -> list[dict]:
    config = node.get("config") or {}
    errors = []
    op = OPERATIONS.get(config.get("native_operation"))
    bindings = config.get("input_bindings") or {}
    try:
        if config.get("output_retention", "auto") not in {"auto", "after_consumers", "retain"}:
            raise ValueError("不支持的输出保留策略")
        if not isinstance(bindings, dict):
            raise ValueError("输入绑定必须是目标参数路径到上游输出的映射")
        incoming = {e["source"] for e in edges if e["target"] == node["id"]}
        for path, raw in bindings.items():
            if not path or any(not part for part in path.split(".")):
                raise ValueError("参数绑定路径不能为空")
            binding = OutputBinding.model_validate(raw)
            if binding.node_id not in incoming:
                raise ValueError(f"输入绑定缺少连线：{binding.node_id} → {node['id']}")
        if config.get("native_operation"):
            if op is None or op.kind != node["type"]:
                raise ValueError("内置操作与节点类型不匹配")
            parameters = config.get("parameters") or {}
            if parameters.get("start") is False or parameters.get("start_scan") is False:
                raise ValueError("执行节点必须启动实际构建任务")
            if (
                config.get("command")
                or parameters.get("command")
                or parameters.get("code_version_id")
            ):
                raise ValueError("内置领域节点只接受结构化参数；自定义命令请使用原有命令节点")
            adapter = parameters.get("adapter_id")
            if adapter and (
                not adapter.startswith("builtin.") or adapter.startswith("builtin.command.")
            ):
                raise ValueError("内置领域节点必须选择对应的内置适配器")
            try:
                op.schema.model_validate(parameters)
            except ValidationError as exc:
                for error in exc.errors():
                    location = ".".join(str(p) for p in error["loc"])
                    if not any(
                        location == p
                        or location.startswith(p + ".")
                        or p.startswith(location + ".")
                        for p in bindings
                    ):
                        raise ValueError(f"{location}: {error['msg']}") from exc
        elif node["type"] in {"preparation", "translation", "fact_translation", "knowledge"}:
            raise ValueError("请选择内置准备操作")
    except (ValueError, TypeError) as exc:
        errors.append(
            {
                "severity": "error",
                "code": "invalid_native_node",
                "node_id": node["id"],
                "message": str(exc),
            }
        )
    return errors


def advance_domain_node(session, run, node, row, context) -> bool:
    config = node["config"]
    op = OPERATIONS[config["native_operation"]]
    parameters = resolve_parameters(config, context)
    resolved_errors = validate_node(
        {**node, "config": {**config, "parameters": parameters, "input_bindings": {}}}, []
    )
    if resolved_errors:
        raise ValueError(resolved_errors[0]["message"])
    request = op.schema.model_validate(parameters)
    if isinstance(request, ModelSelection):
        evaluation = session.get(m.EvaluationRun, request.evaluation_run_id)
        if (
            evaluation is None
            or evaluation.project_id != run.project_id
            or evaluation.status != "succeeded"
        ):
            raise ValueError("选优必须引用当前项目已成功的评测运行")
        scores = []
        reports = dict((evaluation.summary or {}).get("models", {}))
        for identity in request.comparison_evaluation_run_ids:
            comparison = session.get(m.EvaluationRun, identity)
            if (
                comparison is None
                or comparison.project_id != run.project_id
                or comparison.status != "succeeded"
                or comparison.benchmark_signature != evaluation.benchmark_signature
                or comparison.compatibility_hash != evaluation.compatibility_hash
            ):
                raise ValueError("跨阶段选优只接受同协议、同评测配置的已完成运行")
            reports.update((comparison.summary or {}).get("models", {}))
        for model_id, report in reports.items():
            model = session.get(m.ModelVersion, model_id)
            if model is None or model.project_id != run.project_id:
                raise ValueError("评测候选模型不可用")
            value = get_path(report, request.metric_path)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("选优指标必须是有限数值")
            scores.append((value, model_id))
        if not scores:
            raise ValueError("评测没有可比较的候选模型")
        scores.sort(
            key=lambda item: ((-item[0] if request.direction == "max" else item[0]), item[1])
        )
        score, model_id = scores[0]
        model = session.get(m.ModelVersion, model_id)
        if not model.source_checkpoint_id:
            raise ValueError("所选候选没有可引用的 Checkpoint")
        row.input_snapshot = {
            **context,
            "domain_parameters": parameters,
            "consumed_assets": [
                _database_asset_ref(session, "model_version", model_id, role="selected_model")
            ],
        }
        row.output_snapshot = {
            "model_version_id": model_id,
            "checkpoint_id": model.source_checkpoint_id,
            "evaluation_run_id": evaluation.id,
            "score": score,
            "metric_path": request.metric_path,
            "direction": request.direction,
            "candidates": [{"score": v, "model_version_id": k} for v, k in scores],
        }
        row.status = "succeeded"
        row.ended_at = m.utcnow()
        return True
    handler = getattr(import_module(f"..routers.{op.module}", __package__), op.function)
    with domain_transaction(session):
        if isinstance(request, TokenizerLock):
            tokenizer = session.get(m.TokenizerVersion, request.tokenizer_version_id)
            if tokenizer is None or tokenizer.project_id != run.project_id:
                raise ValueError("Tokenizer 不属于当前项目")
            response = handler(request.tokenizer_version_id, session)
        else:
            response = handler(run.project_id, request, session)
        if config["native_operation"] == "data.generation_collection" and response.get("empty"):
            row.input_snapshot = {**context, "domain_parameters": parameters, "consumed_assets": [
                _database_asset_ref(session, "dataset_version", entry["dataset_version_id"], role="empty_batch")
                for entry in request.entries
            ]}
            row.output_snapshot = {**response, "native_operation": config["native_operation"], "produced_assets": []}
            row.status, row.ended_at = "succeeded", m.utcnow()
            return True
        output_id = get_path(response, op.result_path)
        row.output_snapshot = {
            op.output_key: output_id,
            "native_operation": config["native_operation"],
        }
        row.input_snapshot = {**context, "domain_parameters": parameters}
        if config["native_operation"] == "data.generation_collection":
            row.input_snapshot["consumed_assets"] = [
                _database_asset_ref(session, "dataset_version", entry["dataset_version_id"], role="batch")
                for entry in request.entries
            ]
            row.output_snapshot["empty_dataset_version_ids"] = response.get("empty_dataset_version_ids", [])
        if isinstance(request, DataGenerationCreate):
            consumed = [
                _database_asset_ref(session, "dataset_version", request.input_dataset_version_id, role="source"),
                _database_asset_ref(session, "model_weight_version", request.model_weight_version_id, role="generator"),
            ]
            row.input_snapshot = {**row.input_snapshot, "consumed_assets": consumed}
            if request.checkpoint_source_task_id:
                row.input_snapshot["generation_continuation"] = {
                    "source_task_id": request.checkpoint_source_task_id,
                    "source_task_attempt_id": request.checkpoint_source_attempt_id,
                }
        if isinstance(request, s.DataProcessingRunCreate):
            consumed = [_database_asset_ref(session, "dataset_version", request.input_dataset_version_id, role="source")]
            if request.input_mixture_version_id:
                consumed.append(_database_asset_ref(session, "mixture_version", request.input_mixture_version_id, role="selection"))
            row.input_snapshot = {**row.input_snapshot, "consumed_assets": consumed}
        if op.asset_type:
            row.output_snapshot["produced_assets"] = [
                _database_asset_ref(session, op.asset_type, output_id, role="output")
            ]
        # Some domain APIs return the logical record, with the task on its version.
        task_id = response.get("task_id") or (response.get("current_version") or {}).get("task_id")
        if config["native_operation"] in {"data.mixture", "data.generation_collection"}:
            task_id = session.get(m.DataMixtureVersion, output_id).task_id
        task = session.get(m.Task, task_id) if task_id else None
        if task:
            # Corpus identity reuse is a consumption of an existing result, not a new production.
            if task.observed_state == "succeeded":
                row.input_snapshot["consumed_assets"] = row.output_snapshot.pop(
                    "produced_assets", []
                )
                row.output_snapshot["reused_task_id"] = task.id
                row.output_snapshot = {**deepcopy(task.result or {}), **row.output_snapshot}
                row.status = "succeeded"
                row.ended_at = m.utcnow()
            else:
                if (
                    task.payload.get("pipeline_node_run_id")
                    and task.payload["pipeline_node_run_id"] != row.id
                ):
                    raise ValueError("该领域任务正在由另一个节点执行，请等待完成后复用")
                if task.observed_state in {"failed", "cancelled", "lost"}:
                    raise ValueError("已有领域任务未成功，请先恢复该任务")
                task.payload = {
                    **task.payload,
                    "output_retention": config.get("output_retention", "auto"),
                    "pipeline_instance_id": run.instance_id,
                    "pipeline_run_id": run.id,
                    "pipeline_node_run_id": row.id,
                    "pipeline_node_id": row.node_id,
                }
                row.task_id = task.id
                row.status = "queued" if task.observed_state == "queued" else "running"
        else:
            row.status = "succeeded"
            row.ended_at = m.utcnow()
    return True
