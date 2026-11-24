"""Public catalog and state builder for non-configurable training validations."""

from __future__ import annotations

from typing import Any

TRAINING_SYSTEM_VALIDATIONS = (
    {
        "id": "training_plan_valid",
        "label": "训练计划结构有效",
        "description": "阶段选择和有效配置必须形成可执行的冻结计划。",
        "phase": "preflight",
    },
    {
        "id": "data_mixture_ready",
        "label": "训练数据引用就绪",
        "description": "数据混合身份、锁定状态和 Manifest 必须有效。",
        "phase": "preflight",
    },
    {
        "id": "tokenizer_locked",
        "label": "Tokenizer 已冻结",
        "description": "训练必须绑定不可变且可物化的 Tokenizer 版本。",
        "phase": "preflight",
    },
    {
        "id": "tokenizer_model_compatible",
        "label": "Tokenizer 与模型兼容",
        "description": "词表大小和模型声明必须一致。",
        "phase": "preflight",
    },
    {
        "id": "validation_input_ready",
        "label": "验证输入就绪",
        "description": "训练声明的验证集或验证数据流必须可用。",
        "phase": "preflight",
    },
    {
        "id": "resume_checkpoint_compatible",
        "label": "接续 Checkpoint 兼容",
        "description": "权重结构、制品身份和数据游标必须与本次计划兼容。",
        "phase": "preflight",
    },
    {
        "id": "tokenized_dataset_integrity",
        "label": "预分词数据完整",
        "description": "预分词制品必须匹配 Tokenizer、序列长度、容量和批次边界。",
        "phase": "preflight",
    },
    {
        "id": "training_data_view_integrity",
        "label": "训练数据视图完整",
        "description": "训练读取视图及其不可变来源必须就绪。",
        "phase": "preflight",
    },
    {
        "id": "training_data_view_selection",
        "label": "训练数据选择一致",
        "description": "一次运行中的数据准备和读取语义必须一致。",
        "phase": "preflight",
    },
    {
        "id": "training_data_view_capacity",
        "label": "唯一数据容量充足",
        "description": "数据容量必须覆盖接续游标之后的冻结训练范围。",
        "phase": "preflight",
    },
    {
        "id": "code_version_ready",
        "label": "训练代码版本就绪",
        "description": "不可变代码版本、入口和源代码制品必须可用。",
        "phase": "preflight",
    },
    {
        "id": "engine_configuration_valid",
        "label": "训练引擎配置有效",
        "description": "引擎、命令和模板变量必须完整且可解析。",
        "phase": "preflight",
    },
    {
        "id": "worker_capability_available",
        "label": "Worker 路由可满足",
        "description": "资源、能力和制品位置约束必须能路由到同一 Worker。",
        "phase": "preflight",
    },
    {
        "id": "task_input_materialization",
        "label": "冻结输入物化完整",
        "description": "Worker 将复核代码、数据、Tokenizer 和 Checkpoint 的内容身份。",
        "phase": "worker",
    },
    {
        "id": "runtime_training_invariants",
        "label": "运行时训练约束有效",
        "description": "设备能力、内核、批次几何和数据读取契约将在 Worker 上验证。",
        "phase": "runtime",
    },
    {
        "id": "finite_training_state",
        "label": "训练数值保持有限",
        "description": "训练过程中 loss、梯度和关键指标不得出现非有限值。",
        "phase": "runtime",
    },
    {
        "id": "output_artifact_integrity",
        "label": "输出制品完整",
        "description": "Checkpoint、模型和结果制品将在登记时校验。",
        "phase": "output",
    },
)


ISSUE_VALIDATION_IDS = {
    "no_stages": "training_plan_valid",
    "model_checkpoint_contract_invalid": "training_plan_valid",
    "invalid_data_mixture": "data_mixture_ready",
    "data_mixture_not_ready": "data_mixture_ready",
    "missing_tokenizer": "tokenizer_locked",
    "unlocked_tokenizer": "tokenizer_locked",
    "model_vocab_contract_invalid": "tokenizer_model_compatible",
    "tokenizer_model_vocab_mismatch": "tokenizer_model_compatible",
    "invalid_model_specification_revision": "tokenizer_model_compatible",
    "model_specification_not_ready": "tokenizer_model_compatible",
    "training_validation_set_unavailable": "validation_input_ready",
    "validation_mixture_unavailable": "validation_input_ready",
    "missing_training_validation_set": "validation_input_ready",
    "invalid_resume_checkpoint": "resume_checkpoint_compatible",
    "resume_checkpoint_required": "resume_checkpoint_compatible",
    "resume_checkpoint_unavailable": "resume_checkpoint_compatible",
    "resume_checkpoint_artifact_unavailable": "resume_checkpoint_compatible",
    "checkpoint_sequence_offset_unknown": "resume_checkpoint_compatible",
    "invalid_checkpoint_sequence_offset": "resume_checkpoint_compatible",
    "invalid_checkpoint_sample_offset": "resume_checkpoint_compatible",
    "checkpoint_sample_offset_unknown": "resume_checkpoint_compatible",
    "resume_checkpoint_model_incompatible": "resume_checkpoint_compatible",
    "resume_checkpoint_model_compatibility_unverified": "resume_checkpoint_compatible",
    "tokenized_dataset_unavailable": "tokenized_dataset_integrity",
    "tokenized_dataset_batch_alignment_invalid": "tokenized_dataset_integrity",
    "tokenized_dataset_range_insufficient": "tokenized_dataset_integrity",
    "training_data_view_unavailable": "training_data_view_integrity",
    "training_data_preparation_required": "training_data_view_integrity",
    "training_data_configuration_mismatch": "training_data_view_selection",
    "multiple_training_data_views": "training_data_view_selection",
    "training_data_view_range_insufficient": "training_data_view_capacity",
    "invalid_code_version": "code_version_ready",
    "code_version_not_ready": "code_version_ready",
    "code_artifact_unavailable": "code_version_ready",
    "absolute_cwd_with_code_version": "code_version_ready",
    "engine_unavailable": "engine_configuration_valid",
    "missing_executable": "engine_configuration_valid",
    "unresolved_engine_config_template": "engine_configuration_valid",
    "invalid_gpu_request": "worker_capability_available",
    "invalid_worker_selector": "worker_capability_available",
    "worker_selector_unavailable": "worker_capability_available",
    "training_dataset_snapshot_missing": "worker_capability_available",
    "training_dataset_snapshot_unavailable": "worker_capability_available",
    "training_data_worker_location_mismatch": "worker_capability_available",
    "no_matching_worker": "worker_capability_available",
}


def annotate_system_validation_issues(issues: list[dict[str, Any]]) -> None:
    for issue in issues:
        validation_id = ISSUE_VALIDATION_IDS.get(str(issue.get("code") or ""))
        if validation_id:
            issue["validation_id"] = validation_id
            issue["category"] = "system_validation"
            issue["configurable"] = False


def build_training_system_validations(
    issues: list[dict[str, Any]],
    *,
    executed: set[str],
    not_applicable: set[str],
    evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build statuses from explicit execution state, never from error absence."""

    evidence = evidence or {}
    blocked = {
        str(issue.get("validation_id"))
        for issue in issues
        if issue.get("validation_id") and issue.get("severity") == "blocker"
    }
    deferred = {
        str(issue.get("validation_id"))
        for issue in issues
        if issue.get("validation_id") and issue.get("severity") == "warning"
    }
    results: list[dict[str, Any]] = []
    for definition in TRAINING_SYSTEM_VALIDATIONS:
        validation_id = definition["id"]
        if validation_id in blocked:
            status = "blocked"
        elif validation_id in deferred:
            status = "pending"
        elif validation_id in not_applicable:
            status = "not_applicable"
        elif validation_id in executed:
            status = "passed"
        else:
            status = "pending"
        results.append(
            {
                **definition,
                "category": "system_validation",
                "configurable": False,
                "status": status,
                **(evidence.get(validation_id) or {}),
            }
        )
    return results
