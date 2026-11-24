"""Native, immutable performance comparisons and same-stage full-state continuation."""

from copy import deepcopy
from pathlib import PurePosixPath

from fastapi import HTTPException
from llm_platform_core.training_performance import SEARCH_PROTOCOL, automatic_performance_candidates
from sqlalchemy import select

from .. import models as m
from .tasks import create_task
from .transactions import commit_or_flush

PROTOCOL = "forgelm.training-performance.v1"
TUNING_FIELDS = {
    "micro_batch_size", "gradient_accumulation_steps", "activation_checkpointing",
    "checkpoint_every_n_layers", "resource_batch_mode", "resource_batch_profile",
    "dataloader_num_workers", "dataloader_prefetch_factor", "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "optimizations",
}


def _reject(message):
    raise HTTPException(409, detail={"code": "invalid_training_performance", "message": message})


def _pause_receipt(session, source):
    attempt = session.scalar(select(m.TaskAttempt).where(m.TaskAttempt.task_id == source.id)
                             .order_by(m.TaskAttempt.attempt_number.desc()).limit(1))
    event = session.scalar(select(m.TaskEvent).where(
        m.TaskEvent.task_id == source.id, m.TaskEvent.attempt_id == attempt.id,
        m.TaskEvent.event_type == "control.acknowledged",
    ).order_by(m.TaskEvent.sequence.desc()).limit(1)) if attempt else None
    receipt = event.payload if event else {}
    if receipt.get("protocol") != "forgelm-command-control-file-v1" or receipt.get("action") != "safe_pause":
        _reject("来源任务最新执行没有完整状态安全暂停回执")
    return receipt, attempt


def create_comparison(session, project_id, request, version):
    source = session.get(m.Task, request.source_task_id)
    if source is None or source.project_id != project_id or source.kind != "train":
        _reject("性能比较必须引用当前项目的正式训练任务")
    if (source.payload.get("resource_request") or {}).get("gpus", 1) != 1:
        _reject("当前性能比较支持单 GPU 训练，不能用单卡比较替代分布式训练状态")
    if source.observed_state != "paused":
        _reject("请先通过系统安全暂停训练，取得完整 Trainer 状态")
    receipt, attempt = _pause_receipt(session, source)
    checkpoint = str(receipt.get("checkpoint") or "")
    step = int(receipt.get("step") or 0)
    if step < 1:
        _reject("安全暂停必须已完成至少一个优化器更新")
    path = PurePosixPath(checkpoint)
    if not path.is_absolute() or path.name != f"checkpoint-{step}" or path.parent.name != "trainer":
        _reject("安全暂停回执没有合法的完整状态目录")
    if (version.training_contract or {}).get("performance_probe") != PROTOCOL:
        _reject("模型引擎版本没有声明平台训练性能比较协议")
    if (version.entrypoints or {}).get("training") != "train.py":
        _reject("性能比较必须使用模型引擎的正式训练入口")
    artifact = session.get(m.Artifact, version.artifact_id)
    if artifact is None or artifact.availability_state != "available":
        _reject("模型引擎制品不可用")
    if attempt is None or not attempt.worker_id:
        _reject("来源 Worker 不可用")
    payload = deepcopy(source.payload)
    stages = payload.get("stages") or []
    if len(stages) != 1 or stages[0].get("type") != "pretrain" or not payload.get("training_data_view_inputs"):
        _reject("当前性能比较支持单阶段、冻结 Training Data View 的预训练")
    original_stage = deepcopy(stages[0])
    resolved = ((attempt.environment_snapshot or {}).get("resource_batch_plan") or {}).get("selected") or {}
    stages[0].update(resolved.get("values") or {})
    stages[0]["_resolved_resource_batch_values"] = deepcopy(resolved.get("values") or {})
    effective = int(stages[0].get("effective_batch_tokens") or 0)
    length = int(stages[0].get("sequence_length") or 0)
    if length <= 0 or effective < length or effective % length:
        _reject("来源任务缺少有效的固定批量几何")
    baseline = {"micro_batch_size": int(stages[0]["micro_batch_size"]),
                "activation_checkpointing": stages[0].get("activation_checkpointing", True),
                "checkpoint_every_n_layers": stages[0].get("checkpoint_every_n_layers", 1)}
    if stages[0].get("optimizations") is not None:
        baseline["optimizations"] = deepcopy(stages[0]["optimizations"])
    candidates = [baseline]
    mode = request.search_mode or ("manual" if request.candidates else "auto")
    if mode == "auto" and (version.training_contract or {}).get("performance_search") != SEARCH_PROTOCOL:
        _reject("该引擎没有声明自动性能搜索协议，请选择支持的代码版本")
    if mode == "auto" and int((payload.get("validation_data_input") or {}).get("sequence_count") or 0) < 32:
        _reject("自动搜索需要至少32条固定验证序列以比较候选的验证loss")
    if mode == "manual" and not request.candidates:
        _reject("手动比较至少需要一个候选")
    for candidate in request.candidates:
        value = candidate.model_dump(exclude_none=True)
        if value.get("optimizations") and (version.training_contract or {}).get("optimization_profiles") != "forgelm.microlm-optimizations.v1":
            _reject("模型引擎未声明算子组合比较协议")
        if not effective or not length or effective % (length * value["micro_batch_size"]):
            _reject("候选微批量必须整除原始有效批量，不得改变优化器更新预算")
        if value not in candidates:
            candidates.append(value)
    if mode == "auto":
        candidates = automatic_performance_candidates(baseline, effective, length)
    source_range = payload.get("training_data_range") or {}
    if int(source_range.get("planned_steps", 0)) <= step + request.warmup_steps + request.measurement_steps:
        _reject("训练剩余步数不足以执行有界性能比较")
    for key in list(payload):
        if key.startswith("pipeline_") or key in {
            "training_run_id", "plan_id", "revision_id", "resume_checkpoint", "resource_batch_policy",
            "resolved_resource_batch_plan", "system_validation_contract",
        }:
            payload.pop(key)
    payload["code"] = {
        "version_id": version.id, "version": version.version, "artifact_id": artifact.id,
        "content_hash": artifact.content_hash, "git_commit": version.git_commit,
        "entrypoints": version.entrypoints, "dependency_spec": version.dependency_spec,
        "training_contract": version.training_contract,
    }
    model_config = str((source.payload.get("input_model_ref") or {}).get("config_path") or "configs/model.json")
    payload["command"] = {"executable": "__worker_python__", "args": ["train.py", "--model-config", model_config],
                          "data_input_mode": "tokenized_required"}
    payload["resource_request"] = {**payload.get("resource_request", {}), "gpus": 1,
                                   "worker_selector": {"worker_id": attempt.worker_id}}
    payload["output_retention"] = "retain"
    payload["performance_probe"] = {
        "schema_version": PROTOCOL, "source_task_id": source.id,
        "source_training_run_id": source.payload["training_run_id"],
        "source_checkpoint_path": checkpoint, "source_step": step,
        "source_stage": original_stage, "source_resume": source.payload.get("resume_checkpoint") or {},
        "legacy_position_from_frozen_range": (
            (source.payload.get("effective_config", {}).get("engine_config", {}).get("model_implementation")
             == "microlm_306m_v3.modeling:MicroLMForCausalLM")
            and (source.payload.get("code", {}).get("training_contract", {}).get("performance_probe") != PROTOCOL)
        ),
        "candidates": candidates, "warmup_steps": request.warmup_steps,
        "measurement_steps": request.measurement_steps,
        "memory_headroom_mb": request.memory_headroom_mb if request.memory_headroom_mb is not None else (None if mode == "auto" else 2048),
        "search_mode": mode, "max_candidates": request.max_candidates,
        "search_protocol": SEARCH_PROTOCOL if mode == "auto" else None,
    }
    task = create_task(session, project_id=project_id, kind="training_performance",
                       title="训练性能比较：完整状态、固定数据与有效批量", adapter_id="builtin.command.training",
                       payload=payload)
    commit_or_flush(session)
    return {"id": task.id, "task_id": task.id}


def freeze_continuation(session, project_id, revision, stages):
    ref = revision.input_model_ref or {}
    task = session.get(m.Task, ref.get("performance_task_id"))
    if task is None or task.project_id != project_id or task.kind != "training_performance" or task.observed_state != "succeeded":
        _reject("完整状态接续必须引用当前项目成功的性能比较节点")
    probe = task.payload.get("performance_probe") or {}
    report = (task.result or {}).get("performance_report") or {}
    recommendation = report.get("recommendation")
    if report.get("schema_version") != PROTOCOL or not recommendation:
        _reject("性能比较没有可采用的配置")
    source = session.get(m.Task, probe.get("source_task_id"))
    if source is None or source.observed_state not in {"paused", "cancelled"}:
        _reject("来源训练仍在运行，不能从过期状态并发接续")
    receipt, _attempt = _pause_receipt(session, source)
    if int(receipt.get("step", -1)) != probe.get("source_step"):
        _reject("来源训练位置已变化，请重新比较")
    expected = probe["source_stage"]
    if len(stages) != 1 or {k: v for k, v in stages[0].items() if k not in TUNING_FIELDS} != {
        k: v for k, v in expected.items() if k not in TUNING_FIELDS
    }:
        _reject("完整状态接续不可改变数据、预算、学习率或其他训练语义")
    if any(stages[0].get(k, True if k == "activation_checkpointing" else 1) != v for k, v in recommendation.items()):
        _reject("接续配置必须采用性能节点实测通过的推荐值")
    frozen = task.payload
    if revision.code_version_id != frozen["code"]["version_id"] or revision.tokenizer_version_id != frozen["tokenizer"]["version_id"]:
        _reject("接续必须使用已比较的模型代码和同一 Tokenizer")
    record = next((row for row in task.result.get("output_artifacts", []) if row.get("ref") == "full-trainer-state"), {})
    artifact = session.get(m.Artifact, record.get("artifact_id"))
    if artifact is None or artifact.kind != "checkpoint" or artifact.availability_state != "available":
        _reject("完整 Trainer 状态制品不可用")
    source_resume = probe.get("source_resume") or {}
    return {
        "mode": "full_state_continuation", "performance_task_id": task.id,
        "source_task_id": source.id, "training_run_id": probe["source_training_run_id"],
        "step": probe["source_step"], "artifact_id": artifact.id, "content_hash": artifact.content_hash,
        "token_count": int(source_resume.get("token_count") or 0),
        "sequence_offset": int((frozen.get("training_data_range") or {}).get("sequence_start") or 0),
        "training_data_view": frozen["training_data_view_inputs"][0],
        "model_compatibility": frozen.get("checkpoint_compatibility"),
    }
