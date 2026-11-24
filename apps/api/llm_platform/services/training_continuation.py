"""Checkpoint handoffs create a new frozen run; ordinary replay keeps its intent."""
from forgelm_plugin_sdk.training import continuation_policy
from sqlalchemy import select

from ..models import Artifact, Task, TrainingStateCheckpoint
from .artifacts import artifact_storage


def prepare(session, project_id, node, config, inputs, checkpoint_id, mode, keep_optimizer=False):
    checkpoint = session.get(TrainingStateCheckpoint, checkpoint_id)
    if not checkpoint or checkpoint.project_id != project_id:
        raise ValueError("Checkpoint 不属于此项目")
    contract = node.get("training", {})
    previous = checkpoint.manifest
    source_contract = previous.get("contract", {})
    if mode not in contract.get("modes", []) or contract.get("checkpoint_format") != source_contract.get("checkpoint_format"):
        raise ValueError("目标组件不支持此 Checkpoint 格式或接续模式")
    policy = continuation_policy(mode, keep_optimizer=keep_optimizer)
    if mode != "weights" and contract.get("state_compatibility") != source_contract.get("state_compatibility"):
        raise ValueError("训练状态布局不兼容；可检查仅继承权重模式")
    source = session.get(Task, checkpoint.task_id)
    source_spec = source.payload["suite_execution"]
    if mode == "full" and (config != previous.get("config") or inputs != source_spec["inputs"]):
        raise ValueError("完整状态接续要求训练配置和输入保持一致；更换数据或优化方式请使用配置变更接续")
    artifact = session.get(Artifact, checkpoint.artifact_id)
    if not artifact or not artifact_storage.managed_payload_path(session, artifact, project_id):
        raise ValueError("Checkpoint 托管制品不可用")
    return {"schema": "forgelm.training-continuation.v1", "checkpoint_id": checkpoint.id,
            "source_task_id": checkpoint.task_id, "source_attempt_id": checkpoint.attempt_id,
            "source_execution_digest": checkpoint.execution_digest,
            "mode": mode, "state_policy": policy, "manifest": previous,
            "artifact": {"artifact_id": artifact.id, "digest": artifact.content_hash, "size_bytes": artifact.size_bytes},
            "validation": {"status": "runtime_check_required", "checks": ["模型参数名称、形状与类型", "优化器参数映射", "状态文件完整性"]}}


def latest_resume(session, task):
    spec = task.payload.get("suite_execution")
    if not spec:
        return None
    checkpoint = session.scalar(select(TrainingStateCheckpoint).where(TrainingStateCheckpoint.task_id == task.id).order_by(TrainingStateCheckpoint.created_at.desc()).limit(1))
    if not checkpoint:
        return None
    from ..suite_models import SuiteVersion
    version = session.get(SuiteVersion, spec["version_id"])
    node = next(n for n in version.package["nodes"] if n["id"] == spec["node_id"])
    return prepare(session, task.project_id, node, spec["config"], spec["inputs"], checkpoint.id, "full")


def freeze_native(session, project_id, revision, stages):
    """Explicit native checkpoint segment; ordinary retries never call this."""
    from copy import deepcopy

    from ..models import Checkpoint, CodeVersion, TrainingRun
    ref = revision.input_model_ref
    checkpoint = session.get(Checkpoint, ref.get("checkpoint_id"))
    if not checkpoint or checkpoint.project_id != project_id or checkpoint.status != "available":
        raise ValueError("Checkpoint 不属于此项目或不可用")
    run = session.get(TrainingRun, checkpoint.training_run_id)
    source = session.get(Task, run.task_id) if run else None
    artifact = session.get(Artifact, checkpoint.artifact_id)
    if not source or not artifact or artifact.availability_state != "available":
        raise ValueError("Checkpoint 来源训练或制品不可用")
    if artifact_storage.inference_access(session, artifact, project_id)["type"] == "unavailable":
        raise ValueError("Checkpoint 没有可访问的制品副本")
    code = session.get(CodeVersion, revision.code_version_id)
    if not code or (code.training_contract or {}).get("checkpoint_continuation") != "forgelm.native-continuation.v1":
        raise ValueError("目标引擎未声明 Checkpoint 分段接续协议")
    mode = ref.get("mode", "weights")
    policy = continuation_policy(mode, keep_optimizer=ref.get("keep_optimizer", False))
    old = source.payload
    if revision.tokenizer_version_id != (old.get("tokenizer") or {}).get("version_id"):
        raise ValueError("接续不能隐式改变 Tokenizer 语义")
    if mode == "full" and stages != old.get("stages"):
        raise ValueError("完整恢复必须保留训练配置和数据；修改配置请选择配置变更接续")
    original = old.get("resume_checkpoint") or {}
    value = {"mode": "checkpoint_continuation", "continuation_mode": mode,
             "checkpoint_id": checkpoint.id, "source_task_id": source.id, "training_run_id": run.id,
             "step": checkpoint.step, "artifact_id": artifact.id, "content_hash": artifact.content_hash,
             "hash_algorithm": (artifact.metadata_json or {}).get("hash_algorithm"),
             "state_policy": policy, "sequence_offset": int(original.get("sequence_offset") or 0) if mode == "full" else 0,
             "token_count": int(original.get("token_count") or 0) if mode == "full" else 0,
             "model_compatibility": (checkpoint.metrics or {}).get("model_compatibility")}
    if mode == "full" and old.get("training_data_view_inputs"):
        value["training_data_view"] = deepcopy(old["training_data_view_inputs"][0])
    return value
