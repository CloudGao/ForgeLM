from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from llm_platform_core.domain import TaskStatus
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    ProjectPipeline,
    ProjectPipelineInstance,
    ProjectPipelineNodeRun,
    ProjectPipelineRevision,
    ProjectPipelineRun,
    Task,
    TrainingRun,
    utcnow,
)
from ..schemas import (
    ProjectPipelineDefinitionCreate,
    ProjectPipelineDraftUpdate,
    ProjectPipelineInstanceCreate,
    ProjectPipelineInstanceDraftUpdate,
)
from .global_data import asset_descriptor, project_has_asset_access
from .lineage import (
    node_lineage_views,
    record_terminal_run_lineage,
    run_lineage_view,
)
from .project_pipelines import project_pipeline_view
from .tasks import create_task, transition_task

NODE_CATALOG: tuple[dict[str, Any], ...] = (
    {"type": "fact_translation", "label": "三元组翻译", "category": "数据", "description": "保留事实结构与方向，翻译字段后输出中文三元组", "color": "#0e7490", "inputs": [{"id": "in", "type": "data", "label": "原始三元组"}], "outputs": [{"id": "out", "type": "data", "label": "中文三元组"}]},
    {"type": "translation", "label": "数据翻译", "category": "数据", "description": "本地模型批量英→中翻译，关闭 thinking", "color": "#0891b2", "inputs": [{"id": "in", "type": "data", "label": "可选上游数据"}], "outputs": [{"id": "out", "type": "data", "label": "中文 CPT 数据"}]},
    {"type": "knowledge", "label": "知识数据生成", "category": "数据", "description": "本地模型多方式重写知识，关闭 thinking", "color": "#0d9488", "inputs": [{"id": "in", "type": "data", "label": "可选上游数据"}], "outputs": [{"id": "out", "type": "data", "label": "中文 CPT 数据"}]},
    {
        "type": "selection", "label": "模型选优", "category": "模型",
        "description": "根据已完成验证评测的数值指标选择候选模型与 Checkpoint",
        "color": "#eab308",
        "inputs": [{"id": "in", "type": "any", "label": "候选与评测", "multiple": True}],
        "outputs": [{"id": "out", "type": "model", "label": "所选模型"}],
    },
    {
        "type": "preparation", "label": "数据与模型准备", "category": "模型准备",
        "description": "通过内置领域服务创建模板、监督映射、编码 Corpus 和评测配置",
        "color": "#7c3aed",
        "inputs": [{"id": "in", "type": "any", "label": "准备输入", "multiple": True}],
        "outputs": [{"id": "out", "type": "any", "label": "准备产物"}],
    },
    {
        "type": "data",
        "label": "数据",
        "category": "数据",
        "description": "引用项目已绑定的精确数据版本",
        "color": "#3b82f6",
        "inputs": [],
        "outputs": [{"id": "out", "type": "data", "label": "数据"}],
    },
    {
        "type": "transform",
        "label": "数据转换",
        "category": "数据",
        "description": "执行格式转换或文本抽取命令",
        "color": "#06b6d4",
        "inputs": [{"id": "in", "type": "data", "label": "数据"}],
        "outputs": [{"id": "out", "type": "data", "label": "转换结果"}],
    },
    {
        "type": "clean",
        "label": "数据清洗",
        "category": "数据",
        "description": "执行清洗、过滤或去重命令",
        "color": "#14b8a6",
        "inputs": [{"id": "in", "type": "data", "label": "数据"}, {"id": "selection", "type": "data", "label": "文件选择"}, {"id": "recipe", "type": "any", "label": "清洗配方"}],
        "outputs": [{"id": "out", "type": "data", "label": "清洗结果"}],
    },
    {
        "type": "mixture",
        "label": "数据混合",
        "category": "数据",
        "description": "组合多个数据输入，区别于控制流汇合",
        "color": "#0ea5e9",
        "inputs": [{"id": "in", "type": "data", "label": "多个数据", "multiple": True}],
        "outputs": [{"id": "out", "type": "data", "label": "混合数据"}],
    },
    {
        "type": "tokenizer",
        "label": "Tokenizer",
        "category": "模型准备",
        "description": "训练 Tokenizer 或构建编码数据",
        "color": "#8b5cf6",
        "inputs": [{"id": "in", "type": "data", "label": "数据"}],
        "outputs": [{"id": "out", "type": "tokenizer", "label": "Tokenizer"}],
    },
    {
        "type": "training",
        "label": "训练",
        "category": "模型",
        "description": "执行预训练或后训练命令",
        "color": "#f97316",
        "inputs": [{"id": "in", "type": "any", "label": "训练输入", "multiple": True}],
        "outputs": [{"id": "out", "type": "model", "label": "模型"}],
    },
    {
        "type": "evaluation",
        "label": "评测",
        "category": "模型",
        "description": "运行 Benchmark 或自定义评测命令",
        "color": "#ec4899",
        "inputs": [{"id": "in", "type": "any", "label": "模型 / 数据", "multiple": True}],
        "outputs": [{"id": "out", "type": "report", "label": "评测报告"}],
    },
    {
        "type": "check",
        "label": "检查",
        "category": "控制",
        "description": "产生显式检查结果，不默认成为强制门禁",
        "color": "#eab308",
        "inputs": [{"id": "in", "type": "any", "label": "待检查输入", "multiple": True}],
        "outputs": [{"id": "out", "type": "control", "label": "检查结果"}],
    },
    {
        "type": "branch",
        "label": "条件分支",
        "category": "控制",
        "description": "根据声明式条件选择一个出口",
        "color": "#f59e0b",
        "inputs": [{"id": "in", "type": "any", "label": "条件输入"}],
        "outputs": [
            {"id": "true", "type": "control", "label": "通过"},
            {"id": "false", "type": "control", "label": "未通过"},
            {"id": "default", "type": "control", "label": "默认"},
        ],
    },
    {
        "type": "join",
        "label": "汇合",
        "category": "控制",
        "description": "汇合实际被选择的上游分支",
        "color": "#64748b",
        "inputs": [{"id": "in", "type": "any", "label": "多个分支", "multiple": True}],
        "outputs": [{"id": "out", "type": "any", "label": "汇合结果"}],
    },
    {
        "type": "custom",
        "label": "自定义任务",
        "category": "扩展",
        "description": "通过现有无 Shell 结构化命令适配器执行",
        "color": "#6366f1",
        "inputs": [{"id": "in", "type": "any", "label": "输入", "multiple": True}],
        "outputs": [{"id": "out", "type": "any", "label": "输出"}],
    },
    {
        "type": "quantization",
        "label": "量化",
        "category": "交付",
        "description": "执行模型量化命令",
        "color": "#a855f7",
        "inputs": [{"id": "in", "type": "model", "label": "模型"}],
        "outputs": [{"id": "out", "type": "model", "label": "量化模型"}],
    },
    {
        "type": "release",
        "label": "发布",
        "category": "交付",
        "description": "执行发布或导出命令",
        "color": "#22c55e",
        "inputs": [{"id": "in", "type": "model", "label": "模型"}],
        "outputs": [{"id": "out", "type": "release", "label": "发布记录"}],
    },
)

CATALOG_BY_TYPE = {item["type"]: item for item in NODE_CATALOG}
TASK_NODE_TYPES = {
    "translation", "fact_translation", "knowledge",
    "selection",
    "preparation",
    "transform",
    "clean",
    "mixture",
    "tokenizer",
    "training",
    "evaluation",
    "custom",
    "quantization",
    "release",
}
DEFAULT_ADAPTERS = {
    "transform": "builtin.command.data",
    "clean": "builtin.command.data",
    "mixture": "builtin.command.data",
    "tokenizer": "builtin.command.tokenizer",
    "training": "builtin.command.training",
    "evaluation": "builtin.command.evaluation",
    "custom": "builtin.command.data",
    "quantization": "builtin.command.quantization",
    "release": "builtin.command.data",
}
TERMINAL_NODE_STATUSES = {"succeeded", "failed", "cancelled", "skipped"}
ACTIVE_TASK_STATES = {
    TaskStatus.QUEUED,
    TaskStatus.ASSIGNED,
    TaskStatus.STARTING,
    TaskStatus.RUNNING,
    TaskStatus.PAUSING,
    TaskStatus.STOPPING,
}

COMPATIBILITY_METADATA_KEY = "__forgelm"
IMPORTED_INSTANCE_KIND = "imported"
NATIVE_INSTANCE_KIND = "native"
DOMAIN_TASK_OWNER = "domain_task"
PIPELINE_SCHEDULER_OWNER = "pipeline_scheduler"


class PipelineDagError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def pipeline_node_catalog() -> list[dict[str, Any]]:
    from .native_pipeline_nodes import operation_catalog
    return [{**deepcopy(item), "native_operations": operation_catalog(item["type"])} for item in NODE_CATALOG]


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _legacy_graph(
    revision: ProjectPipelineRevision | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    legacy = [
        ("data", "数据", "data", 0.0, 20.0),
        ("tokenizer", "Tokenizer", "tokenizer", 0.0, 180.0),
        ("code", "代码", "custom", 0.0, 340.0),
        ("training", "训练", "training", 320.0, 180.0),
        ("model", "模型", "custom", 640.0, 180.0),
        ("evaluation", "评测", "evaluation", 960.0, 60.0),
        ("quantization", "量化", "quantization", 960.0, 300.0),
        ("release", "发布", "release", 1280.0, 300.0),
    ]
    references = {
        "data": {"data_refs": deepcopy(revision.data_refs or [])} if revision else {},
        "tokenizer": {
            "tokenizer_version_id": revision.tokenizer_version_id,
        }
        if revision
        else {},
        "code": {"code_version_id": revision.code_version_id} if revision else {},
        "training": {
            "plan_revision_id": revision.plan_revision_id,
            "training_run_id": revision.training_run_id,
            "execution_owner": DOMAIN_TASK_OWNER,
        }
        if revision
        else {},
        "model": {"model_version_id": revision.model_version_id} if revision else {},
        "evaluation": {"evaluation_run_id": revision.evaluation_run_id}
        if revision
        else {},
        "quantization": {"quantization_run_id": revision.quantization_run_id}
        if revision
        else {},
        "release": {"release_id": revision.release_id} if revision else {},
    }
    nodes = [
        {
            "id": key,
            "type": node_type,
            "position": {"x": x, "y": y},
            "label": label,
            "description": "兼容既有固定产线映射",
            "config": {
                "legacy_mapping": True,
                **{
                    field: value
                    for field, value in references[key].items()
                    if value not in (None, "", [])
                },
            },
        }
        for key, label, node_type, x, y in legacy
    ]
    dependency_edges = [
        ("data", "training"),
        ("tokenizer", "training"),
        ("code", "training"),
        ("training", "model"),
        ("model", "evaluation"),
        ("model", "quantization"),
        ("quantization", "release"),
    ]
    edges = [
        {
            "id": f"legacy-{source}-{target}",
            "source": source,
            "target": target,
            "source_handle": "out",
            "target_handle": "in",
            "label": "",
        }
        for source, target in dependency_edges
    ]
    return nodes, edges


def revision_graph(revision: ProjectPipelineRevision | None) -> dict[str, Any]:
    if revision is None:
        return {"nodes": [], "edges": [], "variables": {}, "viewport": {}}
    nodes = deepcopy(revision.nodes or [])
    edges = deepcopy(revision.edges or [])
    if not nodes and revision.mapping_snapshot:
        nodes, edges = _legacy_graph(revision)
    return {
        "nodes": nodes,
        "edges": edges,
        "variables": deepcopy(revision.variables or {}),
        "viewport": deepcopy(revision.viewport or {}),
    }


def _normalized_graph_content(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
) -> bytes:
    content = {
        "nodes": sorted(nodes, key=lambda item: str(item.get("id") or "")),
        "edges": sorted(edges, key=lambda item: str(item.get("id") or "")),
        "variables": variables,
    }
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def graph_content_hash(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
) -> str:
    return hashlib.sha256(_normalized_graph_content(nodes, edges, variables)).hexdigest()


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    metadata = (value or {}).get(COMPATIBILITY_METADATA_KEY)
    return metadata if isinstance(metadata, dict) else {}


def _instance_metadata(instance: ProjectPipelineInstance) -> dict[str, Any]:
    return _metadata(instance.variables or {})


def _run_metadata(run: ProjectPipelineRun) -> dict[str, Any]:
    return _metadata(run.context_snapshot or {})


def is_imported_pipeline_instance(instance: ProjectPipelineInstance) -> bool:
    return _instance_metadata(instance).get("instance_kind") == IMPORTED_INSTANCE_KIND or bool(
        (instance.variables or {}).get("__training_history", {}).get("training_run_id")
    )


def is_imported_pipeline_run(run: ProjectPipelineRun) -> bool:
    return _run_metadata(run).get("run_kind") == IMPORTED_INSTANCE_KIND


def _task_pipeline_status(task: Task | None, *, persisted: bool = False) -> str:
    if task is None:
        return "failed"
    state = task.observed_state
    if state in {TaskStatus.QUEUED, TaskStatus.ASSIGNED, TaskStatus.STARTING}:
        return "queued"
    if state in {TaskStatus.RUNNING, TaskStatus.PAUSING}:
        return "running"
    if state == TaskStatus.PAUSED:
        return "running" if persisted else "paused"
    if state == TaskStatus.STOPPING:
        return "cancelling"
    if state == TaskStatus.SUCCEEDED:
        return "succeeded"
    if state in {TaskStatus.FAILED, TaskStatus.LOST}:
        return "failed"
    if state == TaskStatus.CANCELLED:
        return "cancelled"
    return "queued"


def _task_node_status(task: Task | None) -> str:
    status = _task_pipeline_status(task)
    return {
        "cancelling": "running",
    }.get(status, status)


def _unique_imported_instance_name(
    session: Session,
    project_id: str,
    pipeline_name: str,
    training_run_id: str,
) -> str:
    base = f"{pipeline_name} · 现有运行"
    names = set(
        session.scalars(
            select(ProjectPipelineInstance.name).where(
                ProjectPipelineInstance.project_id == project_id
            )
        ).all()
    )
    if base not in names:
        return base
    candidate = f"{base} [{training_run_id[:8]}]"
    if candidate not in names:
        return candidate
    suffix = 2
    while f"{candidate} #{suffix}" in names:
        suffix += 1
    return f"{candidate} #{suffix}"


def create_imported_pipeline_execution(
    session: Session,
    pipeline: ProjectPipeline,
) -> ProjectPipelineInstance | None:
    """Attach an existing named lifecycle pipeline without taking over its Task."""

    revision = (
        session.get(ProjectPipelineRevision, pipeline.current_revision_id)
        if pipeline.current_revision_id
        else None
    )
    if revision is None or revision.nodes or not revision.training_run_id:
        return None
    candidates = session.scalars(
        select(ProjectPipelineInstance).where(
            ProjectPipelineInstance.pipeline_id == pipeline.id,
            ProjectPipelineInstance.definition_revision_id == revision.id,
        )
    ).all()
    existing = next(
        (item for item in candidates if is_imported_pipeline_instance(item)), None
    )
    if existing is not None:
        return existing

    training_run = session.get(TrainingRun, revision.training_run_id)
    task = session.get(Task, training_run.task_id) if training_run is not None else None
    if training_run is None:
        return None
    graph = revision_graph(revision)
    metadata = {
        "instance_kind": IMPORTED_INSTANCE_KIND,
        "run_kind": IMPORTED_INSTANCE_KIND,
        "execution_owner": DOMAIN_TASK_OWNER,
        "source_pipeline_revision_id": revision.id,
        "source_training_run_id": training_run.id,
        "source_task_id": task.id if task else None,
    }
    variables = deepcopy(graph["variables"])
    variables[COMPATIBILITY_METADATA_KEY] = metadata
    instance = ProjectPipelineInstance(
        project_id=pipeline.project_id,
        pipeline_id=pipeline.id,
        definition_revision_id=revision.id,
        name=_unique_imported_instance_name(
            session, pipeline.project_id, pipeline.name, training_run.id
        ),
        description="由现有命名产线自动接入；配置和运行事实由原 TrainingRun/Task 管理。",
        status="ready",
        nodes=deepcopy(graph["nodes"]),
        edges=deepcopy(graph["edges"]),
        variables=variables,
        viewport=deepcopy(graph["viewport"]),
        validation_report={
            "valid": True,
            "errors": [],
            "warnings": [
                {
                    "severity": "warning",
                    "code": "imported_pipeline_read_only",
                    "message": "兼容实例只读展示原任务事实，不能由 DAG 调度器重新执行。",
                }
            ],
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        },
        content_hash=graph_content_hash(graph["nodes"], graph["edges"], variables),
        activated_at=utcnow(),
        locked=True,
    )
    session.add(instance)
    session.flush()

    persisted_status = _task_pipeline_status(task, persisted=True)
    run = ProjectPipelineRun(
        project_id=pipeline.project_id,
        pipeline_id=pipeline.id,
        revision_id=revision.id,
        instance_id=instance.id,
        status=persisted_status,
        trigger_kind="imported",
        input_snapshot={
            "legacy_mapping": deepcopy(revision.mapping_snapshot or {}),
        },
        context_snapshot={
            "pipeline_name": pipeline.name,
            "pipeline_instance_name": instance.name,
            "revision": revision.revision,
            COMPATIBILITY_METADATA_KEY: metadata,
        },
        graph_snapshot={
            "nodes": deepcopy(graph["nodes"]),
            "edges": deepcopy(graph["edges"]),
            "variables": variables,
            "viewport": deepcopy(graph["viewport"]),
        },
        error=deepcopy(task.error or {}) if task and persisted_status == "failed" else {},
        started_at=(
            (task.started_at or task.created_at) if task else training_run.started_at
        ),
        ended_at=task.ended_at if task and persisted_status in {"succeeded", "failed", "cancelled"} else None,
    )
    session.add(run)
    session.flush()

    source_terminal = persisted_status in {"succeeded", "failed", "cancelled"}
    downstream_refs = {
        "model": revision.model_version_id,
        "evaluation": revision.evaluation_run_id,
        "quantization": revision.quantization_run_id,
        "release": revision.release_id,
    }
    task_already_linked = bool(
        task
        and session.scalar(
            select(ProjectPipelineNodeRun.id).where(
                ProjectPipelineNodeRun.task_id == task.id
            )
        )
    )
    for node in graph["nodes"]:
        node_id = str(node["id"])
        if node_id == "training":
            status = _task_node_status(task)
        elif node_id == "data":
            status = "succeeded" if revision.data_refs else "waiting"
        elif node_id == "tokenizer":
            status = "succeeded" if revision.tokenizer_version_id else "waiting"
        elif node_id == "code":
            status = "succeeded" if revision.code_version_id else "waiting"
        elif downstream_refs.get(node_id):
            status = "succeeded"
        else:
            status = "skipped" if source_terminal else "waiting"
        is_training = node_id == "training"
        session.add(
            ProjectPipelineNodeRun(
                pipeline_run_id=run.id,
                node_id=node_id,
                node_type=node["type"],
                label=node["label"],
                status=status,
                config_snapshot=deepcopy(node.get("config") or {}),
                input_snapshot={"source": "existing_project_pipeline"},
                task_id=task.id if task and is_training and not task_already_linked else None,
                error=deepcopy(task.error or {})
                if task and is_training and status == "failed"
                else {},
                started_at=(task.started_at or task.created_at)
                if task and is_training
                else None,
                ended_at=task.ended_at
                if task and is_training and status in TERMINAL_NODE_STATUSES
                else None,
            )
        )
    session.flush()
    imported_rows = list(
        session.scalars(
            select(ProjectPipelineNodeRun).where(
                ProjectPipelineNodeRun.pipeline_run_id == run.id
            )
        ).all()
    )
    _, projected_nodes = _imported_run_projection(session, run, imported_rows)
    record_terminal_run_lineage(
        session,
        run,
        imported_rows,
        projected_nodes=projected_nodes,
        execution_owner=DOMAIN_TASK_OWNER,
    )
    return instance


def _port(catalog: dict[str, Any], direction: str, handle: str | None) -> dict[str, Any] | None:
    ports = list(catalog[direction])
    if not ports:
        return None
    if handle:
        return next((item for item in ports if item["id"] == handle), None)
    return ports[0]


def _command_issues(node: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    config = node.get("config") or {}
    command = config.get("command") or {}
    executable = command.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        issues.append(
            {
                "severity": "error",
                "code": "missing_executable",
                "node_id": node["id"],
                "message": f"节点“{node['label']}”尚未配置可执行命令",
            }
        )
        return issues
    args = command.get("args") or []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        issues.append(
            {
                "severity": "error",
                "code": "invalid_command_args",
                "node_id": node["id"],
                "message": f"节点“{node['label']}”的命令参数必须是字符串数组",
            }
        )
    env = command.get("env") or {}
    if not isinstance(env, dict):
        issues.append(
            {
                "severity": "error",
                "code": "invalid_command_env",
                "node_id": node["id"],
                "message": f"节点“{node['label']}”的环境变量必须是对象",
            }
        )
    return issues


def validate_pipeline_graph(
    session: Session,
    project_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    execution_bindings: bool,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not nodes:
        issues.append(
            {
                "severity": "error",
                "code": "empty_pipeline",
                "message": "Pipeline 至少需要一个节点",
            }
        )
    node_ids = [str(node.get("id") or "") for node in nodes]
    duplicate_nodes = [item for item, count in Counter(node_ids).items() if count > 1]
    for node_id in duplicate_nodes:
        issues.append(
            {
                "severity": "error",
                "code": "duplicate_node_id",
                "node_id": node_id,
                "message": f"节点 ID 重复：{node_id}",
            }
        )
    edge_ids = [str(edge.get("id") or "") for edge in edges]
    for edge_id, count in Counter(edge_ids).items():
        if count > 1:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_edge_id",
                    "edge_id": edge_id,
                    "message": f"连接 ID 重复：{edge_id}",
                }
            )

    node_map = {str(node.get("id") or ""): node for node in nodes}
    indegree = {node_id: 0 for node_id in node_map}
    adjacency: dict[str, list[str]] = defaultdict(list)
    target_connections: Counter[tuple[str, str]] = Counter()
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        edge_id = str(edge.get("id") or "")
        if source == target:
            issues.append(
                {
                    "severity": "error",
                    "code": "self_loop",
                    "edge_id": edge_id,
                    "message": "DAG 不允许节点连接自身",
                }
            )
            continue
        if source not in node_map or target not in node_map:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_edge_endpoint",
                    "edge_id": edge_id,
                    "message": "连接引用了不存在的节点",
                }
            )
            continue
        adjacency[source].append(target)
        indegree[target] += 1
        outgoing[source].append(edge)
        source_catalog = CATALOG_BY_TYPE[node_map[source]["type"]]
        target_catalog = CATALOG_BY_TYPE[node_map[target]["type"]]
        source_port = _port(source_catalog, "outputs", edge.get("source_handle"))
        target_port = _port(target_catalog, "inputs", edge.get("target_handle"))
        if source_port is None or target_port is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_port",
                    "edge_id": edge_id,
                    "message": "连接使用了不存在的输入或输出端口",
                }
            )
            continue
        source_type = source_port["type"]
        target_type = target_port["type"]
        if source_type != "any" and target_type != "any" and source_type != target_type:
            issues.append(
                {
                    "severity": "error",
                    "code": "incompatible_ports",
                    "edge_id": edge_id,
                    "message": f"端口类型不兼容：{source_type} → {target_type}",
                }
            )
        target_key = (target, target_port["id"])
        target_connections[target_key] += 1
        if target_connections[target_key] > 1 and not target_port.get("multiple"):
            issues.append(
                {
                    "severity": "error",
                    "code": "multiple_single_input_edges",
                    "edge_id": edge_id,
                    "message": f"节点“{node_map[target]['label']}”的该输入端口只允许一个来源",
                }
            )

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(node_map):
        issues.append(
            {
                "severity": "error",
                "code": "pipeline_cycle",
                "message": "Pipeline 包含有向环；首版只支持 DAG",
            }
        )

    for node in nodes:
        node_type = node["type"]
        config = node.get("config") or {}
        if node_type == "data" and execution_bindings:
            asset_type = str(config.get("asset_type") or "")
            asset_version_id = str(config.get("asset_version_id") or "")
            if not asset_type or not asset_version_id:
                issues.append(
                    {
                        "severity": "error",
                        "code": "missing_data_reference",
                        "node_id": node["id"],
                        "message": f"节点“{node['label']}”必须选择精确数据版本",
                    }
                )
            elif asset_descriptor(session, asset_type, asset_version_id) is None:
                issues.append(
                    {
                        "severity": "error",
                        "code": "data_reference_not_found",
                        "node_id": node["id"],
                        "message": f"节点“{node['label']}”引用的数据版本不存在",
                    }
                )
            elif not project_has_asset_access(session, project_id, asset_type, asset_version_id):
                issues.append(
                    {
                        "severity": "error",
                        "code": "data_reference_forbidden",
                        "node_id": node["id"],
                        "message": f"当前项目无权消费节点“{node['label']}”的数据版本",
                    }
                )
        elif node_type in TASK_NODE_TYPES and execution_bindings:
            from .native_pipeline_nodes import is_domain_node, validate_node
            if (node.get("config") or {}).get("execution_mode") == "suite":
                from .suite_packages import validate_canvas_node
                issues.extend(validate_canvas_node(session, node, edges))
            elif is_domain_node(node) or node_type in {"preparation", "translation", "fact_translation", "knowledge"}:
                issues.extend(validate_node(node, edges))
            else:
                issues.extend(_command_issues(node))
        elif node_type == "branch":
            handles = {str(edge.get("source_handle") or "") for edge in outgoing[node["id"]]}
            if len(handles & {"true", "false", "default"}) < 2:
                issues.append(
                    {
                        "severity": "error",
                        "code": "incomplete_branch",
                        "node_id": node["id"],
                        "message": f"分支节点“{node['label']}”至少需要两个不同出口",
                    }
                )
            if not str(config.get("path") or ""):
                issues.append(
                    {
                        "severity": "warning",
                        "code": "branch_uses_upstream_result",
                        "node_id": node["id"],
                        "message": f"分支节点“{node['label']}”未设置路径，将直接判断首个上游输出",
                    }
                )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def _validation_report_view(
    report: dict[str, Any] | None,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(report or {})
    errors = value.get("errors") if isinstance(value.get("errors"), list) else []
    warnings = value.get("warnings") if isinstance(value.get("warnings"), list) else []
    return {
        **value,
        "valid": value.get("valid") if isinstance(value.get("valid"), bool) else not errors,
        "errors": errors,
        "warnings": warnings,
        "node_count": value.get("node_count") if isinstance(value.get("node_count"), int) else len(nodes),
        "edge_count": value.get("edge_count") if isinstance(value.get("edge_count"), int) else len(edges),
    }


def _revision_view(revision: ProjectPipelineRevision | None) -> dict[str, Any] | None:
    if revision is None:
        return None
    graph = revision_graph(revision)
    return {
        **revision.to_dict(),
        **graph,
        "validation_report": _validation_report_view(
            revision.validation_report, graph["nodes"], graph["edges"]
        ),
        "executable_graph": bool(revision.nodes),
    }


def _latest_run(session: Session, pipeline_id: str) -> ProjectPipelineRun | None:
    return session.scalar(
        select(ProjectPipelineRun)
        .where(ProjectPipelineRun.pipeline_id == pipeline_id)
        .order_by(ProjectPipelineRun.created_at.desc())
        .limit(1)
    )


def _latest_instance(
    session: Session, pipeline_id: str
) -> ProjectPipelineInstance | None:
    return session.scalar(
        select(ProjectPipelineInstance)
        .where(
            ProjectPipelineInstance.pipeline_id == pipeline_id,
            ProjectPipelineInstance.status != "archived",
        )
        .order_by(ProjectPipelineInstance.created_at.desc())
        .limit(1)
    )


def pipeline_definition_view(session: Session, pipeline: ProjectPipeline) -> dict[str, Any]:
    current = (
        session.get(ProjectPipelineRevision, pipeline.current_revision_id)
        if pipeline.current_revision_id
        else None
    )
    draft = (
        session.get(ProjectPipelineRevision, pipeline.draft_revision_id)
        if pipeline.draft_revision_id
        else None
    )
    revisions = session.scalars(
        select(ProjectPipelineRevision)
        .where(ProjectPipelineRevision.pipeline_id == pipeline.id)
        .order_by(ProjectPipelineRevision.revision.desc())
    ).all()
    latest = _latest_run(session, pipeline.id)
    latest_instance = _latest_instance(session, pipeline.id)
    instance_count = int(
        session.scalar(
            select(func.count(ProjectPipelineInstance.id)).where(
                ProjectPipelineInstance.pipeline_id == pipeline.id,
                ProjectPipelineInstance.status != "archived",
            )
        )
        or 0
    )
    display = draft or current
    graph = revision_graph(display)
    definition_kind = (
        "legacy_compatibility"
        if display is not None and not display.nodes and display.mapping_snapshot
        else "native"
    )
    return {
        **pipeline.to_dict(),
        "state": "draft" if draft is not None else "published" if current is not None else "empty",
        "definition_kind": definition_kind,
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "current_revision": _revision_view(current),
        "draft_revision": _revision_view(draft),
        "revisions": [_revision_view(item) for item in revisions],
        "latest_run": pipeline_run_summary_view(session, latest) if latest else None,
        "instance_count": instance_count,
        "latest_instance": pipeline_instance_view(session, latest_instance)
        if latest_instance
        else None,
    }


def list_pipeline_definitions(session: Session, project_id: str) -> list[dict[str, Any]]:
    pipelines = session.scalars(
        select(ProjectPipeline)
        .where(
            ProjectPipeline.project_id == project_id,
            ProjectPipeline.archived.is_(False),
        )
        .order_by(ProjectPipeline.updated_at.desc(), ProjectPipeline.created_at.desc())
    ).all()
    return [pipeline_definition_view(session, pipeline) for pipeline in pipelines]


def _next_revision(session: Session, pipeline_id: str) -> int:
    return int(
        session.scalar(
            select(func.coalesce(func.max(ProjectPipelineRevision.revision), 0)).where(
                ProjectPipelineRevision.pipeline_id == pipeline_id
            )
        )
        or 0
    ) + 1


def _payload_graph(
    payload: ProjectPipelineDefinitionCreate | ProjectPipelineDraftUpdate,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return (
        [item.model_dump() for item in payload.nodes],
        [item.model_dump() for item in payload.edges],
        deepcopy(payload.variables),
        deepcopy(payload.viewport),
    )


def create_pipeline_definition(
    session: Session,
    project_id: str,
    payload: ProjectPipelineDefinitionCreate,
) -> ProjectPipeline:
    nodes, edges, variables, viewport = _payload_graph(payload)
    pipeline = ProjectPipeline(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description.strip(),
        objective=payload.objective,
    )
    session.add(pipeline)
    try:
        session.flush()
    except IntegrityError as exc:
        raise PipelineDagError(
            "pipeline_name_exists", "项目内已存在同名 Pipeline", status_code=409
        ) from exc
    report = validate_pipeline_graph(
        session, project_id, nodes, edges, execution_bindings=False
    )
    revision = ProjectPipelineRevision(
        pipeline_id=pipeline.id,
        revision=1,
        status="draft",
        nodes=nodes,
        edges=edges,
        variables=variables,
        viewport=viewport,
        validation_report=report,
        locked=False,
    )
    session.add(revision)
    session.flush()
    pipeline.draft_revision_id = revision.id
    return pipeline


def save_pipeline_draft(
    session: Session,
    pipeline: ProjectPipeline,
    payload: ProjectPipelineDraftUpdate,
) -> ProjectPipelineRevision:
    duplicate = session.scalar(
        select(ProjectPipeline).where(
            ProjectPipeline.project_id == pipeline.project_id,
            ProjectPipeline.name == payload.name.strip(),
            ProjectPipeline.id != pipeline.id,
        )
    )
    if duplicate is not None:
        raise PipelineDagError(
            "pipeline_name_exists", "项目内已存在同名 Pipeline", status_code=409
        )
    nodes, edges, variables, viewport = _payload_graph(payload)
    draft = (
        session.get(ProjectPipelineRevision, pipeline.draft_revision_id)
        if pipeline.draft_revision_id
        else None
    )
    if draft is None:
        draft = ProjectPipelineRevision(
            pipeline_id=pipeline.id,
            revision=_next_revision(session, pipeline.id),
            status="draft",
            locked=False,
        )
        session.add(draft)
        session.flush()
        pipeline.draft_revision_id = draft.id
    if draft.status != "draft" or draft.locked:
        raise PipelineDagError("draft_locked", "当前草稿已经发布，不能继续修改", status_code=409)
    pipeline.name = payload.name.strip()
    pipeline.description = payload.description.strip()
    pipeline.objective = payload.objective
    draft.nodes = nodes
    draft.edges = edges
    draft.variables = variables
    draft.viewport = viewport
    draft.validation_report = validate_pipeline_graph(
        session, pipeline.project_id, nodes, edges, execution_bindings=False
    )
    draft.content_hash = None
    return draft


def validate_pipeline_definition(
    session: Session, pipeline: ProjectPipeline
) -> dict[str, Any]:
    revision_id = pipeline.draft_revision_id or pipeline.current_revision_id
    revision = session.get(ProjectPipelineRevision, revision_id) if revision_id else None
    if revision is None:
        raise PipelineDagError("pipeline_revision_missing", "Pipeline 没有可验证的修订版")
    graph = revision_graph(revision)
    report = validate_pipeline_graph(
        session,
        pipeline.project_id,
        graph["nodes"],
        graph["edges"],
        execution_bindings=False,
    )
    if revision.status == "draft":
        revision.validation_report = report
    return {"revision_id": revision.id, **report}


def publish_pipeline_definition(
    session: Session, pipeline: ProjectPipeline
) -> ProjectPipelineRevision:
    draft = (
        session.get(ProjectPipelineRevision, pipeline.draft_revision_id)
        if pipeline.draft_revision_id
        else None
    )
    if draft is None:
        raise PipelineDagError("pipeline_draft_missing", "没有可以发布的草稿", status_code=409)
    report = validate_pipeline_graph(
        session,
        pipeline.project_id,
        draft.nodes or [],
        draft.edges or [],
        execution_bindings=False,
    )
    draft.validation_report = report
    if not report["valid"]:
        raise PipelineDagError(
            "pipeline_validation_failed",
            f"Pipeline 仍有 {len(report['errors'])} 个错误，不能发布",
            status_code=409,
        )
    draft.status = "published"
    draft.locked = True
    draft.content_hash = graph_content_hash(draft.nodes or [], draft.edges or [], draft.variables or {})
    draft.published_at = utcnow()
    pipeline.current_revision_id = draft.id
    pipeline.draft_revision_id = None
    return draft


def _instance_graph_payload(
    payload: ProjectPipelineInstanceDraftUpdate,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return (
        [item.model_dump() for item in payload.nodes],
        [item.model_dump() for item in payload.edges],
        deepcopy(payload.variables),
        deepcopy(payload.viewport),
    )


def _instance_revision(
    session: Session, instance: ProjectPipelineInstance
) -> ProjectPipelineRevision:
    revision = session.get(
        ProjectPipelineRevision, instance.definition_revision_id
    )
    if revision is None:
        raise PipelineDagError(
            "pipeline_revision_missing", "DAG 实例引用的定义修订不存在", status_code=409
        )
    return revision


def _instance_name_available(
    session: Session,
    project_id: str,
    name: str,
    *,
    exclude_id: str | None = None,
) -> bool:
    query = select(ProjectPipelineInstance.id).where(
        ProjectPipelineInstance.project_id == project_id,
        ProjectPipelineInstance.name == name,
    )
    if exclude_id:
        query = query.where(ProjectPipelineInstance.id != exclude_id)
    return session.scalar(query) is None


def create_pipeline_instance(
    session: Session,
    project_id: str,
    payload: ProjectPipelineInstanceCreate,
) -> ProjectPipelineInstance:
    revision = session.get(ProjectPipelineRevision, payload.definition_revision_id)
    pipeline = (
        session.get(ProjectPipeline, revision.pipeline_id) if revision is not None else None
    )
    if (
        revision is None
        or pipeline is None
        or pipeline.project_id != project_id
        or pipeline.archived
        or revision.status != "published"
        or not revision.locked
    ):
        raise PipelineDagError(
            "published_definition_required",
            "只能从当前项目的已发布 Pipeline 定义修订创建实例",
            status_code=409,
        )
    graph = revision_graph(revision)
    if not revision.nodes:
        raise PipelineDagError(
            "legacy_pipeline_not_instantiable",
            "既有固定产线只能查看；请先在 Pipeline 编辑器中创建并发布 DAG 修订版",
            status_code=409,
        )
    name = payload.name.strip()
    if not _instance_name_available(session, project_id, name):
        raise PipelineDagError(
            "pipeline_instance_name_exists", "项目内已存在同名 DAG 实例", status_code=409
        )
    report = validate_pipeline_graph(
        session,
        project_id,
        graph["nodes"],
        graph["edges"],
        execution_bindings=True,
    )
    instance = ProjectPipelineInstance(
        project_id=project_id,
        pipeline_id=pipeline.id,
        definition_revision_id=revision.id,
        name=name,
        description=payload.description.strip(),
        status="draft",
        nodes=graph["nodes"],
        edges=graph["edges"],
        variables=graph["variables"],
        viewport=graph["viewport"],
        validation_report=report,
        locked=False,
    )
    session.add(instance)
    session.flush()
    return instance


def save_pipeline_instance_draft(
    session: Session,
    instance: ProjectPipelineInstance,
    payload: ProjectPipelineInstanceDraftUpdate,
) -> ProjectPipelineInstance:
    if instance.status != "draft" or instance.locked:
        raise PipelineDagError(
            "pipeline_instance_locked",
            "已激活的 DAG 实例不可修改；请从定义创建新实例",
            status_code=409,
        )
    name = payload.name.strip()
    if not _instance_name_available(
        session, instance.project_id, name, exclude_id=instance.id
    ):
        raise PipelineDagError(
            "pipeline_instance_name_exists", "项目内已存在同名 DAG 实例", status_code=409
        )
    nodes, edges, variables, viewport = _instance_graph_payload(payload)
    instance.name = name
    instance.description = payload.description.strip()
    instance.nodes = nodes
    instance.edges = edges
    instance.variables = variables
    instance.viewport = viewport
    instance.validation_report = validate_pipeline_graph(
        session,
        instance.project_id,
        nodes,
        edges,
        execution_bindings=True,
    )
    instance.content_hash = None
    return instance


def validate_pipeline_instance(
    session: Session, instance: ProjectPipelineInstance
) -> dict[str, Any]:
    report = validate_pipeline_graph(
        session,
        instance.project_id,
        instance.nodes or [],
        instance.edges or [],
        execution_bindings=True,
    )
    if instance.status == "draft":
        instance.validation_report = report
    return {"instance_id": instance.id, **report}


def activate_pipeline_instance(
    session: Session, instance: ProjectPipelineInstance
) -> ProjectPipelineInstance:
    if instance.status == "ready" and instance.locked:
        return instance
    if instance.status != "draft" or instance.locked:
        raise PipelineDagError(
            "pipeline_instance_locked", "当前 DAG 实例不能激活", status_code=409
        )
    revision = _instance_revision(session, instance)
    if revision.status != "published" or not revision.locked:
        raise PipelineDagError(
            "published_definition_required", "实例引用的定义修订不再是已发布版本", status_code=409
        )
    report = validate_pipeline_graph(
        session,
        instance.project_id,
        instance.nodes or [],
        instance.edges or [],
        execution_bindings=True,
    )
    instance.validation_report = report
    if not report["valid"]:
        raise PipelineDagError(
            "pipeline_instance_validation_failed",
            f"DAG 实例仍有 {len(report['errors'])} 个错误，不能激活",
            status_code=409,
        )
    instance.status = "ready"
    instance.locked = True
    instance.content_hash = graph_content_hash(
        instance.nodes or [], instance.edges or [], instance.variables or {}
    )
    instance.activated_at = utcnow()
    return instance


def pipeline_instance_view(
    session: Session, instance: ProjectPipelineInstance
) -> dict[str, Any]:
    pipeline = session.get(ProjectPipeline, instance.pipeline_id) if instance.pipeline_id else None
    revision = session.get(ProjectPipelineRevision, instance.definition_revision_id) if instance.definition_revision_id else None
    latest = session.scalar(
        select(ProjectPipelineRun)
        .where(ProjectPipelineRun.instance_id == instance.id)
        .order_by(ProjectPipelineRun.created_at.desc())
        .limit(1)
    )
    run_count = int(
        session.scalar(
            select(func.count(ProjectPipelineRun.id)).where(
                ProjectPipelineRun.instance_id == instance.id
            )
        )
        or 0
    )
    imported = is_imported_pipeline_instance(instance)
    metadata = _instance_metadata(instance)
    display_graph = {
        "nodes": deepcopy(instance.nodes or []),
        "edges": deepcopy(instance.edges or []),
        "variables": deepcopy(instance.variables or {}),
        "viewport": deepcopy(instance.viewport or {}),
    }
    if imported:
        projected_graph = revision_graph(revision)
        if projected_graph["nodes"]:
            display_graph = {
                **projected_graph,
                "variables": deepcopy(instance.variables or {}),
            }
    return {
        **instance.to_dict(),
        **display_graph,
        "validation_report": _validation_report_view(
            instance.validation_report, display_graph["nodes"], display_graph["edges"]
        ),
        "pipeline_name": pipeline.name if pipeline else "",
        "instance_kind": IMPORTED_INSTANCE_KIND if imported else NATIVE_INSTANCE_KIND,
        "execution_owner": metadata.get("execution_owner")
        or (DOMAIN_TASK_OWNER if imported else PIPELINE_SCHEDULER_OWNER),
        "source_refs": {
            "training_run_id": metadata.get("source_training_run_id"),
            "task_id": metadata.get("source_task_id"),
        }
        if imported
        else {},
        "can_run": not imported and instance.status == "ready" and instance.locked,
        "definition_revision": _revision_view(revision),
        "node_count": len(display_graph["nodes"]),
        "edge_count": len(display_graph["edges"]),
        "run_count": run_count,
        "latest_run": pipeline_run_summary_view(session, latest) if latest else None,
    }


def list_pipeline_instances(
    session: Session, project_id: str
) -> list[dict[str, Any]]:
    instances = session.scalars(
        select(ProjectPipelineInstance)
        .where(
            ProjectPipelineInstance.project_id == project_id,
            ProjectPipelineInstance.status != "archived",
        )
        .order_by(
            ProjectPipelineInstance.updated_at.desc(),
            ProjectPipelineInstance.created_at.desc(),
        )
    ).all()
    return [pipeline_instance_view(session, instance) for instance in instances]


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in [item for item in path.split(".") if item]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _condition(config: dict[str, Any], context: dict[str, Any]) -> tuple[bool, Any]:
    path = str(config.get("path") or "")
    actual = _get_path(context, path) if path else next(iter(context.get("upstream", {}).values()), {})
    operator = str(config.get("operator") or "truthy")
    expected = config.get("value")
    try:
        if operator == "truthy":
            return bool(actual), actual
        if operator == "falsy":
            return not bool(actual), actual
        if operator == "equals":
            return actual == expected, actual
        if operator == "not_equals":
            return actual != expected, actual
        if operator == "contains":
            return expected in actual, actual
        if operator == "gt":
            return actual > expected, actual
        if operator == "gte":
            return actual >= expected, actual
        if operator == "lt":
            return actual < expected, actual
        if operator == "lte":
            return actual <= expected, actual
    except (TypeError, ValueError):
        return False, actual
    return False, actual


def _node_context(
    run: ProjectPipelineRun,
    incoming: list[dict[str, Any]],
    node_runs: dict[str, ProjectPipelineNodeRun],
) -> dict[str, Any]:
    upstream = {
        edge["source"]: deepcopy(node_runs[edge["source"]].output_snapshot or {})
        for edge in incoming
        if edge["source"] in node_runs
        and node_runs[edge["source"]].status == "succeeded"
    }
    return {"inputs": deepcopy(run.input_snapshot or {}), "upstream": upstream}


def _task_for_node(
    session: Session,
    run: ProjectPipelineRun,
    node: dict[str, Any],
    node_run: ProjectPipelineNodeRun,
    context: dict[str, Any],
) -> None:
    config = deepcopy(node.get("config") or {})
    if config.get("execution_mode") == "suite":
        from .suite_packages import canvas_task
        canvas_task(session, run, node, node_run, context)
        node_run.started_at = utcnow()
        return
    command = deepcopy(config.get("command") or {})
    command_env = dict(command.get("env") or {})
    command_env.update(
        {
            "LLM_PLATFORM_PIPELINE_RUN_ID": run.id,
            "LLM_PLATFORM_PIPELINE_INSTANCE_ID": run.instance_id,
            "LLM_PLATFORM_PIPELINE_NODE_ID": node_run.node_id,
        }
    )
    command["env"] = command_env
    task = create_task(
        session,
        project_id=run.project_id,
        kind="pipeline_node",
        title=f"Pipeline · {node_run.label}",
        adapter_id=str(config.get("adapter_id") or DEFAULT_ADAPTERS[node_run.node_type]),
        launcher_id=str(config.get("launcher_id") or "builtin.local-process"),
        payload={
            "pipeline_run_id": run.id,
            "pipeline_instance_id": run.instance_id,
            "pipeline_node_run_id": node_run.id,
            "pipeline_node_id": node_run.node_id,
            "pipeline_context": context,
            **({"code": deepcopy(config["code_version"])} if config.get("code_version") else {}),
            "resource_request": deepcopy(config.get("resource_request") or {}),
            "command": command,
        },
    )
    node_run.task_id = task.id
    node_run.status = "queued"
    node_run.input_snapshot = context
    node_run.started_at = utcnow()


def _active_incoming_edges(
    incoming: list[dict[str, Any]],
    node_runs: dict[str, ProjectPipelineNodeRun],
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for edge in incoming:
        source_run = node_runs[edge["source"]]
        if source_run.status == "skipped":
            continue
        if source_run.node_type == "branch" and source_run.status == "succeeded":
            selected = source_run.branch_decision or "default"
            if str(edge.get("source_handle") or "default") != selected:
                continue
        active.append(edge)
    return active


def _update_run_summary(
    run: ProjectPipelineRun, node_runs: list[ProjectPipelineNodeRun]
) -> None:
    scope = (run.context_snapshot or {}).get("__training_instance", {}).get("execution_scope") or {}
    scoped_ids = set(scope.get("selected_node_ids", []))
    counted = [item for item in node_runs if item.node_id in scoped_ids] if scoped_ids else node_runs
    counts = Counter(item.status for item in counted)
    total = len(counted)
    completed = sum(counts[state] for state in TERMINAL_NODE_STATUSES)
    run.node_summary = {
        "total": total,
        "completed": completed,
        "progress": round(completed / total, 4) if total else 1.0,
        "counts": dict(counts),
        **(
            {
                "reused": len(scope.get("reused_node_ids", [])),
                "excluded": len(scope.get("excluded_node_ids", [])),
            }
            if scope else {}
        ),
    }


def advance_pipeline_run(session: Session, run: ProjectPipelineRun) -> None:
    from .native_pipeline_nodes import advance_domain_node, is_domain_node
    from .training_instances import META, advance_native_node, native

    if (run.context_snapshot or {}).get(META) and run.status == "failed":
        return
    if run.status in {"succeeded", "cancelled", "cancelling"}:
        return
    graph = deepcopy(run.graph_snapshot or {})
    if not graph.get("nodes"):
        revision = session.get(ProjectPipelineRevision, run.revision_id)
        graph = revision_graph(revision)
    if not graph.get("nodes"):
        run.status = "failed"
        run.error = {
            "code": "pipeline_graph_snapshot_missing",
            "message": "运行冻结的 DAG 图快照不存在",
        }
        run.ended_at = utcnow()
        return
    node_defs = {node["id"]: node for node in graph.get("nodes") or []}
    edges = list(graph.get("edges") or [])
    incoming_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming_by_node[edge["target"]].append(edge)

    made_progress = True
    while made_progress and run.status not in {"cancelled", "cancelling"}:
        made_progress = False
        rows = session.scalars(
            select(ProjectPipelineNodeRun).where(
                ProjectPipelineNodeRun.pipeline_run_id == run.id
            )
        ).all()
        node_runs = {item.node_id: item for item in rows}
        for node_id, node_run in node_runs.items():
            if node_run.status != "waiting":
                continue
            incoming = incoming_by_node[node_id]
            active_incoming = _active_incoming_edges(incoming, node_runs)
            if incoming and not active_incoming:
                if all(node_runs[edge["source"]].status in TERMINAL_NODE_STATUSES for edge in incoming):
                    node_run.status = "skipped"
                    node_run.ended_at = utcnow()
                    made_progress = True
                continue
            source_runs = [node_runs[edge["source"]] for edge in active_incoming]
            if any(item.status in {"failed", "cancelled"} for item in source_runs):
                node_run.status = "cancelled"
                node_run.error = {
                    "code": "upstream_failed",
                    "message": "上游节点失败或取消，当前节点未启动",
                }
                node_run.ended_at = utcnow()
                made_progress = True
                continue
            if any(item.status not in {"succeeded", "skipped"} for item in source_runs):
                continue
            node = node_defs[node_id]
            context = _node_context(run, active_incoming, node_runs)
            node_run.input_snapshot = {**(node_run.input_snapshot or {}), **context}
            node_run.started_at = node_run.started_at or utcnow()
            node_type = node_run.node_type
            config = node.get("config") or {}
            if node_type == "data":
                if config.get("wait_until_ready") and config.get("asset_type") == "dataset_version":
                    from ..models import DatasetVersion
                    version = session.get(DatasetVersion, config.get("asset_version_id"))
                    if version is None or version.state in {"failed", "cancelled"}:
                        node_run.status = "failed"
                        node_run.error = {"code": "source_dataset_unavailable", "message": "输入数据版本不存在或导入失败"}
                        node_run.ended_at = utcnow()
                        made_progress = True
                        continue
                    if version.state != "ready":
                        continue
                descriptor = asset_descriptor(
                    session,
                    str(config.get("asset_type") or ""),
                    str(config.get("asset_version_id") or ""),
                )
                node_run.output_snapshot = {
                    "asset_type": config.get("asset_type"),
                    "asset_version_id": config.get("asset_version_id"),
                    "dataset_version_id": config.get("asset_version_id") if config.get("asset_type") == "dataset_version" else None,
                    "descriptor": descriptor or {},
                }
                node_run.status = "succeeded"
                node_run.ended_at = utcnow()
            elif node_type == "check":
                passed, actual = _condition(config, context)
                node_run.output_snapshot = {
                    "passed": passed,
                    "actual": actual,
                    "operator": config.get("operator") or "truthy",
                    "expected": config.get("value"),
                    "severity": config.get("severity") or "warning",
                }
                if not passed and config.get("failure_policy") == "fail":
                    node_run.status = "failed"
                    node_run.error = {
                        "code": "pipeline_check_failed",
                        "message": f"检查节点“{node_run.label}”未通过",
                    }
                else:
                    node_run.status = "succeeded"
                node_run.ended_at = utcnow()
            elif node_type == "branch":
                passed, actual = _condition(config, context)
                decision = str(
                    config.get("true_handle" if passed else "false_handle")
                    or ("true" if passed else "false")
                )
                node_run.branch_decision = decision
                node_run.output_snapshot = {
                    "passed": passed,
                    "actual": actual,
                    "selected_handle": decision,
                }
                node_run.status = "succeeded"
                node_run.ended_at = utcnow()
            elif node_type == "join":
                node_run.output_snapshot = {"upstream": context["upstream"]}
                node_run.status = "succeeded"
                node_run.ended_at = utcnow()
            else:
                try:
                    if is_domain_node(node):
                        with session.begin_nested():
                            advance_domain_node(session, run, node, node_run, context)
                    elif native(node):
                        with session.begin_nested():
                            if not advance_native_node(session, run, node, node_run, context):
                                continue
                    else:
                        _task_for_node(session, run, node, node_run, context)
                except (ValueError, HTTPException) as exc:
                    node_run.status = "failed"
                    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                    node_run.error = {"code": "pipeline_task_create_failed", "message": detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)}
                    node_run.ended_at = utcnow()
            made_progress = True

    rows = session.scalars(
        select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.pipeline_run_id == run.id)
    ).all()
    _update_run_summary(run, rows)
    failed_rows = [item for item in rows if item.status == "failed"]
    cancelled_rows = [item for item in rows if item.status == "cancelled"]
    active_rows = [item for item in rows if item.status in {"waiting", "queued", "running", "paused"}]
    if failed_rows or cancelled_rows:
        failed_node = (failed_rows or cancelled_rows)[0]
        run.error = deepcopy(failed_node.error or {}) or {
            "code": "pipeline_node_failed",
            "message": f"节点“{failed_node.label}”执行失败",
        }
        if active_rows:
            run.status = "running"
            run.ended_at = None
        else:
            run.status = "failed"
            run.ended_at = utcnow()
    elif rows and all(item.status in {"succeeded", "skipped"} for item in rows):
        run.status = "succeeded"
        run.ended_at = utcnow()
    elif active_rows:
        run.status = "running"
    record_terminal_run_lineage(
        session,
        run,
        list(rows),
        execution_owner=PIPELINE_SCHEDULER_OWNER,
    )


def start_pipeline_run(
    session: Session,
    instance: ProjectPipelineInstance,
    inputs: dict[str, Any] | None = None,
) -> ProjectPipelineRun:
    if is_imported_pipeline_instance(instance):
        raise PipelineDagError(
            "imported_pipeline_read_only",
            "兼容接入实例引用现有 TrainingRun/Task，不能由 DAG 调度器重新运行",
            status_code=409,
        )
    pipeline = session.get(ProjectPipeline, instance.pipeline_id)
    revision = session.get(
        ProjectPipelineRevision, instance.definition_revision_id
    )
    if (
        pipeline is None
        or revision is None
        or pipeline.project_id != instance.project_id
        or revision.pipeline_id != pipeline.id
    ):
        raise PipelineDagError(
            "pipeline_instance_invalid",
            "DAG 实例与 Pipeline 定义关系无效",
            status_code=409,
        )
    if instance.status != "ready" or not instance.locked:
        raise PipelineDagError(
            "pipeline_instance_not_ready",
            "请先完成实例配置、校验并激活 DAG 实例",
            status_code=409,
        )
    report = validate_pipeline_graph(
        session,
        pipeline.project_id,
        instance.nodes or [],
        instance.edges or [],
        execution_bindings=True,
    )
    if not report["valid"]:
        raise PipelineDagError(
            "pipeline_instance_validation_failed",
            "DAG 实例的当前精确引用已不可执行",
            status_code=409,
        )
    graph_snapshot = {
        "nodes": deepcopy(instance.nodes or []),
        "edges": deepcopy(instance.edges or []),
        "variables": deepcopy(instance.variables or {}),
        "viewport": deepcopy(instance.viewport or {}),
    }
    run = ProjectPipelineRun(
        project_id=pipeline.project_id,
        pipeline_id=pipeline.id,
        revision_id=revision.id,
        instance_id=instance.id,
        status="running",
        trigger_kind="manual",
        input_snapshot=deepcopy(inputs or {}),
        context_snapshot={
            "pipeline_name": pipeline.name,
            "pipeline_instance_name": instance.name,
            "pipeline_instance_content_hash": instance.content_hash,
            "revision": revision.revision,
            "content_hash": revision.content_hash,
        },
        graph_snapshot=graph_snapshot,
        started_at=utcnow(),
    )
    session.add(run)
    session.flush()
    for node in graph_snapshot["nodes"]:
        session.add(
            ProjectPipelineNodeRun(
                pipeline_run_id=run.id,
                node_id=node["id"],
                node_type=node["type"],
                label=node["label"],
                status="waiting",
                config_snapshot=deepcopy(node.get("config") or {}),
            )
        )
    session.flush()
    advance_pipeline_run(session, run)
    return run


def _imported_node_status(
    node_id: str,
    lifecycle_status: str,
    task: Task | None,
    *,
    source_terminal: bool,
) -> str:
    if node_id == "training":
        return _task_node_status(task)
    if lifecycle_status == "ready":
        return "succeeded"
    if lifecycle_status == "in_progress":
        return "running"
    if lifecycle_status == "paused":
        return "paused"
    if lifecycle_status in {"failed", "blocked"}:
        return "failed"
    if lifecycle_status in {"stopped", "cancelled"}:
        return "cancelled"
    return "skipped" if source_terminal else "waiting"


def _imported_run_projection(
    session: Session,
    run: ProjectPipelineRun,
    node_runs: list[ProjectPipelineNodeRun],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _run_metadata(run)
    training_run_id = str(metadata.get("source_training_run_id") or "")
    training_run = session.get(TrainingRun, training_run_id) if training_run_id else None
    task_id = str(metadata.get("source_task_id") or "")
    if not task_id and training_run is not None:
        task_id = training_run.task_id
    task = session.get(Task, task_id) if task_id else None
    pipeline = session.get(ProjectPipeline, run.pipeline_id)
    revision = session.get(ProjectPipelineRevision, run.revision_id)
    lifecycle = project_pipeline_view(session, pipeline) if pipeline is not None else None
    lifecycle_nodes = {
        str(item.get("key") or ""): item
        for item in (lifecycle or {}).get("nodes", [])
    }
    run_status = _task_pipeline_status(task)
    source_terminal = run_status in {"succeeded", "failed", "cancelled"}
    projected_nodes: list[dict[str, Any]] = []
    for item in node_runs:
        lifecycle_node = lifecycle_nodes.get(item.node_id, {})
        status = _imported_node_status(
            item.node_id,
            str(lifecycle_node.get("status") or "not_started"),
            task,
            source_terminal=source_terminal,
        )
        error: dict[str, Any] = {}
        if item.node_id == "training" and status in {"failed", "cancelled"}:
            error = deepcopy(task.error or {}) if task else {}
        elif status == "failed":
            error = {
                "code": "imported_pipeline_node_blocked",
                "message": str(lifecycle_node.get("detail") or "现有产线节点失败或阻断"),
            }
        output_snapshot = {
            **deepcopy(item.output_snapshot or {}),
            "source": "existing_project_pipeline",
            "detail": lifecycle_node.get("detail"),
            "versions": deepcopy(lifecycle_node.get("versions") or []),
            "metrics": deepcopy(lifecycle_node.get("metrics") or {}),
        }
        started_at: Any = item.started_at
        ended_at: Any = item.ended_at
        if item.node_id == "training" and task is not None:
            started_at = task.started_at or task.created_at
            ended_at = task.ended_at if status in TERMINAL_NODE_STATUSES else None
        elif status in TERMINAL_NODE_STATUSES:
            ended_at = lifecycle_node.get("updated_at") or item.ended_at
        projected_nodes.append(
            {
                **item.to_dict(),
                "status": status,
                "task_id": task.id if item.node_id == "training" and task else item.task_id,
                "input_snapshot": {
                    **deepcopy(item.input_snapshot or {}),
                    "legacy_mapping": deepcopy(
                        (run.input_snapshot or {}).get("legacy_mapping") or {}
                    ),
                },
                "output_snapshot": output_snapshot,
                "error": error,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )

    counts = Counter(item["status"] for item in projected_nodes)
    total = len(projected_nodes)
    completed = sum(counts[state] for state in TERMINAL_NODE_STATUSES)
    weighted_completed = float(completed)
    training_node = next(
        (item for item in projected_nodes if item["node_id"] == "training"), None
    )
    if training_node and training_node["status"] not in TERMINAL_NODE_STATUSES and task:
        weighted_completed += max(0.0, min(1.0, float(task.progress or 0)))
    summary = {
        "total": total,
        "completed": completed,
        "progress": round(weighted_completed / total, 4) if total else 1.0,
        "counts": dict(counts),
        "source_progress": round(float(task.progress or 0), 4) if task else 0,
    }
    error = deepcopy(task.error or {}) if task and run_status == "failed" else {}
    if run_status == "failed" and not error:
        error = {
            "code": "imported_source_task_failed",
            "message": "现有产线关联的 Task 执行失败",
        }
    persisted_graph = deepcopy(run.graph_snapshot or {})
    projected_graph = revision_graph(revision)
    if projected_graph["nodes"]:
        projected_graph["variables"] = deepcopy(persisted_graph.get("variables") or {})
    else:
        projected_graph = persisted_graph
    return (
        {
            **run.to_dict(),
            "graph_snapshot": projected_graph,
            "status": run_status,
            "run_kind": IMPORTED_INSTANCE_KIND,
            "execution_owner": DOMAIN_TASK_OWNER,
            "source_refs": {
                "training_run_id": training_run.id if training_run else training_run_id or None,
                "task_id": task.id if task else task_id or None,
            },
            "node_summary": summary,
            "error": error,
            "started_at": (task.started_at or task.created_at) if task else run.started_at,
            "ended_at": task.ended_at if task and source_terminal else None,
        },
        projected_nodes,
    )


def pipeline_run_summary_view(
    session: Session, run: ProjectPipelineRun
) -> dict[str, Any]:
    if not is_imported_pipeline_run(run):
        return {
            **run.to_dict(),
            "run_kind": NATIVE_INSTANCE_KIND,
            "execution_owner": PIPELINE_SCHEDULER_OWNER,
            "source_refs": {},
        }
    node_runs = session.scalars(
        select(ProjectPipelineNodeRun)
        .where(ProjectPipelineNodeRun.pipeline_run_id == run.id)
        .order_by(ProjectPipelineNodeRun.created_at.asc())
    ).all()
    projected, _ = _imported_run_projection(session, run, list(node_runs))
    return projected


def pipeline_run_view(session: Session, run: ProjectPipelineRun) -> dict[str, Any]:
    pipeline = session.get(ProjectPipeline, run.pipeline_id) if run.pipeline_id else None
    revision = session.get(ProjectPipelineRevision, run.revision_id) if run.revision_id else None
    instance = session.get(ProjectPipelineInstance, run.instance_id)
    node_runs = session.scalars(
        select(ProjectPipelineNodeRun)
        .where(ProjectPipelineNodeRun.pipeline_run_id == run.id)
        .order_by(ProjectPipelineNodeRun.created_at.asc())
    ).all()
    imported = is_imported_pipeline_run(run)
    projected_run: dict[str, Any]
    projected_nodes: list[dict[str, Any]]
    if imported:
        projected_run, projected_nodes = _imported_run_projection(
            session, run, list(node_runs)
        )
    else:
        projected_run = pipeline_run_summary_view(session, run)
        projected_nodes = [item.to_dict() for item in node_runs]
    lineage_by_node_run = node_lineage_views(session, run)
    lineage = run_lineage_view(session, run)
    task_map = {
        item.task_id: session.get(Task, item.task_id)
        for item in node_runs
        if item.task_id is not None
    }
    if imported:
        source_task_id = (projected_run.get("source_refs") or {}).get("task_id")
        if source_task_id and source_task_id not in task_map:
            task_map[source_task_id] = session.get(Task, source_task_id)
    return {
        **projected_run,
        "pipeline_name": pipeline.name if pipeline else "",
        "instance_name": instance.name if instance else "",
        "instance": pipeline_instance_view(session, instance) if instance else None,
        "revision": _revision_view(revision),
        "node_runs": [
            {
                **item,
                "lineage": lineage_by_node_run.get(str(item.get("id") or "")),
                "task": (item.get("input_snapshot") or {}).get("task_snapshot") or task_map[item.get("task_id")].to_dict()
                if item.get("task_id") and task_map.get(item.get("task_id"))
                else None,
            }
            for item in projected_nodes
        ],
        "lineage": lineage,
    }


def list_pipeline_runs(
    session: Session, project_id: str, *, instance_id: str | None = None
) -> list[dict[str, Any]]:
    query = select(ProjectPipelineRun).where(
        ProjectPipelineRun.project_id == project_id
    )
    if instance_id:
        query = query.where(ProjectPipelineRun.instance_id == instance_id)
    runs = session.scalars(
        query.order_by(ProjectPipelineRun.created_at.desc())
    ).all()
    return [pipeline_run_view(session, run) for run in runs]


def sync_pipeline_task_status(session: Session, task: Task) -> None:
    node_run = session.scalar(
        select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.task_id == task.id).order_by(ProjectPipelineNodeRun.created_at.desc()).limit(1)
    )
    if node_run is None:
        return
    run = session.get(ProjectPipelineRun, node_run.pipeline_run_id)
    if run is None:
        return
    from .training_instances import META, node_output

    unified = bool((run.context_snapshot or {}).get(META))
    if unified and (run.status in {"succeeded", "failed", "cancelled"} or ((node_run.error or {}).get("code") == "lease_expired" and task.observed_state == TaskStatus.RUNNING)):
        latest_id = session.scalar(select(ProjectPipelineRun.id).where(ProjectPipelineRun.instance_id == run.instance_id).order_by(ProjectPipelineRun.created_at.desc()).limit(1))
        if not (run.status in {"failed", "running"} and latest_id == run.id and (node_run.error or {}).get("code") == "lease_expired" and task.observed_state == TaskStatus.RUNNING and (node_run.input_snapshot or {}).get("task_attempt_id") == task.active_attempt_id):
            return
        # A late heartbeat proved the same process is still alive. This corrects
        # a provisional lease loss before any new execution has been created.
        run.status, run.ended_at, run.error = "running", None, {}
        rows = session.scalars(select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.pipeline_run_id == run.id)).all()
        for row in rows:
            if row.id == node_run.id or (row.error or {}).get("code") == "upstream_failed":
                row.status = "running" if row.id == node_run.id else "waiting"
                row.error, row.ended_at = {}, None
                row.input_snapshot = {k: v for k, v in (row.input_snapshot or {}).items() if k not in {"task_snapshot", "task_attempt_id"}}
    if is_imported_pipeline_run(run):
        node_run.status = _task_node_status(task)
        node_run.started_at = task.started_at or task.created_at
        node_run.ended_at = (
            task.ended_at if node_run.status in TERMINAL_NODE_STATUSES else None
        )
        node_run.error = (
            deepcopy(task.error or {})
            if node_run.status in {"failed", "cancelled"}
            else {}
        )
        run.status = _task_pipeline_status(task, persisted=True)
        run.started_at = task.started_at or task.created_at
        run.ended_at = (
            task.ended_at
            if run.status in {"succeeded", "failed", "cancelled"}
            else None
        )
        run.error = (
            deepcopy(task.error or {}) if run.status == "failed" else {}
        )
        rows = list(
            session.scalars(
                select(ProjectPipelineNodeRun).where(
                    ProjectPipelineNodeRun.pipeline_run_id == run.id
                )
            ).all()
        )
        _, projected_nodes = _imported_run_projection(session, run, rows)
        record_terminal_run_lineage(
            session,
            run,
            rows,
            projected_nodes=projected_nodes,
            execution_owner=DOMAIN_TASK_OWNER,
        )
        return
    state = task.observed_state
    if state in {TaskStatus.QUEUED, TaskStatus.ASSIGNED, TaskStatus.STARTING}:
        node_run.status = "queued"
    elif state in {TaskStatus.RUNNING, TaskStatus.PAUSING, TaskStatus.STOPPING}:
        node_run.status = "running"
        node_run.started_at = node_run.started_at or task.started_at or utcnow()
    elif state == TaskStatus.PAUSED:
        node_run.status = "paused"
    elif state == TaskStatus.SUCCEEDED:
        node_run.status = "succeeded"
        node_run.output_snapshot = node_output(session, task, node_run) if unified else deepcopy(task.result or {})
        node_run.error = {}
        node_run.ended_at = task.ended_at or utcnow()
    elif state in {TaskStatus.FAILED, TaskStatus.LOST}:
        node_run.status = "failed"
        node_run.error = deepcopy(task.error or {})
        node_run.ended_at = task.ended_at or utcnow()
        run.error = node_run.error or {
            "code": "pipeline_node_failed",
            "message": f"节点“{node_run.label}”执行失败",
        }
    elif state == TaskStatus.CANCELLED:
        node_run.status = "cancelled"
        node_run.error = deepcopy(task.error or {})
        node_run.ended_at = task.ended_at or utcnow()
        if run.status != "cancelling":
            run.error = {
                "code": "pipeline_node_cancelled",
                "message": f"节点“{node_run.label}”已取消",
            }
    if unified and node_run.status in TERMINAL_NODE_STATUSES:
        if state != TaskStatus.SUCCEEDED:
            node_run.output_snapshot = node_output(session, task, node_run)
        node_run.input_snapshot = {**(node_run.input_snapshot or {}), "task_attempt_id": task.active_attempt_id, "task_snapshot": deepcopy(task.to_dict())}
    if run.status == "cancelling":
        rows = session.scalars(
            select(ProjectPipelineNodeRun).where(
                ProjectPipelineNodeRun.pipeline_run_id == run.id
            )
        ).all()
        if all(item.status in TERMINAL_NODE_STATUSES for item in rows):
            run.status = "cancelled"
            run.ended_at = utcnow()
            _update_run_summary(run, rows)
        record_terminal_run_lineage(
            session,
            run,
            list(rows),
            execution_owner=PIPELINE_SCHEDULER_OWNER,
        )
        return
    if run.status != "cancelled":
        advance_pipeline_run(session, run)


def cancel_pipeline_run(session: Session, run: ProjectPipelineRun, *, finish_running: bool = False) -> ProjectPipelineRun:
    if is_imported_pipeline_run(run):
        raise PipelineDagError(
            "imported_pipeline_read_only",
            "兼容接入运行由原 TrainingRun/Task 管理；请从训练中心或任务入口控制原任务",
            status_code=409,
        )
    if run.status in {"succeeded", "failed", "cancelled"}:
        raise PipelineDagError(
            "pipeline_run_terminal", "终态 Pipeline 运行不能再次取消", status_code=409
        )
    run.status = "cancelling"
    rows = session.scalars(
        select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.pipeline_run_id == run.id)
    ).all()
    for node_run in rows:
        if node_run.status == "waiting":
            node_run.status = "cancelled"
            node_run.ended_at = utcnow()
            continue
        if not node_run.task_id or node_run.status in TERMINAL_NODE_STATUSES:
            continue
        task = session.get(Task, node_run.task_id)
        if task is None:
            node_run.status = "cancelled"
            node_run.ended_at = utcnow()
            continue
        current = task.observed_state
        if finish_running and current in {TaskStatus.ASSIGNED, TaskStatus.STARTING, TaskStatus.RUNNING}:
            continue
        task.desired_state = TaskStatus.CANCELLED
        if current in {TaskStatus.DRAFT, TaskStatus.QUEUED, TaskStatus.PAUSED}:
            transition_task(session, task, TaskStatus.CANCELLED, event_type="task.cancelled")
            sync_pipeline_task_status(session, task)
        elif current in {
            TaskStatus.ASSIGNED,
            TaskStatus.STARTING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSING,
        }:
            if current == TaskStatus.ASSIGNED:
                transition_task(
                    session, task, TaskStatus.STARTING, event_type="task.stop_requested"
                )
            transition_task(session, task, TaskStatus.STOPPING, event_type="task.stop_requested")
    rows = session.scalars(
        select(ProjectPipelineNodeRun).where(ProjectPipelineNodeRun.pipeline_run_id == run.id)
    ).all()
    if all(item.status in TERMINAL_NODE_STATUSES for item in rows):
        run.status = "cancelled"
        run.ended_at = utcnow()
    _update_run_summary(run, rows)
    record_terminal_run_lineage(
        session,
        run,
        list(rows),
        execution_owner=PIPELINE_SCHEDULER_OWNER,
    )
    return run
