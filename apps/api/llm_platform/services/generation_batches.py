"""Compile bounded generation batches to the existing native DAG protocol."""

from copy import deepcopy

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .. import models as m
from ..routers.data_generation import _continuation_binding
from .global_data import project_has_asset_access


class InterleavedGenerationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=140)
    batch_records: int = Field(default=100000, ge=1, le=1000000)
    checkpoint_source_task_id: str | None = None


def compile_interleaved(session, instance, request, *, preview=False):
    from llm_platform_core.data_generation import DataGenerationCreate, FactTranslationCreate

    nodes = instance.nodes or []
    translation = [n for n in nodes if n.get("config", {}).get("native_operation") == "data.fact_translation"]
    knowledge = [n for n in nodes if n.get("config", {}).get("native_operation") == "data.knowledge"]
    if len(nodes) != 2 or len(translation) != 1 or len(knowledge) != 1:
        raise HTTPException(422, "分批交错入口要求仅含三元组翻译 → 知识生成两个原生节点")
    translation, knowledge = translation[0], knowledge[0]
    binding = knowledge["config"].get("input_bindings", {}).get("input_dataset_version_id", {})
    edges = instance.edges or []
    if (binding != {"node_id": translation["id"], "output_path": "dataset_version_id"}
            or len(edges) != 1 or edges[0]["source"] != translation["id"] or edges[0]["target"] != knowledge["id"]
            or translation["config"].get("input_bindings")
            or set(knowledge["config"].get("input_bindings", {})) != {"input_dataset_version_id"}):
        raise HTTPException(422, "只支持直接绑定翻译输出的两阶段流程")
    translate = FactTranslationCreate.model_validate(translation["config"]["parameters"])
    generate = DataGenerationCreate.model_validate({**knowledge["config"]["parameters"],
                                                   "input_dataset_version_id": translate.input_dataset_version_id})
    if translate.record_range or generate.record_range or translate.checkpoint_source_task_id or generate.checkpoint_source_task_id:
        raise HTTPException(422, "请选择尚未分批的完整两阶段实例")
    if generate.engine != "vllm" or generate.model_weight_version_id != translate.model_weight_version_id:
        raise HTTPException(422, "交错执行要求两个节点使用同一 vLLM 模型")
    source = session.get(m.DatasetVersion, translate.input_dataset_version_id)
    if not source or not project_has_asset_access(session, instance.project_id, "dataset_version", source.id):
        raise HTTPException(422, "来源数据不存在或尚未绑定到项目")
    files = (source.source_snapshot or {}).get("files", [])
    if source.state != "ready" or not source.train_ready or len(files) != 1 or not source.document_count:
        raise HTTPException(422, "首版分批交错要求单文件、已分析且就绪的文本快照")
    if not str(files[0]["path"]).lower().endswith((".parquet", ".jsonl", ".ndjson")):
        raise HTTPException(422, "分批交错支持 Parquet 或 JSONL 快照")
    total = source.document_count
    count = (total + request.batch_records - 1) // request.batch_records
    # Two nodes per batch plus two collectors must respect the existing 200-node limit.
    if count > 99:
        raise HTTPException(422, f"最多 99 批；每批至少 {(total + 98) // 99} 条，才能符合现有 DAG 的 200 节点上限")
    checkpoint = None
    if request.checkpoint_source_task_id:
        task = session.get(m.Task, request.checkpoint_source_task_id)
        if not task or task.project_id != instance.project_id or task.payload.get("pipeline_instance_id") != instance.id:
            raise HTTPException(422, "接续任务必须来自所选原实例")
        checkpoint = session.scalar(select(m.TaskAttempt).where(m.TaskAttempt.task_id == task.id)
                                    .order_by(m.TaskAttempt.attempt_number.desc()).limit(1))
        if not checkpoint:
            raise HTTPException(409, "来源任务没有可接续的尝试")
        translate = translate.model_copy(update={"checkpoint_source_task_id": task.id,
                                                 "checkpoint_source_attempt_id": checkpoint.id})
        if not preview:
            _continuation_binding(session, instance.project_id, translate, "fact_translation")
    order = list(range(count))
    if checkpoint:
        # Alternate old prefix and unfinished tail; scheduling order never implies coverage.
        order = []
        left, right = 0, count - 1
        while left <= right:
            order.append(left)
            if right != left:
                order.append(right)
            left, right = left + 1, right - 1
    result_nodes, result_edges = [], []
    groups = {"translate": [], "generate": []}
    previous = None
    for position, index in enumerate(order):
        start, stop = index * request.batch_records, min(total, (index + 1) * request.batch_records)
        tid, gid = f"translate-{index:04d}", f"generate-{index:04d}"
        tp = translate.model_dump()
        tp.update(name=f"{request.name} · 中文三元组", record_range={"logical_file": files[0]["path"], "start": start, "stop": stop, "total": total})
        gp = generate.model_dump(exclude={"input_dataset_version_id"})
        gp["name"] = f"{request.name} · 中文正文"
        for template, identity, parameters, group, x in (
            (translation, tid, tp, "translate", 0), (knowledge, gid, gp, "generate", 360),
        ):
            config = deepcopy(template["config"])
            config.update(parameters=parameters, output_retention="retain",
                          generation_batch={"index": index, "count": count, "source_start": start, "source_stop": stop})
            config.pop("input_bindings", None)
            if group == "generate":
                config["input_bindings"] = {"input_dataset_version_id": {"node_id": tid, "output_path": "dataset_version_id"}}
            result_nodes.append({"id": identity, "type": template["type"], "label": f"第 {index + 1}/{count} 批 · {template['label']}",
                                 "position": {"x": x, "y": position * 160}, "config": config})
            groups[group].append(identity)
        if previous:
            result_edges.append({"id": f"next-{index}", "source": previous, "target": tid})
        result_edges.append({"id": f"pair-{index}", "source": tid, "target": gid})
        previous = gid
    for group, ids in groups.items():
        identity = f"collect-{group}"
        result_nodes.append({"id": identity, "type": "mixture", "label": "汇集中文三元组" if group == "translate" else "汇集中文正文",
                             "position": {"x": 0 if group == "translate" else 360, "y": count * 160},
                             "config": {"execution_mode": "native", "native_operation": "data.generation_collection", "output_retention": "retain",
                                        "parameters": {"name": f"{request.name} · {group} 全部批次", "entries": [{} for _ in ids]},
                                        "input_bindings": {f"entries.{i}.dataset_version_id": {"node_id": node_id, "output_path": "dataset_version_id"} for i, node_id in enumerate(ids)}}})
        result_edges.extend({"id": f"collect-{node_id}", "source": node_id, "target": identity} for node_id in ids)
    return {"name": request.name, "description": "原生三元组翻译与正文分批交错；每批独立提交，可暂停恢复。",
            "cloned_from_instance_id": instance.id, "nodes": result_nodes, "edges": result_edges,
            "variables": {"generation_interleave": {"schema_version": 1, "batch_records": request.batch_records,
                "source_records": total, "batch_count": count, "source_instance_id": instance.id,
                "checkpoint_source_task_id": request.checkpoint_source_task_id,
                "checkpoint_source_attempt_id": checkpoint.id if checkpoint else None,
                "order": order}}}
