"""Resolve immutable node outputs used at the boundary of a partial execution."""

from copy import deepcopy

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as m
from .global_data import ASSET_TYPES, asset_record, project_has_asset_access
from .lineage import _collect_refs


def execution_scope(
    session: Session,
    instance: m.ProjectPipelineInstance,
    *,
    selection_mode: str = "all",
    node_ids: list[str] | None = None,
    source_attempt_id: str | None = None,
) -> dict:
    """Read-only preview; the caller repeats this under its instance lock."""
    nodes = {node["id"]: node for node in instance.nodes}
    requested = list(node_ids or [])
    errors: list[dict] = []

    def error(code: str, message: str, node_id: str | None = None):
        errors.append({"code": code, "message": message, "node_id": node_id})

    if selection_mode not in {"all", "from_node", "selected"}:
        error("invalid_execution_scope", "执行范围类型无效")
    if len(requested) != len(set(requested)):
        error("duplicate_execution_nodes", "不能重复选择节点")
    for node_id in requested:
        if node_id not in nodes:
            error("execution_node_missing", "所选节点不属于当前实例", node_id)
    if selection_mode == "all" and (requested or source_attempt_id):
        error("ambiguous_execution_scope", "全量执行不接受节点选择或历史输出来源")
    if selection_mode == "from_node" and len(requested) != 1:
        error("execution_start_required", "请选择一个起始节点")
    if selection_mode == "selected" and not requested:
        error("execution_nodes_required", "请至少选择一个执行节点")

    selected = set(nodes) if selection_mode == "all" else set(requested) & nodes.keys()
    if selection_mode == "from_node":
        frontier = list(selected)
        children: dict[str, list[str]] = {}
        for edge in instance.edges:
            children.setdefault(edge["source"], []).append(edge["target"])
        while frontier:
            for child in children.get(frontier.pop(), []):
                if child not in selected:
                    selected.add(child)
                    frontier.append(child)

    boundary = {
        edge["source"]
        for edge in instance.edges
        if edge["target"] in selected and edge["source"] not in selected
    }
    source = session.get(m.ProjectPipelineRun, source_attempt_id) if source_attempt_id else None
    valid_source = source is not None and source.instance_id == instance.id
    if source_attempt_id and not valid_source:
        error("execution_source_missing", "来源执行不存在或不属于当前实例")
    elif valid_source and source.status not in {"succeeded", "failed", "cancelled"}:
        error("execution_source_active", "只能复用已结束执行中的冻结结果")
        valid_source = False
    rows = (
        {
            row.node_id: row
            for row in session.scalars(
                select(m.ProjectPipelineNodeRun).where(
                    m.ProjectPipelineNodeRun.pipeline_run_id == source.id
                )
            ).all()
        }
        if valid_source
        else {}
    )
    reused = []
    for node_id in nodes:
        if node_id not in boundary:
            continue
        previous = rows.get(node_id)
        if (
            previous is None
            or previous.status not in {"succeeded", "skipped"}
            or (previous.input_snapshot or {}).get("excluded_from_execution")
        ):
            error(
                "execution_dependency_unavailable",
                f"上游节点“{nodes[node_id]['label']}”缺少可复用结果；请选择包含该结果的历史执行，或将其加入执行范围",
                node_id,
            )
            continue
        for ref in _collect_refs(previous.output_snapshot):
            artifact_id = None
            if ref["asset_type"] == "artifact":
                artifact_id = ref["asset_id"]
            elif ref["asset_type"] in ASSET_TYPES:
                record = asset_record(session, ref["asset_type"], ref["asset_id"])
                if record is None or not project_has_asset_access(
                    session, instance.project_id, ref["asset_type"], ref["asset_id"]
                ):
                    error(
                        "execution_asset_unavailable",
                        f"上游节点“{nodes[node_id]['label']}”引用的资产版本已不可用",
                        node_id,
                    )
                elif record is not None:
                    artifact_id = getattr(record, "artifact_id", None) or getattr(
                        record, "manifest_artifact_id", None
                    )
            elif ref["asset_type"] in {"checkpoint", "model_version"}:
                model = m.Checkpoint if ref["asset_type"] == "checkpoint" else m.ModelVersion
                record = session.get(model, ref["asset_id"])
                if (
                    record is None
                    or record.project_id != instance.project_id
                    or record.status in {"deleted", "missing", "archived", "failed"}
                ):
                    error(
                        "execution_asset_unavailable",
                        f"上游节点“{nodes[node_id]['label']}”引用的模型或 Checkpoint 已不可用",
                        node_id,
                    )
                elif record is not None:
                    artifact_id = record.artifact_id
            if artifact_id:
                artifact = session.get(m.Artifact, artifact_id)
                if artifact is None or artifact.availability_state != "available":
                    error(
                        "execution_artifact_unavailable",
                        f"上游节点“{nodes[node_id]['label']}”引用的制品已不可用",
                        node_id,
                    )
        reused.append(
            {
                "node_id": node_id,
                "label": nodes[node_id]["label"],
                "status": previous.status,
                "source_attempt_id": source.id,
                "source_node_run_id": previous.id,
                "outputs": deepcopy(previous.output_snapshot),
            }
        )
    excluded = set(nodes) - selected - boundary
    return {
        "valid": not errors,
        "selection_mode": selection_mode,
        "node_ids": requested,
        "source_attempt_id": source_attempt_id,
        "selected_node_ids": [node_id for node_id in nodes if node_id in selected],
        "reused_node_ids": [node_id for node_id in nodes if node_id in boundary],
        "excluded_node_ids": [node_id for node_id in nodes if node_id in excluded],
        "reused_nodes": reused,
        "errors": errors,
        "warnings": [
            "执行范围内的节点仍遵守分支条件；范围外的依赖使用所选历史结果，不消费本次重新生成的中间结果。"
        ]
        if selection_mode != "all"
        else [],
    }
