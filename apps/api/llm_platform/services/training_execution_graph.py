"""Artifact → execution → artifact read model, including recorded workflow steps."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

from sqlalchemy import select

from .. import models as m
from .global_data import ASSET_TYPES, is_global_data_project, project_has_asset_access
from .lineage import _database_asset_ref
from .training_data_lineage import layered_data_edges
from .training_downstream_lineage import downstream_facts
from .training_history import (
    frozen_training_payload,
    history_run,
    source_run_id,
    spec_from_run,
    status,
)
from .training_upstream_lineage import (
    ASSETS,
    LABELS,
    _domain_source,
    _source_details,
    upstream_lineage,
    version_refs,
)

REF_MODELS = {
    **ASSETS,
    "code_version": m.CodeVersion,
    "tokenizer_version": m.TokenizerVersion,
    "artifact": m.Artifact,
    "model_specification_revision": m.ModelSpecificationRevision,
    "evaluation_run": m.EvaluationRun,
}


def refs(value, container="", supervised=False):
    return set(version_refs(
        value, container=container, supervised=supervised, asset_types=REF_MODELS
    ))


def recorded_tasks(value):
    """Only structured IDs; command strings and logs cannot manufacture a DAG."""
    if isinstance(value, list):
        return [item for child in value for item in recorded_tasks(child)]
    if not isinstance(value, dict):
        return []
    result = [value] if isinstance(value.get("task_id"), str) else []
    for key, child in value.items():
        if key not in {"command", "command_result", "logs"}:
            result.extend(recorded_tasks(child))
    return result


def task_access(session, project_id, task):
    return task.project_id == project_id or is_global_data_project(
        session.get(m.Project, task.project_id)
    )


def task_row(session, task, node_id, attempt_id=None, *, context_only=False, before=None):
    attempt = session.get(m.TaskAttempt, attempt_id) if attempt_id else None
    if attempt and attempt.task_id != task.id:
        attempt = None
    if not attempt:
        query = select(m.TaskAttempt).where(m.TaskAttempt.task_id == task.id)
        if before:
            query = query.where(m.TaskAttempt.created_at <= before)
        attempt = session.scalar(query.order_by(m.TaskAttempt.attempt_number.desc()).limit(1))
    training = session.scalar(select(m.TrainingRun).where(m.TrainingRun.task_id == task.id))
    kind = {
        "train": "training",
        "data_process": "clean",
        "data_convert": "transform",
        "data_import": "data",
        "data_mixture": "mixture",
        "mixture_resolve": "mixture",
        "evaluate": "evaluation",
        "evaluation": "evaluation",
    }.get(task.kind, "custom")
    config = {
        "adapter_id": task.adapter_id,
        "launcher_id": task.launcher_id,
        "resource_request": deepcopy((task.payload or {}).get("resource_request") or {}),
        "command": deepcopy((task.payload or {}).get("command") or {}),
        "_recorded_step": True,
        "_context_only": context_only,
    }
    if training:
        kind = "training"
        config.update(
            {
                "execution_mode": "native",
                "training_spec": spec_from_run(training),
                "training_run_id": training.id,
            }
        )
    observed = status(attempt.status if attempt else task.observed_state)
    task_view = task.to_dict()
    task_view["observed_state"] = observed
    if training:
        task_view["payload"] = frozen_training_payload(training, task)
    return {
        "id": f"recorded:{node_id}",
        "node_id": node_id,
        "node_type": kind,
        "label": task.title,
        "status": observed,
        "task_id": task.id,
        "task": task_view,
        "config_snapshot": config,
        "input_snapshot": {
            "task_attempt_id": attempt.id if attempt else None,
            "training_run_id": training.id if training else None,
        },
        "output_snapshot": deepcopy(task.result or {}),
        "error": deepcopy(attempt.error if attempt else task.error or {}),
        "started_at": attempt.started_at.isoformat() if attempt and attempt.started_at else None,
        "ended_at": attempt.ended_at.isoformat() if attempt and attempt.ended_at else None,
    }


def output_refs(session, task, row, *, whole_record=False):
    result = set()
    attempt_id = row.get("input_snapshot", {}).get("task_attempt_id")
    artifacts = (
        session.scalars(
            select(m.Artifact).where(m.Artifact.created_by_task_attempt_id == attempt_id)
        ).all()
        if attempt_id
        else []
    )
    artifact_ids = {artifact.id for artifact in artifacts}
    training = session.scalar(select(m.TrainingRun).where(m.TrainingRun.task_id == task.id))
    if training:
        checkpoints = session.scalars(
            select(m.Checkpoint).where(m.Checkpoint.training_run_id == training.id)
        ).all()
        for checkpoint in checkpoints:
            if whole_record or not attempt_id or checkpoint.artifact_id in artifact_ids:
                result.add(("checkpoint", checkpoint.id))
                result.update(
                    ("model_version", identity)
                    for identity in session.scalars(
                        select(m.ModelVersion.id).where(
                            m.ModelVersion.source_checkpoint_id == checkpoint.id
                        )
                    ).all()
                )
    else:
        for kind, model in REF_MODELS.items():
            if hasattr(model, "task_id") and kind not in {"checkpoint", "artifact"}:
                result.update(
                    (kind, identity)
                    for identity in session.scalars(
                        select(model.id).where(model.task_id == task.id)
                    ).all()
                )
        for model in (m.DataProcessingRun, m.DataConversionRun):
            for record in session.scalars(select(model).where(model.task_id == task.id)).all():
                if record.output_dataset_version_id:
                    result.add(("dataset_version", record.output_dataset_version_id))
        result.update(
            ("artifact", artifact.id)
            for artifact in artifacts
            if artifact.kind not in {"runtime_metrics", "task_report", "logs"}
        )
    return result


def execution_graph(
    session,
    instance,
    attempt_view=None,
    *,
    include_upstream=True,
    max_depth=None,
    center_node_id=None,
):
    graph = deepcopy(
        attempt_view["graph_snapshot"]
        if attempt_view
        else {
            "nodes": instance.nodes,
            "edges": instance.edges,
            "variables": instance.variables,
            "viewport": instance.viewport,
        }
    )
    rows = deepcopy(attempt_view.get("node_runs", []) if attempt_view else [])
    training = history_run(session, instance)
    if training and not rows:
        task = session.get(m.Task, training.task_id)
        rows = [task_row(session, task, "training")] if task else []
        graph["nodes"] = [
            {
                "id": "training",
                "type": "training",
                "label": instance.name,
                "description": "",
                "config": {"training_spec": spec_from_run(training)},
                "position": {"x": 360, "y": 100},
            }
        ]
        graph["edges"] = []
    if not rows:
        rows = [
            {
                "id": node["id"],
                "node_id": node["id"],
                "node_type": node["type"],
                "label": node["label"],
                "config_snapshot": node.get("config", {}),
                "input_snapshot": {},
                "output_snapshot": {},
                "status": "waiting",
                "task_id": None,
            }
            for node in graph["nodes"]
        ]
    warnings = []
    # Expand reported domain steps, while retaining the actual orchestration task.
    known_tasks = {row.get("task_id") for row in rows}
    for parent in list(rows):
        task = session.get(m.Task, parent["task_id"]) if parent.get("task_id") else None
        payload = (
            parent.get("output_snapshot")
            or ((parent.get("task") or {}).get("result") if not attempt_view else {})
            or {}
        )
        for record in recorded_tasks(payload.get("workflow") or {}):
            task_id = record["task_id"]
            if task_id in known_tasks:
                continue
            child = session.get(m.Task, task_id)
            if not child or not task_access(session, instance.project_id, child):
                continue
            # The result explicitly references this task. Older executions are context.
            historical = bool(task and child.created_at < task.created_at)
            row = task_row(
                session,
                child,
                f"task:{child.id}",
                record.get("task_attempt_id"),
                context_only=historical,
                before=task.ended_at if task else None,
            )
            rows.append(row)
            graph["nodes"].append(
                {
                    "id": row["node_id"],
                    "type": row["node_type"],
                    "label": row["label"],
                    "description": "历史引用" if historical else "工作流登记的执行步骤",
                    "config": row["config_snapshot"],
                    "position": {"x": 0, "y": 0},
                }
            )
            known_tasks.add(task_id)
            if len(rows) >= 200:
                warnings.append("登记的执行步骤超过展示上限。")
                break
    assets, edges, consumed, produced = {}, {}, {}, {}
    node_ids = {node["id"] for node in graph["nodes"]}
    if center_node_id and center_node_id not in node_ids:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="当前实例中没有该中心节点。")
    centers = (
        [center_node_id]
        if center_node_id
        else [
            row["node_id"]
            for row in rows
            if not row.get("config_snapshot", {}).get("_context_only")
            and not row.get("input_snapshot", {}).get("excluded_from_execution")
            and not row.get("input_snapshot", {}).get("reused_from_node_run_id")
        ]
    )
    centers = centers or list(node_ids)

    def asset(kind, identity):
        key = (kind, identity)
        if key not in assets:
            model = REF_MODELS.get(kind)
            record = session.get(model, identity) if model else None
            if record is None:
                return None
            if kind == "code_version":
                owner = session.get(m.CodeSource, record.source_id)
                if not owner or owner.project_id != instance.project_id:
                    return None
            elif kind == "model_specification_revision":
                owner = session.get(m.ModelSpecification, record.specification_id)
                if not owner or owner.project_id != instance.project_id:
                    return None
            elif kind == "artifact":
                producer = (
                    session.get(m.TaskAttempt, record.created_by_task_attempt_id)
                    if record.created_by_task_attempt_id
                    else None
                )
                producer_task = session.get(m.Task, producer.task_id) if producer else None
                location = session.scalar(
                    select(m.ArtifactLocation.id)
                    .where(
                        m.ArtifactLocation.artifact_id == identity,
                        m.ArtifactLocation.project_id == instance.project_id,
                    )
                    .limit(1)
                )
                if not location and not (
                    producer_task and task_access(session, instance.project_id, producer_task)
                ):
                    return None
            elif kind in ASSET_TYPES:
                if not project_has_asset_access(session, instance.project_id, kind, identity):
                    return None
            elif getattr(record, "project_id", instance.project_id) != instance.project_id:
                return None
            ref = _database_asset_ref(session, kind, identity)
            label = ref["label"]
            if ref["version"] and f"V{ref['version']}" not in label:
                label = f"{label} · V{ref['version']}"
            if label.startswith(kind) and kind in LABELS:
                label = f"{LABELS[kind]} · {getattr(record, 'version', None) or identity[:8]}"
            if kind == "checkpoint":
                label = f"Checkpoint · Step {record.step}"
            if kind == "evaluation_run":
                label = f"EvalResult · {identity[:8]}"
            if kind == "artifact":
                label = f"{record.kind} · {identity[:8]}"
            assets[key] = {
                "id": f"version:{kind}:{identity}",
                "kind": "asset",
                "label": label,
                "asset_type": kind,
                "asset_id": identity,
                "version": ref["version"],
                "status": ref["status"],
                "depth": 0,
                "context_only": False,
                "read_only": True,
                "source": {},
                "note": "",
                "position": {"x": 0, "y": 0},
            }
        return assets[key]["id"]

    def edge(source, target, context_only=False, label=""):
        if source and target and source != target:
            identity = f"binding:{source}:{target}"
            edges[identity] = {
                "id": identity,
                "source": source,
                "target": target,
                "label": label,
                "context_only": context_only,
            }

    for row in rows:
        node_id = row["node_id"]
        task = session.get(m.Task, row["task_id"]) if row.get("task_id") else None
        frozen_payload = (
            (row.get("task") or {}).get("payload") or (task.payload if task else {}) or {}
        )
        inputs = refs(row.get("config_snapshot", {})) | refs(frozen_payload)
        outputs = output_refs(session, task, row, whole_record=not attempt_view) if task else set()
        consumed[node_id], produced[node_id] = inputs - outputs, outputs
        historical = row.get("config_snapshot", {}).get("_context_only", False)
        for kind, identity in sorted(inputs - outputs):
            edge(asset(kind, identity), node_id, historical, "输入")
        for kind, identity in sorted(outputs):
            edge(node_id, asset(kind, identity), historical, "产出")
    # Explicit domain derivations connect intermediary versions (e.g. Model from Checkpoint).
    for (kind, identity), item in list(assets.items()):
        source = (
            _domain_source(session, kind, session.get(REF_MODELS[kind], identity))
            if kind == "model_version"
            else None
        )
        if source:
            for predecessor in source["inputs"]:
                if predecessor in assets:
                    edge(assets[predecessor]["id"], item["id"], label="版本来源")
    # Keep original control dependencies, plus artifact flow. No order inferred from labels.
    for original in graph["edges"]:
        edge(original["source"], original["target"], label=original.get("label", ""))
    context_nodes = []
    truncated = False
    if include_upstream:
        projected_rows = [SimpleNamespace(**row) for row in rows]
        run = SimpleNamespace(
            id=attempt_view["id"] if attempt_view else instance.id,
            project_id=instance.project_id,
            graph_snapshot=graph,
        )
        context = upstream_lineage(
            session,
            run,
            max_depth=max_depth,
            max_nodes=1000 if max_depth is None else 300 if max_depth > 4 else 120,
            projected_rows=projected_rows,
        )
        remap = {}
        task_nodes = {row["task_id"]: row["node_id"] for row in rows if row.get("task_id")}
        for item in context["nodes"]:
            key = (item.get("asset_type"), item.get("asset_id"))
            if item["kind"] == "asset" and key in assets:
                remap[item["id"]] = assets[key]["id"]
            elif item["kind"] == "process" and item["source"].get("task_id") in task_nodes:
                remap[item["id"]] = task_nodes[item["source"]["task_id"]]
            else:
                context_nodes.append(item)
        for binding in context["edges"]:
            source, target = (
                remap.get(binding["source"], binding["source"]),
                remap.get(binding["target"], binding["target"]),
            )
            if f"binding:{source}:{target}" not in edges:
                edge(source, target, True, binding["label"])
        warnings.extend(context["warnings"])
        truncated = context["truncated"]
        # Only expand descendants of the current centers, not consumers of their
        # ancestors. Otherwise a common dataset would bring in unrelated runs.
        next_ids = {}
        for binding in edges.values():
            next_ids.setdefault(binding["source"], []).append(binding["target"])
        descendants, frontier = set(centers), list(centers)
        for current in frontier:
            for target in next_ids.get(current, []):
                if target not in descendants:
                    descendants.add(target)
                    frontier.append(target)
        canonical = {
            (item["asset_type"], item["asset_id"]): item["id"]
            for item in [*assets.values(), *context_nodes]
            if item["kind"] == "asset"
        }

        def process_key(source, fallback):
            return next(
                (
                    f"{key}:{source[key]}"
                    for key in ("node_run_id", "training_run_id", "task_id")
                    if source.get(key)
                ),
                fallback,
            )

        processes = {
            process_key(item["source"], item["id"].split(f"history:{run.id}:")[-1]): item["id"]
            for item in context_nodes
            if item["kind"] == "process"
        }
        for row in rows:
            source = {
                "training_run_id": row.get("input_snapshot", {}).get("training_run_id"),
                "task_id": row.get("task_id"),
            }
            processes[process_key(source, row["node_id"])] = row["node_id"]
        downstream_pending = deque(
            key for key, identity in canonical.items() if identity in descendants
        )
        downstream_seen = set()
        while downstream_pending:
            key = downstream_pending.popleft()
            if key in downstream_seen:
                continue
            downstream_seen.add(key)
            if len(assets) + len(context_nodes) + len(node_ids) >= 1000:
                truncated = True
                warnings.append("关联链路超过 1000 个节点，已展示部分链路。")
                break
            for fact in downstream_facts(session, instance.project_id, *key):
                outputs = [tuple(output) for output in fact["outputs"]]
                if outputs and all(
                    output in canonical and f"binding:{canonical[key]}:{canonical[output]}" in edges
                    for output in outputs
                ):
                    downstream_pending.extend(outputs)
                    continue
                process_identity = process_key(fact["source"], fact["id"])
                process_id = processes.get(process_identity)
                if not process_id:
                    process_id = f"history:{run.id}:{fact['id']}"
                    processes[process_identity] = process_id
                    context_nodes.append(
                        {
                            "id": process_id,
                            "kind": "process",
                            "label": fact["label"],
                            "status": fact["status"],
                            "source": _source_details(session, instance.project_id, fact["source"]),
                            "depth": 0,
                            "context_only": True,
                            "read_only": True,
                            "note": "关联下游执行，只读。",
                        }
                    )
                edge(canonical[key], process_id, True, "输入")
                for output_key in fact["outputs"]:
                    output_key = tuple(output_key)
                    output_id = canonical.get(output_key) or asset(*output_key)
                    if output_id:
                        canonical[output_key] = output_id
                        edge(process_id, output_id, True, "产出")
                        downstream_pending.append(output_key)
                if len(assets) + len(context_nodes) + len(node_ids) >= 1000:
                    truncated = True
                    warnings.append("关联链路超过 1000 个节点，已展示部分链路。")
                    break
    visible_assets = list(assets.values()) + context_nodes
    all_ids = node_ids | {item["id"] for item in visible_assets}
    outgoing = {identity: [] for identity in all_ids}
    indegree = {identity: 0 for identity in all_ids}
    accepted = {}
    for identity, binding in edges.items():
        source, target = binding["source"], binding["target"]
        if source not in all_ids or target not in all_ids:
            continue
        pending, seen = [target], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(outgoing.get(current, []))
        if source in seen:
            warnings.append("循环来源未连入执行图。")
            continue
        accepted[identity] = binding
        outgoing[source].append(target)
        indegree[target] += 1
    accepted = layered_data_edges(visible_assets, accepted)
    outgoing = {identity: [] for identity in all_ids}
    indegree = dict.fromkeys(all_ids, 0)
    for binding in accepted.values():
        outgoing[binding["source"]].append(binding["target"])
        indegree[binding["target"]] += 1
    # A removed shortcut must not leave an ancestor styled as a current input.
    current_ids = {
        row["node_id"] for row in rows if not row.get("config_snapshot", {}).get("_context_only")
    }
    for item in assets.values():
        connected = [e for e in accepted.values() if item["id"] in {e["source"], e["target"]}]
        item["context_only"] = bool(connected) and not any(
            (e["source"] in current_ids or e["target"] in current_ids) and not e["context_only"]
            for e in connected
        )
    # Walk the two directions separately: never include a sibling execution
    # merely because it consumes the same historical input as a center node.
    incoming = {}
    for binding in accepted.values():
        incoming.setdefault(binding["target"], []).append(binding["source"])

    def reachable(links):
        found, frontier = set(centers), list(centers)
        for identity in frontier:
            for target in links.get(identity, []):
                if target not in found:
                    found.add(target)
                    frontier.append(target)
        return found

    visible_ids = reachable(incoming) | reachable(outgoing)
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] in visible_ids]
    visible_assets = [node for node in visible_assets if node["id"] in visible_ids]
    rows = [row for row in rows if row["node_id"] in visible_ids]
    accepted = {
        identity: binding
        for identity, binding in accepted.items()
        if binding["source"] in visible_ids and binding["target"] in visible_ids
    }
    pending = sorted(identity for identity in all_ids if not indegree[identity])
    levels = dict.fromkeys(all_ids, 0)
    while pending:
        identity = pending.pop(0)
        for target in outgoing[identity]:
            levels[target] = max(levels[target], levels[identity] + 1)
            indegree[target] -= 1
            if not indegree[target]:
                pending.append(target)
    # Keep direct inputs next to their consumers; unrelated ancestor depths must
    # not stretch a single training step across the entire historical chain.
    for identity in sorted(all_ids, key=lambda key: levels[key], reverse=True):
        if outgoing[identity]:
            levels[identity] = min(levels[target] for target in outgoing[identity]) - 1
    slots = {}
    for item in [*graph["nodes"], *visible_assets]:
        level = levels[item["id"]]
        slot = slots.get(level, 0)
        slots[level] = slot + 1
        item["position"] = {"x": level * 360, "y": slot * 230}
    graph["edges"] = []  # All visible relationships use the unified binding edge set.
    return {
        "graph": graph,
        "node_runs": rows,
        "artifacts": visible_assets,
        "bindings": list(accepted.values()),
        "warnings": sorted(set(warnings)),
        "execution_node_ids": [node["id"] for node in instance.nodes]
        if not source_run_id(instance)
        else [],
        "expanded_step_count": len(rows),
        "center_node_ids": centers,
        "truncated": truncated,
    }
