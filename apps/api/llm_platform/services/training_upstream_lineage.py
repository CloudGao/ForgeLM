"""Read-only upstream context, deliberately separate from the schedulable DAG."""

from collections import deque

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models as m
from .global_data import ASSET_TYPES, is_global_data_project, project_has_asset_access
from .lineage import _config_refs, _database_asset_ref, _normalize_asset_ref
from .training_data_lineage import layered_data_edges

ASSETS = {
    "dataset_version": m.DatasetVersion,
    "mixture_version": m.DataMixtureVersion,
    "supervision_dataset_version": m.SupervisionDatasetVersion,
    "supervision_mixture_version": m.SupervisionMixtureVersion,
    "eval_dataset_version": m.EvalDatasetVersion,
    "model_version": m.ModelVersion,
    "checkpoint": m.Checkpoint,
    "model_weight_version": m.ModelWeightVersion,
    "training_data_view_version": m.TrainingDataViewVersion,
    "tokenized_corpus_version": m.TokenizedCorpusVersion,
    "tokenized_dataset_version": m.TokenizedDatasetVersion,
    "supervision_corpus_version": m.SupervisionCorpusVersion,
    "supervision_training_view_version": m.SupervisionTrainingViewVersion,
}
LABELS = {
    "dataset_version": "数据集",
    "mixture_version": "数据混合",
    "supervision_dataset_version": "监督数据",
    "supervision_mixture_version": "监督混合",
    "eval_dataset_version": "评测数据",
    "model_version": "模型",
    "checkpoint": "Checkpoint",
    "model_weight_version": "外部模型权重",
    "training_data_view_version": "训练读取索引",
    "tokenized_corpus_version": "编码 Corpus",
    "tokenized_dataset_version": "编码数据",
    "supervision_corpus_version": "监督 Corpus",
    "supervision_training_view_version": "监督训练 View",
}
VERSION_FIELDS = {
    "validation_mixture_version_id": "mixture_version",
    "input_dataset_version_id": "dataset_version",
    "source_checkpoint_id": "checkpoint",
    "reference_checkpoint_id": "checkpoint",
    "training_data_view_id": "training_data_view_version",
    "supervision_training_view_id": "supervision_training_view_version",
}
VERSION_CONTAINERS = {
    "code": "code_version",
    "tokenizer": "tokenizer_version",
    "model_specification": "model_specification_revision",
    "training_data_view_inputs": "training_data_view_version",
    "prefix_training_data_view": "training_data_view_version",
    "supervision_training_view_inputs": "supervision_training_view_version",
    "supervision_corpus": "supervision_corpus_version",
    "tokenized_corpora": "tokenized_corpus_version",
    "source_corpora": "tokenized_corpus_version",
    "corpora": "tokenized_corpus_version",
    "resume_checkpoint": "checkpoint",
    "reference_checkpoint": "checkpoint",
}


def version_refs(value, *, supervised=False, container="", asset_types=None):
    """Recognize explicit version bindings; never infer edges from names or dates."""
    asset_types = ASSETS if asset_types is None else asset_types
    fields = {f"{kind}_id": kind for kind in asset_types} | VERSION_FIELDS
    refs = []
    if isinstance(value, list):
        for item in value:
            refs.extend(version_refs(
                item, supervised=supervised, container=container, asset_types=asset_types
            ))
    elif isinstance(value, dict):
        supervised = (
            supervised
            or value.get("type") in {"sft", "dpo"}
            or container in {"supervision_corpus", "supervision_training_view_inputs"}
        )
        direct = _normalize_asset_ref(value)
        if direct and direct["asset_type"] in asset_types:
            refs.append((direct["asset_type"], direct["asset_id"]))
        container_kind = VERSION_CONTAINERS.get(container)
        if container_kind in asset_types:
            identity = (
                value.get("version_id") or value.get("revision_id")
                or value.get("checkpoint_id") or value.get("id")
            )
            if isinstance(identity, str) and identity:
                refs.append((container_kind, identity))
        for field, item in value.items():
            if field in {
                "metadata", "task_snapshot", "result", "command", "effective_config",
                "output_dataset_version_id", "output_artifact_id", "artifact_id", "artifact_ids", "artifacts",
            }:
                continue
            kind = fields.get(field)
            if field == "mixture_version_id" and supervised:
                kind = "supervision_mixture_version"
            if field == "data_ref_id":
                kind = value.get("data_ref_type") or (
                    "supervision_mixture_version" if supervised else "mixture_version"
                )
            if kind in asset_types and isinstance(item, str) and item:
                refs.append((kind, item))
            elif (
                field.endswith("_ids") and fields.get(field[:-1]) in asset_types
                and isinstance(item, list)
            ):
                refs.extend(
                    (fields[field[:-1]], identity) for identity in item
                    if isinstance(identity, str) and identity
                )
            elif isinstance(item, (dict, list)):
                refs.extend(version_refs(
                    item, supervised=supervised, container=field, asset_types=asset_types
                ))
    return list(dict.fromkeys(refs))


def _training_inputs(session, run):
    refs = version_refs(run.config_snapshot)
    if not refs:
        revision = session.get(m.PlanRevision, run.plan_revision_id)
        if revision:
            refs = version_refs(
                {"stages": revision.stages, "input_model_ref": revision.input_model_ref}
            )
    return refs


def _domain_source(session, kind, record):
    """Return only explicitly recorded producer facts and their version inputs."""
    inputs = []
    label = ""
    source = {"asset_type": kind, "asset_id": record.id}
    status = getattr(record, "status", "")
    task_id = getattr(record, "task_id", None)
    producer_id = f"version:{kind}:{record.id}"
    if kind == "dataset_version":
        if record.parent_version_id:
            inputs = [("dataset_version", record.parent_version_id)]
            label = "数据派生"
        for field, model, title in (
            ("conversion_run_id", m.DataConversionRun, "数据转换"),
            ("processing_run_id", m.DataProcessingRun, "数据清洗"),
        ):
            run_id = getattr(record, field)
            run = session.get(model, run_id) if run_id else None
            if run and run.output_dataset_version_id == record.id:
                inputs = [("dataset_version", run.input_dataset_version_id)]
                label, task_id, status = title, run.task_id, run.status
                source.update({"run_id": run.id, "recipe_version_id": run.recipe_version_id})
        if not label and task_id:
            label = "数据接入"
        if task_id:
            task = session.get(m.Task, task_id)
            if task and (task.payload or {}).get("dataset_version_id") == record.id:
                parents = version_refs((task.payload or {}).get("data_inputs", []))
                if parents:
                    inputs = list(dict.fromkeys(inputs + parents))
                    label = "数据派生"
    elif kind in {"mixture_version", "supervision_mixture_version"}:
        label = "数据混合" if kind == "mixture_version" else "监督数据混合"
        inputs = version_refs(record.entries, supervised=kind == "supervision_mixture_version")
    elif kind == "supervision_dataset_version":
        label, inputs = "监督数据映射", [("dataset_version", record.dataset_version_id)]
        inputs.extend(
            ("dataset_version", str(lookup.get("dataset_version_id")))
            for lookup in (record.mapping_config or {}).get("_lookups") or []
            if lookup.get("dataset_version_id")
        )
    elif kind == "model_version":
        label = "模型登记"
        if record.source_checkpoint_id:
            inputs = [("checkpoint", record.source_checkpoint_id)]
        elif record.model_weight_version_id:
            inputs = [("model_weight_version", record.model_weight_version_id)]
        elif record.parent_model_version_id:
            label, inputs = "模型派生", [("model_version", record.parent_model_version_id)]
        else:
            label = ""
    elif kind == "checkpoint":
        run = session.get(m.TrainingRun, record.training_run_id)
        if run:
            label, task_id, status = "模型训练", run.task_id, run.status
            producer_id = f"training:{run.id}"
            source.update({"training_run_id": run.id})
            inputs = _training_inputs(session, run)
    elif kind == "model_weight_version":
        label = "模型权重接入"
        source.update({"commit": record.resolved_commit})
    elif kind in {"training_data_view_version", "tokenized_dataset_version"}:
        label, inputs = LABELS[kind], [("mixture_version", record.mixture_version_id)]
        # Corpus mappings are frozen inputs of the view, when recorded.
        inputs.extend(version_refs(record.config_snapshot))
    elif kind == "tokenized_corpus_version":
        label, inputs = "数据编码", [("dataset_version", record.dataset_version_id)]
        if (record.selection_config or {}).get("mixture_version_id"):
            inputs.append(("mixture_version", record.selection_config["mixture_version_id"]))
    elif kind == "supervision_corpus_version":
        label, inputs = "监督数据编码", [("supervision_mixture_version", record.mixture_version_id)]
    elif kind == "supervision_training_view_version":
        label, inputs = "监督读取索引", [("supervision_corpus_version", record.corpus_version_id)]
    if not label:
        return None
    source["task_id"] = task_id
    return {
        "id": producer_id,
        "label": label,
        "status": status,
        "source": source,
        "inputs": list(dict.fromkeys(inputs)),
    }


def _source_details(session, project_id, source):
    """Resolve navigable history objects without inventing an execution attempt."""
    source = dict(source)
    previous = session.get(m.ProjectPipelineRun, source["attempt_id"]) if source.get("attempt_id") else None
    if previous and previous.project_id == project_id:
        source["instance_id"] = previous.instance_id
    task = session.get(m.Task, source["task_id"]) if source.get("task_id") else None
    if task and (task.project_id == project_id or is_global_data_project(session.get(m.Project, task.project_id))):
        source["task_name"] = task.title
        source["training_run_id"] = source.get("training_run_id") or (task.payload or {}).get("training_run_id")
    elif task:
        source["task_id"] = None
    training = session.get(m.TrainingRun, source["training_run_id"]) if source.get("training_run_id") else None
    if training and training.project_id == project_id:
        source["training_name"] = training.current_stage or source.get("task_name") or f"训练 {training.id[:8]}"
        source["instance_id"] = source.get("instance_id") or (
            (training.config_snapshot or {}).get("revision", {}).get("effective_config", {}).get("training_instance_id")
        )
        if not source.get("instance_id"):
            from .training_history import source_run_id
            source["instance_id"] = next((item.id for item in session.scalars(select(m.ProjectPipelineInstance).where(m.ProjectPipelineInstance.project_id == project_id)).all() if source_run_id(item) == training.id), None)
    elif source.get("training_run_id"):
        source.pop("training_run_id", None)
    instance = session.get(m.ProjectPipelineInstance, source["instance_id"]) if source.get("instance_id") else None
    if instance and instance.project_id == project_id:
        source["instance_name"] = instance.name
    else:
        source.pop("instance_id", None)
    return source


def upstream_lineage(session: Session, run: m.ProjectPipelineRun, *, max_depth=4, max_nodes=120, projected_rows=None):
    graph = run.graph_snapshot or {}
    rows = projected_rows if projected_rows is not None else list(
        session.scalars(
            select(m.ProjectPipelineNodeRun).where(
                m.ProjectPipelineNodeRun.pipeline_run_id == run.id
            )
        ).all()
    )
    ledger = list(
        session.scalars(
            select(m.LineageExecution).where(m.LineageExecution.pipeline_run_id == run.id)
        ).all()
    )
    relations = (
        list(
            session.scalars(
                select(m.LineageAssetRelation).where(
                    m.LineageAssetRelation.execution_id.in_([e.id for e in ledger])
                )
            ).all()
        )
        if ledger
        else []
    )
    by_execution = {e.id: e for e in ledger}
    produced = {(r.asset_type, r.asset_id) for r in relations if r.relation_kind == "produced"}
    consumed = {}
    for relation in relations:
        if relation.relation_kind == "consumed":
            consumed.setdefault(by_execution[relation.execution_id].node_id, []).append(
                (relation.asset_type, relation.asset_id)
            )
    roots = []
    for row in rows:
        config = row.config_snapshot or {}
        refs = consumed.get(row.node_id, []) + version_refs(row.input_snapshot)
        refs += [(r["asset_type"], r["asset_id"]) for r in _config_refs(session, config)]
        refs += version_refs(config.get("training_spec") or config)
        training_id = config.get("training_run_id") or (row.input_snapshot or {}).get(
            "training_run_id"
        )
        training = session.get(m.TrainingRun, training_id) if training_id else None
        if training and training.project_id == run.project_id:
            refs += _training_inputs(session, training)
        for kind, record_id in dict.fromkeys(refs):
            if kind in ASSETS and (kind, record_id) not in produced:
                roots.append((kind, record_id, row.node_id))
    nodes, edges, warnings = {}, {}, set()
    namespace = f"history:{run.id}:"
    while any(n["id"].startswith(namespace) for n in graph.get("nodes", [])):
        namespace += ":"
    outgoing = {}
    pending = deque()
    expanded = set()
    truncated = False

    def add_edge(source, target, label=""):
        if source == target:
            return False
        frontier, visited = [target], set()
        while frontier:
            current = frontier.pop()
            if current == source:
                warnings.add("检测到历史血缘循环，已停止沿该路径展开。")
                return False
            if current not in visited:
                visited.add(current)
                frontier.extend(outgoing.get(current, []))
        key = f"{source}>{target}"
        edges[key] = {
            "id": key,
            "source": source,
            "target": target,
            "label": label,
            "context_only": True,
        }
        outgoing.setdefault(source, set()).add(target)
        return True

    def add_asset(kind, record_id, target, depth, ancestors, *, inherited=False):
        nonlocal truncated
        key = (kind, record_id)
        if kind not in ASSETS:
            return
        if key in ancestors:
            warnings.add("检测到历史血缘循环，已停止沿该路径展开。")
            return
        record = session.get(ASSETS[kind], record_id)
        if record is not None:
            if kind in ASSET_TYPES:
                if not inherited and not project_has_asset_access(
                    session, run.project_id, kind, record_id
                ):
                    warnings.add("部分历史输入不可见，未展开其来源。")
                    return
            elif getattr(record, "project_id", run.project_id) != run.project_id:
                warnings.add("部分历史输入不可见，未展开其来源。")
                return
        asset_id = f"{namespace}asset:{kind}:{record_id}"
        if asset_id not in nodes:
            if len(nodes) >= max_nodes:
                truncated = True
                return
            ref = _database_asset_ref(session, kind, record_id)
            label = ref["label"]
            if kind in ASSET_TYPES and ref["version"]:
                label = f"{label} · V{ref['version']}"
            if label.startswith(kind):
                version = (
                    getattr(record, "version", None) or getattr(record, "identity_hash", "")[:8]
                )
                label = f"{LABELS[kind]} · {version or record_id[:8]}"
            nodes[asset_id] = {
                "id": asset_id,
                "kind": "asset",
                "label": label,
                "asset_type": kind,
                "asset_id": record_id,
                "version": ref["version"],
                "status": ref["status"],
                "depth": depth * 2,
                "context_only": True,
                "read_only": True,
                "source": {},
                "note": "",
            }
        if not add_edge(asset_id, target, "冻结输入" if not inherited else "输入"):
            return
        if key not in expanded:
            pending.append((kind, record_id, asset_id, depth, ancestors | {key}, record))

    for kind, record_id, node_id in roots:
        add_asset(kind, record_id, node_id, 0, frozenset())
    while pending:
        kind, record_id, asset_id, depth, ancestors, record = pending.popleft()
        key = (kind, record_id)
        if key in expanded:
            continue
        expanded.add(key)
        if record is None:
            nodes[asset_id]["note"] = "版本记录缺失，仅保留冻结引用。"
        if max_depth is not None and depth >= max_depth:
            nodes[asset_id]["note"] = "已到展开深度，可继续展开。"
            truncated = True
            continue
        # Only this project's actual producers; never enumerate foreign executions.
        producers = list(
            session.scalars(
                select(m.LineageExecution)
                .join(
                    m.LineageAssetRelation,
                    m.LineageAssetRelation.execution_id == m.LineageExecution.id,
                )
                .where(
                    m.LineageAssetRelation.project_id == run.project_id,
                    m.LineageExecution.project_id == run.project_id,
                    m.LineageAssetRelation.asset_type == kind,
                    m.LineageAssetRelation.asset_id == record_id,
                    m.LineageAssetRelation.relation_kind == "produced",
                    m.LineageExecution.pipeline_run_id != run.id,
                )
                .order_by(m.LineageExecution.created_at.asc())
                .limit(13)
            )
            .unique()
            .all()
        )
        if len(producers) > 12:
            truncated = True
            producers = producers[:12]
        facts = []
        for producer in producers:
            node = session.get(m.ProjectPipelineNodeRun, producer.node_run_id)
            if node and (node.input_snapshot or {}).get("reused_from_node_run_id"):
                continue
            inputs = session.scalars(
                select(m.LineageAssetRelation).where(
                    m.LineageAssetRelation.execution_id == producer.id,
                    m.LineageAssetRelation.relation_kind == "consumed",
                )
            ).all()
            facts.append(
                {
                    "id": f"execution:{producer.id}",
                    "label": node.label if node else producer.node_id,
                    "status": producer.status,
                    "source": {
                        "attempt_id": producer.pipeline_run_id,
                        "node_run_id": producer.node_run_id,
                        "task_id": producer.task_id,
                        "task_attempt_id": (node.input_snapshot or {}).get("task_attempt_id") if node else None,
                        "instance_id": (producer.execution_identity or {}).get(
                            "training_instance_id"
                        ),
                    },
                    "inputs": [
                        (r.asset_type, r.asset_id) for r in inputs if r.asset_type in ASSETS
                    ],
                }
            )
        if not facts and record is not None:
            fact = _domain_source(session, kind, record)
            if fact:
                facts.append(fact)
        if not facts:
            nodes[asset_id]["note"] = nodes[asset_id]["note"] or "未登记上游加工来源。"
        for fact in facts:
            process_id = f"{namespace}{fact['id']}"
            if process_id not in nodes:
                if len(nodes) >= max_nodes:
                    truncated = True
                    break
                nodes[process_id] = {
                    "id": process_id,
                    "kind": "process",
                    "label": fact["label"],
                    "status": fact["status"],
                    "source": _source_details(session, run.project_id, fact["source"]),
                    "depth": depth * 2 + 1,
                    "context_only": True,
                    "read_only": True,
                    "note": "历史加工记录，不属于本次执行。",
                }
            add_edge(
                process_id,
                asset_id,
                "产出" if not fact["id"].startswith("version:model_version:") else "登记",
            )
            for input_kind, input_id in fact["inputs"]:
                add_asset(input_kind, input_id, process_id, depth + 1, ancestors, inherited=True)
    if truncated:
        warnings.add("上游血缘已按深度或节点数量截断，当前展示不是完整链路。")
    edges = layered_data_edges(list(nodes.values()), edges)
    outgoing = {}
    for binding in edges.values():
        outgoing.setdefault(binding["source"], set()).add(binding["target"])
    # Shared ancestors occupy their longest downstream distance, so their edges
    # still read left-to-right even when the same version is also a direct root.
    layers = dict.fromkeys(nodes, 0)
    incoming = {}
    remaining = {}
    for node_id in nodes:
        targets = [target for target in outgoing.get(node_id, []) if target in nodes]
        remaining[node_id] = len(targets)
        for target in targets:
            incoming.setdefault(target, []).append(node_id)
    pending_layers = deque(node_id for node_id, count in remaining.items() if not count)
    while pending_layers:
        node_id = pending_layers.popleft()
        for predecessor in incoming.get(node_id, []):
            layers[predecessor] = max(layers[predecessor], layers[node_id] + 1)
            remaining[predecessor] -= 1
            if not remaining[predecessor]:
                pending_layers.append(predecessor)
    for node_id, node in nodes.items():
        node["depth"] = layers[node_id]
    return {
        "attempt_id": run.id,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "warnings": sorted(warnings),
        "truncated": truncated,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "dag_node_ids": [n["id"] for n in graph.get("nodes", [])],
    }
