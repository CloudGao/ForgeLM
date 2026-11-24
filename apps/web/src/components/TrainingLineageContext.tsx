import { Handle, MarkerType, Position, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { ArrowUpRight, Database, History } from "lucide-react";
import type { TrainingUpstreamLineage, TrainingLineageNode } from "../types";
import { StatusBadge } from "./ui";
import { DisplayHandles, type DisplayPort } from "./TrainingGraphLayout";

export type ContextFlowNode = Node<{ context: TrainingLineageNode; displayPorts?: DisplayPort[] }, "lineageContext">;
const assetLabels: Record<string, string> = {
  dataset_version: "数据集", mixture_version: "数据混合", supervision_dataset_version: "监督数据", supervision_mixture_version: "监督混合",
  model_version: "模型版本", checkpoint: "Checkpoint", model_weight_version: "模型权重", eval_dataset_version: "评测数据",
  tokenized_corpus_version: "编码 Corpus", training_data_view_version: "训练读取索引", tokenized_dataset_version: "编码数据",
  code_version: "代码版本", tokenizer_version: "Tokenizer", evaluation_run: "EvalResult", artifact: "Artifact", model_specification_revision: "模型结构版本", supervision_corpus_version: "监督 Corpus", supervision_training_view_version: "监督训练 View",
};

export function LineageContextNode({ data, selected }: NodeProps<ContextFlowNode>) {
  const item = data.context;
  const Icon = item.kind === "asset" ? Database : History;
  const title = item.source.instance_name || item.source.training_name || item.label;
  return <>{data.displayPorts && <DisplayHandles ports={data.displayPorts} />}<article className={`training-lineage-node ${item.context_only ? "is-historical" : "is-current-version"} ${selected ? "is-selected" : ""}`} aria-label={`${item.label} · ${item.context_only ? "历史血缘" : "绑定版本"}，只读`}>
    {!data.displayPorts && <Handle type="target" position={Position.Left} id="in" isConnectable={false} />}
    <div className="training-lineage-kind"><Icon size={14} /><span>{item.kind === "asset" ? item.context_only ? "上游版本" : "绑定版本" : item.source.instance_id ? "历史训练实例" : item.source.training_run_id ? "历史训练" : "历史加工"}</span><small>只读</small></div>
    <strong>{title}</strong>
    {title !== item.label && <span className="training-lineage-caption">来源步骤：{item.label}</span>}
    <span className="training-lineage-caption">{item.kind === "asset" ? assetLabels[item.asset_type || ""] || item.asset_type : item.source.attempt_id ? `Attempt · ${item.source.attempt_id.slice(0, 8)}` : "历史领域记录"}</span>
    {item.status && <StatusBadge status={item.status} />}
    {item.note && <p>{item.note}</p>}
    {(item.source.instance_id || item.source.training_run_id) && <span className="training-lineage-caption">点击打开完整训练详情 ↗</span>}
    {!data.displayPorts && <Handle type="source" position={Position.Right} id="out" isConnectable={false} />}
  </article></>;
}

export function contextFlowGraph(context: TrainingUpstreamLineage, existing: { id: string; position: { x: number; y: number } }[], selectedId: string) {
  const left = Math.min(0, ...existing.map((n) => n.position.x));
  const top = Math.min(0, ...existing.map((n) => n.position.y));
  const layers = new Map<number, number>();
  const nodes: ContextFlowNode[] = context.nodes.map((item) => {
    const row = layers.get(item.depth) || 0;
    layers.set(item.depth, row + 1);
    return { id: item.id, type: "lineageContext", data: { context: item }, position: item.position || { x: left - 330 * (item.depth + 1), y: top + row * 230 },
      selected: item.id === selectedId, draggable: false, connectable: false, deletable: false, selectable: true, focusable: true, className: "training-context-flow-node" };
  });
  const dagIds = new Set(existing.map((node) => node.id));
  const edges: Edge[] = context.edges.map((edge) => ({ ...edge, sourceHandle: "out", targetHandle: dagIds.has(edge.target) ? "context-in" : "in", type: "smoothstep",
    className: edge.context_only ? "training-lineage-edge" : "training-binding-edge", style: edge.context_only ? { strokeDasharray: "6 5", opacity: 0.55 } : { strokeWidth: 2, opacity: 1 }, animated: false,
    deletable: false, reconnectable: false, selectable: false, markerEnd: { type: MarkerType.ArrowClosed, color: "#94a3b8" } }));
  return { nodes, edges };
}

const sourceLabels: Record<string, string> = { attempt_id: "来源 Attempt", instance_id: "来源训练实例", node_run_id: "来源节点执行", task_id: "任务", training_run_id: "训练运行", run_id: "加工运行", recipe_version_id: "配方版本", commit: "代码 / 权重 Commit" };

export function LineageContextInspector({ node, onOpenHistory }: { node: TrainingLineageNode; onOpenHistory?: (node: TrainingLineageNode) => void }) {
  return <><header><div><span>{node.context_only ? "历史血缘 · 只读" : "绑定版本"}</span><h2>{node.label}</h2></div></header><div className="pipeline-inspector-body">
    <p className="training-muted">{node.context_only ? "用于解释本次输入的来源，不参与本次调度或进度统计。" : "本次输入或产出的精确版本，连线表示实际消费与产出关系。"}</p>
    {node.status && <StatusBadge status={node.status} />}
    {node.note && <p>{node.note}</p>}
    {onOpenHistory && (node.source.instance_id || node.source.training_run_id || node.source.task_id) && <button className="button secondary" onClick={() => onOpenHistory(node)}><ArrowUpRight size={15} />{node.source.instance_id ? "打开来源实例与执行详情" : node.source.training_run_id ? "打开历史训练详情" : "打开历史任务详情"}</button>}
    <dl className="training-lineage-details">
      {node.asset_type && <><dt>资产类型</dt><dd>{assetLabels[node.asset_type] || node.asset_type}</dd><dt>精确版本 ID</dt><dd>{node.asset_id}</dd></>}
      {node.version && <><dt>版本</dt><dd>{node.version}</dd></>}
      {Object.entries(node.source).filter(([key, value]) => value && sourceLabels[key]).map(([key, value]) => <div key={key}><dt>{sourceLabels[key]}</dt><dd>{value}</dd></div>)}
    </dl>
  </div></>;
}
