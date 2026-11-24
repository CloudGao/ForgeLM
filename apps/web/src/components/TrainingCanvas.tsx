import { addEdge, applyEdgeChanges, applyNodeChanges, Background, Controls, MarkerType, MiniMap, ReactFlow, useReactFlow, useNodesInitialized, type Connection } from "@xyflow/react";
import { Activity, Bot, CircleDotDashed, Play, Plus, Settings2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { SuiteCanvasNode } from "./SuiteCanvasFields";
import type { Json, PipelineNodeCatalogItem, PipelineNodeKind, ProjectPipelineDagNode, ProjectPipelineDagEdge, ProjectPipelineNodeRun, ProjectPipelineOptions, ProjectPipelineRun, TrainingUpstreamLineage, TrainingLineageNode, TrainingExecutionGraph } from "../types";
import { defaultAdapters, fromFlowEdges, fromFlowNodes, LineageExecutionFacts, nodeIcons, nodeTypes, NodeInspector, ReadOnlyNodeInspector, runPresentations, toFlowEdges, toFlowNodes, type FlowNode, type PipelineNodeData } from "./TrainingGraph";
import { ErrorMessage, Field, StatusBadge } from "./ui";
import type { ExecutionSelectionMode } from "./TrainingExecutionDialog";
import { TaskTiming } from "./TaskTiming";
import { contextFlowGraph, LineageContextInspector, LineageContextNode } from "./TrainingLineageContext";
import { layoutTrainingGraph, RoutedTrainingEdge, type GraphLayout } from "./TrainingGraphLayout";

const trainingNodeTypes = { ...nodeTypes, lineageContext: LineageContextNode };
const trainingEdgeTypes = { routedTraining: RoutedTrainingEdge };

export type TrainingGraphValue = { id?: string; project_id?: string; name: string; description: string; nodes: ProjectPipelineDagNode[]; edges: ProjectPipelineDagEdge[]; variables: Json; viewport: Json; cloned_from_instance_id?: string | null };

export function TrainingCanvas({ value, catalog, options, readOnly = false, attempt, onChange, onConfigureTraining, onTask, onTrainingRun, onExecuteNode, onOpenHistory }: {
  value: TrainingGraphValue;
  catalog: PipelineNodeCatalogItem[];
  options?: ProjectPipelineOptions;
  readOnly?: boolean;
  attempt?: ProjectPipelineRun;
  onChange?: (value: TrainingGraphValue) => void;
  onConfigureTraining?: (node: ProjectPipelineDagNode) => void;
  onExecuteNode?: (nodeId: string, mode: ExecutionSelectionMode) => void;
  onTask?: (node: ProjectPipelineNodeRun) => void;
  onTrainingRun?: (id: string, taskAttemptId?: string) => void;
  onOpenHistory?: (node: TrainingLineageNode) => void;
}) {
  const { screenToFlowPosition, fitView } = useReactFlow();
  const nodesInitialized = useNodesInitialized();
  const [measurements, setMeasurements] = useState<Record<string, { width: number; height: number }>>({});
  const [layout, setLayout] = useState<GraphLayout>();
  const [layoutError, setLayoutError] = useState("");
  const [layoutRevision, setLayoutRevision] = useState(0);
  const [selectedId, setSelectedId] = useState("");
  const [paletteOpen, setPaletteOpen] = useState(true);
  const [showContext, setShowContext] = useState(true);
  const [centerNodeId, setCenterNodeId] = useState("");
  const [contextDepth, setContextDepth] = useState(4);
  const projection = useQuery({
    queryKey: ["training-execution-graph", value.id, attempt?.id, attempt?.status, showContext, centerNodeId],
    queryFn: () => api<TrainingExecutionGraph>(`/projects/${value.project_id}/training-instances/${value.id}/execution-graph?include_upstream=${showContext}${centerNodeId ? `&center_node_id=${encodeURIComponent(centerNodeId)}` : ""}${attempt ? `&attempt_id=${attempt.id}` : ""}`),
    enabled: readOnly && Boolean(value.id && value.project_id), retry: false, staleTime: 15_000,
    refetchInterval: attempt && ["queued", "running", "paused"].includes(attempt.status) ? 15_000 : false,
  });
  const context = useQuery({
    queryKey: ["training-upstream-lineage", attempt?.id, contextDepth, attempt?.status],
    queryFn: () => api<TrainingUpstreamLineage>(`/projects/${attempt!.project_id}/training-instances/${attempt!.instance_id}/attempts/${attempt!.id}/upstream-lineage?max_depth=${contextDepth}&max_nodes=${contextDepth > 4 ? 300 : 120}`),
    enabled: Boolean(attempt) && showContext && !value.id, retry: false, staleTime: 30_000,
    refetchInterval: attempt && ["queued", "running", "paused"].includes(attempt.status) ? 15_000 : false,
  });
  const graph = projection.data?.graph || attempt?.graph_snapshot || value;
  const rows = projection.data?.node_runs || attempt?.node_runs || [];
  const statuses = new Map(rows.map((n) => [n.node_id, n.status]));
  const nodes = toFlowNodes(graph.nodes, catalog, statuses, runPresentations(attempt)).map((n) => {
    const row = rows.find((item) => item.node_id === n.id);
    const scopeLabel = row?.input_snapshot?.excluded_from_execution ? "范围外 · 未执行" : row?.input_snapshot?.reused_from_node_run_id ? "复用历史结果" : undefined;
    return { ...n, className: n.data.config._context_only ? "training-historical-step" : undefined, selected: n.id === selectedId, data: { ...n.data, ...(scopeLabel ? { executionLabel: scopeLabel } : n.data.config._context_only ? { executionLabel: "历史执行 · 只读" } : n.data.config._recorded_step ? { executionLabel: "已登记的执行步骤" } : {}) } };
  });
  const edges = toFlowEdges(graph.edges);
  const contextData: TrainingUpstreamLineage | undefined = projection.data ? {
    attempt_id: attempt?.id || value.id!, nodes: projection.data.artifacts, edges: projection.data.bindings,
    warnings: projection.data.warnings, truncated: projection.data.truncated, max_depth: contextDepth, max_nodes: 300,
  } : showContext && context.data?.attempt_id === attempt?.id ? context.data : undefined;
  const history = contextData ? contextFlowGraph(contextData, nodes, selectedId) : { nodes: [], edges: [] };
  const historySelection = contextData?.nodes.find((node) => node.id === selectedId);
  const layoutKey = history.nodes.map((node) => node.id).join("|");
  const combinedNodes = [...nodes.map(node => ({ ...node, data: { ...node.data, contextInput: history.edges.some(edge => edge.target === node.id) } })), ...history.nodes]
    .map(node => ({ ...node, measured: measurements[node.id] }));
  const combinedEdges = [...edges, ...history.edges];
  const geometryKey = JSON.stringify([combinedNodes.map(node => [node.id, node.measured?.width, node.measured?.height]), combinedEdges.map(edge => [edge.id, edge.source, edge.target])]);
  useEffect(() => {
    if (!readOnly || !combinedNodes.length || (value.id && projection.isPending)) return;
    let stale = false;
    setLayoutError("");
    layoutTrainingGraph(combinedNodes, combinedEdges).then(next => { if (!stale) setLayout(next); })
      .catch(() => { if (!stale) setLayoutError("自动布局失败，请重新布局。"); });
    return () => { stale = true; };
  }, [readOnly, geometryKey, layoutRevision, projection.isPending]);
  const displayNodes = combinedNodes.map(node => {
    const placed = readOnly && layout?.nodes[node.id];
    return placed ? { ...node, position: placed.position, style: { ...node.style, height: placed.height },
      className: `${node.className || ""} training-layout-node`, data: { ...node.data, displayPorts: placed.ports } } : node;
  });
  const displayEdges = combinedEdges.map(edge => {
    const route = readOnly && layout?.edges[edge.id];
    return route ? { ...edge, type: "routedTraining", sourceHandle: route.sourceHandle, targetHandle: route.targetHandle, data: { points: route.points } } : edge;
  });
  const allCenterIds = projection.data?.center_node_ids || nodes.filter(node => !node.data.config._context_only).map(node => node.id);
  const connectedCenterIds = allCenterIds.filter(id => combinedEdges.some(edge => edge.source === id || edge.target === id));
  const centerIds = connectedCenterIds.length ? connectedCenterIds : allCenterIds;
  const centerKey = centerIds.join("|");
  const focusOptions = useMemo(() => ({ nodes: centerIds.slice(0, 1).map(id => ({ id })), minZoom: 0.7, maxZoom: 1, padding: 0.25 }), [centerKey, layoutKey, attempt?.id, value.id]);
  useEffect(() => {
    if (!readOnly || !nodesInitialized || !layout || (value.id && projection.isPending)) return;
    const frame = requestAnimationFrame(() => { fitView(focusOptions); });
    return () => cancelAnimationFrame(frame);
  }, [focusOptions, showContext, nodesInitialized, projection.isPending, fitView, layout]);
  const suiteNodes = useQuery({ queryKey: ["suite-nodes"], queryFn: () => api<SuiteCanvasNode[]>("/suite-nodes"), enabled: !readOnly });
  const selected = nodes.find((n) => n.id === selectedId);
  const execution = rows.find((n) => n.node_id === selectedId);
  const update = (patch: Partial<TrainingGraphValue>) => { if (!readOnly) onChange?.({ ...value, ...patch }); };
  function updateNode(patch: Partial<PipelineNodeData>) {
    update({ nodes: fromFlowNodes(nodes.map((n) => n.id === selectedId ? { ...n, data: { ...n.data, ...patch } } : n)) });
  }
  function addNode(kind: PipelineNodeKind, position?: { x: number; y: number }) {
    if (readOnly) return;
    const item = catalog.find((n) => n.type === kind);
    if (!item) return;
    const id = `${kind}-${crypto.randomUUID().slice(0, 8)}`;
    const config = ["translation", "fact_translation", "knowledge"].includes(kind) ? { execution_mode: "native", native_operation: kind === "fact_translation" ? "data.fact_translation" : kind === "translation" ? "data.translation" : "data.knowledge", parameters: { name: item.label, engine: kind === "fact_translation" ? "vllm" : "transformers", ...(kind === "fact_translation" ? { prompt_version: "cpt-generation-v8", document_mode: "whole", styles: ["facts"], proper_name_policy: "source_exact", validation_retry_limit: 0 } : {}), enable_thinking: false, batch_size: "auto", top_p: kind === "translation" ? 0.6 : 0.8 }, input_bindings: {} }
      : kind === "training" ? { execution_mode: "native", training_spec: {} }
      : defaultAdapters[kind] ? { adapter_id: defaultAdapters[kind], command: { executable: "", args: [] }, resource_request: { gpus: 0 } }
      : kind === "branch" ? { path: "", operator: "truthy", true_handle: "true", false_handle: "false" } : {};
    update({ nodes: [...value.nodes, { id, type: kind, label: item.label, description: "", position: position || { x: 90 + value.nodes.length % 3 * 300, y: 90 + Math.floor(value.nodes.length / 3) * 180 }, config }] });
    setSelectedId(id);
  }
  function addSuiteNode(item: SuiteCanvasNode) {
    const id = `suite-${crypto.randomUUID().slice(0, 8)}`;
    update({ nodes: [...value.nodes, { id, type: "custom", label: item.displayName, description: `${item.suite_id} ${item.version}`, position: { x: 90, y: 90 + value.nodes.length * 100 }, config: { execution_mode: "suite", suite_version_id: item.suite_version_id, suite_node_id: item.id, suite_package_digest: item.digest, suite_config: {}, suite_inputs: {} } }] });
    setSelectedId(id);
  }
  function connect(connection: Connection) {
    if (readOnly || !nodes.some((node) => node.id === connection.source) || !nodes.some((node) => node.id === connection.target)) return;
    update({ edges: fromFlowEdges(addEdge({ ...connection, id: `edge-${crypto.randomUUID().slice(0, 8)}`, markerEnd: { type: MarkerType.ArrowClosed } }, edges)) });
  }
  const nativeSelected = selected?.data.kind === "training" && selected.data.config.execution_mode === "native";
  return <>
    {readOnly && (attempt || value.id) && <div className="training-context-toolbar">
      <div className="training-lineage-legend"><span><i />本次执行</span><span><i className="historical" />关联历史 · 只读</span><span>上游 ← 当前节点 → 下游</span></div>
      <button className="button ghost small" aria-pressed={showContext} onClick={() => { setShowContext(!showContext); setSelectedId(""); }}>{showContext ? "收起扩展血缘" : "显示扩展血缘"}</button>
      {centerNodeId && <button className="button ghost small" onClick={() => { setCenterNodeId(""); setSelectedId(""); }}>本次全部节点</button>}
      {Boolean(contextData?.nodes.length) && <button className="button ghost small" onClick={() => fitView(focusOptions)}>定位当前节点</button>}
      {Boolean(contextData?.nodes.length) && <button className="button ghost small" onClick={() => fitView({ maxZoom: 1, padding: 0.15 })}>适配全部</button>}
      <button className="button ghost small" onClick={() => setLayoutRevision(revision => revision + 1)}>重新布局</button>
      {layoutError && <span role="alert">{layoutError}</span>}
      {!value.id && showContext && contextData?.truncated && contextDepth < 12 && <button className="button secondary small" disabled={context.isFetching || projection.isFetching} onClick={() => setContextDepth(Math.min(contextDepth + 4, 12))}>展开更多上游</button>}
      {showContext && (context.isFetching || projection.isFetching) && <span role="status">正在读取上游来源…</span>}
      {showContext && context.error && <span role="alert">上游血缘加载失败，执行图仍可查看。<button className="button ghost small" onClick={() => context.refetch()}>重新加载血缘</button></span>}
      {showContext && contextData && !contextData.nodes.length && <span className="training-muted">暂无可展示的上游版本来源</span>}
      {contextData?.warnings.map((warning) => <p className="training-context-warning" key={warning}>{warning}</p>)}
    </div>}
    {projection.data && projection.data.expanded_step_count > value.nodes.length && <div className="training-recorded-steps" aria-label="工作流执行步骤"><span>工作流步骤</span>{nodes.map((node) => <button key={node.id} className="button ghost small" title={node.data.label} onClick={() => { setSelectedId(node.id); fitView({ nodes: [{ id: node.id }], maxZoom: 0.85, padding: 0.5 }); }}>{node.data.kind === "training" ? node.data.label.includes("SFT") ? "SFT 训练" : node.data.label.includes("CPT") ? "CPT 训练" : "模型训练" : catalog.find((item) => item.type === node.data.kind)?.label || node.data.label}<StatusBadge status={node.data.runStatus || "waiting"} /></button>)}</div>}
    <ErrorMessage error={projection.error} />
    {!readOnly && <div className="training-canvas-toolbar"><button className="button ghost small" onClick={() => setPaletteOpen(!paletteOpen)}><Plus size={15} />{paletteOpen ? "收起节点库" : "展开节点库"}</button><span>连接节点组织执行顺序，点击节点配置参数</span></div>}
    <div className={`pipeline-builder-grid training-builder ${readOnly || !paletteOpen ? "is-locked" : ""} ${readOnly && !selected && !historySelection ? "inspector-collapsed" : ""}`}>
      {!readOnly && paletteOpen && <aside className="pipeline-palette"><header><span>节点库</span><small>拖拽或点击添加</small></header>{!!suiteNodes.data?.length && <section><h3>套件节点</h3>{suiteNodes.data.filter(n => n.execution.kind === "task").map(n => <button key={`${n.suite_version_id}:${n.id}`} onClick={() => addSuiteNode(n)}><i><Settings2 size={15} /></i><span><strong>{n.displayName}</strong><small>{n.suite_id} {n.version}</small></span><Plus size={14} /></button>)}</section>}{[...new Set(catalog.map((n) => n.category))].map((category) => <section key={category}><h3>{category}</h3>{catalog.filter((n) => n.category === category).map((item) => { const Icon = nodeIcons[item.type]; return <button key={item.type} draggable onClick={() => addNode(item.type)} onDragStart={(event) => { event.dataTransfer.setData("application/forgelm-pipeline-node", item.type); }}><i style={{ background: item.color }}><Icon size={15} /></i><span><strong>{item.label}</strong><small>{item.description}</small></span><Plus size={14} /></button>; })}</section>)}</aside>}
      <section className="pipeline-canvas" aria-label={attempt ? "执行状态图" : "训练配置图"} onDragOver={(event) => { if (!readOnly) event.preventDefault(); }} onDrop={(event) => { if (readOnly) return; event.preventDefault(); const position = screenToFlowPosition({ x: event.clientX, y: event.clientY }); addNode(event.dataTransfer.getData("application/forgelm-pipeline-node") as PipelineNodeKind, Number.isFinite(position.x) ? position : undefined); }}>
        <ReactFlow nodes={displayNodes} edges={displayEdges} nodeTypes={trainingNodeTypes} edgeTypes={trainingEdgeTypes} onNodesChange={(changes) => { const sizes = changes.filter((change) => change.type === "dimensions" && change.dimensions); if (sizes.length) setMeasurements((previous) => { const next = { ...previous }; let changed = false; for (const change of sizes) { if (change.type === "dimensions" && change.dimensions && (next[change.id]?.width !== change.dimensions.width || next[change.id]?.height !== change.dimensions.height)) { next[change.id] = change.dimensions; changed = true; } } return changed ? next : previous; }); if (!readOnly && changes.some((c) => c.type === "position" || c.type === "remove")) { const next = applyNodeChanges<FlowNode>(changes.filter((change) => change.type === "position" || change.type === "remove"), nodes); const ids = new Set(next.map((n) => n.id)); update({ nodes: fromFlowNodes(next), edges: value.edges.filter((e) => ids.has(e.source) && ids.has(e.target)) }); } }} onEdgesChange={(changes) => { if (!readOnly && changes.some((c) => c.type === "remove")) update({ edges: fromFlowEdges(applyEdgeChanges(changes, edges)) }); }} onConnect={connect} onNodeClick={(_, n) => { const historyNode = contextData?.nodes.find((item) => item.id === n.id); const recorded = rows.find((row) => row.node_id === n.id); if (recorded?.config_snapshot?._context_only && recorded.input_snapshot?.training_run_id && onOpenHistory) { onOpenHistory({ id: n.id, kind: "process", label: recorded.label, status: recorded.status, depth: 0, context_only: true, read_only: true, source: { training_run_id: recorded.input_snapshot.training_run_id }, note: "" }); return; } if (historyNode && onOpenHistory && (historyNode.source.instance_id || historyNode.source.training_run_id)) onOpenHistory(historyNode); else setSelectedId(n.id); }} onPaneClick={() => setSelectedId("")} onMoveEnd={(event, viewport) => { if (event && !readOnly) update({ viewport }); }} nodesDraggable={!readOnly} nodesConnectable={!readOnly} edgesReconnectable={!readOnly} deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]} fitView={!readOnly} fitViewOptions={focusOptions} minZoom={0.05} maxZoom={1.8}>
          <Background gap={22} size={1} color="#d9e2ec" /><Controls position="bottom-left" showInteractive={!readOnly} /><MiniMap pannable zoomable nodeColor={(n) => (n.data as PipelineNodeData).catalog?.color || "#94a3b8"} />
        </ReactFlow>
      </section>
      {(!readOnly || selected || historySelection) && <aside className="pipeline-inspector">
        {readOnly && selected && !selected.data.config._context_only && value.id && <div className="training-node-execution-actions"><button className="button secondary small" onClick={() => setCenterNodeId(selected.id)}>以此节点为中心展开</button></div>}
        {readOnly && selected && onExecuteNode && (!projection.data || projection.data.execution_node_ids.includes(selected.id)) && <div className="training-node-execution-actions"><button className="button secondary small" onClick={() => onExecuteNode(selected.id, "from_node")}><Play size={14} />从此节点继续</button><button className="button secondary small" onClick={() => onExecuteNode(selected.id, "selected")}>只运行此节点</button></div>}
        {historySelection ? <LineageContextInspector node={historySelection} onOpenHistory={onOpenHistory} /> : execution ? <><header><div><span>{execution.config_snapshot?._context_only ? "历史执行 · 只读" : "节点执行"}</span><h2>{execution.label}</h2></div><StatusBadge status={execution.status} /></header><div className="pipeline-inspector-body">
          {execution.input_snapshot?.excluded_from_execution && <p className="training-muted">该节点不在本次执行范围内，未运行，也未复用其结果。</p>}
          {execution.input_snapshot?.reused_from_node_run_id && <p className="training-muted">复用此前的冻结结果，未重新执行。</p>}
          {execution.input_snapshot?.preparation && !execution.task_id && <p>{execution.input_snapshot.preparation.message}</p>}
          {execution.error?.message && <p role="alert" className="form-error">{execution.error.message}</p>}
          {execution.task && <TaskTiming task={execution.task} />}
          {execution.config_snapshot?._context_only && execution.input_snapshot?.training_run_id && onOpenHistory && <button className="button secondary" onClick={() => onOpenHistory({ id: execution.node_id, kind: "process", label: execution.label, status: execution.status, depth: 0, context_only: true, read_only: true, source: { training_run_id: execution.input_snapshot.training_run_id }, note: "" })}>打开所属训练实例</button>}
          {execution.task_id && <button className="button secondary" onClick={() => onTask?.(execution)}><Activity size={15} />资源与日志</button>}
          {execution.task?.payload?.training_run_id && <button className="button secondary" onClick={() => onTrainingRun?.(String(execution.task?.payload?.training_run_id), execution.input_snapshot?.task_attempt_id)}><Bot size={15} />训练指标与产物</button>}
          <LineageExecutionFacts node={execution} />
          {selected && <details><summary>冻结节点配置</summary><ReadOnlyNodeInspector node={selected} /></details>}
          <details><summary>实际输入</summary><pre>{JSON.stringify(execution.input_snapshot, null, 2)}</pre></details><details><summary>本次产出</summary><pre>{JSON.stringify(execution.output_snapshot, null, 2)}</pre></details>
        </div></> : selected ? <>
          {nativeSelected && !readOnly && <div className="training-native-config"><strong>训练参数</strong><p>{selected.data.config.training_spec?.stages?.map((s: Json) => s.name).join(" → ") || "配置训练数据、模型、代码和超参数"}</p><button className="button primary" onClick={() => onConfigureTraining?.(value.nodes.find((n) => n.id === selectedId)!)}><Settings2 size={15} />配置训练参数</button></div>}
          {readOnly ? <ReadOnlyNodeInspector node={selected} /> : <NodeInspector node={selected} options={options} onChange={updateNode} onDelete={() => { update({ nodes: value.nodes.filter((n) => n.id !== selectedId), edges: value.edges.filter((e) => e.source !== selectedId && e.target !== selectedId) }); setSelectedId(""); }} />}
        </> : <><header><div><span>{attempt ? "执行概况" : "实例配置"}</span><h2>{attempt ? "选择节点查看执行" : "训练实例"}</h2></div></header><div className="pipeline-inspector-body"><Field label="实例名称"><input value={value.name} disabled={readOnly} onChange={(event) => update({ name: event.target.value })} maxLength={180} /></Field><Field label="说明"><textarea value={value.description} disabled={readOnly} onChange={(event) => update({ description: event.target.value })} /></Field><div className="pipeline-inspector-note"><CircleDotDashed size={16} /><p>{readOnly ? "配置已冻结。调整训练配置请复制为新实例。" : "创建后配置不可修改。外部输入绑定精确版本，内部输出在执行时登记。"}</p></div></div></>}
      </aside>}
    </div>
  </>;
}
