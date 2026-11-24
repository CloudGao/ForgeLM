import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingCanvas } from "./TrainingCanvas";
import type { ProjectPipelineRun, TrainingUpstreamLineage } from "../types";

let flow: any;
const fitView = vi.fn();
vi.mock("@xyflow/react", () => ({
  useNodesInitialized: () => true,
  useReactFlow: () => ({ screenToFlowPosition: (p: unknown) => p, fitView }),
  Handle: () => null, Controls: () => null, MiniMap: () => null, Background: () => null,
  Position: { Left: "left", Right: "right" }, MarkerType: { ArrowClosed: "arrowclosed" },
  addEdge: vi.fn(), applyEdgeChanges: vi.fn(), applyNodeChanges: vi.fn(),
  ReactFlow: (props: any) => { flow = props; return <div aria-label="图节点">{props.nodes.map((node: any) => <button key={node.id} onClick={() => props.onNodeClick({}, node)}>{node.data.context?.label || node.data.label}</button>)}</div>; },
}));

const graph = { name: "训练实例", description: "", variables: {}, viewport: {}, nodes: [
  { id: "data", type: "data" as const, label: "本次输入", position: { x: 0, y: 0 }, config: {} },
  { id: "train", type: "training" as const, label: "本次训练", position: { x: 400, y: 0 }, config: {} },
], edges: [{ id: "current-edge", source: "data", target: "train", source_handle: "out", target_handle: "in", label: "" }] };
const attempt: ProjectPipelineRun = { id: "attempt", project_id: "p", pipeline_id: null, pipeline_name: "", instance_id: "instance", instance_name: "训练实例", revision_id: null, revision: null, status: "succeeded", trigger_kind: "manual", input_snapshot: {}, context_snapshot: {}, graph_snapshot: graph, node_summary: {}, error: {}, node_runs: [], created_at: "2026-09-08T00:00:00Z" };
const context: TrainingUpstreamLineage = { attempt_id: "attempt", max_depth: 4, max_nodes: 120, warnings: [], truncated: false,
  nodes: [
    { id: "history-asset", kind: "asset", label: "清洗数据 V3", asset_type: "dataset_version", asset_id: "version-3", version: "3", status: "ready", depth: 0, context_only: true, read_only: true, source: {}, note: "" },
    { id: "history-process", kind: "process", label: "历史数据清洗", status: "succeeded", depth: 1, context_only: true, read_only: true, source: { attempt_id: "old-attempt", instance_id: "old-instance" }, note: "历史加工记录，不属于本次执行。" },
  ], edges: [
    { id: "history-edge", source: "history-process", target: "history-asset", label: "产出", context_only: true },
    { id: "input-edge", source: "history-asset", target: "data", label: "冻结输入", context_only: true },
  ],
};
let requestFails = false;
beforeEach(() => {
  requestFails = false; flow = null; fitView.mockClear();
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(requestFails ? { detail: "unavailable" } : context), { status: requestFails ? 503 : 200, headers: { "Content-Type": "application/json" } })));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function show(withAttempt = true, onOpenHistory?: (node: any) => void) {
  const onChange = vi.fn(), onExecuteNode = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><TrainingCanvas value={graph} catalog={[]} readOnly attempt={withAttempt ? attempt : undefined} onChange={onChange} onExecuteNode={onExecuteNode} onOpenHistory={onOpenHistory} /></QueryClientProvider>);
  return { onChange, onExecuteNode };
}

describe("Attempt upstream context", () => {
  it("renders recorded workflow steps and solid version bindings without making them schedulable", async () => {
    const recorded = { id: "recorded-train", type: "training", label: "已登记 SFT", position: { x: 500, y: 0 }, config: { _recorded_step: true } };
    const artifact = { ...context.nodes[0], id: "input-version", context_only: false, position: { x: 0, y: 0 } };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ graph: { nodes: [graph.nodes[1], recorded], edges: [] }, node_runs: [], artifacts: [artifact, context.nodes[1]], bindings: [{ id: "version-input", source: artifact.id, target: recorded.id, label: "输入", context_only: false }], execution_node_ids: ["train"], expanded_step_count: 2, warnings: [], truncated: false }), { headers: { "Content-Type": "application/json" } })));
    const onExecuteNode = vi.fn();
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><TrainingCanvas value={{ ...graph, id: "instance", project_id: "p" }} catalog={[]} readOnly onExecuteNode={onExecuteNode} /></QueryClientProvider>);
    await screen.findByRole("button", { name: "已登记 SFT" });
    expect(String(vi.mocked(fetch).mock.calls[0][0])).toContain("/execution-graph?");
    expect(flow.edges.find((edge: any) => edge.id === "version-input").style.strokeDasharray).toBeUndefined();
    expect(flow.nodes.find((node: any) => node.id === "input-version").position).toEqual({ x: 0, y: 0 });
    fireEvent.click(screen.getByRole("button", { name: "已登记 SFT" }));
    expect(screen.queryByRole("button", { name: "只运行此节点" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "本次训练" }));
    fireEvent.click(screen.getByRole("button", { name: "只运行此节点" }));
    expect(onExecuteNode).toHaveBeenCalledWith("train", "selected");
  });
  it("opens the original instance and attempt when historical training is clicked", async () => {
    const onOpenHistory = vi.fn();
    const { onChange, onExecuteNode } = show(true, onOpenHistory);
    fireEvent.click(await screen.findByRole("button", { name: "历史数据清洗" }));
    expect(onOpenHistory).toHaveBeenCalledWith(context.nodes[1]);
    expect(onChange).not.toHaveBeenCalled();
    expect(onExecuteNode).not.toHaveBeenCalled();
    expect(screen.queryByText("历史血缘 · 只读")).not.toBeInTheDocument();
  });
  it("keeps current edges solid and historical nodes and edges read-only", async () => {
    const { onChange, onExecuteNode } = show();
    await screen.findByRole("button", { name: "历史数据清洗" });
    const historyNode = flow.nodes.find((n: any) => n.id === "history-process");
    expect(historyNode).toMatchObject({ draggable: false, connectable: false, deletable: false });
    expect(flow.edges.find((e: any) => e.id === "history-edge")).toMatchObject({ style: { strokeDasharray: "6 5" }, deletable: false, reconnectable: false });
    expect(flow.edges.find((e: any) => e.id === "current-edge").style?.strokeDasharray).toBeUndefined();
    expect(flow.edges.find((e: any) => e.id === "input-edge").targetHandle).toBe("context-in");
    expect(flow.nodes.find((n: any) => n.id === "data").data.contextInput).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "历史数据清洗" }));
    expect(screen.getByText("历史血缘 · 只读")).toBeInTheDocument();
    expect(screen.getByText("old-attempt")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "从此节点继续" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "只运行此节点" })).not.toBeInTheDocument();
    flow.onConnect({ source: "history-process", target: "train" });
    flow.onNodesChange([{ type: "remove", id: "history-process" }]);
    expect(onChange).not.toHaveBeenCalled();
    expect(onExecuteNode).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "本次训练" }));
    fireEvent.click(screen.getByRole("button", { name: "从此节点继续" }));
    expect(onExecuteNode).toHaveBeenCalledWith("train", "from_node");
  });

  it("can hide history without changing the frozen DAG", async () => {
    const snapshot = structuredClone(graph);
    show();
    await screen.findByRole("button", { name: "清洗数据 V3" });
    fireEvent.click(screen.getByRole("button", { name: "收起扩展血缘" }));
    expect(screen.queryByRole("button", { name: "清洗数据 V3" })).not.toBeInTheDocument();
    expect(flow.nodes).toHaveLength(2);
    expect(flow.edges).toHaveLength(1);
    expect(graph).toEqual(snapshot);
    fireEvent.click(screen.getByRole("button", { name: "显示扩展血缘" }));
    await screen.findByRole("button", { name: "清洗数据 V3" });
  });

  it("leaves the execution graph usable if lineage cannot be loaded", async () => {
    requestFails = true;
    show();
    expect(await screen.findByRole("alert")).toHaveTextContent("上游血缘加载失败");
    expect(screen.getByRole("button", { name: "本次训练" })).toBeInTheDocument();
    requestFails = false;
    fireEvent.click(screen.getByRole("button", { name: "重新加载血缘" }));
    await screen.findByRole("button", { name: "历史数据清洗" });
  });

  it("does not attach history to the frozen configuration view", async () => {
    show(false);
    await waitFor(() => expect(flow.nodes).toHaveLength(2));
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "收起扩展血缘" })).not.toBeInTheDocument();
  });
});
