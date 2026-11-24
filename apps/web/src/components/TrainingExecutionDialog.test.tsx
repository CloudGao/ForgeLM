import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProjectPipelineInstance, ProjectPipelineRun } from "../types";
import { executionScopeLabel, TrainingExecutionDialog } from "./TrainingExecutionDialog";

const instance = { id: "instance", project_id: "project", can_run: true, nodes: ["a", "b", "c"].map((id) => ({ id, label: `节点 ${id}` })) } as ProjectPipelineInstance;
const source: ProjectPipelineRun = {
  id: "source-attempt", project_id: "project", pipeline_id: null, pipeline_name: "", instance_id: "instance", instance_name: "测试实例",
  revision_id: null, revision: null, status: "succeeded", trigger_kind: "manual", created_at: "2026-09-08T00:00:00Z",
  context_snapshot: { __training_instance: { attempt_number: 1 } }, input_snapshot: {},
  graph_snapshot: { nodes: [], edges: [], variables: {}, viewport: {} }, node_summary: {}, error: {}, node_runs: [],
};
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
let requests: { url: string; body: any }[];
let available: boolean;

beforeEach(() => {
  requests = []; available = true;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const body = JSON.parse(String(init?.body));
    requests.push({ url, body });
    if (url.endsWith("/preview")) return response({ valid: available, selected_node_ids: body.selection_mode === "from_node" ? ["b", "c"] : body.node_ids, reused_node_ids: ["a"], excluded_node_ids: [], reused_nodes: [{ node_id: "a", label: "节点 a", status: "succeeded", source_attempt_id: "source-attempt", outputs: { version: "output-v1" } }], errors: available ? [] : [{ code: "missing", message: "上游节点 a 缺少可复用结果" }], warnings: [] });
    return response({ id: "new-attempt", instance_id: "instance", status: "running" }, 201);
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function show(mode: "from_node" | "selected" = "selected", initialNodeId?: string) {
  const onStarted = vi.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><TrainingExecutionDialog instance={instance} attempts={[source]} initialMode={mode} initialNodeId={initialNodeId} onClose={() => {}} onStarted={onStarted} /></QueryClientProvider>);
  return onStarted;
}

describe("training execution scope", () => {
  it("submits only checked nodes with their explicit upstream source", async () => {
    const onStarted = show();
    expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "节点 b b" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "节点 c c" }));
    await screen.findByRole("heading", { name: "本次执行 2 个节点" });
    await waitFor(() => expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeEnabled());
    expect(requests.every((r) => r.url.endsWith("/preview"))).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "创建 Attempt 并执行" }));
    await waitFor(() => expect(onStarted).toHaveBeenCalled());
    expect(requests.find((r) => !r.url.endsWith("/preview"))?.body).toEqual({ selection_mode: "selected", node_ids: ["b", "c"], source_attempt_id: "source-attempt" });
  });

  it("previews downstream expansion from a node shortcut", async () => {
    show("from_node", "b");
    expect(screen.getByLabelText("起始节点")).toHaveValue("b");
    await screen.findByRole("heading", { name: "本次执行 2 个节点" });
    expect(requests[0].body).toEqual({ selection_mode: "from_node", node_ids: ["b"], source_attempt_id: "source-attempt" });
    expect(screen.getByText("复用的上游结果")).toBeInTheDocument();
  });

  it("blocks submission when upstream results are missing", async () => {
    available = false;
    show("selected", "c");
    expect(await screen.findByRole("alert")).toHaveTextContent("上游节点 a 缺少可复用结果");
    expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeDisabled();
    expect(requests.every((r) => r.url.endsWith("/preview"))).toBe(true);
  });

  it("clears approval of a previous preview when selection changes", async () => {
    show("selected", "b");
    await waitFor(() => expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeEnabled());
    available = false;
    fireEvent.click(screen.getByRole("checkbox", { name: "节点 c c" }));
    expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeDisabled();
    await screen.findByRole("alert");
    expect(screen.getByRole("button", { name: "创建 Attempt 并执行" })).toBeDisabled();
  });

  it("keeps partial scope visible in a retried attempt", () => {
    expect(executionScopeLabel({ ...source, trigger_kind: "retry", context_snapshot: { __training_instance: { execution_scope: { selection_mode: "selected", selected_node_ids: ["b"] } } } })).toBe("恢复执行 · 选定执行 · 1 个节点");
  });
});
