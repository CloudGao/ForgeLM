import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Training } from "./Training";
import { trainingRunsFor } from "../components/TrainingInstanceOverview";

vi.mock("../components/TrainingCanvas", () => ({ TrainingCanvas: ({ value, readOnly, onChange, attempt, onConfigureTraining }: any) => <div aria-label="测试画布"><input aria-label="实例名称" value={value.name} disabled={readOnly} onChange={(e) => onChange({ ...value, name: e.target.value })} />{!readOnly && <button onClick={() => onConfigureTraining(value.nodes[0])}>配置训练参数</button>}{attempt && <span>执行 {attempt.id}</span>}</div> }));
vi.mock("../components/NativeTrainingConfig", () => ({ NativeTrainingConfig: ({ onSave }: any) => <button onClick={() => onSave({ stages: [{ name: "SFT", type: "sft", learning_rate: 0.0001 }], code_version_id: "code-7", tokenizer_version_id: "tokenizer-3" })}>保存节点配置</button> }));
const node = { id: "training", type: "training", label: "模型训练", position: { x: 100, y: 100 }, config: { execution_mode: "native", training_spec: { stages: [{ name: "基础训练", target_tokens: 1000 }], code_version_id: "code-1" }, plan_revision_id: "private-revision", _native_plan_id: "private-plan" } };
const original = { id: "instance-1", project_id: "p1", name: "实验 A", description: "已冻结", status: "ready", locked: true, can_run: true, nodes: [node], edges: [], variables: { __training_instance: { schema_version: 1 } }, viewport: {}, node_count: 1, edge_count: 0, run_count: 0, created_at: "2026-09-08T00:00:00Z", updated_at: "2026-09-08T00:00:00Z" };
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
let requests: { url: string; method: string; body: any }[];
let runs: any[];
let instances: any[];
let trainingRuns: any[];

beforeEach(() => {
  sessionStorage.clear(); requests = []; runs = []; trainingRuns = []; instances = [structuredClone(original)];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const method = init?.method || "GET"; const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    requests.push({ url, method, body });
    if (url.endsWith("/training-instances") && method === "POST") { const created = { ...original, ...body, id: "instance-2", can_run: true }; instances.push(created); return response(created, 201); }
    if (url.includes("/training-instances?") && method === "GET") return response(instances);
    if (url.endsWith("/attempts")) {
      if (method === "POST") { const run = { ...runs[0], id: "attempt-new", instance_id: "instance-1", status: "running", node_runs: [], node_summary: {}, context_snapshot: {}, created_at: "2026-09-08T01:00:00Z" }; runs = [run, ...runs]; return response(run, 201); }
      return response(runs);
    }
    if (/training-instances\/instance-\d$/.test(url)) return response(instances.find((i) => url.endsWith(i.id)));
    if (url.endsWith("/validate")) return response({ valid: true, errors: [], warnings: [] });
    if (url.endsWith("/pipeline-options")) return response({ data_versions: [] });
    if (url.includes("/resolve-run/")) return response({ instance_id: instances.find((item) => item.source_refs?.training_run_id === url.split("/").at(-1))?.id });
    if (url.endsWith("/training-runs")) return response(trainingRuns);
    if (url.includes("/training-runs/")) return response(trainingRuns.find((run) => url.endsWith(run.id)));
    if (url.includes("/models?")) return response({ checkpoints: [], models: [] });
    if (url.endsWith("/pipeline-node-catalog")) return response([]);
    return response({ detail: "not found" }, 404);
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function show(path = "/projects/p1/training") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><Routes><Route path="/projects/:projectId/training" element={<Training />} /></Routes></MemoryRouter></QueryClientProvider>);
}

describe("unified training", () => {
  it("retains complete training records reported by an old workflow and does not infer them from command text", () => {
    const training = { id: "training-run", task_id: "training-task" } as any;
    const attempt = { id: "attempt", node_runs: [{ task_id: "workflow-task", output_snapshot: { workflow: { training: { id: training.id, task_id: training.task_id } } } }] } as any;
    expect(trainingRunsFor(original as any, [attempt], [training])).toEqual([training]);
    attempt.node_runs[0].output_snapshot = { command_result: { id: training.id, task_id: training.task_id } };
    expect(trainingRunsFor(original as any, [attempt], [training])).toEqual([]);
  });
  it("lists instances without definitions or a separate plan list", async () => {
    show();
    expect(await screen.findByText("实验 A")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "训练" })).toBeInTheDocument();
    expect(screen.queryByText("Pipeline 定义")).not.toBeInTheDocument();
    expect(screen.queryByText("训练计划")).not.toBeInTheDocument();
    expect(requests.some((r) => r.url.includes("pipeline-definitions") || r.url.endsWith("training-plans"))).toBe(false);
  });
  it("creates a frozen instance in one submission after configuring its native training node", async () => {
    show(); await screen.findByText("实验 A");
    fireEvent.click(screen.getByRole("button", { name: "新建训练" }));
    fireEvent.change(screen.getByLabelText("实例名称"), { target: { value: "新训练" } });
    fireEvent.click(screen.getByText("配置训练参数"));
    fireEvent.click(screen.getByText("保存节点配置"));
    expect(requests.filter((r) => r.method === "POST")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "创建并冻结实例" }));
    expect(await screen.findByRole("button", { name: "复制并修改" })).toBeInTheDocument();
    const writes = requests.filter((r) => r.method === "POST");
    expect(writes).toHaveLength(1);
    expect(writes[0].body.nodes[0].config.training_spec.code_version_id).toBe("code-7");
    expect(screen.getByRole("tab", { name: "工作流与配置" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("测试画布")).toBeInTheDocument();
    expect(screen.getByLabelText("实例名称")).toBeDisabled();
  });
  it("clones configuration into an editable draft and preserves the source", async () => {
    show(); await screen.findByText("实验 A");
    fireEvent.click(screen.getByRole("button", { name: "复制 实验 A" }));
    expect(await screen.findByLabelText("实例名称")).toHaveValue("实验 A · 副本");
    fireEvent.change(screen.getByLabelText("实例名称"), { target: { value: "实验 B" } });
    fireEvent.click(screen.getByRole("button", { name: "创建并冻结实例" }));
    await screen.findByRole("button", { name: "复制并修改" });
    const body = requests.find((r) => r.method === "POST")!.body;
    expect(body.cloned_from_instance_id).toBe("instance-1");
    expect(body.nodes[0].config.plan_revision_id).toBeUndefined();
    expect(body.nodes[0].config.training_spec.code_version_id).toBe("code-1");
    expect(instances[0].name).toBe("实验 A");
  });
  it("copies historical training configuration instead of importing a read-only projection", async () => {
    instances[0] = { ...original, instance_kind: "imported", source_refs: { training_run_id: "legacy-run" } };
    const fetcher = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/training-runs/legacy-run")) return response({ id: "legacy-run", project_id: "p1", config_snapshot: { revision: { code_version_id: "legacy-code", stages: [{ name: "旧阶段" }], effective_config: { engine_config: { executable: "__worker_python__" } } }, selected_stages: [{ name: "选定阶段" }], resume_checkpoint: { checkpoint_id: "checkpoint-9" } } });
      return fetcher(input, init);
    }));
    show(); await screen.findByText("实验 A");
    fireEvent.click(screen.getByRole("button", { name: "复制 实验 A" }));
    await screen.findByLabelText("实例名称");
    fireEvent.click(screen.getByRole("button", { name: "创建并冻结实例" }));
    await screen.findByRole("button", { name: "复制并修改" });
    const body = requests.find((r) => r.method === "POST")!.body;
    expect(body.cloned_from_instance_id).toBe("instance-1");
    expect(body.nodes).toHaveLength(1);
    expect(body.nodes[0].config.training_spec).toMatchObject({ code_version_id: "legacy-code", stages: [{ name: "选定阶段" }], input_model_ref: { checkpoint_id: "checkpoint-9" } });
  });
  it("retries the same instance using only the prior attempt identity", async () => {
    runs = [{ id: "attempt-failed", status: "failed", instance_id: "instance-1", node_runs: [], node_summary: {}, context_snapshot: {}, created_at: "2026-09-08T00:00:00Z" }];
    show("/projects/p1/training?instance=instance-1");
    fireEvent.click(await screen.findByRole("button", { name: "重试并恢复" }));
    await waitFor(() => expect(requests.some((r) => r.method === "POST")).toBe(true));
    expect(requests.find((r) => r.method === "POST")!.body).toEqual({ retry_from_attempt_id: "attempt-failed" });
    expect(requests.find((r) => r.method === "POST")!.url).toContain("instance-1/attempts");
  });
  it("keeps failed loading distinct from a successful instance creation", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response({ detail: { message: "服务暂时不可用" } }, 503)));
    show();
    expect(await screen.findByText(/服务暂时不可用/)).toBeInTheDocument();
    expect(screen.queryByText("配置检查通过")).not.toBeInTheDocument();
  });
  it("opens the complete graph for both instance and attempt, with overview and metrics still reachable", async () => {
    runs = [{ id: "attempt-1", instance_id: "instance-1", status: "succeeded", node_runs: [], node_summary: {}, context_snapshot: {}, created_at: original.created_at }];
    show("/projects/p1/training?instance=instance-1");
    expect(await screen.findByLabelText("测试画布")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "实例概览" }));
    expect(screen.getByLabelText("训练实例概览")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: /Attempt 1.*查看执行详情/ }));
    expect(await screen.findByText("执行 attempt-1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "执行详情" }));
    expect(screen.getByLabelText("Attempt 执行详情")).toBeInTheDocument();
  });
  it("keeps an imported training's metrics and complete detail reachable directly from its instance", async () => {
    instances[0] = { ...original, instance_kind: "imported", source_refs: { training_run_id: "history-1" } };
    trainingRuns = [{ id: "history-1", project_id: "p1", current_stage: "基座预训练", current_step: 400, current_tokens: 10000, latest_metrics: { loss: 2.4 }, config_snapshot: {}, status: "succeeded", created_at: original.created_at }];
    show("/projects/p1/training?instance=instance-1");
    fireEvent.click(await screen.findByRole("tab", { name: "实例概览" }));
    fireEvent.click(await screen.findByRole("button", { name: "打开训练详情" }));
    expect(await screen.findByRole("heading", { name: "训练运行 history-" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "训练输出与血缘" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "运行日志" })).toBeInTheDocument();
  });
  it("opens adopted historical training in the same instance detail, including old run links", async () => {
    instances.push({ ...original, id: "instance-2", name: "旧 SFT", instance_kind: "imported", can_run: false, source_refs: { training_run_id: "history-2" } });
    show();
    await screen.findByText("旧 SFT");
    fireEvent.change(screen.getByLabelText("搜索训练实例"), { target: { value: "旧 SFT" } });
    expect(screen.queryByText("实验 A")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看详情" }));
    expect(await screen.findByLabelText("测试画布")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "工作流与配置" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("heading", { name: /训练运行/ })).not.toBeInTheDocument();
    cleanup();
    show("/projects/p1/training?run=history-2");
    expect(await screen.findByLabelText("测试画布")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "旧 SFT" })).toBeInTheDocument();
    expect(requests.filter((request) => request.method !== "GET")).toHaveLength(0);
  });
});
