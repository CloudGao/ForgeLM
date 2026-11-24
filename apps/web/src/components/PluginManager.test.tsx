import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginManager } from "./PluginManager";
import type { PluginVersion } from "./pluginManagement";

vi.mock("./TaskDetail", () => ({ TaskDetail: ({ taskId, onClose }: any) => <div role="dialog" aria-label="任务详情">任务 {taskId}<button onClick={onClose}>关闭日志</button></div> }));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const base: PluginVersion = {
  id: "v1", plugin_id: "plugin.llm-compressor.quantization", version: "0.1.0", kind: "quantization", entrypoint: "plugin.llm-compressor.quantization", enabled: true,
  source: { type: "builtin-catalog" }, runtime: { python: ">=3.11", isolation: "venv", requirements: ["llmcompressor==0.10.0"] },
  config_schema: { type: "object", properties: { budget: { type: "integer", default: 99 }, verify: { type: "boolean", default: true }, targets: { type: "array", items: { type: "string" }, default: ["Linear"] } } },
};
const newer: PluginVersion = { ...base, id: "v2", version: "0.2.0" };
const bundled: PluginVersion = { ...base, id: "builtin", plugin_id: "builtin.command.training", kind: "training_engine", source: { type: "builtin" } };
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function setup({ route = "/execution/plugins?project=p1", projectId, partialFailure = false, directoryError = false, legacyTasks = false }: { route?: string; projectId?: string; partialFailure?: boolean; directoryError?: boolean; legacyTasks?: boolean } = {}) {
  const calls: Array<{ url: string; method: string; body: any }> = [];
  let bindings = [{ plugin_id: base.plugin_id, plugin_version_id: "v1", enabled: true, default_config: { budget: 0, verify: false, targets: ["module,with,commas"], extra: { keep: true } } }];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input), method = init?.method || "GET", body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body });
    if (url === "/api/plugins") return directoryError ? response({ detail: { message: "目录服务不可用" } }, 500) : response([base, newer, bundled]);
    if (url === "/api/projects") return response([{ id: "p1", name: "实验项目" }]);
    if (url === "/api/workers") return response([
      { id: "w1", name: "GPU A", status: "online", capabilities: ["builtin.plugin-manager"] },
      { id: "w2", name: "GPU B", status: "online", capabilities: ["builtin.plugin-manager"] },
      { id: "offline", name: "离线机器", status: "offline", capabilities: ["builtin.plugin-manager"] },
    ]);
    if (url === "/api/plugin-installations") return response([{ id: "i1", worker_id: "w1", plugin_id: base.plugin_id, state: "ready", current_plugin_version: base, installed_versions: [{ plugin_version_id: "v1" }], last_verified_at: "2026-09-13T00:00:00Z" }]);
    if (url === "/api/projects/p1/plugin-bindings") return response(bindings);
    if (url.includes("/plugin-bindings/") && method === "PUT") { bindings = [{ ...bindings[0], ...body }]; return response(bindings[0]); }
    if (url.includes("/plugin-installations") && method === "POST") {
      if (partialFailure && body?.worker_id === "w2") return response({ detail: { message: "Worker 已离线" } }, 409);
      return response({ task: { id: "install-1" } }, 201);
    }
    if (url.startsWith("/api/tasks?")) return response([{ id: "operation-1", kind: legacyTasks ? "train" : "plugin_install", project_id: "p1", title: "更新插件到 0.2.0", observed_state: "failed", created_at: "2026-09-13T00:00:00Z", payload: { resource_request: { worker_selector: { worker_id: "w1" } } } }]);
    return response({ detail: { message: `Unexpected ${url}` } }, 404);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[route]}><PluginManager projectId={projectId} /></MemoryRouter></QueryClientProvider>);
  return calls;
}

describe("unified plugin management", () => {
  it("shows distinct installation and project state, and lists bundled adapters without installation", async () => {
    setup();
    expect(await screen.findByText("LLM Compressor")).toBeInTheDocument();
    expect(await screen.findByText("已启用 0.1.0")).toBeInTheDocument();
    expect(screen.getByText("1 台")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "builtin.command.training" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("安装方式"), { target: { value: "all" } });
    fireEvent.click(await screen.findByRole("button", { name: "builtin.command.training" }));
    expect(screen.getByRole("button", { name: "安装到 Worker" })).toBeDisabled();
    expect(screen.getByText(/实际能力由 Worker 与对应业务入口决定/)).toBeInTheDocument();
  });

  it("submits an explicit version for each Worker and retains partial failure results", async () => {
    const calls = setup({ partialFailure: true });
    fireEvent.click(await screen.findByRole("button", { name: "LLM Compressor" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "安装到 Worker" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "安装到 Worker" }));
    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByLabelText(/GPU A/));
    fireEvent.click(within(dialog).getByLabelText(/GPU B/));
    expect(within(dialog).getByLabelText(/离线机器/)).toBeDisabled();
    fireEvent.click(within(dialog).getByRole("button", { name: "安装到 2 台 Worker" }));
    expect(await screen.findByText("提交失败：Worker 已离线")).toBeInTheDocument();
    expect(screen.getByText("任务已提交，等待实际安装结果")).toBeInTheDocument();
    expect(calls.find((call) => call.url === "/api/plugin-installations/i1/update")?.body).toEqual({ project_id: "p1", target_plugin_version_id: "v2" });
    expect(calls.find((call) => call.url === "/api/projects/p1/plugin-installations")?.body).toEqual({ worker_id: "w2", plugin_version_id: "v2" });
    expect(calls.some((call) => call.method === "PUT")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "提交剩余目标" }));
    await vi.waitFor(() => expect(calls.filter((call) => call.url === "/api/projects/p1/plugin-installations").length).toBe(2));
    expect(calls.filter((call) => call.url === "/api/plugin-installations/i1/update")).toHaveLength(1);
  });

  it("preserves existing falsy, array and extension values when switching a binding version", async () => {
    const calls = setup();
    fireEvent.click(await screen.findByRole("button", { name: "LLM Compressor" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "项目启用与配置" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "项目启用与配置" }));
    expect(screen.getByLabelText("budget")).toHaveValue(0);
    expect(screen.getByLabelText("verify")).toHaveValue("false");
    fireEvent.click(screen.getByLabelText("允许项目使用此插件"));
    fireEvent.click(screen.getByRole("button", { name: "保存项目配置" }));
    await vi.waitFor(() => expect(calls.some((call) => call.method === "PUT")).toBe(true));
    expect(calls.find((call) => call.method === "PUT")?.body).toEqual({ plugin_version_id: "v2", enabled: false, default_config: { budget: 0, verify: false, targets: ["module,with,commas"], extra: { keep: true } } });
    await vi.waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("blocks malformed JSON and lets optional fields be removed", async () => {
    setup();
    fireEvent.click(await screen.findByRole("button", { name: "LLM Compressor" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "项目启用与配置" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "项目启用与配置" }));
    fireEvent.change(screen.getByLabelText("targets"), { target: { value: "[broken" } });
    expect(screen.getByRole("button", { name: /保存|提交/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "移除 targets" }));
    expect(screen.getByRole("button", { name: "保存项目配置" })).toBeEnabled();
    expect(screen.getByLabelText("targets")).not.toHaveAttribute("aria-invalid", "true");
  });

  it("requires a project for mutation and exposes failures without inventing empty results", async () => {
    const calls = setup({ route: "/execution/plugins" });
    fireEvent.click(await screen.findByRole("button", { name: "LLM Compressor" }));
    expect(screen.getByRole("button", { name: "安装到 Worker" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "项目启用与配置" })).toBeDisabled();
    expect(calls.filter((call) => call.method !== "GET")).toHaveLength(0);
  });

  it("shows matrix exact versions and reads persistent operation logs with a kind filter", async () => {
    const calls = setup({ route: "/execution/plugins?project=p1&pluginTab=workers" });
    expect(await screen.findByText("已安装")).toBeInTheDocument();
    const oldRow = screen.getByText("0.1.0", { selector: "th" }).closest("tr")!;
    expect(within(oldRow).getAllByText("未安装")).toHaveLength(2);
    fireEvent.click(screen.getByRole("tab", { name: "操作记录" }));
    expect(await screen.findByText("更新插件到 0.2.0")).toBeInTheDocument();
    expect(calls.some((call) => call.url.includes("kind=plugin_install&include_events=false") && call.url.includes("project_id=p1"))).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "查看日志" }));
    expect(screen.getByRole("dialog", { name: "任务详情" })).toHaveTextContent("operation-1");
    expect(screen.queryByRole("button", { name: "取消安装" })).not.toBeInTheDocument();
  });

  it("reports a directory error", async () => {
    setup({ directoryError: true });
    expect(await screen.findByText("目录服务不可用")).toBeInTheDocument();
    expect(screen.queryByText("目录中没有插件")).not.toBeInTheDocument();
  });

  it("rejects unfiltered task data from an old control plane", async () => {
    setup({ route: "/execution/plugins?pluginTab=operations", legacyTasks: true });
    expect(await screen.findByText(/控制面尚未加载插件任务过滤更新/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查看日志" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "取消安装" })).not.toBeInTheDocument();
  });
});
