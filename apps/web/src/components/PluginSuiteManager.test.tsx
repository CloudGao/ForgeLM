import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { PluginSuiteManager, type PluginSuite } from "./PluginSuiteManager";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const suite: PluginSuite = { id: "suite-v1", suite_id: "suite.microlm", name: "MicroLM", version: "0.1.0", enabled: true, description: "原生模型生命周期", architectures: ["microlm"], manifest_hash: "fixed", components: [
  { key: "training", label: "预训练 / SFT / DPO", node_kind: "training", plugin_id: "builtin.command.training", plugin_version_id: "train-v1", version: "0.1.0", deployment: "project_code", note: "需项目代码", plugin: null },
  { key: "quantization", label: "W8A16 量化", node_kind: "quantization", plugin_id: "plugin.microlm.quantization", plugin_version_id: "quant-v1", version: "0.1.0", deployment: "installable", note: "", plugin: null },
  { key: "export", label: "Native 导出", node_kind: "export", plugin_id: "plugin.forgelm.native-exporter", plugin_version_id: "export-v1", version: "0.1.0", deployment: "installable", note: "", plugin: null },
] };
const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
function setup(route = "/execution/plugins?project=p1") {
  const writes: Array<{ url: string; body: any }> = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (init?.method === "POST") { writes.push({ url, body: JSON.parse(String(init.body)) }); return response({ results: [{ key: "quantization", status: "installed" }, { key: "export", status: "failed", error: { message: "组件环境冲突" } }] }); }
    if (url === "/api/plugin-suites") return response([suite]);
    if (url === "/api/plugin-suite-deployments") return response([{ suite_version_id: suite.id, worker_id: "w1", worker_name: "GPU Worker", worker_status: "online", state: "partial", ready_count: 2, total_count: 3, project_configuration_required: true, components: [{ key: "training", state: "project_required" }, { key: "quantization", state: "installed" }, { key: "export", state: "missing" }] }]);
    if (url === "/api/workers") return response([{ id: "w1", name: "GPU Worker", status: "online", capabilities: ["builtin.plugin-manager"] }]);
    if (url === "/api/projects") return response([{ id: "p1", name: "测试项目" }]);
    return response([]);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[route]}><PluginSuiteManager /></MemoryRouter></QueryClientProvider>);
  return writes;
}

it("uses suites as catalog entries and exposes component roles in detail", async () => {
  setup();
  expect(await screen.findByRole("button", { name: "MicroLM" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "W8A16 量化" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "MicroLM" }));
  expect(await screen.findByRole("heading", { name: "MicroLM · 0.1.0" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "W8A16 量化" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "预训练 / SFT / DPO" })).toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: "前往配置节点" })[0]).toHaveAttribute("href", "/projects/p1/training");
});

it("groups Worker deployment by suite and expands exact components", async () => {
  setup("/execution/plugins?pluginTab=workers");
  expect(await screen.findByText("部分部署")).toBeInTheDocument();
  const table = screen.getByRole("table");
  expect(within(table).getByRole("button", { name: "MicroLM" })).toBeInTheDocument();
  expect(within(table).queryByRole("button", { name: "MicroLM W8A16" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByText("展开 3 个组件"));
  expect(screen.getByText("已安装精确版本")).toBeVisible();
  expect(screen.getByText("需项目代码配置")).toBeVisible();
  expect(screen.getByRole("button", { name: "安装 / 补齐组件" })).toBeDisabled();
});

it("installs selected suite components and displays per-component outcomes", async () => {
  const writes = setup();
  fireEvent.click(await screen.findByRole("button", { name: "MicroLM" }));
  fireEvent.click(await screen.findByRole("button", { name: "安装套件 / 按需安装" }));
  const dialog = screen.getByRole("dialog");
  expect(within(dialog).queryByRole("checkbox", { name: /预训练/ })).not.toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /GPU Worker/ }));
  fireEvent.click(within(dialog).getByRole("button", { name: "提交套件安装" }));
  expect(await screen.findByText("已安装，直接复用")).toBeInTheDocument();
  expect(screen.getByText("失败：组件环境冲突")).toBeInTheDocument();
  expect(writes).toEqual([{ url: "/api/projects/p1/plugin-suites/suite-v1/installations", body: { worker_id: "w1", component_keys: ["quantization", "export"] } }]);
});

it("finds suites through the node capability dimension", async () => {
  setup("/execution/plugins?pluginTab=capabilities");
  const choice = await screen.findByRole("button", { name: "MicroLM 0.1.0 · W8A16 量化" });
  fireEvent.click(choice);
  expect(await screen.findByRole("heading", { name: "MicroLM · 0.1.0" })).toBeInTheDocument();
});
