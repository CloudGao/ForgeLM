import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("system routing", () => {
  it("exposes plugin management under compute without adding a system navigation item", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/control-plane")) return jsonResponse({ managed: false, restart_supported: false });
      if (["/api/plugins", "/api/plugin-suites", "/api/plugin-suite-deployments", "/api/workers", "/api/projects", "/api/plugin-installations"].some((path) => url.endsWith(path))) return jsonResponse([]);
      return new Response(null, { status: 404 });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/execution/plugins"]}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByRole("heading", { name: "插件管理" })).toBeInTheDocument();
    const system = screen.getByRole("navigation", { name: "系统导航" });
    expect(within(system).getByRole("link", { name: "计算" })).toHaveAttribute("aria-current", "page");
    expect(within(system).queryByRole("link", { name: "插件管理" })).not.toBeInTheDocument();
    const compute = screen.getByRole("navigation", { name: "计算管理" });
    expect(within(compute).getByRole("link", { name: "插件管理" })).toHaveAttribute("aria-current", "page");
    expect(await screen.findByRole("heading", { name: "没有匹配的插件套件" })).toBeInTheDocument();
  });

  it("uses the system overview as the homepage and keeps the project center on its own route", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects")) return jsonResponse([{
        id: "project-1",
        name: "ExampleLM",
        target_model_name: "ExampleLM-300M",
        target_parameters: "300M",
        workspace_uri: "F:\\models\\example",
        project_type: "pretrain",
        primary_languages: ["zh"],
        target_capabilities: ["chat"],
        lifecycle_stage: "数据准备",
        archived: false,
        recent_training_status: "not_started",
        data_progress: { dataset_count: 1, ready_versions: 1, percent: 100 },
        created_at: "2026-08-01T00:00:00Z",
        updated_at: "2026-08-15T00:00:00Z",
      }]);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([{ id: "dataset-1" }]);
      if (url.endsWith("/api/catalog/evaluation-datasets")) return jsonResponse([{ id: "benchmark-1" }]);
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "Local", hostname: "localhost", status: "online" }]);
      if (url.endsWith("/api/control-plane")) return jsonResponse({ managed: false, restart_supported: false, state: "unmanaged", instance_id: "test", pid: 1 });
      return new Response(null, { status: 404 });
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>
      </QueryClientProvider>,
    );

    const navigation = screen.getByRole("navigation", { name: "系统导航" });
    expect(navigation).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /从全局资产出发/ })).toBeInTheDocument();
    expect(await screen.findByText("ExampleLM")).toBeInTheDocument();
    expect(screen.getByText("个在线 Worker · 共 1 个节点")).toBeInTheDocument();

    expect(within(navigation).getByRole("link", { name: "首页" })).toHaveAttribute("aria-current", "page");
    const mobileMenu = screen.getByRole("button", { name: "打开系统导航" });
    expect(mobileMenu).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(mobileMenu);
    expect(mobileMenu).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(within(navigation).getByRole("link", { name: "项目中心" }));

    expect(await screen.findByRole("heading", { name: "项目清单" })).toBeInTheDocument();
    expect(mobileMenu).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/从数据到发布/)).not.toBeInTheDocument();
    expect(screen.queryByText("当前工作区")).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "项目清单" })).toBeInTheDocument();
    expect(within(navigation).getByRole("link", { name: "项目中心" })).toHaveAttribute("aria-current", "page");
  });
});
