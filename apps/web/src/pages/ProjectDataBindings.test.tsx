import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProjectDataBindings } from "./ProjectDataBindings";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/data"]}><Routes><Route path="/projects/:projectId/data" element={<ProjectDataBindings />} /></Routes></MemoryRouter></QueryClientProvider>);
}

const binding = {
  id: "binding-1",
  asset_type: "dataset_version",
  asset_version_id: "version-1234567890",
  purpose: "training",
  alias: "",
  created_at: "2026-08-16T00:00:00Z",
  asset: {
    name: "中文预训练语料",
    version: 3,
    status: "ready",
    origin_project_name: "系统数据工作区",
    global_managed: true,
    tags: ["zh", "pretrain"],
    asset: {},
  },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Project datasets", () => {
  it("positions the page as project dataset CRUD without project-side data creation", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/data-bindings")) return jsonResponse([binding]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderPage();

    expect(await screen.findByText("中文预训练语料")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "项目数据集", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("集中管理本项目使用的数据集。数据集的创建、导入、加工和版本管理在全局数据中心完成。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "添加已有数据集" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "打开全局数据中心" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "项目数据集清单" })).toBeInTheDocument();
    expect(screen.getByText("中文预训练语料")).toBeInTheDocument();
    expect(screen.queryByText("绑定意味着什么？")).not.toBeInTheDocument();
    expect(screen.queryByText("Tokenize 与训练准备结果")).not.toBeInTheDocument();
  });

  it("edits project-only configuration and confirms removal", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    let current = binding;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-bindings") && !init?.method) return jsonResponse([current]);
      if (url.endsWith("/api/projects/project-1/data-bindings/binding-1") && init?.method === "PATCH") {
        current = { ...current, ...JSON.parse(String(init.body)) };
        return jsonResponse(current);
      }
      if (url.endsWith("/api/projects/project-1/data-bindings/binding-1") && init?.method === "DELETE") return jsonResponse({ deleted: true, id: "binding-1" });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderPage();
    await screen.findByText("中文预训练语料");
    fireEvent.click(screen.getByRole("button", { name: "编辑" }));
    expect(screen.getByRole("dialog", { name: "编辑项目数据集" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: /项目别名/ }), { target: { value: "主训练集" } });
    fireEvent.change(screen.getByRole("combobox", { name: /项目用途/ }), { target: { value: "validation" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() => expect(requests.some((request) => request.init?.method === "PATCH")).toBe(true));
    const updated = requests.find((request) => request.init?.method === "PATCH");
    expect(JSON.parse(String(updated?.init?.body))).toEqual({ purpose: "validation", alias: "主训练集" });

    expect(await screen.findByText("主训练集")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "移除" }));
    expect(screen.getByRole("dialog", { name: "从项目移除数据集" })).toBeInTheDocument();
    expect(requests.some((request) => request.init?.method === "DELETE")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "确认从项目移除" }));
    await waitFor(() => expect(requests.some((request) => request.init?.method === "DELETE")).toBe(true));
  });

  it("adds only an existing global version to the project", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-bindings") && init?.method === "POST") return jsonResponse({ ...binding, id: "binding-2", asset_version_id: "version-2" }, 201);
      if (url.endsWith("/api/projects/project-1/data-bindings")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([{
        name: "Global Corpus",
        origin_project_name: "全局数据中心",
        tags: ["code"],
        versions: [{ id: "version-2", version: 2, state: "ready", document_count: 100, token_estimate: 5000, tier: "gold" }],
      }]);
      if (url.includes("/api/catalog/")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderPage();
    await screen.findByRole("heading", { name: "项目还没有数据集" });
    fireEvent.click(screen.getByRole("button", { name: "添加已有数据集" }));
    expect(screen.getByRole("dialog", { name: "添加已有数据集" })).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "加入项目 Global Corpus V2" }));

    await waitFor(() => expect(requests.some((request) => request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toEqual({
      asset_type: "dataset_version",
      asset_version_id: "version-2",
      purpose: "training",
      alias: "",
      metadata: {},
    });
  });
});
