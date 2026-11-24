import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Overview } from "./Overview";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function dashboard(workspaceUri: string) {
  return {
    project: { id: "project-1", name: "ExampleLM", workspace_uri: workspaceUri, lifecycle_stage: "data_preparation" },
    lifecycle: ["data_preparation", "tokenizer", "pretraining", "post_training", "evaluation", "quantization", "release"],
    current: {},
    active_tasks: [],
    alerts: [],
  };
}

function renderOverview(initialEntry = "/projects/project-1/overview") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={[initialEntry]}><Routes><Route path="/projects/:projectId"><Route path="overview" element={<Overview />} /><Route path="evaluations" element={<div>评测目标页面</div>} /></Route></Routes></MemoryRouter></QueryClientProvider>);
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Project overview address", () => {
  it("keeps Pipeline management out of the project overview", async () => {
    const requests: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith("/api/projects/project-1/dashboard")) return jsonResponse(dashboard("F:\\projects\\example-lm"));
      if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderOverview();

    expect(await screen.findByText("F:\\projects\\example-lm")).toBeInTheDocument();
    expect(screen.getByText("PROJECT OVERVIEW")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /数据准备：/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /创建产线/ })).not.toBeInTheDocument();
    expect(screen.queryByText("还没有产线")).not.toBeInTheDocument();
    expect(requests.some((url) => url.endsWith("/pipeline-options") || url.endsWith("/pipelines"))).toBe(false);
  });

  it("does not render Dashboard lifecycle nodes in the overview", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/dashboard")) return jsonResponse({
        ...dashboard("F:\\projects\\example-lm"),
        derived_lifecycle_stage: "pretraining",
        lifecycle_nodes: [
          { key: "data_preparation", label: "数据准备", status: "ready", raw_status: "ready", version: { kind: "dataset_version", id: "data-v3", label: "Global Corpus · V3" }, versions: [], detail: "1/1 个精确版本可用于下游", updated_at: "2026-08-16T01:00:00Z", target: "data", is_current: false },
          { key: "tokenizer", label: "Tokenizer", status: "ready", raw_status: "passed", version: { kind: "tokenizer_version", id: "tok-v4", label: "MicroLM Tokenizer · V4" }, versions: [], detail: "已通过并锁定，尚未设为默认", updated_at: "2026-08-16T01:01:00Z", target: "tokenizers", is_current: false },
          { key: "pretraining", label: "预训练", status: "stopped", raw_status: "cancelled", version: { kind: "model_version", id: "model-v4", label: "MicroLM · V4" }, versions: [], detail: "最近运行 cancelled；已有可用模型", updated_at: "2026-08-16T01:02:00Z", target: "training", is_current: true },
          { key: "post_training", label: "后训练", status: "not_started", raw_status: "", version: null, versions: [], detail: "尚无后训练运行或模型", updated_at: null, target: "training", is_current: false },
          { key: "evaluation", label: "评测", status: "ready", raw_status: "succeeded", version: { kind: "eval_suite_version", id: "suite-v2", label: "Release Gate · V2" }, versions: [], detail: "1 个模型 · 1 个套件", updated_at: "2026-08-16T01:03:00Z", target: "evaluations", is_current: false },
          { key: "quantization", label: "量化", status: "ready", raw_status: "succeeded", version: { kind: "quantization_run", id: "quant-1", label: "GPTQ · Q4" }, versions: [], detail: "源模型 MicroLM V4", updated_at: "2026-08-16T01:04:00Z", target: "releases", is_current: false },
          { key: "release", label: "发布", status: "ready", raw_status: "published", version: { kind: "release", id: "release-1", label: "MicroLM Local · 2026.08" }, versions: [], detail: "目标：local", updated_at: "2026-08-16T01:05:00Z", target: "releases", is_current: false },
        ],
        current: {
          data_asset: { name: "Global Corpus", version: 3, token_estimate: 8_000_000, document_count: 1_024, purposes: ["training"] },
          tokenizer: { name: "MicroLM Tokenizer", version: 4, vocab_size: 32_000, locked: true, is_default: false },
        },
      });
      if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderOverview();

    expect((await screen.findAllByText("Global Corpus · V3")).length).toBeGreaterThan(0);
    expect(screen.getByText("MicroLM Tokenizer · V4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /预训练：MicroLM · V4/ })).not.toBeInTheDocument();
    expect(screen.queryByText("Release Gate · V2")).not.toBeInTheDocument();
    expect(screen.queryByText("GPTQ · Q4")).not.toBeInTheDocument();
    expect(screen.queryByText("MicroLM Local · 2026.08")).not.toBeInTheDocument();
  });

  it("opens code import when linked from training", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/dashboard")) return jsonResponse(dashboard(""));
      if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderOverview("/projects/project-1/overview?action=import-code");

    expect(await screen.findByRole("dialog", { name: "导入模型代码" })).toBeInTheDocument();
  });

  it("shows and updates the project workspace URI", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/dashboard")) return jsonResponse(dashboard("F:\\projects\\example-lm"));
      if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1") && init?.method === "PATCH") {
        const body = JSON.parse(String(init.body));
        return jsonResponse({ ...dashboard(body.workspace_uri).project, ...body });
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderOverview();
    expect(await screen.findByText("F:\\projects\\example-lm")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "配置地址" }));
    const dialog = screen.getByRole("dialog", { name: "配置项目地址" });
    const input = within(dialog).getByRole("textbox", { name: /项目地址/ });
    fireEvent.change(input, { target: { value: "file:///workspace/example-lm" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1") && request.init?.method === "PATCH")).toBe(true));
    const update = requests.find((request) => request.url.endsWith("/api/projects/project-1") && request.init?.method === "PATCH");
    expect(JSON.parse(String(update?.init?.body))).toEqual({ workspace_uri: "file:///workspace/example-lm" });
    expect(await screen.findByText("file:///workspace/example-lm")).toBeInTheDocument();
  });

  it("imports a Git code source as an immutable version", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/dashboard")) return jsonResponse(dashboard(""));
      if (url.endsWith("/api/projects/project-1/code-sources") && init?.method === "POST") return jsonResponse({ id: "source-1", name: "Example model", versions: [] }, 201);
      if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderOverview();
    fireEvent.click(await screen.findByRole("button", { name: "导入代码" }));
    const dialog = screen.getByRole("dialog", { name: "导入模型代码" });
    fireEvent.change(within(dialog).getByRole("textbox", { name: "代码源名称" }), { target: { value: "Example model" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /来源地址/ }), { target: { value: "https://github.com/example/model.git" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /Revision/ }), { target: { value: "main" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /训练入口/ }), { target: { value: "train.py" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "导入并创建版本" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/code-sources") && request.init?.method === "POST")).toBe(true));
    const create = requests.find((request) => request.url.endsWith("/api/projects/project-1/code-sources") && request.init?.method === "POST");
    expect(JSON.parse(String(create?.init?.body))).toMatchObject({
      name: "Example model",
      source_type: "git",
      source_uri: "https://github.com/example/model.git",
      source_config: { revision: "main" },
      entrypoints: { training: "train.py" },
    });
  });
});
