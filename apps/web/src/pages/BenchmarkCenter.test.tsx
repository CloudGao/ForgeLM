import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BenchmarkCenter } from "./BenchmarkCenter";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function renderBenchmarkCenter() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<MemoryRouter><QueryClientProvider client={queryClient}><BenchmarkCenter /></QueryClientProvider></MemoryRouter>);
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("global Benchmark asset management", () => {
  it("registers a standard definition without exposing project evaluation actions", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([{ id: "mmlu", name: "MMLU", description: "多任务知识评测", task_id: "multiple_choice", languages: ["en"], dimensions: ["knowledge"], primary_metric: "accuracy", estimated_cost: "medium", license: "MIT", framework: "lm-eval" }]);
      if (url.endsWith("/api/catalog/evaluation-datasets") && (!init?.method || init.method === "GET")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/evaluation-datasets/from-benchmark/mmlu") && init?.method === "POST") return jsonResponse({ id: "asset-1", name: "MMLU", version: 1, source_type: "benchmark", benchmark_config: { id: "mmlu", task_id: "multiple_choice" }, dimensions: ["knowledge"], license: "MIT" });
      if (url.endsWith("/api/catalog/evaluation-datasets/asset-1")) return jsonResponse({ version: { id: "asset-1", project_id: "global", name: "MMLU", version: 1, source_type: "benchmark", task_type: "multiple_choice", benchmark_config: { id: "mmlu", task_id: "multiple_choice", primary_metric: "accuracy" }, dimensions: ["knowledge"], contamination_tags: ["public-benchmark"], license: "MIT", global_managed: true }, versions: [{ id: "asset-1", version: 1, task_type: "multiple_choice", license: "MIT", dimensions: ["knowledge"] }], artifact: null, references: [], can_delete: true, delete_blockers: [] });
      return jsonResponse([]);
    }));

    renderBenchmarkCenter();
    expect(await screen.findByRole("heading", { name: "评测集（Benchmark）" })).toBeInTheDocument();
    expect(await screen.findByText("运行时获取")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /标准目录|在线来源|评测资产/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /发起评测|开始评测|选择模型/ })).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "添加评测集" }));

    await waitFor(() => expect(requests.some(({ url, init }) => url.endsWith("/api/catalog/evaluation-datasets/from-benchmark/mmlu") && init?.method === "POST")).toBe(true));
    expect(await screen.findByRole("heading", { name: "MMLU" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除版本" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "编辑并创建新版本" })).not.toBeInTheDocument();
  });

  it("offers validated upload and data-version construction as asset creation paths", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/evaluation-datasets/inspect-upload") && init?.method === "POST") return jsonResponse({
        filename: "eval.jsonl", format: "jsonl", size_bytes: 35, sample_count: 1,
        valid_sample_count: 1, invalid_sample_count: 0, fields: ["id", "prompt", "answer"],
        missing_mapping_fields: [], preview: [{ id: "1", prompt: "2+2?", answer: "4" }],
      });
      return jsonResponse([]);
    }));

    renderBenchmarkCenter();
    fireEvent.click((await screen.findAllByRole("button", { name: "上传或从数据创建" }))[0]);
    expect(await screen.findByRole("dialog", { name: "构建评测数据版本" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上传并构建" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "从数据版本构建" })).toBeInTheDocument();

    const file = new File(['{"id":"1","prompt":"2+2?","answer":"4"}\n'], "eval.jsonl", { type: "application/x-ndjson" });
    const fileInput = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, { target: { files: [file] } });
    expect(await screen.findByText("字段映射覆盖全部样本")).toBeInTheDocument();
    expect(screen.getByText("2+2?", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "冻结评测版本" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: /发起评测|开始评测|选择模型/ })).not.toBeInTheDocument();
  });

  it("shows downloaded online candidates and builds from their immutable version", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([{
        id: "dataset-1", name: "OpenBookQA", source_type: "huggingface", source_uri: "allenai/openbookqa",
        language: "en", license: "apache-2.0", tags: ["评测候选", "在线下载"], status: "ready",
        current_version: { id: "version-1", version: 3, state: "ready", artifact_id: "artifact-1", document_count: 4957, raw_size_bytes: 2048, source_snapshot: { file_count: 2 } },
        versions: [{ id: "version-1", version: 3, state: "ready", artifact_id: "artifact-1", document_count: 4957, raw_size_bytes: 2048 }],
      }]);
      return jsonResponse([]);
    }));

    renderBenchmarkCenter();
    expect(await screen.findByText("allenai/openbookqa", { exact: false })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "在线导入" })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "配置评测集" }));

    expect(await screen.findByRole("dialog", { name: "构建评测数据版本" })).toBeInTheDocument();
    const versionSelect = document.querySelector<HTMLSelectElement>('select[name="dataset_version_id"]');
    expect(versionSelect).toHaveValue("version-1");
    expect(screen.getAllByText("OpenBookQA · V3").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "冻结评测版本" })).toBeEnabled();
  });

  it("groups versions into an asset workspace and performs versioned edit and safe delete", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const v1 = { id: "eval-v1", project_id: "global", name: "Internal Eval", version: 1, source_type: "upload", task_type: "generation", license: "internal", dimensions: ["reasoning"], contamination_tags: ["private"], sample_count: 2, source_config: { preview: [{ prompt: "Q1", answer: "A1" }] }, schema_config: { prompt_field: "prompt", answer_field: "answer" }, scoring_config: { metric: "exact_match" }, created_at: "2026-08-14T00:00:00Z", updated_at: "2026-08-14T00:00:00Z" };
    const v2 = { ...v1, id: "eval-v2", version: 2, license: "internal-v2", dimensions: ["reasoning", "math"], created_at: "2026-08-15T00:00:00Z", updated_at: "2026-08-15T00:00:00Z" };
    const v3 = { ...v2, id: "eval-v3", version: 3, license: "internal-v3", created_at: "2026-08-15T01:00:00Z", updated_at: "2026-08-15T01:00:00Z" };
    const detail = (version: any, versions: any[]) => ({ version: { ...version, origin_project_name: "全局数据资产", global_managed: true }, versions, artifact: { id: "artifact-1", content_hash: "a".repeat(64), size_bytes: 128, media_type: "application/x-ndjson", availability_state: "available" }, references: [], can_delete: true, delete_blockers: [] });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/evaluation-datasets") && (!init?.method || init.method === "GET")) return jsonResponse([v2, v1]);
      if (url.endsWith("/api/catalog/evaluation-datasets/eval-v2") && (!init?.method || init.method === "GET")) return jsonResponse(detail(v2, [v2, v1]));
      if (url.endsWith("/api/catalog/evaluation-datasets/eval-v2/versions") && init?.method === "POST") return jsonResponse(detail(v3, [v3, v2, v1]), 201);
      if (url.endsWith("/api/catalog/evaluation-datasets/eval-v3") && init?.method === "DELETE") return new Response(null, { status: 204 });
      if (url.endsWith("/api/catalog/evaluation-datasets/eval-v3")) return jsonResponse(detail(v3, [v3, v2, v1]));
      return jsonResponse([]);
    }));

    renderBenchmarkCenter();
    expect(await screen.findAllByText("Internal Eval")).toHaveLength(1);
    expect(screen.getByText("2 个版本")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看评测集" }));

    expect(await screen.findByRole("heading", { name: "Internal Eval" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "数据查看" }));
    expect(screen.getByRole("textbox", { name: "搜索数据内容" })).toBeInTheDocument();
    await waitFor(() => expect(requests.some(({ url }) => url.endsWith("eval-v2/viewer/query"))).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "编辑并创建新版本" }));
    const license = screen.getByRole("textbox", { name: "许可证" });
    fireEvent.change(license, { target: { value: "internal-v3" } });
    fireEvent.click(screen.getByRole("button", { name: "创建新版本" }));
    await waitFor(() => expect(requests.some(({ url, init }) => url.endsWith("/api/catalog/evaluation-datasets/eval-v2/versions") && init?.method === "POST")).toBe(true));
    expect(await screen.findByText("当前查看 V3", { exact: false })).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "删除版本" }).at(-1)!);
    fireEvent.change(screen.getByRole("textbox", { name: "输入资产名称确认" }), { target: { value: "Internal Eval" } });
    fireEvent.click(screen.getAllByRole("button", { name: "删除版本" }).at(-1)!);
    await waitFor(() => expect(requests.some(({ url, init }) => url.endsWith("/api/catalog/evaluation-datasets/eval-v3") && init?.method === "DELETE")).toBe(true));
  });
});
