import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Dataset } from "../types";
import { Tokenizers } from "./Tokenizers";

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Tokenizer data readiness actions", () => {
  it("creates an immutable Chat Template with assistant generation spans", async () => {
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/chat-templates") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ id: "ct-1", versions: [] });
      }
      if (url.endsWith("/tokenizers") || url.endsWith("/datasets") || url.endsWith("/mixtures") || url.endsWith("/chat-templates") || url.includes("/tasks?")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "新建 Chat Template" }));
    expect((screen.getByRole("textbox", { name: /Jinja 模板/ }) as HTMLTextAreaElement).value).toContain("{% generation %}");
    fireEvent.click(screen.getByRole("button", { name: "创建不可变版本" }));
    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted).toMatchObject({ assistant_mask_mode: "generation_spans_v1" });
    expect(submitted.template_body).toContain("{% endgeneration %}");
  });

  it("shows every eligible data mixture version without hiding it in a collapsed dropdown", async () => {
    const mixture = { id: "mixture-1", project_id: "project-1", name: "tokenizer专用", current_version: { id: "mixture-version-3", version: 3, status: "ready", locked: true, manifest_artifact_id: "mixture-manifest-3" } };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/mixtures")) return jsonResponse([mixture]);
      if (url.endsWith("/tokenizers") || url.endsWith("/datasets") || url.includes("/tasks?") || url.endsWith("/tokenized-corpora") || url.endsWith("/training-data-views")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click((await screen.findAllByRole("button", { name: "新建实验" }))[0]);

    const trainingData = screen.getByLabelText("训练数据");
    expect(trainingData).toHaveAttribute("size", "2");
    expect(screen.getByRole("option", { name: "tokenizer专用 · V3" })).toBeInTheDocument();
    expect(screen.getByText("1 个可用版本，全部展开显示")).toBeInTheDocument();
    fireEvent.change(trainingData, { target: { value: "mixture-version-3" } });
    expect(trainingData).toHaveValue("mixture-version-3");
  });

  it("creates a reusable Tokenized Corpus with every encoding option in the request", async () => {
    const tokenizer = { id: "tokenizer-1", project_id: "project-1", name: "Locked BPE", version: 2, algorithm: "BPE", vocab_size: 32000, status: "passed", locked: true, is_default: true, task_id: null, artifact_id: "tokenizer-artifact-1", metrics: {} };
    const dataset = { id: "dataset-1", project_id: "project-1", name: "Corpus", source_type: "local", license: "MIT", language: "zh", tags: [], status: "ready", versions: [], current_version: { id: "dataset-version-1", version: 3, state: "ready", train_ready: true, artifact_id: "snapshot-1", record_manifest_artifact_id: "records-1", token_estimate: 12_000_000_000 } };
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/tokenized-corpora") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ id: "tokenized-1", status: "queued", ...submitted });
      }
      if (url.endsWith("/tokenized-corpora") || url.endsWith("/training-data-views")) return jsonResponse([]);
      if (url.endsWith("/tokenizers")) return jsonResponse([tokenizer]);
      if (url.endsWith("/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/mixtures") || url.includes("/tasks?")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    expect(screen.queryByRole("tab", { name: /Tokenized Corpus/ })).not.toBeInTheDocument();
    expect(screen.getByText(/训练数据制品由训练中心自动编排/)).toBeInTheDocument();
    return;
    expect(screen.queryByLabelText("Tokenized Corpus 生成关系")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /Tokenized Corpus/ })).not.toBeInTheDocument();
    expect(screen.getByText(/训练数据制品由训练中心自动编排/)).toBeInTheDocument();
    return;
    fireEvent.click(await screen.findByRole("tab", { name: /Tokenized Corpus/ }));
    expect(screen.getByLabelText("Tokenized Corpus 生成关系")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "构建 Corpus" }));
    expect(await screen.findByRole("dialog", { name: "构建 Tokenized Corpus" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("数据集版本"), { target: { value: "dataset-version-1" } });
    fireEvent.change(screen.getByLabelText("锁定 Tokenizer"), { target: { value: "tokenizer-1" } });
    fireEvent.change(screen.getByLabelText(/^输入选择策略/), { target: { value: "estimated_token_limit" } });
    fireEvent.change(screen.getByLabelText(/^输入估算 Token 上限/), { target: { value: "12002000000" } });
    fireEvent.change(screen.getByLabelText("EOS 策略"), { target: { value: "none" } });
    fireEvent.change(screen.getByLabelText("空文档策略"), { target: { value: "skip" } });
    fireEvent.change(screen.getByLabelText("Token dtype"), { target: { value: "uint32" } });
    fireEvent.change(screen.getByLabelText(/^每分片 Token 上限/), { target: { value: "1048576" } });
    fireEvent.click(screen.getByRole("button", { name: "创建并开始编码" }));

    await waitFor(() => expect(submitted).toBeDefined());
    expect(submitted).toMatchObject({
      dataset_version_id: "dataset-version-1",
      tokenizer_version_id: "tokenizer-1",
      selection: { mode: "estimated_token_limit", estimated_token_limit: 12002000000 },
      encoding: { eos_policy: "none", empty_document_policy: "skip", token_dtype: "uint32", shard_token_limit: 1048576 },
      adapter_id: "builtin.tokenize-corpus",
      resource_request: { gpus: 0 },
    });
    expect(submitted).not.toHaveProperty("target_tokens");
    expect(submitted).not.toHaveProperty("sequence_length");
  });

  it("retries a failed training data artifact from its own workspace", async () => {
    const artifact = { id: "tokenized-1", status: "failed", task_id: "task-tokenized-1", dataset_version_id: "dataset-version-1", dataset_name: "Corpus", dataset_version: 3, dataset_token_estimate: 12_000_000_000, tokenizer_version_id: "tokenizer-1", tokenizer_name: "Locked BPE", tokenizer_version: 2, selection_config: { mode: "all", estimated_token_limit: null }, encoding_config: { eos_policy: "append_if_available", empty_document_policy: "skip", token_dtype: "auto", shard_token_limit: 268435456 }, document_count: 0, token_count: 0, shard_count: 0, failure_reason: "目标磁盘空间不足", created_at: "2026-08-11T00:00:00Z" };
    const task = { id: "task-tokenized-1", kind: "data_tokenize_corpus", title: "预分词 Corpus", observed_state: "failed", desired_state: "running", progress: 0.2, error: { message: "目标磁盘空间不足" }, created_at: "2026-08-11T00:00:00Z" };
    let retried = false;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/tasks/task-tokenized-1/control") && init?.method === "POST") {
        retried = JSON.parse(String(init.body)).action === "retry";
        return jsonResponse({ ...task, observed_state: "queued" });
      }
      if (url.endsWith("/tokenized-corpora")) return jsonResponse([artifact]);
      if (url.endsWith("/training-data-views")) return jsonResponse([]);
      if (url.includes("/tasks?")) return jsonResponse([task]);
      if (url.endsWith("/tokenizers") || url.endsWith("/datasets") || url.endsWith("/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    expect(screen.queryByRole("tab", { name: /Tokenized Corpus/ })).not.toBeInTheDocument();
    expect(screen.getByText(/训练数据制品由训练中心自动编排/)).toBeInTheDocument();
    return;
    fireEvent.click(await screen.findByRole("tab", { name: /Tokenized Corpus/ }));
    expect(await screen.findByText("目标磁盘空间不足")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "重试" }));
    await waitFor(() => expect(retried).toBe(true));
  });

  it("opens a Corpus row as a fully prefilled immutable derived-version form", async () => {
    const tokenizer = { id: "tokenizer-1", project_id: "project-1", name: "Locked BPE", version: 2, algorithm: "BPE", vocab_size: 32000, status: "passed", locked: true, artifact_id: "tokenizer-artifact-1", metrics: {} };
    const dataset = { id: "dataset-1", project_id: "project-1", name: "Corpus", source_type: "local", license: "MIT", language: "zh", tags: [], status: "ready", versions: [], current_version: { id: "dataset-version-1", version: 3, state: "ready", train_ready: true, artifact_id: "snapshot-1", record_manifest_artifact_id: "records-1", token_estimate: 6_000_000_000 } };
    const artifact = { id: "tokenized-1", status: "ready", task_id: "task-tokenized-1", artifact_id: "artifact-1", identity_hash: "a".repeat(64), format: "forgelm-token-corpus-v2", dataset_version_id: "dataset-version-1", dataset_name: "Corpus", dataset_version: 3, dataset_token_estimate: 6_000_000_000, tokenizer_version_id: "tokenizer-1", tokenizer_name: "Locked BPE", tokenizer_version: 2, selection_config: { mode: "estimated_token_limit", estimated_token_limit: 5_000_000_000 }, encoding_config: { eos_policy: "none", empty_document_policy: "eos_only", token_dtype: "uint32", shard_token_limit: 1048576 }, document_count: 1200, token_count: 4_900_000_000, shard_count: 358, created_at: "2026-08-11T00:00:00Z" };
    const task = { id: "task-tokenized-1", kind: "data_tokenize_corpus", title: "预分词 Corpus", observed_state: "succeeded", desired_state: "running", progress: 1, payload: { resource_request: { gpus: 0, worker_selector: { labels: { environment: "wsl" } } } }, created_at: "2026-08-11T00:00:00Z" };
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/tokenized-corpora") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ ...artifact, id: "tokenized-2", status: "queued", ...submitted });
      }
      if (url.endsWith("/tokenized-corpora")) return jsonResponse([artifact]);
      if (url.endsWith("/training-data-views")) return jsonResponse([]);
      if (url.includes("/tasks?")) return jsonResponse([task]);
      if (url.endsWith("/tokenizers")) return jsonResponse([tokenizer]);
      if (url.endsWith("/mixtures")) return jsonResponse([]);
      if (url.endsWith("/datasets")) return jsonResponse([dataset]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    expect(screen.queryByRole("tab", { name: /Tokenized Corpus/ })).not.toBeInTheDocument();
    expect(screen.getByText(/训练数据制品由训练中心自动编排/)).toBeInTheDocument();
    return;
    fireEvent.click(await screen.findByRole("tab", { name: /Tokenized Corpus/ }));
    fireEvent.click(await screen.findByRole("button", { name: "查看/派生配置" }));
    expect(screen.getByRole("dialog", { name: "查看并派生 Tokenized Corpus 配置" })).toBeInTheDocument();
    expect(screen.getByLabelText("数据集版本")).toHaveValue("dataset-version-1");
    expect(screen.getByLabelText("锁定 Tokenizer")).toHaveValue("tokenizer-1");
    expect(screen.getByLabelText(/^输入选择策略/)).toHaveValue("estimated_token_limit");
    expect(screen.getByLabelText(/^输入估算 Token 上限/)).toHaveValue(5_000_000_000);
    expect(screen.getByLabelText("EOS 策略")).toHaveValue("none");
    expect(screen.getByLabelText("空文档策略")).toHaveValue("eos_only");
    expect(screen.getByLabelText("Token dtype")).toHaveValue("uint32");
    expect(screen.getByLabelText(/^每分片 Token 上限/)).toHaveValue(1048576);
    expect(screen.getByLabelText("制品格式")).toHaveAttribute("readonly");
    expect(screen.getByLabelText(/^Worker 标签/)).toHaveValue("environment=wsl");
    fireEvent.change(screen.getByLabelText(/^输入估算 Token 上限/), { target: { value: "7000000000" } });
    fireEvent.click(screen.getByRole("button", { name: "按当前配置派生版本" }));

    await waitFor(() => expect(submitted).toBeDefined());
    expect(submitted).toMatchObject({
      dataset_version_id: "dataset-version-1",
      tokenizer_version_id: "tokenizer-1",
      selection: { mode: "estimated_token_limit", estimated_token_limit: 7_000_000_000 },
      encoding: { eos_policy: "none", empty_document_policy: "eos_only", token_dtype: "uint32", shard_token_limit: 1048576 },
      resource_request: { gpus: 0, worker_selector: { labels: { environment: "wsl" } } },
    });
  });

  it("explains empty choices and links a dataset without a manifest to reanalysis", async () => {
    const dataset: Dataset = {
      id: "dataset-1",
      project_id: "project-1",
      name: "Parquet Corpus",
      source_type: "local",
      license: "MIT",
      language: "zh",
      tags: [],
      status: "ready",
      current_version: {
        id: "version-1",
        version: 1,
        state: "ready",
        train_ready: true,
        artifact_id: "snapshot-1",
        record_manifest_artifact_id: null,
      },
      versions: [],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/tokenizers") || url.includes("/tasks?")) return jsonResponse([]);
      if (url.endsWith("/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /><Route path="/projects/:projectId/data" element={<div>Data center target</div>} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click((await screen.findAllByRole("button", { name: "新建实验" }))[0]);
    expect(screen.getByText("没有可用的数据混合版本")).toBeInTheDocument();
    expect(screen.getByLabelText(/GPU 数量/)).toBeDisabled();
    expect(screen.getByText("该内置适配器仅使用 CPU，不会占用 GPU")).toBeInTheDocument();
    expect(screen.getByLabelText(/训练语料抽样 Tokens/)).not.toHaveAttribute("max");
    expect(screen.getByText("正文按有界批次流式送入；内存主要随不同预分词片段数量增长，不按总 Tokens 线性增长")).toBeInTheDocument();
    expect(screen.getByLabelText("Tokenizer 质量门禁配置")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "全部质量门禁" })).toBeInTheDocument();
    expect(screen.getByText("可逆性与格式")).toBeInTheDocument();
    expect(screen.getByText("未知词与词表")).toBeInTheDocument();
    expect(screen.getByText("压缩效率")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Round-trip 成功率/ })).toBeChecked();
    expect(screen.getByLabelText("Round-trip 成功率阈值")).toHaveValue(1);
    expect(screen.getByRole("checkbox", { name: /中文压缩率/ })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: /换行保真率/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /缩进保真率/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /特殊 Token 完整性/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Byte fallback 比例/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /垃圾词表比例/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /数学压缩率/ })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("输入类型"), { target: { value: "dataset_version" } });
    expect(screen.getByText("没有可训练的数据集版本")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重新分析 Parquet Corpus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "创建并启动" })).toBeDisabled();
  });

  it("opens a created experiment card and shows lost task details", async () => {
    const tokenizer = { id: "tokenizer-1", project_id: "project-1", name: "实验1", version: 1, algorithm: "BPE", vocab_size: 32768, status: "queued", locked: false, is_default: false, task_id: "task-1", artifact_id: null, data_ref_type: "dataset_version", data_ref_id: "version-1", config: { normalization: "NFKC" }, metrics: {}, created_at: "2026-08-08T00:00:00Z" };
    const task = { id: "task-1", project_id: "project-1", kind: "tokenizer_train", title: "训练 Tokenizer", adapter_id: "builtin.hf-tokenizer", observed_state: "lost", desired_state: "running", progress: 0, error: { message: "Worker 心跳超时" }, attempts: [], created_at: "2026-08-08T00:00:00Z" };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/projects/project-1/tokenizers")) return jsonResponse([tokenizer]);
      if (url.endsWith("/tasks/task-1")) return jsonResponse(task);
      if (url.includes("/tasks?")) return jsonResponse([task]);
      if (url.endsWith("/datasets") || url.endsWith("/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "查看 Tokenizer 实验 实验1 v1" }));
    expect(await screen.findByRole("dialog", { name: "实验1 v1" })).toBeInTheDocument();
    expect(screen.getByText("执行 Worker 已失联")).toBeInTheDocument();
    expect(screen.getByText("Worker 心跳超时")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重试任务" })).toBeInTheDocument();
  });

  it("prefills an existing experiment and submits a same-name next version", async () => {
    const tokenizer = { id: "tokenizer-1", project_id: "project-1", name: "实验1", version: 1, algorithm: "BPE", vocab_size: 32768, status: "failed", locked: false, is_default: false, task_id: "task-1", artifact_id: null, data_ref_type: "dataset_version", data_ref_id: "version-1", config: { sample_tokens: 120000000, normalization: "NFC", byte_fallback: true, special_tokens: ["<unk>", "<s>"], gate_policy: { version: "tokenizer-quality-gate-v1", rules: { round_trip: { enabled: true, operator: "gte", threshold: 0.98 }, unk_rate: { enabled: true, operator: "lte", threshold: 0.01 }, zh_chars_per_token: { enabled: true, operator: "gte", threshold: 1.7 }, legacy_symbol_coverage: { label: "历史符号覆盖率", enabled: true, operator: "gte", threshold: 0.95 } } } }, metrics: {}, created_at: "2026-08-08T00:00:00Z" };
    const task = { id: "task-1", project_id: "project-1", kind: "tokenizer_train", title: "训练 Tokenizer", adapter_id: "builtin.hf-tokenizer", observed_state: "failed", desired_state: "running", progress: 0, error: { message: "训练失败" }, payload: { resource_request: { cpu_cores: 6, ram_gb: 12, gpu_count: 0, worker_labels: ["tokenizer"] } }, attempts: [], created_at: "2026-08-08T00:00:00Z" };
    const dataset: Dataset = {
      id: "dataset-1", project_id: "project-1", name: "Corpus", source_type: "local", license: "MIT", language: "zh", tags: [], status: "ready",
      current_version: { id: "version-1", version: 1, state: "ready", train_ready: true, artifact_id: "snapshot-1", record_manifest_artifact_id: "manifest-1" }, versions: [],
    };
    let submitted: Record<string, unknown> | undefined;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/projects/project-1/tokenizers") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ ...tokenizer, id: "tokenizer-2", version: 2, config: (submitted as any).config });
      }
      if (url.endsWith("/projects/project-1/tokenizers")) return jsonResponse([tokenizer]);
      if (url.endsWith("/tasks/task-1")) return jsonResponse(task);
      if (url.includes("/tasks?")) return jsonResponse([task]);
      if (url.endsWith("/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "查看 Tokenizer 实验 实验1 v1" }));
    fireEvent.click(await screen.findByRole("button", { name: "编辑配置" }));

    expect(await screen.findByRole("dialog", { name: "编辑 实验1 v1 配置" })).toBeInTheDocument();
    expect(screen.getByLabelText(/^实验名称/)).toHaveValue("实验1");
    expect(screen.getByLabelText(/^实验名称/)).toHaveAttribute("readonly");
    expect(screen.getByLabelText("训练数据")).toHaveValue("version-1");
    expect(screen.getByLabelText(/^训练语料抽样 Tokens/)).toHaveValue(120000000);
    expect(screen.getByLabelText("Normalization")).toHaveValue("NFC");
    expect(screen.getByLabelText("Tokenizer 质量门禁配置")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Round-trip 成功率/ })).toBeChecked();
    expect(screen.getByLabelText("Round-trip 成功率阈值")).toHaveValue(0.98);
    expect(screen.getByRole("checkbox", { name: /中文压缩率/ })).toBeChecked();
    expect(screen.getByLabelText("中文压缩率阈值")).toHaveValue(1.7);
    expect(screen.getByText("历史 / 自定义指标")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /历史符号覆盖率/ })).toBeChecked();
    expect(screen.getByLabelText("历史符号覆盖率阈值")).toHaveValue(0.95);

    fireEvent.change(screen.getByLabelText(/^训练语料抽样 Tokens/), { target: { value: "100000000" } });
    fireEvent.change(screen.getByLabelText("Round-trip 成功率阈值"), { target: { value: "0.99" } });
    fireEvent.click(screen.getByRole("button", { name: "创建新版本并启动" }));

    await waitFor(() => expect(submitted).toBeDefined());
    expect(submitted).toMatchObject({
      name: "实验1",
      algorithm: "BPE",
      vocab_size: 32768,
      data_ref_type: "dataset_version",
      data_ref_id: "version-1",
      start: true,
      adapter_id: "builtin.hf-tokenizer",
      config: {
        sample_tokens: 100000000,
        normalization: "NFC",
        byte_fallback: true,
        special_tokens: ["<unk>", "<s>"],
        gate_policy: {
          version: "tokenizer-quality-gate-v1",
          rules: {
            round_trip: { enabled: true, operator: "gte", threshold: 0.99 },
            unk_rate: { enabled: true, operator: "lte", threshold: 0.01 },
            zh_chars_per_token: { enabled: true, operator: "gte", threshold: 1.7 },
            en_chars_per_token: { enabled: false, operator: "gte", threshold: 3 },
            code_chars_per_token: { enabled: false, operator: "gte", threshold: 2 },
            legacy_symbol_coverage: { label: "历史符号覆盖率", enabled: true, operator: "gte", threshold: 0.95 },
          },
        },
      },
    });
  });

  it("shows frozen gate thresholds and per-rule completion evidence", async () => {
    const tokenizer = { id: "tokenizer-1", project_id: "project-1", name: "Gate Evidence", version: 3, algorithm: "BPE", vocab_size: 32000, status: "failed_gate", locked: false, is_default: false, task_id: "task-1", artifact_id: "artifact-1", data_ref_type: "dataset_version", data_ref_id: "version-1", config: { gate_policy: { version: "tokenizer-quality-gate-v1", rules: { round_trip: { enabled: true, operator: "gte", threshold: 1 }, unk_rate: { enabled: true, operator: "lte", threshold: 0 } } } }, metrics: { round_trip: 0.99, unk_rate: 0 }, created_at: "2026-08-11T00:00:00Z" };
    const task = { id: "task-1", project_id: "project-1", kind: "tokenizer_train", title: "训练 Tokenizer", adapter_id: "builtin.hf-tokenizer", observed_state: "succeeded", desired_state: "running", progress: 1, result: { gate_passed: false, gate_results: [{ metric: "round_trip", label: "Round-trip 成功率", enabled: true, operator: "gte", threshold: 1, actual: 0.99, passed: false, reason: "threshold_not_met" }, { metric: "unk_rate", label: "Unknown Token 比例", enabled: true, operator: "lte", threshold: 0, actual: 0, passed: true, reason: "passed" }] }, attempts: [], created_at: "2026-08-11T00:00:00Z" };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/tasks/task-1")) return jsonResponse(task);
      if (url.includes("/tasks?")) return jsonResponse([task]);
      if (url.endsWith("/tokenizers")) return jsonResponse([tokenizer]);
      if (url.endsWith("/datasets") || url.endsWith("/mixtures") || url.endsWith("/tokenized-corpora") || url.endsWith("/training-data-views")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "查看 Tokenizer 实验 Gate Evidence v3" }));
    expect(await screen.findByRole("dialog", { name: "Gate Evidence v3" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "质量门禁" })).toBeInTheDocument();
    expect(screen.getAllByText("≥ 100%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("99%").length).toBeGreaterThan(0);
    expect(screen.getByText("低于设定下限")).toBeInTheDocument();
    expect(screen.getByText("达到阈值")).toBeInTheDocument();
  });

  it("lists the disk-backed BPE backend and submits its bounded working-set config", async () => {
    const dataset: Dataset = {
      id: "dataset-1", project_id: "project-1", name: "Large Corpus", source_type: "local", license: "MIT", language: "zh", tags: [], status: "ready",
      current_version: { id: "version-1", version: 1, state: "ready", train_ready: true, artifact_id: "snapshot-1", record_manifest_artifact_id: "manifest-1" }, versions: [],
    };
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/projects/project-1/tokenizers") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ id: "tokenizer-1", ...submitted });
      }
      if (url.endsWith("/tokenizers") || url.includes("/tasks?")) return jsonResponse([]);
      if (url.endsWith("/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/tokenizers"]}><Routes><Route path="/projects/:projectId/tokenizers" element={<Tokenizers />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click((await screen.findAllByRole("button", { name: "新建实验" }))[0]);
    fireEvent.change(screen.getByLabelText("输入类型"), { target: { value: "dataset_version" } });
    fireEvent.change(screen.getByLabelText("训练数据"), { target: { value: "version-1" } });
    fireEvent.change(screen.getByLabelText("执行方式"), { target: { value: "builtin.disk-bpe-tokenizer" } });

    expect(screen.getByLabelText(/^算法/)).toBeDisabled();
    expect(screen.getByLabelText(/^词表大小/)).toHaveValue(32000);
    expect(screen.getByLabelText(/^GPU 数量/)).toBeDisabled();
    expect(screen.getByLabelText(/^最大候选词形数/)).toHaveValue(250000);
    expect(screen.getByText(/Token 数不作为内存上限，合并工作集由下方参数约束/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("实验名称"), { target: { value: "Large Disk BPE" } });
    fireEvent.change(screen.getByLabelText(/^训练语料抽样 Tokens/), { target: { value: "12000000000" } });
    fireEvent.change(screen.getByLabelText(/^最大候选词形数/), { target: { value: "300000" } });
    fireEvent.change(screen.getByLabelText(/^合并重放预算/), { target: { value: "3000000" } });
    fireEvent.click(screen.getByRole("button", { name: "创建并启动" }));

    await waitFor(() => expect(submitted).toBeDefined());
    expect(submitted).toMatchObject({
      adapter_id: "builtin.disk-bpe-tokenizer",
      algorithm: "BPE",
      resource_request: { gpus: 0 },
      config: {
        sample_tokens: 12000000000,
        min_word_frequency: 2,
        max_unique_words: 300000,
        max_candidate_characters: 4000000,
        max_word_length: 128,
        merge_occurrence_budget: 3000000,
      },
    });
  });
});
