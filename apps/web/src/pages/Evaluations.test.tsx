import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Evaluations } from "./Evaluations";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderEvaluations() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/projects/project-1/evaluations"]}>
        <Routes>
          <Route path="/projects/:projectId/evaluations" element={<Evaluations />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Unified project evaluation datasets", () => {
  it("keeps frozen versions when selecting alongside a new built-in and prevents incompatible engine combinations", async () => {
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/benchmarks/catalog")) return jsonResponse([{ id: "lm-eval:c3", name: "C3" }, { id: "lm-eval:piqa", name: "PIQA" }]);
      if (url.endsWith("/evaluation-datasets")) return jsonResponse([
        { id: "frozen-c3", name: "C3", version: 2, source_type: "benchmark", benchmark_config: { id: "lm-eval:c3", execution: { num_fewshot: 3 } } },
        { id: "private", name: "私有题目", version: 1, source_type: "upload", artifact_id: "snapshot" },
      ]);
      if (url.endsWith("/evaluation-suites") && init?.method === "POST") { submitted = JSON.parse(String(init.body)); return jsonResponse({ id: "suite-new" }, 201); }
      return jsonResponse([]);
    }));
    renderEvaluations();
    fireEvent.click(screen.getByRole("button", { name: /^评测集/ }));
    expect(screen.queryByRole("button", { name: /^Benchmark/ })).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("checkbox", { name: "选择 C3 V2" }));
    expect(screen.queryByRole("checkbox", { name: "选择 C3" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择 PIQA" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "选择 私有题目 V1" }));
    fireEvent.click(screen.getByRole("button", { name: /用所选项组装套件/ }));
    expect(screen.getByRole("alert")).toHaveTextContent("不同执行器");
    expect(screen.getByRole("button", { name: "创建不可变套件" })).toBeDisabled();
    const dialog = screen.getByRole("dialog");
    fireEvent.click(dialog.querySelector<HTMLInputElement>('input[aria-label="选择 私有题目 V1"]')!);
    fireEvent.click(screen.getByRole("button", { name: "创建不可变套件" }));
    await waitFor(() => expect(submitted).toMatchObject({ datasets: [{ dataset_version_id: "frozen-c3", weight: 1 }], benchmark_ids: ["lm-eval:piqa"], release_gate: {} }));
  });
});

describe("Evaluation run archive", () => {
  it("opens one round with its goal, process, benchmark scores, and comparable baseline", async () => {
    const baseRun = {
      id: "run-1-previous",
      project_id: "project-1",
      task_id: "task-previous",
      status: "succeeded",
      model_version_ids: ["model-1"],
      suite_version_ids: ["suite-1"],
      inference_config: { sampling: { method: "stable_hash_v1", percentage: 1, seed: 42 } },
      compatibility_hash: "compatible-signature",
      created_at: "2026-08-19T00:00:00Z",
      config_snapshot: { models: [{ id: "model-1", name: "MicroLM", version: 6 }], suites: [{ id: "suite-1", name: "标准套件", version: 1, datasets: [{ task_id: "mmlu", name: "MMLU" }] }], benchmarks: [{ task_id: "mmlu", name: "MMLU" }], scoring_config: { aggregator: "weighted_mean" } },
      summary: { overall: 0.2, dimensions: { 知识: 0.2 }, models: { "model-1": { tasks: { mmlu: { benchmark_id: "lm-eval:mmlu", score: 0.2, metric: "acc", weight: 1, raw_metrics: { sample_len: 100 } } } } } },
    };
    const latestRun = {
      ...baseRun,
      id: "run-2-latest",
      task_id: "task-latest",
      model_version_ids: ["model-1", "model-2"],
      created_at: "2026-08-20T00:00:00Z",
      config_snapshot: { ...baseRun.config_snapshot, primary_model_version_id: "model-1", models: [{ id: "model-1", name: "MicroLM", version: 7 }, { id: "model-2", name: "Qwen", version: 1 }] },
      summary: { primary_model_version_id: "model-1", overall: 0.2396, dimensions: { 知识: 0.26 }, aggregation: { method: "weighted_mean" }, execution: { batch_policy: "auto", requested_batch_size: "auto", sampling: { benchmarks: { mmlu: { plan_hash: "abcdef1234567890" } } }, models: { "model-1": { mmlu: { effective_batch_sizes: [4], oom_backoffs: 1 } } } }, models: { "model-1": { overall: 0.2396, dimensions: { 知识: 0.26 }, tasks: { mmlu: { benchmark_id: "lm-eval:mmlu", score: 0.2381, metric: "acc", weight: 1, raw_metrics: { sample_len: 168 } } } }, "model-2": { overall: 0.31, dimensions: { 知识: 0.34 }, tasks: { mmlu: { benchmark_id: "lm-eval:mmlu", score: 0.31, metric: "acc", weight: 1, raw_metrics: { sample_len: 168 } } } } } },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([latestRun, baseRun]);
      if (url.includes("/api/tasks/task-latest?include_events=false")) return jsonResponse({ id: "task-latest", kind: "evaluate", title: "模型评测", observed_state: "succeeded", desired_state: "running", current_phase: "completed", progress: 1, created_at: latestRun.created_at, started_at: latestRun.created_at, ended_at: "2026-08-20T02:44:00Z", timing_estimate: { status: "complete", actual_duration_seconds: 9840, sample_count: 0 }, attempts: [{ id: "attempt-1", attempt_number: 1, status: "succeeded", worker_id: "worker-1", exit_code: 0 }] });
      if (url.includes("/api/projects/project-1/evaluations/run-2-latest/samples?")) return jsonResponse({ items: [{ id: "question-1", question: "Which answer is correct?", choices: ["No", "Yes"], reference_answer: { index: 1, label: "B", value: "Yes" }, model_answer: { index: 1, label: "B", value: "Yes" }, metric: "acc", score: 1, status: "passed", metrics: { acc: 1 }, metadata: { subject: "logic" } }], offset: 0, limit: 20, has_more: false, source: "lm-eval-report/model-1/mmlu/samples.jsonl" });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    expect(await screen.findByText("评测轮次")).toBeInTheDocument();
    expect(screen.getAllByRole("cell", { name: /MicroLM v7/ })).toHaveLength(2);
    expect(screen.getByRole("cell", { name: /Qwen v1 31.0%/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("row", { name: "打开第 2 轮评测" }));

    expect(await screen.findByRole("dialog")).toHaveTextContent("第 2 轮评测");
    expect(screen.getByRole("dialog")).toHaveTextContent("评测目标");
    expect(screen.getByRole("dialog")).toHaveTextContent("标准套件 v1");
    expect(screen.getByRole("dialog")).toHaveTextContent("评测过程");
    await waitFor(() => expect(screen.getByRole("dialog")).toHaveTextContent("worker-1"));
    expect(screen.getByRole("dialog")).toHaveTextContent("实际执行 2 小时 44 分钟");
    expect(screen.getByRole("dialog")).toHaveTextContent("自动（请求 auto）");
    expect(screen.getByRole("dialog")).toHaveTextContent("1% · Seed 42 · stable_hash_v1");
    expect(screen.getByRole("dialog")).toHaveTextContent("abcdef123456");
    expect(screen.getByRole("dialog")).toHaveTextContent("4 · OOM 降档 1 次");
    expect(screen.getByRole("dialog")).toHaveTextContent("23.96%");
    expect(screen.getByRole("dialog")).toHaveTextContent("Qwen v1");
    expect(screen.getByRole("dialog")).toHaveTextContent("31.00%");
    expect(screen.getByRole("dialog")).toHaveTextContent("Σ（计入项目得分 × 项目权重）÷ Σ项目权重");
    expect(screen.getByRole("dialog")).toHaveTextContent("+3.96 pp");
    expect(screen.getByRole("dialog")).toHaveTextContent("168");
    expect(screen.getByRole("dialog")).toHaveTextContent("23.81%");
    fireEvent.click(screen.getByRole("button", { name: "查看 MMLU MicroLM v7 逐题结果" }));
    const samplesDialog = await screen.findByRole("dialog", { name: /MMLU · 逐题结果/ });
    await waitFor(() => expect(samplesDialog).toHaveTextContent("Which answer is correct?"));
    expect(samplesDialog).toHaveTextContent("模型回答");
    expect(samplesDialog).toHaveTextContent("逐题报告已持久化");
    expect(samplesDialog).toHaveTextContent("B. Yes");
    expect(samplesDialog).toHaveTextContent("100.00%");
  });
});

describe("Benchmark catalog", () => {
  it("applies the pretrained base-model preset with frozen zero-shot overrides", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const benchmarkIds = ["wikitext", "lambada_openai", "hellaswag", "winogrande", "piqa", "arc_easy", "truthfulqa_mc2"];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse(benchmarkIds.map((taskId) => ({ id: `lm-eval:${taskId}`, provider: "lm-eval", task_id: taskId, name: taskId, description: `${taskId} description`, languages: ["en"], dimensions: ["语言"], primary_metric: taskId === "wikitext" ? "word_perplexity" : "acc", estimated_cost: "medium", defaults: { num_fewshot: taskId === "hellaswag" ? 10 : 0 } })));
      if (url.endsWith("/api/benchmarks/suite-presets")) return jsonResponse([{ id: "pretrained-base-language-general-v1", name: "预训练基模 · 语言与通识", description: "基模能力", benchmark_ids: benchmarkIds.map((id) => `lm-eval:${id}`), overrides: { "lm-eval:hellaswag": { num_fewshot: 0 }, "lm-eval:winogrande": { num_fewshot: 0 } }, scoring_config: { aggregator: "weighted_mean", preset_id: "pretrained-base-language-general-v1", recommended_inference_config: { apply_chat_template: false, sample_percentage: 10, sample_seed: 42 } }, notes: ["困惑度单独展示"] }]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/benchmark-suites") && init?.method === "POST") return jsonResponse({ id: "suite-base", datasets: [] }, 201);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    fireEvent.click(await screen.findByRole("button", { name: /评测套件/ }));
    expect(screen.getByText("套件模板（尚未创建）")).toBeInTheDocument();
    expect(screen.getByText("已创建的评测套件")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "从模板创建套件" }));
    expect(screen.getByRole("dialog").querySelectorAll(".selected-benchmark-list article")).toHaveLength(7);
    expect(Array.from(screen.getByRole("dialog").querySelectorAll(".selected-benchmark-list b")).map((item) => item.textContent)).toEqual(Array(7).fill("0-shot"));
    expect(screen.getByLabelText("套件名称")).toHaveValue("预训练基模 · 语言与通识");
    fireEvent.click(screen.getByRole("button", { name: "创建不可变套件" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/benchmark-suites") && request.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/api/projects/project-1/benchmark-suites") && item.init?.method === "POST");
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({
      benchmark_ids: benchmarkIds.map((id) => `lm-eval:${id}`),
      overrides: { "lm-eval:hellaswag": { num_fewshot: 0 }, "lm-eval:winogrande": { num_fewshot: 0 } },
      scoring_config: { preset_id: "pretrained-base-language-general-v1" },
      release_gate: {},
    });
  });

  it("keeps presets out of the Benchmark catalog and labels created suites by asset type", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([{ id: "lm-eval:c3", provider: "lm-eval", task_id: "c3", name: "C3", description: "中文阅读理解", languages: ["zh"], dimensions: ["中文阅读"], primary_metric: "acc_norm", estimated_cost: "medium", defaults: { num_fewshot: 0 } }]);
      if (url.endsWith("/api/benchmarks/suite-presets")) return jsonResponse([{ id: "zh-base", name: "中文基模", description: "中文能力组合", benchmark_ids: ["lm-eval:c3"] }]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([{ id: "dataset-c3", name: "C3 中文阅读", task_type: "multiple_choice", benchmark_config: { name: "C3 中文阅读", task_id: "c3", task_type: "multiple_choice", primary_metric: "acc_norm", execution: { num_fewshot: 0 }, config_hash: "1234567890abcdef" } }]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([{ id: "suite-1", name: "中文诊断", version: 2, created_at: "2026-08-27T10:00:00Z", datasets: [{ dataset_version_id: "dataset-c3", benchmark_id: "lm-eval:c3", task_id: "c3", weight: 1, config_hash: "1234567890abcdef" }], scoring_config: { asset_type: "benchmark", aggregator: "weighted_mean", preset_id: "zh-base", recommended_inference_config: { sample_percentage: 5, sample_seed: 42, apply_chat_template: false } }, release_gate: { overall: 0.6 } }]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    fireEvent.click(await screen.findByRole("button", { name: /^评测集/ }));
    expect(screen.getByText("评测集（Benchmark）")).toBeInTheDocument();
    expect(screen.queryByText("套件模板（尚未创建）")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /评测套件/ }));
    expect(await screen.findByText("标准套件")).toBeInTheDocument();
    expect(screen.getAllByText("1 个评测项目")).toHaveLength(2);
    expect(screen.getByText("加权平均")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "查看套件 中文诊断 V2" }));
    const dialog = screen.getByRole("dialog", { name: "中文诊断 · V2" });
    expect(dialog).toHaveTextContent("标准 Benchmark 套件");
    expect(dialog).toHaveTextContent("C3 中文阅读");
    expect(dialog).toHaveTextContent("acc_norm");
    expect(dialog).toHaveTextContent("题目比例5%");
    expect(dialog).toHaveTextContent("抽样 Seed42");
    expect(dialog).toHaveTextContent("Chat Template关闭");
  });

  it("creates a versioned suite from selected catalog tasks", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([
        {
          id: "lm-eval:mmlu",
          provider: "lm-eval",
          task_id: "mmlu",
          name: "MMLU",
          description: "多学科知识与推理",
          languages: ["en"],
          dimensions: ["知识", "推理"],
          primary_metric: "acc",
          estimated_cost: "high",
          defaults: { num_fewshot: 5 },
        },
      ]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites") && init?.method === "POST") {
        return jsonResponse({ id: "suite-1", name: "Regression suite", datasets: [] }, 201);
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    fireEvent.click(await screen.findByRole("button", { name: /^评测集/ }));
    fireEvent.click(await screen.findByLabelText("选择 MMLU"));
    fireEvent.click(screen.getByRole("button", { name: /用所选项组装套件/ }));
    fireEvent.change(screen.getByLabelText("套件名称"), { target: { value: "Regression suite" } });
    fireEvent.click(screen.getByRole("button", { name: "创建不可变套件" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/evaluation-suites") && request.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/api/projects/project-1/evaluation-suites") && item.init?.method === "POST");
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({
      name: "Regression suite",
      benchmark_ids: ["lm-eval:mmlu"],
      scoring_config: { aggregator: "weighted_mean" },
    });
  });

  it("disables empty suites and explains how to create a runnable suite", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [{ id: "model-1", name: "ExampleLM", version: 1, kind: "sft", status: "candidate", metadata_json: {} }] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([{ id: "suite-empty", name: "Empty suite", version: 1, datasets: [], scoring_config: {}, release_gate: {} }]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    const launchButtons = await screen.findAllByRole("button", { name: "发起评测" });
    fireEvent.click(launchButtons[0]);

    expect(screen.getByRole("checkbox", { name: /Empty suite/ })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("请先选择评测集创建套件");
    expect(screen.queryByLabelText("可执行文件")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "启动评测" })).toBeDisabled();
  });

  it("selects a registered endpoint as a comparison target and freezes the primary model", async () => {
    let submitted: any;
    const localModel = { id: "model-local", name: "MicroLM", version: 7, kind: "base", status: "candidate", metadata_json: { lm_eval: { backend: "hf", model_args: "pretrained=/models/microlm" } }, inference_source: { type: "managed" } };
    const endpointModel = { id: "model-endpoint", name: "Ollama Qwen", version: 1, kind: "sft", status: "registered", metadata_json: { lm_eval: { backend: "local-completions", model_args: "model=qwen,base_url=http://localhost:11434/v1/completions", capabilities: ["generate", "loglikelihood"], endpoint_mode: "completions" }, openai_endpoint: { mode: "completions" } }, inference_source: { type: "api", endpoint_url: "http://localhost:11434/v1/completions", endpoint_model: "qwen", endpoint_mode: "completions" } };
    const suite = { id: "suite-1", name: "Standard suite", version: 1, datasets: [{ benchmark_id: "lm-eval:mmlu", task_id: "mmlu", task_type: "multiple_choice" }], scoring_config: { asset_type: "benchmark" } };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [localModel, endpointModel] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([suite]);
      if (url.endsWith("/api/projects/project-1/evaluations") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ id: "run-created" }, 201);
      }
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    fireEvent.click((await screen.findAllByRole("button", { name: "发起评测" }))[0]);
    fireEvent.click(screen.getByLabelText("选择 Ollama Qwen v1"));
    expect(screen.getByText("Completions · qwen")).toBeInTheDocument();
    expect(screen.getByLabelText("Batch 策略")).toHaveValue("auto");
    expect(screen.getByLabelText("手动 Batch size")).toHaveValue(1);
    expect(screen.getByLabelText("使用题目比例（%）")).toHaveValue(100);
    expect(screen.getByLabelText("抽样 Seed")).toHaveValue(42);
    fireEvent.click(screen.getByRole("button", { name: "启动评测" }));

    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted.model_version_ids).toEqual(["model-local", "model-endpoint"]);
    expect(submitted.primary_model_version_id).toBe("model-local");
    expect(submitted.inference_config.batch_size).toBe("auto");
    expect(submitted.inference_config.sample_percentage).toBe(100);
    expect(submitted.inference_config.sample_seed).toBe(42);
    expect(submitted.inference_config.limit).toBeUndefined();
    expect(submitted.inference_config.models).toBeUndefined();
  });

  it("uses the same native MicroLM inference profile as Playground", async () => {
    let submitted: any;
    const nativeModel = { id: "model-microlm", name: "MicroLM", version: 8, kind: "base", status: "candidate", metadata_json: { inference: { format: "microlm-hybrid-safetensors", adapter_id: "builtin.microlm-inference" } }, inference_source: { type: "native", adapter_id: "builtin.microlm-inference", display_name: "MicroLM 原生推理" } };
    const suite = { id: "suite-1", name: "Standard suite", version: 1, datasets: [{ benchmark_id: "lm-eval:mmlu", task_id: "mmlu", task_type: "multiple_choice" }], scoring_config: { asset_type: "benchmark" } };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [nativeModel] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([suite]);
      if (url.endsWith("/api/projects/project-1/evaluations") && init?.method === "POST") { submitted = JSON.parse(String(init.body)); return jsonResponse({ id: "run-created" }, 201); }
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    fireEvent.click((await screen.findAllByRole("button", { name: "发起评测" }))[0]);
    expect(screen.getByText("MicroLM 原生推理")).toBeInTheDocument();
    expect(screen.queryByLabelText(/MicroLM v8 backend/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "启动评测" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "启动评测" }));
    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted.inference_config.models).toBeUndefined();
  });

  it("uploads a custom evaluation dataset with frozen field mappings", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets/upload") && init?.method === "POST") {
        return jsonResponse({ id: "eval-data-1", name: "Private regression", version: 1 }, 201);
      }
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    await screen.findByText("EVALUATION CENTER");
    const datasetTab = screen.getByRole("button", { name: /^评测集/ });
    expect(datasetTab).not.toBeNull();
    fireEvent.click(datasetTab!);
    const createButton = screen.getByRole("button", { name: "添加评测集" });
    expect(createButton).not.toBeNull();
    fireEvent.click(createButton!);

    const form = document.querySelector<HTMLFormElement>(".modal form");
    expect(form).not.toBeNull();
    fireEvent.change(form!.querySelector<HTMLInputElement>('input[name="name"]')!, { target: { value: "Private regression" } });
    const file = new File(['{"prompt":"2+2","answer":"4"}\n'], "private.jsonl", { type: "application/x-ndjson" });
    fireEvent.change(form!.querySelector<HTMLInputElement>('input[name="file"]')!, { target: { files: [file] } });
    fireEvent.submit(form!);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/evaluation-datasets/upload") && request.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/api/projects/project-1/evaluation-datasets/upload") && item.init?.method === "POST");
    expect(request?.init?.body).toBeInstanceOf(FormData);
    const body = request?.init?.body as FormData;
    expect(body.get("name")).toBe("Private regression");
    expect(body.get("file")).toBeInstanceOf(File);
    expect(JSON.parse(String(body.get("schema_config")))).toMatchObject({
      prompt_field: "prompt",
      answer_field: "answer",
    });
    expect(JSON.parse(String(body.get("scoring_config")))).toEqual({ metric: "exact_match" });
  });

  it("reuses an immutable Data Center version for evaluation", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [] });
      if (url.endsWith("/api/benchmarks/catalog")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluation-datasets/from-data-version") && init?.method === "POST") {
        return jsonResponse({ id: "eval-data-2", name: "HF regression", version: 1 }, 201);
      }
      if (url.endsWith("/api/projects/project-1/evaluation-datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([{
        id: "dataset-1",
        name: "HF prompts",
        source_type: "huggingface",
        versions: [{ id: "data-version-1", version: 3, artifact_id: "artifact-1", document_count: 120 }],
      }]);
      if (url.endsWith("/api/projects/project-1/evaluation-suites")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/evaluations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderEvaluations();
    await screen.findByText("EVALUATION CENTER");
    fireEvent.click(screen.getByRole("button", { name: /^评测集/ })!);
    fireEvent.click(screen.getByRole("button", { name: "添加评测集" })!);
    fireEvent.click(document.querySelector<HTMLButtonElement>(".source-mode-switch button:nth-child(2)")!);

    const form = document.querySelector<HTMLFormElement>(".modal form")!;
    fireEvent.change(form.querySelector<HTMLInputElement>('input[name="name"]')!, { target: { value: "HF regression" } });
    fireEvent.change(form.querySelector<HTMLSelectElement>('select[name="dataset_version_id"]')!, { target: { value: "data-version-1" } });
    fireEvent.submit(form);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/evaluation-datasets/from-data-version") && request.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/api/projects/project-1/evaluation-datasets/from-data-version") && item.init?.method === "POST");
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({
      name: "HF regression",
      dataset_version_id: "data-version-1",
      schema_config: { prompt_field: "prompt", answer_field: "answer" },
      scoring_config: { metric: "exact_match" },
    });
  });
});
