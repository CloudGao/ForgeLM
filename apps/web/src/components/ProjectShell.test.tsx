import "@testing-library/jest-dom/vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Task } from "../types";
import { ProjectShell } from "./ProjectShell";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function downloadTask(id: string, title: string, state: string): Task {
  return {
    id,
    project_id: "project-1",
    kind: "data_import",
    title,
    adapter_id: "builtin.data-import",
    observed_state: state,
    desired_state: state === "paused" ? "paused" : "running",
    progress: 0.42,
    current_phase: "downloading",
    payload: { source_type: "huggingface" },
    created_at: "2026-08-06T00:00:00Z",
  };
}

function renderShell(dataPage: ReactNode = <div>Data page</div>, initialEntry = "/projects/project-1/data") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/projects/:projectId" element={<ProjectShell />}>
            <Route path="overview" element={<div>Overview page</div>} />
            <Route path="pipelines" element={<div>Pipeline page</div>} />
            <Route path="data" element={dataPage} />
            <Route path="tokenizers" element={<div>Tokenizer page</div>} />
            <Route path="training" element={<TrainingDestination />} />
            <Route path="models" element={<div>Models page</div>} />
            <Route path="evaluations" element={<div>Evaluations page</div>} />
            <Route path="releases" element={<div>Releases page</div>} />
            <Route path="playground" element={<div>Playground page</div>} />
          </Route>
          <Route path="/projects" element={<div>Project center destination</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function TrainingDestination() {
  const location = useLocation();
  return <div>Training destination {location.search}</div>;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Task center controls", () => {
  it("makes training and project asset workspaces peers", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM" });
      if (url.includes("/api/tasks?project_id=project-1") || url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({}, 404);
    }));
    renderShell();
    const navigation = screen.getByRole("navigation", { name: "项目导航" });
    expect(within(navigation).getByRole("link", { name: "训练" })).toHaveAttribute("href", "/projects/project-1/training");
    expect(within(navigation).getByRole("link", { name: "项目数据集" })).toHaveClass("active");
    expect(within(navigation).getAllByRole("link")).toHaveLength(8);
    expect(within(navigation).queryByRole("link", { name: "Pipeline" })).not.toBeInTheDocument();
    expect(within(navigation).queryByRole("group")).not.toBeInTheDocument();
    expect(within(navigation).getByRole("link", { name: "推理 Playground" })).toBeInTheDocument();
  });

  it("keeps the project shell visible while a page module is suspended", () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse([]);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    const neverResolves = new Promise<never>(() => {});
    function SuspendedPage(): never { throw neverResolves; }
    renderShell(<SuspendedPage />);

    expect(screen.getByText("ForgeLM")).toBeInTheDocument();
    expect(screen.getByRole("navigation")).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "正在加载页面内容…" })).toBeInTheDocument();
  });

  it("keeps the project shell visible when a page throws during rendering", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse([]);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    function BrokenPage(): never { throw new Error("Pipeline 校验报告字段缺失"); }
    renderShell(<BrokenPage />);

    expect(screen.getByText("ForgeLM")).toBeInTheDocument();
    expect(screen.getByRole("navigation")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("这个页面遇到了异常");
    expect(screen.getByRole("alert")).toHaveTextContent("Pipeline 校验报告字段缺失");
  });

  it("returns to the dedicated project center route", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse([]);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: "项目中心" }));
    expect(await screen.findByText("Project center destination")).toBeInTheDocument();
  });

  it("shows frozen training scope and opens the originating run configuration", async () => {
    const trainingTask: Task = {
      id: "task-training",
      project_id: "project-1",
      kind: "train",
      title: "训练：MicroLM-306M-v3 基座预训练 r23",
      adapter_id: "builtin.command.training",
      observed_state: "succeeded",
      desired_state: "running",
      progress: 1,
      current_phase: "completed",
      payload: {
        training_run_id: "run-100m",
        stages: [{ name: "Bootstrap · 100M 新链路重跑", target_tokens: 100_000_000, cumulative_target_tokens: 100_000_000 }],
        training_data_range: { planned_steps: 382, planned_train_tokens: 100_139_008 },
      },
      created_at: "2026-08-06T00:00:00Z",
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse([trainingTask]);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: /任务中心/ }));
    expect(await screen.findByText("Bootstrap · 100M 新链路重跑 · 本次 1亿 Tokens · 382 Steps")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "打开训练运行配置" })).toBeInTheDocument();

    fireEvent.click(screen.getByText("训练：MicroLM-306M-v3 基座预训练 r23"));
    expect(await screen.findByText("Training destination ?run=run-100m")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "任务中心" })).not.toBeInTheDocument();
  });

  it("paginates task history without rendering the next-page probe row", async () => {
    const firstPage = Array.from({ length: 21 }, (_, index) => downloadTask(`recent-${index}`, `最近任务 ${index}`, "succeeded"));
    const historicalTraining: Task = {
      id: "task-6b",
      project_id: "project-1",
      kind: "train",
      title: "训练：MicroLM-306M-v3 基座预训练 r25",
      adapter_id: "builtin.command.training",
      observed_state: "cancelled",
      desired_state: "cancelled",
      progress: 0.5116,
      current_phase: "Gate 2 · 6B 中期架构门禁",
      payload: {
        training_run_id: "run-6b",
        stages: [{ name: "Gate 2 · 6B 中期架构门禁", target_tokens: 5_900_000_000, cumulative_target_tokens: 6_000_000_000 }],
      },
      created_at: "2026-08-12T16:41:09Z",
    };
    const requestedOffsets: string[] = [];
    const requestedScopes: (string | null)[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) {
        const searchParams = new URL(url, "http://localhost").searchParams;
        const offset = searchParams.get("offset") || "0";
        requestedOffsets.push(offset);
        requestedScopes.push(searchParams.get("scope"));
        return jsonResponse(offset === "20" ? [historicalTraining] : firstPage);
      }
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: /任务中心/ }));
    expect(await screen.findByText("最近任务 0")).toBeInTheDocument();
    expect(screen.queryByText("最近任务 20")).not.toBeInTheDocument();
    expect(screen.getByText("第 1 页 · 每页 20 条")).toBeInTheDocument();
    expect(requestedScopes).toContain("pipeline");

    fireEvent.click(screen.getByRole("button", { name: "下一页" }));
    expect(await screen.findByText("Gate 2 · 6B 中期架构门禁 · 本次 59亿 Tokens · 累计到 60亿 Tokens")).toBeInTheDocument();
    expect(screen.getByText("第 2 页 · 每页 20 条")).toBeInTheDocument();
    expect(requestedOffsets).toContain("20");
    expect(requestedScopes.every((scope) => scope === "pipeline")).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "上一页" }));
    expect(await screen.findByText("最近任务 0")).toBeInTheDocument();
  });

  it("pauses a queued download, resumes a paused download, and stops any active task", async () => {
    let tasks = [
      downloadTask("task-queued", "排队下载", "queued"),
      downloadTask("task-paused", "暂停下载", "paused"),
      {
        ...downloadTask("task-tokenizer", "训练 Tokenizer", "running"),
        kind: "tokenizer_train",
        adapter_id: "builtin.hf-tokenizer",
        current_phase: "loading_corpus",
        payload: {},
        timing_estimate: {
          status: "available",
          actual_duration_seconds: 600,
          estimated_total_seconds: 1200,
          remaining_seconds: 600,
          estimated_completion_at: "2026-08-06T00:20:00Z",
          sample_count: 3,
        },
      },
      {
        id: "task-stopped-training",
        project_id: "project-1",
        kind: "train",
        title: "已停止的训练",
        adapter_id: "builtin.command.training",
        observed_state: "cancelled",
        desired_state: "cancelled",
        progress: 0.36,
        current_phase: "pretraining",
        payload: {
          training_run_id: "run-stopped",
          stages: [{ name: "预训练", target_tokens: 100_000_000 }],
        },
        created_at: "2026-08-06T00:00:00Z",
      },
    ];
    const controls: Array<{ id: string; body: any }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "42M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse(tasks);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      if (url.includes("/api/tasks/task-tokenizer?include_events=false") && (!init?.method || init.method === "GET")) return jsonResponse({
        ...tasks.find((task) => task.id === "task-tokenizer"),
        attempts: [{
          id: "attempt-1",
          attempt_number: 1,
          worker_id: "worker-1",
          status: "running",
          environment_snapshot: { cpu_count: 8, memory_total_bytes: 16 * 1024 ** 3 },
          resource_summary: { sample_count: 2, process: { peak_cpu_percent: 75, peak_rss_bytes: 512 * 1024 ** 2 }, gpus: [] },
          runtime_metrics_artifact_id: "metrics-artifact-1",
          log_summary: { line_count: 1, stdout_lines: 1 },
        }],
      });
      if (url.includes("/api/tasks/task-tokenizer/event-window?event_group=resources")) return jsonResponse({ items: [
          { id: 1, attempt_id: "attempt-1", sequence: 1, event_type: "resource.sample", payload: { process: { cpu_percent: 50, rss_bytes: 256 * 1024 ** 2 }, system: { memory_percent: 40 }, gpus: [] }, created_at: "2026-08-06T00:00:01Z" },
        ], has_more: false });
      if (url.includes("/api/tasks/task-tokenizer/event-window?event_group=logs")) return jsonResponse({ items: [
          { id: 2, attempt_id: "attempt-1", sequence: 2, event_type: "log", payload: { stream: "stdout", message: "loading corpus" }, created_at: "2026-08-06T00:00:02Z" },
        ], has_more: false });
      if (url.includes("/api/tasks/task-tokenizer/event-window?event_group=timeline")) return jsonResponse({ items: [], has_more: false });
      const match = url.match(/\/api\/tasks\/([^/]+)\/control$/);
      if (match && init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        controls.push({ id: match[1], body });
        tasks = tasks.map((task) => task.id !== match[1] ? task : {
          ...task,
          observed_state: body.action === "pause" ? "paused" : body.action === "resume" ? "queued" : "stopping",
          desired_state: body.action === "pause" ? "paused" : body.action === "resume" ? "running" : "cancelled",
        });
        return jsonResponse(tasks.find((task) => task.id === match[1]));
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: /任务中心/ }));
    expect(await screen.findByText("预计总执行 20 分钟")).toBeInTheDocument();
    expect(screen.getByText("剩余约 10 分钟")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "暂停下载 排队下载" }));
    await waitFor(() => expect(controls[0]).toMatchObject({ id: "task-queued", body: { action: "pause" } }));

    fireEvent.click(await screen.findByRole("button", { name: "继续下载 暂停下载" }));
    await waitFor(() => expect(controls[1]).toMatchObject({ id: "task-paused", body: { action: "resume" } }));

    fireEvent.click(await screen.findByRole("button", { name: "停止任务 训练 Tokenizer" }));
    await waitFor(() => expect(controls[2]).toEqual({
      id: "task-tokenizer",
      body: { action: "cancel", reason: "用户从任务中心停止任务" },
    }));
    expect(await screen.findByRole("button", { name: "停止任务 训练 Tokenizer" })).toBeDisabled();

    fireEvent.click(await screen.findByRole("button", { name: "继续执行 已停止的训练" }));
    await waitFor(() => expect(controls[3]).toEqual({
      id: "task-stopped-training",
      body: { action: "resume", reason: "用户从任务中心继续执行训练" },
    }));

    const tokenizerRow = screen.getByText("训练 Tokenizer").closest("article");
    expect(tokenizerRow).not.toBeNull();
    fireEvent.click(within(tokenizerRow!).getByRole("button", { name: "查看资源与日志" }));
    expect(await screen.findByRole("dialog", { name: "训练 Tokenizer" })).toBeInTheDocument();
    expect(screen.getAllByText("预计总执行 20 分钟")).toHaveLength(2);
    expect(screen.getByText(/预计完成/)).toBeInTheDocument();
    expect(screen.getByText("loading corpus")).toBeInTheDocument();
    expect(screen.getByLabelText("资源使用百分比折线图")).toBeInTheDocument();
    expect(screen.getByText(/1 条记录 · 实时跟随/)).toBeInTheDocument();
  });

  it("safely pauses checkpoint-capable training while retaining immediate stop", async () => {
    const safePauseControl = {
      safe_pause: { protocol: "forgelm-command-control-file-v1" },
    };
    let tasks: Task[] = [
      {
        id: "task-safe-running",
        project_id: "project-1",
        kind: "train",
        title: "安全暂停训练",
        adapter_id: "builtin.command.training",
        observed_state: "running",
        desired_state: "running",
        progress: 0.5,
        payload: {
          training_run_id: "run-safe",
          stages: [{ name: "预训练", target_tokens: 100_000_000 }],
          command: { control: safePauseControl },
        },
        created_at: "2026-08-21T00:00:00Z",
      },
      {
        id: "task-safe-paused",
        project_id: "project-1",
        kind: "train",
        title: "已安全暂停训练",
        adapter_id: "builtin.command.training",
        observed_state: "paused",
        desired_state: "paused",
        progress: 0.25,
        payload: {
          training_run_id: "run-paused",
          stages: [{ name: "预训练", target_tokens: 100_000_000 }],
          command: { control: safePauseControl },
        },
        created_at: "2026-08-21T00:00:00Z",
      },
    ];
    const controls: Array<{ id: string; body: any }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1")) return jsonResponse({ id: "project-1", name: "ExampleLM", target_model_name: "ExampleLM", target_parameters: "306M" });
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse(tasks);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      const match = url.match(/\/api\/tasks\/([^/]+)\/control$/);
      if (match && init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        controls.push({ id: match[1], body });
        tasks = tasks.map((task) => task.id !== match[1] ? task : {
          ...task,
          observed_state: body.action === "pause" ? "pausing" : body.action === "resume" ? "queued" : "stopping",
          desired_state: body.action === "pause" ? "paused" : body.action === "resume" ? "running" : "cancelled",
        });
        return jsonResponse(tasks.find((task) => task.id === match[1]));
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: /任务中心/ }));
    expect(await screen.findByRole("button", { name: "安全暂停 安全暂停训练" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "立即停止 安全暂停训练" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "安全暂停 安全暂停训练" }));
    await waitFor(() => expect(controls[0]).toEqual({
      id: "task-safe-running",
      body: { action: "pause", reason: "用户从任务中心安全暂停训练" },
    }));
    expect(await screen.findByText("正在完成当前 Step 并保存 Checkpoint…")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "立即停止 安全暂停训练" }));
    await waitFor(() => expect(controls[1]).toEqual({
      id: "task-safe-running",
      body: { action: "cancel", reason: "用户从任务中心立即停止训练" },
    }));

    fireEvent.click(screen.getByRole("button", { name: "继续执行 已安全暂停训练" }));
    await waitFor(() => expect(controls[2]).toEqual({
      id: "task-safe-paused",
      body: { action: "resume", reason: "用户从任务中心继续执行训练" },
    }));
  });
});
