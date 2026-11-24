import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TrainingRunDetail } from "./TrainingRunDetail";
import type { TrainingRun } from "../types";

const run: TrainingRun = { id: "run-1", project_id: "p", plan_revision_id: "revision", task_id: "task-1", status: "succeeded", current_step: 100, current_tokens: 1000, latest_metrics: {}, config_snapshot: {}, created_at: "2026-09-08T00:00:00Z" };
let requests: string[];
beforeEach(() => {
  requests = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost"); requests.push(url.href);
    const payload = url.pathname.endsWith("event-window") ? { items: url.searchParams.get("event_group") === "logs" ? [{ id: 1, sequence: 1, event_type: "log", attempt_id: url.searchParams.get("attempt_id"), payload: { message: `日志 ${url.searchParams.get("attempt_id")}`, stream: "stdout" }, created_at: run.created_at }] : [], has_more: false } : {
      id: "task-1", title: "训练", observed_state: "succeeded", progress: 1, attempts: [
        { id: "new-attempt", attempt_number: 2, status: "succeeded", started_at: run.created_at },
        { id: "old-attempt", attempt_number: 1, status: "failed", started_at: run.created_at },
      ],
    };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function show(initialAttemptId: string, displayRun: TrainingRun = run) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><TrainingRunDetail run={displayRun} initialAttemptId={initialAttemptId} onClose={vi.fn()} onOpenModel={vi.fn()} onOpenCheckpoint={vi.fn()} /></QueryClientProvider>);
}

it("shows frozen warmup ratio and resolved optimizer steps", () => {
  show("new-attempt", { ...run, config_snapshot: { learning_rate_schedules: [
    { stage_name: "预训练", warmup_ratio: 0.015, warmup_steps: 0, total_steps: 57221, resolved_warmup_steps: 859 },
  ] } });
  expect(screen.getByText(/比例 1.5%/)).toHaveTextContent("预热 859 / 总计 57,221 步");
});

it("does not present a pending packed budget as zero warmup", () => {
  show("new-attempt", { ...run, config_snapshot: { learning_rate_schedules: [
    { stage_name: "SFT", warmup_ratio: 0.03, total_steps: null, resolved_warmup_steps: null },
  ] } });
  expect(screen.getByText(/比例 3%/)).toHaveTextContent("等待引擎确定优化器总步数");
});

it("pins historical logs to the source attempt and supports explicit switching", async () => {
  show("old-attempt");
  expect(await screen.findByText("日志 old-attempt")).toBeInTheDocument();
  expect(requests.filter((url) => url.includes("event-window")).every((url) => url.includes("attempt_id=old-attempt"))).toBe(true);
  fireEvent.change(screen.getByRole("combobox", { name: "查看执行尝试" }), { target: { value: "new-attempt" } });
  expect(await screen.findByText("日志 new-attempt")).toBeInTheDocument();
  expect(screen.queryByText("日志 old-attempt")).not.toBeInTheDocument();
});

it("does not substitute the latest attempt when an explicit history reference is missing", async () => {
  show("missing-attempt");
  expect(await screen.findByRole("alert")).toHaveTextContent("该执行尝试不存在");
  await waitFor(() => expect(requests.some((url) => url.includes("task-1"))).toBe(true));
  expect(requests.some((url) => url.includes("event-window"))).toBe(false);
});
