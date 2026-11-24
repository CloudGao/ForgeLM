import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TaskEvent } from "../types";
import { GenerationThroughput, ResourceUsageChart, TerminalLogViewer } from "./TaskObservability";

function logEvent(index: number): TaskEvent {
  return {
    id: index + 1,
    sequence: index + 1,
    event_type: "log",
    payload: { stream: "stdout", message: `line-${index}` },
    created_at: `2026-08-08T12:${String(Math.floor(index / 60)).padStart(2, "0")}:${String(index % 60).padStart(2, "0")}Z`,
  };
}

describe("task observability", () => {
  afterEach(cleanup);
  it("separates raw and accepted throughput and keeps missing historical metrics explicit", () => {
    render(<GenerationThroughput events={[{ ...logEvent(1), event_type: "progress", payload: {
      phase: "generating", recent_tokens_per_second: 150, recent_raw_tokens_per_second: 1500,
      rate_window_seconds: 120, raw_completed_requests: 500, retry_generated_tokens: 7000,
      rejected_generated_tokens: 8000, document_count: 100, batch_size: 32, submission_window: 32,
    }}]} />);
    expect(screen.getByText("150.0")).toBeInTheDocument();
    expect(screen.getByText("1500.0")).toBeInTheDocument();
    expect(screen.getByText("本次未收录 tokens")).toBeInTheDocument();
  });
  it("shows raw core usage without clipping a multi-core process to 100%", () => {
    render(<ResourceUsageChart events={[{id: 2, sequence: 2, event_type: "resource.sample", created_at: "2026-09-11T00:00:00Z", payload: {
      process: {cpu_percent: 147, cpu_allocation_percent: 18.375, allocated_cpus: 8}, system: {}, gpus: [],
    }}]} />);
    expect(screen.getByText(/进程实际使用 1.47 核/)).toBeInTheDocument();
    expect(screen.getByText("18.4%")).toBeInTheDocument();
    expect(screen.queryByText("100%")).not.toBeInTheDocument();
  });
  it("renders resource samples as one percentage chart", () => {
    const events: TaskEvent[] = [{
      id: 1,
      sequence: 1,
      event_type: "resource.sample",
      created_at: "2026-08-08T12:00:00Z",
      payload: {
        process: { cpu_percent: 42, rss_bytes: 2_000 },
        system: { cpu_percent: 36, memory_percent: 58, memory_total_bytes: 10_000 },
        gpus: [{ utilization_percent: 75, memory_used_mb: 8_000, memory_total_mb: 16_000 }],
      },
    }];

    render(<ResourceUsageChart events={events} />);

    expect(screen.getByLabelText("资源使用百分比折线图")).toBeInTheDocument();
    expect(screen.getByText("系统 CPU")).toBeInTheDocument();
    expect(screen.getByText("系统内存")).toBeInTheDocument();
    expect(screen.getByText("GPU 显存")).toBeInTheDocument();
  });

  it("keeps a bounded terminal, pauses follow on scroll, and can request older records", () => {
    const onLoadOlder = vi.fn();
    render(<TerminalLogViewer events={Array.from({ length: 300 }, (_, index) => logEvent(index))} hasMore onLoadOlder={onLoadOlder} />);

    expect(screen.getByText("300 条记录 · 实时跟随")).toBeInTheDocument();
    const viewport = screen.getByLabelText("运行日志");
    Object.defineProperties(viewport, {
      scrollHeight: { configurable: true, value: 1200 },
      clientHeight: { configurable: true, value: 300 },
    });
    viewport.scrollTop = 200;
    fireEvent.scroll(viewport);
    expect(screen.getByText("300 条记录 · 已暂停跟随")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "回到底部" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "加载更早" }));
    expect(onLoadOlder).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "回到底部" }));
    expect(viewport.scrollTop).toBe(1200);
  });
});
