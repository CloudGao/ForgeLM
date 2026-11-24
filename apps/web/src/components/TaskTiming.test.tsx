import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Task } from "../types";
import { formatDuration, TaskTiming } from "./TaskTiming";

function task(status: Task["timing_estimate"]): Task {
  return {
    id: "task-1",
    kind: "train",
    title: "训练",
    observed_state: "running",
    desired_state: "running",
    progress: 0.5,
    created_at: "2026-08-10T03:00:00Z",
    timing_estimate: status,
  };
}

describe("TaskTiming", () => {
  it("formats compact Chinese durations", () => {
    expect(formatDuration(45)).toBe("不到 1 分钟");
    expect(formatDuration(3660)).toBe("1 小时 1 分钟");
    expect(formatDuration(180000)).toBe("2 天 2 小时");
  });

  it("shows total, remaining and completion time for an available estimate", () => {
    render(<TaskTiming detail task={task({
      status: "available",
      actual_duration_seconds: 600,
      estimated_total_seconds: 1200,
      remaining_seconds: 600,
      estimated_completion_at: "2026-08-10T04:10:00Z",
      sample_count: 3,
    })} />);

    expect(screen.getByText("预计总执行 20 分钟")).toBeInTheDocument();
    expect(screen.getByText("剩余约 10 分钟")).toBeInTheDocument();
    expect(screen.getByText(/预计完成/)).toBeInTheDocument();
  });

  it("does not invent a duration while collecting samples", () => {
    render(<TaskTiming task={task({
      status: "estimating",
      actual_duration_seconds: 20,
      estimated_total_seconds: null,
      remaining_seconds: null,
      estimated_completion_at: null,
      sample_count: 1,
    })} />);

    expect(screen.getByText("正在积累有效进度，稍后估算耗时")).toBeInTheDocument();
  });
});
