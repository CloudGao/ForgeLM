import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingOptimizationComparison } from "./TrainingOptimizationComparison";
import type { Task } from "../types";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("freezes the selected provider in a new comparison and excludes unsupported engines", async () => {
  const writes: any[] = [];
  vi.stubGlobal("fetch", vi.fn(async (_url, init) => {
    if (init?.method === "POST") {
      writes.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ task_id: "comparison" }));
    }
    return new Response(JSON.stringify([{ name: "MicroLM", versions: [
      { id: "old", version: 1, state: "ready", training_contract: {} },
      { id: "new", version: 2, state: "ready", training_contract: { performance_probe: "forgelm.training-performance.v1", optimization_profiles: "forgelm.microlm-optimizations.v1" } },
    ] }]));
  }));
  const onOpen = vi.fn();
  const task: Task = { id: "source", project_id: "project", kind: "train", title: "source", observed_state: "paused", desired_state: "paused", progress: 0.5, created_at: "2026-09-16", payload: { stages: [{ micro_batch_size: 2 }] } };
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><TrainingOptimizationComparison task={task} onOpen={onOpen} /></QueryClientProvider>);
  expect(screen.getByRole("button", { name: "创建完整训练比较" })).toBeDisabled();
  await screen.findByRole("option", { name: "MicroLM · V2" });
  expect(screen.queryByRole("option", { name: "MicroLM · V1" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("比较引擎版本"), { target: { value: "new" } });
  fireEvent.change(screen.getByLabelText("交叉熵"), { target: { value: "torch_chunked" } });
  fireEvent.click(screen.getByRole("button", { name: "创建完整训练比较" }));
  await waitFor(() => expect(onOpen).toHaveBeenCalledWith("comparison"));
  expect(writes[0]).toMatchObject({ source_task_id: "source", engine_version_id: "new", candidates: [{ optimizations: { partial_rope: "flash_attn", linear_cross_entropy: "torch_chunked" } }] });
  expect(task.payload?.stages[0]).toEqual({ micro_batch_size: 2 });
});
