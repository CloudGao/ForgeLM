import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { ProjectPipelineInstance } from "../types";
import { InterleavedGenerationDialog } from "./InterleavedGenerationDialog";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("previews frozen batches and creates through the native instance entry", async () => {
  const calls: { url: string; body: any }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(init.body) });
    return new Response(JSON.stringify(String(url).endsWith("-preview")
      ? { summary: { source_records: 6642359, batch_count: 133 } } : { id: "new-instance" }));
  }));
  const created = vi.fn();
  render(<QueryClientProvider client={new QueryClient()}><InterleavedGenerationDialog
    instance={{ id: "original", project_id: "project", name: "原任务" } as ProjectPipelineInstance}
    sourceTaskId="old-task" onClose={vi.fn()} onCreated={created} /></QueryClientProvider>);
  expect(screen.getByRole("button", { name: "创建交错实例" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "预览批次" }));
  await screen.findByRole("status");
  fireEvent.click(screen.getByRole("button", { name: "创建交错实例" }));
  await waitFor(() => expect(created).toHaveBeenCalledWith({ id: "new-instance" }));
  expect(calls).toHaveLength(2);
  expect(calls[1].url).toContain("/training-instances/original/interleaved");
  expect(calls[1].body).toMatchObject({ batch_records: 100000, checkpoint_source_task_id: "old-task" });
  expect(calls.some(call => call.url.includes("/attempts") || call.url.includes("/control"))).toBe(false);
});
