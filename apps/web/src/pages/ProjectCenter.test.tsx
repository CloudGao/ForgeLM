import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { ProjectCenter } from "./ProjectCenter";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it.each([true, false])("requires an unblocked preview and exact name before deletion (%s)", async (allowed) => {
  const sent: unknown[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    let body: unknown = [];
    if (url.endsWith("/projects")) body = [{ id: "p1", name: "实验", target_model_name: "test", workspace_uri: "/tmp/project", project_type: "pretrain", target_parameters: "50M", lifecycle_stage: "data_preparation", recent_training_status: "not_started", updated_at: "2026-09-22T00:00:00Z", data_progress: { dataset_count: 0, ready_versions: 0, percent: 0 } }];
    if (url.endsWith("/deletion-preview")) body = { fingerprint: "f".repeat(64), can_delete: allowed, record_counts: { projects: 1 }, paths: [], blockers: allowed ? [] : [{ kind: "active_task", id: "running" }] };
    if (url.endsWith("/delete")) { sent.push(JSON.parse(String(init?.body))); body = { deleted: true }; }
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter><ProjectCenter /></MemoryRouter></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "删除项目 实验" }));
  const submit = await screen.findByRole("button", { name: "永久删除项目及列出的目录" });
  expect(submit).toBeDisabled();
  fireEvent.change(screen.getByRole("textbox", { name: "确认删除的项目名称" }), { target: { value: "实验" } });
  if (allowed) {
    fireEvent.click(submit);
    await waitFor(() => expect(sent).toEqual([{ confirmation: "实验", fingerprint: "f".repeat(64) }]));
  } else {
    expect(submit).toBeDisabled();
    expect(sent).toEqual([]);
  }
});
