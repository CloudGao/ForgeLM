import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkerStorage } from "./WorkerStorage";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it("selects a Worker scan candidate and never submits protected paths", async () => {
  const requests: any[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: any, init: any) => {
    const url = String(input);
    requests.push({ url, init });
    let body: any = [];
    if (url.endsWith("/projects")) body = [{ id: "p", name: "实验项目" }];
    if (url.endsWith("/workers")) body = [{ id: "w", name: "AutoDL", capabilities: ["builtin.storage-maintenance"] }];
    if (url.endsWith("/policy")) body = { cache_cleanup: "auto", intermediate_retention: "retain", grace_hours: 24 };
    if (url.endsWith("/storage/worker-jobs")) body = [{ id: "scan", project_id: "p", title: "盘点", observed_state: "succeeded",
      payload: { resource_request: { worker_selector: { worker_id: "w" } } }, result: { storage_action: "scan", items: [
        { id: "cache", path: "/cache", category: "scratch", size_bytes: 100, reclaimable: true },
        { id: "model", path: "/checkpoint", category: "checkpoint", size_bytes: 900, reclaimable: false },
      ] } }];
    return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  }));
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><WorkerStorage /></QueryClientProvider>);
  await screen.findByText("实验项目");
  fireEvent.change(screen.getByLabelText("存储维护项目"), { target: { value: "p" } });
  fireEvent.change(screen.getByLabelText("存储维护 Worker"), { target: { value: "w" } });
  expect(await screen.findByLabelText("回收 /checkpoint")).toBeDisabled();
  fireEvent.click(screen.getByLabelText("回收 /cache"));
  fireEvent.click(screen.getByRole("button", { name: "回收选中项" }));
  await waitFor(() => expect(requests.some((r) => r.init?.method === "POST")).toBe(true));
  const posted = JSON.parse(requests.find((r) => r.init?.method === "POST").init.body);
  expect(posted).toEqual({ worker_id: "w", action: "cleanup", scan_task_id: "scan", item_ids: ["cache"] });
});
