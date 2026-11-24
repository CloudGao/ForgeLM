import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { StorageLayout } from "./StorageLayout";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it("shows protected paths and uses the relocation action with exact confirmation", async () => {
  const requests: string[] = [];
  const plan = { id: "plan", status: "interrupted", confirmation_phrase: "迁移已核验的存储位置", summary: { item_count: 1, bytes: 4, blocked: [{ source: "/old/paused", reason: "暂停任务恢复目录保留" }] }, items: [{ id: "item", path: "/old/data", size_bytes: 4, status: "moved", metadata_json: { destination: "/new/dataset/data" } }] };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (init?.method === "POST") requests.push(url);
    const body = url.endsWith("/storage/layout") ? { root: "/new", directories: { dataset: "/new/dataset" }, projects: [], plans: [] } : plan;
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><StorageLayout /></QueryClientProvider>);
  fireEvent.click(await screen.findByRole("button", { name: "预览资产迁移" }));
  expect(await screen.findByText("暂停任务恢复目录保留")).toBeInTheDocument();
  const execute = screen.getByRole("button", { name: "校验并继续迁移" });
  expect(execute).toBeDisabled();
  fireEvent.change(screen.getByLabelText("迁移确认"), { target: { value: plan.confirmation_phrase } });
  fireEvent.click(execute);
  await waitFor(() => expect(requests).toEqual(["/api/storage/relocations/preview", "/api/storage/relocations/plan/execute"]));
});
