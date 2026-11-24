import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StorageCenter } from "./StorageCenter";

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Storage center", () => {
  it("explains an outdated backend and never submits selected items to it", async () => {
    const requests: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (init?.method === "POST") requests.push(url);
      if (url.endsWith("/storage/overview")) return response({
        latest_successful_scan: { id: "old", status: "succeeded", summary: {}, items: [
          { id: "cache", item_type: "cleanup_candidate", category: "scratch", path: "F:/cache", reclaimable: true, size_bytes: 1024, file_count: 1 },
        ] }, cleanup_plans: [],
      });
      return response([]);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><StorageCenter /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText(/后台尚未加载新版数据清理/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择当前可清理项" }));
    const preview = screen.getByRole("button", { name: "生成清理预览" });
    expect(preview).toBeDisabled();
    fireEvent.click(preview);
    expect(requests).toEqual([]);
  });

  it("filters selection, disables protected rows and displays per-item results", async () => {
    const items = [
      { id: "cache", project_id: "p1", item_type: "cleanup_candidate", category: "scratch", path: "F:/p1/cache", size_bytes: 1024, file_count: 1, reclaimable: true, reason: "任务已结束", metadata_json: { project_name: "项目一", impact: "保留评测结果" } },
      { id: "protected", project_id: "p1", item_type: "cleanup_candidate", category: "intermediate_checkpoint", path: "F:/p1/checkpoint-10", size_bytes: 4096, file_count: 1, reclaimable: false, reason: "最终恢复包校验失败", metadata_json: { impact: "不能恢复到中间步数" } },
      { id: "orphan", project_id: "p2", item_type: "cleanup_candidate", category: "orphan_artifact", path: "F:/p2/old-model", size_bytes: 2048, file_count: 1, reclaimable: true, reason: "模型已删除", metadata_json: { project_name: "项目二" } },
    ];
    const previewBodies: any[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/storage/overview")) return response({
        cleanup_selection_supported: true,
        current_scan: { id: "scan", status: "succeeded" },
        latest_successful_scan: { id: "scan", status: "succeeded", summary: { projects: [
          { project_id: "p1", project_name: "项目一", categories: {} }, { project_id: "p2", project_name: "项目二", categories: {} },
        ] }, items },
        cleanup_plans: [{ id: "previous", status: "partial", summary: { item_count: 1, actual_reclaimed_bytes: 0 } }],
      });
      if (url.endsWith("/cleanup-plans/previous")) return response({
        id: "previous", status: "partial", summary: { actual_reclaimed_bytes: 0 },
        items: [{ ...items[1], status: "skipped", actual_reclaimed_bytes: 0, error: { message: "文件已变化，跳过" } }],
      });
      if (url.endsWith("/cleanup-plans") && init?.method === "POST") {
        previewBodies.push(JSON.parse(String(init.body)));
        return response({ id: "plan", status: "planned", confirmation_phrase: "清理 1 项", summary: { item_count: 1, estimated_reclaimable_bytes: 1024 }, items: [items[0]] });
      }
      return response([], 200);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><StorageCenter /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText("最终恢复包校验失败")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "选择 F:/p1/checkpoint-10" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "选择当前可清理项" }));
    expect(screen.getByRole("status")).toHaveTextContent("已选 2 项");
    fireEvent.change(screen.getByLabelText("清理项目"), { target: { value: "p1" } });
    expect(screen.getByRole("status")).toHaveTextContent("已选 0 项");
    expect(screen.queryByRole("checkbox", { name: "选择 F:/p2/old-model" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("清理类型"), { target: { value: "scratch" } });
    fireEvent.click(screen.getByRole("checkbox", { name: "选择当前可清理项" }));
    fireEvent.click(screen.getByRole("button", { name: "生成清理预览" }));
    const preview = await screen.findByRole("dialog", { name: "确认执行存储清理" });
    expect(within(preview).getByText("保留评测结果")).toBeInTheDocument();
    expect(previewBodies).toEqual([{ scan_id: "scan", item_ids: ["cache"] }]);
    fireEvent.click(within(preview).getByRole("button", { name: "取消" }));
    fireEvent.click(screen.getByRole("button", { name: "查看结果" }));
    const result = await screen.findByRole("dialog", { name: "清理执行结果" });
    expect(within(result).getByText("文件已变化，跳过")).toBeInTheDocument();
  });

  it("shows physical accounting and requires the cleanup plan phrase", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const candidate = {
      id: "item-1", project_id: "project-1", item_type: "cleanup_candidate", category: "scratch",
      path: "F:\\workspace\\.llm-platform\\runs\\task-1\\attempt-1\\inputs", size_bytes: 1024,
      file_count: 1, reclaimable: true, reason: "终态 attempt 的可重建 scratch",
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/storage/cleanup-plans") && init?.method === "POST") return response({
        id: "plan-1", scan_id: "scan-1", status: "planned", confirmation_phrase: "清理 1 项",
        summary: { item_count: 1, estimated_reclaimable_bytes: 1024 }, error: {}, expires_at: "2026-08-09T12:10:00Z",
        items: [{ ...candidate, status: "planned", actual_reclaimed_bytes: 0, error: {} }],
      }, 201);
      if (url.endsWith("/api/storage/cleanup-plans/plan-1/execute") && init?.method === "POST") return response({
        id: "plan-1", scan_id: "scan-1", status: "executing", confirmation_phrase: "清理 1 项", summary: {}, error: {}, expires_at: "2026-08-09T12:10:00Z", items: [],
      }, 202);
      if (url.endsWith("/api/storage/overview")) return response({
        cleanup_selection_supported: true,
        current_scan: { id: "scan-1", status: "succeeded", summary: {}, error: {}, created_at: "2026-08-09T12:00:00Z" },
        latest_successful_scan: {
          id: "scan-1", status: "succeeded", error: {}, ended_at: "2026-08-09T12:00:00Z",
          summary: {
            managed_bytes: 4096, reclaimable_bytes: 1024, candidate_count: 1, project_count: 1, unscanned_project_count: 0,
            volumes: [{ id: "device:1", root: "F:\\workspace", total_bytes: 10000, used_bytes: 4000, free_bytes: 6000, reserved_bytes: 1000, available_for_tasks_bytes: 5000, managed_bytes: 4096, severity: "healthy" }],
            projects: [{ project_id: "project-1", project_name: "ExampleLM", workspace: "F:\\workspace", managed_bytes: 4096, reclaimable_bytes: 1024, scanned: true, categories: { durable_artifact: 3072, run_state: 1024 } }],
            workers: [],
          },
          items: [candidate],
        },
        cleanup_plans: [],
      });
      return response({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><StorageCenter /></MemoryRouter></QueryClientProvider>);

    expect((await screen.findAllByText("ExampleLM")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.0 KiB").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "生成清理预览" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("请先勾选下方可清理项，再生成预览。");
    fireEvent.click(screen.getByRole("checkbox", { name: "选择当前可清理项" }));
    expect(screen.getByRole("status")).not.toHaveTextContent("请先勾选");
    fireEvent.click(screen.getByRole("button", { name: "生成清理预览" }));
    const dialog = await screen.findByRole("dialog", { name: "确认执行存储清理" });
    const execute = within(dialog).getByRole("button", { name: "执行精确清理" });
    expect(execute).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText("清理确认文字"), { target: { value: "清理 1 项" } });
    expect(execute).toBeEnabled();
    fireEvent.click(execute);

    await waitFor(() => expect(requests.some((item) => item.url.endsWith("/plan-1/execute"))).toBe(true));
    const preview = requests.find((item) => item.url.endsWith("/api/storage/cleanup-plans"));
    expect(JSON.parse(String(preview?.init?.body))).toEqual({ scan_id: "scan-1", item_ids: ["item-1"] });
    const execution = requests.find((item) => item.url.endsWith("/plan-1/execute"));
    expect(JSON.parse(String(execution?.init?.body))).toEqual({ confirmation: "清理 1 项" });
  });
});
