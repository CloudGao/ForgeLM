import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { DatasetTableViewer } from "./DataCenter";

afterEach(() => vi.unstubAllGlobals());

it("browses full evaluation results, searches, jumps and reuses the initial snapshot", async () => {
  const queries: any[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/viewer/query")) {
      queries.push(JSON.parse(String(init?.body)));
      return Response.json({ id: `query-${queries.length}`, observed_state: "queued" }, { status: 202 });
    }
    const index = Number(url.match(/query-(\d+)/)?.[1]) - 1;
    const query = queries[index];
    return Response.json({ id: `query-${index + 1}`, observed_state: "succeeded", result: {
      sql: query.sql, base_sql: query.sql, search: query.search, source_task_id: "query-1", configuration: "task-a", configurations: ["task-a", "task-b"], split: "test", splits: ["test"], format: "jsonl", elapsed_ms: 1,
      columns: [{ name: "question", type: "VARCHAR" }], rows: [[`真实题目-${query.offset}`]], offset: query.offset, limit: 100, total_rows: 250, has_more: true, labelable: false,
    } });
  }));
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><DatasetTableViewer version={{ id: "eval-1" }} queryEndpoint="/catalog/evaluation-datasets/eval-1/viewer/query" enableLabels={false} /></QueryClientProvider>);
  expect(await screen.findByText("真实题目-0")).toBeInTheDocument();
  expect(screen.queryByText("当前结果不可打标")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Dataset SQL 查询"), { target: { value: "SELECT question FROM dataset" } });
  fireEvent.click(screen.getByRole("button", { name: "下一页" }));
  expect(await screen.findByText("真实题目-1")).toBeInTheDocument();
  expect(queries[1]).toMatchObject({ offset: 1, sql: "SELECT * FROM dataset", source_task_id: "query-1" });
  fireEvent.change(screen.getByLabelText("搜索数据内容"), { target: { value: "任意题目" } });
  fireEvent.click(screen.getByRole("button", { name: "搜索" }));
  await waitFor(() => expect(queries[2]).toMatchObject({ search: "任意题目", offset: 0, sql: "SELECT question FROM dataset", source_task_id: "query-1" }));
  await screen.findByText("真实题目-0");
  fireEvent.change(screen.getByLabelText("跳转行号"), { target: { value: "201" } });
  fireEvent.click(screen.getByRole("button", { name: "跳转" }));
  expect(await screen.findByText("真实题目-200")).toBeInTheDocument();
  expect(queries[3]).toMatchObject({ offset: 200, search: "任意题目", source_task_id: "query-1" });
  fireEvent.change(screen.getByLabelText("数据子集"), { target: { value: "task-b" } });
  await waitFor(() => expect(queries[4]).toMatchObject({ configuration: "task-b", split: "", offset: 0, source_task_id: "query-1" }));
});
