import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { InferenceOutput } from "./InferenceOutput";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows streamed text before completion, ignores stale attempts and closes the stream", () => {
  let listener: (event: any) => void = () => {};
  const close = vi.fn();
  vi.stubGlobal("EventSource", class { addEventListener(_name: string, fn: any) { listener = fn; } close = close; });
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const view = render(<QueryClientProvider client={client}><InferenceOutput projectId="p" record={{ output: "" }} task={{ id: "t", active_attempt_id: "a", observed_state: "running" }} /></QueryClientProvider>);
  act(() => listener({ data: JSON.stringify({ attempt_id: "a", event_type: "task.assigned", payload: {} }) }));
  expect(screen.getByText("正在派发到 Worker")).toBeInTheDocument();
  act(() => listener({ data: JSON.stringify({ attempt_id: "old", event_type: "inference.snapshot", payload: { output: "过期内容" } }) }));
  expect(screen.queryByText("过期内容")).toBeNull();
  act(() => listener({ data: JSON.stringify({ attempt_id: "a", event_type: "inference.snapshot", payload: { output: "即时正文", phase: "generating" } }) }));
  expect(screen.getByText("即时正文")).toBeInTheDocument();
  expect(screen.getByText("正在生成")).toBeInTheDocument();
  act(() => listener({ data: JSON.stringify({ attempt_id: "a", event_type: "task.succeeded", payload: {} }) }));
  expect(invalidate).toHaveBeenCalledTimes(2);
  expect(close).toHaveBeenCalled();
  view.unmount();
});

it("keeps partial text visible with an explicit failure without presenting success", () => {
  render(<QueryClientProvider client={new QueryClient()}><InferenceOutput projectId="p"
    record={{ output: "", performance: { stream: { attempt_id: "a", output: "未完成正文" } } }}
    task={{ id: "t", active_attempt_id: "a", observed_state: "failed", error: { message: "保存失败" } }} /></QueryClientProvider>);
  expect(screen.getByText("未完成正文")).toBeInTheDocument();
  expect(screen.getByRole("alert")).toHaveTextContent("保存失败");
});
