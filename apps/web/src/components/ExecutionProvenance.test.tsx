import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { configurationDiff, ExecutionProvenance } from "./ExecutionProvenance";
import { SuiteTrainingContinuation } from "./SuiteTrainingContinuation";
import { useState } from "react";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const provider = (child: React.ReactNode) => <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{child}</QueryClientProvider>;

it("compares nested config additions, removals and array changes", () => {
  expect(configurationDiff({ lr: 1, old: true, backends: ["a"] }, { lr: 2, backends: ["b"] })).toEqual([
    { path: "backends", before: ["a"], after: ["b"] }, { path: "lr", before: 1, after: 2 }, { path: "old", before: true, after: undefined },
  ]);
  expect(configurationDiff({ lr: 1, seed: 42 }, { seed: 42, lr: 1 })).toEqual([]);
});

it("shows the selected attempt receipt and compares its frozen config", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify([
    { id: "old", attempt_id: "a1", phase: "intent", digest: "a".repeat(64), snapshot: { adapter_id: "native", payload: { config: { lr: 1 } } } },
    { id: "new", attempt_id: "a2", phase: "intent", digest: "b".repeat(64), snapshot: { adapter_id: "native", payload: { config: { lr: 2 } } } },
  ]))));
  render(provider(<ExecutionProvenance taskId="t1" attemptId="a2" />));
  fireEvent.change(await screen.findByLabelText("与此前尝试比较配置"), { target: { value: "old" } });
  expect(screen.getByRole("cell", { name: "config.lr" })).toBeInTheDocument();
});

it("resets optimizer inheritance when switching continuation mode", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify([{ id: "cp", task_id: "task1", attempt_id: "attempt1", step: 12, manifest: {} }]))));
  function Form() {
    const [value, setValue] = useState({ checkpoint_id: "cp", mode: "configured", keep_optimizer: true });
    return <SuiteTrainingContinuation projectId="p" value={value} onChange={setValue} />;
  }
  render(provider(<Form />));
  const mode = await screen.findByLabelText("状态继承方式");
  expect(screen.getByRole("checkbox")).toBeChecked();
  fireEvent.change(mode, { target: { value: "weights" } });
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  fireEvent.change(mode, { target: { value: "configured" } });
  expect(screen.getByRole("checkbox")).not.toBeChecked();
});

it("reuses source configuration and exact artifact inputs for a continuation", async () => {
  const useSource = vi.fn();
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(String(input).includes("training-state-checkpoints")
    ? [{ id: "cp", task_id: "source", attempt_id: "a1", step: 4, manifest: {} }]
    : { payload: { suite_execution: { config: { optimizer_backend: "auto", seed: 42 }, inputs: { model: { artifact_id: "model1" }, data: { artifact_id: "data1" } } } } }))));
  render(provider(<SuiteTrainingContinuation projectId="p" value={{ checkpoint_id: "cp", mode: "full", keep_optimizer: false }} onChange={() => {}} onUseSource={useSource} />));
  const button = await screen.findByRole("button", { name: "沿用源训练配置与输入" });
  await vi.waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(useSource).toHaveBeenCalledWith({ optimizer_backend: "auto", seed: 42 }, { model: "model1", data: "data1" });
});
