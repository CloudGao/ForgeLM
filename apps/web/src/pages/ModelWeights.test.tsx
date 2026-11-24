import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ModelWeights } from "./ModelWeights";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it.each([
  ["downloading", "下载中", 0.37, "37"],
  ["verifying", "校验中", 0.82, "82"],
  ["downloading", "下载中", 0, "0"],
])("shows weight download phase %s and progress %s", async (phase, label, progress, percent) => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const body = url.endsWith("/model-weights")
      ? { items: [{ id: "weight", state: "running", task_id: "download", asset: { name: "Test" }, locations: [] }] }
      : url.includes("/tasks/download")
        ? { observed_state: "running", current_phase: phase, progress }
        : [];
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ModelWeights /></QueryClientProvider>);
  expect(await screen.findByText(`总进度 ${percent}%`)).toBeInTheDocument();
  expect(screen.getByText(label)).toBeInTheDocument();
  expect(screen.getByLabelText(`进度 ${percent}%`)).toBeInTheDocument();
  expect(screen.queryByText("运行中")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /^下载$/ })).toBeDisabled();
  client.clear();
});
