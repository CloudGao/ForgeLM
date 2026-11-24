import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Releases } from "./Releases";

function response(body: unknown) {
  return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("quantization plugins", () => {
  it("shows project-bound jobs and manually creates a run instance", async () => {
    const calls: Array<{ url: string; method: string }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      calls.push({ url, method });
      if (url.endsWith("/api/plugins")) return response([
        { id: "plugin-version-1", plugin_id: "plugin.llm-compressor.quantization", kind: "quantization", version: "0.1.0", support_level: "experimental", source: { type: "builtin-catalog" }, capabilities: { framework: "llm-compressor" } },
        { id: "plugin-version-2", plugin_id: "plugin.microlm.quantization", kind: "quantization", version: "0.1.0", support_level: "experimental", source: { type: "builtin-catalog" }, capabilities: { framework: "forgelm-microlm" } },
      ]);
      if (url.endsWith("/api/workers")) return response([{ id: "worker-1", name: "GPU Worker", status: "online", capabilities: ["builtin.plugin-manager"] }]);
      if (url.endsWith("/api/plugin-suites")) return response([{ id: "suite-1", suite_id: "suite.microlm", name: "MicroLM", version: "0.1.0", enabled: true, description: "原生套件", architectures: ["microlm"], components: [], manifest_hash: "hash" }]);
      if (url.endsWith("/api/plugin-suite-deployments")) return response([]);
      if (url.endsWith("/api/plugin-installations")) return response([{ id: "installation-1", plugin_id: "plugin.llm-compressor.quantization", state: "ready", worker: { name: "GPU Worker" }, current_plugin_version: { version: "0.1.0" } }]);
      if (url.endsWith("/api/projects/project-1/plugin-bindings")) return response([{ plugin_id: "plugin.llm-compressor.quantization", plugin_version_id: "plugin-version-1", enabled: true }]);
      if (url.endsWith("/api/projects/project-1/models")) return response({ models: [{ id: "model-1", name: "Base Model", version: 1, artifact_id: "artifact-1" }] });
      if (url.endsWith("/api/projects/project-1/quantization-jobs/job-1/runs") && method === "POST") return response({ id: "run-2", run_number: 2 });
      if (url.endsWith("/api/projects/project-1/quantization-jobs")) return response([{ id: "job-1", name: "Release FP8", model_version_id: "model-1", enabled: true, config: { algorithm: "RTN", scheme: "FP8_DYNAMIC" }, runs: [{ id: "run-1" }] }]);
      if (url.endsWith("/api/projects/project-1/quantizations")) return response([]);
      if (url.endsWith("/api/projects/project-1/release-jobs")) return response([]);
      if (url.endsWith("/api/projects/project-1/releases")) return response([]);
      return new Response(null, { status: 404 });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/projects/project-1/releases"]}><Routes><Route path="/projects/:projectId/releases" element={<Releases />} /></Routes></MemoryRouter></QueryClientProvider>);

    expect(await screen.findByText("Release FP8")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行" }));
    await vi.waitFor(() => expect(calls.some((call) => call.method === "POST" && call.url.endsWith("/quantization-jobs/job-1/runs"))).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: /插件管理/ }));
    expect(await screen.findByRole("button", { name: "MicroLM" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "MicroLM W8A16" })).not.toBeInTheDocument();
  });

  it("separates runtime build from target delivery and renders plugin schemas", async () => {
    const calls: Array<{ url: string; method: string }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      calls.push({ url, method });
      if (url.endsWith("/api/plugins")) return response([
        { id: "exporter-v1", plugin_id: "plugin.forgelm.native-exporter", kind: "release_exporter", version: "0.1.0", support_level: "experimental", config_schema: { type: "object", properties: { verify_smoke: { type: "boolean", default: true } } }, capabilities: { output_formats: ["forgelm-native-model-directory-v1"], input_formats: ["microlm-hybrid-safetensors"] } },
        { id: "publisher-v1", plugin_id: "plugin.local-directory.publisher", kind: "release_publisher", version: "0.1.0", support_level: "experimental", config_schema: { type: "object", properties: { target_subdirectory: { type: "string", default: "published-models" } }, required: ["target_subdirectory"] }, capabilities: { input_formats: ["forgelm-native-model-directory-v1"], target_kind: "local_directory" } },
      ]);
      if (url.endsWith("/api/workers")) return response([]);
      if (url.endsWith("/api/plugin-installations")) return response([]);
      if (url.endsWith("/api/projects/project-1/plugin-bindings")) return response([
        { id: "binding-exporter", plugin_id: "plugin.forgelm.native-exporter", plugin_version_id: "exporter-v1", enabled: true },
        { id: "binding-publisher", plugin_id: "plugin.local-directory.publisher", plugin_version_id: "publisher-v1", enabled: true },
      ]);
      if (url.endsWith("/api/projects/project-1/models")) return response({ models: [{ id: "model-1", name: "MicroLM", version: 1, artifact_id: "artifact-1", metadata_json: { inference: { format: "microlm-hybrid-safetensors" } } }] });
      if (url.endsWith("/api/projects/project-1/quantization-jobs")) return response([]);
      if (url.endsWith("/api/projects/project-1/quantizations")) return response([]);
      if (url.endsWith("/api/projects/project-1/release-jobs/job-1/runs") && method === "POST") return response({ id: "build-run-1", task_id: "task-build-1" });
      if (url.endsWith("/api/projects/project-1/release-jobs")) return response([{ id: "job-1", name: "Native runtime", source_kind: "model_version", source_id: "model-1", enabled: true, exporter_plugin_version: { plugin_id: "plugin.forgelm.native-exporter", version: "0.1.0", capabilities: { output_formats: ["forgelm-native-model-directory-v1"] } }, runs: [] }]);
      if (url.endsWith("/delivery-preflight") && method === "POST") return response({ status: "passed" });
      if (url.endsWith("/delivery-jobs") && method === "POST") return response({ id: "delivery-job-1" });
      if (url.endsWith("/delivery-jobs/delivery-job-1/runs") && method === "POST") return response({ id: "delivery-run-1" });
      if (url.endsWith("/api/projects/project-1/releases")) return response([{ id: "release-1", name: "MicroLM", version: "v1.0.0", release_kind: "runtime", created_at: "2026-08-23T00:00:00Z", gate_result: { status: "bypassed" }, runtime_snapshot: { format: "forgelm-native-model-directory-v1" }, delivery_jobs: [] }]);
      return new Response(null, { status: 404 });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/projects/project-1/releases"]}><Routes><Route path="/projects/:projectId/releases" element={<Releases />} /></Routes></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: /发布构建/ }));
    expect(await screen.findByText("Native runtime")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "运行构建" }));
    fireEvent.change(screen.getByLabelText("发布名称"), { target: { value: "MicroLM" } });
    fireEvent.change(screen.getByLabelText("新版本号"), { target: { value: "v1.1.0" } });
    fireEvent.click(screen.getByRole("button", { name: "创建构建运行" }));
    await vi.waitFor(() => expect(calls.some((call) => call.method === "POST" && call.url.endsWith("/release-jobs/job-1/runs"))).toBe(true));

    fireEvent.click(screen.getByRole("button", { name: /正式版本/ }));
    expect(await screen.findByText("MicroLM")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "添加交付目标" }));
    expect(screen.getByLabelText("target_subdirectory")).toHaveValue("published-models");
    fireEvent.change(screen.getByLabelText("目标名称"), { target: { value: "Local release" } });
    fireEvent.click(screen.getByRole("button", { name: "保存并交付" }));
    await vi.waitFor(() => expect(calls.some((call) => call.url.endsWith("/delivery-jobs/delivery-job-1/runs") && call.method === "POST")).toBe(true));
  });
});
