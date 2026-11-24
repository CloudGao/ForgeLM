import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { formatNumber } from "../api";
import type { ModelArchitecture } from "../types";
import { ModelSpecificationEditor } from "./ModelSpecificationEditor";

const architectures: ModelArchitecture[] = [{
  id: "llama_dense",
  version: "2",
  name: "Llama 风格 Dense Decoder",
  description: "Dense decoder",
  category: "Dense Decoder",
  source_type: "standard",
  implementation: { config_class: "LlamaConfig" },
  capabilities: ["causal_lm", "gqa"],
  fields: [
    { key: "vocab_size", label: "词表大小", group: "基础形状", type: "integer", default: 32000, minimum: 256, maximum: 1000000, step: 1 },
    { key: "num_hidden_layers", label: "层数", group: "基础形状", type: "integer", default: 24, minimum: 1, maximum: 256, step: 1 },
    { key: "tie_word_embeddings", label: "共享 Embedding 与 LM Head", group: "输出层", type: "boolean", default: false },
  ],
  presets: [],
}];

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("visual model specification editor", () => {
  it("uses typed controls and exposes generated JSON and Python before saving", async () => {
    const requests: Array<{ url: string; body: any }> = [];
    const resolved = {
      status: "ready",
      issues: [],
      parameter_summary: {
        total_parameters: 1_100_000_000,
        active_parameters_per_token: 1_100_000_000,
        trainable_parameters: 1_100_000_000,
        method: "architecture_formula_v2",
        verified: true,
      },
      generated_json: { schema_version: "forgelm.model-spec.v2", config: { num_hidden_layers: 32 } },
      generated_code: "from transformers import LlamaConfig\nconfig = LlamaConfig(num_hidden_layers=32)",
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = JSON.parse(String(init?.body || "{}"));
      requests.push({ url, body });
      if (url.endsWith("/api/projects/project-1/model-specifications/resolve")) return jsonResponse(resolved);
      if (url.endsWith("/api/projects/project-1/model-specifications")) return jsonResponse({ id: "spec-1", current_revision_id: "revision-1" }, 201);
      return jsonResponse({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    const onSaved = vi.fn();
    render(<QueryClientProvider client={queryClient}><ModelSpecificationEditor projectId="project-1" architectures={architectures} codeVersions={[]} onCancel={() => undefined} onSaved={onSaved} /></QueryClientProvider>);

    expect(screen.getByRole("combobox", { name: "架构实现族" })).toHaveValue("llama_dense");
    expect(screen.getByLabelText("层数")).toHaveValue(24);
    expect(screen.getByRole("button", { name: "保存模型规格" })).toBeDisabled();

    fireEvent.change(screen.getByLabelText("层数"), { target: { value: "32" } });
    fireEvent.change(screen.getByLabelText("共享 Embedding 与 LM Head"), { target: { value: "true" } });
    fireEvent.click(screen.getByRole("button", { name: "生成运行预览" }));

    expect((await screen.findAllByText(formatNumber(1_100_000_000))).length).toBeGreaterThan(0);
    expect(screen.getByText(/forgelm.model-spec.v2/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "实例化代码" }));
    expect(screen.getAllByText(/LlamaConfig/).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "保存模型规格" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "保存模型规格" }));

    await waitFor(() => expect(onSaved).toHaveBeenCalled());
    expect(requests[0].body).toMatchObject({
      source_type: "standard",
      architecture_family_id: "llama_dense",
      preset_id: null,
      config: { num_hidden_layers: 32, tie_word_embeddings: true },
      initialization: { type: "random" },
    });
    expect(requests[1].body).toMatchObject({ name: "我的模型规格", architecture_family_id: "llama_dense" });
  });
});
