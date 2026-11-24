import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Models } from "./Models";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const activeModel = {
  id: "model-1",
  project_id: "project-1",
  name: "ExampleLM",
  version: 3,
  kind: "sft",
  parent_model_version_id: null,
  source_checkpoint_id: null,
  artifact_id: null,
  tokenizer_version_id: null,
  parameter_count: 306_000_000,
  status: "candidate",
  is_best: false,
  is_candidate: true,
  is_official: false,
  metadata_json: { description: "Production candidate", registry_source: { type: "huggingface", repository_id: "example/model", revision: "1".repeat(40) }, inference: { pretrained: "example/model", revision: "1".repeat(40) } },
  inference_source: { type: "pretrained", value: "example/model", globally_accessible: true, worker_ids: [] },
  artifact: null,
  source_checkpoint: null,
  source_training_run: null,
  tokenizer: null,
  parent_model: null,
  child_model_count: 0,
  created_at: "2026-08-16T00:00:00Z",
  updated_at: "2026-08-16T00:00:00Z",
};

function renderModels(initialEntry = "/projects/project-1/models") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={[initialEntry]}><Routes><Route path="/projects/:projectId/models" element={<Models />} /></Routes></MemoryRouter></QueryClientProvider>);
}

function baseFetch(registry: any, onRequest?: (url: string, init?: RequestInit) => Response | undefined) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const custom = onRequest?.(url, init);
    if (custom) return custom;
    if (url.endsWith("/api/projects/project-1/models?include_archived=true")) return jsonResponse(registry);
    if (url.endsWith("/api/projects/project-1/model-specifications")) return jsonResponse([]);
    if (url.endsWith("/api/model-architectures")) return jsonResponse({ architectures: [] });
    if (url.endsWith("/api/projects/project-1/code-sources")) return jsonResponse([]);
    if (url.endsWith("/api/projects/project-1/tokenizers")) return jsonResponse([]);
    if (url.endsWith("/api/workers")) return jsonResponse([]);
    throw new Error(`Unexpected request: ${url}`);
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Model registry management", () => {
  it.each(["normal", "cascade"])("previews and confirms %s deletion from the more menu", async (mode) => {
    const actionName = mode === "cascade" ? "级联删除" : "删除";
    let previewBody: any;
    let executed: any;
    const plan: any = {
      id: "delete-plan", status: "planned", confirmation_phrase: `${actionName} V3`,
      summary: { blockers: [], estimated_reclaimable_bytes: 1024, actual_reclaimed_bytes: 0 },
      items: [
        { id: "i1", path: "/workspace/model.bin", category: "model", status: "planned", reason: "独占模型文件", size_bytes: 1024, metadata_json: { worker_name: "本机" } },
        { id: "i2", path: "/workspace/shared", category: "checkpoint", status: "protected", reason: "被 V9 共用", size_bytes: 2048, metadata_json: { worker_name: "远端" } },
      ],
    };
    vi.stubGlobal("fetch", baseFetch({ models: [activeModel], checkpoints: [], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (url.endsWith("/models/model-1/deletion-plans")) {
        previewBody = JSON.parse(String(init?.body));
        return jsonResponse(plan, 201);
      }
      if (url.endsWith("/deletion-plans/delete-plan/execute")) {
        executed = JSON.parse(String(init?.body));
        return jsonResponse({ ...plan, status: "succeeded", summary: { ...plan.summary, actual_reclaimed_bytes: 1024 } }, 202);
      }
      if (url.endsWith("/deletion-plans/delete-plan")) return jsonResponse(plan);
      return undefined;
    }));
    renderModels();
    fireEvent.click(await screen.findByText("更多"));
    fireEvent.click(screen.getByRole("button", { name: actionName }));
    const dialog = await screen.findByRole("dialog", { name: `${actionName} ExampleLM V3` });
    expect(await within(dialog).findByText("被 V9 共用")).toBeInTheDocument();
    expect(previewBody).toEqual({ mode });
    const confirm = within(dialog).getByRole("button", { name: `确认${actionName}` });
    expect(confirm).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText(`输入“${actionName} V3”确认`), { target: { value: `${actionName} V3` } });
    fireEvent.click(confirm);
    await waitFor(() => expect(executed).toEqual({ confirmation: `${actionName} V3` }));
    expect(await within(dialog).findByText(/已回收/)).toBeInTheDocument();
  });

  it("shows active-user blockers and offers deletion in model details", async () => {
    const plan = { id: "blocked-plan", status: "planned", confirmation_phrase: "级联删除 V3", items: [],
      summary: { blockers: ["未完成任务：正在训练"], estimated_reclaimable_bytes: 0, actual_reclaimed_bytes: 0 } };
    vi.stubGlobal("fetch", baseFetch({ models: [activeModel], checkpoints: [], lineage: { nodes: [], edges: [] } }, url => {
      if (url.includes("/deletion-plans")) return jsonResponse(plan);
      return undefined;
    }));
    renderModels("/projects/project-1/models?model=model-1");
    const detail = await screen.findByRole("dialog", { name: "ExampleLM V3" });
    fireEvent.click(within(detail).getByText("更多"));
    fireEvent.click(within(detail).getByRole("button", { name: "级联删除" }));
    const dialog = await screen.findByRole("dialog", { name: "级联删除 ExampleLM V3" });
    expect(await within(dialog).findByText("未完成任务：正在训练")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "确认级联删除" })).toBeDisabled();
  });

  it("opens on real model versions instead of the model specification editor", async () => {
    const archived = { ...activeModel, id: "model-archived", name: "OldLM", version: 1, status: "archived", is_candidate: false };
    vi.stubGlobal("fetch", baseFetch({ models: [activeModel, archived], checkpoints: [], lineage: { nodes: [], edges: [] } }));
    renderModels();

    expect(await screen.findByRole("tab", { name: /模型版本/ })).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText("ExampleLM")).toBeInTheDocument();
    expect(screen.queryByText("OldLM")).not.toBeInTheDocument();
    expect(screen.getByText("模型版本是下游稳定引用；模型规格和训练 Checkpoint 分别在独立页签管理。")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("模型状态"), { target: { value: "archived" } });
    expect(screen.getByText("OldLM")).toBeInTheDocument();
    expect(screen.queryByText("ExampleLM")).not.toBeInTheDocument();
  });

  it("opens a model detail from a stable training-output deep link", async () => {
    vi.stubGlobal("fetch", baseFetch({ models: [activeModel], checkpoints: [], lineage: { nodes: [], edges: [] } }));
    renderModels("/projects/project-1/models?model=model-1");

    expect(await screen.findByRole("dialog", { name: "ExampleLM V3" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /模型版本/ })).toHaveAttribute("aria-selected", "true");
  });

  it("opens a checkpoint detail from a stable training-output deep link", async () => {
    const checkpoint = {
      id: "checkpoint-1",
      project_id: "project-1",
      training_run_id: "run-1",
      step: 120,
      token_count: 24_000_000,
      artifact_id: "artifact-1",
      artifact: { id: "artifact-1", kind: "checkpoint", size_bytes: 1024, content_hash: "b".repeat(64), availability_state: "available" },
      metrics: { loss: 2.4 },
      training_position: {},
      training_run: { id: "run-1", current_stage: "Stage 1", status: "succeeded" },
      status: "available",
      created_at: "2026-08-16T00:00:00Z",
    };
    vi.stubGlobal("fetch", baseFetch({ models: [], checkpoints: [checkpoint], lineage: { nodes: [], edges: [] } }));
    renderModels("/projects/project-1/models?tab=checkpoints&checkpoint=checkpoint-1");

    expect(await screen.findByRole("dialog", { name: "Checkpoint checkpoi" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Checkpoint/ })).toHaveAttribute("aria-selected", "true");
  });

  it("registers an immutable Hugging Face model version with an exact commit", async () => {
    let submitted: any;
    vi.stubGlobal("fetch", baseFetch({ models: [], checkpoints: [], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (url.endsWith("/api/projects/project-1/models") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ ...activeModel, id: "model-created", name: submitted.name, version: 1, metadata_json: submitted.metadata }, 201);
      }
      return undefined;
    }));
    renderModels();

    fireEvent.click((await screen.findAllByRole("button", { name: "登记模型版本" }))[0]);
    const dialog = screen.getByRole("dialog", { name: "登记模型版本" });
    fireEvent.change(within(dialog).getByLabelText("模型名称"), { target: { value: "ImportedLM" } });
    fireEvent.change(within(dialog).getByLabelText("模型类型"), { target: { value: "sft" } });
    fireEvent.change(within(dialog).getByLabelText("仓库 ID"), { target: { value: "example/imported-lm" } });
    fireEvent.change(within(dialog).getByLabelText(/Commit SHA/), { target: { value: "a".repeat(40) } });
    fireEvent.change(within(dialog).getByLabelText("说明"), { target: { value: "Exact imported weights" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "创建不可变模型版本" }));

    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted).toMatchObject({
      name: "ImportedLM",
      kind: "sft",
      source_type: "huggingface",
      huggingface_id: "example/imported-lm",
      huggingface_revision: "a".repeat(40),
      is_candidate: true,
      metadata: { description: "Exact imported weights" },
    });
    expect(submitted).not.toHaveProperty("version");
  });

  it("starts model registration from an available Checkpoint", async () => {
    let submitted: any;
    const checkpoint = {
      id: "checkpoint-1",
      project_id: "project-1",
      training_run_id: "run-1",
      step: 120,
      token_count: 24_000_000,
      artifact_id: "artifact-1",
      artifact: { id: "artifact-1", kind: "checkpoint", size_bytes: 1024, content_hash: "b".repeat(64), availability_state: "available" },
      metrics: { loss: 2.4 },
      training_position: {},
      training_run: { id: "run-1", current_stage: "Stage 1", status: "succeeded" },
      status: "available",
      created_at: "2026-08-16T00:00:00Z",
    };
    vi.stubGlobal("fetch", baseFetch({ models: [], checkpoints: [checkpoint], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (url.endsWith("/api/projects/project-1/models") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ ...activeModel, id: "model-from-checkpoint", name: submitted.name, version: 1 }, 201);
      }
      return undefined;
    }));
    renderModels();

    fireEvent.click(await screen.findByRole("tab", { name: /Checkpoint/ }));
    fireEvent.click(await screen.findByRole("button", { name: "登记模型" }));
    const dialog = screen.getByRole("dialog", { name: "登记模型版本" });
    expect(within(dialog).getByText("项目 Checkpoint").closest("label")).toHaveClass("selected");
    fireEvent.change(within(dialog).getByLabelText("模型名称"), { target: { value: "CheckpointLM" } });
    expect(within(dialog).getByLabelText("可用 Checkpoint")).toHaveValue("checkpoint-1");
    fireEvent.click(within(dialog).getByRole("button", { name: "创建不可变模型版本" }));

    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted).toMatchObject({ name: "CheckpointLM", source_type: "checkpoint", source_checkpoint_id: "checkpoint-1" });
    expect(submitted).not.toHaveProperty("huggingface_id");
  });

  it("registers an OpenAI-compatible endpoint without sending a secret value", async () => {
    let submitted: any;
    vi.stubGlobal("fetch", baseFetch({ models: [], checkpoints: [], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (url.endsWith("/api/projects/project-1/models") && init?.method === "POST") {
        submitted = JSON.parse(String(init.body));
        return jsonResponse({ ...activeModel, id: "endpoint-model", name: submitted.name, version: 1, metadata_json: submitted.metadata }, 201);
      }
      return undefined;
    }));
    renderModels();

    fireEvent.click((await screen.findAllByRole("button", { name: "登记模型版本" }))[0]);
    const dialog = screen.getByRole("dialog", { name: "登记模型版本" });
    const endpointChoice = within(dialog).getByText("OpenAI 兼容端点").closest("label");
    fireEvent.click(endpointChoice!.querySelector("input")!);
    fireEvent.change(within(dialog).getByLabelText("模型名称"), { target: { value: "Ollama Qwen" } });
    fireEvent.change(dialog.querySelector('input[name="endpoint_url"]')!, { target: { value: "http://localhost:11434/v1" } });
    fireEvent.change(dialog.querySelector('input[name="endpoint_model"]')!, { target: { value: "qwen3.5:0.8b" } });
    fireEvent.change(dialog.querySelector('input[name="endpoint_tokenizer"]')!, { target: { value: "Qwen/Qwen3.5-0.8B" } });
    fireEvent.change(dialog.querySelector('input[name="endpoint_auth_env_var"]')!, { target: { value: "OLLAMA_EVAL_KEY" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "创建不可变模型版本" }));

    await waitFor(() => expect(submitted).toBeTruthy());
    expect(submitted).toMatchObject({
      name: "Ollama Qwen",
      source_type: "openai_compatible",
      endpoint_url: "http://localhost:11434/v1",
      endpoint_model: "qwen3.5:0.8b",
      endpoint_mode: "completions",
      endpoint_tokenizer: "Qwen/Qwen3.5-0.8B",
      endpoint_auth_env_var: "OLLAMA_EVAL_KEY",
      endpoint_verify_tls: true,
    });
    expect(JSON.stringify(submitted)).not.toContain("top-secret");
  });

  it("exposes candidate, best, official and archive actions for a model version", async () => {
    const requests: string[] = [];
    vi.stubGlobal("fetch", baseFetch({ models: [activeModel], checkpoints: [], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (init?.method === "POST" && url.includes("/api/models/model-1/")) {
        requests.push(url);
        return jsonResponse(activeModel);
      }
      return undefined;
    }));
    renderModels();

    await screen.findByText("ExampleLM");
    fireEvent.click(screen.getByRole("button", { name: "设为最佳" }));
    fireEvent.click(screen.getByRole("button", { name: "晋升" }));
    fireEvent.click(screen.getByRole("button", { name: "归档" }));
    await waitFor(() => {
      expect(requests.some((url) => url.endsWith("/api/models/model-1/mark-best"))).toBe(true);
      expect(requests.some((url) => url.endsWith("/api/models/model-1/promote"))).toBe(true);
      expect(requests.some((url) => url.endsWith("/api/models/model-1/archive"))).toBe(true);
    });
  });

  it("lets the user remove the best marker", async () => {
    const requests: string[] = [];
    const bestModel = { ...activeModel, is_best: true };
    vi.stubGlobal("fetch", baseFetch({ models: [bestModel], checkpoints: [], lineage: { nodes: [], edges: [] } }, (url, init) => {
      if (init?.method === "POST" && url.endsWith("/api/models/model-1/unmark-best")) {
        requests.push(url);
        return jsonResponse({ ...bestModel, is_best: false });
      }
      return undefined;
    }));
    renderModels();

    fireEvent.click(await screen.findByRole("button", { name: "取消最佳" }));
    await waitFor(() => expect(requests).toHaveLength(1));
  });
});
