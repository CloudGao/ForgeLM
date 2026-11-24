import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Playground } from "./Playground";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function renderPlayground() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/projects/project-1/playground"]}><Routes><Route path="/projects/:projectId/playground" element={<Playground />} /></Routes></MemoryRouter></QueryClientProvider>);
}

const model = {
  id: "model-1",
  name: "Tiny Chat",
  version: 1,
  is_official: true,
  artifact_id: null,
  metadata_json: { inference: { pretrained: "org/tiny-chat" } },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Playground real inference", () => {
  it.each(["new-session", "select-model"])("recovers from a deleted historical model via %s", async (action) => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const nativeModel = {
      ...model,
      artifact_id: "native-artifact",
      inference_source: {
        type: "native", adapter_id: "builtin.microlm-inference",
        display_name: "MicroLM 原生推理", globally_accessible: true,
        resource_request: { gpus: 1 },
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/models")) return jsonResponse({ models: [nativeModel] });
      if (url.endsWith("/workers")) return jsonResponse([{
        id: "worker-1", status: "online", resources: { gpus: [{}] },
        capabilities: ["builtin.microlm-inference", "builtin.local-process"],
      }]);
      if (url.endsWith("/playground/sessions")) return jsonResponse([{
        id: "session-deleted-model", mode: "single", model_version_ids: ["deleted-model"],
        records: [{ id: "record-old", prompt: "历史问题", output: "历史回答", performance: {} }],
      }]);
      if (url.includes("/tasks?") || url.endsWith("/feedback")) return jsonResponse([]);
      if (url.endsWith("/playground/generate") && init?.method === "POST") return jsonResponse({ session: { id: "session-new" } }, 202);
      return jsonResponse({}, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("历史回答")).toBeInTheDocument();
    const select = screen.getByRole("combobox", { name: "模型版本" });
    await waitFor(() => expect(select).toHaveValue("deleted-model"));
    expect(screen.getByRole("option", { name: "模型 deleted-（已不可用）" })).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "继续提问" } });
    const run = screen.getByRole("button", { name: "运行" });
    expect(run).toBeDisabled();
    expect(run).toHaveAttribute("title", "所选模型已不可用，请选择模型或开始新会话");

    if (action === "new-session") fireEvent.click(screen.getByRole("button", { name: "新会话" }));
    else fireEvent.change(select, { target: { value: nativeModel.id } });
    await waitFor(() => expect(select).toHaveValue(nativeModel.id));
    expect(screen.queryByText("历史回答")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("combobox", { name: "推理适配器" })).toHaveValue("builtin.microlm-inference"));
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "什么是光合作用" } });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);
    await waitFor(() => expect(requests.some((r) => r.url.endsWith("/playground/generate"))).toBe(true));
    const submitted = requests.find((r) => r.url.endsWith("/playground/generate"));
    const body = JSON.parse(String(submitted?.init?.body));
    expect(body).toMatchObject({ model_version_ids: [nativeModel.id], adapter_id: "builtin.microlm-inference", prompt: "什么是光合作用" });
    expect(body.session_id).toBeUndefined();
  });

  it("restores the historical session model instead of the latest default", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const latestModel = { ...model, id: "model-latest", name: "Latest Model", is_official: true };
    const sessionModel = {
      ...model,
      id: "model-session",
      name: "Session Model",
      is_official: false,
      metadata_json: { inference: { pretrained: "org/session-chat" } },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [latestModel, sessionModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "GPU Worker", status: "online", capabilities: ["builtin.transformers-inference", "builtin.local-process"] }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) {
        return jsonResponse([{
          id: "session-old",
          mode: "single",
          model_version_ids: ["model-session"],
          records: [{ id: "record-1", prompt: "旧问题", output: "旧回答", task_id: "task-old", performance: {}, input_tokens: 1, output_tokens: 1 }],
        }]);
      }
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") {
        return jsonResponse({ session: { id: "session-old" }, record: { id: "record-2" }, task: { id: "task-2" } }, 202);
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("旧问题")).toBeInTheDocument();
    const modelSelect = await screen.findByRole("combobox", { name: "模型版本" });
    await waitFor(() => expect(modelSelect).toHaveValue("model-session"));
    expect(screen.getByText(/切换模型会开始新会话/)).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "继续问" } });
    const run = screen.getByRole("button", { name: "运行" });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      session_id: "session-old",
      model_version_ids: ["model-session"],
      prompt: "继续问",
    });
  });

  it("expands historical sessions and switches the active conversation", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) {
        return jsonResponse({
          models: [
            { ...model, id: "model-a", name: "Model A" },
            { ...model, id: "model-b", name: "Model B", is_official: false },
          ],
        });
      }
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "GPU Worker", status: "online", capabilities: ["builtin.transformers-inference", "builtin.local-process"] }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) {
        return jsonResponse([
          {
            id: "session-new",
            mode: "single",
            updated_at: "2026-09-06T10:00:00Z",
            model_version_ids: ["model-a"],
            records: [{ id: "record-new", prompt: "最新问题", output: "最新回答", task_id: "task-new", performance: {}, input_tokens: 2, output_tokens: 2 }],
          },
          {
            id: "session-old",
            mode: "single",
            updated_at: "2026-09-05T08:00:00Z",
            model_version_ids: ["model-b"],
            records: [{ id: "record-old", prompt: "历史问题", output: "历史回答", task_id: "task-old", performance: {}, input_tokens: 1, output_tokens: 1 }],
          },
        ]);
      }
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("最新问题")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "历史会话" }));
    expect(await screen.findByLabelText("历史会话列表")).toBeInTheDocument();
    expect(screen.getByText("2 条")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /历史问题/ }));
    expect(await screen.findByText("历史回答")).toBeInTheDocument();
    expect(screen.queryByText("最新回答")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("combobox", { name: "模型版本" })).toHaveValue("model-b"));
  });

  it("starts a new session when the model selector changes", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const latestModel = { ...model, id: "model-latest", name: "Latest Model", is_official: true };
    const sessionModel = {
      ...model,
      id: "model-session",
      name: "Session Model",
      is_official: false,
      metadata_json: { inference: { pretrained: "org/session-chat" } },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [latestModel, sessionModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "GPU Worker", status: "online", capabilities: ["builtin.transformers-inference", "builtin.local-process"] }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) {
        return jsonResponse([{
          id: "session-old",
          mode: "single",
          model_version_ids: ["model-session"],
          records: [{ id: "record-1", prompt: "旧问题", output: "旧回答", task_id: "task-old", performance: {}, input_tokens: 1, output_tokens: 1 }],
        }]);
      }
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") {
        return jsonResponse({ session: { id: "session-new" }, record: { id: "record-2" }, task: { id: "task-2" } }, 202);
      }
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    const modelSelect = await screen.findByRole("combobox", { name: "模型版本" });
    await waitFor(() => expect(modelSelect).toHaveValue("model-session"));
    fireEvent.change(modelSelect, { target: { value: "model-latest" } });
    await waitFor(() => expect(screen.queryByText("旧问题")).not.toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "新会话提问" } });
    const run = screen.getByRole("button", { name: "运行" });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      model_version_ids: ["model-latest"],
      prompt: "新会话提问",
    });
    expect(JSON.parse(String(submitted?.init?.body)).session_id).toBeUndefined();
  });

  it("submits the built-in adapter only when a capable Worker and model source exist", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [model] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "GPU Worker", status: "online", capabilities: ["builtin.transformers-inference", "builtin.local-process"] }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") return jsonResponse({ session: { id: "session-1" }, record: { id: "record-1" }, task: { id: "task-1" } }, 202);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("来源：org/tiny-chat")).toBeInTheDocument();
    const prompt = screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送");
    fireEvent.change(prompt, { target: { value: "你好" } });
    const run = screen.getByRole("button", { name: "运行" });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      adapter_id: "builtin.transformers-inference",
      model_version_ids: ["model-1"],
      prompt: "你好",
      config: { prompt_mode: "auto", device: "auto", dtype: "auto", max_tokens: 512 },
      resource_request: { gpus: 0 },
      command: {},
    });
  });

  it("freezes an explicitly selected compatible Worker", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [model] });
      if (url.endsWith("/api/workers")) return jsonResponse([
        { id: "worker-1", name: "Windows Worker", status: "online", labels: { environment: "native" }, capabilities: ["builtin.transformers-inference", "builtin.local-process"], resources: { cpu_count: 8, gpus: [] } },
        { id: "worker-2", name: "WSL Worker", status: "online", labels: { environment: "wsl" }, capabilities: ["builtin.transformers-inference", "builtin.local-process"], resources: { cpu_count: 8, gpus: [] } },
      ]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") return jsonResponse({ session: { id: "session-1" }, record: { id: "record-1" }, task: { id: "task-1" } }, 202);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    const worker = await screen.findByRole("combobox", { name: "执行 Worker" });
    await screen.findByRole("option", { name: "WSL Worker · wsl" });
    expect(worker).toHaveValue("");
    fireEvent.change(worker, { target: { value: "worker-2" } });
    await waitFor(() => expect(worker).toHaveValue("worker-2"));
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "你好" } });
    fireEvent.click(screen.getByRole("button", { name: "运行" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      resource_request: { gpus: 0, worker_selector: { worker_id: "worker-2" } },
    });
  });

  it("shows the missing runtime honestly and blocks submission", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [model] });
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("没有能访问该模型的 Transformers Worker")).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "你好" } });
    expect(screen.getByRole("button", { name: "运行" })).toBeDisabled();
  });

  it("requires a Transformers Worker that can access an external model", async () => {
    const externalModel = {
      ...model,
      artifact_id: "artifact-1",
      metadata_json: {},
      inference_source: {
        type: "external",
        artifact_id: "artifact-1",
        worker_ids: ["wsl-worker"],
        globally_accessible: false,
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [externalModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([
        { id: "windows-worker", status: "online", capabilities: ["builtin.transformers-inference"] },
        { id: "wsl-worker", status: "online", capabilities: ["builtin.local-process"] },
      ]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("没有能访问该模型的 Transformers Worker")).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "你好" } });
    expect(screen.getByRole("button", { name: "运行" })).toBeDisabled();
  });

  it("blocks a training checkpoint that has not been packaged for Transformers", async () => {
    const checkpointModel = {
      ...model,
      artifact_id: "checkpoint-1",
      metadata_json: {},
      inference_source: {
        type: "unsupported",
        artifact_id: "checkpoint-1",
        artifact_kind: "checkpoint",
        reason: "训练 Checkpoint 尚未封装为 Transformers 模型制品",
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [checkpointModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", status: "online", capabilities: ["builtin.transformers-inference", "builtin.local-process"] }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("模型制品不能直接推理")).toBeInTheDocument();
    expect(screen.getByText("训练 Checkpoint 尚未封装为 Transformers 模型制品")).toBeInTheDocument();
  });

  it("automatically uses a registered project inference profile", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const projectModel = {
      ...model,
      artifact_id: "checkpoint-1",
      metadata_json: {},
      inference_source: {
        type: "command",
        artifact_id: "checkpoint-1",
        adapter_id: "test.project-inference.v1",
        display_name: "示例项目原生推理",
        resource_request: { gpus: 1 },
        worker_ids: ["wsl-worker"],
        globally_accessible: false,
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [projectModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "wsl-worker", status: "online", capabilities: ["test.project-inference.v1", "builtin.local-process"], resources: { gpus: [{}] } }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") return jsonResponse({ session: { id: "session-1" }, record: { id: "record-1" }, task: { id: "task-1" } }, 202);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("来源：示例项目原生推理 · 制品 checkpoi")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("combobox", { name: "推理适配器" })).toHaveValue("test.project-inference.v1"));
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "你好" } });
    const run = screen.getByRole("button", { name: "运行" });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      adapter_id: "test.project-inference.v1",
      resource_request: { gpus: 1 },
      command: {},
    });
  });

  it("automatically uses the native MicroLM adapter for a packaged checkpoint", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const nativeModel = {
      ...model,
      artifact_id: "microlm-artifact-1",
      metadata_json: { inference: { format: "microlm-hybrid-safetensors" } },
      inference_source: {
        type: "native",
        artifact_id: "microlm-artifact-1",
        adapter_id: "builtin.microlm-inference",
        display_name: "MicroLM 原生推理",
        format: "microlm-hybrid-safetensors",
        resource_request: { gpus: 1 },
        worker_ids: ["wsl-worker"],
        globally_accessible: false,
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/models")) return jsonResponse({ models: [nativeModel] });
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "wsl-worker", status: "online", capabilities: ["builtin.microlm-inference", "builtin.local-process"], resources: { gpus: [{}] } }]);
      if (url.endsWith("/api/projects/project-1/playground/sessions")) return jsonResponse([]);
      if (url.includes("/api/tasks?")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/feedback")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/playground/generate") && init?.method === "POST") return jsonResponse({ session: { id: "session-1" }, record: { id: "record-1" }, task: { id: "task-1" } }, 202);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderPlayground();

    expect(await screen.findByText("来源：MicroLM 原生推理 · 制品 microlm-")).toBeInTheDocument();
    const adapter = screen.getByRole("combobox", { name: "推理适配器" });
    await waitFor(() => expect(adapter).toHaveValue("builtin.microlm-inference"));
    expect(screen.queryByRole("option", { name: "内置 Transformers 推理" })).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /^设备/ })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: /^设备/ })).toHaveValue("cuda");
    fireEvent.change(screen.getByPlaceholderText("输入 Prompt，Ctrl + Enter 发送"), { target: { value: "你好" } });
    const run = screen.getByRole("button", { name: "运行" });
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST")).toBe(true));
    const submitted = requests.find((request) => request.url.endsWith("/playground/generate") && request.init?.method === "POST");
    expect(JSON.parse(String(submitted?.init?.body))).toMatchObject({
      adapter_id: "builtin.microlm-inference",
      resource_request: { gpus: 1 },
      command: {},
    });
  });
});
