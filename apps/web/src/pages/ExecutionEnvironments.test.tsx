import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExecutionEnvironments } from "./ExecutionEnvironments";

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Execution environments", () => {
  it("discovers a local WSL environment and submits an injection", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/execution-environments/local/operations")) return response([]);
      if (url.endsWith("/api/execution-environments/local/inject")) return response({ id: "operation-1", status: "running" }, 202);
      if (url.endsWith("/api/execution-environments/local")) return response([
        { id: "host:native", kind: "windows", name: "Windows Native", state: "running", injection_supported: false, start_supported: true, description: "native", worker: { name: "local-host", status: "online" }, supervision: { managed: false, state: "stopped" } },
        { id: "wsl:MicroLM-Ubuntu", kind: "wsl", name: "MicroLM-Ubuntu", state: "stopped", injection_supported: true, start_supported: false, description: "wsl", worker: null, supervision: { managed: false, state: "stopped", restart_count: 0 } },
      ]);
      return response({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);

    expect(await screen.findByText("MicroLM-Ubuntu")).toBeInTheDocument();
    expect(screen.getByText("未托管")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "注入 Worker" }));
    const dialog = screen.getByRole("dialog", { name: "向 MicroLM-Ubuntu 注入 Worker" });
    fireEvent.change(within(dialog).getByRole("combobox", { name: "运行配置" }), { target: { value: "base" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "开始注入" }));

    await waitFor(() => expect(requests.some((item) => item.url.endsWith("/inject") && item.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/inject") && item.init?.method === "POST");
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({
      environment_id: "wsl:MicroLM-Ubuntu",
      profile: "base",
    });
  });

  it("starts an offline Windows Native Worker", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/execution-environments/local/operations")) return response([]);
      if (url.endsWith("/api/execution-environments/local/start")) return response({ managed: true, state: "running", pid: 123 }, 202);
      if (url.endsWith("/api/execution-environments/local")) return response([
        { id: "host:native", kind: "windows", name: "Windows Native", state: "running", injection_supported: false, start_supported: true, description: "native", worker: { name: "local-host", status: "offline" }, supervision: { managed: false, state: "stopped" } },
      ]);
      return response({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "启动 Worker" }));

    await waitFor(() => expect(requests.some((item) => item.url.endsWith("/start") && item.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/start") && item.init?.method === "POST");
    expect(JSON.parse(String(request?.init?.body))).toEqual({ environment_id: "host:native" });
  });

  it("counts and lists multiple Workers in one WSL environment", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/execution-environments/local/operations")) return response([]);
      if (url.endsWith("/api/execution-environments/local")) return response([
        {
          id: "wsl:MicroLM-Ubuntu",
          kind: "wsl",
          name: "MicroLM-Ubuntu",
          state: "running",
          injection_supported: true,
          start_supported: false,
          description: "wsl",
          worker: { name: "data-wsl-host", status: "online" },
          workers: [
            { name: "data-wsl-host", status: "online" },
            { name: "microlm-wsl-host", status: "online" },
          ],
          supervision: { managed: true, state: "running" },
        },
      ]);
      return response({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);

    expect(await screen.findByText("data-wsl-host, microlm-wsl-host")).toBeInTheDocument();
    expect(screen.getByText("在线 Worker").parentElement).toHaveTextContent("2");
  });

  it("adds an SSH Worker without sending a private key or Agent token", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/execution-environments/local/operations")) return response([]);
      if (url.endsWith("/api/execution-environments/local")) return response([]);
      if (url.endsWith("/api/execution-environments/remote") && init?.method === "POST") return response({ id: "remote-1", name: "gpu-lab-01", status: "registered" }, 201);
      if (url.endsWith("/api/execution-environments/remote")) return response([]);
      return response({ message: "not found" }, 404);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "添加 SSH Worker" }));
    const dialog = screen.getByRole("dialog", { name: "添加 SSH 远端 Worker" });
    fireEvent.change(within(dialog).getByRole("textbox", { name: "环境名称" }), { target: { value: "gpu-lab-01" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: "SSH 主机" }), { target: { value: "10.0.0.21" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /^SSH 用户/ }), { target: { value: "forgelm" } });
    fireEvent.change(within(dialog).getByRole("combobox", { name: "登录方式" }), { target: { value: "reference" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /^私钥环境变量/ }), { target: { value: "FORGELM_SSH_KEY_GPU_LAB_01" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "固定 Host Key 并部署" }));

    await waitFor(() => expect(requests.some((item) => item.url.endsWith("/execution-environments/remote") && item.init?.method === "POST")).toBe(true));
    const request = requests.find((item) => item.url.endsWith("/execution-environments/remote") && item.init?.method === "POST");
    const body = JSON.parse(String(request?.init?.body));
    expect(body).toMatchObject({
      name: "gpu-lab-01",
      ssh_host: "10.0.0.21",
      ssh_user: "forgelm",
      credential_ref: "FORGELM_SSH_KEY_GPU_LAB_01",
      profile: "general",
      transport: "ssh_tunnel",
    });
    expect(body).not.toHaveProperty("private_key");
    expect(body).not.toHaveProperty("agent_token");
  });
  it("deploys an AutoDL command with a one-time password and Hugging Face environment", async () => {
    let submitted: any;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "POST") { submitted = JSON.parse(String(init.body)); return response({ id: "remote" }, 201); }
      return response([]);
    }));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);
    fireEvent.click(screen.getByRole("button", { name: "添加 SSH Worker" }));
    const dialog = screen.getByRole("dialog", { name: "添加 SSH 远端 Worker" });
    expect(within(dialog).queryByRole("textbox", { name: /^控制面公开地址/ })).not.toBeInTheDocument();
    fireEvent.change(within(dialog).getByRole("textbox", { name: "环境名称" }), { target: { value: "autodl-gpu" } });
    fireEvent.change(within(dialog).getByRole("textbox", { name: /^SSH 登录指令/ }), { target: { value: "ssh -p 31234 root@autodl.test" } });
    fireEvent.change(within(dialog).getByLabelText(/^SSH 密码/), { target: { value: "one-time-password" } });
    fireEvent.change(within(dialog).getByRole("combobox", { name: "运行环境" }), { target: { value: "inference" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "固定 Host Key 并部署" }));
    await waitFor(() => expect(submitted).toMatchObject({ ssh_command: "ssh -p 31234 root@autodl.test", ssh_password: "one-time-password", transport: "ssh_tunnel", runtime: "inference", service_mode: "auto" }));
    expect(submitted).not.toHaveProperty("credential_ref");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("shows the address only when direct HTTP is selected", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response([])));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><MemoryRouter><ExecutionEnvironments /></MemoryRouter></QueryClientProvider>);
    fireEvent.click(screen.getByRole("button", { name: "添加 SSH Worker" }));
    fireEvent.change(screen.getByRole("combobox", { name: "连接方式" }), { target: { value: "direct" } });
    expect(screen.getByRole("textbox", { name: /^控制面公开地址/ })).toBeInTheDocument();
  });

});
