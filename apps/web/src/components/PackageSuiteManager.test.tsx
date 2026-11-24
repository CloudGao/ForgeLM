import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import { PackageSuiteManager } from "./PackageSuiteManager";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const version = { id: "v1", suite_id: "s1", version: "1.0.0", digest: "a".repeat(64), trusted: false, enabled: true, suite: { id: "s1", suite_id: "acme.demo", name: "Demo", archived: false }, manifest: { name: "Demo", publisher: "acme", description: "真实发布包", license: "MIT", components: [{ id: "normalize", distribution: "acme-demo" }], targets: [{ id: "windows-x86_64-cp312-cpu" }], executionRequirements: { network: false } }, nodes: [{ id: "acme.demo.normalize", displayName: "转换文本", component: "normalize", capability: "transform", execution: { kind: "task" }, inputs: [], outputs: [], config_schema: { type: "object", properties: {} } }], package: { size_bytes: 100, expanded_bytes: 200 } };
function setup(route = "/execution/plugins?package=v1") {
  const writes: { url: string; body: any }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    let value: any = [];
    if (init?.method === "POST") {
      const body = init.body instanceof FormData ? init.body : JSON.parse(String(init.body || "{}"));
      writes.push({ url, body });
      if (url === "/api/suite-package-imports") value = { id: "import", package: { ...version.package, digest: version.digest, manifest: version.manifest } };
      if (url === "/api/suite-operation-plans") value = { id: "plan", spec: { trusted: false, items: [{ worker_id: "w1", worker_name: "CPU Worker", in_use: [], config_references: [], download_bytes: 100, disk_bytes: 400, blocked: false, diff: { added_nodes: [], removed_nodes: [], changed_nodes: [] } }] } };
    } else if (url === "/api/suite-packages") value = [version];
    else if (url === "/api/workers") value = [{ id: "w1", name: "CPU Worker", status: "online", capabilities: ["builtin.suite-manager"] }];
    return new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
  }));
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[route]}><PackageSuiteManager /></MemoryRouter></QueryClientProvider>);
  return writes;
}

it("installs the entire package without a project and requires digest trust", async () => {
  const writes = setup();
  fireEvent.click(await screen.findByRole("button", { name: "安装套件" }));
  const dialog = screen.getByRole("dialog");
  expect(within(dialog).queryByText("选择项目")).not.toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /CPU Worker/ }));
  fireEvent.click(within(dialog).getByRole("button", { name: "检查操作计划" }));
  const execute = await within(dialog).findByRole("button", { name: "执行安装" });
  expect(execute).toBeDisabled();
  fireEvent.click(within(dialog).getByRole("checkbox", { name: /我信任此包/ }));
  fireEvent.click(execute);
  await screen.findByText("尚无套件包操作。");
  expect(writes[0]).toEqual({ url: "/api/suite-operation-plans", body: { action: "install", version_id: "v1", worker_ids: ["w1"], drain: false } });
  expect(writes[1].body.trust_digest).toBe(version.digest);
  expect(writes[1].body.idempotency_key).toBeTruthy();
  expect(writes.some(w => "component_keys" in w.body)).toBe(false);
});

it("previews a file before registration and does not install on import", async () => {
  const writes = setup("/execution/plugins");
  fireEvent.click(screen.getByRole("button", { name: "新增套件" }));
  const dialog = screen.getByRole("dialog");
  fireEvent.change(within(dialog).getByLabelText("套件发布包"), { target: { files: [new File(["archive"], "demo.flsuite")] } });
  fireEvent.click(within(dialog).getByRole("button", { name: "检查发布包" }));
  fireEvent.click(await within(dialog).findByRole("button", { name: "登记套件" }));
  await screen.findByRole("button", { name: "管理套件" });
  expect(writes.map(w => w.url)).toEqual(["/api/suite-package-imports", "/api/suite-package-imports/import/register"]);
});
