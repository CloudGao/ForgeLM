import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ControlPlaneControl } from "./ControlPlaneControl";

const managedStatus = {
  managed: true,
  restart_supported: true,
  state: "running",
  instance_id: "instance-a",
  pid: 101,
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function renderControl(fetchImplementation: typeof fetch, onRecovered = vi.fn()) {
  vi.stubGlobal("fetch", vi.fn(fetchImplementation));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <ControlPlaneControl onRecovered={onRecovered} recoveryDelay={1} recoveryAttempts={5} />
    </QueryClientProvider>,
  );
  return onRecovered;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ControlPlaneControl", () => {
  it("disables restart when the API is not supervised", async () => {
    renderControl(async () => jsonResponse({ ...managedStatus, managed: false, restart_supported: false }));

    expect(await screen.findByText("控制面未受管")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重启控制面" })).toBeDisabled();
  });

  it("confirms restart and waits for a different managed instance", async () => {
    let restartAccepted = false;
    let recoveryReads = 0;
    const onRecovered = renderControl(async (_input, init) => {
      if (init?.method === "POST") {
        restartAccepted = true;
        return jsonResponse({ accepted: true, request_id: "request-a", instance_id: "instance-a" }, 202);
      }
      if (!restartAccepted) return jsonResponse(managedStatus);
      recoveryReads += 1;
      if (recoveryReads === 1) throw new TypeError("connection reset");
      return jsonResponse({ ...managedStatus, instance_id: "instance-b", pid: 202 });
    });

    expect(await screen.findByText("控制面受管")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "重启控制面" }));
    expect(screen.getByRole("dialog", { name: "重启控制面？" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "确认重启" }));

    expect(await screen.findByText("正在重启…")).toBeInTheDocument();
    await waitFor(() => expect(onRecovered).toHaveBeenCalledOnce());
    expect(recoveryReads).toBe(2);
  });
});
