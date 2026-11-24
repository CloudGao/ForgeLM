import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { executionPayload, Progress, StatusBadge } from "./ui";

describe("shared status UI", () => {
  it("renders localized task state", () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText("运行中")).toBeInTheDocument();
  });

  it("clamps progress to a percentage", () => {
    render(<Progress value={1.5} />);
    expect(screen.getByLabelText("进度 100%")).toBeInTheDocument();
  });

  it("builds a structured command without a shell string", () => {
    const form = new FormData();
    form.set("adapter_id", "builtin.command.training");
    form.set("execution_executable", "python");
    form.set("execution_args", "engine.py\n--config\ntrain.json");
    form.set("execution_outputs", "checkpoint=outputs/final");
    form.set("execution_gpus", "2");
    expect(executionPayload(form)).toEqual({
      adapter_id: "builtin.command.training",
      launcher_id: "builtin.local-process",
      resource_request: { gpus: 2 },
      command: {
        executable: "python",
        cwd: null,
        args: ["engine.py", "--config", "train.json"],
        env: {},
        outputs: [{ kind: "checkpoint", path: "outputs/final" }],
      },
    });
  });

  it("uses the built-in tokenizer adapter without requiring a command", () => {
    const form = new FormData();
    form.set("adapter_id", "builtin.hf-tokenizer");
    form.set("execution_gpus", "0");
    expect(executionPayload(form)).toEqual({
      adapter_id: "builtin.hf-tokenizer",
      launcher_id: "builtin.local-process",
      resource_request: { gpus: 0 },
      command: {},
    });
  });
});
