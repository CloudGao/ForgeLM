import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { TrainingPerformanceReport } from "./TrainingPerformanceReport";

it("shows measured throughput, memory and the selected configuration", () => {
  const configuration = { micro_batch_size: 8, activation_checkpointing: true, checkpoint_every_n_layers: 2 };
  render(<TrainingPerformanceReport report={{ schema_version: "forgelm.training-performance.v1", source_step: 359, memory_headroom_mb: 2048, recommendation: configuration, candidates: [{ configuration, status: "passed", tokens_per_second: 50000, peak_reserved_mb: 24576, loss_absolute_difference: 0, gradient_relative_l2: 0 }] }} />);
  expect(screen.getByText("已选用")).toBeInTheDocument();
  expect(screen.getByText("50,000")).toBeInTheDocument();
  expect(screen.getByText("24 GiB")).toBeInTheDocument();
  expect(screen.getByText("每 2 层")).toBeInTheDocument();
});
