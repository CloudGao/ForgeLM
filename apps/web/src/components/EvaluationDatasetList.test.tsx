import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { EvaluationDatasetList, evaluationDatasetRows } from "./EvaluationDatasetList";

afterEach(cleanup);

describe("unified evaluation datasets", () => {
  const catalog = [{ id: "lm-eval:c3", name: "C3", primary_metric: "acc" }, { id: "lm-eval:mmlu", name: "MMLU", primary_metric: "acc" }];
  const assets = [
    { id: "c3-v1", project_id: "global", name: "C3", version: 1, source_type: "benchmark", benchmark_config: { id: "lm-eval:c3", primary_metric: "acc" } },
    { id: "hf-v1", project_id: "global", name: "阅读题", version: 1, source_type: "huggingface", source_config: { dataset_version_id: "data-v1" }, artifact_id: "snapshot", sample_count: 10, scoring_config: { metric: "f1" } },
    { id: "hf-v2", project_id: "global", name: "阅读题", version: 2, source_type: "huggingface", source_config: { dataset_version_id: "data-v1" }, artifact_id: "snapshot", sample_count: 10, scoring_config: { metric: "exact_match" } },
    { id: "local-v1", project_id: "global", name: "我的题目", version: 1, source_type: "upload", artifact_id: "upload", sample_count: 2 },
  ];

  it("shows all sources together and filters without duplicating registered catalog items or configured imports", () => {
    const rows = evaluationDatasetRows(catalog, assets, [{ id: "data", name: "已构建来源", source_type: "huggingface", current_version: { id: "data-v1", state: "ready" } }]);
    render(<EvaluationDatasetList rows={rows} />);
    expect(screen.getAllByText("C3")).toHaveLength(1);
    expect(screen.getAllByText("阅读题")).toHaveLength(1);
    expect(screen.queryByText("已构建来源")).not.toBeInTheDocument();
    expect(screen.getByText("2 个版本")).toBeInTheDocument();
    expect(screen.getByText("MMLU")).toBeInTheDocument();
    expect(screen.getByText("我的题目")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("筛选评测集来源"), { target: { value: "Hugging Face" } });
    expect(screen.getByText("阅读题")).toBeInTheDocument();
    expect(screen.queryByText("C3")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("搜索评测集"), { target: { value: "f1" } });
    expect(screen.getByText("阅读题")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("搜索评测集"), { target: { value: "missing" } });
    expect(screen.getByText("没有匹配的评测集")).toBeInTheDocument();
  });

  it("retains a newly imported source version and shows preparation separately from runtime acquisition", () => {
    const rows = evaluationDatasetRows(catalog, assets, [
      { id: "data", name: "新来源版本", source_type: "huggingface", current_version: { id: "data-v2", version: 2, state: "ready", artifact_id: "new" } },
      { id: "waiting", name: "下载中的题目", source_type: "http", current_version: { id: "waiting-v1", state: "running" } },
    ]);
    render(<EvaluationDatasetList rows={rows} />);
    expect(screen.getByText("新来源版本")).toBeInTheDocument();
    expect(screen.getByText("待配置评测")).toBeInTheDocument();
    expect(screen.getByText("下载中的题目")).toBeInTheDocument();
    expect(screen.getAllByText("运行时获取")).toHaveLength(2);
  });

  it("keeps exact project versions and distinct assets with the same name", () => {
    const rows = evaluationDatasetRows(catalog, [...assets, { ...assets[0], id: "other", project_id: "legacy", source_type: "upload", benchmark_config: {} }], [], false);
    expect(rows.filter((row) => row.item.name === "C3")).toHaveLength(2);
    expect(rows.filter((row) => row.item.name === "阅读题").map((row) => row.item.id)).toEqual(["hf-v1", "hf-v2"]);
    expect(rows.some((row) => row.kind === "builtin" && row.item.id === "lm-eval:c3")).toBe(false);
  });
});
