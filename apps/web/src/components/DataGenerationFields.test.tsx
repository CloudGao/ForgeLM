import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DataGenerationFields } from "./DataGenerationFields";

vi.mock("../api", () => ({ api: async (path: string) => path === "/model-weights" ? { items: [{ id: "hy", state: "ready", version: 1, asset: { repository_id: "tencent/Hy-MT2-1.8B" } }] } : [] }));
afterEach(cleanup);

describe("CPT generation configuration", () => {
  it("switches mother-source QA to a bounded native vLLM configuration", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: {} }} onChange={onChange} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText("正文表达协议"), { target: { value: "dense-qa-v2" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: expect.objectContaining({ prompt_version: "dense-qa-v2", text_field: "dense_note", qa_limit: 100, qa_primary_limit: 50, batch_size: "auto", engine: "vllm", enable_thinking: false }) });
  });
  it("exposes structured translation as a native node with safe fixed defaults", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="fact_translation" config={{ parameters: { name: "三元组" } }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getByLabelText("推理引擎")).toBeDisabled();
    expect(screen.getByLabelText("推理引擎")).toHaveValue("vllm");
    expect(screen.queryByLabelText("正文表达协议")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("质量不通过时")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("输出数据集名称"), { target: { value: "中文三元组" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: expect.objectContaining({ name: "中文三元组", styles: ["facts"], document_mode: "whole", validation_retry_limit: 0, proper_name_policy: "source_exact" }) });
  });
  it("allows source names without changing weights and clears conflicting language options", () => {
    const onChange = vi.fn();
    const parameters = { engine: "vllm", prompt_version: "cpt-generation-v8", styles: ["facts"], model_weight_version_id: "original-model", chinese_output_constraint: true, fact_input_selection: "chinese_ready" };
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getByLabelText(/无可靠译名的专名/)).toHaveValue("translate");
    fireEvent.change(screen.getByLabelText(/无可靠译名的专名/), { target: { value: "source_exact" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { ...parameters, proper_name_policy: "source_exact", chinese_output_constraint: false, fact_input_selection: "standard", enable_thinking: false } });
  });

  it("configures preflight selection and single-pass fallback without changing model", () => {
    const onChange = vi.fn();
    const parameters = { engine: "vllm", prompt_version: "cpt-generation-v8", styles: ["facts"], model_weight_version_id: "original-model" };
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getByLabelText(/生成前选卡/)).toHaveValue("standard");
    expect(screen.getByLabelText(/质量不通过时/)).toHaveValue("1");
    fireEvent.change(screen.getByLabelText(/生成前选卡/), { target: { value: "chinese_ready" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { ...parameters, fact_input_selection: "chinese_ready", enable_thinking: false } });
    fireEvent.change(screen.getByLabelText(/质量不通过时/), { target: { value: "0" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { ...parameters, validation_retry_limit: 0, enable_thinking: false } });
  });
  it("keeps language constraints opt-in with an explicit factual quality warning", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { engine: "vllm", prompt_version: "cpt-generation-v7", styles: ["facts"] } }} onChange={onChange} /></QueryClientProvider>);
    expect((screen.getByLabelText(/前置纯中文约束/) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText(/不能保证译名或事实正确/)).toBeTruthy();
    fireEvent.click(screen.getByLabelText(/前置纯中文约束/));
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { engine: "vllm", prompt_version: "cpt-generation-v7", styles: ["facts"], chinese_output_constraint: true, enable_thinking: false } });
  });
  it("selects interleaved styles while preserving generation parameters", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { prompt_version: "cpt-generation-v7", styles: ["textbook", "facts"] } }} onChange={onChange} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText(/文体生成顺序/), { target: { value: "record_major" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v7", styles: ["textbook", "facts"], style_order: "record_major", enable_thinking: false } });
  });
  it("switches from document styles to textbook and facts only", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { prompt_version: "cpt-generation-v6", styles: ["retrospective"] } }} onChange={onChange} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText("正文表达协议"), { target: { value: "cpt-generation-v7" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v7", styles: ["textbook", "facts"], document_mode: "whole", enable_thinking: false } });
  });
  it("switches to two document styles and permits one separate output", () => {
    const onChange = vi.fn();
    const client = new QueryClient();
    const { rerender } = render(<QueryClientProvider client={client}><DataGenerationFields kind="knowledge" config={{ parameters: { prompt_version: "cpt-generation-v3", styles: ["plain"], temperature: 0.2 } }} onChange={onChange} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText("正文表达协议"), { target: { value: "cpt-generation-v5" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v5", styles: ["retrospective", "thematic"], temperature: 0.2, document_mode: "whole", enable_thinking: false } });
    rerender(<QueryClientProvider client={client}><DataGenerationFields kind="knowledge" config={{ parameters: { prompt_version: "cpt-generation-v5", styles: ["retrospective", "thematic"] } }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
    fireEvent.click(screen.getByLabelText("主题式"));
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v5", styles: ["retrospective"], enable_thinking: false } });
    fireEvent.change(screen.getByLabelText("正文表达协议"), { target: { value: "cpt-generation-v4" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v4", styles: ["textbook", "plain", "causal", "comparison", "distinction"], enable_thinking: false } });
  });
  it("selects local weights and fixes thinking off", async () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="translation" config={{ parameters: { name: "译文", enable_thinking: false } }} onChange={onChange} /></QueryClientProvider>);
    await screen.findByText(/tencent\/Hy-MT2-1.8B · V1/);
    fireEvent.change(screen.getByLabelText(/本地模型权重版本/), { target: { value: "hy" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { name: "译文", enable_thinking: false, model_weight_version_id: "hy" } });
    expect(screen.getByText(/Thinking 已强制关闭/)).toBeTruthy();
    expect(screen.getByLabelText("批处理模式")).toHaveValue("auto");
  });

  it("offers all five rewrite styles and switches to an upstream binding", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { name: "知识", input_dataset_version_id: "source" } }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getAllByRole("checkbox")).toHaveLength(5);
    fireEvent.change(screen.getByLabelText("输入数据来源"), { target: { value: "upstream" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { name: "知识" }, input_bindings: { input_dataset_version_id: { node_id: "", output_path: "dataset_version_id" } } });
  });

  it("shows whole-article filtering without a chunk control", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { engine: "vllm", quantization: "fp8", document_mode: "whole", max_source_characters: 3000, prompt_version: "cpt-generation-v2" } }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getByLabelText("文章处理方式")).toHaveValue("whole");
    expect(screen.getByLabelText(/原文字符上限/)).toHaveValue(3000);
    expect(screen.queryByLabelText("每块最大字符")).toBeNull();
    expect(screen.getByLabelText("正文表达协议")).toHaveValue("cpt-generation-v2");
    expect(screen.getByLabelText(/权重量化/)).toHaveValue("fp8");
    expect(screen.getByLabelText(/输出达到 token 上限/)).toHaveValue("skip");
    fireEvent.change(screen.getByLabelText(/输出达到 token 上限/), { target: { value: "fail" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { engine: "vllm", quantization: "fp8", document_mode: "whole", max_source_characters: 3000, prompt_version: "cpt-generation-v2", output_length_limit_policy: "fail", enable_thinking: false } });
  });

  it("selects structural rewriting explicitly without changing existing settings", () => {
    const onChange = vi.fn();
    render(<QueryClientProvider client={new QueryClient()}><DataGenerationFields kind="knowledge" config={{ parameters: { prompt_version: "cpt-generation-v3", temperature: 0.2, styles: ["textbook"] } }} onChange={onChange} /></QueryClientProvider>);
    expect(screen.getByLabelText("正文表达协议")).toHaveValue("cpt-generation-v3");
    fireEvent.change(screen.getByLabelText("正文表达协议"), { target: { value: "cpt-generation-v4" } });
    expect(onChange).toHaveBeenLastCalledWith({ parameters: { prompt_version: "cpt-generation-v4", temperature: 0.2, styles: ["textbook"], enable_thinking: false } });
  });
});
