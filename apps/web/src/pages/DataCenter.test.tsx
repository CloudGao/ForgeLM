import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Dataset, Mixture } from "../types";
import { DataCenter } from "./DataCenter";
import { GlobalDataCenter } from "./GlobalDataCenter";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function renderDataCenter(initialEntry = "/projects/project-1/data") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={[initialEntry]}><Routes><Route path="/projects/:projectId/data" element={<DataCenter />} /></Routes></MemoryRouter></QueryClientProvider>);
}

function renderGlobalDataCenter(initialEntry = "/data") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={[initialEntry]}><DataCenter projectIdOverride="catalog-project" globalMode /></MemoryRouter></QueryClientProvider>);
}

function renderGlobalDataEntry() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={["/data"]}><GlobalDataCenter /></MemoryRouter></QueryClientProvider>);
}

const dataset: Dataset = {
  id: "dataset-1",
  project_id: "project-1",
  name: "Real Corpus",
  source_type: "local",
  source_uri: "F:\\data\\real.jsonl",
  source_config: { copy_mode: "managed" },
  license: "MIT",
  language: "zh",
  tags: ["真实导入"],
  status: "ready",
  current_version: {
    id: "version-1",
    version: 1,
    tier: "bronze",
    state: "ready",
    raw_size_bytes: 128,
    document_count: 1,
    token_estimate: 4,
    automatic_quality_score: 100,
    train_ready: true,
    artifact_id: "snapshot-1",
    record_manifest_artifact_id: "records-1",
    materialization: {
      kind: "imported",
      state: "ready",
      primary_location: { id: "location-primary", storage_type: "managed", uri: "file:///managed/snapshot-1", worker_id: null, last_verified_at: "2026-08-30T00:00:00Z" },
      replicas: [],
      recovery: { can_replay: false, replay_adapter_id: null },
    },
    quality_report: { analyzed_document_count: 1, parsed_file_count: 1, garbled_rate: 0, duplicate_rate: 0, pii_rate: 0, ad_template_rate: 0, language_distribution: { zh: 100 }, length_distribution: { "0-128": 100 }, unsupported_files: [] },
    sample_preview: [{ sample_key: "real.jsonl:1", source_file: "real.jsonl", raw_text: "真实样本", cleaned_text: "真实样本", language: "zh", length: 4, flags: [] }],
    source_snapshot: { fetched: true, file_count: 1, raw_size_bytes: 128, files: [{ path: "real.jsonl", size_bytes: 128, sha256: "a".repeat(64) }] },
  },
  versions: [],
};
dataset.versions = [dataset.current_version!];

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Data center loading continuity", () => {
  it("keeps a visible workspace while the global catalog context is loading", async () => {
    let resolveContext!: (response: Response) => void;
    const contextResponse = new Promise<Response>((resolve) => { resolveContext = resolve; });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/catalog/context")) return contextResponse;
      if (url.endsWith("/api/catalog/datasets") || url.endsWith("/api/catalog/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderGlobalDataEntry();

    expect(screen.getByRole("status", { name: "正在连接全局数据工作区…" })).toBeInTheDocument();

    resolveContext(jsonResponse({ catalog_project_id: "catalog-project", workspace_uri: "F:\\catalog" }));
    expect(await screen.findByRole("heading", { name: "数据集" })).toBeInTheDocument();
  });

  it("renders a clicked dataset from the catalog snapshot while its detail request is pending", async () => {
    let resolveDetail!: (response: Response) => void;
    const detailResponse = new Promise<Response>((resolve) => { resolveDetail = resolve; });
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return detailResponse;
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderDataCenter();
    fireEvent.click(await screen.findByText("Real Corpus"));

    expect(screen.getByRole("heading", { name: "Real Corpus", level: 1 })).toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "正在读取数据集详情" })).not.toBeInTheDocument();

    resolveDetail(jsonResponse({ ...dataset, name: "Fresh Corpus" }));
    expect(await screen.findByRole("heading", { name: "Fresh Corpus", level: 1 })).toBeInTheDocument();
  });
});

describe("Dataset materialization", () => {
  it("shows the primary location and submits the automatically selected route", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      if (url.endsWith("/api/workers")) return jsonResponse([{ id: "worker-1", name: "gpu-lab-01", status: "online" }]);
      if (url.endsWith("/api/dataset-versions/version-1/materializations") && init?.method === "POST") return jsonResponse({ id: "materialization-1", status: "queued", chosen_route: { strategy: "managed_copy" }, task_id: "task-copy" }, 202);
      if (url.endsWith("/api/dataset-versions/version-1/materializations")) return jsonResponse([]);
      if (url.includes("/api/dataset-versions/version-1/materialization-routes?target_worker_id=worker-1")) return jsonResponse({ routes: [{ route_id: "managed-copy:location-primary", strategy: "managed_copy", feasible: true, reason: "从控制面托管位置下载并校验" }] });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();
    fireEvent.click(await screen.findByText("Real Corpus"));

    expect(await screen.findByText("数据物化与副本")).toBeInTheDocument();
    const primaryLocation = screen.getByRole("article", { name: "主位置 控制面托管" });
    expect(within(primaryLocation).getByText("不绑定 Worker")).toBeInTheDocument();
    expect(within(primaryLocation).getByText("file:///managed/snapshot-1")).toBeInTheDocument();
    expect(within(primaryLocation).getByText("控制面托管位置")).toBeInTheDocument();
    fireEvent.change(await screen.findByLabelText("目标 Worker"), { target: { value: "worker-1" } });
    expect(await screen.findByDisplayValue(/managed_copy/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "同步到 Worker" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/dataset-versions/version-1/materializations") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/dataset-versions/version-1/materializations") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toEqual({
      target_worker_id: "worker-1",
      selection_mode: "auto",
      route_id: null,
    });
  });

  it("shows an HF version as copyable and replay-downloadable", async () => {
    const hfVersion = {
      ...dataset.current_version!,
      materialization: {
        ...dataset.current_version!.materialization!,
        kind: "download",
        primary_location: {
          id: "location-primary",
          storage_type: "external",
          uri: "file:///data/forgelm/data-worker/coig",
          worker_id: "worker-data",
          availability_state: "available",
          last_verified_at: "2026-08-30T00:00:00Z",
        },
        replicas: [{
          id: "location-replica",
          storage_type: "external",
          uri: "file:///srv/datasets/coig",
          worker_id: "worker-replica",
          availability_state: "available",
          last_verified_at: "2026-08-30T01:00:00Z",
        }],
        recovery: { can_replay: true, replay_adapter_id: "builtin.data-import", source_type: "huggingface", source_uri: "BAAI/COIG" },
      },
    };
    const hfDataset = {
      ...dataset,
      source_type: "huggingface",
      source_uri: "BAAI/COIG",
      current_version: hfVersion,
      versions: [hfVersion],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([hfDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(hfDataset);
      if (url.endsWith("/api/workers")) return jsonResponse([
        { id: "worker-data", name: "data-wsl-DESKTOP-B41QTPJ", status: "online" },
        { id: "worker-replica", name: "gpu-lab-01", status: "offline" },
      ]);
      if (url.endsWith("/api/dataset-versions/version-1/materializations")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderDataCenter();
    fireEvent.click(await screen.findByText("Real Corpus"));

    expect(
      await screen.findByText("download · 可从主位置复制 / 按冻结来源重新下载"),
    ).toBeInTheDocument();
    expect(screen.getByText("huggingface · BAAI/COIG")).toBeInTheDocument();
    const primaryLocation = await screen.findByRole("article", { name: "主位置 data-wsl-DESKTOP-B41QTPJ" });
    expect(within(primaryLocation).getByText("worker-data")).toBeInTheDocument();
    expect(within(primaryLocation).getByText("file:///data/forgelm/data-worker/coig")).toBeInTheDocument();
    expect(within(primaryLocation).getByText("Worker 私有外部位置")).toBeInTheDocument();
    const replicaLocation = screen.getByRole("article", { name: "已验证副本 gpu-lab-01" });
    expect(within(replicaLocation).getByText("worker-replica")).toBeInTheDocument();
    expect(within(replicaLocation).getByText("file:///srv/datasets/coig")).toBeInTheDocument();
  });

  it("does not mislabel a missing backend contract as imported and pending", async () => {
    const legacyVersion = { ...dataset.current_version!, materialization: undefined };
    const legacyDataset = {
      ...dataset,
      current_version: legacyVersion,
      versions: [legacyVersion],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([legacyDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(legacyDataset);
      if (url.endsWith("/api/workers")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));

    renderDataCenter();
    fireEvent.click(await screen.findByText("Real Corpus"));

    expect(await screen.findByText("物化契约尚未加载")).toBeInTheDocument();
    expect(screen.queryByText(/imported ·/)).not.toBeInTheDocument();
  });
});

describe("Data center real imports", () => {
  it("makes the structured SQL viewer the default Data Studio and keeps manifest filtering", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/catalog/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && init?.method === "POST") return jsonResponse({ id: "viewer-task-1", observed_state: "queued", kind: "data_query", title: "查询数据", created_at: "2026-08-15T00:00:00Z" }, 202);
      if (url.includes("/api/tasks/viewer-task-1")) return jsonResponse({ id: "viewer-task-1", observed_state: "succeeded", kind: "data_query", title: "查询数据", created_at: "2026-08-15T00:00:00Z", result: { engine: "duckdb_dataset_viewer_v1", sql: "SELECT * FROM dataset", configuration: "default", configurations: ["default"], split: "train", splits: ["train", "test"], format: "parquet", files: ["data/train.parquet"], columns: [{ name: "messages", type: "STRUCT(role VARCHAR, content VARCHAR)[]" }, { name: "source", type: "VARCHAR" }], rows: [[[ { role: "user", content: "你好" } ], "真实样本"]], offset: 0, limit: 100, total_rows: 1, has_more: false, truncated_cells: 0, result_limited: false, elapsed_ms: 8 } });
      if (url.endsWith("/api/catalog/explorer/query") && init?.method === "POST") return jsonResponse({
        total_rows: 1,
        total_tokens: 4,
        materializable_as_mixture: true,
        sources: [{ dataset_version_id: "version-1" }],
        rows: [{ dataset_version_id: "version-1", dataset_name: "Real Corpus", dataset_version: 1, record_id: "record-1", logical_path: "real.jsonl", record_index: 0, language: "zh", character_length: 4, token_estimate: 4, content_hash: "a".repeat(64), preview: { raw_text: "真实样本" } }],
      });
      return jsonResponse([]);
    }));

    renderGlobalDataCenter();
    expect(await screen.findByRole("heading", { name: "数据集" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Data Studio" }));
    expect(await screen.findByRole("heading", { name: "Data Studio" })).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset SQL 查询")).toHaveValue("SELECT * FROM dataset");
    expect(await screen.findByRole("region", { name: "数据集二维表格" })).toBeInTheDocument();
    expect(screen.getByText("messages")).toBeInTheDocument();
    expect(screen.getByTitle("STRUCT(role VARCHAR, content VARCHAR)[]")).toBeInTheDocument();
    expect(screen.getByText("真实样本")).toBeInTheDocument();
    const viewed = requests.find((request) => request.url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && request.init?.method === "POST");
    expect(JSON.parse(String(viewed?.init?.body))).toMatchObject({ sql: "SELECT * FROM dataset", offset: 0, limit: 100 });
    fireEvent.change(screen.getByLabelText("Dataset SQL 查询"), { target: { value: "SELECT source FROM dataset" } });
    fireEvent.click(screen.getByRole("button", { name: "重置" }));
    expect(screen.getByLabelText("Dataset SQL 查询")).toHaveValue("SELECT * FROM dataset");
    await waitFor(() => expect(requests.filter((request) => request.url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && request.init?.method === "POST")).toHaveLength(2));
    const resetQuery = requests.filter((request) => request.url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && request.init?.method === "POST").at(-1);
    expect(JSON.parse(String(resetQuery?.init?.body))).toMatchObject({ sql: "SELECT * FROM dataset", offset: 0, limit: 100 });

    fireEvent.click(screen.getByRole("tab", { name: "稳定索引筛选" }));
    expect(screen.queryByRole("dialog", { name: /DuckDB/ })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("确定性抽样比例")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /Real Corpus · V1/ }));
    fireEvent.click(screen.getByRole("button", { name: "搜索记录" }));

    expect(await screen.findByText("真实样本", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("把可复现条件冻结为混合集")).toBeInTheDocument();
    const searched = requests.find((request) => request.url.endsWith("/api/catalog/explorer/query") && request.init?.method === "POST");
    expect(JSON.parse(String(searched?.init?.body))).toMatchObject({ dataset_version_ids: ["version-1"], sample_ratio: 1, seed: 42, offset: 0, limit: 100 });
  });

  it("labels a complete Data Studio result page with a modality scheme", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const scheme = { id: "scheme-1", code: "quality", name: "文本质量", modality: "text", description: "", status: "active", current_version_id: "scheme-v1", version_count: 1, current_version: { id: "scheme-v1", scheme_id: "scheme-1", version: 1, description: "", created_by: "tester", immutable: true, groups: [{ id: "group-1", scheme_version_id: "scheme-v1", code: "quality", name: "质量", description: "", selection_mode: "single", required: false, sort_order: 0, labels: [{ id: "label-good", group_id: "group-1", code: "good", name: "可用", description: "", color: "#22c55e", sort_order: 0 }] }] } };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([{ ...dataset, modality: "text" }]);
      if (url.endsWith("/api/catalog/mixtures")) return jsonResponse([]);
      if (url.includes("/api/catalog/label-schemes?modality=text")) return jsonResponse([scheme]);
      if (url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && init?.method === "POST") return jsonResponse({ id: "viewer-task-label", observed_state: "queued", kind: "data_query", title: "查询数据", created_at: "2026-08-23T00:00:00Z" }, 202);
      if (url.includes("/api/tasks/viewer-task-label")) return jsonResponse({ id: "viewer-task-label", observed_state: "succeeded", kind: "data_query", title: "查询数据", created_at: "2026-08-23T00:00:00Z", result: { engine: "duckdb_dataset_viewer_v2", sql: "SELECT * FROM dataset", configuration: "default", configurations: ["default"], split: "train", splits: ["train"], format: "parquet", files: ["real.parquet"], columns: [{ name: "text", type: "VARCHAR" }, { name: "_forgelm_labels", type: "VARCHAR[]" }], rows: [["真实样本", []]], row_identities: [{ record_id: "record-1", record_hash: "a".repeat(64) }], labelable: true, labelability_reason: "", offset: 0, limit: 100, total_rows: 1, has_more: false, truncated_cells: 0, result_limited: false, elapsed_ms: 8 } });
      if (url.endsWith("/api/catalog/dataset-versions/version-1/label-operations") && init?.method === "POST") return jsonResponse({ id: "operation-1", task_id: "label-task-1", task: { id: "label-task-1", observed_state: "queued", kind: "data_label", title: "打标数据", created_at: "2026-08-23T00:00:00Z" } }, 202);
      if (url.includes("/api/tasks/label-task-1")) return jsonResponse({ id: "label-task-1", observed_state: "running", kind: "data_label", title: "打标数据", current_phase: "freezing_label_selection", created_at: "2026-08-23T00:00:00Z" });
      return jsonResponse([]);
    }));

    renderGlobalDataCenter();
    fireEvent.click(await screen.findByRole("button", { name: "Data Studio" }));
    fireEvent.click(await screen.findByRole("button", { name: "打标签" }));
    fireEvent.change(screen.getByLabelText("质量标签"), { target: { value: "label-good" } });
    fireEvent.click(screen.getByRole("button", { name: "创建打标任务" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/catalog/dataset-versions/version-1/label-operations") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/catalog/dataset-versions/version-1/label-operations") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toMatchObject({ source_query_task_id: "viewer-task-label", scope: "page", action: "apply", scheme_version_id: "scheme-v1", label_ids: ["label-good"], records: [{ record_id: "record-1" }] });
  });

  it("shows first-class post-training datasets, mixtures, corpora, and views", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/supervision-datasets")) return jsonResponse([{ id: "sd-1", project_id: "project-1", name: "Instruction", description: "", current_version_id: "sdv-1", versions: [{ id: "sdv-1", version: 1, dataset_version_id: "version-1", dataset_name: "Real Corpus", dataset_version: 1, kind: "conversation", mapping_config: { messages_path: "messages" }, filter_config: {}, status: "ready", accepted_record_count: 10, rejected_record_count: 1, validation_report: {}, sample_preview: [], identity_hash: "a".repeat(64) }] }]);
      if (url.endsWith("/api/projects/project-1/supervision-mixtures")) return jsonResponse([{ id: "sm-1", project_id: "project-1", name: "SFT mix", description: "", current_version_id: "smv-1", versions: [{ id: "smv-1", version: 1, kind: "conversation", entries: [{}], seed: 42, status: "ready", identity_hash: "b".repeat(64) }] }]);
      if (url.endsWith("/api/projects/project-1/supervision-corpora")) return jsonResponse([{ id: "sc-1", format: "forgelm-sft-corpus-v1", objective_kind: "sft", example_count: 10, pair_count: 0, supervised_token_count: 100, status: "ready", size_bytes: 2048 }]);
      if (url.endsWith("/api/projects/project-1/supervision-training-views")) return jsonResponse([{ id: "sv-1", format: "forgelm-sft-training-index-v1", objective_kind: "sft", example_count: 10, pair_count: 0, budget_value: 100, budget_unit: "supervised_tokens", status: "ready", size_bytes: 80 }]);
      return jsonResponse([]);
    }));
    renderDataCenter("/projects/project-1/data?tab=post-training");
    expect(await screen.findByRole("button", { name: "后训练数据" })).toHaveClass("active");
    expect(screen.getByText("后训练数据是平台一等制品")).toBeInTheDocument();
    expect(await screen.findByText("Instruction")).toBeInTheDocument();
    expect(screen.getByText("SFT mix")).toBeInTheDocument();
    expect(screen.getByText("forgelm-sft-corpus-v1")).toBeInTheDocument();
    expect(screen.getByText("forgelm-sft-training-index-v1")).toBeInTheDocument();
  });

  it("filters and combines supervision sources with independent weights", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const supervisionDatasets = [
      { id: "sd-zh", project_id: "project-1", name: "Chinese Instructions", description: "通用中文", current_version_id: "sdv-zh", versions: [{ id: "sdv-zh", version: 1, dataset_version_id: "dataset-zh-v1", dataset_name: "Chinese Raw", dataset_version: 1, kind: "conversation", status: "ready", accepted_record_count: 120, rejected_record_count: 2, identity_hash: "a".repeat(64), mapping_config: {}, filter_config: {}, validation_report: {}, sample_preview: [] }] },
      { id: "sd-code", project_id: "project-1", name: "Code Assistant", description: "代码指令", current_version_id: "sdv-code", versions: [{ id: "sdv-code", version: 2, dataset_version_id: "dataset-code-v3", dataset_name: "Code Raw", dataset_version: 3, kind: "conversation", status: "ready", accepted_record_count: 80, rejected_record_count: 1, identity_hash: "b".repeat(64), mapping_config: {}, filter_config: {}, validation_report: {}, sample_preview: [] }] },
      { id: "sd-pref", project_id: "project-1", name: "Preference Pairs", description: "偏好数据", current_version_id: "sdv-pref", versions: [{ id: "sdv-pref", version: 1, dataset_version_id: "dataset-pref-v1", dataset_name: "Preference Raw", dataset_version: 1, kind: "preference", status: "ready", accepted_record_count: 60, rejected_record_count: 0, identity_hash: "c".repeat(64), mapping_config: {}, filter_config: {}, validation_report: {}, sample_preview: [] }] },
    ];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/supervision-datasets")) return jsonResponse(supervisionDatasets);
      if (url.endsWith("/api/projects/project-1/supervision-mixtures") && init?.method === "POST") return jsonResponse({ id: "mix-version-new" }, 201);
      if (url.endsWith("/api/projects/project-1/supervision-mixtures")) return jsonResponse([]);
      return jsonResponse([]);
    }));
    renderDataCenter("/projects/project-1/data?tab=post-training");

    await screen.findByText("Chinese Instructions");
    fireEvent.click(screen.getByRole("button", { name: "创建后训练混合" }));
    const dialog = await screen.findByRole("dialog", { name: "创建后训练混合" });
    expect(within(dialog).getByRole("button", { name: "保存后训练混合" })).toBeDisabled();

    fireEvent.click(within(dialog).getByRole("button", { name: "筛选来源" }));
    fireEvent.change(within(dialog).getByRole("textbox", { name: "搜索后训练混合来源" }), { target: { value: "Code" } });
    expect(within(dialog).getByText("Code Assistant")).toBeInTheDocument();
    expect(within(dialog).queryByText("Chinese Instructions")).not.toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("button", { name: "选择筛选结果" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "重置筛选" }));
    fireEvent.click(within(dialog).getByRole("checkbox", { name: "Chinese Instructions V1 来源选择" }));
    fireEvent.change(within(dialog).getByRole("spinbutton", { name: "Chinese Instructions V1 采样权重" }), { target: { value: "3" } });
    expect(within(dialog).getByText("75.0%")).toBeInTheDocument();
    expect(within(dialog).getByText("25.0%")).toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText("名称"), { target: { value: "Balanced SFT" } });
    fireEvent.change(within(dialog).getByLabelText("说明"), { target: { value: "中文与代码组合" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "保存后训练混合" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/supervision-mixtures") && request.init?.method === "POST")).toBe(true));
    const saved = requests.find((request) => request.url.endsWith("/api/projects/project-1/supervision-mixtures") && request.init?.method === "POST");
    expect(JSON.parse(String(saved?.init?.body))).toEqual({
      name: "Balanced SFT",
      description: "中文与代码组合",
      entries: [
        { supervision_dataset_version_id: "sdv-zh", weight: 3 },
        { supervision_dataset_version_id: "sdv-code", weight: 1 },
      ],
      seed: 42,
    });
  });

  it("opens supervision mixture details, edits metadata, and deletes from the drawer", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const supervisionDataset = { id: "sd-1", project_id: "project-1", name: "Instruction", description: "", current_version_id: "sdv-1", versions: [{ id: "sdv-1", version: 1, dataset_version_id: "version-1", dataset_name: "Real Corpus", dataset_version: 1, kind: "conversation", mapping_config: {}, filter_config: {}, status: "ready", accepted_record_count: 10, rejected_record_count: 1, validation_report: {}, sample_preview: [], identity_hash: "a".repeat(64) }] };
    const mixture = { id: "sm-crud", project_id: "project-1", name: "SFT CRUD", description: "可维护混合", current_version_id: "smv-1", versions: [{ id: "smv-1", mixture_id: "sm-crud", version: 1, kind: "conversation", entries: [{ supervision_dataset_version_id: "sdv-1", supervision_name: "Instruction", supervision_version: 1, dataset_name: "Real Corpus", dataset_version: 1, accepted_record_count: 10, weight: 2, normalized_weight: 1 }], seed: 42, status: "ready", identity_hash: "b".repeat(64), created_at: "2026-08-14T00:00:00Z" }] };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/supervision-datasets")) return jsonResponse([supervisionDataset]);
      if (url.endsWith("/api/projects/project-1/supervision-mixtures")) return jsonResponse([mixture]);
      if (url.endsWith("/api/supervision-mixtures/sm-crud") && init?.method === "PATCH") return jsonResponse({ ...mixture, name: "SFT CRUD Renamed", description: "新说明" });
      if (url.endsWith("/api/supervision-mixtures/sm-crud") && init?.method === "DELETE") return new Response(null, { status: 204 });
      return jsonResponse([]);
    }));
    renderDataCenter("/projects/project-1/data?tab=post-training");

    fireEvent.click(await screen.findByRole("row", { name: "查看后训练混合 SFT CRUD" }));
    expect(await screen.findByRole("heading", { name: "SFT CRUD" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "来源与权重" })).toBeInTheDocument();
    expect(screen.getByText("weighted deficit V1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "编辑信息" }));
    const editDialog = await screen.findByRole("dialog", { name: "编辑后训练混合信息" });
    fireEvent.change(within(editDialog).getByLabelText("名称"), { target: { value: "SFT CRUD Renamed" } });
    fireEvent.change(within(editDialog).getByLabelText("说明"), { target: { value: "新说明" } });
    fireEvent.click(within(editDialog).getByRole("button", { name: "保存信息" }));
    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/supervision-mixtures/sm-crud") && request.init?.method === "PATCH")).toBe(true));
    const updated = requests.find((request) => request.url.endsWith("/api/supervision-mixtures/sm-crud") && request.init?.method === "PATCH");
    expect(JSON.parse(String(updated?.init?.body))).toEqual({ name: "SFT CRUD Renamed", description: "新说明" });

    fireEvent.click(screen.getByRole("button", { name: "删除" }));
    const deleteDialog = await screen.findByRole("dialog", { name: "删除 SFT CRUD" });
    fireEvent.click(within(deleteDialog).getByRole("button", { name: "确认删除" }));
    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/supervision-mixtures/sm-crud") && request.init?.method === "DELETE")).toBe(true));
  });

  it("shows automatically prepared training data and its full effective configuration", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets") || url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/tokenized-corpora")) return jsonResponse([{ id: "corpus-1", status: "ready", dataset_name: "Real Corpus", dataset_version: 1, tokenizer_name: "Locked BPE", tokenizer_version: 2, token_count: 4096, document_count: 10, shard_count: 1, selection_config: { mode: "all" }, encoding_config: { eos_policy: "append_if_available", token_dtype: "auto" }, identity_hash: "a".repeat(64), warnings: [] }]);
      if (url.endsWith("/api/projects/project-1/training-data-views")) return jsonResponse([{ id: "index-1", status: "ready", mixture_name: "Stable mixture", mixture_version: 1, tokenizer_name: "Locked BPE", sequence_length: 16, requested_tokens: 4096, token_count: 4096, sequence_count: 256, config_snapshot: { configuration: { consumption_policy: "unique_only", tail_policy: "drop", seed: 42 } }, warnings: [] }]);
      if (url.includes("/api/tasks?project_id=project-1")) return jsonResponse([
        { id: "task-current", kind: "data_tokenize_corpus", title: "Tokenized Corpus：Real Corpus V1 · Locked BPE V2", observed_state: "succeeded", desired_state: "running", progress: 1, error: {}, payload: { trigger_context: { source: "training_plan_prepare_data", action: "准备训练数据与运行提示", training_plan_id: "plan-1", training_plan_name: "300M 预训练", plan_revision_id: "revision-2", plan_revision: 2, selected_stages: [{ index: 0, name: "预训练", type: "pretrain" }] } }, result: {}, created_at: "2026-08-12T00:00:00Z" },
        { id: "task-failed", kind: "data_tokenize", title: "预分词：旧混合 V3 · 旧 Tokenizer V4", observed_state: "failed", desired_state: "running", progress: 0.88, error: { type: "ValueError", message: "旧流程错误仍须可追溯" }, payload: { target_tokens: 12000000000 }, result: {}, created_at: "2026-08-11T00:00:00Z" },
      ]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter("/projects/project-1/data?tab=prepared");

    expect(await screen.findByRole("button", { name: /已准备数据/ })).toHaveClass("active");
    expect(screen.getByText("这里是训练中心的自动准备结果")).toBeInTheDocument();
    expect(screen.getByText("编码数据")).toBeInTheDocument();
    expect(screen.getByText("训练读取索引")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "构建记录" })).toBeInTheDocument();
    expect(await screen.findByText("预分词：旧混合 V3 · 旧 Tokenizer V4")).toBeInTheDocument();
    expect(screen.getByText("历史预分词")).toBeInTheDocument();
    expect(screen.getByText("300M 预训练")).toBeInTheDocument();
    expect(screen.getByText("Revision 2 · 预训练（pretrain）")).toBeInTheDocument();
    expect(screen.getByText("旧版预分词入口")).toBeInTheDocument();
    expect(screen.getByText("历史任务未记录训练计划与阶段")).toBeInTheDocument();
    expect(screen.getByText("旧流程错误仍须可追溯")).toBeInTheDocument();
    expect(await screen.findByText("Real Corpus")).toBeInTheDocument();
    expect(screen.getAllByText("查看全部配置")).toHaveLength(2);
    expect(screen.getByText((_text, element) => element?.tagName === "STRONG" && element.textContent === "4,096 / 4,096")).toBeInTheDocument();
  });

  it("creates an ordered document conversion recipe without converting during import", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const catalog = { engine_adapter_id: "builtin.data-conversion", engine_version: "document-conversion-v1", operators: [
      { name: "detect_format", label: "自动识别格式", category: "识别", description: "识别文档容器", parameters: {} },
      { name: "extract_pdf", label: "提取 PDF 文本", category: "抽取", description: "提取文本型 PDF", parameters: {} },
      { name: "extract_docx", label: "提取 DOCX", category: "抽取", description: "提取段落与表格", parameters: { include_tables: { type: "boolean", default: true } } },
      { name: "extract_epub", label: "提取 EPUB", category: "抽取", description: "按 spine 提取章节", parameters: {} },
      { name: "normalize_whitespace", label: "规范空白", category: "加工", description: "统一空白", parameters: { max_blank_lines: { type: "integer", default: 1 } } },
      { name: "filter_empty", label: "过滤空记录", category: "加工", description: "过滤空记录", parameters: { min_characters: { type: "integer", default: 1 } } },
      { name: "chunk_text", label: "定长文本分块", category: "加工", description: "切分长文本", parameters: { max_characters: { type: "integer", default: 8192 }, overlap_characters: { type: "integer", default: 0 } } },
    ] };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-conversion-recipes") && init?.method === "POST") return jsonResponse({ id: "conversion-recipe-1", name: "文档统一转文本", versions: [] }, 201);
      if (url.endsWith("/api/projects/project-1/data-conversion-recipes")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/data-conversion-runs")) return jsonResponse([]);
      if (url.endsWith("/api/data-conversion/operators")) return jsonResponse(catalog);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据转换/ }));
    fireEvent.click((await screen.findAllByRole("button", { name: "新建转换配方" }))[0]);
    fireEvent.change(screen.getByLabelText("转换配方名称"), { target: { value: "文档统一转文本" } });
    fireEvent.change(screen.getByLabelText("添加转换算子"), { target: { value: "chunk_text" } });
    expect(screen.getByLabelText("转换步骤 7 算子")).toHaveValue("chunk_text");
    fireEvent.click(screen.getByRole("button", { name: "创建转换配方" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/data-conversion-recipes") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/data-conversion-recipes") && request.init?.method === "POST");
    const body = JSON.parse(String(created?.init?.body));
    expect(body).toMatchObject({ name: "文档统一转文本", engine_adapter_id: "builtin.data-conversion", execution_config: {} });
    expect(body.operators.map((operator: Record<string, unknown>) => Object.keys(operator)[0])).toEqual(["detect_format", "extract_pdf", "extract_docx", "extract_epub", "normalize_whitespace", "filter_empty", "chunk_text"]);
  });

  it("explicitly starts conversion from a ready raw snapshot", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const rawDataset: Dataset = { ...dataset, current_version: { ...dataset.current_version, train_ready: false }, versions: [{ ...dataset.current_version, train_ready: false }] };
    const recipe = { id: "conversion-recipe-1", project_id: "project-1", name: "文档转文本", description: "", current_version_id: "conversion-version-1", current_version: { id: "conversion-version-1", version: 1, operators: [{ detect_format: {} }, { extract_pdf: {} }] }, versions: [{ id: "conversion-version-1", version: 1, operators: [{ detect_format: {} }, { extract_pdf: {} }] }] };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-conversion-runs") && init?.method === "POST") return jsonResponse({ id: "conversion-run-1", status: "queued" }, 202);
      if (url.endsWith("/api/projects/project-1/data-conversion-recipes")) return jsonResponse([recipe]);
      if (url.endsWith("/api/projects/project-1/data-conversion-runs")) return jsonResponse([]);
      if (url.endsWith("/api/data-conversion/operators")) return jsonResponse({ engine_adapter_id: "builtin.data-conversion", engine_version: "document-conversion-v1", operators: [] });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([rawDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据转换/ }));
    expect(await screen.findByText("文档转文本")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "发起转换" })[0]);
    expect(await screen.findByRole("heading", { name: "发起数据转换" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /待转换/ })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("CPU 请求"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "创建转换运行" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/data-conversion-runs") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/data-conversion-runs") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toEqual({ recipe_version_id: "conversion-version-1", input_dataset_version_id: "version-1", resource_request: { cpus: 4, gpus: 0 } });
  });

  it("creates an immutable Data-Juicer recipe from the data processing tab", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-processing-recipes") && init?.method === "POST") return jsonResponse({ id: "recipe-1", name: "基础清洗", versions: [] }, 201);
      if (url.endsWith("/api/projects/project-1/data-processing-recipes")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/data-processing-runs")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据清洗/ }));
    fireEvent.click((await screen.findAllByRole("button", { name: "新建清洗配方" }))[0]);
    fireEvent.change(screen.getByLabelText("配方名称"), { target: { value: "基础清洗" } });
    fireEvent.change(screen.getByRole("textbox", { name: /并行进程数/ }), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("添加算子"), { target: { value: "clean_html_mapper" } });
    expect(screen.getByLabelText("步骤 3 算子")).toHaveValue("clean_html_mapper");
    fireEvent.click(screen.getByRole("button", { name: "创建清洗配方" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/data-processing-recipes") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/data-processing-recipes") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toMatchObject({ name: "基础清洗", engine_adapter_id: "builtin.data-juicer", execution_config: { np: 4 }, operators: [{ whitespace_normalization_mapper: {} }, { text_length_filter: { min_len: 10, max_len: 100000 } }, { clean_html_mapper: {} }] });
  });

  it("keeps catalog-external Data-Juicer operators in advanced JSON mode", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-processing-recipes") && init?.method === "POST") return jsonResponse({ id: "recipe-1", name: "自定义处理", versions: [] }, 201);
      if (url.endsWith("/api/projects/project-1/data-processing-recipes")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/data-processing-runs")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据清洗/ }));
    fireEvent.click((await screen.findAllByRole("button", { name: "新建清洗配方" }))[0]);
    fireEvent.change(screen.getByLabelText("配方名称"), { target: { value: "自定义处理" } });
    fireEvent.click(screen.getByRole("button", { name: "高级 JSON" }));
    fireEvent.change(screen.getByLabelText("完整算子 JSON"), { target: { value: '[{"project_custom_mapper":{"mode":"strict"}}]' } });
    fireEvent.click(screen.getByRole("button", { name: "创建清洗配方" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/data-processing-recipes") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/data-processing-recipes") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body)).operators).toEqual([{ project_custom_mapper: { mode: "strict" } }]);
  });

  it("freezes the selected recipe and input when starting data processing", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const recipe = { id: "recipe-1", project_id: "project-1", name: "基础清洗", description: "", current_version_id: "recipe-version-1", current_version: { id: "recipe-version-1", version: 1, operators: [{ whitespace_normalization_mapper: {} }], execution_config: { np: 2 } }, versions: [{ id: "recipe-version-1", version: 1, operators: [{ whitespace_normalization_mapper: {} }], execution_config: { np: 2 } }] };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/data-processing-runs") && init?.method === "POST") return jsonResponse({ id: "run-1", status: "queued" }, 202);
      if (url.endsWith("/api/projects/project-1/data-processing-recipes")) return jsonResponse([recipe]);
      if (url.endsWith("/api/projects/project-1/data-processing-runs")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据清洗/ }));
    expect(await screen.findByText("基础清洗")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "发起清洗" })[0]);
    expect(await screen.findByRole("heading", { name: "发起数据清洗" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("CPU 请求"), { target: { value: "6" } });
    fireEvent.click(screen.getByRole("button", { name: "创建清洗运行" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/projects/project-1/data-processing-runs") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/data-processing-runs") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toEqual({ recipe_version_id: "recipe-version-1", input_dataset_version_id: "version-1", output_tier: "silver", resource_request: { cpus: 6, gpus: 0 } });
  });

  it("opens the reanalysis confirmation from a tokenizer repair link", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/datasets/dataset-1")) return jsonResponse(dataset);
      if (url.endsWith("/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/projects/project-1/mixtures")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    }));

    renderDataCenter("/projects/project-1/data?tab=datasets&dataset=dataset-1&action=reanalyze");

    expect(await screen.findByRole("heading", { name: "重新分析数据" })).toBeInTheDocument();
    expect(screen.getByText(/重新计算文件 SHA-256/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始重新分析" })).toBeInTheDocument();
  });

  it("opens a saved mixture and explains normalized token consumption", async () => {
    const secondDataset: Dataset = { ...dataset, id: "dataset-2", name: "English Corpus", language: "en", license: "Apache-2.0", current_version: { ...dataset.current_version, id: "version-2", token_estimate: 1_000_000_000 }, versions: [{ ...dataset.current_version, id: "version-2", token_estimate: 1_000_000_000 }] };
    const mixture: Mixture = { id: "mixture-1", name: "12B Mix", description: "中英文预训练", current_version: { id: "mix-version-1", version: 1, target_tokens: 12_000_000_000, status: "ready", locked: true, seed: 42, manifest_artifact_id: "manifest-1", manifest_hash: "a".repeat(64), resolved_document_count: 2_000_000, estimated_steps: 0, entries: [{ dataset_version_id: "version-1", name: "Real Corpus", weight: 3, normalized_weight: .75, available_tokens: 3_000_000_000 }, { dataset_version_id: "version-2", name: "English Corpus", weight: 1, normalized_weight: .25, available_tokens: 1_000_000_000 }], gate_result: { status: "passed", issues: [] } }, versions: [] };
    mixture.versions = [mixture.current_version!];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset, secondDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([mixture]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据混合/ }));
    fireEvent.click(await screen.findByRole("button", { name: "打开数据混合 12B Mix" }));

    expect(await screen.findByText("版本交付固定的数据流")).toBeInTheDocument();
    expect(screen.getByText("aaaaaaaaaaaa")).toBeInTheDocument();
    expect(screen.getByText("90亿")).toBeInTheDocument();
    expect(screen.getByText("30亿")).toBeInTheDocument();
    expect(screen.getAllByText("3.00×")).toHaveLength(2);
  });

  it("filters mixture sources and creates a new immutable version", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const recipeVersion = { ...dataset.current_version, source_snapshot: { ...dataset.current_version?.source_snapshot, files: [{ path: "data/train/part-1.jsonl", size_bytes: 80 }, { path: "data/test/held-out.jsonl", size_bytes: 48 }] } };
    const recipeDataset: Dataset = { ...dataset, current_version: recipeVersion, versions: [recipeVersion] };
    const secondDataset: Dataset = { ...dataset, id: "dataset-2", name: "English Corpus", language: "en", current_version: { ...dataset.current_version, id: "version-2", version: 2, token_estimate: 10 }, versions: [{ ...dataset.current_version, id: "version-2", version: 2, token_estimate: 10 }] };
    const mixture: Mixture = { id: "mixture-1", name: "Editable Mix", description: "", current_version: { id: "mix-version-1", version: 1, target_tokens: 100, status: "ready", locked: true, seed: 42, manifest_artifact_id: "manifest-1", manifest_hash: "a".repeat(64), entries: [{ dataset_version_id: "version-1", name: "Real Corpus", weight: 1, normalized_weight: 1, available_tokens: 4 }], gate_result: { status: "passed", issues: [] } }, versions: [] };
    mixture.versions = [mixture.current_version!];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/mixtures/mixture-1/versions") && init?.method === "POST") return jsonResponse({ id: "mix-version-2", version: 2 }, 201);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([recipeDataset, secondDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([mixture]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据混合/ }));
    fireEvent.click(await screen.findByRole("button", { name: "打开数据混合 Editable Mix" }));
    fireEvent.click(screen.getAllByRole("button", { name: "创建新版本" }).at(-1)!);
    const sourceFilter = screen.getAllByRole("button", { name: "筛选子集" }).find((button) => !(button as HTMLButtonElement).disabled)!;
    fireEvent.click(sourceFilter);
    fireEvent.change(screen.getByLabelText("确定性抽样比例"), { target: { value: "50" } });
    fireEvent.change(screen.getByLabelText("Token 上限"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("data/train 路径规则"), { target: { value: "include" } });
    fireEvent.change(screen.getByLabelText("data/test/held-out.jsonl 路径规则"), { target: { value: "exclude" } });
    fireEvent.click(screen.getByRole("button", { name: "应用筛选" }));
    fireEvent.change(screen.getByLabelText("搜索混合来源"), { target: { value: "English" } });
    expect(screen.getByText("English Corpus")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getAllByRole("button", { name: "创建新版本" }).at(-1)!);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/mixtures/mixture-1/versions") && request.init?.method === "POST")).toBe(true));
    const saved = requests.find((request) => request.url.endsWith("/api/mixtures/mixture-1/versions") && request.init?.method === "POST");
    const savedEntries = JSON.parse(String(saved?.init?.body)).entries;
    expect(savedEntries.map((entry: any) => entry.dataset_version_id)).toEqual(["version-1", "version-2"]);
    expect(savedEntries[0].selection).toMatchObject({ sample_ratio: .5, max_tokens: 2, include_paths: ["data/train/**"], exclude_paths: ["data/test/held-out.jsonl"], seed: 42 });
    expect(savedEntries[0].selected_tokens).toBe(2);
  });

  it("uses the global catalog endpoint when versioning a legacy mixture", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const mixture: Mixture = { id: "legacy-mixture", name: "Legacy Global Mix", description: "", current_version: { id: "legacy-v1", version: 1, target_tokens: 100, status: "ready", locked: true, seed: 42, manifest_artifact_id: "manifest-1", manifest_hash: "a".repeat(64), entries: [{ dataset_version_id: "version-1", name: "Real Corpus", weight: 1, normalized_weight: 1, available_tokens: 4 }], gate_result: { status: "passed", issues: [] } }, versions: [] };
    mixture.versions = [mixture.current_version!];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.endsWith("/api/catalog/mixtures/legacy-mixture/versions") && init?.method === "POST") return jsonResponse({ id: "legacy-v2", version: 2 }, 201);
      if (url.endsWith("/api/catalog/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/catalog/mixtures")) return jsonResponse([mixture]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderGlobalDataCenter("/data?tab=mixtures");

    fireEvent.click(await screen.findByRole("button", { name: "打开数据混合 Legacy Global Mix" }));
    fireEvent.click(screen.getAllByRole("button", { name: "创建新版本" }).at(-1)!);
    fireEvent.click(screen.getAllByRole("button", { name: "创建新版本" }).at(-1)!);

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/catalog/mixtures/legacy-mixture/versions") && request.init?.method === "POST")).toBe(true));
  });

  it("opens the data import wizard from the mixture editor", async () => {
    const mixture: Mixture = { id: "mixture-1", name: "Needs Source", description: "", current_version: { id: "mix-version-1", version: 1, target_tokens: 100, entries: [{ dataset_version_id: "version-1", name: "Real Corpus", weight: 1, normalized_weight: 1, available_tokens: 4 }], gate_result: { status: "passed", issues: [] } }, versions: [] };
    mixture.versions = [mixture.current_version!];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([mixture]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: /数据混合/ }));
    fireEvent.click(await screen.findByRole("button", { name: "打开数据混合 Needs Source" }));
    fireEvent.click(screen.getByRole("button", { name: "创建新版本" }));
    fireEvent.click(screen.getByRole("button", { name: "添加新数据源" }));

    const wizard = screen.getByRole("dialog", { name: "新数据" });
    expect(within(wizard).getByText("Hugging Face")).toBeInTheDocument();
    expect(within(wizard).getByText("HTTP / HTTPS")).toBeInTheDocument();
    expect(within(wizard).getByText("本地文件")).toBeInTheDocument();
    expect(within(wizard).getByText("Git 仓库")).toBeInTheDocument();
    fireEvent.click(within(wizard).getByRole("button", { name: "关闭" }));
    expect(screen.getByRole("dialog", { name: "编辑 Needs Source" })).toBeInTheDocument();
  });

  it("shows the external transfer tool and live throughput", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const runningDataset: Dataset = {
      ...dataset,
      status: "running",
      source_type: "huggingface",
      source_uri: "org/corpus",
      source_config: { hf_endpoint: "https://hf-mirror.com" },
      current_version: { ...dataset.current_version, state: "running", train_ready: false, task_id: "task-live" },
      versions: [{ ...dataset.current_version, state: "running", train_ready: false, task_id: "task-live" }],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/tasks/task-live/control") && init?.method === "POST") return jsonResponse({ id: "task-live", observed_state: "stopping", desired_state: "cancelled" });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([runningDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(runningDataset);
      if (url.endsWith("/api/tasks/task-live")) return jsonResponse({
        id: "task-live",
        kind: "data_import",
        title: "导入数据",
        observed_state: "running",
        desired_state: "running",
        progress: 0.25,
        current_phase: "downloading",
        created_at: "2026-08-05T00:00:00Z",
        events: [{ id: 1, sequence: 1, event_type: "progress", created_at: "2026-08-05T00:00:01Z", payload: { tool: "hfd + aria2c", downloaded_bytes: 1_500_000, total_bytes: 6_000_000, speed_bps: 500_000, file_count_completed: 2, file_count_total: 8, eta: "00:09" } }],
      });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));

    const transfer = await screen.findByText(/hfd \+ aria2c.*ETA 00:09/);
    expect(transfer).toHaveTextContent(/1\.43 MB.*5\.72 MB.*488\.3 KB\/s.*2.*8/);
    fireEvent.click(screen.getByRole("button", { name: "停止分析" }));
    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/tasks/task-live/control") && request.init?.method === "POST")).toBe(true));
    const stopped = requests.find((request) => request.url.endsWith("/api/tasks/task-live/control") && request.init?.method === "POST");
    expect(JSON.parse(String(stopped?.init?.body))).toEqual({ action: "cancel", reason: "用户从数据详情停止分析" });
  });

  it("restarts a failed dataset download from the list", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const failedDataset: Dataset = {
      ...dataset,
      status: "failed",
      current_version: { ...dataset.current_version, state: "failed", train_ready: false, task_id: "task-1" },
      versions: [{ ...dataset.current_version, state: "failed", train_ready: false, task_id: "task-1" }],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/tasks/task-1/control") && init?.method === "POST") {
        return jsonResponse({ id: "task-1", observed_state: "queued" });
      }
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([failedDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByRole("button", { name: "重新下载" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/tasks/task-1/control") && request.init?.method === "POST")).toBe(true));
    const restarted = requests.find((request) => request.url.endsWith("/api/tasks/task-1/control") && request.init?.method === "POST");
    expect(JSON.parse(String(restarted?.init?.body))).toEqual({ action: "retry", reason: "用户从数据中心重新下载" });
    expect(screen.queryByRole("separator", { name: "调整数据集详情面板宽度" })).not.toBeInTheDocument();
  });

  it("explains an empty network timeout error in dataset detail", async () => {
    const failedDataset: Dataset = {
      ...dataset,
      status: "failed",
      current_version: { ...dataset.current_version, state: "failed", train_ready: false, task_id: "task-1" },
      versions: [{ ...dataset.current_version, state: "failed", train_ready: false, task_id: "task-1" }],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([failedDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(failedDataset);
      if (url.endsWith("/api/tasks/task-1")) return jsonResponse({ id: "task-1", observed_state: "failed", error: { type: "ConnectTimeout", message: "" }, attempts: [] });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));

    expect(await screen.findByText("连接数据源超时；请检查 Worker 网络后重新下载。")).toBeInTheDocument();
  });

  it("opens a dataset as a full-width resource page", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    expect(await screen.findByRole("heading", { name: "Real Corpus", level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "数据查看器" })).toBeInTheDocument();
    expect(screen.queryByRole("separator", { name: /调整数据集详情/ })).not.toBeInTheDocument();
    expect(document.querySelector(".detail-drawer")).not.toBeInTheDocument();
  });

  it("shows original dataset columns and read-only SQL in the detail viewer", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      if (url.endsWith("/api/catalog/dataset-versions/version-1/viewer/query") && init?.method === "POST") return jsonResponse({ id: "viewer-task-1", observed_state: "queued", kind: "data_query", title: "查询数据", created_at: "2026-08-15T00:00:00Z" }, 202);
      if (url.includes("/api/tasks/viewer-task-1")) return jsonResponse({ id: "viewer-task-1", observed_state: "succeeded", kind: "data_query", title: "查询数据", created_at: "2026-08-15T00:00:00Z", result: { engine: "duckdb_dataset_viewer_v1", sql: "SELECT * FROM dataset", configuration: "default", configurations: ["default"], split: "train", splits: ["train"], format: "parquet", files: ["real.parquet"], columns: [{ name: "messages", type: "STRUCT(role VARCHAR, content VARCHAR)[]" }, { name: "source", type: "VARCHAR" }], rows: [[[ { role: "user", content: "真实样本" } ], "smol-talk"]], offset: 0, limit: 100, total_rows: 1, has_more: false, truncated_cells: 0, result_limited: false, elapsed_ms: 8 } });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    fireEvent.click(await screen.findByRole("button", { name: "数据查看器" }));
    expect(await screen.findByRole("region", { name: "数据集二维表格" })).toBeInTheDocument();
    expect(screen.getByText("messages")).toBeInTheDocument();
    expect(screen.getByText("smol-talk")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset SQL 查询")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "标注" })).not.toBeInTheDocument();
    expect(requests.some((request) => request.url.includes("/annotations"))).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "统计与分布" }));
    expect(screen.getAllByText("0.00%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("未检测").length).toBeGreaterThan(0);
  });

  it("resolves, selects and plans Hugging Face files before creating a worker task", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/projects/project-1/datasets") && init?.method === "POST") return jsonResponse({ ...dataset, current_version: { ...dataset.current_version, state: "queued", task_id: "task-1" } }, 201);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/projects/project-1/datasets/huggingface/scan") && init?.method === "POST") return jsonResponse({ id: "scan-1", observed_state: "queued", kind: "hf_repository_scan", title: "scan", created_at: "2026-08-05T00:00:00Z" }, 202);
      if (url.endsWith("/api/tasks/scan-1")) return jsonResponse({ id: "scan-1", observed_state: "succeeded", kind: "hf_repository_scan", title: "scan", created_at: "2026-08-05T00:00:00Z", result: { repo_id: "org/corpus", requested_revision: "main", commit_hash: "a".repeat(40), total_size_bytes: 384, file_count: 2, files: [{ path: "data/train.parquet", size_bytes: 256 }, { path: "README.md", size_bytes: 128 }] } });
      if (url.endsWith("/api/projects/project-1/datasets/huggingface/plans") && init?.method === "POST") return jsonResponse({ id: "plan-1", observed_state: "queued", kind: "hf_download_plan", title: "plan", created_at: "2026-08-05T00:00:00Z" }, 202);
      if (url.endsWith("/api/tasks/plan-1")) return jsonResponse({ id: "plan-1", observed_state: "succeeded", kind: "hf_download_plan", title: "plan", created_at: "2026-08-05T00:00:00Z", result: { repo_id: "org/corpus", commit_hash: "a".repeat(40), file_count: 1, selected_paths: ["data/train.parquet"], selected_bytes: 256, cached_bytes: 64, download_bytes: 192, disk_free_bytes: 4096, disk_space_sufficient: true, endpoint_warning: "mirror metadata unavailable" } });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click((await screen.findAllByRole("button", { name: /新数据/ }))[0]);
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }));
    fireEvent.change(screen.getByLabelText("数据集名称"), { target: { value: "Downloaded Corpus" } });
    fireEvent.change(screen.getByLabelText("Repo ID"), { target: { value: "org/corpus" } });
    fireEvent.change(screen.getByLabelText(/^Hub 端点/), { target: { value: "mirror" } });
    fireEvent.click(screen.getByRole("button", { name: /解析仓库/ }));
    expect(await screen.findByText("data/train.parquet")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText(/data\/train\.parquet/));
    fireEvent.click(screen.getByRole("button", { name: /预检所选文件/ }));
    expect(await screen.findByText("空间检查通过")).toBeInTheDocument();
    expect(screen.getByText("镜像节点已回退到 Hugging Face 官方")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }));
    fireEvent.click(screen.getByRole("button", { name: "创建并开始下载" }));
    expect(await screen.findByText("下载任务已创建")).toBeInTheDocument();

    const created = requests.find((request) => request.url.endsWith("/api/projects/project-1/datasets") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toMatchObject({ modality: "text", source_uri: "org/corpus", source_config: { hf_plan_task_id: "plan-1" }, start_import: true, adapter_id: "builtin.data-import", resource_request: { gpus: 0 } });
    const planned = requests.find((request) => request.url.endsWith("/api/projects/project-1/datasets/huggingface/plans") && request.init?.method === "POST");
    expect(JSON.parse(String(planned?.init?.body))).toEqual({ scan_task_id: "scan-1", selected_paths: ["data/train.parquet"] });
    const scanned = requests.find((request) => request.url.endsWith("/api/projects/project-1/datasets/huggingface/scan") && request.init?.method === "POST");
    expect(JSON.parse(String(scanned?.init?.body))).toMatchObject({ hf_endpoint: "https://hf-mirror.com" });
  });

  it("keeps a planned Hugging Face physical source frozen while editing metadata", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const failedDataset: Dataset = {
      ...dataset,
      source_type: "huggingface",
      source_uri: "org/corpus",
      source_config: { revision: "main" },
      status: "failed",
      current_version: { ...dataset.current_version, state: "failed", task_id: "task-failed", train_ready: false },
      versions: [{ ...dataset.current_version, state: "failed", task_id: "task-failed", train_ready: false }],
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/datasets/dataset-1") && init?.method === "PATCH") return jsonResponse({ ...failedDataset, language: "zh-Hans" });
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([failedDataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(failedDataset);
      if (url.endsWith("/api/tasks/task-failed")) return jsonResponse({ id: "task-failed", observed_state: "failed", desired_state: "running", progress: 0, kind: "data_import", title: "导入失败", error: { type: "ConnectTimeout" }, created_at: "2026-08-04T00:00:00Z" });
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    fireEvent.click(await screen.findByRole("button", { name: "设置" }));
    expect(screen.getByLabelText("数据集仓库 ID")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("语言声明"), { target: { value: "zh-Hans" } });
    fireEvent.click(screen.getByRole("button", { name: "保存修改" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/datasets/dataset-1") && request.init?.method === "PATCH")).toBe(true));
    const updated = requests.find((request) => request.url.endsWith("/api/datasets/dataset-1") && request.init?.method === "PATCH");
    expect(JSON.parse(String(updated?.init?.body))).toMatchObject({ source_uri: "org/corpus", source_config: { revision: "main" }, language: "zh-Hans" });
  });

  it("edits metadata without treating unchanged local config as a new source", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    fireEvent.click(await screen.findByRole("button", { name: "设置" }));

    expect(screen.getByRole("button", { name: "保存修改" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "保存并重新导入" })).not.toBeInTheDocument();
  });

  it("deletes an unreferenced dataset after name confirmation", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    let deleted = false;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/datasets/dataset-1") && init?.method === "DELETE") { deleted = true; return new Response(null, { status: 204 }); }
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse(deleted ? [] : [dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    fireEvent.click(await screen.findByRole("button", { name: "删除" }));
    fireEvent.change(screen.getByLabelText(/输入“Real Corpus”确认/), { target: { value: "Real Corpus" } });
    fireEvent.click(screen.getByRole("button", { name: "确认删除" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/datasets/dataset-1") && request.init?.method === "DELETE")).toBe(true));
  });

  it("creates an analysis revision without changing the dataset version", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/api/datasets/dataset-1/reanalyze") && init?.method === "POST") {
        return jsonResponse({ ...dataset, current_version: { ...dataset.current_version, active_analysis: { id: "analysis-2", revision: 2, status: "queued", task_id: "task-2" } } }, 202);
      }
      if (url.endsWith("/api/projects/project-1/datasets")) return jsonResponse([dataset]);
      if (url.endsWith("/api/projects/project-1/mixtures")) return jsonResponse([]);
      if (url.endsWith("/api/datasets/dataset-1")) return jsonResponse(dataset);
      return jsonResponse({ message: `Unhandled request ${url}` }, 404);
    }));
    renderDataCenter();

    fireEvent.click(await screen.findByText("Real Corpus"));
    fireEvent.click(await screen.findByRole("button", { name: "重新分析" }));
    expect(screen.getByText("数据版本不会变化")).toBeInTheDocument();
    expect(screen.getByText(/不改变内容版本/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "开始重新分析" }));

    await waitFor(() => expect(requests.some((request) => request.url.endsWith("/api/datasets/dataset-1/reanalyze") && request.init?.method === "POST")).toBe(true));
    const created = requests.find((request) => request.url.endsWith("/api/datasets/dataset-1/reanalyze") && request.init?.method === "POST");
    expect(JSON.parse(String(created?.init?.body))).toEqual({ resource_request: { gpus: 0 } });
  });
});
