import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  Code2,
  Database,
  FileCheck2,
  FileJson,
  File as FileIcon,
  Filter,
  Folder,
  HardDrive,
  Image as ImageIcon,
  AudioLines,
  Video,
  Shapes,
  Layers3,
  MessageSquareText,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Trash2,
  Tags,
  Workflow,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { api, formatDate, formatNumber, patch, post } from "../api";
import { RouteLoading } from "../components/RouteLoading";
import { TaskDetail } from "../components/TaskDetail";
import { PostTrainingDataPanel } from "../components/PostTrainingDataPanel";
import {
  Alert,
  EmptyState,
  ErrorMessage,
  Field,
  FormActions,
  Modal,
  PageHeader,
  Panel,
  Progress,
  StatusBadge,
} from "../components/ui";
import type {
  DataConversionOperatorCatalog,
  DataConversionRecipe,
  DataConversionRun,
  DataProcessingRecipe,
  DataProcessingRun,
  Dataset,
  DatasetPreflight,
  DatasetViewerResult,
  LabelScheme,
  Mixture,
  Task,
} from "../types";

const sources = [
  ["huggingface", "Hugging Face", "填写 Hub 数据集 ID，由 Worker 下载匹配文件"],
  ["http", "HTTP / HTTPS", "下载一个直接文件地址，不抓取网页链接"],
  ["local", "本地文件", "复制原始文件或目录到托管制品库；PDF、DOCX、EPUB 需后续显式转换"],
  ["git", "Git 仓库", "克隆仓库并冻结实际内容"],
] as const;

const modalities = [
  ["text", "文本", "文档、表格、JSONL、Parquet", FileJson],
  ["image", "图片", "原图与标注文件", ImageIcon],
  ["audio", "音频", "音频文件与转写标注", AudioLines],
  ["video", "视频", "视频文件与时序标注", Video],
  ["multimodal", "多模态", "多种媒体及配对关系", Layers3],
  ["other", "其他", "保留原始内容，稍后接解析器", Shapes],
] as const;

const modalityLabels: Record<string, string> = Object.fromEntries(
  modalities.map(([key, label]) => [key, label]),
);

const HF_ENDPOINTS = {
  deployment: undefined,
  official: "https://huggingface.co",
  mirror: "https://hf-mirror.com",
} as const;

const TERMINAL_STATES = new Set(["succeeded", "failed", "cancelled", "lost"]);
const PREPARATION_TASK_KINDS = new Set(["data_tokenize_corpus", "training_data_index", "data_tokenize"]);
const RETRYABLE_DOWNLOAD_STATES = new Set(["failed", "lost"]);
type DataCenterTab = "datasets" | "explorer" | "labels" | "conversion" | "processing" | "mixtures" | "post-training" | "prepared";

function initialDataCenterTab(value: string | null): DataCenterTab {
  const supported = new Set<DataCenterTab>(["datasets", "explorer", "labels", "conversion", "processing", "mixtures", "post-training", "prepared"]);
  return value && supported.has(value as DataCenterTab) ? value as DataCenterTab : "datasets";
}

function transferTool(sourceType: string, endpoint?: string) {
  if (sourceType === "huggingface") {
    return "huggingface_hub.snapshot_download";
  }
  if (["http", "https"].includes(sourceType)) return "curl";
  if (sourceType === "git") return "git clone";
  return "本地文件复制";
}

function latestTransferProgress(task?: Task) {
  return task?.events?.slice().reverse().find((event) => event.event_type === "progress")?.payload;
}

function transferDetail(progress?: Record<string, any>) {
  if (!progress) return "";
  const parts = [progress.tool];
  const processedBytes = progress.downloaded_bytes ?? progress.processed_bytes;
  if (processedBytes != null) {
    parts.push(progress.total_bytes ? `${formatBytes(processedBytes)} / ${formatBytes(progress.total_bytes)}` : formatBytes(processedBytes));
  }
  if (progress.speed_bps != null) parts.push(`${formatBytes(progress.speed_bps)}/s`);
  if (progress.file_count_total != null) parts.push(`${formatNumber(progress.file_count_completed)} / ${formatNumber(progress.file_count_total)} 文件`);
  if (progress.processed_rows != null) parts.push(`${formatNumber(progress.processed_rows)} / ${formatNumber(progress.total_rows)} 行`);
  if (progress.documents_per_second != null) parts.push(`${formatNumber(progress.documents_per_second)} 文档/s`);
  if (progress.eta) parts.push(`ETA ${progress.eta}`);
  return parts.filter(Boolean).join(" · ");
}

function downloadErrorDetail(error?: Record<string, any>) {
  const message = String(error?.message || "").trim();
  if (message) return message;
  if (error?.type === "ConnectTimeout") return "连接数据源超时；请检查 Worker 网络后重新下载。";
  if (["ReadError", "RemoteProtocolError"].includes(String(error?.type || ""))) return "下载连接在文件传输完成前中断；重新下载会从已有数据续传。";
  if (error?.type) return `Worker 返回 ${error.type}，请检查任务日志后重新下载。`;
  return "请检查 Worker 状态或任务日志后重新下载";
}

function useRetryDownload() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (taskId: string) => post<Task>(`/tasks/${taskId}/control`, {
      action: "retry",
      reason: "用户从数据中心重新下载",
    }),
    onSuccess: (_task, taskId) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["dataset"] });
      queryClient.invalidateQueries({ queryKey: ["task", taskId] });
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
}

export function DataCenter({
  projectIdOverride,
  globalMode = false,
  workspaceUri = "",
}: {
  projectIdOverride?: string;
  globalMode?: boolean;
  workspaceUri?: string;
} = {}) {
  const { projectId: routeProjectId = "" } = useParams();
  const projectId = projectIdOverride || routeProjectId;
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const initialTab = searchParams.get("tab");
  const [tab, setTab] = useState<DataCenterTab>(() => initialDataCenterTab(initialTab));
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(searchParams.get("dataset"));
  const [wizard, setWizard] = useState(false);
  const [mixtureOpen, setMixtureOpen] = useState(false);
  const [recipeOpen, setRecipeOpen] = useState<DataProcessingRecipe | true | null>(null);
  const [processingRunOpen, setProcessingRunOpen] = useState(false);
  const [conversionRecipeOpen, setConversionRecipeOpen] = useState<DataConversionRecipe | true | null>(null);
  const [conversionRunOpen, setConversionRunOpen] = useState(false);
  const [selectedMixtureId, setSelectedMixtureId] = useState<string | null>(null);
  const [filterOpen, setFilterOpen] = useState(false);
  const [columnsOpen, setColumnsOpen] = useState(false);
  const [selectedPreparedTaskId, setSelectedPreparedTaskId] = useState("");
  const [filters, setFilters] = useState({ language: "", license: "", readiness: "" });
  const [visibleColumns, setVisibleColumns] = useState({ stage: true, documents: true, tokens: true, quality: true, license: true, status: true });
  const retryDownload = useRetryDownload();
  const datasets = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api<Dataset[]>(globalMode ? "/catalog/datasets" : `/projects/${projectId}/datasets`),
    refetchInterval: (query) => (query.state.data || []).some((item) => !TERMINAL_STATES.has(item.status) && item.status !== "ready" && item.status !== "needs_review" && item.status !== "registered") ? 1500 : false,
  });
  const mixtures = useQuery({ queryKey: ["mixtures", projectId, globalMode], queryFn: () => api<Mixture[]>(globalMode ? "/catalog/mixtures" : `/projects/${projectId}/mixtures`), refetchInterval: (query) => (query.state.data || []).some((item) => ["resolving", "queued", "assigned", "starting", "running"].includes(String(item.current_version?.status || ""))) ? 1500 : false });
  const preparedCorpora = useQuery({ queryKey: ["tokenized-corpora", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/tokenized-corpora`), enabled: tab === "prepared", refetchInterval: 2000 });
  const preparedIndexes = useQuery({ queryKey: ["training-data-views", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/training-data-views`), enabled: tab === "prepared", refetchInterval: 2000 });
  const preparationTasks = useQuery({ queryKey: ["preparation-tasks", projectId], queryFn: () => api<Task[]>(`/tasks?project_id=${projectId}&limit=100`), enabled: tab === "prepared", refetchInterval: 2000 });
  const preparedTaskRecords = (preparationTasks.data || []).filter((task) => PREPARATION_TASK_KINDS.has(task.kind));
  const recipes = useQuery({ queryKey: ["data-processing-recipes", projectId], queryFn: () => api<DataProcessingRecipe[]>(`/projects/${projectId}/data-processing-recipes`), enabled: tab === "processing" });
  const processingRuns = useQuery({ queryKey: ["data-processing-runs", projectId], queryFn: () => api<DataProcessingRun[]>(`/projects/${projectId}/data-processing-runs`), enabled: tab === "processing", refetchInterval: (query) => (query.state.data || []).some((item) => !TERMINAL_STATES.has(item.status)) ? 1500 : false });
  const conversionRecipes = useQuery({ queryKey: ["data-conversion-recipes", projectId], queryFn: () => api<DataConversionRecipe[]>(`/projects/${projectId}/data-conversion-recipes`), enabled: tab === "conversion" });
  const conversionRuns = useQuery({ queryKey: ["data-conversion-runs", projectId], queryFn: () => api<DataConversionRun[]>(`/projects/${projectId}/data-conversion-runs`), enabled: tab === "conversion", refetchInterval: (query) => (query.state.data || []).some((item) => !TERMINAL_STATES.has(item.status)) ? 1500 : false });
  const conversionCatalog = useQuery({ queryKey: ["data-conversion-operators"], queryFn: () => api<DataConversionOperatorCatalog>("/data-conversion/operators"), enabled: tab === "conversion" });
  useEffect(() => {
    if ((conversionRuns.data || []).some((run) => TERMINAL_STATES.has(run.status))) {
      queryClient.invalidateQueries({ queryKey: ["datasets", projectId] });
    }
  }, [conversionRuns.dataUpdatedAt, projectId, queryClient]);
  const filtered = (datasets.data || []).filter((item) => {
    const version = item.current_version;
    return `${item.name} ${item.language} ${item.license}`.toLowerCase().includes(search.toLowerCase())
      && (!filters.language || item.language === filters.language)
      && (!filters.license || item.license === filters.license)
      && (!filters.readiness || (filters.readiness === "ready") === Boolean(version?.train_ready));
  });
  const activeFilterCount = Object.values(filters).filter(Boolean).length;

  function selectTab(nextTab: DataCenterTab) {
    setTab(nextTab);
    setSelectedId(null);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("dataset");
    nextParams.delete("version");
    const defaultTab = "datasets";
    if (nextTab === defaultTab) nextParams.delete("tab");
    else nextParams.set("tab", nextTab);
    setSearchParams(nextParams, { replace: true });
  }

  function openDataset(datasetId: string) {
    setSelectedId(datasetId);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("tab");
    nextParams.set("dataset", datasetId);
    setSearchParams(nextParams, { replace: true });
  }

  function closeDataset() {
    setSelectedId(null);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("dataset");
    nextParams.delete("action");
    setSearchParams(nextParams, { replace: true });
  }

  function exploreDatasetVersion(versionId: string) {
    setSelectedId(null);
    setTab("explorer");
    const nextParams = new URLSearchParams();
    nextParams.set("tab", "explorer");
    nextParams.set("version", versionId);
    setSearchParams(nextParams, { replace: true });
  }

  const pageMeta: Record<DataCenterTab, { eyebrow: string; title: string; description: string }> = {
    datasets: { eyebrow: "DATASETS", title: "数据集", description: "发现、导入并管理系统内全部不可变数据集版本。" },
    explorer: { eyebrow: "DATA STUDIO", title: "Data Studio", description: "像 Hugging Face 一样用二维表格浏览原始字段，并使用只读 SQL 查询真实数据。" },
    labels: { eyebrow: "LABEL SYSTEM", title: "标签管理", description: "按数据模态管理版本化标签体系、单选组和多选组。" },
    conversion: { eyebrow: "DATA WORKFLOW", title: "文档转换", description: "将 PDF、DOCX、EPUB 等原始文件显式转换为文本数据版本。" },
    processing: { eyebrow: "DATA WORKFLOW", title: "清洗与质量加工", description: "使用版本化配方生成 Silver 或 Gold 数据版本。" },
    mixtures: { eyebrow: "PRETRAIN RECIPES", title: "预训练混合集", description: "组合不可变数据版本、子集条件与采样权重。" },
    "post-training": { eyebrow: "POST-TRAINING DATA", title: "后训练数据", description: "管理 SFT、偏好数据映射和后训练混合。" },
    prepared: { eyebrow: "PROJECT ARTIFACTS", title: "已准备数据", description: "查看项目生成的 Tokenized Corpus 与训练读取索引。" },
  };
  const meta = pageMeta[tab];

  if (selectedId) {
    return <DatasetDetailPage datasetId={selectedId} initialDataset={(datasets.data || []).find((item) => item.id === selectedId)} initialAction={searchParams.get("action")} onClose={closeDataset} onExplore={exploreDatasetVersion} />;
  }

  return <>
    <header className="data-command-header">
      <div><p className="eyebrow">{meta.eyebrow}</p><h1>{meta.title}</h1><p>{meta.description}</p>{workspaceUri && <span className="data-workspace-path"><HardDrive size={13} />{workspaceUri}</span>}</div>
      <div className="data-command-actions">
        {(tab === "datasets" || tab === "explorer") && <button className="button primary" aria-label="新数据 · 导入数据" onClick={() => setWizard(true)}><Plus size={17} />导入数据</button>}
        {tab === "conversion" && <button className="button primary" onClick={() => setConversionRecipeOpen(true)}><Plus size={17} />新建转换配方</button>}
        {tab === "processing" && <button className="button primary" onClick={() => setRecipeOpen(true)}><Plus size={17} />新建清洗配方</button>}
        {tab === "mixtures" && <button className="button primary" onClick={() => setMixtureOpen(true)}><Plus size={17} />新建混合</button>}
      </div>
    </header>
    <div className="data-workbench">
      <nav className="data-workbench-nav" aria-label="数据资产导航">
        <button className={tab === "datasets" ? "active" : ""} onClick={() => selectTab("datasets")}><Database size={17} /><strong>数据集目录</strong><b>{datasets.data?.length || 0}</b></button>
        {globalMode && <button className={tab === "explorer" ? "active" : ""} onClick={() => selectTab("explorer")}><Search size={17} /><strong>Data Studio</strong></button>}
        {globalMode && <button className={tab === "labels" ? "active" : ""} onClick={() => selectTab("labels")}><Tags size={17} /><strong>标签管理</strong></button>}
        <button className={tab === "conversion" ? "active" : ""} onClick={() => selectTab("conversion")}><FileJson size={17} /><strong>数据转换</strong><b>{conversionRuns.data?.length || 0}</b></button>
        <button className={tab === "processing" ? "active" : ""} onClick={() => selectTab("processing")}><Workflow size={17} /><strong>数据清洗</strong><b>{processingRuns.data?.length || 0}</b></button>
        <button aria-label={`数据混合 ${mixtures.data?.length || 0}`} className={tab === "mixtures" ? "active" : ""} onClick={() => selectTab("mixtures")}><Layers3 size={17} /><strong>预训练混合</strong><b>{mixtures.data?.length || 0}</b></button>
        <button className={tab === "post-training" ? "active" : ""} onClick={() => selectTab("post-training")}><MessageSquareText size={17} /><strong>后训练数据</strong></button>
        {!globalMode && <button className={tab === "prepared" ? "active" : ""} onClick={() => selectTab("prepared")}><HardDrive size={17} /><strong>已准备数据</strong><b>{(preparedCorpora.data?.length || 0) + (preparedIndexes.data?.length || 0) + preparedTaskRecords.length}</b></button>}
      </nav>
      <section className="data-workbench-content">
        {tab === "post-training" ? <PostTrainingDataPanel projectId={projectId} datasets={datasets.data || []} globalMode={globalMode} /> : tab === "labels" ? <LabelManagementPanel /> : tab === "explorer" ? <DatasetExplorerWorkspace datasets={datasets.data || []} initialVersionId={searchParams.get("version")} onMixtureCreated={() => { queryClient.invalidateQueries({ queryKey: ["mixtures", projectId] }); selectTab("mixtures"); }} /> : tab === "datasets" ? <>
          <div className="catalog-toolbar">
            <div className="search-box catalog-search"><Search size={16} /><input aria-label="搜索数据集" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称、语言、许可证或来源" /></div>
            <button className={`button ghost ${filterOpen ? "active" : ""}`} onClick={() => setFilterOpen(!filterOpen)}><Filter size={16} />筛选{activeFilterCount ? ` ${activeFilterCount}` : ""}</button>
            <button className={`button ghost ${columnsOpen ? "active" : ""}`} onClick={() => setColumnsOpen(!columnsOpen)}><SlidersHorizontal size={16} />显示列</button>
          </div>
          {filterOpen && <div className="catalog-filter-strip"><Field label="语言"><select value={filters.language} onChange={(event) => setFilters({ ...filters, language: event.target.value })}><option value="">全部语言</option>{Array.from(new Set((datasets.data || []).map((item) => item.language).filter(Boolean))).map((value) => <option key={value} value={value}>{value}</option>)}</select></Field><Field label="许可证"><select value={filters.license} onChange={(event) => setFilters({ ...filters, license: event.target.value })}><option value="">全部许可证</option>{Array.from(new Set((datasets.data || []).map((item) => item.license).filter(Boolean))).map((value) => <option key={value} value={value}>{value}</option>)}</select></Field><Field label="训练状态"><select value={filters.readiness} onChange={(event) => setFilters({ ...filters, readiness: event.target.value })}><option value="">全部状态</option><option value="ready">可训练</option><option value="blocked">不可训练</option></select></Field><button className="button ghost small" onClick={() => setFilters({ language: "", license: "", readiness: "" })}>重置</button></div>}
          {columnsOpen && <div className="catalog-column-strip"><span>显示字段</span>{Object.entries({ stage: "处理阶段", documents: "文档数", tokens: "Token 估算", quality: "质量", license: "许可证", status: "状态" }).map(([key, label]) => <label className="check-field" key={key}><input type="checkbox" checked={visibleColumns[key as keyof typeof visibleColumns]} onChange={(event) => setVisibleColumns({ ...visibleColumns, [key]: event.target.checked })} /><span>{label}</span></label>)}</div>}
          <Panel className="table-panel catalog-table-panel">
            <div className="catalog-table-caption"><span>共 {filtered.length} 个数据集</span><small>点击任意行查看版本、样本、质量和来源</small></div>
            {filtered.length ? <div className="data-table-wrap"><table className="data-table catalog-table"><thead><tr><th>数据集</th><th>当前版本</th>{visibleColumns.stage && <th>阶段</th>}{visibleColumns.documents && <th>文档数</th>}{visibleColumns.tokens && <th>Token 估算</th>}{visibleColumns.quality && <th>质量</th>}{visibleColumns.license && <th>许可证</th>}{visibleColumns.status && <th>状态</th>}<th aria-label="打开详情" /></tr></thead><tbody>{filtered.map((dataset) => {
              const version = dataset.current_version || {};
              const status = version.train_ready ? "ready" : dataset.status;
              const canRetryDownload = Boolean(version.task_id && RETRYABLE_DOWNLOAD_STATES.has(status));
              const retrying = retryDownload.isPending && retryDownload.variables === version.task_id;
              return <tr key={dataset.id} onClick={() => openDataset(dataset.id)}><td><div className="dataset-name-cell"><span className="dataset-source-mark">{dataset.name.slice(0, 2).toUpperCase()}</span><span><strong>{dataset.name}</strong><small>{modalityLabels[dataset.modality || "text"]} · {dataset.source_type} · {dataset.language}</small>{dataset.tags?.length ? <span className="dataset-inline-tags">{dataset.tags.slice(0, 3).map((tag) => <i key={tag}>{tag}</i>)}</span> : null}</span></div></td><td><b>{String(version.tier || "—").toUpperCase()} V{version.version || "—"}</b></td>{visibleColumns.stage && <td>{version.tier || "—"}</td>}{visibleColumns.documents && <td>{formatNumber(version.document_count)}</td>}{visibleColumns.tokens && <td>{formatNumber(version.token_estimate)}</td>}{visibleColumns.quality && <td><div className="score-cell"><b>{version.automatic_quality_score ?? "—"}</b><span>自动</span><b>{version.human_score ?? "—"}</b><span>人工</span></div></td>}{visibleColumns.license && <td>{dataset.license}</td>}{visibleColumns.status && <td><div className="dataset-status-cell"><StatusBadge status={status} />{canRetryDownload && <button type="button" className="button secondary small" disabled={retrying} onClick={(event) => { event.stopPropagation(); retryDownload.mutate(String(version.task_id)); }}><RotateCcw size={13} />{retrying ? "启动中…" : "重新下载"}</button>}</div></td>}<td><ArrowRight size={16} /></td></tr>;
            })}</tbody></table></div> : <EmptyState title="没有匹配的数据集" description={datasets.data?.length ? "调整搜索词或清除筛选条件。" : "从 Hugging Face、本地文件、HTTP 或 Git 导入第一份真实数据。"} action={datasets.data?.length ? <button className="button secondary" onClick={() => { setSearch(""); setFilters({ language: "", license: "", readiness: "" }); }}>清除筛选</button> : <button className="button primary" onClick={() => setWizard(true)}><Plus size={16} />导入数据</button>} />}
          </Panel>
          <ErrorMessage error={retryDownload.error} />
        </> : tab === "conversion" ? <DataConversionPanel recipes={conversionRecipes.data || []} runs={conversionRuns.data || []} loading={conversionRecipes.isLoading || conversionRuns.isLoading || conversionCatalog.isLoading} onCreateRecipe={() => setConversionRecipeOpen(true)} onVersionRecipe={(recipe) => setConversionRecipeOpen(recipe)} onCreateRun={() => setConversionRunOpen(true)} /> : tab === "processing" ? <DataProcessingPanel recipes={recipes.data || []} runs={processingRuns.data || []} loading={recipes.isLoading || processingRuns.isLoading} onCreateRecipe={() => setRecipeOpen(true)} onVersionRecipe={(recipe) => setRecipeOpen(recipe)} onCreateRun={() => setProcessingRunOpen(true)} /> : tab === "mixtures" ? <MixtureList mixtures={mixtures.data || []} onCreate={() => setMixtureOpen(true)} onSelect={setSelectedMixtureId} /> : <PreparedDataPanel corpora={preparedCorpora.data || []} indexes={preparedIndexes.data || []} tasks={preparedTaskRecords} onOpenTask={setSelectedPreparedTaskId} />}
      </section>
    </div>
    {wizard && <DatasetWizard projectId={projectId} globalMode={globalMode} onClose={() => setWizard(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["datasets", projectId] }); setWizard(false); }} />}
    {mixtureOpen && <MixtureModal projectId={projectId} datasets={datasets.data || []} globalMode={globalMode} onClose={() => setMixtureOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["mixtures", projectId] }); setMixtureOpen(false); }} />}
    {recipeOpen && <DataProcessingRecipeModal projectId={projectId} recipe={recipeOpen === true ? undefined : recipeOpen} onClose={() => setRecipeOpen(null)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["data-processing-recipes", projectId] }); setRecipeOpen(null); }} />}
    {processingRunOpen && <DataProcessingRunModal projectId={projectId} datasets={datasets.data || []} recipes={recipes.data || []} onClose={() => setProcessingRunOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["data-processing-runs", projectId] }); queryClient.invalidateQueries({ queryKey: ["datasets", projectId] }); setProcessingRunOpen(false); }} />}
    {conversionRecipeOpen && conversionCatalog.data && <DataConversionRecipeModal projectId={projectId} recipe={conversionRecipeOpen === true ? undefined : conversionRecipeOpen} catalog={conversionCatalog.data} onClose={() => setConversionRecipeOpen(null)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["data-conversion-recipes", projectId] }); setConversionRecipeOpen(null); }} />}
    {conversionRunOpen && <DataConversionRunModal projectId={projectId} datasets={datasets.data || []} recipes={conversionRecipes.data || []} onClose={() => setConversionRunOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["data-conversion-runs", projectId] }); queryClient.invalidateQueries({ queryKey: ["datasets", projectId] }); setConversionRunOpen(false); }} />}
    {selectedMixtureId && <MixtureDrawer projectId={projectId} mixture={(mixtures.data || []).find((item) => item.id === selectedMixtureId)} datasets={datasets.data || []} globalMode={globalMode} onClose={() => setSelectedMixtureId(null)} onChanged={() => queryClient.invalidateQueries({ queryKey: ["mixtures", projectId] })} />}
    {selectedPreparedTaskId && <TaskDetail taskId={selectedPreparedTaskId} onClose={() => setSelectedPreparedTaskId("")} />}
  </>;
}

type LabelDraft = { code: string; name: string; description: string; color: string };
type LabelGroupDraft = { code: string; name: string; description: string; selection_mode: "single" | "multiple"; required: boolean; labels: LabelDraft[] };

function emptyLabelGroup(): LabelGroupDraft {
  return { code: "category", name: "分类", description: "", selection_mode: "single", required: false, labels: [{ code: "label", name: "标签", description: "", color: "" }] };
}

function LabelManagementPanel() {
  const queryClient = useQueryClient();
  const [modality, setModality] = useState("");
  const [editing, setEditing] = useState<LabelScheme | true | null>(null);
  const schemes = useQuery({ queryKey: ["label-schemes", modality, "all"], queryFn: () => api<LabelScheme[]>(`/catalog/label-schemes?include_archived=true${modality ? `&modality=${modality}` : ""}`) });
  const archive = useMutation({ mutationFn: ({ id, status }: { id: string; status: "active" | "archived" }) => patch<LabelScheme>(`/catalog/label-schemes/${id}`, { status }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["label-schemes"] }) });
  return <section className="label-management">
    <div className="catalog-toolbar label-management-toolbar">
      <Field label="数据模态"><select aria-label="筛选标签模态" value={modality} onChange={(event) => setModality(event.target.value)}><option value="">全部模态</option>{modalities.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></Field>
      <button className="button primary" onClick={() => setEditing(true)}><Plus size={16} />新建标签体系</button>
    </div>
    <ErrorMessage error={schemes.error || archive.error} />
    {schemes.data?.length ? <div className="label-scheme-grid">{schemes.data.map((scheme) => <Panel key={scheme.id} className="label-scheme-card">
      <header><div><span>{modalityLabels[scheme.modality] || scheme.modality}</span><h3>{scheme.name}</h3><code>{scheme.code}</code></div><StatusBadge status={scheme.status} /></header>
      <p>{scheme.description || "暂无说明"}</p>
      <div className="label-scheme-groups">{scheme.current_version?.groups.map((group) => <div key={group.id}><strong>{group.name}</strong><small>{group.selection_mode === "single" ? "单选" : "多选"}</small><div>{group.labels.map((label) => <span key={label.id} style={label.color ? { borderColor: label.color } : undefined}>{label.name}</span>)}</div></div>)}</div>
      <footer><span>当前 V{scheme.current_version?.version || "—"} · 共 {scheme.version_count} 个版本</span><div>{scheme.status === "active" && <button className="button secondary small" onClick={() => setEditing(scheme)}>创建新版本</button>}<button className="button ghost small" disabled={archive.isPending} onClick={() => archive.mutate({ id: scheme.id, status: scheme.status === "active" ? "archived" : "active" })}>{scheme.status === "active" ? "归档" : "恢复"}</button></div></footer>
    </Panel>)}</div> : !schemes.isLoading && <EmptyState title="还没有标签体系" description="创建一个与数据模态绑定的标签体系，再到 Data Studio 对查询结果打标。" action={<button className="button primary" onClick={() => setEditing(true)}><Plus size={16} />新建标签体系</button>} />}
    {editing && <LabelSchemeEditorModal scheme={editing === true ? undefined : editing} onClose={() => setEditing(null)} onSaved={() => { queryClient.invalidateQueries({ queryKey: ["label-schemes"] }); setEditing(null); }} />}
  </section>;
}

function LabelSchemeEditorModal({ scheme, onClose, onSaved }: { scheme?: LabelScheme; onClose: () => void; onSaved: () => void }) {
  const current = scheme?.current_version;
  const [name, setName] = useState(scheme?.name || "");
  const [code, setCode] = useState(scheme?.code || "");
  const [modality, setModality] = useState(scheme?.modality || "text");
  const [description, setDescription] = useState(scheme?.description || current?.description || "");
  const [createdBy, setCreatedBy] = useState("local-user");
  const [groups, setGroups] = useState<LabelGroupDraft[]>(() => current?.groups.map((group) => ({ code: group.code, name: group.name, description: group.description, selection_mode: group.selection_mode, required: group.required, labels: group.labels.map((label) => ({ code: label.code, name: label.name, description: label.description, color: label.color })) })) || [emptyLabelGroup()]);
  const save = useMutation({ mutationFn: () => {
    const body = { description, created_by: createdBy, groups };
    return scheme ? post<LabelScheme>(`/catalog/label-schemes/${scheme.id}/versions`, body) : post<LabelScheme>("/catalog/label-schemes", { ...body, name, code, modality });
  }, onSuccess: onSaved });
  function updateGroup(index: number, values: Partial<LabelGroupDraft>) { setGroups(groups.map((group, position) => position === index ? { ...group, ...values } : group)); }
  function updateLabel(groupIndex: number, labelIndex: number, values: Partial<LabelDraft>) { updateGroup(groupIndex, { labels: groups[groupIndex].labels.map((label, position) => position === labelIndex ? { ...label, ...values } : label) }); }
  const valid = Boolean((scheme || (name.trim() && code.trim())) && groups.length && groups.every((group) => group.name.trim() && group.code.trim() && group.labels.length && group.labels.every((label) => label.name.trim() && label.code.trim())));
  return <Modal title={scheme ? `创建 ${scheme.name} 新版本` : "新建标签体系"} description="版本发布后结构不可修改；以后调整标签时创建新版本，历史打标仍引用原版本。" onClose={onClose} wide><form className="form-grid" onSubmit={(event) => { event.preventDefault(); save.mutate(); }}>
    {!scheme && <div className="form-grid two-columns"><Field label="体系名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="体系编码" hint="小写字母开头，可含数字、_、-"><input required value={code} onChange={(event) => setCode(event.target.value)} /></Field><Field label="数据模态"><select value={modality} onChange={(event) => setModality(event.target.value)}>{modalities.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></Field><Field label="创建人"><input value={createdBy} onChange={(event) => setCreatedBy(event.target.value)} /></Field></div>}
    {scheme && <Field label="创建人"><input value={createdBy} onChange={(event) => setCreatedBy(event.target.value)} /></Field>}
    <Field label="版本说明"><textarea value={description} onChange={(event) => setDescription(event.target.value)} /></Field>
    <div className="label-editor-groups">{groups.map((group, groupIndex) => <section key={groupIndex} className="label-editor-group"><header><strong>标签组 {groupIndex + 1}</strong><button type="button" className="button ghost small" disabled={groups.length === 1} onClick={() => setGroups(groups.filter((_, index) => index !== groupIndex))}>删除组</button></header><div className="form-grid two-columns"><Field label="组名称"><input value={group.name} onChange={(event) => updateGroup(groupIndex, { name: event.target.value })} /></Field><Field label="组编码"><input value={group.code} onChange={(event) => updateGroup(groupIndex, { code: event.target.value })} /></Field><Field label="选择方式"><select value={group.selection_mode} onChange={(event) => updateGroup(groupIndex, { selection_mode: event.target.value as "single" | "multiple" })}><option value="single">单选</option><option value="multiple">多选</option></select></Field><label className="check-field"><input type="checkbox" checked={group.required} onChange={(event) => updateGroup(groupIndex, { required: event.target.checked })} /><span>标记为必填组</span></label></div><div className="label-editor-labels">{group.labels.map((label, labelIndex) => <div key={labelIndex}><input aria-label={`标签 ${labelIndex + 1} 名称`} placeholder="标签名称" value={label.name} onChange={(event) => updateLabel(groupIndex, labelIndex, { name: event.target.value })} /><input aria-label={`标签 ${labelIndex + 1} 编码`} placeholder="label_code" value={label.code} onChange={(event) => updateLabel(groupIndex, labelIndex, { code: event.target.value })} /><input aria-label={`标签 ${labelIndex + 1} 颜色`} placeholder="#4f46e5（可选）" value={label.color} onChange={(event) => updateLabel(groupIndex, labelIndex, { color: event.target.value })} /><button type="button" className="button ghost small" disabled={group.labels.length === 1} onClick={() => updateGroup(groupIndex, { labels: group.labels.filter((_, index) => index !== labelIndex) })}>删除</button></div>)}</div><button type="button" className="button secondary small" onClick={() => updateGroup(groupIndex, { labels: [...group.labels, { code: `label_${group.labels.length + 1}`, name: `标签 ${group.labels.length + 1}`, description: "", color: "" }] })}><Plus size={13} />添加标签</button></section>)}</div>
    <button type="button" className="button secondary" onClick={() => setGroups([...groups, { ...emptyLabelGroup(), code: `group_${groups.length + 1}`, name: `标签组 ${groups.length + 1}` }])}><Plus size={14} />添加标签组</button>
    <ErrorMessage error={save.error} /><FormActions submitting={save.isPending} disabled={!valid} onCancel={onClose} submitLabel={scheme ? "发布新版本" : "创建并发布 V1"} />
  </form></Modal>;
}

function DatasetExplorerWorkspace({ datasets, initialVersionId, onMixtureCreated }: { datasets: Dataset[]; initialVersionId?: string | null; onMixtureCreated: () => void }) {
  const [mode, setMode] = useState<"studio" | "manifest">("studio");
  const candidates = useMemo(
    () => datasets.flatMap((dataset) => (dataset.versions || [])
      .filter((version: any) => version.state === "ready" && version.artifact_id)
      .map((version: any) => ({ dataset, version }))),
    [datasets],
  );
  const initialCandidate = candidates.find(({ version }) => version.id === initialVersionId) || candidates[0];
  const [versionId, setVersionId] = useState(() => String(initialCandidate?.version.id || ""));
  const selected = candidates.find(({ version }) => version.id === versionId) || initialCandidate;
  useEffect(() => {
    if (!versionId && initialCandidate?.version.id) setVersionId(String(initialCandidate.version.id));
  }, [initialCandidate?.version.id, versionId]);
  return <section className="dataset-studio-workspace">
    <div className="dataset-studio-mode" role="tablist" aria-label="数据检索模式">
      <button role="tab" aria-selected={mode === "studio"} className={mode === "studio" ? "active" : ""} onClick={() => setMode("studio")}><Database size={15} />Data Studio</button>
      <button role="tab" aria-selected={mode === "manifest"} className={mode === "manifest" ? "active" : ""} onClick={() => setMode("manifest")}><Filter size={15} />稳定索引筛选</button>
    </div>
    {mode === "studio" ? <>
      <div className="dataset-studio-source">
        <div><span>Dataset</span><strong>选择一个不可变版本，用二维表格浏览原始字段</strong></div>
        <select aria-label="选择 Data Studio 数据版本" value={selected?.version.id || ""} onChange={(event) => setVersionId(event.target.value)}>
          {candidates.map(({ dataset, version }) => <option key={version.id} value={version.id}>{dataset.name} · V{version.version} · {String(version.tier).toUpperCase()}</option>)}
        </select>
      </div>
      {selected ? <DatasetTableViewer key={selected.version.id} version={selected.version} datasetName={selected.dataset.name} modality={selected.dataset.modality || selected.version.modality || "text"} /> : <EmptyState title="没有可浏览的数据版本" description="导入 Parquet、JSON、CSV 或 TSV 数据并等待快照就绪后，即可在这里查询。" />}
    </> : <RecordManifestExplorerWorkspace datasets={datasets} initialVersionId={initialVersionId} onMixtureCreated={onMixtureCreated} />}
  </section>;
}

function viewerCellText(value: any): string {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

export function DatasetTableViewer({ version, datasetName, modality = "text", queryEndpoint, enableLabels = true }: { version: any; datasetName?: string; modality?: string; queryEndpoint?: string; enableLabels?: boolean }) {
  const [sql, setSql] = useState("SELECT * FROM dataset");
  const [search, setSearch] = useState("");
  const [configuration, setConfiguration] = useState("default");
  const [sourceTaskId, setSourceTaskId] = useState("");
  const [jumpRow, setJumpRow] = useState("1");
  const [split, setSplit] = useState("");
  const [taskId, setTaskId] = useState("");
  const [expanded, setExpanded] = useState<{ column: string; value: any } | null>(null);
  const [autoStarted, setAutoStarted] = useState(false);
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [labelingOpen, setLabelingOpen] = useState(false);
  const [labelTaskId, setLabelTaskId] = useState("");
  const startQuery = useMutation({
    mutationFn: ({ offset, selectedSplit, querySql, querySearch = search, selectedConfiguration = configuration }: { offset: number; selectedSplit?: string; querySql: string; querySearch?: string; selectedConfiguration?: string }) => post<Task>(queryEndpoint || `/catalog/dataset-versions/${version.id}/viewer/query`, {
      sql: querySql,
      search: querySearch,
      configuration: selectedConfiguration,
      ...(queryEndpoint && sourceTaskId ? { source_task_id: sourceTaskId } : {}),
      split: selectedSplit ?? split,
      offset,
      limit: 100,
      timeout_seconds: 30,
    }),
    onSuccess: (task) => setTaskId(task.id),
  });
  const task = useQuery({
    queryKey: ["dataset-viewer-task", taskId],
    queryFn: () => api<Task>(`/tasks/${taskId}?include_events=false`),
    enabled: Boolean(taskId),
    refetchInterval: (query) => TERMINAL_STATES.has(query.state.data?.observed_state || "") ? false : 350,
  });
  const result = task.data?.observed_state === "succeeded" ? task.data.result as DatasetViewerResult : undefined;
  const labelSchemes = useQuery({ queryKey: ["label-schemes", modality], queryFn: () => api<LabelScheme[]>(`/catalog/label-schemes?modality=${modality}`), enabled: enableLabels && Boolean(result?.labelable) });
  const labelTask = useQuery({ queryKey: ["data-label-task", labelTaskId], queryFn: () => api<Task>(`/tasks/${labelTaskId}?include_events=false`), enabled: Boolean(labelTaskId), refetchInterval: (query) => TERMINAL_STATES.has(query.state.data?.observed_state || "") ? false : 500 });
  const pending = startQuery.isPending || Boolean(taskId && (!task.data || !TERMINAL_STATES.has(task.data.observed_state)));
  const activeSplit = split || result?.split || "";
  function runQuery(offset = 0, selectedSplit = activeSplit, querySql = sql, querySearch = search, selectedConfiguration = configuration) {
    if (pending) return;
    setTaskId("");
    setSelectedRows(new Set());
    startQuery.mutate({ offset, selectedSplit, querySql, querySearch, selectedConfiguration });
  }
  function browse(offset: number) {
    runQuery(offset, result?.split || activeSplit, result?.base_sql || result?.sql || sql, result?.search || "", result?.configuration || configuration);
  }
  useEffect(() => {
    if (result?.source_task_id) setSourceTaskId(result.source_task_id);
    if (result?.configuration) setConfiguration(result.configuration);
  }, [result]);
  useEffect(() => {
    if (!autoStarted && version?.id) {
      setAutoStarted(true);
      startQuery.mutate({ offset: 0, selectedSplit: "", querySql: "SELECT * FROM dataset" });
    }
  }, [autoStarted, version?.id]);
  useEffect(() => {
    if (labelTask.data?.observed_state === "succeeded") {
      setLabelTaskId("");
      browse(result?.offset || 0);
    }
  }, [labelTask.data?.observed_state]);
  const stopQuery = useMutation({ mutationFn: () => post<Task>(`/tasks/${taskId}/control`, { action: "cancel", reason: "用户停止数据查看器查询" }) });
  const queryError = startQuery.error || task.error;
  const taskFailure = task.data && ["failed", "lost"].includes(task.data.observed_state) ? task.data.error : null;
  return <section className="hf-dataset-viewer">
    <header className="hf-viewer-toolbar">
      <div className="hf-viewer-selectors">
        <label><span>Subset</span><select aria-label="数据子集" value={configuration} disabled={pending || (result?.configurations?.length || 0) < 2} onChange={(event) => { const value = event.target.value; setConfiguration(value); setSplit(""); runQuery(0, "", sql, search, value); }}>{(result?.configurations || [configuration]).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label><span>Split</span><select aria-label="数据划分" value={activeSplit} disabled={!result?.splits?.length || pending} onChange={(event) => { const value = event.target.value; setSplit(value); runQuery(0, value); }}>{(result?.splits || (activeSplit ? [activeSplit] : [])).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      </div>
      <div className="hf-viewer-dataset-meta"><strong>{datasetName || `V${version.version}`}</strong><span>{result ? `${formatNumber(result.total_rows)} rows · ${result.format.toUpperCase()} · ${result.elapsed_ms} ms` : "正在读取结构化快照"}</span></div>
    </header>
    <form className="dataset-viewer-search" onSubmit={(event) => { event.preventDefault(); runQuery(0); }}><input aria-label="搜索数据内容" maxLength={2000} placeholder="搜索全部记录，支持题目、答案及嵌套字段" value={search} onChange={(event) => setSearch(event.target.value)} /><button className="button secondary small" disabled={pending}><Search size={14} />搜索</button><button type="button" className="button ghost small" disabled={pending || !search} onClick={() => { setSearch(""); runQuery(0, activeSplit, sql, ""); }}>清除搜索</button></form>
    <div className="dataset-sql-panel">
      <div className="dataset-sql-heading"><div><Code2 size={16} /><span>SQL</span><small>只读 DuckDB · 表名为 <code>dataset</code></small></div><button className="button ghost small" onClick={() => { const defaultSql = "SELECT * FROM dataset"; setSql(defaultSql); runQuery(0, activeSplit, defaultSql); }}>重置</button></div>
      <textarea aria-label="Dataset SQL 查询" spellCheck={false} value={sql} onChange={(event) => setSql(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runQuery(0); }} />
      <div className="dataset-sql-actions"><span>Ctrl / ⌘ + Enter 运行 · 每页 100 行 · 搜索作用于全部查询结果</span>{pending ? <button className="button danger small" disabled={!taskId || stopQuery.isPending} onClick={() => stopQuery.mutate()}><Square size={13} />停止</button> : null}<button className="button primary small" disabled={pending || !sql.trim()} onClick={() => runQuery(0)}><Search size={14} />{pending ? "查询中…" : "运行 SQL"}</button></div>
    </div>
    <ErrorMessage error={queryError || stopQuery.error} />
    {taskFailure && <Alert severity="error" title="SQL 查询失败" detail={String(taskFailure.message || taskFailure.type || "Worker 未能完成查询")} />}
    {task.data?.observed_state === "cancelled" && <Alert severity="info" title="查询已停止" detail="可调整搜索条件后重新查询。" />}
    {pending && !result && <div className="dataset-viewer-loading"><RefreshCw size={20} /><strong>正在查询真实数据快照…</strong><span>{task.data?.current_phase || task.data?.observed_state || "等待 Worker"}</span></div>}
    {result && <>
      {result.result_limited && <Alert severity="warning" title="结果达到响应大小上限" detail="已返回完整行之前的安全范围；请收窄 SQL 字段或条件。" />}
      {enableLabels && !result.labelable && <Alert severity="info" title="当前结果不可打标" detail={result.labelability_reason || "该数据版本缺少稳定 Record Manifest，或查询没有保留内部行身份。"} />}
      {enableLabels && result.labelable && <div className="dataset-label-toolbar"><div><strong>{selectedRows.size ? `已选择 ${selectedRows.size} 行` : "可对查询结果打标"}</strong><span>标签关系将保存为不可变 Parquet 快照</span></div><button className="button primary small" disabled={Boolean(labelTaskId)} onClick={() => setLabelingOpen(true)}><Tags size={14} />打标签</button></div>}
      {labelTaskId && !TERMINAL_STATES.has(labelTask.data?.observed_state || "") && <Alert severity="info" title="正在执行打标任务" detail={labelTask.data?.current_phase || "Worker 正在冻结选择集并生成标签快照"} />}
      {labelTask.data && ["failed", "lost", "cancelled"].includes(labelTask.data.observed_state) && <Alert severity="error" title="打标任务失败" detail={String(labelTask.data.error?.message || labelTask.data.observed_state)} />}
      <div className="hf-data-grid" role="region" aria-label="数据集二维表格" tabIndex={0}>
        <table><thead><tr>{enableLabels && result.labelable && <th className="dataset-select-column"><input aria-label="选择当前页全部数据行" type="checkbox" checked={result.rows.length > 0 && selectedRows.size === result.rows.length} onChange={(event) => setSelectedRows(event.target.checked ? new Set(result.rows.map((_, index) => index)) : new Set())} /></th>}{result.columns.map((column) => <th key={column.name}><strong title={column.name}>{column.name}</strong><span title={column.type}>{column.type}</span></th>)}</tr></thead><tbody>{result.rows.map((row, rowIndex) => <tr key={`${result.offset}:${rowIndex}`}>{enableLabels && result.labelable && <td className="dataset-select-column"><input aria-label={`选择第 ${result.offset + rowIndex + 1} 行`} type="checkbox" checked={selectedRows.has(rowIndex)} onChange={(event) => { const next = new Set(selectedRows); if (event.target.checked) next.add(rowIndex); else next.delete(rowIndex); setSelectedRows(next); }} /></td>}{result.columns.map((column, columnIndex) => { const value = row[columnIndex]; const text = viewerCellText(value); return <td key={`${column.name}:${columnIndex}`}><button type="button" className="dataset-cell" title="点击查看完整单元格" onClick={() => setExpanded({ column: column.name, value })}>{text}</button></td>; })}</tr>)}</tbody></table>
      </div>
      <footer className="hf-viewer-pagination"><span>第 {result.total_rows ? formatNumber(result.offset + 1) : 0}–{formatNumber(result.offset + result.rows.length)} 行，共 {formatNumber(result.total_rows)} 行{result.truncated_cells ? ` · ${result.truncated_cells} 个超长单元格已截断` : ""}</span><div><button className="button secondary small" disabled={pending || result.offset === 0} onClick={() => browse(Math.max(0, result.offset - result.limit))}><ArrowLeft size={14} />上一页</button><button className="button secondary small" disabled={pending || !result.has_more || !result.rows.length} onClick={() => browse(result.offset + result.rows.length)}>下一页<ArrowRight size={14} /></button></div></footer>
    </>}
    {result && <form className="dataset-viewer-jump" onSubmit={(event) => { event.preventDefault(); const row = Number(jumpRow); if (Number.isInteger(row) && row >= 1 && row <= result.total_rows) browse(row - 1); }}><label>跳转到第 <input aria-label="跳转行号" type="number" min={1} max={result.total_rows} value={jumpRow} onChange={(event) => setJumpRow(event.target.value)} /> 行</label><button className="button secondary small" disabled={pending || !Number.isInteger(Number(jumpRow)) || Number(jumpRow) < 1 || Number(jumpRow) > result.total_rows}>跳转</button></form>}
    {expanded && <Modal title={expanded.column} description="当前查询结果中的原始单元格值" onClose={() => setExpanded(null)} wide><pre className="dataset-cell-expanded">{viewerCellText(expanded.value)}</pre></Modal>}
    {labelingOpen && result && <DatasetLabelModal versionId={version.id} sourceTaskId={taskId} result={result} selectedRows={selectedRows} schemes={labelSchemes.data || []} onClose={() => setLabelingOpen(false)} onCreated={(createdTaskId) => { setLabelingOpen(false); setSelectedRows(new Set()); setLabelTaskId(createdTaskId); }} />}
  </section>;
}

function DatasetLabelModal({ versionId, sourceTaskId, result, selectedRows, schemes, onClose, onCreated }: { versionId: string; sourceTaskId: string; result: DatasetViewerResult; selectedRows: Set<number>; schemes: LabelScheme[]; onClose: () => void; onCreated: (taskId: string) => void }) {
  const [scope, setScope] = useState<"selected" | "page" | "query">(selectedRows.size ? "selected" : "page");
  const [action, setAction] = useState<"apply" | "remove">("apply");
  const [schemeId, setSchemeId] = useState(schemes[0]?.id || "");
  const [selections, setSelections] = useState<Record<string, string[]>>({});
  const [annotator, setAnnotator] = useState("local-user");
  const [note, setNote] = useState("");
  const scheme = schemes.find((item) => item.id === schemeId) || schemes[0];
  const labelIds = Object.values(selections).flat();
  const create = useMutation({ mutationFn: () => {
    const indexes = scope === "selected" ? Array.from(selectedRows) : scope === "page" ? result.row_identities.map((_, index) => index) : [];
    const records = indexes.map((index) => result.row_identities[index]).filter(Boolean);
    return post<{ task: Task }>(`/catalog/dataset-versions/${versionId}/label-operations`, { source_query_task_id: sourceTaskId, scope, action, scheme_version_id: scheme?.current_version?.id, label_ids: labelIds, records, annotator, note });
  }, onSuccess: (response) => onCreated(response.task.id) });
  function chooseSingle(groupId: string, labelId: string) { setSelections({ ...selections, [groupId]: labelId ? [labelId] : [] }); }
  function toggleMultiple(groupId: string, labelId: string, checked: boolean) { const current = selections[groupId] || []; setSelections({ ...selections, [groupId]: checked ? [...current, labelId] : current.filter((id) => id !== labelId) }); }
  return <Modal title="对查询结果打标签" description="提交后先冻结稳定 Record ID，再生成选择集、变更集和新的有效标签 Parquet 快照。" onClose={onClose} wide>{schemes.length ? <form className="form-grid" onSubmit={(event) => { event.preventDefault(); create.mutate(); }}>
    <div className="form-grid two-columns"><Field label="作用范围"><select value={scope} onChange={(event) => setScope(event.target.value as typeof scope)}><option value="selected" disabled={!selectedRows.size}>已勾选行（{selectedRows.size}）</option><option value="page">当前结果页（{result.rows.length}）</option><option value="query">全部查询结果（{result.total_rows}）</option></select></Field><Field label="动作"><select value={action} onChange={(event) => setAction(event.target.value as typeof action)}><option value="apply">应用标签</option><option value="remove">移除标签</option></select></Field><Field label="标签体系"><select value={scheme?.id || ""} onChange={(event) => { setSchemeId(event.target.value); setSelections({}); }}>{schemes.map((item) => <option key={item.id} value={item.id}>{item.name} · V{item.current_version?.version}</option>)}</select></Field><Field label="打标人"><input value={annotator} onChange={(event) => setAnnotator(event.target.value)} /></Field></div>
    <div className="dataset-label-groups">{scheme?.current_version?.groups.map((group) => <section key={group.id}><header><strong>{group.name}</strong><span>{group.selection_mode === "single" ? "单选" : "多选"}</span></header>{group.selection_mode === "single" ? <select aria-label={`${group.name}标签`} value={selections[group.id]?.[0] || ""} onChange={(event) => chooseSingle(group.id, event.target.value)}><option value="">不操作此组</option>{group.labels.map((label) => <option key={label.id} value={label.id}>{label.name}</option>)}</select> : <div>{group.labels.map((label) => <label className="check-field" key={label.id}><input type="checkbox" checked={(selections[group.id] || []).includes(label.id)} onChange={(event) => toggleMultiple(group.id, label.id, event.target.checked)} /><span>{label.name}</span></label>)}</div>}</section>)}</div>
    <Field label="备注（可选）"><textarea value={note} onChange={(event) => setNote(event.target.value)} /></Field>
    {scope === "query" && <Alert severity="warning" title="将处理全部查询结果" detail={`Worker 会重新执行这条已冻结 SQL，并先物化 ${formatNumber(result.total_rows)} 行对应的去重 Record ID；后续数据或标签变化不会改变本次选择集。`} />}
    <ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!scheme?.current_version?.id || !labelIds.length || !annotator.trim()} onCancel={onClose} submitLabel={action === "apply" ? "创建打标任务" : "创建移除任务"} />
  </form> : <><EmptyState title="当前模态还没有可用标签体系" description="请先到数据资产的“标签管理”子页创建并发布 V1。" /><div className="form-actions"><button className="button secondary" onClick={onClose}>关闭</button></div></>}</Modal>;
}

function RecordManifestExplorerWorkspace({ datasets, initialVersionId, onMixtureCreated }: { datasets: Dataset[]; initialVersionId?: string | null; onMixtureCreated: () => void }) {
  const candidates = useMemo(() => datasets.flatMap((dataset) => (dataset.versions || []).filter((version: any) => version.state === "ready" && version.record_manifest_artifact_id).map((version: any) => ({ dataset, version }))), [datasets]);
  const [selected, setSelected] = useState<Record<string, boolean>>(() => initialVersionId ? { [initialVersionId]: true } : {});
  const [sourceSearch, setSourceSearch] = useState("");
  const [includePaths, setIncludePaths] = useState("");
  const [excludePaths, setExcludePaths] = useState("");
  const [language, setLanguage] = useState("");
  const [minLength, setMinLength] = useState(0);
  const [maxLength, setMaxLength] = useState(0);
  const [offset, setOffset] = useState(0);
  const [mixtureName, setMixtureName] = useState("检索混合集");
  const selectedIds = candidates.filter(({ version }) => selected[version.id]).map(({ version }) => version.id);
  const visibleCandidates = candidates.filter(({ dataset, version }) => `${dataset.name} ${dataset.language} ${dataset.license} ${(dataset.tags || []).join(" ")} ${version.tier}`.toLowerCase().includes(sourceSearch.trim().toLowerCase()));
  const languages = Array.from(new Set(candidates.map(({ version, dataset }) => String(version.quality_report?.dominant_language || dataset.language || "")).filter(Boolean))).sort();
  const patterns = (value: string) => value.split(/[\n,，]/).map((item) => item.trim()).filter(Boolean);
  const payload = (pageOffset = offset) => ({ dataset_version_ids: selectedIds, include_paths: patterns(includePaths), exclude_paths: patterns(excludePaths), languages: language ? [language] : [], min_length: minLength, max_length: maxLength, sample_ratio: 1, seed: 42, offset: pageOffset, limit: 100 });
  const query = useMutation({ mutationFn: (pageOffset: number) => post<any>("/catalog/explorer/query", payload(pageOffset)) });
  const createMixture = useMutation({ mutationFn: () => post<any>("/catalog/explorer/mixtures", { ...payload(0), name: mixtureName, description: "由 DuckDB 行级检索条件构建", target_tokens: query.data?.total_tokens || 0, weights: {} }), onSuccess: onMixtureCreated });
  function runQuery(pageOffset = 0) { setOffset(pageOffset); query.mutate(pageOffset); }
  function toggleAll() {
    const next = selectedIds.length !== candidates.length;
    setSelected(Object.fromEntries(candidates.map(({ version }) => [version.id, next])));
  }
  return <section className="record-explorer-workspace">
    <div className="record-explorer-note"><span><Database size={16} />DuckDB</span><p>真实查询全部已选版本的稳定记录索引。当前索引字段：数据集、版本、路径、语言、字符长度、Token 估算和稳定记录 ID。</p></div>
    <Alert severity="warning" title="正文和记录级标签还没有进入全局索引" detail="当前不会提供假的正文或标签搜索。Record Manifest 不保存正文，现有人工标注也只覆盖导入时冻结的有限分析样本；完整能力需要独立标签版本和全文/列式查询快照。" />
    <div className="record-explorer-builder">
      <section className="dataset-explorer-sources"><header><div><b>1</b><div><strong>查询范围</strong><span>按名称、数据集标签、语言或许可证查找版本</span></div></div><button className="button ghost small" onClick={toggleAll}>{selectedIds.length === candidates.length ? "清空" : "全选"}</button></header><div className="search-box explorer-source-search"><Search size={14} /><input aria-label="搜索可检索数据版本" value={sourceSearch} onChange={(event) => setSourceSearch(event.target.value)} placeholder="搜索数据集、标签、语言或许可证" /></div><div className="explorer-source-list">{visibleCandidates.map(({ dataset, version }) => <label key={version.id} className="check-field"><input type="checkbox" checked={Boolean(selected[version.id])} onChange={(event) => setSelected((current) => ({ ...current, [version.id]: event.target.checked }))} /><span><strong>{dataset.name} · V{version.version}</strong><small>{String(version.tier).toUpperCase()} · {formatNumber(version.document_count)} 行 · {(dataset.tags || []).join(" / ") || "无数据集标签"}</small></span></label>)}</div><footer>已选择 <strong>{selectedIds.length}</strong> / {candidates.length} 个版本</footer></section>
      <section className="record-explorer-query"><header><b>2</b><div><strong>记录条件</strong><span>筛选真实索引字段，不在搜索阶段抽样</span></div></header><div className="form-grid two-columns"><Field label="语言"><select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="">全部语言</option>{languages.map((value) => <option key={value} value={value}>{value}</option>)}</select></Field><Field label="字符长度"><div className="range-inputs"><input aria-label="最小字符数" type="number" min="0" value={minLength} onChange={(event) => setMinLength(Number(event.target.value))} placeholder="最小" /><span>至</span><input aria-label="最大字符数" type="number" min="0" value={maxLength} onChange={(event) => setMaxLength(Number(event.target.value))} placeholder="不限" /></div></Field><Field label="包含路径" hint="逗号或换行分隔 glob"><textarea value={includePaths} onChange={(event) => setIncludePaths(event.target.value)} placeholder="例如 train/**" /></Field><Field label="排除路径"><textarea value={excludePaths} onChange={(event) => setExcludePaths(event.target.value)} placeholder="例如 **/test/**" /></Field></div><div className="record-query-summary"><span>{selectedIds.length} 个版本</span><span>{language || "全部语言"}</span><span>{patterns(includePaths).length + patterns(excludePaths).length} 条路径规则</span><button className="button primary" disabled={!selectedIds.length || query.isPending || Boolean(maxLength && maxLength < minLength)} onClick={() => runQuery(0)}><Search size={16} />{query.isPending ? "查询中…" : "搜索记录"}</button></div></section>
    </div>
    <ErrorMessage error={query.error || createMixture.error} />
    {query.data ? <section className="dataset-explorer-results"><header className="explorer-results-header"><div><b>3</b><div><strong>查询结果</strong><span>完整命中统计与当前分页记录</span></div></div><small>第 {formatNumber(offset + 1)}–{formatNumber(Math.min(offset + query.data.rows.length, query.data.total_rows))} 行</small></header><div className="metrics-grid four"><article className="metric-card"><span>匹配行数</span><strong>{formatNumber(query.data.total_rows)}</strong></article><article className="metric-card"><span>Token 估算</span><strong>{formatNumber(query.data.total_tokens)}</strong></article><article className="metric-card"><span>数据版本</span><strong>{query.data.sources.length}</strong></article><article className="metric-card"><span>查询引擎</span><strong>DuckDB</strong></article></div><div className="data-table-wrap"><table className="data-table"><thead><tr><th>数据版本</th><th>逻辑路径 / 行</th><th>语言</th><th>字符</th><th>Tokens</th><th>记录 ID</th></tr></thead><tbody>{query.data.rows.map((row: any) => <tr key={`${row.dataset_version_id}:${row.record_id}`}><td><strong>{row.dataset_name}</strong><span>V{row.dataset_version}</span></td><td><strong>{row.logical_path}</strong><span>row {row.record_index}{row.preview?.raw_text ? ` · ${String(row.preview.raw_text).slice(0, 120)}` : ""}</span></td><td>{row.language || "—"}</td><td>{formatNumber(row.character_length)}</td><td>{formatNumber(row.token_estimate)}</td><td><code>{String(row.record_id).slice(0, 14)}</code></td></tr>)}</tbody></table></div><div className="explorer-pagination"><button className="button secondary small" disabled={offset === 0 || query.isPending} onClick={() => runQuery(Math.max(0, offset - 100))}><ArrowLeft size={14} />上一页</button><span>每页最多 100 行</span><button className="button secondary small" disabled={offset + query.data.rows.length >= query.data.total_rows || query.isPending} onClick={() => runQuery(offset + 100)}>下一页<ArrowRight size={14} /></button></div><div className="dataset-explorer-materialize"><Field label="新混合集名称"><input value={mixtureName} onChange={(event) => setMixtureName(event.target.value)} /></Field><div><strong>把可复现条件冻结为混合集</strong><span>{query.data.materializable_as_mixture ? "保存精确版本、路径和长度条件；抽样权重在混合编辑器中另行设置。" : "当前语言条件还不能由既有混合选择器复现，请清除语言条件后构建。"}</span></div><button className="button primary" disabled={!query.data.materializable_as_mixture || !mixtureName.trim() || !query.data.total_rows || createMixture.isPending} onClick={() => createMixture.mutate()}><Layers3 size={16} />{createMixture.isPending ? "正在构建…" : "创建混合集"}</button></div></section> : <section className="explorer-empty-state"><Search size={28} /><strong>选择版本并搜索记录</strong><span>结果、来源汇总和可复现构建入口会显示在这里。</span></section>}
  </section>;
}

function PreparedDataPanel({ corpora, indexes, tasks, onOpenTask }: { corpora: any[]; indexes: any[]; tasks: Task[]; onOpenTask: (taskId: string) => void }) {
  const ready = [...corpora, ...indexes].filter((item) => item.status === "ready").length;
  const interrupted = tasks.filter((task) => ["failed", "lost", "cancelled"].includes(task.observed_state)).length;
  return <section className="tokenized-workspace">
    <section className="metrics-grid four"><Metric label="制品版本" value={corpora.length + indexes.length} /><Metric label="可复用" value={ready} /><Metric label="构建记录" value={tasks.length} /><Metric label="失败 / 中断" value={interrupted} /></section>
    <Alert severity="info" title="这里是训练中心的自动准备结果" detail="训练计划声明数据意图；系统按完整配置签名复用或构建这些不可变制品。实际 Token 数是结果，不是隐藏成功门槛。" />
    <Panel className="table-panel" title="编码数据" description="数据集版本按锁定 Tokenizer 编码后的连续 Token 与文档索引。">
      {corpora.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>数据集</th><th>Tokenizer</th><th>实际产量</th><th>状态</th><th>有效配置</th></tr></thead><tbody>{corpora.map((item) => <tr key={item.id}><td><strong>{item.dataset_name || item.dataset_version_id}</strong><span>V{item.dataset_version || "—"} · {String(item.id).slice(0, 8)}</span></td><td><strong>{item.tokenizer_name || item.tokenizer_version_id}</strong><span>V{item.tokenizer_version || "—"}</span></td><td><strong>{formatNumber(item.token_count)} Tokens</strong><span>{formatNumber(item.document_count)} 文档 · {formatNumber(item.shard_count)} 分片</span></td><td><StatusBadge status={item.status} />{item.warnings?.length ? <span>{item.warnings.length} 条提示</span> : null}</td><td><details className="collapsed-config"><summary>查看全部配置</summary><pre className="json-preview">{JSON.stringify({ selection: item.selection_config, encoding: item.encoding_config, identity_hash: item.identity_hash }, null, 2)}</pre></details></td></tr>)}</tbody></table></div> : <EmptyState title="还没有已准备编码数据" description="从训练计划执行“准备并预检”后，系统会在此记录自动生成的制品。" />}
    </Panel>
    <Panel className="table-panel" title="训练读取索引" description="在编码数据之上冻结序列、尾部、混合采样和数据消费策略；不会复制 Token 数据。">
      {indexes.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>数据混合</th><th>序列配置</th><th>请求 / 实际</th><th>状态</th><th>有效配置</th></tr></thead><tbody>{indexes.map((item) => { const config = item.config_snapshot?.configuration || {}; return <tr key={item.id}><td><strong>{item.mixture_name || item.mixture_version_id}</strong><span>V{item.mixture_version || "—"} · {item.tokenizer_name || "Tokenizer"}</span></td><td><strong>{formatNumber(item.sequence_length)} Tokens</strong><span>{config.consumption_policy || "—"} · {config.tail_policy || "—"}</span></td><td><strong>{formatNumber(item.requested_tokens)} / {formatNumber(item.token_count)}</strong><span>{formatNumber(item.sequence_count)} sequences</span></td><td><StatusBadge status={item.status} />{item.warnings?.length ? <span>{item.warnings.length} 条提示</span> : null}</td><td><details className="collapsed-config"><summary>查看全部配置</summary><pre className="json-preview">{JSON.stringify(item.config_snapshot || {}, null, 2)}</pre></details></td></tr>; })}</tbody></table></div> : <EmptyState title="还没有训练读取索引" description="索引由训练中心根据目标 Tokens、序列长度和消费策略自动生成。" />}
    </Panel>
    <Panel className="table-panel" title="构建记录" description="训练数据准备的每次执行都在这里留痕；失败或取消不会冒充可复用制品，但错误、输入配置和日志必须可追溯。">
      {tasks.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>任务</th><th>类型</th><th>发起来源</th><th>进度</th><th>状态</th><th>错误与配置</th><th>详情</th></tr></thead><tbody>{tasks.map((task) => {
        const taskType = task.kind === "data_tokenize_corpus" ? "编码 Corpus" : task.kind === "training_data_index" ? "训练读取索引" : "历史预分词";
        const origin = preparationTaskOrigin(task);
        const errorMessage = String(task.error?.message || "").trim();
        return <tr key={task.id}><td><strong>{task.title}</strong><span>{formatDate(task.created_at)} · {task.id.slice(0, 8)}</span></td><td><strong>{taskType}</strong><span>{task.kind === "data_tokenize" ? "旧流程，只读追溯" : "当前流程"}</span></td><td><strong>{origin.title}</strong><span>{origin.detail}</span><span>{origin.action}</span></td><td><strong>{Math.round((task.progress || 0) * 100)}%</strong><Progress value={task.progress || 0} /></td><td><StatusBadge status={task.observed_state} /></td><td>{errorMessage && <span className="inline-error">{errorMessage}</span>}<details className="collapsed-config"><summary>查看输入配置与错误</summary><pre className="json-preview">{JSON.stringify({ trigger_context: task.payload?.trigger_context || null, error: task.error || {}, payload: task.payload || {}, result: task.result || {} }, null, 2)}</pre></details></td><td><button type="button" className="button ghost small" onClick={() => onOpenTask(task.id)}><Activity size={13} />查看日志与资源</button></td></tr>;
      })}</tbody></table></div> : <EmptyState title="还没有训练数据构建记录" description="从训练计划执行“准备并预检”后，成功、失败和取消的执行记录都会出现在这里。" />}
    </Panel>
  </section>;
}

function preparationTaskOrigin(task: Task) {
  const context = task.payload?.trigger_context || {};
  if (context.source === "training_plan_prepare_data") {
    const stages = Array.isArray(context.selected_stages) ? context.selected_stages : [];
    const stageLabel = stages.length
      ? stages.map((stage: any) => `${stage.name || `Stage ${stage.index}`}（${stage.type || "training"}）`).join("、")
      : "阶段未记录";
    return {
      title: context.training_plan_name || context.training_plan_id || "训练计划",
      detail: `Revision ${context.plan_revision ?? "—"} · ${stageLabel}`,
      action: context.action || "准备训练数据与运行提示",
    };
  }
  if (task.kind === "data_tokenize") {
    return {
      title: "旧版预分词入口",
      detail: "历史任务未记录训练计划与阶段",
      action: "旧流程直接构建 packed sequences",
    };
  }
  if (context.source === "direct_api") {
    return {
      title: "直接领域 API",
      detail: "未关联训练计划或阶段",
      action: context.action || "直接创建数据制品",
    };
  }
  return {
    title: "来源未记录",
    detail: "该任务创建时没有冻结发起上下文",
    action: "—",
  };
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return <article className="metric-card"><span>{label}</span><strong>{String(value)}</strong></article>;
}

function locationWorker(location: any, workers: any[]) {
  return location?.worker_id ? workers.find((worker) => worker.id === location.worker_id) : null;
}

function locationWorkerName(location: any, workers: any[]) {
  if (!location?.worker_id) return location?.storage_type === "managed" ? "控制面托管" : "共享位置";
  return locationWorker(location, workers)?.name || `Worker ${location.worker_id}`;
}

function locationStorageLabel(location: any) {
  if (location?.storage_type === "managed") return "控制面托管位置";
  if (location?.storage_type === "external") return location.worker_id ? "Worker 私有外部位置" : "共享外部位置";
  return location?.storage_type || "未知存储类型";
}

function MaterializationLocationCard({ role, location, workers }: { role: "主位置" | "已验证副本"; location: any; workers: any[] }) {
  const worker = locationWorker(location, workers);
  const workerName = locationWorkerName(location, workers);
  return <article className={`materialization-location${role === "主位置" ? " primary" : ""}`} aria-label={`${role} ${workerName}`}>
    <header>
      <div><span>{role}</span><strong>{workerName}</strong></div>
      <StatusBadge status={worker?.status || location.availability_state} />
    </header>
    <dl>
      <div><dt>Worker ID</dt><dd><code>{location.worker_id || "不绑定 Worker"}</code></dd></div>
      <div><dt>存储性质</dt><dd>{locationStorageLabel(location)}</dd></div>
      <div><dt>最近校验</dt><dd>{location.last_verified_at ? formatDate(location.last_verified_at) : "尚未校验"}</dd></div>
      <div className="materialization-location-uri"><dt>具体位置</dt><dd><code title={location.uri}>{location.uri}</code></dd></div>
    </dl>
  </article>;
}

function DatasetMaterializationPanel({ version }: { version: any }) {
  const queryClient = useQueryClient();
  const contract = version.materialization || null;
  const workers = useQuery({ queryKey: ["workers", "dataset-materialization"], queryFn: () => api<any[]>("/workers") });
  const [targetWorkerId, setTargetWorkerId] = useState("");
  const [selectionMode, setSelectionMode] = useState<"auto" | "manual">("auto");
  const [routeId, setRouteId] = useState("");
  const operations = useQuery({
    queryKey: ["dataset-materializations", version.id],
    queryFn: () => api<any[]>(`/dataset-versions/${version.id}/materializations`),
    enabled: Boolean(contract && version.id && version.artifact_id),
    refetchInterval: (query) => (query.state.data || []).some((item) => ["queued", "running"].includes(item.status)) ? 1500 : false,
  });
  const routes = useQuery({
    queryKey: ["dataset-materialization-routes", version.id, targetWorkerId],
    queryFn: () => api<any>(`/dataset-versions/${version.id}/materialization-routes?target_worker_id=${encodeURIComponent(targetWorkerId)}`),
    enabled: Boolean(contract && version.artifact_id && targetWorkerId),
  });
  const create = useMutation({
    mutationFn: () => post<any>(`/dataset-versions/${version.id}/materializations`, {
      target_worker_id: targetWorkerId,
      selection_mode: selectionMode,
      route_id: selectionMode === "manual" ? routeId : null,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset-materializations", version.id] });
      queryClient.invalidateQueries({ queryKey: ["dataset", version.dataset_id] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
  const latest = operations.data?.[0];
  useEffect(() => {
    if (latest?.status !== "succeeded") return;
    queryClient.invalidateQueries({ queryKey: ["dataset", version.dataset_id] });
    queryClient.invalidateQueries({ queryKey: ["datasets"] });
    queryClient.invalidateQueries({ queryKey: ["dataset-materialization-routes", version.id] });
  }, [latest?.id, latest?.status, queryClient, version.dataset_id, version.id]);
  if (!contract) {
    return <Panel title="数据物化与副本" description="等待控制面提供该版本的物化契约。">
      <Alert severity="warning" title="物化契约尚未加载" detail="当前 API 尚未返回主位置与恢复语义，不能据此把版本判断为 imported 或 pending。请完成数据库迁移并重启控制面后刷新。" />
    </Panel>;
  }
  const activeForTarget = (operations.data || []).find((item: any) =>
    item.target_worker_id === targetWorkerId && ["queued", "running"].includes(item.status));
  const feasibleRoutes = (routes.data?.routes || []).filter((route: any) => route.feasible);
  const selectedRoute = (routes.data?.routes || []).find((route: any) => route.route_id === routeId);
  const primary = contract.primary_location;
  const replicas = contract.replicas || [];
  const availableWorkers = workers.data || [];
  const canReplay = Boolean(contract.recovery?.can_replay);
  const kindLabel = contract.kind === "download"
    ? canReplay ? "可从主位置复制 / 按冻结来源重新下载" : "仅可复制已有位置（缺少冻结下载规格）"
    : contract.kind === "derived"
      ? canReplay ? "可从主位置复制 / 按 Recipe 重建" : "仅可复制已有位置（缺少冻结 Recipe）"
      : "可从主位置或已验证副本复制";
  const recoveryLabel = contract.kind === "download"
    ? canReplay ? `${contract.recovery.source_type || "冻结来源"} · ${contract.recovery.source_uri || "重新下载"}` : "仅复制已有位置"
    : contract.kind === "derived"
      ? canReplay ? "按 Recipe 重建" : "仅复制已有位置"
      : "仅复制已有位置";
  return <Panel title="数据物化与副本" description="数据身份由不可变版本与哈希确定；主位置唯一，其他位置必须校验后才成为副本。" actions={<StatusBadge status={contract.state || "pending"} />}>
    <div className="metrics-grid four">
      <Metric label="物化类型" value={`${contract.kind || "imported"} · ${kindLabel}`} />
      <article className="metric-card materialization-primary-metric">
        <span>主位置</span>
        <strong>{primary ? locationWorkerName(primary, availableWorkers) : "尚未建立"}</strong>
        {primary && <small title={primary.uri}>{primary.uri}</small>}
      </article>
      <Metric label="已验证副本" value={`${replicas.length} 个位置`} />
      <article className="metric-card materialization-recovery-metric">
        <span>缺失时恢复</span>
        <strong>{recoveryLabel}</strong>
        <small>{canReplay ? contract.recovery.replay_adapter_id : "无可重放规格"}</small>
      </article>
    </div>
    <div className="materialization-location-list" aria-label="已登记数据位置">
      {primary ? <MaterializationLocationCard role="主位置" location={primary} workers={availableWorkers} /> : <Alert severity="warning" title="主位置缺失" detail="版本处于 pending/degraded 时需要先恢复一个通过校验的位置。" />}
      {replicas.map((replica: any) => <MaterializationLocationCard key={replica.id} role="已验证副本" location={replica} workers={availableWorkers} />)}
    </div>
    {version.artifact_id && <div className="form-grid three-columns">
      <Field label="目标 Worker"><select value={targetWorkerId} onChange={(event) => { setTargetWorkerId(event.target.value); setRouteId(""); }}><option value="">选择 Worker</option>{(workers.data || []).map((worker) => <option key={worker.id} value={worker.id}>{worker.name} · {worker.status}</option>)}</select></Field>
      <Field label="选路方式"><select value={selectionMode} onChange={(event) => { setSelectionMode(event.target.value as "auto" | "manual"); setRouteId(""); }}><option value="auto">系统自动选择最低成本路线</option><option value="manual">手动指定可行路线</option></select></Field>
      {selectionMode === "manual" ? <Field label="物化路线"><select value={routeId} onChange={(event) => setRouteId(event.target.value)}><option value="">选择路线</option>{(routes.data?.routes || []).map((route: any) => <option key={route.route_id} value={route.route_id} disabled={!route.feasible}>{route.strategy} · {route.reason}</option>)}</select></Field> : <Field label="自动路线"><input readOnly value={feasibleRoutes[0] ? `${feasibleRoutes[0].strategy} · ${feasibleRoutes[0].reason}` : targetWorkerId ? "没有可行路线" : "选择 Worker 后计算"} /></Field>}
    </div>}
    {targetWorkerId && routes.data && !feasibleRoutes.length && <Alert severity="warning" title="没有可行路线" detail={(routes.data.routes || []).map((route: any) => route.reason).filter(Boolean).join("；") || "目标 Worker 不具备所需能力或没有可达数据位置。"} />}
    {selectionMode === "manual" && selectedRoute && <p className="muted">手动路线会随任务冻结：{selectedRoute.reason}</p>}
    {latest && <Alert severity={latest.status === "failed" ? "error" : latest.status === "succeeded" ? "success" : "info"} title={`最近同步：${latest.status}`} detail={`${latest.chosen_route?.strategy || "—"}${latest.task_id ? ` · 任务 ${latest.task_id}` : ""}`} />}
    <ErrorMessage error={workers.error || routes.error || operations.error || create.error} />
    <div className="form-actions"><button className="button primary" disabled={!targetWorkerId || !feasibleRoutes.length || (selectionMode === "manual" && !routeId) || create.isPending || Boolean(activeForTarget)} onClick={() => create.mutate()}><RefreshCw size={15} />{create.isPending ? "正在创建…" : "同步到 Worker"}</button></div>
  </Panel>;
}

function DatasetDetailPage({ datasetId, initialDataset, initialAction, onClose, onExplore }: { datasetId: string; initialDataset?: Dataset; initialAction?: string | null; onClose: () => void; onExplore: (versionId: string) => void }) {
  const [tab, setTab] = useState("overview");
  const [reanalyzeOpen, setReanalyzeOpen] = useState(initialAction === "reanalyze");
  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const queryClient = useQueryClient();
  const retryDownload = useRetryDownload();
  const dataset = useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => api<Dataset>(`/datasets/${datasetId}`),
    placeholderData: initialDataset,
    refetchInterval: (query) => {
      const state = query.state.data?.current_version?.state;
      const analysisState = query.state.data?.current_version?.active_analysis?.status;
      return (analysisState && !TERMINAL_STATES.has(analysisState)) || (state && !TERMINAL_STATES.has(state) && state !== "ready") ? 1500 : false;
    },
  });
  const version = dataset.data?.current_version;
  const mostRecentAnalysis = version?.analysis_revisions?.[0];
  const analysisTaskId = version?.active_analysis?.task_id || mostRecentAnalysis?.task_id || version?.latest_analysis?.task_id || version?.task_id;
  const task = useQuery({
    queryKey: ["task", analysisTaskId],
    queryFn: () => api<Task>(`/tasks/${analysisTaskId}`),
    enabled: Boolean(analysisTaskId),
    refetchInterval: (query) => TERMINAL_STATES.has(query.state.data?.observed_state || "") ? false : 1200,
  });
  const reanalyze = useMutation({
    mutationFn: () => post<any>(`/datasets/${datasetId}/reanalyze`, { resource_request: { gpus: 0 } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      setReanalyzeOpen(false);
    },
  });
  const stopAnalysis = useMutation({
    mutationFn: (taskId: string) => post<Task>(`/tasks/${taskId}/control`, {
      action: "cancel",
      reason: "用户从数据详情停止分析",
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["task", analysisTaskId] });
      queryClient.invalidateQueries({ queryKey: ["dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  if (!dataset.data || !version) return <section className="dataset-resource-page"><button className="button ghost small" onClick={onClose}><ArrowLeft size={16} />全部数据集</button>{dataset.error ? <ErrorMessage error={dataset.error} /> : <RouteLoading label="正在读取数据集详情…" />}</section>;
  const item = dataset.data;
  const canExplore = Boolean(version.state === "ready" && version.artifact_id);
  const isAnalysisRevisionTask = Boolean(task.data?.payload?.analysis_revision_id);
  const snapshotDrift = String(task.data?.error?.message || "").includes("snapshot_drift");
  return <section className="dataset-resource-page">
    <header className="dataset-resource-header"><button className="button ghost small" onClick={onClose}><ArrowLeft size={16} />全部数据集</button><div className="dataset-resource-identity"><span className="dataset-source-mark large">{item.name.slice(0, 2).toUpperCase()}</span><div><span className="eyebrow">DATASET</span><h1>{item.name}</h1><p>{modalityLabels[item.modality || "text"]} · {item.source_type} · {item.language} · {item.license}</p><div className="dataset-resource-tags"><StatusBadge status={version.train_ready ? "ready" : item.status} /><span>{String(version.tier || "").toUpperCase()} V{version.version}</span>{(item.tags || []).map((tag) => <i key={tag}>{tag}</i>)}</div></div></div><div className="dataset-resource-actions"><button type="button" className="button secondary" disabled={!canExplore} onClick={() => onExplore(version.id)}><Search size={15} />检索此版本</button><button type="button" className="button secondary" onClick={() => setEditOpen(true)}><Pencil size={14} />设置</button><button type="button" className="button danger" onClick={() => setDeleteOpen(true)}><Trash2 size={14} />删除</button></div></header>
    <nav className="dataset-resource-tabs" aria-label="数据集详情导航">{[["overview", "数据集卡片"], ["samples", "数据查看器"], ["quality", "统计与分布"], ["versions", "文件与版本"], ["license", "来源与许可"]].map(([key, label]) => <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>{label}</button>)}</nav>
    <div className="dataset-resource-content">
      <Panel title="数据分析" description="从冻结内容生成文件清单、质量报告和有限样本；数据版本保持不变。" actions={<button className="button secondary small" disabled={Boolean(version.active_analysis) || Boolean(task.data && !TERMINAL_STATES.has(task.data.observed_state))} onClick={() => setReanalyzeOpen(true)}><RefreshCw size={14} />重新分析</button>}>
        <p className="muted">当前为 V{version.version} · 分析 A{version.analysis_revision || "—"}；重新分析只增加分析修订，不制造新的内容版本。</p>
      </Panel>
      {task.data && !TERMINAL_STATES.has(task.data.observed_state) && <Panel title="真实导入任务" description={task.data.current_phase || "等待 Worker 领取"} actions={<button type="button" className="button danger small" disabled={stopAnalysis.isPending || task.data.observed_state === "stopping"} onClick={() => stopAnalysis.mutate(task.data!.id)}><Square size={13} />{stopAnalysis.isPending || task.data.observed_state === "stopping" ? "正在停止…" : "停止分析"}</button>}><Progress value={task.data.progress || 0} /><p className="muted">{Math.round((task.data.progress || 0) * 100)}% · {task.data.observed_state}</p>{transferDetail(latestTransferProgress(task.data)) && <p className="muted">{transferDetail(latestTransferProgress(task.data))}</p>}<ErrorMessage error={stopAnalysis.error} /></Panel>}
      {task.data && RETRYABLE_DOWNLOAD_STATES.has(task.data.observed_state) && <div className="download-failure"><Alert severity="error" title={snapshotDrift ? "冻结快照已发生漂移" : task.data.observed_state === "lost" ? (isAnalysisRevisionTask ? "数据分析已中断" : "数据导入已中断") : (isAnalysisRevisionTask ? "数据分析失败" : "数据导入失败")} detail={snapshotDrift ? "源文件集合或内容校验和已变化。当前数据版本和上一次成功分析均未修改；请把变化后的来源作为新数据版本重新导入。" : downloadErrorDetail(task.data.error)} />{!snapshotDrift && <button type="button" className="button secondary small" disabled={retryDownload.isPending} onClick={() => retryDownload.mutate(task.data!.id)}><RotateCcw size={14} />{retryDownload.isPending ? "启动中…" : isAnalysisRevisionTask ? "重试分析" : "重新下载"}</button>}<ErrorMessage error={retryDownload.error} /></div>}
      {tab === "overview" && <><DatasetMaterializationPanel version={version} /><DatasetOverview version={version} /></>}
      {tab === "samples" && <DataViewerPanel version={version} datasetName={item.name} />}
      {tab === "quality" && <QualityReport report={version.quality_report} />}
      {tab === "versions" && <VersionHistory versions={item.versions} task={task.data} />}
      {tab === "license" && <SourcePanel dataset={item} version={version} />}
    </div>
    {reanalyzeOpen && <Modal title="重新分析数据" description={`将在 ${item.name} V${version.version} 上创建新的分析修订，不改变内容版本。`} onClose={() => setReanalyzeOpen(false)}><div className="preflight-list"><Alert severity="info" title="数据版本不会变化" detail="成功后会切换到最新分析结果；旧分析修订、Record Manifest 和任务日志继续保留。" /><Alert severity="warning" title="会完整复核冻结快照" detail="Worker 会重新计算文件 SHA-256；发现内容漂移时拒绝发布分析结果，并要求作为新数据版本重新导入。" /></div><ErrorMessage error={reanalyze.error} /><div className="form-actions"><button className="button secondary" onClick={() => setReanalyzeOpen(false)}>取消</button><button className="button primary" disabled={reanalyze.isPending} onClick={() => reanalyze.mutate()}>{reanalyze.isPending ? "创建中…" : "开始重新分析"}</button></div></Modal>}
    {editOpen && <DatasetEditModal dataset={item} onClose={() => setEditOpen(false)} onSaved={() => { queryClient.invalidateQueries({ queryKey: ["dataset", datasetId] }); queryClient.invalidateQueries({ queryKey: ["datasets"] }); queryClient.invalidateQueries({ queryKey: ["tasks"] }); setEditOpen(false); }} />}
    {deleteOpen && <DatasetDeleteModal dataset={item} onClose={() => setDeleteOpen(false)} onDeleted={() => { queryClient.invalidateQueries({ queryKey: ["datasets"] }); onClose(); }} />}
  </section>;
}

function DatasetEditModal({ dataset, onClose, onSaved }: { dataset: Dataset; onClose: () => void; onSaved: () => void }) {
  const configuredEndpoint = String(dataset.source_config?.hf_endpoint || "");
  const initialEndpointMode = !configuredEndpoint ? "deployment" : configuredEndpoint === HF_ENDPOINTS.official ? "official" : configuredEndpoint === HF_ENDPOINTS.mirror ? "mirror" : "custom";
  const [form, setForm] = useState({
    name: dataset.name,
    source_uri: dataset.source_uri || "",
    language: dataset.language,
    license: dataset.license,
    tags: (dataset.tags || []).join(", "),
    revision: String(dataset.source_config?.revision || ""),
    config: String(dataset.source_config?.config || ""),
    split: String(dataset.source_config?.split || ""),
    include_patterns: Array.isArray(dataset.source_config?.include_patterns) ? dataset.source_config!.include_patterns.join(", ") : String(dataset.source_config?.include_patterns || ""),
    hf_endpoint_mode: initialEndpointMode,
    custom_hf_endpoint: initialEndpointMode === "custom" ? configuredEndpoint : "",
  });
  const set = (key: keyof typeof form, value: string) => setForm((current) => ({ ...current, [key]: value }));
  const hfEndpoint = form.hf_endpoint_mode === "custom" ? form.custom_hf_endpoint.trim() : HF_ENDPOINTS[form.hf_endpoint_mode as keyof typeof HF_ENDPOINTS];
  const sourceConfig: Record<string, any> = { ...(dataset.source_config || {}) };
  for (const [key, value] of [["revision", form.revision], ["config", form.config], ["split", form.split]] as const) {
    if (value) sourceConfig[key] = value;
    else delete sourceConfig[key];
  }
  const includePatterns = form.include_patterns.split(/[,，]/).map((value) => value.trim()).filter(Boolean);
  if (includePatterns.length || Object.hasOwn(dataset.source_config || {}, "include_patterns")) sourceConfig.include_patterns = includePatterns;
  else delete sourceConfig.include_patterns;
  if (dataset.source_type === "huggingface" && hfEndpoint) sourceConfig.hf_endpoint = hfEndpoint;
  else if (dataset.source_type === "huggingface") delete sourceConfig.hf_endpoint;
  const sourceChanged = form.source_uri.trim() !== (dataset.source_uri || "") || JSON.stringify(sourceConfig) !== JSON.stringify(dataset.source_config || {});
  const customEndpointValid = form.hf_endpoint_mode !== "custom" || isHttpUrl(form.custom_hf_endpoint);
  const update = useMutation({
    mutationFn: async () => {
      if (sourceChanged) {
        const checked = await post<DatasetPreflight>(`/projects/${dataset.project_id}/datasets/preflight`, { source_type: dataset.source_type, source_uri: form.source_uri.trim(), source_config: sourceConfig });
        if (!checked.accessible || !checked.disk_space_sufficient) throw new Error(checked.error || "来源预检未通过，请检查地址、镜像和磁盘空间");
      }
      return patch<Dataset>(`/datasets/${dataset.id}`, {
        name: form.name.trim(),
        source_uri: form.source_uri.trim(),
        source_config: sourceConfig,
        language: form.language.trim(),
        license: form.license.trim(),
        tags: form.tags.split(/[,，]/).map((value) => value.trim()).filter(Boolean),
        resource_request: { gpus: 0 },
      });
    },
    onSuccess: onSaved,
  });
  const canSave = Boolean(form.name.trim() && form.source_uri.trim() && customEndpointValid);
  const hfFrozen = dataset.source_type === "huggingface";
  return <Modal title="编辑数据集" description={hfFrozen ? "HF 物理来源、commit 和文件清单已由下载计划冻结；此处只修改数据集元数据。" : sourceChanged ? "来源变化会创建新的不可变 Bronze 版本，旧版本与失败日志仍会保留。" : "只修改数据集元数据，不创建新版本。"} onClose={onClose} wide><form className="form-grid two-columns" onSubmit={(event) => { event.preventDefault(); update.mutate(); }}><Field label="数据集名称"><input value={form.name} onChange={(event) => set("name", event.target.value)} /></Field><Field label="来源类型"><input value={dataset.source_type} disabled /></Field><Field label={dataset.source_type === "huggingface" ? "数据集仓库 ID" : dataset.source_type === "local" ? "本地绝对路径" : "数据地址"}><input value={form.source_uri} disabled={hfFrozen} onChange={(event) => set("source_uri", event.target.value)} /></Field><Field label="语言声明"><input value={form.language} onChange={(event) => set("language", event.target.value)} /></Field><Field label="许可证声明"><input value={form.license} onChange={(event) => set("license", event.target.value)} /></Field><Field label="标签" hint="逗号分隔"><input value={form.tags} onChange={(event) => set("tags", event.target.value)} /></Field>{["huggingface", "git"].includes(dataset.source_type) && <Field label="Revision / Branch" hint={hfFrozen ? "已固定为远端 commit" : undefined}><input value={form.revision} disabled={hfFrozen} onChange={(event) => set("revision", event.target.value)} /></Field>}{dataset.source_type === "huggingface" && <><Field label="Hub 端点" hint="新端点需要重新创建接入计划"><input value={configuredEndpoint || "部署默认（HF_ENDPOINT）"} disabled /></Field><Field label="Config（逻辑元数据）"><input value={form.config} disabled /></Field><Field label="Split（逻辑元数据）"><input value={form.split} disabled /></Field><Alert severity="info" title="物理文件选择不可在元数据编辑中修改" detail="请通过“新数据”向导重新解析仓库并创建新的不可变 HF 数据资产。" /></>}<ErrorMessage error={update.error} /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button type="submit" className="button primary" disabled={!canSave || update.isPending}>{update.isPending ? "保存中…" : "保存修改"}</button></div></form></Modal>;
}

function DatasetDeleteModal({ dataset, onClose, onDeleted }: { dataset: Dataset; onClose: () => void; onDeleted: () => void }) {
  const [confirmation, setConfirmation] = useState("");
  const remove = useMutation({ mutationFn: () => api<void>(`/datasets/${dataset.id}`, { method: "DELETE" }), onSuccess: onDeleted });
  return <Modal title="删除数据集" description="删除数据集记录、版本和标注；独立任务日志与内容寻址制品不会被删除。" onClose={onClose}><div className="form-grid"><Alert severity="warning" title="存在活动任务或下游引用时会阻止删除" detail="请先移除数据混合、Tokenizer 或训练计划中的引用。" /><Field label={`输入“${dataset.name}”确认`}><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} autoFocus /></Field><ErrorMessage error={remove.error} /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button type="button" className="button danger" disabled={confirmation !== dataset.name || remove.isPending} onClick={() => remove.mutate()}>{remove.isPending ? "删除中…" : "确认删除"}</button></div></div></Modal>;
}

function DatasetOverview({ version }: { version: any }) {
  const language = version.quality_report?.language_distribution || {};
  const lengths = version.quality_report?.length_distribution || {};
  return <>
    <section className="drawer-metrics"><div><span>原始大小</span><strong>{formatBytes(version.raw_size_bytes)}</strong></div><div><span>文档数</span><strong>{formatNumber(version.document_count)}</strong></div><div><span>Token 估算</span><strong>{formatNumber(version.token_estimate)}</strong></div><div><span>自动质量分</span><strong>{version.automatic_quality_score ?? "—"}</strong></div></section>
    <DistributionPanel title="语言分布" distribution={language} empty="导入尚未生成语言分布。" />
    <DistributionPanel title="长度分布" distribution={lengths} empty="导入尚未生成长度分布。" />
    <Panel title="解析与使用契约"><dl className="detail-list"><div><dt>数据模态</dt><dd>{modalityLabels[version.modality || "text"]}</dd></div><div><dt>解析器</dt><dd>{version.parser_id ? `${version.parser_id}@${version.parser_version}` : "尚未接入"}</dd></div><div><dt>解析状态</dt><dd><StatusBadge status={version.parser_status || "pending"} /></dd></div><div><dt>识别格式</dt><dd>{(version.detected_formats || []).join("、") || "—"}</dd></div><div><dt>使用能力</dt><dd>{(version.usage_capabilities || []).join("、") || "无"}</dd></div><div><dt>版本状态</dt><dd><StatusBadge status={version.state} /></dd></div><div><dt>可用于文本训练</dt><dd><StatusBadge status={version.train_ready ? "ready" : "blocked"} /></dd></div><div><dt>制品</dt><dd>{version.artifact_id || "导入完成后生成"}</dd></div></dl></Panel>
  </>;
}

function DistributionPanel({ title, distribution, empty }: { title: string; distribution: Record<string, number>; empty: string }) {
  return <Panel title={title}>{Object.keys(distribution).length ? <div className="distribution-bars">{Object.entries(distribution).map(([label, value]) => <Distribution key={label} label={label} value={Number(value)} />)}</div> : <div className="quiet-message">{empty}</div>}</Panel>;
}

function Distribution({ label, value }: { label: string; value: number }) {
  return <div><span>{label}<b>{value.toFixed(1)}%</b></span><i><em style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }} /></i></div>;
}

function DataViewerPanel({ version, datasetName }: { version: any; datasetName: string }) {
  if (version.state !== "ready" || !version.artifact_id) return <EmptyState title="等待数据快照就绪" description="Worker 完成导入并冻结数据后，原始二维表格会出现在这里。" />;
  return <DatasetTableViewer version={version} datasetName={datasetName} />;
}

function QualityReport({ report = {} }: { report?: any }) {
  const items = [
    ["乱码率", formatRate(report.garbled_rate)],
    ["重复率", formatRate(report.duplicate_rate)],
    ["PII 风险率", formatRate(report.pii_rate)],
    ["广告/模板率", formatRate(report.ad_template_rate)],
    ["教育价值", formatCheck(report.education_value)],
    ["事实密度", formatCheck(report.factual_density)],
    ["代码质量", formatCheck(report.code_quality)],
    ["许可证检查", report.license_check === "declared_only" ? "仅核对声明" : formatCheck(report.license_check)],
    ["评测污染", formatCheck(report.contamination)],
  ];
  if (!Object.keys(report).length) return <EmptyState title="等待质量分析" description="质量值只会来自真实导入分析；未运行时不会显示为 0%。" />;
  return <>
    <Panel title="自动质量报告" description={`已分析 ${formatNumber(report.analyzed_document_count)} 个文档；自动结果与人工评分分开保存。`}><div className="quality-grid">{items.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></Panel>
    <DistributionPanel title="语言检测" distribution={report.language_distribution || {}} empty="未生成语言结果。" />
    <DistributionPanel title="长度检测" distribution={report.length_distribution || {}} empty="未生成长度结果。" />
    <Panel title="文件解析覆盖"><dl className="detail-list"><div><dt>已解析文件</dt><dd>{formatNumber(report.parsed_file_count)}</dd></div><div><dt>未解析文件</dt><dd>{report.unsupported_files?.length ? report.unsupported_files.join("；") : "无"}</dd></div><div><dt>评分方法</dt><dd>{report.quality_score_method || "未登记"}</dd></div></dl></Panel>
  </>;
}

function VersionHistory({ versions, task }: { versions: any[]; task?: Task }) {
  return <><div className="version-timeline">{versions.map((version) => <article key={version.id}><i /><div><span>{String(version.tier).toUpperCase()} V{version.version}</span><strong>{version.state}</strong><small>{formatNumber(version.document_count)} 文档 · {formatNumber(version.token_estimate)} Tokens</small>{Boolean(version.analysis_revisions?.length) && <small>{version.analysis_revisions.map((analysis: any) => `A${analysis.revision} ${analysis.status}${analysis.parser_id ? ` · ${analysis.parser_id}@${analysis.parser_version}` : ""}`).join("；")}</small>}</div><StatusBadge status={version.train_ready ? "ready" : version.state} /></article>)}</div>{task && <Panel title="当前分析任务"><dl className="detail-list"><div><dt>任务 ID</dt><dd>{task.id}</dd></div><div><dt>执行器</dt><dd>{task.adapter_id}</dd></div><div><dt>状态</dt><dd><StatusBadge status={task.observed_state} /></dd></div><div><dt>阶段</dt><dd>{task.current_phase || "—"}</dd></div></dl></Panel>}</>;
}

function SourcePanel({ dataset, version }: { dataset: Dataset; version: any }) {
  const snapshot = version.source_snapshot || {};
  const toolName = snapshot.transfer?.tool || transferTool(dataset.source_type, dataset.source_config?.hf_endpoint);
  const tool = snapshot.transfer?.version ? `${toolName} · ${snapshot.transfer.version}` : toolName;
  return <><Panel title="来源快照"><dl className="detail-list"><div><dt>来源类型</dt><dd>{dataset.source_type}</dd></div><div><dt>执行工具</dt><dd>{tool}</dd></div><div><dt>地址</dt><dd>{dataset.source_uri}</dd></div><div><dt>许可证</dt><dd>{dataset.license}</dd></div><div><dt>获取状态</dt><dd>{snapshot.fetched ? "已真实获取" : "尚未获取"}</dd></div><div><dt>获取时间</dt><dd>{snapshot.fetched_at ? new Date(snapshot.fetched_at).toLocaleString("zh-CN") : "—"}</dd></div><div><dt>文件数</dt><dd>{formatNumber(snapshot.file_count)}</dd></div><div><dt>总大小</dt><dd>{formatBytes(snapshot.raw_size_bytes)}</dd></div></dl></Panel><Panel title="冻结配置"><pre className="json-preview">{JSON.stringify(dataset.source_config || {}, null, 2)}</pre></Panel><Panel title="文件与校验和">{snapshot.files?.length ? <div className="source-file-list">{snapshot.files.slice(0, 100).map((file: any) => <article key={file.path}><strong>{file.path}</strong><span>{formatBytes(file.size_bytes)}</span><code>{file.sha256 || "下载前未计算"}</code></article>)}</div> : <div className="quiet-message">导入完成后显示实际文件和 SHA-256。</div>}</Panel></>;
}

function DataConversionPanel({ recipes, runs, loading, onCreateRecipe, onVersionRecipe, onCreateRun }: { recipes: DataConversionRecipe[]; runs: DataConversionRun[]; loading: boolean; onCreateRecipe: () => void; onVersionRecipe: (recipe: DataConversionRecipe) => void; onCreateRun: () => void }) {
  if (loading) return <Panel><div className="quiet-message">正在读取转换配方和运行…</div></Panel>;
  return <div className="processing-layout">
    <section>
      <div className="section-heading"><div><h2>转换配方</h2><p>配方冻结格式识别、文档抽取和文本加工算子的执行顺序。</p></div><button className="button secondary" onClick={onCreateRecipe}><Plus size={16} />新建转换配方</button></div>
      {recipes.length ? <div className="processing-recipe-grid">{recipes.map((recipe) => { const version = recipe.current_version || {}; return <Panel key={recipe.id} title={recipe.name} description={recipe.description || "未填写说明"} actions={<StatusBadge status="ready" />}><div className="processing-recipe-meta"><span>当前版本 <b>V{version.version}</b></span><span>执行引擎 <b>内置文档转换</b></span><span>算子 <b>{version.operators?.length || 0}</b></span><span>输出 <b>文本 JSONL</b></span></div><div className="operator-chips">{(version.operators || []).map((operator: any, index: number) => <code key={`${Object.keys(operator)[0]}-${index}`}>{Object.keys(operator)[0]}</code>)}</div><footer><button type="button" className="button ghost small" onClick={() => onVersionRecipe(recipe)}><FileJson size={14} />创建新版本</button></footer></Panel>; })}</div> : <EmptyState title="还没有转换配方" description="先组合格式识别、PDF/DOCX/EPUB 抽取和文本加工算子。转换不会在导入时自动执行。" action={<button className="button primary" onClick={onCreateRecipe}><Plus size={16} />新建转换配方</button>} />}
    </section>
    <section>
      <div className="section-heading"><div><h2>转换运行</h2><p>显式选择原始版本；成功后才创建并切换到新的 Bronze 文本版本。</p></div><button className="button primary" onClick={onCreateRun} disabled={!recipes.length}><Workflow size={16} />发起转换</button></div>
      {runs.length ? <Panel className="table-panel"><div className="data-table-wrap"><table className="data-table"><thead><tr><th>数据集 / 配方</th><th>版本流转</th><th>转换结果</th><th>引擎</th><th>状态</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.dataset_name || "未知数据集"}</strong><span>{run.recipe_name || run.recipe_version_id} · V{run.recipe_version?.version || "—"}</span></td><td><b>BRONZE V{run.input_version?.version || "—"} → BRONZE V{run.output_version?.version || "—"}</b></td><td>{run.status === "succeeded" ? `${formatNumber(run.summary?.converted_file_count || 0)} 文件 · ${formatNumber(run.summary?.output_record_count || run.output_version?.document_count)} 记录` : "等待完成"}</td><td>{run.engine_version || "document-conversion-v1"}</td><td><StatusBadge status={run.status} />{run.failure_reason && <small className="processing-error">{run.failure_reason}</small>}</td></tr>)}</tbody></table></div></Panel> : <EmptyState title="还没有转换运行" description={recipes.length ? "选择原始数据版本和转换配方，产出文本 JSONL 新版本。" : "创建转换配方后才能发起运行。"} action={recipes.length ? <button className="button primary" onClick={onCreateRun}><Workflow size={16} />发起转换</button> : undefined} />}
      <Alert severity="info" title="导入与转换保持分离" detail="PDF、DOCX、EPUB 先原样进入原始 Bronze 版本；这里只处理用户显式发起的转换。文本型 PDF 可抽取，扫描 PDF 暂不做 OCR。" />
    </section>
  </div>;
}

function conversionDefaultParameters(operator: DataConversionOperatorCatalog["operators"][number]) {
  return Object.fromEntries(Object.entries(operator.parameters || {}).map(([name, schema]) => [name, schema.default]));
}

function defaultConversionOperators(catalog: DataConversionOperatorCatalog) {
  return catalog.operators.filter((operator) => operator.name !== "chunk_text").map((operator) => ({ [operator.name]: conversionDefaultParameters(operator) }));
}

function DataConversionRecipeModal({ projectId, recipe, catalog, onClose, onCreated }: { projectId: string; recipe?: DataConversionRecipe; catalog: DataConversionOperatorCatalog; onClose: () => void; onCreated: () => void }) {
  const initial = recipe?.current_version;
  const defaults = initial?.operators || defaultConversionOperators(catalog);
  const [validationError, setValidationError] = useState<Error | null>(null);
  const [editorMode, setEditorMode] = useState<"visual" | "json">("visual");
  const [operatorToAdd, setOperatorToAdd] = useState("");
  const [steps, setSteps] = useState<OperatorStep[]>(() => operatorStepsFromValue(defaults));
  const [advancedJson, setAdvancedJson] = useState(() => JSON.stringify(defaults, null, 2));
  const groups = Array.from(new Set(catalog.operators.map((operator) => operator.category)));
  const create = useMutation({ mutationFn: (payload: any) => recipe ? post(`/data-conversion-recipes/${recipe.id}/versions`, payload) : post(`/projects/${projectId}/data-conversion-recipes`, payload), onSuccess: onCreated });
  const entryFor = (name: string) => catalog.operators.find((operator) => operator.name === name);
  function addOperator(name: string) {
    const entry = entryFor(name);
    if (!entry) return;
    setSteps((current) => [...current, { key: nextOperatorStepKey(), name, parametersJson: JSON.stringify(conversionDefaultParameters(entry), null, 2) }]);
    setOperatorToAdd("");
    setValidationError(null);
  }
  function replaceOperator(key: string, name: string) {
    const entry = entryFor(name);
    if (!entry) return;
    setSteps((current) => current.map((step) => step.key === key ? { ...step, name, parametersJson: JSON.stringify(conversionDefaultParameters(entry), null, 2) } : step));
  }
  function moveOperator(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= steps.length) return;
    setSteps((current) => { const reordered = [...current]; [reordered[index], reordered[target]] = [reordered[target], reordered[index]]; return reordered; });
  }
  function toJsonMode() {
    try {
      const operators = operatorsFromSteps(steps);
      setAdvancedJson(JSON.stringify(operators, null, 2));
      setEditorMode("json");
      setValidationError(null);
    } catch (error) { setValidationError(error instanceof Error ? error : new Error("算子配置无效")); }
  }
  function toVisualMode() {
    try {
      setSteps(operatorStepsFromValue(JSON.parse(advancedJson)));
      setEditorMode("visual");
      setValidationError(null);
    } catch (error) { setValidationError(error instanceof Error ? error : new Error("完整算子 JSON 无效")); }
  }
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    try {
      const operators = editorMode === "visual" ? operatorsFromSteps(steps) : normalizeDataJuicerOperators(JSON.parse(advancedJson));
      setValidationError(null);
      create.mutate({ ...(!recipe ? { name: String(form.get("name") || "").trim(), description: String(form.get("description") || "").trim() } : {}), engine_adapter_id: "builtin.data-conversion", operators, execution_config: {} });
    } catch (error) { setValidationError(error instanceof Error ? error : new Error("算子配置必须是合法 JSON")); }
  }
  return <Modal title={recipe ? `创建 ${recipe.name} 新版本` : "新建转换配方"} description="按顺序组合格式识别、文档抽取与轻量文本加工；运行时会冻结完整配方。" onClose={onClose} wide><form className="form-grid" onSubmit={submit}>
    {!recipe && <div className="form-grid two-columns"><Field label="转换配方名称"><input name="name" required maxLength={180} placeholder="文档统一转文本" /></Field><Field label="说明"><input name="description" maxLength={4000} placeholder="识别 PDF、DOCX、EPUB 并输出 JSONL" /></Field></div>}
    <div className="operator-editor">
      <div className="operator-editor-heading"><div><strong>转换步骤</strong><span>识别必须位于首步，抽取算子必须位于加工算子之前。</span></div><div className="operator-mode-toggle"><button type="button" className={editorMode === "visual" ? "active" : ""} aria-pressed={editorMode === "visual"} onClick={toVisualMode}><Workflow size={14} />可视化</button><button type="button" className={editorMode === "json" ? "active" : ""} aria-pressed={editorMode === "json"} onClick={toJsonMode}><Code2 size={14} />高级 JSON</button></div></div>
      {editorMode === "visual" ? <><Field label="添加转换算子"><select aria-label="添加转换算子" value={operatorToAdd} onChange={(event) => { setOperatorToAdd(event.target.value); addOperator(event.target.value); }}><option value="">选择后添加到末尾…</option>{groups.map((group) => <optgroup key={group} label={group}>{catalog.operators.map((entry) => entry.category === group && <option key={entry.name} value={entry.name} disabled={steps.some((step) => step.name === entry.name)}>{entry.label} · {entry.name}</option>)}</optgroup>)}</select></Field><div className="operator-step-list">{steps.map((step, index) => { const entry = entryFor(step.name); return <article className="operator-step" key={step.key}><header><div><i>{index + 1}</i><span><strong>{entry?.label || "未知算子"}</strong><code>{step.name}</code></span></div><div className="operator-step-actions"><button type="button" className="icon-button" aria-label={`上移转换步骤 ${index + 1}`} disabled={index === 0 || step.name === "detect_format" || steps[index - 1]?.name === "detect_format"} onClick={() => moveOperator(index, -1)}><ChevronUp size={15} /></button><button type="button" className="icon-button" aria-label={`下移转换步骤 ${index + 1}`} disabled={index === steps.length - 1 || step.name === "detect_format"} onClick={() => moveOperator(index, 1)}><ChevronDown size={15} /></button><button type="button" className="icon-button danger" aria-label={`删除转换步骤 ${index + 1}`} disabled={steps.length === 1 || step.name === "detect_format"} onClick={() => setSteps((current) => current.filter((item) => item.key !== step.key))}><Trash2 size={15} /></button></div></header><div className="operator-step-grid"><Field label={`转换步骤 ${index + 1} 算子`}><select aria-label={`转换步骤 ${index + 1} 算子`} value={step.name} disabled={step.name === "detect_format"} onChange={(event) => replaceOperator(step.key, event.target.value)}>{catalog.operators.map((item) => <option key={item.name} value={item.name} disabled={item.name !== step.name && steps.some((current) => current.name === item.name)}>{item.label} · {item.name}</option>)}</select></Field><Field label={`转换步骤 ${index + 1} 参数 JSON`} hint="参数必须匹配服务端算子定义"><textarea aria-label={`转换步骤 ${index + 1} 参数 JSON`} value={step.parametersJson} onChange={(event) => setSteps((current) => current.map((item) => item.key === step.key ? { ...item, parametersJson: event.target.value } : item))} /></Field></div><p>{entry?.description || "服务端不支持这个算子。"}</p></article>; })}</div></> : <Field label="完整转换算子 JSON" hint="服务端会校验算子白名单、参数和执行顺序。"><textarea aria-label="完整转换算子 JSON" className="recipe-editor" value={advancedJson} onChange={(event) => setAdvancedJson(event.target.value)} required /></Field>}
    </div>
    <ErrorMessage error={validationError || create.error} />
    <FormActions submitting={create.isPending} onCancel={onClose} submitLabel={recipe ? "创建新版本" : "创建转换配方"} />
  </form></Modal>;
}

function DataConversionRunModal({ projectId, datasets, recipes, onClose, onCreated }: { projectId: string; datasets: Dataset[]; recipes: DataConversionRecipe[]; onClose: () => void; onCreated: () => void }) {
  const inputs = datasets.flatMap((dataset) => (dataset.versions || []).filter((version: any) => version.state === "ready" && version.artifact_id).map((version: any) => ({ ...version, dataset_name: dataset.name })));
  const recipeVersions = recipes.flatMap((recipe) => (recipe.versions || []).map((version: any) => ({ ...version, recipe_name: recipe.name })));
  const create = useMutation({ mutationFn: (payload: any) => post(`/projects/${projectId}/data-conversion-runs`, payload), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    create.mutate({ recipe_version_id: form.get("recipe_version_id"), input_dataset_version_id: form.get("input_dataset_version_id"), resource_request: { cpus: Number(form.get("cpus") || 0), gpus: 0 } });
  }
  return <Modal title="发起数据转换" description="转换是显式步骤：输入原始快照保持不变，成功后新增文本 JSONL 版本并切换当前版本。" onClose={onClose}>{inputs.length && recipeVersions.length ? <form className="form-grid" onSubmit={submit}><Field label="原始数据版本"><select name="input_dataset_version_id" required>{inputs.map((version) => <option key={version.id} value={version.id}>{version.dataset_name} · {String(version.tier).toUpperCase()} V{version.version} · {version.train_ready ? "已是文本" : "待转换"}</option>)}</select></Field><Field label="转换配方版本"><select name="recipe_version_id" required>{recipeVersions.map((version) => <option key={version.id} value={version.id}>{version.recipe_name} · V{version.version} · {version.operators?.length || 0} 算子</option>)}</select></Field><Field label="CPU 请求"><input name="cpus" type="number" min="0" defaultValue={0} /></Field><Alert severity="warning" title="转换不会覆盖原始版本" detail="运行失败时当前版本保持不变；成功时输出新的 Bronze 文本版本，之后再进入数据清洗。" /><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="创建转换运行" /></form> : <><EmptyState title="没有可转换的输入或配方" description="输入需要处于 ready 且具有可用 Snapshot；项目中还需要至少一个转换配方版本。" /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>关闭</button></div></>}</Modal>;
}

function DataProcessingPanel({ recipes, runs, loading, onCreateRecipe, onVersionRecipe, onCreateRun }: { recipes: DataProcessingRecipe[]; runs: DataProcessingRun[]; loading: boolean; onCreateRecipe: () => void; onVersionRecipe: (recipe: DataProcessingRecipe) => void; onCreateRun: () => void }) {
  if (loading) return <Panel><div className="quiet-message">正在读取处理配方和运行…</div></Panel>;
  return <div className="processing-layout">
    <section>
      <div className="section-heading"><div><h2>清洗配方</h2><p>配方版本冻结 Data-Juicer 算子和并行度；修改会创建新版本。</p></div><button className="button secondary" onClick={onCreateRecipe}><Plus size={16} />新建清洗配方</button></div>
      {recipes.length ? <div className="processing-recipe-grid">{recipes.map((recipe) => { const version = recipe.current_version || {}; return <Panel key={recipe.id} title={recipe.name} description={recipe.description || "未填写说明"} actions={<StatusBadge status="ready" />}><div className="processing-recipe-meta"><span>当前版本 <b>V{version.version}</b></span><span>执行引擎 <b>Data-Juicer</b></span><span>算子 <b>{version.operators?.length || 0}</b></span><span>并行进程 <b>{version.execution_config?.np || 1}</b></span></div><div className="operator-chips">{(version.operators || []).slice(0, 8).map((operator: any, index: number) => <code key={`${Object.keys(operator)[0]}-${index}`}>{Object.keys(operator)[0]}</code>)}</div><footer><button type="button" className="button ghost small" onClick={() => onVersionRecipe(recipe)}><FileJson size={14} />创建新版本</button></footer></Panel>; })}</div> : <EmptyState title="还没有清洗配方" description="先创建一个不可变 Data-Juicer 算子配方，再从转换后的 Bronze/Silver 文本版本发起清洗。" action={<button className="button primary" onClick={onCreateRecipe}><Plus size={16} />新建清洗配方</button>} />}
    </section>
    <section>
      <div className="section-heading"><div><h2>清洗运行</h2><p>运行冻结转换后的文本输入和配方，成功后才把新的 Silver/Gold 版本设为当前版本。</p></div><button className="button primary" onClick={onCreateRun} disabled={!recipes.length}><Workflow size={16} />发起清洗</button></div>
      {runs.length ? <Panel className="table-panel"><div className="data-table-wrap"><table className="data-table"><thead><tr><th>数据集 / 配方</th><th>版本流转</th><th>文档变化</th><th>引擎</th><th>状态</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.dataset_name || "未知数据集"}</strong><span>{run.recipe_name || run.recipe_version_id} · V{run.recipe_version?.version || "—"}</span></td><td><b>{String(run.input_version?.tier || "—").toUpperCase()} V{run.input_version?.version || "—"} → {run.output_tier.toUpperCase()} V{run.output_version?.version || "—"}</b></td><td>{run.status === "succeeded" ? `${formatSigned(run.summary?.document_delta || 0)} · 保留 ${formatNumber(run.summary?.output_document_count)}` : "等待完成"}</td><td>{run.engine_version ? `Data-Juicer ${run.engine_version}` : "Data-Juicer"}</td><td><StatusBadge status={run.status} />{run.failure_reason && <small className="processing-error">{run.failure_reason}</small>}</td></tr>)}</tbody></table></div></Panel> : <EmptyState title="还没有清洗运行" description={recipes.length ? "选择文本数据版本和配方，生成第一个 Silver 或 Gold 版本。" : "创建配方后才能发起数据清洗。"} action={recipes.length ? <button className="button primary" onClick={onCreateRun}><Workflow size={16} />发起清洗</button> : undefined} />}
      <Alert severity="info" title="执行能力按 Worker 显式声明" detail={'目标 Worker 需要安装项目可选依赖 ".[data-juicer]"。首版只调用单机 dj-process，不宣称 Ray、多机、暂停或断点恢复。'} />
    </section>
  </div>;
}

const DEFAULT_DATA_JUICER_OPERATORS = [
  { whitespace_normalization_mapper: {} },
  { text_length_filter: { min_len: 10, max_len: 100000 } },
];

type DataJuicerOperator = Record<string, Record<string, unknown> | null>;
type OperatorStep = { key: string; name: string; parametersJson: string };
type OperatorCatalogEntry = {
  name: string;
  label: string;
  group: "规范化与清理" | "质量过滤" | "去重";
  description: string;
  defaultParameters: Record<string, unknown>;
};

const DATA_JUICER_OPERATOR_GROUPS = ["规范化与清理", "质量过滤", "去重"] as const;
const DATA_JUICER_OPERATOR_CATALOG: OperatorCatalogEntry[] = [
  { name: "whitespace_normalization_mapper", label: "空白规范化", group: "规范化与清理", description: "把不同空白字符规范为标准空格。", defaultParameters: {} },
  { name: "punctuation_normalization_mapper", label: "标点规范化", group: "规范化与清理", description: "把 Unicode 标点规范为一致形式。", defaultParameters: {} },
  { name: "fix_unicode_mapper", label: "修复 Unicode", group: "规范化与清理", description: "修复常见 Unicode 文本错误。", defaultParameters: {} },
  { name: "clean_html_mapper", label: "清理 HTML", group: "规范化与清理", description: "移除 HTML 标记并保留纯文本。", defaultParameters: {} },
  { name: "clean_email_mapper", label: "清理邮箱", group: "规范化与清理", description: "移除文本中的邮箱地址。", defaultParameters: {} },
  { name: "clean_ip_mapper", label: "清理 IP", group: "规范化与清理", description: "移除 IPv4 和 IPv6 地址。", defaultParameters: {} },
  { name: "clean_links_mapper", label: "清理链接", group: "规范化与清理", description: "移除 HTTP、HTTPS 和 FTP 链接。", defaultParameters: {} },
  { name: "remove_repeat_sentences_mapper", label: "移除重复句子", group: "规范化与清理", description: "删除单个样本内重复出现的句子。", defaultParameters: {} },
  { name: "text_length_filter", label: "文本长度过滤", group: "质量过滤", description: "只保留总文本长度位于指定区间的样本。", defaultParameters: { min_len: 10, max_len: 100000 } },
  { name: "special_characters_filter", label: "特殊字符比例过滤", group: "质量过滤", description: "按特殊字符比例过滤样本。", defaultParameters: {} },
  { name: "character_repetition_filter", label: "字符重复过滤", group: "质量过滤", description: "按字符级 n-gram 重复比例过滤样本。", defaultParameters: {} },
  { name: "word_repetition_filter", label: "词语重复过滤", group: "质量过滤", description: "按词级 n-gram 重复比例过滤样本。", defaultParameters: {} },
  { name: "document_deduplicator", label: "文档精确去重", group: "去重", description: "使用完全匹配删除重复文档。", defaultParameters: {} },
  { name: "document_minhash_deduplicator", label: "文档 MinHash 去重", group: "去重", description: "使用 MinHash LSH 删除近似重复文档。", defaultParameters: {} },
];

let dataJuicerOperatorStepSequence = 0;

function nextOperatorStepKey() {
  dataJuicerOperatorStepSequence += 1;
  return `data-juicer-operator-${dataJuicerOperatorStepSequence}`;
}

function normalizeDataJuicerOperators(value: unknown): DataJuicerOperator[] {
  if (!Array.isArray(value) || !value.length) throw new Error("配方至少需要一个算子");
  return value.map((operator, index) => {
    if (!operator || typeof operator !== "object" || Array.isArray(operator)) throw new Error(`第 ${index + 1} 个算子必须是 JSON 对象`);
    const entries = Object.entries(operator as Record<string, unknown>);
    if (entries.length !== 1) throw new Error(`第 ${index + 1} 个算子只能包含一个算子名`);
    const [name, parameters] = entries[0];
    if (!/^[a-z][a-z0-9_]*$/.test(name)) throw new Error(`第 ${index + 1} 个算子名称不合法`);
    if (parameters !== null && (typeof parameters !== "object" || Array.isArray(parameters))) throw new Error(`第 ${index + 1} 个算子参数必须是 JSON 对象或 null`);
    return { [name]: parameters as Record<string, unknown> | null };
  });
}

function operatorStepsFromValue(value: unknown): OperatorStep[] {
  return normalizeDataJuicerOperators(value).map((operator) => {
    const [name, parameters] = Object.entries(operator)[0];
    return { key: nextOperatorStepKey(), name, parametersJson: JSON.stringify(parameters, null, 2) };
  });
}

function operatorsFromSteps(steps: OperatorStep[]): DataJuicerOperator[] {
  return normalizeDataJuicerOperators(steps.map((step, index) => {
    let parameters: unknown;
    try {
      parameters = JSON.parse(step.parametersJson || "{}");
    } catch {
      throw new Error(`步骤 ${index + 1} 的参数不是合法 JSON`);
    }
    return { [step.name]: parameters };
  }));
}

function catalogEntry(name: string) {
  return DATA_JUICER_OPERATOR_CATALOG.find((entry) => entry.name === name);
}

function DataProcessingRecipeModal({ projectId, recipe, onClose, onCreated }: { projectId: string; recipe?: DataProcessingRecipe; onClose: () => void; onCreated: () => void }) {
  const initial = recipe?.current_version;
  const [validationError, setValidationError] = useState<Error | null>(null);
  const [editorMode, setEditorMode] = useState<"visual" | "json">("visual");
  const [operatorToAdd, setOperatorToAdd] = useState("");
  const [steps, setSteps] = useState<OperatorStep[]>(() => operatorStepsFromValue(initial?.operators || DEFAULT_DATA_JUICER_OPERATORS));
  const [advancedJson, setAdvancedJson] = useState(() => JSON.stringify(initial?.operators || DEFAULT_DATA_JUICER_OPERATORS, null, 2));
  const create = useMutation({
    mutationFn: (payload: any) => recipe ? post(`/data-processing-recipes/${recipe.id}/versions`, payload) : post(`/projects/${projectId}/data-processing-recipes`, payload),
    onSuccess: onCreated,
  });
  function addOperator(name: string) {
    const entry = catalogEntry(name);
    if (!entry) return;
    setSteps((current) => [...current, { key: nextOperatorStepKey(), name, parametersJson: JSON.stringify(entry.defaultParameters, null, 2) }]);
    setOperatorToAdd("");
    setValidationError(null);
  }
  function replaceOperator(key: string, name: string) {
    const entry = catalogEntry(name);
    setSteps((current) => current.map((step) => step.key === key ? { ...step, name, parametersJson: entry ? JSON.stringify(entry.defaultParameters, null, 2) : step.parametersJson } : step));
  }
  function moveOperator(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= steps.length) return;
    setSteps((current) => {
      const reordered = [...current];
      [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
      return reordered;
    });
  }
  function showAdvancedJson() {
    try {
      const operators = operatorsFromSteps(steps);
      setAdvancedJson(JSON.stringify(operators, null, 2));
      setEditorMode("json");
      setValidationError(null);
    } catch (error) {
      setValidationError(error instanceof Error ? error : new Error("算子配置无效"));
    }
  }
  function showVisualEditor() {
    try {
      const operators = normalizeDataJuicerOperators(JSON.parse(advancedJson));
      setSteps(operatorStepsFromValue(operators));
      setEditorMode("visual");
      setValidationError(null);
    } catch (error) {
      setValidationError(error instanceof Error ? error : new Error("完整算子 JSON 无效"));
    }
  }
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    let operators: DataJuicerOperator[];
    try {
      operators = editorMode === "visual" ? operatorsFromSteps(steps) : normalizeDataJuicerOperators(JSON.parse(advancedJson));
    } catch (error) {
      setValidationError(error instanceof Error ? error : new Error("算子配置必须是合法 JSON"));
      return;
    }
    setValidationError(null);
    create.mutate({
      ...(!recipe ? { name: String(form.get("name") || "").trim(), description: String(form.get("description") || "").trim() } : {}),
      engine_adapter_id: "builtin.data-juicer",
      operators,
      execution_config: { np: form.get("np") === "auto" ? "auto" : Number(form.get("np")) },
    });
  }
  return <Modal title={recipe ? `创建 ${recipe.name} 新版本` : "新建清洗配方"} description="从常用目录选择算子并排列步骤；输入、输出路径仍由 Worker 安全生成。" onClose={onClose} wide><form className="form-grid" onSubmit={submit}>
    {!recipe && <div className="form-grid two-columns"><Field label="配方名称"><input name="name" required maxLength={180} placeholder="基础文本清洗" /></Field><Field label="说明"><input name="description" maxLength={4000} placeholder="规范空白并过滤异常长度文本" /></Field></div>}
    <Field label="并行进程数" hint="auto 按任务 CPU 配额和可用内存自动选择，也可填写 1–256；手动值受实际配额上限约束"><input name="np" pattern="auto|[0-9]+" defaultValue={initial?.execution_config?.np ?? "auto"} required /></Field>
    <div className="operator-editor">
      <div className="operator-editor-heading"><div><strong>处理步骤</strong><span>顺序会原样写入 Data-Juicer process 配置。</span></div><div className="operator-mode-toggle"><button type="button" className={editorMode === "visual" ? "active" : ""} aria-pressed={editorMode === "visual"} onClick={showVisualEditor}><Workflow size={14} />可视化</button><button type="button" className={editorMode === "json" ? "active" : ""} aria-pressed={editorMode === "json"} onClick={showAdvancedJson}><Code2 size={14} />高级 JSON</button></div></div>
      {editorMode === "visual" ? <>
        <Field label="添加算子" hint="这里只列出常用稳定文本算子；目录外算子可使用高级 JSON。"><select aria-label="添加算子" value={operatorToAdd} onChange={(event) => { setOperatorToAdd(event.target.value); addOperator(event.target.value); }}><option value="">选择后添加到末尾…</option>{DATA_JUICER_OPERATOR_GROUPS.map((group) => <optgroup key={group} label={group}>{DATA_JUICER_OPERATOR_CATALOG.filter((entry) => entry.group === group).map((entry) => <option key={entry.name} value={entry.name}>{entry.label} · {entry.name}</option>)}</optgroup>)}</select></Field>
        <div className="operator-step-list">{steps.map((step, index) => { const entry = catalogEntry(step.name); return <article className="operator-step" key={step.key}>
          <header><div><i>{index + 1}</i><span><strong>{entry?.label || "自定义算子"}</strong><code>{step.name}</code></span></div><div className="operator-step-actions"><button type="button" className="icon-button" aria-label={`上移步骤 ${index + 1}`} disabled={index === 0} onClick={() => moveOperator(index, -1)}><ChevronUp size={15} /></button><button type="button" className="icon-button" aria-label={`下移步骤 ${index + 1}`} disabled={index === steps.length - 1} onClick={() => moveOperator(index, 1)}><ChevronDown size={15} /></button><button type="button" className="icon-button danger" aria-label={`删除步骤 ${index + 1}`} disabled={steps.length === 1} onClick={() => setSteps((current) => current.filter((item) => item.key !== step.key))}><Trash2 size={15} /></button></div></header>
          <div className="operator-step-grid"><Field label={`步骤 ${index + 1} 算子`}><select aria-label={`步骤 ${index + 1} 算子`} value={step.name} onChange={(event) => replaceOperator(step.key, event.target.value)}>{!entry && <option value={step.name}>自定义 · {step.name}</option>}{DATA_JUICER_OPERATOR_GROUPS.map((group) => <optgroup key={group} label={group}>{DATA_JUICER_OPERATOR_CATALOG.filter((item) => item.group === group).map((item) => <option key={item.name} value={item.name}>{item.label} · {item.name}</option>)}</optgroup>)}</select></Field><Field label={`步骤 ${index + 1} 参数 JSON`} hint="参数必须是 JSON 对象或 null"><textarea aria-label={`步骤 ${index + 1} 参数 JSON`} value={step.parametersJson} onChange={(event) => setSteps((current) => current.map((item) => item.key === step.key ? { ...item, parametersJson: event.target.value } : item))} /></Field></div>
          <p>{entry?.description || "该算子不在当前策划目录中；运行时由目标 Worker 的 Data-Juicer 版本校验。"}</p>
        </article>; })}</div>
      </> : <Field label="完整算子 JSON" hint="数组中每项只能包含一个算子名和参数对象；可使用目录外算子。"><textarea aria-label="完整算子 JSON" className="recipe-editor" value={advancedJson} onChange={(event) => setAdvancedJson(event.target.value)} required /></Field>}
    </div>
    <ErrorMessage error={validationError || create.error} />
    <FormActions submitting={create.isPending} onCancel={onClose} submitLabel={recipe ? "创建新版本" : "创建清洗配方"} />
  </form></Modal>;
}

function DataProcessingRunModal({ projectId, datasets, recipes, onClose, onCreated }: { projectId: string; datasets: Dataset[]; recipes: DataProcessingRecipe[]; onClose: () => void; onCreated: () => void }) {
  const inputs = datasets.flatMap((dataset) => (dataset.versions || []).filter((version: any) => version.state === "ready" && version.train_ready && version.artifact_id && version.tier !== "gold").map((version: any) => ({ ...version, dataset_name: dataset.name })));
  const recipeVersions = recipes.flatMap((recipe) => (recipe.versions || []).map((version: any) => ({ ...version, recipe_name: recipe.name })));
  const [inputId, setInputId] = useState(inputs[0]?.id || "");
  const selectedInput = inputs.find((version) => version.id === inputId);
  const [outputTier, setOutputTier] = useState<"silver" | "gold">(selectedInput?.tier === "silver" ? "gold" : "silver");
  const create = useMutation({ mutationFn: (payload: any) => post(`/projects/${projectId}/data-processing-runs`, payload), onSuccess: onCreated });
  function selectInput(id: string) {
    setInputId(id);
    const input = inputs.find((version) => version.id === id);
    setOutputTier(input?.tier === "silver" ? "gold" : "silver");
  }
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    create.mutate({ recipe_version_id: form.get("recipe_version_id"), input_dataset_version_id: inputId, output_tier: outputTier, resource_request: { cpus: Number(form.get("cpus") || 0), gpus: Number(form.get("gpus") || 0) } });
  }
  return <Modal title="发起数据清洗" description="输入和配方会被完整冻结；只有任务成功后才切换数据集当前版本。" onClose={onClose}>{inputs.length && recipeVersions.length ? <form className="form-grid" onSubmit={submit}><Field label="输入数据版本"><select value={inputId} onChange={(event) => selectInput(event.target.value)} required>{inputs.map((version) => <option key={version.id} value={version.id}>{version.dataset_name} · {String(version.tier).toUpperCase()} V{version.version} · {formatNumber(version.document_count)} 文档</option>)}</select></Field><Field label="清洗配方版本"><select name="recipe_version_id" required>{recipeVersions.map((version) => <option key={version.id} value={version.id}>{version.recipe_name} · V{version.version} · {version.operators?.length || 0} 算子</option>)}</select></Field><Field label="输出层级"><select value={outputTier} onChange={(event) => setOutputTier(event.target.value as "silver" | "gold")} disabled={selectedInput?.tier === "silver"}><option value="silver">Silver</option><option value="gold">Gold</option></select></Field><Field label="CPU 请求"><input name="cpus" type="number" min="0" defaultValue={0} /></Field><Field label="GPU 请求" hint="只有配方中的算子真实需要 GPU 时才填写"><input name="gpus" type="number" min="0" defaultValue={0} /></Field><Alert severity="warning" title="停止不会保留可恢复状态" detail="首版 Data-Juicer 适配器可以终止子进程，但不提供暂停或断点恢复。" /><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="创建清洗运行" /></form> : <><EmptyState title="没有可清洗的输入或配方" description="输入必须是具有可用 Snapshot 的 Bronze/Silver 可训练文本版本，并且项目中至少有一个清洗配方版本。" /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>关闭</button></div></>}</Modal>;
}

function MixtureList({ mixtures, onCreate, onSelect }: { mixtures: Mixture[]; onCreate: () => void; onSelect: (id: string) => void }) {
  if (!mixtures.length) return <EmptyState title="还没有数据混合" description="数据混合引用不可变数据版本，不会复制底层数据。" action={<button className="button primary" onClick={onCreate}><Plus size={16} />新建混合</button>} />;
  return <section className="mixture-grid">{mixtures.map((mixture) => { const version = mixture.current_version || {}; return <button type="button" className="mixture-card-button" key={mixture.id} onClick={() => onSelect(mixture.id)} aria-label={`打开数据混合 ${mixture.name}`}><Panel title={mixture.name} description={mixture.description} actions={<StatusBadge status={version.status || "legacy_unresolved"} />}><div className="mixture-summary"><div><span>目标 Tokens</span><strong>{formatNumber(version.target_tokens)}</strong></div><div><span>版本</span><strong>V{version.version || "—"}</strong></div><div><span>已解析文档</span><strong>{formatNumber(version.resolved_document_count)}</strong></div></div><div className="mixture-bars">{(version.entries || []).map((entry: any) => <div key={entry.dataset_version_id}><span>{entry.name || entry.dataset_version_id.slice(0, 8)}<b>{Math.round((entry.normalized_weight || entry.weight) * 100)}%</b></span><i><em style={{ width: `${(entry.normalized_weight || entry.weight) * 100}%` }} /></i></div>)}</div><div className="gate-strip"><ShieldCheck size={17} /><div><strong>稳定数据 Manifest</strong><span>{version.manifest_hash ? version.manifest_hash.slice(0, 12) : "等待解析"}</span></div><StatusBadge status={version.status || "legacy_unresolved"} /></div><span className="mixture-open-hint">打开详情与版本操作 <ArrowRight size={14} /></span></Panel></button>; })}</section>;
}

function MixtureDrawer({ projectId, mixture, datasets, globalMode = false, onClose, onChanged }: { projectId: string; mixture?: Mixture; datasets: Dataset[]; globalMode?: boolean; onClose: () => void; onChanged: () => void }) {
  const [selectedVersionId, setSelectedVersionId] = useState(mixture?.current_version?.id || "");
  const [compareVersionId, setCompareVersionId] = useState("");
  const [editOpen, setEditOpen] = useState(false);
  const versions = mixture?.versions || [];
  const version = versions.find((item: any) => item.id === selectedVersionId) || mixture?.current_version;
  const comparison = versions.find((item: any) => item.id === compareVersionId);
  if (!mixture || !version) return null;
  return <aside className="detail-drawer mixture-drawer">
    <header><button className="icon-button" onClick={onClose}><ArrowRight size={19} /></button><div className="drawer-title"><span className="eyebrow">DATA MIXTURE</span><h2>{mixture.name}</h2><p>所有修改都会创建并重新解析新版本，历史版本不会被覆盖。</p></div><div className="drawer-header-actions"><button type="button" className="button secondary small" onClick={() => setEditOpen(true)}><Pencil size={13} />创建新版本</button></div></header>
    <div className="drawer-content">
      <Alert severity="info" title="版本交付固定的数据流" detail="平台已经确定实际入选文档、整数权重、seed 和采样器版本；后续算法只接收不可变 Manifest。" />
      <div className="mixture-version-controls"><Field label="查看版本"><select value={version.id} onChange={(event) => setSelectedVersionId(event.target.value)}>{versions.map((item: any) => <option key={item.id} value={item.id}>V{item.version} · {item.status || "legacy_unresolved"}</option>)}</select></Field><Field label="对比版本"><select value={compareVersionId} onChange={(event) => setCompareVersionId(event.target.value)}><option value="">不对比</option>{versions.filter((item: any) => item.id !== version.id).map((item: any) => <option key={item.id} value={item.id}>V{item.version}</option>)}</select></Field></div>
      <div className="drawer-metrics"><div><span>目标 Tokens</span><strong>{formatNumber(version.target_tokens)}</strong></div><div><span>状态</span><strong>{version.status || "legacy_unresolved"}</strong></div><div><span>已解析文档</span><strong>{formatNumber(version.resolved_document_count)}</strong></div><div><span>固定 Seed</span><strong>{version.seed ?? "—"}</strong></div></div>
      {comparison && <Alert severity="info" title={`V${version.version} 对比 V${comparison.version}`} detail={`目标 Tokens ${formatSigned((version.target_tokens || 0) - (comparison.target_tokens || 0))}；来源数 ${formatSigned((version.entries?.length || 0) - (comparison.entries?.length || 0))}`} />}
      <Panel title="数据配方" description="每个来源先应用自己的子集筛选，再按规范化权重参与训练采样。"><div className="mixture-detail-table"><header><span>数据版本 / 子集</span><span>权重 / 比例</span><span>预计消费</span><span>重复采样</span></header>{(version.entries || []).map((entry: any) => { const ratio = Number(entry.normalized_weight ?? entry.weight ?? 0); const expected = Number(version.target_tokens || 0) * ratio; const selectedTokens = Number(entry.selected_tokens ?? selectedTokenEstimate(entry.available_tokens, entry.selection)); const repeats = selectedTokens ? expected / selectedTokens : null; const compared = comparison?.entries?.find((item: any) => item.dataset_version_id === entry.dataset_version_id); return <div key={entry.dataset_version_id}><strong>{entry.name || entry.dataset_version_id.slice(0, 8)}<small>{selectionSummary(entry.selection)} · 估算 {formatNumber(selectedTokens)} Tokens</small></strong><span>{entry.weight} / {(ratio * 100).toFixed(1)}%{compared && <small>对比 {(Number(compared.normalized_weight ?? compared.weight) * 100).toFixed(1)}%</small>}</span><span>{formatNumber(expected)}</span><span>{repeats == null ? "未知" : `${repeats.toFixed(repeats >= 10 ? 1 : 2)}×`}</span></div>; })}</div></Panel>
      <Panel title="质量与许可证门禁"><div className="gate-strip"><ShieldCheck size={17} /><div><strong>{version.gate_result?.status === "passed" ? "检查通过" : "需要检查"}</strong><span>{version.gate_result?.issues?.length || 0} 个问题</span></div><StatusBadge status={version.gate_result?.status || "draft"} /></div></Panel>
      {version.failure_reason && <Alert severity="error" title="Manifest 解析失败" detail={version.failure_reason} />}
    </div>
    {editOpen && <MixtureModal projectId={projectId} mixture={mixture} baseVersion={version} datasets={datasets} globalMode={globalMode} onClose={() => setEditOpen(false)} onCreated={() => { setEditOpen(false); onChanged(); }} />}
  </aside>;
}

function formatSigned(value: number) {
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

type SourceSelection = {
  sample_ratio: number;
  max_tokens: number | null;
  min_length: number | null;
  max_length: number | null;
  include_paths: string[];
  exclude_paths: string[];
  seed: number;
};

function normalizedSelection(value?: Partial<SourceSelection>): SourceSelection {
  return {
    sample_ratio: Math.min(1, Math.max(0.0001, Number(value?.sample_ratio ?? 1))),
    max_tokens: value?.max_tokens ? Number(value.max_tokens) : null,
    min_length: value?.min_length ? Number(value.min_length) : null,
    max_length: value?.max_length ? Number(value.max_length) : null,
    include_paths: (value?.include_paths || []).filter(Boolean),
    exclude_paths: (value?.exclude_paths || []).filter(Boolean),
    seed: Number(value?.seed ?? 42),
  };
}

function selectedTokenEstimate(availableTokens?: number, value?: Partial<SourceSelection>) {
  const selection = normalizedSelection(value);
  const sampled = Math.floor(Number(availableTokens || 0) * selection.sample_ratio);
  return selection.max_tokens ? Math.min(sampled, selection.max_tokens) : sampled;
}

function selectionSummary(value?: Partial<SourceSelection>) {
  const selection = normalizedSelection(value);
  const parts = [selection.sample_ratio < 1 ? `抽样 ${(selection.sample_ratio * 100).toFixed(1)}%` : "全量"];
  if (selection.max_tokens) parts.push(`上限 ${formatNumber(selection.max_tokens)} Tokens`);
  if (selection.min_length || selection.max_length) parts.push(`长度 ${selection.min_length || 0}–${selection.max_length || "不限"}`);
  if (selection.include_paths.length) parts.push(`包含 ${selection.include_paths.join("、")}`);
  if (selection.exclude_paths.length) parts.push(`排除 ${selection.exclude_paths.join("、")}`);
  return parts.join(" · ");
}

type HfRepoFile = { path: string; size_bytes: number };

type DatasetWizardInitialValues = Partial<{
  modality: string;
  source_type: string;
  name: string;
  source_uri: string;
  language: string;
  license: string;
  revision: string;
  config: string;
  split: string;
  hf_endpoint_mode: string;
  custom_hf_endpoint: string;
}>;

export function DatasetWizard({ projectId, globalMode = false, assetContext = "dataset", initialValues = {}, onClose, onCreated }: { projectId: string; globalMode?: boolean; assetContext?: "dataset" | "evaluation"; initialValues?: DatasetWizardInitialValues; onClose: () => void; onCreated: (dataset: Dataset) => void }) {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({ modality: "text", source_type: "huggingface", name: "", source_uri: "", language: "zh", license: "unknown", revision: "main", config: "", split: "", hf_endpoint_mode: "deployment", custom_hf_endpoint: "", ...initialValues });
  const [created, setCreated] = useState<Dataset | null>(null);
  const [scanTaskId, setScanTaskId] = useState("");
  const [planTaskId, setPlanTaskId] = useState("");
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [fileFilter, setFileFilter] = useState("");
  const isHf = form.source_type === "huggingface";
  const finalStep = isHf ? 6 : 5;
  const hfEndpoint = form.hf_endpoint_mode === "custom" ? form.custom_hf_endpoint.trim() : HF_ENDPOINTS[form.hf_endpoint_mode as keyof typeof HF_ENDPOINTS];
  const sourceConfig = { revision: form.revision || undefined };
  const executionTool = transferTool(form.source_type, hfEndpoint);
  const preflight = useMutation({ mutationFn: () => post<DatasetPreflight>(globalMode ? "/catalog/datasets/preflight" : `/projects/${projectId}/datasets/preflight`, { source_type: form.source_type, source_uri: form.source_uri, source_config: sourceConfig }), onSuccess: () => setStep(3) });
  const scan = useMutation({ mutationFn: () => post<Task>(globalMode ? "/catalog/datasets/huggingface/scan" : `/projects/${projectId}/datasets/huggingface/scan`, { repo_id: form.source_uri.trim(), revision: form.revision.trim() || "main", hf_endpoint: hfEndpoint || undefined, config: form.config.trim() || undefined, split: form.split.trim() || undefined, resource_request: { gpus: 0 } }), onSuccess: (task) => { setScanTaskId(task.id); setPlanTaskId(""); setSelectedPaths([]); setStep(3); } });
  const scanTask = useQuery({ queryKey: ["hf-scan", scanTaskId], queryFn: () => api<Task>(`/tasks/${scanTaskId}`), enabled: Boolean(scanTaskId), refetchInterval: (query) => TERMINAL_STATES.has(query.state.data?.observed_state || "") ? false : 800 });
  const scanResult = scanTask.data?.result || {};
  const hfFiles = (scanResult.files || []) as HfRepoFile[];
  const fileGroups = useMemo(() => {
    const groups = new Map<string, HfRepoFile[]>();
    hfFiles.forEach((file) => { const slash = file.path.lastIndexOf("/"); const directory = slash < 0 ? "仓库根目录" : file.path.slice(0, slash); groups.set(directory, [...(groups.get(directory) || []), file]); });
    return [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [hfFiles]);
  const selectedSet = useMemo(() => new Set(selectedPaths), [selectedPaths]);
  const selectedBytes = hfFiles.reduce((sum, file) => sum + (selectedSet.has(file.path) ? file.size_bytes : 0), 0);
  const updateSelection = (paths: string[]) => { setSelectedPaths(paths); setPlanTaskId(""); };
  const toggleFile = (path: string) => updateSelection(selectedSet.has(path) ? selectedPaths.filter((value) => value !== path) : [...selectedPaths, path]);
  const toggleGroup = (files: HfRepoFile[]) => { const allSelected = files.every((file) => selectedSet.has(file.path)); const paths = new Set(selectedPaths); files.forEach((file) => allSelected ? paths.delete(file.path) : paths.add(file.path)); updateSelection([...paths]); };
  const plan = useMutation({ mutationFn: () => post<Task>(globalMode ? "/catalog/datasets/huggingface/plans" : `/projects/${projectId}/datasets/huggingface/plans`, { scan_task_id: scanTaskId, selected_paths: selectedPaths }), onSuccess: (task) => { setPlanTaskId(task.id); setStep(4); } });
  const planTask = useQuery({ queryKey: ["hf-plan", planTaskId], queryFn: () => api<Task>(`/tasks/${planTaskId}`), enabled: Boolean(planTaskId), refetchInterval: (query) => TERMINAL_STATES.has(query.state.data?.observed_state || "") ? false : 800 });
  const planResult = planTask.data?.result || {};
  const create = useMutation({ mutationFn: () => post<Dataset>(globalMode ? "/catalog/datasets" : `/projects/${projectId}/datasets`, { name: form.name, modality: form.modality, source_type: form.source_type, source_uri: form.source_uri, language: form.language, license: form.license, tags: assetContext === "evaluation" ? ["评测候选", "在线下载"] : ["真实导入"], source_config: isHf ? { hf_plan_task_id: planTaskId } : sourceConfig, start_import: true, adapter_id: "builtin.data-import", resource_request: { gpus: 0 }, initial_version: { tier: "bronze", source_snapshot: isHf ? { download_plan: planResult } : { preflight: preflight.data } } }), onSuccess: (dataset) => { setCreated(dataset); setStep(finalStep); } });
  const set = (key: keyof typeof form, value: string) => { setForm((current) => ({ ...current, [key]: value })); if (["source_type", "source_uri", "revision", "config", "split", "hf_endpoint_mode", "custom_hf_endpoint"].includes(key)) { setScanTaskId(""); setPlanTaskId(""); setSelectedPaths([]); } };
  const customEndpointValid = form.hf_endpoint_mode !== "custom" || isHttpUrl(form.custom_hf_endpoint);
  const canConfigure = Boolean(form.name.trim() && form.source_uri.trim() && customEndpointValid);
  const steps = isHf ? ["选择来源", "配置与解析", "选择文件", "下载预检", "启动下载", "任务已创建"] : ["选择来源", "配置来源", "真实预检", "启动下载", "任务已创建"];
  const visibleGroups = fileGroups.map(([directory, files]) => [directory, files.filter((file) => !fileFilter || file.path.toLowerCase().includes(fileFilter.toLowerCase()))] as const).filter(([, files]) => files.length);
  return <Modal title={assetContext === "evaluation" ? "在线下载评测数据" : "新数据"} description={isHf ? "解析并固定 Hub 版本，选择物理文件后由同一 Worker 预检、下载和登记本地资产。" : "填写真实地址，预检成功后由 Worker 下载并创建不可变 Bronze V1。"} onClose={onClose} wide>
    <div className="wizard-steps">{steps.map((label, index) => <div className={step === index + 1 ? "active" : step > index + 1 ? "done" : ""} key={label}><i>{step > index + 1 ? <Check size={13} /> : index + 1}</i><span>{label}</span></div>)}</div>
    <div className="wizard-body">
      {step === 1 && <div className="wizard-choice-stack"><section><header><strong>数据类型</strong><span>类型决定解析策略和可用的下游能力</span></header><div className="source-grid modality-grid">{modalities.map(([key, label, detail, Icon]) => <button type="button" key={key} className={form.modality === key ? "selected" : ""} onClick={() => set("modality", key)}><Icon size={20} /><strong>{label}</strong><span>{detail}</span></button>)}</div></section><section><header><strong>数据来源</strong><span>来源只决定内容如何获取</span></header><div className="source-grid">{sources.map(([key, label, detail]) => <button type="button" key={key} className={form.source_type === key ? "selected" : ""} onClick={() => set("source_type", key)}><HardDrive size={20} /><strong>{label}</strong><span>{detail}</span></button>)}</div></section>{form.modality !== "text" && <Alert severity="info" title={`${modalityLabels[form.modality]}解析器尚未接入`} detail="本次仍会下载原始内容、计算文件指纹并创建不可变版本；在语义解析器接入前不会进入文本训练、Tokenizer 或文本检索流程。" />}</div>}
      {step === 2 && <div className="form-grid two-columns"><Field label="数据集名称"><input value={form.name} onChange={(event) => set("name", event.target.value)} placeholder="例如：中文预训练语料" /></Field><Field label={isHf ? "Repo ID" : form.source_type === "local" ? "本地绝对路径" : "数据地址"}><input value={form.source_uri} onChange={(event) => set("source_uri", event.target.value)} placeholder={sourcePlaceholder(form.source_type)} /></Field><Field label="语言声明"><input value={form.language} onChange={(event) => set("language", event.target.value)} /></Field><Field label="许可证声明"><input value={form.license} onChange={(event) => set("license", event.target.value)} /></Field>{["huggingface", "git"].includes(form.source_type) && <Field label="Revision / Branch" hint={isHf ? "解析时立即固定为 commit hash" : "留空使用默认分支"}><input value={form.revision} onChange={(event) => set("revision", event.target.value)} placeholder={isHf ? "main" : "默认分支"} /></Field>}{isHf && <><Field label="Hub 端点" hint="所有节点统一由官方 huggingface_hub SDK 访问"><select value={form.hf_endpoint_mode} onChange={(event) => set("hf_endpoint_mode", event.target.value)}><option value="deployment">部署默认（HF_ENDPOINT）</option><option value="official">Hugging Face 官方</option><option value="mirror">HF-Mirror 镜像</option><option value="custom">自定义兼容端点</option></select></Field>{form.hf_endpoint_mode === "custom" && <Field label="自定义端点"><input type="url" value={form.custom_hf_endpoint} onChange={(event) => set("custom_hf_endpoint", event.target.value)} placeholder="https://your-compatible-hub.example.com" /></Field>}<Field label="Config（可选）" hint="逻辑元数据，不参与物理路径匹配"><input value={form.config} onChange={(event) => set("config", event.target.value)} /></Field><Field label="Split（可选）" hint="逻辑元数据，不参与物理路径匹配"><input value={form.split} onChange={(event) => set("split", event.target.value)} placeholder="train" /></Field></>}<ErrorMessage error={scan.error || preflight.error} /></div>}
      {isHf && step === 3 && <div className="hf-file-picker">{scanTask.data?.observed_state && scanTask.data.observed_state !== "succeeded" && !TERMINAL_STATES.has(scanTask.data.observed_state) && <div className="download-ready"><RefreshCw className="spin" size={32} /><h3>正在目标 Worker 解析仓库</h3><p>{scanTask.data.current_phase || "等待 Worker 领取"}</p></div>}{scanTask.data?.observed_state === "failed" && <Alert severity="error" title="仓库解析失败" detail={downloadErrorDetail(scanTask.data.error)} />}{scanTask.data?.observed_state === "succeeded" && <><section className="hf-summary"><div><span>固定 Commit</span><strong title={String(scanResult.commit_hash)}>{String(scanResult.commit_hash).slice(0, 12)}</strong></div><div><span>仓库总大小</span><strong>{formatBytes(scanResult.total_size_bytes)}</strong></div><div><span>文件数量</span><strong>{formatNumber(scanResult.file_count)}</strong></div><div><span>已选择</span><strong>{formatNumber(selectedPaths.length)} · {formatBytes(selectedBytes)}</strong></div></section><div className="hf-picker-actions"><input aria-label="筛选仓库文件" value={fileFilter} onChange={(event) => setFileFilter(event.target.value)} placeholder="筛选路径" /><button className="button secondary small" onClick={() => updateSelection(hfFiles.map((file) => file.path))}>全选</button><button className="button secondary small" onClick={() => updateSelection(hfFiles.filter((file) => matchesModalityFile(file.path, form.modality)).map((file) => file.path))}>选择{modalityLabels[form.modality]}文件</button><button className="button secondary small" onClick={() => updateSelection([])}>清空</button></div><div className="hf-tree">{visibleGroups.map(([directory, files]) => <section key={directory}><label className="hf-directory"><input type="checkbox" checked={files.every((file) => selectedSet.has(file.path))} onChange={() => toggleGroup(files)} /><strong>{directory}</strong><span>{files.length} 个文件 · {formatBytes(files.reduce((sum, file) => sum + file.size_bytes, 0))}</span></label>{files.map((file) => <label className="hf-file" key={file.path}><input type="checkbox" checked={selectedSet.has(file.path)} onChange={() => toggleFile(file.path)} /><code>{file.path}</code><span>{formatBytes(file.size_bytes)}</span></label>)}</section>)}</div><div className="hf-links"><a href={`https://huggingface.co/datasets/${form.source_uri}`} target="_blank" rel="noreferrer">在 HF 查看数据卡</a><a href={`https://huggingface.co/datasets/${form.source_uri}/viewer`} target="_blank" rel="noreferrer">打开 HF 数据预览</a><span>大数据集的筛选、搜索或预览可能只覆盖前 5GB，不代表全量分布。</span></div></>}</div>}
      {isHf && step === 4 && <div className="preflight-list">{planTask.data?.observed_state && planTask.data.observed_state !== "succeeded" && !TERMINAL_STATES.has(planTask.data.observed_state) && <Alert severity="info" title="正在目标 Worker 检查缓存和磁盘" detail={planTask.data.current_phase || "等待 Worker 领取"} />}{planTask.data?.observed_state === "failed" && <Alert severity="error" title="下载预检失败" detail={downloadErrorDetail(planTask.data.error)} />}{planTask.data?.observed_state === "succeeded" && <><Alert severity="success" title={`已固定 ${formatNumber(planResult.file_count)} 个物理文件`} detail={`Commit ${String(planResult.commit_hash).slice(0, 12)}`} />{planResult.endpoint_warning && <Alert severity="warning" title="镜像节点已回退到 Hugging Face 官方" detail={String(planResult.endpoint_warning)} />}<Alert severity="info" title={`选择数据量：${formatBytes(planResult.selected_bytes)}`} detail={`完整文件缓存：${formatBytes(planResult.cached_bytes)} · 计划获取：${formatBytes(planResult.download_bytes)}`} /><Alert severity={planResult.disk_space_sufficient ? "success" : "error"} title={`目标磁盘剩余：${formatBytes(planResult.disk_free_bytes)}`} detail={planResult.disk_space_sufficient ? "空间检查通过" : "空间不足，不能启动下载"} /><Alert severity="warning" title="计划获取量是逻辑文件大小" detail="Xet 分块缓存和执行期间缓存变化可能使实际网络流量不同。" /></>}</div>}
      {!isHf && step === 3 && preflight.data && <div className="preflight-list"><Alert severity={preflight.data.accessible ? "success" : "error"} title={preflight.data.accessible ? "来源真实可访问" : "来源无法访问"} detail={preflight.data.error || preflight.data.source_uri} /><Alert severity={preflight.data.disk_space_sufficient ? "success" : "error"} title={`制品库可用空间：${formatBytes(preflight.data.disk_free_bytes)}`} detail={preflight.data.total_size_bytes == null ? "来源未提供完整大小" : `已知下载大小：${formatBytes(preflight.data.total_size_bytes)}`} /><Alert severity="info" title={`已发现 ${formatNumber(preflight.data.file_count)} 个文件`} detail={preflight.data.files.slice(0, 4).map((file) => file.path).join("；") || "文件列表将在下载时确定"} />{preflight.data.warnings.map((warning) => <Alert key={warning} severity="warning" title={warning} />)}</div>}
      {step === (isHf ? 5 : 4) && <div className="download-ready"><FileCheck2 size={42} /><h3>准备通过 {executionTool} 执行</h3><p>{isHf ? "正式下载固定到预检 Worker、目标目录、commit 和精确文件清单；完成后登记为 Worker 本地不可变数据资产。" : `平台负责调度、日志、状态和制品登记；协议、续传与重试由 ${executionTool} 负责。`}</p><ErrorMessage error={create.error} /></div>}
      {step === finalStep && <div className="download-ready"><div className="success-orb"><Check size={25} /></div><h3>下载任务已创建</h3><p>Bronze V1 当前状态为 {created?.current_version?.state || "queued"}。关闭向导后列表和详情会持续刷新真实任务进度。</p><code>{created?.current_version?.task_id}</code></div>}
    </div>
    <footer className="wizard-actions">{step > 1 && step < finalStep && <button className="button secondary" onClick={() => setStep(step - 1)}><ArrowLeft size={16} />上一步</button>}<span />{step === 1 && <button className="button primary" onClick={() => setStep(2)}>下一步<ArrowRight size={16} /></button>}{step === 2 && isHf && <button className="button primary" disabled={!canConfigure || scan.isPending} onClick={() => scan.mutate()}>{scan.isPending ? "提交中…" : "解析仓库"}<ArrowRight size={16} /></button>}{step === 2 && !isHf && <button className="button primary" disabled={!canConfigure || preflight.isPending} onClick={() => preflight.mutate()}>{preflight.isPending ? "检查中…" : "真实检查来源"}<ArrowRight size={16} /></button>}{isHf && step === 3 && <button className="button primary" disabled={scanTask.data?.observed_state !== "succeeded" || !selectedPaths.length || plan.isPending} onClick={() => plan.mutate()}>{plan.isPending ? "提交中…" : "预检所选文件"}<ArrowRight size={16} /></button>}{isHf && step === 4 && <button className="button primary" disabled={planTask.data?.observed_state !== "succeeded" || !planResult.disk_space_sufficient} onClick={() => setStep(5)}>下一步<ArrowRight size={16} /></button>}{!isHf && step === 3 && <button className="button primary" disabled={!preflight.data?.accessible || !preflight.data?.disk_space_sufficient} onClick={() => setStep(4)}>下一步<ArrowRight size={16} /></button>}{step === (isHf ? 5 : 4) && <button className="button primary" disabled={create.isPending} onClick={() => create.mutate()}>{create.isPending ? "创建中…" : "创建并开始下载"}</button>}{step === finalStep && <button className="button primary" onClick={() => created && onCreated(created)}>完成</button>}</footer>
  </Modal>;
}

function MixtureModal({ projectId, datasets, mixture, baseVersion, globalMode = false, onClose, onCreated }: { projectId: string; datasets: Dataset[]; mixture?: Mixture; baseVersion?: any; globalMode?: boolean; onClose: () => void; onCreated: () => void }) {
  const queryClient = useQueryClient();
  const [addSourceOpen, setAddSourceOpen] = useState(false);
  const candidates = useMemo(() => datasets.flatMap((dataset) => (dataset.versions || []).filter((version: any) => version.train_ready && version.state === "ready" && version.artifact_id && version.record_manifest_artifact_id).map((version: any) => ({ dataset, version }))), [datasets]);
  const initialEntries = baseVersion?.entries || [];
  const [weights, setWeights] = useState<Record<string, number>>(() => Object.fromEntries(initialEntries.map((entry: any) => [entry.dataset_version_id, Number(entry.weight || 0)])));
  const [selected, setSelected] = useState<Record<string, boolean>>(() => Object.fromEntries(initialEntries.map((entry: any) => [entry.dataset_version_id, true])));
  const [selections, setSelections] = useState<Record<string, SourceSelection>>(() => Object.fromEntries(initialEntries.map((entry: any) => [entry.dataset_version_id, normalizedSelection(entry.selection)])));
  const [filteringVersionId, setFilteringVersionId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [language, setLanguage] = useState("");
  const [license, setLicense] = useState("");
  const visible = candidates.filter(({ dataset }) => `${dataset.name} ${dataset.language} ${dataset.license}`.toLowerCase().includes(search.toLowerCase()) && (!language || dataset.language === language) && (!license || dataset.license === license));
  const selectedCandidates = candidates.filter(({ version }) => selected[version.id]);
  const total = useMemo(() => selectedCandidates.reduce((sum, { version }) => sum + Number(weights[version.id] || 0), 0), [selectedCandidates, weights]);
  const create = useMutation({ mutationFn: (data: any) => post(mixture ? (globalMode ? `/catalog/mixtures/${mixture.id}/versions` : `/mixtures/${mixture.id}/versions`) : (globalMode ? "/catalog/mixtures" : `/projects/${projectId}/mixtures`), data), onSuccess: onCreated });
  function toggle(versionId: string, checked: boolean) { setSelected({ ...selected, [versionId]: checked }); if (checked && !weights[versionId]) setWeights({ ...weights, [versionId]: 1 }); }
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const data = new FormData(event.currentTarget); const entries = selectedCandidates.filter(({ version }) => Number(weights[version.id]) > 0).map(({ dataset, version }) => { const selection = normalizedSelection(selections[version.id]); return { dataset_version_id: version.id, name: dataset.name, weight: Number(weights[version.id]), available_tokens: version.token_estimate, selected_tokens: selectedTokenEstimate(version.token_estimate, selection), selection, language: dataset.language, license: dataset.license, tier: version.tier, version: version.version }; }); const shared = { entries, target_tokens: Number(data.get("target_tokens")), seed: Number(data.get("seed")), gate_result: baseVersion?.gate_result || { status: "passed", issues: [] }, distribution: {}, estimated_steps: 0 }; create.mutate(mixture ? shared : { ...shared, name: data.get("name"), description: data.get("description") }); }
  return <><Modal title={mixture ? `编辑 ${mixture.name}` : "新建数据混合"} description={mixture ? `基于 V${baseVersion?.version} 创建新的不可变版本。` : "选择可训练的不可变数据版本，并设置跨数据集采样权重。"} onClose={onClose} wide><form onSubmit={submit}>
    <div className="form-grid two-columns">{!mixture && <><Field label="混合名称"><input name="name" required placeholder="ExampleLM-Mixture-v2" /></Field><Field label="说明"><input name="description" placeholder="训练用途与选择依据" /></Field></>}<Field label="目标 Tokens" hint="例如 12B 填 12000000000"><input name="target_tokens" type="number" min="1" required defaultValue={baseVersion?.target_tokens || 6_000_000_000} /></Field><Field label="固定 Seed" hint="修改 seed 会创建一条不同的稳定数据流"><input name="seed" type="number" required defaultValue={baseVersion?.seed ?? 42} /></Field></div>
    <div className="mixture-source-heading"><div><strong>参与混合的数据源</strong><span>从已有可训练版本中选择，或者先导入新的数据源。</span></div><button type="button" className="button secondary small" onClick={() => setAddSourceOpen(true)}><Plus size={14} />添加新数据源</button></div>
    <div className="mixture-source-filters"><div className="search-box"><Search size={15} /><input aria-label="搜索混合来源" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称、语言或许可证" /></div><select aria-label="按语言筛选" value={language} onChange={(event) => setLanguage(event.target.value)}><option value="">全部语言</option>{Array.from(new Set(candidates.map(({ dataset }) => dataset.language).filter(Boolean))).map((value) => <option key={value} value={value}>{value}</option>)}</select><select aria-label="按许可证筛选" value={license} onChange={(event) => setLicense(event.target.value)}><option value="">全部许可证</option>{Array.from(new Set(candidates.map(({ dataset }) => dataset.license).filter(Boolean))).map((value) => <option key={value} value={value}>{value}</option>)}</select></div>
    <div className="weight-editor selectable"><header><span>选择 / 数据版本</span><span>选中 / 可用 Tokens</span><span>采样权重</span><span>比例 / 子集</span></header>{visible.map(({ dataset, version }) => { const rule = normalizedSelection(selections[version.id]); const selectedTokens = selectedTokenEstimate(version.token_estimate, rule); return <div key={version.id}><label className="mixture-source-choice"><input type="checkbox" checked={Boolean(selected[version.id])} onChange={(event) => toggle(version.id, event.target.checked)} /><strong>{dataset.name}<small>{String(version.tier).toUpperCase()} V{version.version} · {dataset.language} · {dataset.license}</small></strong></label><span>{formatNumber(selectedTokens)} / {formatNumber(version.token_estimate)}</span><input aria-label={`${dataset.name} V${version.version} 采样权重`} type="number" min="0" step="0.1" disabled={!selected[version.id]} value={weights[version.id] || 0} onChange={(event) => setWeights({ ...weights, [version.id]: Number(event.target.value) })} /><div className="mixture-rule-cell"><b>{total && selected[version.id] ? ((weights[version.id] || 0) / total * 100).toFixed(1) : 0}%</b><button type="button" className="button secondary small" disabled={!selected[version.id]} onClick={() => setFilteringVersionId(version.id)}>筛选子集</button><small>{selectionSummary(rule)}</small></div></div>; })}</div>
    {!candidates.length && <Alert severity="warning" title="还没有导入成功且可训练的数据版本" />}{Boolean(candidates.length) && !visible.length && <Alert severity="warning" title="没有匹配筛选条件的数据版本" />}
    <div className="preflight-list compact"><Alert severity="success" title={selectedCandidates.length >= 2 ? `已选择 ${selectedCandidates.length} 个来源进行跨来源混合` : "单一来源数据配方有效"} detail={selectedCandidates.length >= 2 ? "每个来源先应用自己的子集规则，再按权重混合。" : "权重固定为 100%，仍可通过“筛选子集”选择该来源内部的数据。"} /><Alert severity="info" title="保存后会生成稳定 Manifest" detail="平台确定实际入选文档和采样语义；只有解析成功的版本可以进入下游。" /></div>
    <ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!selectedCandidates.length || total <= 0} onCancel={onClose} submitLabel={mixture ? "创建新版本" : "保存混合版本"} />
  </form></Modal>{addSourceOpen && <DatasetWizard projectId={projectId} onClose={() => setAddSourceOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["datasets", projectId] }); setAddSourceOpen(false); }} />}{filteringVersionId && <SourceSelectionModal datasetName={candidates.find(({ version }) => version.id === filteringVersionId)?.dataset.name || "数据源"} availableTokens={candidates.find(({ version }) => version.id === filteringVersionId)?.version.token_estimate || 0} sourceFiles={candidates.find(({ version }) => version.id === filteringVersionId)?.version.source_snapshot?.files || []} value={normalizedSelection(selections[filteringVersionId])} onClose={() => setFilteringVersionId(null)} onSave={(value) => { setSelections({ ...selections, [filteringVersionId]: value }); setFilteringVersionId(null); }} />}</>;
}

type SourcePathFile = { path: string; size_bytes?: number | null };

function SourceSelectionModal({ datasetName, availableTokens, sourceFiles, value, onClose, onSave }: { datasetName: string; availableTokens: number; sourceFiles: SourcePathFile[]; value: SourceSelection; onClose: () => void; onSave: (value: SourceSelection) => void }) {
  const [draft, setDraft] = useState(value);
  const [pathSearch, setPathSearch] = useState("");
  const estimated = selectedTokenEstimate(availableTokens, draft);
  const pathNodes = useMemo(() => buildSourcePathNodes(sourceFiles), [sourceFiles]);
  const visiblePathNodes = pathNodes.filter((node) => !pathSearch || node.path.toLowerCase().includes(pathSearch.toLowerCase())).slice(0, 500);
  const knownRules = new Set(pathNodes.map((node) => node.rule));
  const customIncludes = draft.include_paths.filter((rule) => !knownRules.has(rule));
  const customExcludes = draft.exclude_paths.filter((rule) => !knownRules.has(rule));
  function updatePathRule(node: SourcePathNode, state: "none" | "include" | "exclude") {
    const descendantPrefix = node.type === "directory" ? `${node.path}/` : "";
    const keep = (rule: string) => rule !== node.rule && (!descendantPrefix || !rule.startsWith(descendantPrefix));
    const include_paths = draft.include_paths.filter(keep);
    const exclude_paths = draft.exclude_paths.filter(keep);
    if (state === "include") include_paths.push(node.rule);
    if (state === "exclude") exclude_paths.push(node.rule);
    setDraft({ ...draft, include_paths, exclude_paths });
  }
  function removeCustom(rule: string) { setDraft({ ...draft, include_paths: draft.include_paths.filter((item) => item !== rule), exclude_paths: draft.exclude_paths.filter((item) => item !== rule) }); }
  return <Modal title={`筛选子集 · ${datasetName}`} description="规则作用于这个数据版本内部，并随数据配方版本冻结。" onClose={onClose} wide><div className="form-grid two-columns">
    <Field label="确定性抽样比例" hint="0–100%；使用随机种子保证可复现"><input aria-label="确定性抽样比例" type="number" min="0.01" max="100" step="0.01" value={draft.sample_ratio * 100} onChange={(event) => setDraft({ ...draft, sample_ratio: Number(event.target.value) / 100 })} /></Field>
    <Field label="Token 上限" hint="留空表示不额外限制"><input aria-label="Token 上限" type="number" min="1" value={draft.max_tokens || ""} onChange={(event) => setDraft({ ...draft, max_tokens: event.target.value ? Number(event.target.value) : null })} /></Field>
    <Field label="最短文档长度（字符）"><input aria-label="最短文档长度" type="number" min="0" value={draft.min_length || ""} onChange={(event) => setDraft({ ...draft, min_length: event.target.value ? Number(event.target.value) : null })} /></Field>
    <Field label="最长文档长度（字符）"><input aria-label="最长文档长度" type="number" min="1" value={draft.max_length || ""} onChange={(event) => setDraft({ ...draft, max_length: event.target.value ? Number(event.target.value) : null })} /></Field>
    <section className="source-path-section span-two"><header><div><strong>目录与文件范围</strong><span>直接对冻结文件清单选择包含、排除或不限制。</span></div><div className="search-box"><Search size={14} /><input aria-label="搜索目录或文件" value={pathSearch} onChange={(event) => setPathSearch(event.target.value)} placeholder="搜索目录或文件" /></div></header>{pathNodes.length ? <div className="source-path-tree"><div className="source-path-head"><span>目录 / 文件</span><span>大小</span><span>规则</span></div>{visiblePathNodes.map((node) => { const state = draft.include_paths.includes(node.rule) ? "include" : draft.exclude_paths.includes(node.rule) ? "exclude" : "none"; return <div className={`source-path-row path-${node.type}`} key={`${node.type}:${node.path}`}><span style={{ paddingLeft: `${node.depth * 16}px` }}>{node.type === "directory" ? <Folder size={15} /> : <FileIcon size={14} />}<code>{node.path}</code></span><small>{node.type === "file" ? formatBytes(node.size_bytes) : `${node.file_count} 个文件`}</small><select aria-label={`${node.path} 路径规则`} value={state} onChange={(event) => updatePathRule(node, event.target.value as "none" | "include" | "exclude")}><option value="none">不限制</option><option value="include">包含</option><option value="exclude">排除</option></select></div>; })}</div> : <Alert severity="warning" title="这个版本没有可选择的冻结文件清单" detail="请先完成数据导入或重新分析；在文件清单可用前不能可靠地选择路径。" />}{pathNodes.length > 500 && <p className="muted">当前显示前 500 项，请使用搜索框缩小范围。</p>}{Boolean(customIncludes.length || customExcludes.length) && <div className="custom-path-rules"><strong>已有自定义规则</strong><span>这些规则无法映射到当前文件树，仍会保留。</span><div>{customIncludes.map((rule) => <button type="button" key={`include:${rule}`} onClick={() => removeCustom(rule)}>包含：{rule} ×</button>)}{customExcludes.map((rule) => <button type="button" key={`exclude:${rule}`} onClick={() => removeCustom(rule)}>排除：{rule} ×</button>)}</div></div>}</section>
    <Field label="随机种子"><input aria-label="随机种子" type="number" value={draft.seed} onChange={(event) => setDraft({ ...draft, seed: Number(event.target.value) })} /></Field>
    <div className="selection-estimate"><span>筛选后估算</span><strong>{formatNumber(estimated)} Tokens</strong><small>原始 {formatNumber(availableTokens)} Tokens；路径和长度条件需由执行引擎扫描后得到精确值。</small></div>
    <div className="form-actions span-two"><button type="button" className="button secondary" onClick={onClose}>取消</button><button type="button" className="button primary" disabled={draft.sample_ratio <= 0 || draft.sample_ratio > 1 || Boolean(draft.min_length && draft.max_length && draft.min_length > draft.max_length)} onClick={() => onSave(normalizedSelection(draft))}>应用筛选</button></div>
  </div></Modal>;
}

type SourcePathNode = { type: "directory" | "file"; path: string; rule: string; depth: number; size_bytes?: number | null; file_count?: number };

function buildSourcePathNodes(files: SourcePathFile[]): SourcePathNode[] {
  const cleanFiles = files.filter((file) => file?.path).map((file) => ({ ...file, path: String(file.path).replaceAll("\\", "/").replace(/^\.\//, "") }));
  const directories = new Map<string, number>();
  cleanFiles.forEach((file) => { const parts = file.path.split("/"); for (let index = 1; index < parts.length; index += 1) { const directory = parts.slice(0, index).join("/"); directories.set(directory, (directories.get(directory) || 0) + 1); } });
  return [
    ...[...directories].map(([path, file_count]) => ({ type: "directory" as const, path, rule: `${path}/**`, depth: path.split("/").length - 1, file_count })),
    ...cleanFiles.map((file) => ({ type: "file" as const, path: file.path, rule: file.path, depth: file.path.split("/").length - 1, size_bytes: file.size_bytes })),
  ].sort((left, right) => left.path.localeCompare(right.path) || (left.type === "directory" ? -1 : 1));
}

function sourcePlaceholder(sourceType: string): string {
  if (sourceType === "huggingface") return "openai/gsm8k";
  if (sourceType === "local") return "F:\\datasets\\corpus.jsonl";
  if (sourceType === "git") return "https://github.com/org/dataset.git";
  return "https://example.com/corpus.jsonl";
}

function matchesModalityFile(path: string, modality: string): boolean {
  const patterns: Record<string, RegExp> = {
    text: /\.(parquet|arrow|feather|jsonl|ndjson|json|csv|tsv|txt|text|md|markdown|gz|zst)$/i,
    image: /\.(png|jpe?g|webp|gif|bmp|tiff?|svg|json|jsonl|csv|parquet)$/i,
    audio: /\.(wav|mp3|flac|m4a|aac|ogg|opus|json|jsonl|csv|parquet)$/i,
    video: /\.(mp4|mov|mkv|webm|avi|m4v|json|jsonl|csv|parquet)$/i,
    multimodal: /\.(png|jpe?g|webp|wav|mp3|flac|mp4|mov|mkv|webm|json|jsonl|csv|parquet)$/i,
    other: /./,
  };
  return (patterns[modality] || patterns.other).test(path);
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value.trim());
    return ["http:", "https:"].includes(parsed.protocol) && Boolean(parsed.hostname);
  } catch {
    return false;
  }
}

function formatBytes(value?: number | null): string {
  if (value == null) return "未知";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
}

function formatRate(value: unknown): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "未检测";
}

function formatCheck(value: unknown): string {
  if (value == null || value === "") return "未检测";
  if (typeof value === "boolean") return value ? "通过" : "未通过";
  return String(value);
}
