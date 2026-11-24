import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Download, Eye, Hammer, Link2, Pencil, Plus, Trash2 } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, formatBytes, formatDate, formatNumber, post, postForm } from "../api";
import { Alert, EmptyState, ErrorMessage, Field, FormActions, Modal, Panel, StatusBadge } from "../components/ui";
import type { Dataset } from "../types";
import { DatasetTableViewer, DatasetWizard } from "./DataCenter";
import { EvaluationDatasetList, evaluationDatasetRows, evaluationSource } from "../components/EvaluationDatasetList";

type AssetSourceMode = "upload" | "data_version";
type EvaluationInspection = {
  filename: string;
  format: string;
  size_bytes: number;
  sample_count: number;
  valid_sample_count: number;
  invalid_sample_count: number;
  fields: string[];
  missing_mapping_fields: string[];
  preview: Array<Record<string, unknown>>;
};
type EvaluationAssetDetail = {
  version: any;
  versions: any[];
  artifact: any | null;
  references: any[];
  can_delete: boolean;
  delete_blockers: any[];
};
export function BenchmarkCenter() {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [createOpen, setCreateOpen] = useState(false);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [initialDatasetVersionId, setInitialDatasetVersionId] = useState("");
  const [initialAssetName, setInitialAssetName] = useState("");
  const selectedAssetId = searchParams.get("asset") || "";
  const standards = useQuery({ queryKey: ["benchmark-catalog"], queryFn: () => api<any[]>("/benchmarks/catalog") });
  const definitionDetail = (standards.data || []).find((item) => item.id === searchParams.get("builtin"));
  const assets = useQuery({ queryKey: ["global-evaluation-datasets"], queryFn: () => api<any[]>("/catalog/evaluation-datasets") });
  const datasets = useQuery({
    queryKey: ["datasets", "benchmark-builder"],
    queryFn: () => api<Dataset[]>("/catalog/datasets"),
    refetchInterval: (query) => (query.state.data || []).some((dataset: Dataset) => !["ready", "failed", "cancelled"].includes(String(dataset.current_version?.state || ""))) ? 1500 : false,
  });
  const register = useMutation({
    mutationFn: (benchmarkId: string) => post<any>(`/catalog/evaluation-datasets/from-benchmark/${encodeURIComponent(benchmarkId)}`, { overrides: {} }),
    onSuccess: (asset) => {
      queryClient.invalidateQueries({ queryKey: ["global-evaluation-datasets"] });
      openAsset(asset.id);
    },
  });

  const assetDetail = useQuery({
    queryKey: ["global-evaluation-dataset", selectedAssetId],
    queryFn: () => api<EvaluationAssetDetail>(`/catalog/evaluation-datasets/${encodeURIComponent(selectedAssetId)}`),
    enabled: Boolean(selectedAssetId),
  });
  const rows = useMemo(() => evaluationDatasetRows(standards.data || [], assets.data || [], (datasets.data || []).filter((dataset) => (dataset.tags || []).includes("评测候选"))), [standards.data, assets.data, datasets.data]);

  function openBuilder(datasetVersionId = "", assetName = "") {
    setInitialDatasetVersionId(datasetVersionId);
    setInitialAssetName(assetName);
    setCreateOpen(true);
  }

  function openAsset(versionId: string) {
    const next = new URLSearchParams(searchParams);
    next.delete("builtin");
    next.set("asset", versionId);
    setSearchParams(next);
  }

  function setDefinitionDetail(item: any | null) {
    const next = new URLSearchParams(searchParams);
    if (item) next.set("builtin", item.id);
    else next.delete("builtin");
    setSearchParams(next);
  }

  function closeAsset() {
    const next = new URLSearchParams(searchParams);
    next.delete("asset");
    setSearchParams(next, { replace: true });
  }

  if (selectedAssetId) {
    return <BenchmarkAssetWorkspace
      detail={assetDetail.data}
      loading={assetDetail.isLoading}
      error={assetDetail.error}
      onBack={closeAsset}
      onSelectVersion={openAsset}
      onChanged={(versionId) => {
        queryClient.invalidateQueries({ queryKey: ["global-evaluation-datasets"] });
        queryClient.invalidateQueries({ queryKey: ["global-evaluation-dataset"] });
        if (versionId) openAsset(versionId);
        else closeAsset();
      }}
    />;
  }

  return <main className="system-page benchmark-management-page">
    <header className="benchmark-command-header">
      <div><p className="eyebrow">EVALUATION DATASETS</p><h1>评测集（Benchmark）</h1><p>统一管理评测题目、评分配置和版本，通过来源区分内置与导入的评测集。</p></div>
      <div className="benchmark-asset-actions"><button className="button secondary" onClick={() => setDownloadOpen(true)}><Download size={16} />在线导入</button><button className="button primary" onClick={() => openBuilder()}><Plus size={16} />上传或从数据创建</button></div>
    </header>
    <EvaluationDatasetList rows={rows} loading={standards.isLoading || assets.isLoading || datasets.isLoading} renderActions={(row) => {
      if (row.kind === "version") return <button className="button secondary small" onClick={() => openAsset(row.item.id)}><Eye size={14} />查看评测集</button>;
      if (row.kind === "builtin") return <><button className="button ghost small" onClick={() => setDefinitionDetail(row.item)}><Eye size={14} />查看评测集</button><button className="button secondary small" disabled={register.isPending} onClick={() => register.mutate(row.item.id)}>添加评测集</button></>;
      const ready = row.item.current_version?.state === "ready" && Boolean(row.item.current_version?.artifact_id);
      return <><Link className="button ghost small" to={`/data?dataset=${encodeURIComponent(row.item.id)}`}>查看导入</Link><button className="button secondary small" disabled={!ready} onClick={() => openBuilder(row.item.current_version.id, row.item.name)}><Hammer size={14} />配置评测集</button></>;
    }} />
    <ErrorMessage error={standards.error || assets.error || datasets.error || register.error} />
    {definitionDetail && <StandardDefinitionDetail item={definitionDetail} onClose={() => setDefinitionDetail(null)} pending={register.isPending} error={register.error} onAdd={() => register.mutate(definitionDetail.id)} />}
    {downloadOpen && <DatasetWizard projectId="" globalMode assetContext="evaluation" onClose={() => setDownloadOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["datasets", "benchmark-builder"] }); setDownloadOpen(false); }} />}
    {createOpen && <BenchmarkAssetCreateModal datasets={datasets.data || []} initialDatasetVersionId={initialDatasetVersionId} initialName={initialAssetName} onClose={() => setCreateOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["global-evaluation-datasets"] }); setCreateOpen(false); }} />}
  </main>;
}

function StandardDefinitionDetail({ item, onClose, onAdd, pending, error }: { item: any; onClose: () => void; onAdd: () => void; pending: boolean; error: unknown }) {
  const [view, setView] = useState<"overview" | "data">("overview");
  return <Modal title={item.name || item.id} description="来源：内置目录 · 添加后可管理版本和项目引用" onClose={onClose} wide>
    <nav className="benchmark-asset-tabs"><button onClick={() => setView("overview")} className={view === "overview" ? "active" : ""}>概要</button><button onClick={() => setView("data")} className={view === "data" ? "active" : ""}>数据查看</button></nav>
    {view === "data" ? <DatasetTableViewer key={item.id} version={{ id: item.id }} datasetName={item.name || item.id} queryEndpoint={`/catalog/benchmarks/${encodeURIComponent(item.id)}/viewer/query`} enableLabels={false} /> : <>
    <div className="benchmark-detail-grid">
      <Panel title="定义身份"><dl className="detail-list"><div><dt>框架</dt><dd>{item.framework}</dd></div><div><dt>任务 ID</dt><dd>{item.task_id}</dd></div><div><dt>许可证</dt><dd>{item.license || "unknown"}</dd></div><div><dt>数据</dt><dd>运行时由 lm-eval 获取</dd></div></dl></Panel>
      <Panel title="能力与评分"><div className="benchmark-tags">{(item.dimensions || []).map((value: string) => <i key={value}>{value}</i>)}</div><dl className="detail-list"><div><dt>主要指标</dt><dd>{item.primary_metric || "未声明"}</dd></div><div><dt>预计成本</dt><dd>{item.estimated_cost || "—"}</dd></div></dl></Panel>
    </div>
    <Panel title="目录配置"><pre className="json-preview">{JSON.stringify(item, null, 2)}</pre></Panel>
    </>}
    <ErrorMessage error={error} /><div className="form-actions"><button className="button secondary" onClick={onClose}>关闭</button><button className="button primary" disabled={pending} onClick={onAdd}>添加评测集</button></div>
  </Modal>;
}

function BenchmarkAssetWorkspace({ detail, loading, error, onBack, onSelectVersion, onChanged }: { detail?: EvaluationAssetDetail; loading: boolean; error: unknown; onBack: () => void; onSelectVersion: (versionId: string) => void; onChanged: (versionId?: string) => void }) {
  const [view, setView] = useState<"overview" | "data" | "versions" | "governance">("overview");
  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const remove = useMutation({
    mutationFn: (versionId: string) => api<void>(`/catalog/evaluation-datasets/${encodeURIComponent(versionId)}`, { method: "DELETE" }),
    onSuccess: () => onChanged(),
  });
  if (loading) return <main className="system-page benchmark-asset-workspace"><button className="button ghost" onClick={onBack}><ArrowLeft size={16} />返回评测集</button><div className="quiet-message">正在读取评测集…</div></main>;
  if (!detail) return <main className="system-page benchmark-asset-workspace"><button className="button ghost" onClick={onBack}><ArrowLeft size={16} />返回评测集</button><ErrorMessage error={error || new Error("评测集不存在")} /></main>;
  const item = detail.version;
  const standard = item.source_type === "benchmark";
  const deleteLabel = "删除版本";
  return <main className="system-page benchmark-asset-workspace">
    <header className="benchmark-asset-header">
      <div><button className="button ghost small" onClick={onBack}><ArrowLeft size={15} />评测集</button><p className="eyebrow">EVALUATION ASSET</p><h1>{item.name}</h1><p>{standard ? `内置目录 · ${item.benchmark_config?.task_id}` : `${evaluationSource(item)} · ${item.task_type}`} · 当前查看 V{item.version}</p></div>
      <div className="benchmark-asset-actions">{!standard && <button className="button primary" onClick={() => setEditOpen(true)}><Pencil size={15} />编辑并创建新版本</button>}<button className="button danger" disabled={!detail.can_delete} onClick={() => { setConfirmation(""); setDeleteOpen(true); }}><Trash2 size={15} />{deleteLabel}</button></div>
    </header>

    <div className="benchmark-version-strip" aria-label="评测集版本历史">{detail.versions.map((version) => <button key={version.id} className={version.id === item.id ? "active" : ""} onClick={() => onSelectVersion(version.id)}><b>V{version.version}</b><span>{formatDate(version.created_at)}</span></button>)}</div>
    <nav className="benchmark-asset-tabs" aria-label="评测集管理视图"><button className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}>概要</button><button className={view === "data" ? "active" : ""} onClick={() => setView("data")}>数据查看</button><button className={view === "versions" ? "active" : ""} onClick={() => setView("versions")}>版本历史</button><button className={view === "governance" ? "active" : ""} onClick={() => setView("governance")}>配置与引用</button></nav>

    {view === "overview" && <div className="benchmark-asset-content">
      <div className="benchmark-asset-metrics"><article><span>当前版本</span><strong>V{item.version}</strong><small>{detail.versions.length} 个不可变版本</small></article><article><span>样本</span><strong>{standard ? "运行时" : formatNumber(item.sample_count)}</strong><small>{standard ? "由 lm-eval 获取" : "冻结有效样本"}</small></article><article><span>引用</span><strong>{detail.references.length}</strong><small>{detail.can_delete ? "当前可安全删除" : "删除被下游阻止"}</small></article><article><span>制品</span><strong>{detail.artifact ? formatBytes(detail.artifact.size_bytes) : "定义"}</strong><small>{detail.artifact?.media_type || "无托管题目快照"}</small></article></div>
      <div className="benchmark-detail-grid"><Panel title="资产身份"><dl className="detail-list"><div><dt>来源类型</dt><dd>{evaluationSource(item)}</dd></div><div><dt>任务类型</dt><dd>{item.task_type}</dd></div><div><dt>许可证</dt><dd>{item.license}</dd></div><div><dt>所有者</dt><dd>{item.global_managed ? "全局评测集" : item.origin_project_name || "历史项目"}</dd></div></dl></Panel><Panel title="能力与治理"><div className="benchmark-tags">{(item.dimensions || []).map((value: string) => <i key={value}>{value}</i>)}{(item.contamination_tags || []).map((value: string) => <i key={value}>{value}</i>)}</div><dl className="detail-list"><div><dt>评分指标</dt><dd>{item.benchmark_config?.primary_metric || item.scoring_config?.metric || "未声明"}</dd></div><div><dt>兼容性哈希</dt><dd><code>{String(item.compatibility_hash || "—").slice(0, 24)}</code></dd></div></dl></Panel></div>
      {!detail.can_delete && <Alert severity="blocker" title="这个版本不能删除" detail={`仍有 ${detail.references.length} 个项目绑定、评测套件或历史运行引用。请先在对应项目解除引用。`} />}
    </div>}

    {view === "data" && <div className="benchmark-asset-content"><DatasetTableViewer key={item.id} version={item} datasetName={item.name} queryEndpoint={`/catalog/evaluation-datasets/${encodeURIComponent(item.id)}/viewer/query`} enableLabels={false} /></div>}

    {view === "versions" && <div className="benchmark-asset-content"><section className="benchmark-version-history"><header><div><span>IMMUTABLE HISTORY</span><h2>版本历史</h2></div><small>{detail.versions.length} 个版本</small></header>{detail.versions.map((version) => <article key={version.id} className={version.id === item.id ? "active" : ""}><div className="evaluation-version">V{version.version}</div><div><strong>{version.task_type} · {version.license}</strong><span>{formatDate(version.created_at)} · {formatNumber(version.sample_count)} 条样本</span></div><div className="benchmark-tags">{(version.dimensions || []).slice(0, 4).map((value: string) => <i key={value}>{value}</i>)}</div><button className="button ghost small" disabled={version.id === item.id} onClick={() => onSelectVersion(version.id)}>{version.id === item.id ? "正在查看" : "查看版本"}</button></article>)}</section></div>}

    {view === "governance" && <div className="benchmark-asset-content benchmark-governance-grid"><Panel title="引用关系" description="删除保护扫描项目绑定、评测套件、运行和任务快照。">{detail.references.length ? <div className="benchmark-reference-list">{detail.references.map((reference) => <article key={`${reference.kind}:${reference.id}`}><Link2 size={16} /><div><strong>{reference.label}</strong><span>{reference.kind} · {reference.purpose || reference.status || reference.project_name || "已引用"}</span></div></article>)}</div> : <EmptyState title="没有下游引用" description="当前版本可以安全删除；内容寻址制品不会随元数据版本删除。" />}</Panel><Panel title="冻结配置"><pre className="json-preview">{JSON.stringify({ source_config: item.source_config, schema_config: item.schema_config, scoring_config: item.scoring_config, benchmark_config: item.benchmark_config, artifact: detail.artifact }, null, 2)}</pre></Panel></div>}

    {editOpen && <BenchmarkAssetVersionModal item={item} onClose={() => setEditOpen(false)} onCreated={(versionId) => { setEditOpen(false); onChanged(versionId); }} />}
    {deleteOpen && <Modal title={deleteLabel} description={`${item.name} · V${item.version}`} onClose={() => setDeleteOpen(false)}><div className="form-grid"><Alert severity="blocker" title="该操作会删除版本元数据" detail="不会删除内容寻址制品，但版本 ID 将不能再被绑定。请输入资产名称确认。" /><Field label="输入资产名称确认"><input autoFocus value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></Field><ErrorMessage error={remove.error} /><div className="form-actions"><button className="button secondary" onClick={() => setDeleteOpen(false)}>取消</button><button className="button danger" disabled={confirmation !== item.name || remove.isPending} onClick={() => remove.mutate(item.id)}>{remove.isPending ? "正在删除…" : deleteLabel}</button></div></div></Modal>}
  </main>;
}

function BenchmarkAssetVersionModal({ item, onClose, onCreated }: { item: any; onClose: () => void; onCreated: (versionId: string) => void }) {
  const create = useMutation({
    mutationFn: (data: any) => post<EvaluationAssetDetail>(`/catalog/evaluation-datasets/${encodeURIComponent(item.id)}/versions`, data),
    onSuccess: (detail) => onCreated(detail.version.id),
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const list = (name: string) => String(form.get(name) || "").split(/[,，]/).map((value) => value.trim()).filter(Boolean);
    create.mutate({ task_type: form.get("task_type"), license: form.get("license"), dimensions: list("dimensions"), contamination_tags: list("contamination_tags"), schema_config: { ...item.schema_config, prompt_field: form.get("prompt_field"), answer_field: form.get("answer_field"), id_field: form.get("id_field"), choices_field: form.get("choices_field") }, scoring_config: { ...item.scoring_config, metric: form.get("metric") } });
  }
  return <Modal title="编辑并创建新版本" description={`${item.name} · 基于 V${item.version} 创建下一不可变版本`} onClose={onClose} wide><form className="form-grid" onSubmit={submit}><Alert severity="info" title="旧版本不会被修改" detail="来源制品和样本数量保持冻结；元数据、映射与评分配置保存为新版本。若要更换题目文件，请使用“构建评测版本”。" /><div className="form-grid two-columns"><Field label="任务类型"><select name="task_type" defaultValue={item.task_type}><option value="generation">generation</option><option value="classification">classification</option><option value="multiple_choice">multiple_choice</option></select></Field><Field label="许可证"><input name="license" required defaultValue={item.license} /></Field><Field label="能力维度"><input name="dimensions" defaultValue={(item.dimensions || []).join(", ")} /></Field><Field label="污染标签"><input name="contamination_tags" defaultValue={(item.contamination_tags || []).join(", ")} /></Field></div><Panel title="字段映射"><div className="form-grid two-columns"><Field label="Prompt 字段"><input name="prompt_field" required defaultValue={item.schema_config?.prompt_field || "prompt"} /></Field><Field label="答案字段"><input name="answer_field" required defaultValue={item.schema_config?.answer_field || "answer"} /></Field><Field label="样本 ID 字段"><input name="id_field" defaultValue={item.schema_config?.id_field || "id"} /></Field><Field label="选项字段"><input name="choices_field" defaultValue={item.schema_config?.choices_field || "choices"} /></Field></div></Panel><Field label="评分指标"><select name="metric" defaultValue={item.scoring_config?.metric || "exact_match"}><option value="exact_match">exact_match</option><option value="accuracy">accuracy</option><option value="f1">f1</option><option value="pass@1">pass@1</option><option value="llm_judge">llm_judge</option></select></Field><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="创建新版本" /></form></Modal>;
}

function BenchmarkAssetCreateModal({ datasets, initialDatasetVersionId, initialName, onClose, onCreated }: { datasets: Dataset[]; initialDatasetVersionId?: string; initialName?: string; onClose: () => void; onCreated: () => void }) {
  const [sourceMode, setSourceMode] = useState<AssetSourceMode>(initialDatasetVersionId ? "data_version" : "upload");
  const [file, setFile] = useState<File | null>(null);
  const [promptField, setPromptField] = useState("prompt");
  const [answerField, setAnswerField] = useState("answer");
  const [inspection, setInspection] = useState<EvaluationInspection | null>(null);
  const sourceVersions = datasets.flatMap((dataset) => (dataset.versions || []).filter((version: any) => version.state === "ready" && version.artifact_id).map((version: any) => ({ dataset, version })));
  const selectedVersionId = initialDatasetVersionId || "";
  const [datasetVersionId, setDatasetVersionId] = useState(selectedVersionId);
  const selectedSource = sourceVersions.find(({ version }) => version.id === datasetVersionId);
  const inspect = useMutation({
    mutationFn: ({ selectedFile, prompt, answer }: { selectedFile: File; prompt: string; answer: string }) => {
      const data = new FormData();
      data.set("file", selectedFile);
      data.set("schema_config", JSON.stringify({ prompt_field: prompt, answer_field: answer }));
      return postForm<EvaluationInspection>("/catalog/evaluation-datasets/inspect-upload", data);
    },
    onSuccess: setInspection,
  });
  const create = useMutation({ mutationFn: ({ data, form }: { data?: any; form?: FormData }) => form ? postForm("/catalog/evaluation-datasets/upload", form) : post("/catalog/evaluation-datasets/from-data-version", data), onSuccess: onCreated });

  function inspectSelected(selectedFile: File | null = file) {
    if (!selectedFile || !promptField.trim() || !answerField.trim()) return;
    setInspection(null);
    inspect.mutate({ selectedFile, prompt: promptField.trim(), answer: answerField.trim() });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (sourceMode === "upload" && (!file || !inspection || inspection.invalid_sample_count > 0 || inspection.missing_mapping_fields.length > 0)) return;
    const form = new FormData(event.currentTarget);
    const dimensions = String(form.get("dimensions") || "").split(/[,，]/).map((value) => value.trim()).filter(Boolean);
    const contaminationTags = String(form.get("contamination_tags") || "").split(/[,，]/).map((value) => value.trim()).filter(Boolean);
    const schemaConfig = { prompt_field: promptField.trim(), answer_field: answerField.trim(), id_field: String(form.get("id_field") || "").trim(), choices_field: String(form.get("choices_field") || "").trim() };
    const scoringConfig = { metric: String(form.get("metric") || "exact_match") };
    if (sourceMode === "upload") {
      form.set("file", file!);
      form.set("dimensions", JSON.stringify(dimensions));
      form.set("contamination_tags", JSON.stringify(contaminationTags));
      form.set("schema_config", JSON.stringify(schemaConfig));
      form.set("scoring_config", JSON.stringify(scoringConfig));
      create.mutate({ form });
    } else {
      create.mutate({ data: { name: form.get("name"), dataset_version_id: datasetVersionId, task_type: form.get("task_type"), license: form.get("license"), dimensions, contamination_tags: contaminationTags, schema_config: schemaConfig, scoring_config: scoringConfig } });
    }
  }

  const uploadValid = Boolean(file && inspection && inspection.invalid_sample_count === 0 && inspection.missing_mapping_fields.length === 0);
  return <Modal title="构建评测数据版本" description="上传并校验自定义题目，或把在线下载/全局数据集的一个不可变版本映射为评测集。" onClose={onClose} wide>
    <form className="form-grid" onSubmit={submit}>
      <div className="source-mode-switch" role="group" aria-label="评测数据来源"><button type="button" className={sourceMode === "upload" ? "active" : ""} onClick={() => setSourceMode("upload")}>上传并构建</button><button type="button" className={sourceMode === "data_version" ? "active" : ""} onClick={() => setSourceMode("data_version")}>从数据版本构建</button></div>
      <Field label="资产名称" hint="同名再次创建会生成递增的不可变版本"><input name="name" required defaultValue={initialName || ""} /></Field>
      {sourceMode === "upload" ? <>
        <Field label="评测文件" hint="UTF-8 JSONL、NDJSON、CSV、TSV 或 Parquet，最大 256 MiB"><input name="file" type="file" accept=".jsonl,.ndjson,.csv,.tsv,.parquet" required onChange={(event) => { const selected = event.target.files?.[0] || null; setFile(selected); setInspection(null); if (selected) inspect.mutate({ selectedFile: selected, prompt: promptField, answer: answerField }); }} /></Field>
      </> : <Field label="全局数据版本">{sourceVersions.length ? <select name="dataset_version_id" required value={datasetVersionId} onChange={(event) => setDatasetVersionId(event.target.value)}><option value="" disabled>选择不可变版本</option>{sourceVersions.map(({ dataset, version }) => <option key={version.id} value={version.id}>{(dataset.tags || []).includes("评测候选") ? "[在线] " : ""}{dataset.name} · V{version.version} · {formatNumber(version.document_count)} 条</option>)}</select> : <div className="quiet-message">当前没有具有可用数据快照的版本。请先使用“在线导入”下载，或前往数据资产导入。</div>}</Field>}

      {sourceMode === "data_version" && selectedSource && <Panel title="来源快照" description="构建只引用这个不可变 Snapshot，不复制大型正文。"><dl className="detail-list"><div><dt>数据集</dt><dd>{selectedSource.dataset.name} · V{selectedSource.version.version}</dd></div><div><dt>文档数</dt><dd>{formatNumber(selectedSource.version.document_count)}</dd></div><div><dt>大小</dt><dd>{formatBytes(selectedSource.version.raw_size_bytes)}</dd></div><div><dt>来源</dt><dd>{selectedSource.dataset.source_type}</dd></div></dl></Panel>}

      <div className="form-grid two-columns">
        <Field label="任务类型"><select name="task_type"><option value="generation">generation</option><option value="classification">classification</option><option value="multiple_choice">multiple_choice</option></select></Field>
        <Field label="许可证"><input name="license" defaultValue={sourceMode === "upload" ? "internal" : selectedSource?.dataset.license || "unknown"} /></Field>
        <Field label="能力维度"><input name="dimensions" placeholder="知识, 推理" /></Field>
        <Field label="污染标签"><input name="contamination_tags" defaultValue={sourceMode === "upload" ? "private-eval" : "public-source"} /></Field>
      </div>
      <Panel title="字段映射" description="Prompt 和答案字段必须覆盖每条上传样本；修改字段后需要重新检查。">
        <div className="form-grid two-columns">
          <Field label="Prompt 字段"><input name="prompt_field" required value={promptField} onChange={(event) => { setPromptField(event.target.value); setInspection(null); }} /></Field>
          <Field label="答案字段"><input name="answer_field" required value={answerField} onChange={(event) => { setAnswerField(event.target.value); setInspection(null); }} /></Field>
          <Field label="样本 ID 字段"><input name="id_field" defaultValue="id" /></Field>
          <Field label="选项字段"><input name="choices_field" defaultValue="choices" /></Field>
        </div>
        {sourceMode === "upload" && <div className="evaluation-inspection-actions"><button type="button" className="button secondary small" disabled={!file || inspect.isPending} onClick={() => inspectSelected()}>{inspect.isPending ? "检查中…" : "检查字段与样本"}</button><span>最终保存会执行同一套完整校验。</span></div>}
      </Panel>

      {sourceMode === "upload" && inspection && <EvaluationInspectionPanel inspection={inspection} />}
      <Field label="评分指标"><select name="metric" defaultValue="exact_match"><option value="exact_match">exact_match</option><option value="accuracy">accuracy</option><option value="f1">f1</option><option value="pass@1">pass@1</option><option value="llm_judge">llm_judge</option></select></Field>
      <Alert severity="info" title="创建的是全局评测集" detail="这里不选择模型也不发起评测。项目绑定此版本后，才在项目评测中心创建套件和运行。" />
      <ErrorMessage error={inspect.error || create.error} />
      <FormActions submitting={create.isPending} disabled={sourceMode === "upload" ? !uploadValid : !datasetVersionId} onCancel={onClose} submitLabel="冻结评测版本" />
    </form>
  </Modal>;
}

function EvaluationInspectionPanel({ inspection }: { inspection: EvaluationInspection }) {
  const valid = inspection.invalid_sample_count === 0 && inspection.missing_mapping_fields.length === 0;
  return <Panel title="文件检查结果" description={`${inspection.filename} · ${inspection.format} · ${formatBytes(inspection.size_bytes)}`}>
    <div className="evaluation-inspection-summary"><div><span>总样本</span><strong>{formatNumber(inspection.sample_count)}</strong></div><div><span>有效</span><strong>{formatNumber(inspection.valid_sample_count)}</strong></div><div className={valid ? "" : "danger"}><span>无效</span><strong>{formatNumber(inspection.invalid_sample_count)}</strong></div><div><span>字段</span><strong>{inspection.fields.length}</strong></div></div>
    <div className="benchmark-tags">{inspection.fields.map((field) => <i key={field}>{field}</i>)}</div>
    <Alert severity={valid ? "success" : "error"} title={valid ? "字段映射覆盖全部样本" : "存在无法用于评测的样本"} detail={inspection.missing_mapping_fields.length ? `文件缺少字段：${inspection.missing_mapping_fields.join("、")}` : valid ? "可以冻结为不可变评测版本。" : `${formatNumber(inspection.invalid_sample_count)} 条记录的 Prompt 或答案为空。`} />
    <div className="evaluation-preview-list">{inspection.preview.map((row, index) => <pre key={index}>{JSON.stringify(row, null, 2)}</pre>)}</div>
  </Panel>;
}
