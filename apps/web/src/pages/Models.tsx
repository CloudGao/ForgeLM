import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  Boxes,
  Braces,
  Code2,
  Edit3,
  ExternalLink,
  GitCompareArrows,
  GitFork,
  Network,
  Package,
  Play,
  Plus,
  RefreshCcw,
  Rocket,
  Search,
  Star,
  Tag,
  Upload,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, formatBytes, formatDate, formatNumber, post } from "../api";
import { ModelSpecificationEditor } from "../components/ModelSpecificationEditor";
import { ModelDeletionDialog, ModelDeletionMenu, type ModelDeletionMode } from "../components/ModelDeletionDialog";
import {
  Alert,
  EmptyState,
  ErrorMessage,
  Field,
  FormActions,
  MetricCard,
  Modal,
  PageHeader,
  Panel,
  StatusBadge,
} from "../components/ui";
import type {
  Checkpoint,
  CodeSource,
  CodeVersion,
  ModelArchitecture,
  ModelRegistry,
  ModelSpecification,
  ModelVersion,
  Worker,
} from "../types";

type RegistryTab = "models" | "lineage" | "checkpoints" | "specifications";
type ModelAction = "mark-candidate" | "mark-best" | "unmark-best" | "promote" | "archive" | "restore";

function requestedRegistryTab(value: string | null): RegistryTab {
  return value === "lineage" || value === "checkpoints" || value === "specifications" ? value : "models";
}

const importExample = `{
  "source_key": "existing-model-registry",
  "artifacts": [{
    "ref": "model-v1",
    "kind": "model",
    "uri": "file:///data/models/example-v1",
    "content_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    "size_bytes": 0,
    "availability_state": "available"
  }],
  "training_runs": [{
    "external_id": "imported-models",
    "name": "历史模型登记",
    "models": [{
      "external_id": "example-v1",
      "name": "ExampleLM",
      "version": 1,
      "kind": "base",
      "artifact_ref": "model-v1",
      "status": "candidate"
    }]
  }]
}`;

function canRunModel(model: ModelVersion): boolean {
  const type = model.inference_source?.type;
  return Boolean(type && !["missing", "unavailable", "unsupported"].includes(type));
}

function canUsePlayground(model: ModelVersion): boolean {
  return canRunModel(model) && model.inference_source?.type !== "api";
}

function inferenceLabel(model: ModelVersion): string {
  const source = model.inference_source || {};
  if (source.type === "pretrained") return `HF · ${source.value}`;
  if (source.type === "managed") return "托管模型制品";
  if (source.type === "external") return source.globally_accessible ? "外部共享位置" : `外部位置 · ${source.worker_ids?.length || 0} Worker`;
  if (source.type === "command") return source.display_name || "项目原生推理";
  if (source.type === "native") return source.display_name || "内置原生推理";
  if (source.type === "api") return `评测 API · ${source.endpoint_model || "OpenAI 兼容模型"}`;
  if (source.type === "unsupported") return "原始 Checkpoint";
  if (source.type === "unavailable") return "制品位置不可用";
  return "未配置推理来源";
}

function modelSourceLabel(model: ModelVersion): string {
  if (model.source_checkpoint) return `Checkpoint ${model.source_checkpoint.id.slice(0, 8)} · Step ${formatNumber(model.source_checkpoint.step)}`;
  const source = model.metadata_json?.registry_source;
  if (source?.type === "huggingface") return `${source.repository_id} @ ${String(source.revision || "").slice(0, 12)}`;
  if (source?.type === "external_artifact") return "外部模型制品";
  if (source?.type === "openai_compatible") return `OpenAI API · ${source.endpoint?.model || "外部模型"}`;
  if (model.parent_model) return `${model.parent_model.name} V${model.parent_model.version}`;
  return model.artifact_id ? `制品 ${model.artifact_id.slice(0, 8)}` : "外部登记";
}

export function Models() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<RegistryTab>(() => requestedRegistryTab(searchParams.get("tab")));
  const [importOpen, setImportOpen] = useState(false);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [registrationOpen, setRegistrationOpen] = useState(false);
  const [registrationCheckpointId, setRegistrationCheckpointId] = useState("");
  const [createSpecificationOpen, setCreateSpecificationOpen] = useState(false);
  const [selectedSpecificationId, setSelectedSpecificationId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(searchParams.get("model"));
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string | null>(searchParams.get("checkpoint"));
  const [manifest, setManifest] = useState(importExample);
  const [deletion, setDeletion] = useState<{ model: ModelVersion; mode: ModelDeletionMode } | null>(null);
  const [showDeleted, setShowDeleted] = useState(false);

  const registry = useQuery({
    queryKey: ["models", projectId, "all", showDeleted],
    queryFn: () => api<ModelRegistry>(`/projects/${projectId}/models?include_archived=true${showDeleted ? "&include_deleted=true" : ""}`),
    refetchInterval: 3000,
  });
  const specifications = useQuery({
    queryKey: ["model-specifications", projectId],
    queryFn: () => api<ModelSpecification[]>(`/projects/${projectId}/model-specifications`),
  });
  const architectureCatalog = useQuery({
    queryKey: ["model-architectures"],
    queryFn: () => api<{ architectures: ModelArchitecture[] }>("/model-architectures"),
  });
  const codeSources = useQuery({
    queryKey: ["code-sources", projectId],
    queryFn: () => api<CodeSource[]>(`/projects/${projectId}/code-sources`),
  });
  const tokenizers = useQuery({
    queryKey: ["tokenizers", projectId],
    queryFn: () => api<any[]>(`/projects/${projectId}/tokenizers`),
  });
  const workers = useQuery({
    queryKey: ["workers"],
    queryFn: () => api<Worker[]>("/workers"),
    refetchInterval: 5000,
  });
  const codeVersions: Array<CodeVersion & { source_name: string }> = (codeSources.data || [])
    .flatMap((source) => source.versions.map((version) => ({ ...version, source_name: source.name })))
    .filter((version) => version.state === "ready");
  const action = useMutation({
    mutationFn: ({ id, action: actionName }: { id: string; action: ModelAction }) => post(`/models/${id}/${actionName}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models", projectId] }),
  });
  const historyImport = useMutation({
    mutationFn: () => post(`/projects/${projectId}/history-imports`, JSON.parse(manifest)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models", projectId] });
      queryClient.invalidateQueries({ queryKey: ["runs", projectId] });
      queryClient.invalidateQueries({ queryKey: ["tokenizers", projectId] });
      setImportOpen(false);
    },
  });
  const models = registry.data?.models || [];
  const checkpoints = registry.data?.checkpoints || [];
  const selectedModel = models.find((item) => item.id === selectedModelId);
  const selectedCheckpoint = checkpoints.find((item) => item.id === selectedCheckpointId);
  const selectedSpecification = (specifications.data || []).find((item) => item.id === selectedSpecificationId);

  useEffect(() => {
    const modelId = searchParams.get("model");
    const checkpointId = searchParams.get("checkpoint");
    setSelectedModelId(modelId);
    setSelectedCheckpointId(checkpointId);
    setTab(modelId ? "models" : checkpointId ? "checkpoints" : requestedRegistryTab(searchParams.get("tab")));
  }, [searchParams]);

  function selectTab(nextTab: RegistryTab) {
    const next = new URLSearchParams(searchParams);
    next.set("tab", nextTab);
    next.delete("model");
    next.delete("checkpoint");
    setSearchParams(next);
    setTab(nextTab);
    setSelectedModelId(null);
    setSelectedCheckpointId(null);
  }

  function openModel(modelId: string) {
    const next = new URLSearchParams(searchParams);
    next.set("tab", "models");
    next.set("model", modelId);
    next.delete("checkpoint");
    setSearchParams(next);
    setTab("models");
    setSelectedModelId(modelId);
    setSelectedCheckpointId(null);
  }

  function closeModel() {
    const next = new URLSearchParams(searchParams);
    next.delete("model");
    setSearchParams(next, { replace: true });
    setSelectedModelId(null);
  }

  function openCheckpoint(checkpointId: string) {
    const next = new URLSearchParams(searchParams);
    next.set("tab", "checkpoints");
    next.set("checkpoint", checkpointId);
    next.delete("model");
    setSearchParams(next);
    setTab("checkpoints");
    setSelectedCheckpointId(checkpointId);
    setSelectedModelId(null);
  }

  function closeCheckpoint() {
    const next = new URLSearchParams(searchParams);
    next.delete("checkpoint");
    setSearchParams(next, { replace: true });
    setSelectedCheckpointId(null);
  }

  function refreshSpecifications() {
    queryClient.invalidateQueries({ queryKey: ["model-specifications", projectId] });
    queryClient.invalidateQueries({ queryKey: ["plans", projectId] });
  }

  function openRegistration(checkpointId = "") {
    setRegistrationCheckpointId(checkpointId);
    setRegistrationOpen(true);
  }

  let tabContent;
  if (tab === "models") {
    tabContent = <ModelRepository
      key={String(showDeleted)}
      initialStatus={showDeleted ? "all" : "active"}
      models={models}
      onDelete={(model, mode) => setDeletion({ model, mode })}
      actionError={action.error}
      onAction={(id, actionName) => action.mutate({ id, action: actionName })}
      onOpen={openModel}
      onRegister={() => openRegistration()}
      onNavigate={navigate}
    />;
  } else if (tab === "lineage") {
    tabContent = <ModelLineage
      models={models.filter((model) => model.status !== "archived" && model.status !== "deleted")}
      onOpen={openModel}
      onAction={(id, actionName) => action.mutate({ id, action: actionName })}
      onNavigate={navigate}
    />;
  } else if (tab === "checkpoints") {
    tabContent = <CheckpointRepository
      checkpoints={checkpoints}
      onOpen={openCheckpoint}
      onRegister={openRegistration}
      onNavigate={navigate}
    />;
  } else {
    tabContent = <ModelSpecifications
      specifications={specifications.data || []}
      architectures={architectureCatalog.data?.architectures || []}
      onOpen={setSelectedSpecificationId}
      onCreate={() => setCreateSpecificationOpen(true)}
    />;
  }

  return <>
    <PageHeader
      eyebrow="MODEL REGISTRY"
      title="模型与 Checkpoint"
      description="管理可供评测、推理、量化和发布稳定引用的模型版本，并追溯训练 Checkpoint 与模型规格。"
      actions={<>
        <button className="button primary" onClick={() => openRegistration()}><Plus size={16} />登记模型版本</button>
        {tab === "specifications" && <button className="button secondary" onClick={() => setCreateSpecificationOpen(true)}><Boxes size={16} />新建模型规格</button>}
        <button className="button secondary" onClick={() => setImportOpen(true)}><Upload size={16} />批量接管历史</button>
        <button className="button secondary" disabled={models.length < 2} title={models.length < 2 ? "至少需要两个模型版本" : undefined} onClick={() => setComparisonOpen(true)}><GitCompareArrows size={16} />比较版本</button>
      </>}
    />
    <div className="segmented-tabs" role="tablist" aria-label="模型仓库视图">
      <button role="tab" aria-selected={tab === "models"} className={tab === "models" ? "active" : ""} onClick={() => selectTab("models")}><Package size={16} />模型版本 <b>{models.length}</b></button>
      <button role="tab" aria-selected={tab === "lineage"} className={tab === "lineage" ? "active" : ""} onClick={() => selectTab("lineage")}><Network size={16} />模型血缘 <b>{models.filter((item) => item.status !== "archived").length}</b></button>
      <button role="tab" aria-selected={tab === "checkpoints"} className={tab === "checkpoints" ? "active" : ""} onClick={() => selectTab("checkpoints")}><Tag size={16} />Checkpoint <b>{checkpoints.length}</b></button>
      <button role="tab" aria-selected={tab === "specifications"} className={tab === "specifications" ? "active" : ""} onClick={() => selectTab("specifications")}><Boxes size={16} />模型规格 <b>{specifications.data?.length || 0}</b></button>
    </div>
    <ErrorMessage error={registry.error} />
    <label className="check-field"><input type="checkbox" checked={showDeleted} onChange={event => setShowDeleted(event.target.checked)} /><span>显示已删除版本的历史记录</span></label>
    {tabContent}

    {registrationOpen && <RegisterModelModal
      projectId={projectId}
      models={models.filter(model => model.status !== "deleted")}
      checkpoints={checkpoints}
      tokenizers={tokenizers.data || []}
      workers={workers.data || []}
      initialCheckpointId={registrationCheckpointId}
      onClose={() => setRegistrationOpen(false)}
      onCreated={(model) => {
        queryClient.invalidateQueries({ queryKey: ["models", projectId] });
        setRegistrationOpen(false);
        openModel(model.id);
      }}
    />}
    {selectedModel && <ModelDetailModal
      model={selectedModel}
      onDelete={(mode) => { closeModel(); setDeletion({ model: selectedModel, mode }); }}
      onClose={closeModel}
      onAction={(actionName) => action.mutate({ id: selectedModel.id, action: actionName })}
      actionError={action.error}
      onNavigate={navigate}
    />}
    {deletion && <ModelDeletionDialog model={deletion.model} mode={deletion.mode} onClose={() => {
      setDeletion(null);
      queryClient.invalidateQueries({ queryKey: ["models", projectId] });
    }} />}
    {selectedCheckpoint && <CheckpointDetailModal
      checkpoint={selectedCheckpoint}
      onClose={closeCheckpoint}
      onRegister={() => { closeCheckpoint(); openRegistration(selectedCheckpoint.id); }}
      onNavigate={navigate}
    />}
    {createSpecificationOpen && <Modal title="新建模型规格" description="用可视化字段配置成熟架构，保存前查看平台解析的 JSON 和代码。" onClose={() => setCreateSpecificationOpen(false)} wide><ModelSpecificationEditor projectId={projectId} architectures={architectureCatalog.data?.architectures || []} codeVersions={codeVersions} onCancel={() => setCreateSpecificationOpen(false)} onSaved={() => { refreshSpecifications(); setCreateSpecificationOpen(false); }} /></Modal>}
    {selectedSpecification && <ModelSpecificationDetail specification={selectedSpecification} architectures={architectureCatalog.data?.architectures || []} codeVersions={codeVersions} onClose={() => setSelectedSpecificationId(null)} onSaved={(updated) => { refreshSpecifications(); setSelectedSpecificationId(updated.id); }} />}
    {importOpen && <Modal title="批量接管历史" description="用清单登记外部制品、训练运行、Checkpoint、模型、评测和门禁；不会复制外部文件。" onClose={() => setImportOpen(false)} wide><form onSubmit={(event) => { event.preventDefault(); historyImport.mutate(); }}><Field label="历史清单 JSON" hint="单模型请使用“登记模型版本”；这里适合完整历史的幂等增量接管"><textarea rows={20} value={manifest} onChange={(event) => setManifest(event.target.value)} /></Field><ErrorMessage error={historyImport.error} /><FormActions submitting={historyImport.isPending} onCancel={() => setImportOpen(false)} submitLabel="验证并导入" /></form></Modal>}
    {comparisonOpen && <ModelComparison models={models} onClose={() => setComparisonOpen(false)} />}
  </>;
}

function ModelRepository({ models, actionError, onAction, onOpen, onRegister, onNavigate, onDelete, initialStatus }: { models: ModelVersion[]; actionError: unknown; onAction: (id: string, action: ModelAction) => void; onOpen: (id: string) => void; onRegister: () => void; onNavigate: (to: string) => void; onDelete: (model: ModelVersion, mode: ModelDeletionMode) => void; initialStatus: string }) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState(initialStatus);
  const [kind, setKind] = useState("all");
  const kinds = useMemo(() => Array.from(new Set(models.map((model) => model.kind))).sort(), [models]);
  const filtered = useMemo(() => models.filter((model) => {
    const haystack = `${model.name} ${model.kind} ${model.version} ${model.id}`.toLowerCase();
    if (search.trim() && !haystack.includes(search.trim().toLowerCase())) return false;
    if (status === "active" && ["archived", "deleted"].includes(model.status)) return false;
    if (status !== "active" && status !== "all" && model.status !== status) return false;
    return kind === "all" || model.kind === kind;
  }), [kind, models, search, status]);
  const active = models.filter((model) => model.status !== "archived" && model.status !== "deleted");
  const runnable = active.filter(canRunModel);

  return <div className="model-registry-workspace">
    <div className="metrics-grid four model-registry-summary">
      <MetricCard label="当前正式" value={active.filter((model) => model.is_official).length} detail={active.find((model) => model.is_official)?.name || "尚未晋升"} tone="accent" />
      <MetricCard label="候选版本" value={active.filter((model) => model.is_candidate).length} detail="等待评测或晋升" />
      <MetricCard label="可用于推理" value={runnable.length} detail="具有可访问权重来源" />
      <MetricCard label="全部版本" value={models.length} detail={`${models.filter((model) => model.status === "archived").length} 个已归档`} />
    </div>
    <Panel className="model-registry-panel" title="模型版本仓库" description="模型版本是下游稳定引用；模型规格和训练 Checkpoint 分别在独立页签管理。">
      <div className="model-registry-toolbar">
        <label className="search-box model-registry-search"><Search size={15} /><input aria-label="搜索模型" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称、类型、版本或 ID" /></label>
        <select aria-label="模型状态" value={status} onChange={(event) => setStatus(event.target.value)}><option value="active">活跃版本</option><option value="all">全部状态</option><option value="official">正式</option><option value="candidate">候选</option><option value="registered">已登记</option><option value="archived">已归档</option><option value="deleted">已删除</option></select>
        <select aria-label="模型类型" value={kind} onChange={(event) => setKind(event.target.value)}><option value="all">全部类型</option>{kinds.map((value) => <option value={value} key={value}>{value}</option>)}</select>
        <span>{filtered.length} / {models.length} 个版本</span>
      </div>
      <ErrorMessage error={actionError} />
      {filtered.length ? <div className="model-registry-list">{filtered.map((model) => <article className={model.is_official ? "official" : model.status === "archived" ? "archived" : ""} key={model.id}>
        <button className="model-registry-main" onClick={() => onOpen(model.id)} aria-label={`查看 ${model.name} V${model.version}`}>
          <span className="model-version-mark">{model.kind.slice(0, 4).toUpperCase()}</span>
          <span className="model-registry-identity"><strong>{model.name} <small>V{model.version}</small></strong><span>{model.id}</span><i>{model.metadata_json?.description || modelSourceLabel(model)}</i></span>
        </button>
        <div className="model-registry-source"><span>来源</span><strong>{modelSourceLabel(model)}</strong><small>{model.artifact ? `${formatBytes(model.artifact.size_bytes)} · ${model.artifact.content_hash.slice(0, 10)}` : "未绑定本地制品"}</small></div>
        <div className="model-registry-runtime"><span>运行可用性</span><strong className={canRunModel(model) ? "ready" : "muted"}>{inferenceLabel(model)}</strong><small>{model.tokenizer ? `${model.tokenizer.name} V${model.tokenizer.version}` : "Tokenizer 未登记"}</small></div>
        <div className="model-registry-flags">{model.is_best && <span className="best-flag"><Star size={13} />最佳</span>}<StatusBadge status={model.status} /></div>
        <div className="model-registry-actions">
          <ModelDeletionMenu onSelect={mode => onDelete(model, mode)} />
          <button className="button ghost small" onClick={() => onOpen(model.id)}>详情</button>
          {!model.is_candidate && !model.is_official && model.status !== "archived" && model.status !== "deleted" && <button className="button secondary small" onClick={() => onAction(model.id, "mark-candidate")}>标记候选</button>}
          {model.is_best
            ? <button className="button secondary small" onClick={() => onAction(model.id, "unmark-best")}><Star size={13} />取消最佳</button>
            : model.status !== "archived" && model.status !== "deleted" && <button className="button secondary small" onClick={() => onAction(model.id, "mark-best")}><Star size={13} />设为最佳</button>}
          {!model.is_official && model.status !== "archived" && model.status !== "deleted" && <button className="button primary small" onClick={() => onAction(model.id, "promote")}><Rocket size={13} />晋升</button>}
          {model.status === "archived" ? <button className="button secondary small" onClick={() => onAction(model.id, "restore")}><RefreshCcw size={13} />恢复</button> : !model.is_official && model.status !== "deleted" && <button className="button ghost small" onClick={() => onAction(model.id, "archive")}><Archive size={13} />归档</button>}
          <button className="button ghost small" disabled={!canUsePlayground(model)} title={!canUsePlayground(model) ? model.inference_source?.type === "api" ? "外部评测端点首版不接入 Playground" : inferenceLabel(model) : undefined} onClick={() => onNavigate("../playground")}><Play size={13} />Playground</button>
        </div>
      </article>)}</div> : <EmptyState title={models.length ? "没有匹配的模型" : "模型仓库为空"} description={models.length ? "调整搜索或筛选条件。" : "登记 Hugging Face、OpenAI 兼容端点、外部制品或已有 Checkpoint，创建第一个稳定模型版本。"} action={!models.length ? <button className="button primary" onClick={onRegister}><Plus size={16} />登记模型版本</button> : undefined} />}
    </Panel>
  </div>;
}

function ModelLineage({ models, onOpen, onAction, onNavigate }: { models: ModelVersion[]; onOpen: (id: string) => void; onAction: (id: string, action: ModelAction) => void; onNavigate: (to: string) => void }) {
  const modelIds = new Set(models.map((model) => model.id));
  const children = new Map<string, ModelVersion[]>();
  models.forEach((model) => {
    if (!model.parent_model_version_id || !modelIds.has(model.parent_model_version_id)) return;
    children.set(model.parent_model_version_id, [...(children.get(model.parent_model_version_id) || []), model]);
  });
  const roots = models.filter((model) => !model.parent_model_version_id || !modelIds.has(model.parent_model_version_id));
  const renderNode = (model: ModelVersion, visited: Set<string>): any => {
    if (visited.has(model.id)) return null;
    const nextVisited = new Set(visited).add(model.id);
    return <div className="lineage-branch" key={model.id}>
      <article className={`model-node ${model.is_official ? "official-node" : ""}`}>
        <button className="model-node-main" onClick={() => onOpen(model.id)} aria-label={`查看血缘节点 ${model.name} V${model.version}`}><span className="model-kind">{model.kind.toUpperCase()}</span><span><small>{model.parameter_count ? `${formatNumber(model.parameter_count)} 参数` : "参数量待登记"}</small><strong>{model.name} <i>V{model.version}</i></strong><em>{modelSourceLabel(model)}</em>{model.source_checkpoint && <em>Step {formatNumber(model.source_checkpoint.step)} · {formatNumber(model.source_checkpoint.token_count)} Tokens · Loss {model.source_checkpoint.metrics?.loss ?? "—"}</em>}</span></button>
        <div className="model-node-flags">{model.is_best && <span><Star size={14} />最佳</span>}<StatusBadge status={model.status} /></div>
        <div className="model-node-actions"><button className="button ghost small" disabled title="从任意模型创建分叉计划的控制面契约尚未实现"><GitFork size={14} />继续训练</button><button className="button ghost small" disabled={!canUsePlayground(model)} title={!canUsePlayground(model) ? model.inference_source?.type === "api" ? "外部评测端点首版不接入 Playground" : inferenceLabel(model) : undefined} onClick={() => onNavigate("../playground")}><Play size={14} />Playground</button>{model.is_best ? <button className="button secondary small" onClick={() => onAction(model.id, "unmark-best")}><Star size={14} />取消最佳</button> : <button className="button secondary small" onClick={() => onAction(model.id, "mark-best")}><Star size={14} />设为最佳</button>}{!model.is_official && <button className="button primary small" onClick={() => onAction(model.id, "promote")}><Rocket size={14} />晋升正式</button>}</div>
      </article>
      {(children.get(model.id) || []).map((child) => <div className="lineage-child" key={child.id}><span className="lineage-connector">└─ 派生版本</span>{renderNode(child, nextVisited)}</div>)}
    </div>;
  };
  return <Panel className="lineage-panel" title="模型血缘" description="父模型与来源 Checkpoint 构成不可变血缘；训练配置和完整元数据可从节点详情检查。">{models.length ? <div className="lineage-tree">{roots.map((model) => renderNode(model, new Set()))}</div> : <EmptyState title="还没有模型血缘" description="训练输出或模型登记完成后会在这里形成稳定节点。" />}</Panel>;
}

function CheckpointRepository({ checkpoints, onOpen, onRegister, onNavigate }: { checkpoints: Checkpoint[]; onOpen: (id: string) => void; onRegister: (id: string) => void; onNavigate: (to: string) => void }) {
  return <Panel className="table-panel" title="训练 Checkpoint" description="Checkpoint 用于恢复训练；登记为模型版本不会自动把原始训练目录转换成 Transformers 模型包。">
    {checkpoints.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Checkpoint</th><th>训练运行</th><th>训练位置</th><th>指标</th><th>制品</th><th>创建时间</th><th>操作</th></tr></thead><tbody>{checkpoints.map((checkpoint: any) => <tr key={checkpoint.id}><td><button className="table-link" onClick={() => onOpen(checkpoint.id)}><strong>{checkpoint.id.slice(0, 8)}</strong><StatusBadge status={checkpoint.status} /></button></td><td><strong>{checkpoint.training_run?.current_stage || checkpoint.training_run_id.slice(0, 8)}</strong><span>{checkpoint.training_run?.status || "历史运行"}</span></td><td><strong>Step {formatNumber(checkpoint.step)}</strong><span>{formatNumber(checkpoint.token_count)} Tokens</span></td><td><strong>Loss {checkpoint.metrics?.loss ?? "—"}</strong><span>{Object.keys(checkpoint.metrics || {}).length} 个指标</span></td><td><strong>{checkpoint.artifact ? formatBytes(checkpoint.artifact.size_bytes) : "待登记"}</strong><span>{checkpoint.artifact?.content_hash?.slice(0, 12) || "没有权重制品"}</span></td><td>{formatDate(checkpoint.created_at)}</td><td><div className="checkpoint-row-actions"><button className="button ghost small" onClick={() => onNavigate(`../training?run=${checkpoint.training_run_id}`)}>查看训练</button><button className="button secondary small" disabled={checkpoint.status !== "available" || !checkpoint.artifact_id} title={!checkpoint.artifact_id ? "Checkpoint 没有权重制品" : checkpoint.status !== "available" ? "Checkpoint 当前不可用" : undefined} onClick={() => onRegister(checkpoint.id)}>登记模型</button></div></td></tr>)}</tbody></table></div> : <EmptyState title="没有 Checkpoint" description="训练引擎成功登记可恢复权重后会出现在这里。" />}
  </Panel>;
}

function RegisterModelModal({ projectId, models, checkpoints, tokenizers, workers, initialCheckpointId, onClose, onCreated }: { projectId: string; models: ModelVersion[]; checkpoints: Checkpoint[]; tokenizers: any[]; workers: Worker[]; initialCheckpointId: string; onClose: () => void; onCreated: (model: ModelVersion) => void }) {
  const eligibleCheckpoints = checkpoints.filter((checkpoint) => checkpoint.status === "available" && checkpoint.artifact_id);
  const [sourceType, setSourceType] = useState<"huggingface" | "checkpoint" | "external_artifact" | "openai_compatible">(initialCheckpointId ? "checkpoint" : "huggingface");
  const [endpointMode, setEndpointMode] = useState<"completions" | "chat_completions">("completions");
  const create = useMutation({ mutationFn: (payload: any) => post<ModelVersion>(`/projects/${projectId}/models`, payload), onSuccess: onCreated });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const versionText = String(form.get("version") || "").trim();
    const description = String(form.get("description") || "").trim();
    const payload: any = {
      name: String(form.get("name") || "").trim(),
      kind: String(form.get("kind") || "base"),
      source_type: sourceType,
      parent_model_version_id: form.get("parent_model_version_id") || null,
      tokenizer_version_id: form.get("tokenizer_version_id") || null,
      parameter_count: Number(form.get("parameter_count") || 0),
      is_candidate: form.get("is_candidate") === "on",
      metadata: description ? { description } : {},
      ...(versionText ? { version: Number(versionText) } : {}),
    };
    if (sourceType === "huggingface") {
      payload.huggingface_id = String(form.get("huggingface_id") || "").trim();
      payload.huggingface_revision = String(form.get("huggingface_revision") || "").trim();
    } else if (sourceType === "checkpoint") {
      payload.source_checkpoint_id = String(form.get("source_checkpoint_id") || "");
    } else if (sourceType === "external_artifact") {
      payload.artifact = {
        kind: "model",
        uri: String(form.get("artifact_uri") || "").trim(),
        content_hash: String(form.get("content_hash") || "").trim(),
        size_bytes: Number(form.get("size_bytes") || 0),
        media_type: String(form.get("media_type") || "application/x-directory"),
        availability_state: String(form.get("availability_state") || "unknown"),
        worker_id: form.get("worker_id") || null,
      };
    } else {
      payload.endpoint_url = String(form.get("endpoint_url") || "").trim();
      payload.endpoint_model = String(form.get("endpoint_model") || "").trim();
      payload.endpoint_mode = endpointMode;
      payload.endpoint_tokenizer = endpointMode === "completions" ? String(form.get("endpoint_tokenizer") || "").trim() : null;
      payload.endpoint_auth_env_var = String(form.get("endpoint_auth_env_var") || "").trim() || null;
      payload.endpoint_num_concurrent = Number(form.get("endpoint_num_concurrent") || 1);
      payload.endpoint_timeout_seconds = Number(form.get("endpoint_timeout_seconds") || 300);
      payload.endpoint_verify_tls = form.get("endpoint_verify_tls") === "on";
    }
    create.mutate(payload);
  }

  return <Modal title="登记模型版本" description="创建不可变 ModelVersion，供评测、推理、量化和发布稳定引用。" onClose={onClose} wide>
    <form className="model-registration-form" onSubmit={submit}>
      <section className="model-registration-section"><header><span>1</span><div><h3>模型身份</h3><p>同名版本号留空时自动递增；保存后身份和权重来源不原地修改。</p></div></header><div className="form-grid three-columns">
        <Field label="模型名称"><input name="name" required placeholder="ExampleLM" /></Field>
        <Field label="版本号" hint="留空自动分配下一版本"><input name="version" type="number" min="1" placeholder="自动" /></Field>
        <Field label="模型类型"><select name="kind" defaultValue="base"><option value="base">Base</option><option value="sft">SFT</option><option value="dpo">DPO</option><option value="continued-pretrain">Continued pretrain</option><option value="merged">Merged</option></select></Field>
        <Field label="参数量" hint="未知可填 0"><input name="parameter_count" type="number" min="0" defaultValue="0" /></Field>
        <Field label="父模型"><select name="parent_model_version_id" defaultValue=""><option value="">无 / 外部根节点</option>{models.map((model) => <option value={model.id} key={model.id}>{model.name} V{model.version} · {model.status}</option>)}</select></Field>
        <Field label="Tokenizer"><select name="tokenizer_version_id" defaultValue=""><option value="">未登记</option>{tokenizers.map((tokenizer) => <option value={tokenizer.id} key={tokenizer.id}>{tokenizer.name} V{tokenizer.version} · {tokenizer.status}</option>)}</select></Field>
      </div></section>
      <section className="model-registration-section"><header><span>2</span><div><h3>模型来源与运行端点</h3><p>来源决定 Worker 如何加载或调用模型；厂商托管端点冻结调用配置，但不宣称冻结远端权重。</p></div></header>
        <div className="model-source-choices">
          <label className={sourceType === "huggingface" ? "selected" : ""}><input type="radio" name="source_type_choice" checked={sourceType === "huggingface"} onChange={() => setSourceType("huggingface")} /><strong>Hugging Face</strong><span>固定仓库与 Commit</span></label>
          <label className={sourceType === "checkpoint" ? "selected" : ""}><input type="radio" name="source_type_choice" checked={sourceType === "checkpoint"} onChange={() => setSourceType("checkpoint")} /><strong>项目 Checkpoint</strong><span>复用训练权重制品</span></label>
          <label className={sourceType === "external_artifact" ? "selected" : ""}><input type="radio" name="source_type_choice" checked={sourceType === "external_artifact"} onChange={() => setSourceType("external_artifact")} /><strong>外部模型制品</strong><span>登记 URI 与 SHA-256</span></label>
          <label className={sourceType === "openai_compatible" ? "selected" : ""}><input type="radio" name="source_type_choice" checked={sourceType === "openai_compatible"} onChange={() => setSourceType("openai_compatible")} /><strong>OpenAI 兼容端点</strong><span>Ollama、vLLM 或厂商云 API</span></label>
        </div>
        {sourceType === "huggingface" && <div className="form-grid two-columns model-source-fields"><Field label="仓库 ID"><input name="huggingface_id" required placeholder="organization/model-name" /></Field><Field label="Commit SHA" hint="必须填写 40 位提交哈希，不能使用会移动的 main 分支"><input name="huggingface_revision" required pattern="[A-Fa-f0-9]{40}" placeholder="40 位 commit SHA" /></Field></div>}
        {sourceType === "checkpoint" && <div className="model-source-fields">{eligibleCheckpoints.length ? <Field label="可用 Checkpoint"><select name="source_checkpoint_id" required defaultValue={initialCheckpointId || eligibleCheckpoints[0]?.id}>{eligibleCheckpoints.map((checkpoint: any) => <option value={checkpoint.id} key={checkpoint.id}>Step {formatNumber(checkpoint.step)} · {formatNumber(checkpoint.token_count)} Tokens · {checkpoint.id.slice(0, 8)}</option>)}</select></Field> : <Alert severity="warning" title="没有可登记的 Checkpoint" detail="Checkpoint 必须处于 available 且已经绑定权重制品。" />}</div>}
        {sourceType === "external_artifact" && <div className="form-grid two-columns model-source-fields">
          <Field label="外部 URI" hint="文件或目录只登记位置，不复制负载"><input name="artifact_uri" required placeholder="file:///models/example-v1 或共享存储 URI" /></Field>
          <Field label="SHA-256"><input name="content_hash" required pattern="[A-Fa-f0-9]{64}" placeholder="64 位内容哈希" /></Field>
          <Field label="大小（Bytes）"><input name="size_bytes" type="number" min="0" defaultValue="0" /></Field>
          <Field label="媒体类型"><input name="media_type" defaultValue="application/x-directory" /></Field>
          <Field label="可用状态"><select name="availability_state" defaultValue="unknown"><option value="unknown">未知 / 待验证</option><option value="available">已确认可用</option><option value="missing">缺失</option><option value="offline">离线</option></select></Field>
          <Field label="可访问 Worker" hint="共享位置可留空；节点本地路径必须选择 Worker"><select name="worker_id" defaultValue=""><option value="">共享 / 任意 Worker</option>{workers.map((worker) => <option value={worker.id} key={worker.id}>{worker.name} · {worker.status}</option>)}</select></Field>
        </div>}
        {sourceType === "openai_compatible" && <div className="form-grid two-columns model-source-fields">
          <Field label="接口地址" hint="可填写 http://localhost:11434/v1；路径会按协议补全"><input name="endpoint_url" type="url" required placeholder="http://localhost:11434/v1" /></Field>
          <Field label="远端模型名"><input name="endpoint_model" required placeholder="qwen3.5:0.8b" /></Field>
          <Field label="接口协议"><select name="endpoint_mode" value={endpointMode} onChange={(event) => setEndpointMode(event.target.value as "completions" | "chat_completions")}><option value="completions">Completions · 生成 + Logprobs</option><option value="chat_completions">Chat Completions · 仅生成评测</option></select></Field>
          {endpointMode === "completions" && <Field label="Tokenizer" hint="Log-likelihood 需要与服务模型一致的 Hugging Face Tokenizer"><input name="endpoint_tokenizer" required placeholder="Qwen/Qwen3.5-0.8B" /></Field>}
          <Field label="认证环境变量" hint="只保存变量名；密钥值配置在执行 Worker"><input name="endpoint_auth_env_var" pattern="[A-Za-z_][A-Za-z0-9_]*" placeholder="VENDOR_API_KEY；Ollama 可留空" /></Field>
          <Field label="并发请求数"><input name="endpoint_num_concurrent" type="number" min="1" max="128" defaultValue="1" /></Field>
          <Field label="超时（秒）"><input name="endpoint_timeout_seconds" type="number" min="1" max="3600" defaultValue="300" /></Field>
          <label className="check-field"><input name="endpoint_verify_tls" type="checkbox" defaultChecked /><span><b>验证 HTTPS 证书</b><small>仅可信本地 HTTP 端点不使用 TLS</small></span></label>
        </div>}
      </section>
      <section className="model-registration-section"><header><span>3</span><div><h3>仓库状态</h3><p>候选、最佳与正式是可审计状态动作，不改变模型权重内容。</p></div></header><Field label="说明"><textarea name="description" rows={3} placeholder="训练目标、适用范围或来源说明" /></Field><label className="check-field"><input name="is_candidate" type="checkbox" defaultChecked /><span>登记后标记为候选模型</span></label></section>
      <ErrorMessage error={create.error} />
      <FormActions submitting={create.isPending} disabled={sourceType === "checkpoint" && !eligibleCheckpoints.length} onCancel={onClose} submitLabel="创建不可变模型版本" />
    </form>
  </Modal>;
}

function ModelDetailModal({ model, actionError, onClose, onAction, onNavigate, onDelete }: { model: ModelVersion; actionError: unknown; onClose: () => void; onAction: (action: ModelAction) => void; onNavigate: (to: string) => void; onDelete: (mode: ModelDeletionMode) => void }) {
  return <Modal title={`${model.name} V${model.version}`} description={model.metadata_json?.description || "不可变模型版本详情"} onClose={onClose} wide>
    <div className="model-version-detail">
      <ModelDeletionMenu onSelect={onDelete} />
      <div className="model-version-detail-head"><div><StatusBadge status={model.status} />{model.is_best && <span className="best-flag"><Star size={13} />项目最佳</span>}<span>{model.kind}</span><span>ID {model.id}</span></div><div><button className="button secondary small" onClick={() => onNavigate("../evaluations")}>发起评测</button><button className="button secondary small" onClick={() => onNavigate("../releases")}>量化与发布</button><button className="button primary small" disabled={!canUsePlayground(model)} title={!canUsePlayground(model) ? model.inference_source?.type === "api" ? "外部评测端点首版不接入 Playground" : inferenceLabel(model) : undefined} onClick={() => onNavigate("../playground")}><Play size={14} />Playground</button></div></div>
      <div className="model-parameter-cards"><div><span>参数量</span><strong>{formatNumber(model.parameter_count)}</strong></div><div><span>来源</span><strong>{modelSourceLabel(model)}</strong></div><div><span>模型制品</span><strong>{model.artifact ? formatBytes(model.artifact.size_bytes) : "未绑定"}</strong></div><div><span>子版本</span><strong>{formatNumber(model.child_model_count)}</strong></div></div>
      <div className="model-detail-grid">
        <section><h3>身份与权重</h3><dl><div><dt>名称 / 版本</dt><dd>{model.name} V{model.version}</dd></div><div><dt>类型</dt><dd>{model.kind}</dd></div><div><dt>父模型</dt><dd>{model.parent_model ? `${model.parent_model.name} V${model.parent_model.version}` : "无"}</dd></div><div><dt>Checkpoint</dt><dd>{model.source_checkpoint?.id || "无"}</dd></div><div><dt>Artifact</dt><dd>{model.artifact?.id || "无"}</dd></div><div><dt>内容哈希</dt><dd>{model.artifact?.content_hash || "无"}</dd></div></dl></section>
        <section><h3>运行契约</h3><dl><div><dt>推理来源</dt><dd>{inferenceLabel(model)}</dd></div><div><dt>Tokenizer</dt><dd>{model.tokenizer ? `${model.tokenizer.name} V${model.tokenizer.version} · ${formatNumber(model.tokenizer.vocab_size)}` : "未登记"}</dd></div><div><dt>训练运行</dt><dd>{model.source_training_run?.id || "外部登记"}</dd></div><div><dt>训练阶段</dt><dd>{model.source_training_run?.current_stage || "—"}</dd></div><div><dt>创建时间</dt><dd>{formatDate(model.created_at)}</dd></div><div><dt>更新时间</dt><dd>{formatDate(model.updated_at)}</dd></div></dl></section>
      </div>
      {model.source_training_run && <div className="model-detail-training"><div><strong>来源训练运行</strong><span>Step {formatNumber(model.source_checkpoint?.step)} · {formatNumber(model.source_checkpoint?.token_count)} Tokens · Loss {model.source_checkpoint?.metrics?.loss ?? "—"}</span></div><button className="button secondary small" onClick={() => onNavigate(`../training?run=${model.source_training_run?.id}`)}><ExternalLink size={14} />打开训练详情</button></div>}
      <details className="collapsed-config"><summary>完整模型元数据</summary><pre className="json-preview">{JSON.stringify(model.metadata_json || {}, null, 2)}</pre></details>
      <ErrorMessage error={actionError} />
      <div className="model-detail-actions">
        {!model.is_candidate && !model.is_official && model.status !== "archived" && model.status !== "deleted" && <button className="button secondary" onClick={() => onAction("mark-candidate")}>标记候选</button>}
        {model.is_best
          ? <button className="button secondary" onClick={() => onAction("unmark-best")}><Star size={14} />取消最佳</button>
          : model.status !== "archived" && model.status !== "deleted" && <button className="button secondary" onClick={() => onAction("mark-best")}><Star size={14} />设为最佳</button>}
        {!model.is_official && model.status !== "archived" && model.status !== "deleted" && <button className="button primary" onClick={() => onAction("promote")}><Rocket size={14} />晋升正式</button>}
        {model.status === "archived" ? <button className="button secondary" onClick={() => onAction("restore")}><RefreshCcw size={14} />恢复到仓库</button> : !model.is_official && model.status !== "deleted" && <button className="button ghost" onClick={() => onAction("archive")}><Archive size={14} />归档</button>}
      </div>
    </div>
  </Modal>;
}

function CheckpointDetailModal({ checkpoint, onClose, onRegister, onNavigate }: { checkpoint: any; onClose: () => void; onRegister: () => void; onNavigate: (to: string) => void }) {
  return <Modal title={`Checkpoint ${checkpoint.id.slice(0, 8)}`} description="训练恢复状态与权重制品；它不是下游稳定模型版本。" onClose={onClose} wide><div className="model-version-detail">
    <div className="model-version-detail-head"><div><StatusBadge status={checkpoint.status} /><span>ID {checkpoint.id}</span></div><div><button className="button secondary small" onClick={() => onNavigate(`../training?run=${checkpoint.training_run_id}`)}>打开训练详情</button><button className="button primary small" disabled={checkpoint.status !== "available" || !checkpoint.artifact_id} onClick={onRegister}>登记为模型版本</button></div></div>
    <div className="model-parameter-cards"><div><span>Step</span><strong>{formatNumber(checkpoint.step)}</strong></div><div><span>累计 Tokens</span><strong>{formatNumber(checkpoint.token_count)}</strong></div><div><span>Loss</span><strong>{checkpoint.metrics?.loss ?? "—"}</strong></div><div><span>制品大小</span><strong>{checkpoint.artifact ? formatBytes(checkpoint.artifact.size_bytes) : "未登记"}</strong></div></div>
    <div className="model-detail-grid"><section><h3>训练位置</h3><pre className="json-preview">{JSON.stringify(checkpoint.training_position || {}, null, 2)}</pre></section><section><h3>Checkpoint 指标</h3><pre className="json-preview">{JSON.stringify(checkpoint.metrics || {}, null, 2)}</pre></section></div>
    {checkpoint.artifact && <section className="model-checkpoint-artifact"><h3>权重制品</h3><dl><div><dt>Artifact ID</dt><dd>{checkpoint.artifact.id}</dd></div><div><dt>类型</dt><dd>{checkpoint.artifact.kind}</dd></div><div><dt>SHA-256</dt><dd>{checkpoint.artifact.content_hash}</dd></div><div><dt>可用状态</dt><dd>{checkpoint.artifact.availability_state}</dd></div></dl></section>}
    <Alert severity="info" title="登记不会转换文件格式" detail="若该制品只是训练 Checkpoint，内置 Transformers 推理仍会显示不支持；需要训练引擎输出完整模型目录或登记项目原生推理配置。" />
  </div></Modal>;
}

function ModelComparison({ models, onClose }: { models: ModelVersion[]; onClose: () => void }) {
  return <Modal title="模型版本对比" description="并排检查仓库身份、来源与运行可用性；评测差异请在项目评测中比较。" onClose={onClose} wide><div className="data-table-wrap"><table className="data-table"><thead><tr><th>模型</th><th>类型</th><th>参数量</th><th>来源</th><th>Tokenizer</th><th>推理</th><th>仓库状态</th></tr></thead><tbody>{models.map((model) => <tr key={model.id}><td><strong>{model.name}</strong><span>V{model.version} · {model.id.slice(0, 8)}</span></td><td>{model.kind}</td><td>{formatNumber(model.parameter_count)}</td><td>{modelSourceLabel(model)}</td><td>{model.tokenizer ? `${model.tokenizer.name} V${model.tokenizer.version}` : "—"}</td><td>{inferenceLabel(model)}</td><td><div className="model-registry-flags">{model.is_best && <span className="best-flag"><Star size={13} />最佳</span>}<StatusBadge status={model.status} /></div></td></tr>)}</tbody></table></div><div className="form-actions"><span /><button className="button primary" onClick={onClose}>关闭</button></div></Modal>;
}

function ModelSpecifications({ specifications, architectures, onOpen, onCreate }: { specifications: ModelSpecification[]; architectures: ModelArchitecture[]; onOpen: (id: string) => void; onCreate: () => void }) {
  if (!specifications.length) return <Panel><EmptyState title="还没有模型规格" description="模型规格定义训练前结构，不等于已经存在的模型版本。" action={<button className="button primary" onClick={onCreate}><Plus size={16} />新建模型规格</button>} /></Panel>;
  return <div className="model-spec-grid">{specifications.map((specification) => {
    const revision = specification.current_revision;
    const family = architectures.find((item) => item.id === revision.architecture_family_id);
    return <Panel className="model-spec-card" key={specification.id}>
      <header><div className="model-spec-icon"><Boxes size={20} /></div><div><span>{family?.name || revision.architecture_family_id}</span><h3>{specification.name} <small>V{revision.revision}</small></h3><p>{specification.description}</p></div><StatusBadge status={specification.status} /></header>
      <div className="model-spec-metrics"><div><span>总参数</span><strong>{formatNumber(revision.parameter_summary.total_parameters)}</strong></div><div><span>每 Token 激活</span><strong>{formatNumber(revision.parameter_summary.active_parameters_per_token)}</strong></div><div><span>层数</span><strong>{formatNumber(revision.resolved_config.num_hidden_layers)}</strong></div><div><span>词表</span><strong>{formatNumber(revision.resolved_config.vocab_size)}</strong></div></div>
      <footer><span>{revision.source_type === "custom_code" ? "自定义代码 · 用户声明参数" : `标准架构 · ${revision.parameter_summary.method}`}</span><button className="button secondary small" onClick={() => onOpen(specification.id)}>查看配置</button></footer>
    </Panel>;
  })}</div>;
}

function ModelSpecificationDetail({ specification, architectures, codeVersions, onClose, onSaved }: { specification: ModelSpecification; architectures: ModelArchitecture[]; codeVersions: Array<CodeVersion & { source_name: string }>; onClose: () => void; onSaved: (specification: ModelSpecification) => void }) {
  const [revisionId, setRevisionId] = useState(specification.current_revision_id);
  const [editing, setEditing] = useState(false);
  const [previewMode, setPreviewMode] = useState<"json" | "code">("json");
  const revision = specification.revisions.find((item) => item.id === revisionId) || specification.current_revision;
  const family = architectures.find((item) => item.id === revision.architecture_family_id);
  return <Modal title={specification.name} description={specification.description || "模型规格详情"} onClose={onClose} wide>
    {editing ? <ModelSpecificationEditor projectId={specification.project_id} architectures={architectures} codeVersions={codeVersions} specification={specification} baseRevision={revision} onCancel={() => setEditing(false)} onSaved={(updated) => { onSaved(updated); setRevisionId(updated.current_revision_id); setEditing(false); }} /> : <div className="model-spec-detail">
      <div className="model-spec-detail-head"><div><StatusBadge status={specification.status} /><span>{family?.name || revision.architecture_family_id} · 架构版本 {revision.architecture_version}</span><span>签名 {revision.config_signature.slice(0, 12)}</span></div><button className="button primary small" onClick={() => setEditing(true)}><Edit3 size={14} />基于此版本编辑</button></div>
      <div className="revision-selector">{specification.revisions.map((item) => <button type="button" className={item.id === revision.id ? "active" : ""} key={item.id} onClick={() => setRevisionId(item.id)}><strong>V{item.revision}</strong><span>{formatDate(item.created_at)}{item.locked ? " · 已锁定" : ""}</span></button>)}</div>
      <div className="model-parameter-cards"><div><span>总参数量</span><strong>{formatNumber(revision.parameter_summary.total_parameters)}</strong></div><div><span>每 Token 激活</span><strong>{formatNumber(revision.parameter_summary.active_parameters_per_token)}</strong></div><div><span>可训练参数</span><strong>{formatNumber(revision.parameter_summary.trainable_parameters)}</strong></div><div><span>统计可信度</span><strong>{revision.parameter_summary.verified ? "精确公式" : "用户声明"}</strong></div></div>
      <div className="preview-tabs"><button type="button" className={previewMode === "json" ? "active" : ""} onClick={() => setPreviewMode("json")}><Braces size={14} />冻结 JSON</button><button type="button" className={previewMode === "code" ? "active" : ""} onClick={() => setPreviewMode("code")}><Code2 size={14} />Python 代码</button></div>
      <pre className="json-preview model-code-preview">{previewMode === "json" ? JSON.stringify(revision.generated_json, null, 2) : revision.generated_code}</pre>
    </div>}
  </Modal>;
}
