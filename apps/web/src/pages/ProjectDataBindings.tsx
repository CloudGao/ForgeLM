import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Blocks, Database, Layers3, Pencil, Plus, Search, Trash2 } from "lucide-react";
import { type FormEvent, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatNumber, patch, post } from "../api";
import { EmptyState, ErrorMessage, Field, FormActions, Modal, StatusBadge, SubmitForm } from "../components/ui";

type AssetType = "dataset_version" | "mixture_version" | "supervision_dataset_version" | "supervision_mixture_version" | "eval_dataset_version";
type Purpose = "training" | "validation" | "evaluation" | "reference";
type CatalogAsset = {
  asset_type: AssetType;
  asset_version_id: string;
  name: string;
  version: number;
  status: string;
  origin: string;
  tags: string[];
  detail: string;
};
type Binding = {
  id: string;
  asset_type: AssetType;
  asset_version_id: string;
  purpose: Purpose;
  alias: string;
  created_at: string;
  asset?: {
    name: string;
    version: number;
    status: string;
    origin_project_name: string;
    global_managed?: boolean;
    tags: string[];
    asset?: Record<string, any>;
  };
};

const typeLabels: Record<AssetType, string> = {
  dataset_version: "数据集",
  mixture_version: "预训练混合集",
  supervision_dataset_version: "后训练数据集",
  supervision_mixture_version: "后训练混合集",
  eval_dataset_version: "评测数据集",
};

const purposeLabels: Record<Purpose, string> = {
  training: "训练",
  validation: "验证",
  evaluation: "评测",
  reference: "参考",
};

async function loadCatalogAssets(): Promise<CatalogAsset[]> {
  const [datasets, mixtures, supervisionDatasets, supervisionMixtures, evalDatasets] = await Promise.all([
    api<any[]>("/catalog/datasets"),
    api<any[]>("/catalog/mixtures"),
    api<any[]>("/catalog/supervision-datasets"),
    api<any[]>("/catalog/supervision-mixtures"),
    api<any[]>("/catalog/evaluation-datasets"),
  ]);
  return [
    ...datasets.flatMap((item) => (item.versions || []).map((version: any) => ({
      asset_type: "dataset_version" as const,
      asset_version_id: version.id,
      name: item.name,
      version: version.version,
      status: version.state,
      origin: item.origin_project_name,
      tags: item.tags || [],
      detail: `${formatNumber(version.document_count)} 行 · ${formatNumber(version.token_estimate)} tokens · ${String(version.tier || "").toUpperCase()}`,
    }))),
    ...mixtures.flatMap((item) => (item.versions || []).map((version: any) => ({
      asset_type: "mixture_version" as const,
      asset_version_id: version.id,
      name: item.name,
      version: version.version,
      status: version.status,
      origin: item.origin_project_name,
      tags: [],
      detail: `${formatNumber(version.resolved_document_count)} 行 · ${formatNumber(version.resolved_token_estimate)} tokens`,
    }))),
    ...supervisionDatasets.flatMap((item) => (item.versions || []).map((version: any) => ({
      asset_type: "supervision_dataset_version" as const,
      asset_version_id: version.id,
      name: item.name,
      version: version.version,
      status: version.status,
      origin: item.origin_project_name,
      tags: [version.kind],
      detail: `${formatNumber(version.accepted_record_count)} 条可用记录 · ${version.kind}`,
    }))),
    ...supervisionMixtures.flatMap((item) => (item.versions || []).map((version: any) => ({
      asset_type: "supervision_mixture_version" as const,
      asset_version_id: version.id,
      name: item.name,
      version: version.version,
      status: version.status,
      origin: item.origin_project_name,
      tags: [version.kind],
      detail: `${version.entries?.length || 0} 个来源 · ${version.kind}`,
    }))),
    ...evalDatasets.map((item) => ({
      asset_type: "eval_dataset_version" as const,
      asset_version_id: item.id,
      name: item.name,
      version: item.version,
      status: "ready",
      origin: item.origin_project_name,
      tags: item.dimensions || [],
      detail: `${formatNumber(item.sample_count)} 个样本 · ${item.task_type}`,
    })),
  ];
}

function assetIcon(type: AssetType) {
  if (type === "eval_dataset_version") return <Blocks size={18} />;
  if (type.includes("mixture")) return <Layers3 size={18} />;
  return <Database size={18} />;
}

function bindingName(binding: Binding) {
  return binding.alias || binding.asset?.name || binding.asset_version_id.slice(0, 8);
}

function sourceLabel(binding: Binding) {
  if (binding.asset?.global_managed) return "全局数据中心";
  return binding.asset?.origin_project_name || "全局数据中心";
}

export function ProjectDataBindings() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [editBinding, setEditBinding] = useState<Binding | null>(null);
  const [removeBinding, setRemoveBinding] = useState<Binding | null>(null);
  const [search, setSearch] = useState("");
  const [purposeFilter, setPurposeFilter] = useState<Purpose | "all">("all");
  const [typeFilter, setTypeFilter] = useState<AssetType | "all">("all");
  const [catalogSearch, setCatalogSearch] = useState("");
  const [catalogType, setCatalogType] = useState<AssetType | "all">("all");
  const [pickerPurpose, setPickerPurpose] = useState<Purpose>("training");

  const bindings = useQuery({
    queryKey: ["data-bindings", projectId],
    queryFn: () => api<Binding[]>(`/projects/${projectId}/data-bindings`),
  });
  const catalog = useQuery({
    queryKey: ["global-catalog-assets"],
    queryFn: loadCatalogAssets,
    enabled: pickerOpen,
  });
  const bind = useMutation({
    mutationFn: (asset: CatalogAsset) => post<Binding>(`/projects/${projectId}/data-bindings`, {
      asset_type: asset.asset_type,
      asset_version_id: asset.asset_version_id,
      purpose: pickerPurpose,
      alias: "",
      metadata: {},
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data-bindings", projectId] });
      setPickerOpen(false);
    },
  });
  const update = useMutation({
    mutationFn: ({ bindingId, purpose, alias }: { bindingId: string; purpose: Purpose; alias: string }) => patch<Binding>(`/projects/${projectId}/data-bindings/${bindingId}`, { purpose, alias }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data-bindings", projectId] });
      setEditBinding(null);
    },
  });
  const remove = useMutation({
    mutationFn: (bindingId: string) => api(`/projects/${projectId}/data-bindings/${bindingId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data-bindings", projectId] });
      setRemoveBinding(null);
    },
  });

  const boundRefs = new Set((bindings.data || []).map((item) => `${item.asset_type}:${item.asset_version_id}:${item.purpose}`));
  const visibleBindings = useMemo(() => (bindings.data || []).filter((binding) => {
    const haystack = `${bindingName(binding)} ${binding.asset?.name || ""} ${sourceLabel(binding)} ${(binding.asset?.tags || []).join(" ")}`.toLowerCase();
    return (purposeFilter === "all" || binding.purpose === purposeFilter)
      && (typeFilter === "all" || binding.asset_type === typeFilter)
      && haystack.includes(search.trim().toLowerCase());
  }), [bindings.data, purposeFilter, search, typeFilter]);
  const visibleAssets = useMemo(() => (catalog.data || []).filter((item) => {
    const haystack = `${item.name} ${item.origin} ${item.tags.join(" ")} ${item.detail}`.toLowerCase();
    return (catalogType === "all" || item.asset_type === catalogType)
      && haystack.includes(catalogSearch.trim().toLowerCase());
  }), [catalog.data, catalogSearch, catalogType]);

  function openPicker() {
    bind.reset();
    setCatalogSearch("");
    setCatalogType("all");
    setPickerOpen(true);
  }

  function openEditor(binding: Binding) {
    update.reset();
    setEditBinding(binding);
  }

  function submitEdit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!editBinding) return;
    const form = new FormData(event.currentTarget);
    update.mutate({
      bindingId: editBinding.id,
      purpose: String(form.get("purpose")) as Purpose,
      alias: String(form.get("alias") || "").trim(),
    });
  }

  function openRemove(binding: Binding) {
    remove.reset();
    setRemoveBinding(binding);
  }

  return <div className="project-data-workspace">
    <header className="project-data-command">
      <div>
        <p className="eyebrow">PROJECT DATASETS</p>
        <h1>项目数据集</h1>
        <p>集中管理本项目使用的数据集。数据集的创建、导入、加工和版本管理在全局数据中心完成。</p>
      </div>
      <div>
        <button className="button secondary" onClick={() => navigate("/data")}><Database size={16} />打开全局数据中心</button>
        <button className="button primary" onClick={openPicker}><Plus size={16} />添加已有数据集</button>
      </div>
    </header>

    <section className="project-dataset-section">
      <header>
        <div><span>PROJECT DATASETS</span><h2>本项目使用的数据集</h2></div>
        <small>共 {(bindings.data || []).length} 项</small>
      </header>
      <div className="project-dataset-toolbar">
        <div className="search-box"><Search size={16} /><input aria-label="搜索项目数据集" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称、标签或来源" /></div>
        <label><span>用途</span><select aria-label="按用途筛选" value={purposeFilter} onChange={(event) => setPurposeFilter(event.target.value as Purpose | "all")}><option value="all">全部用途</option>{(Object.entries(purposeLabels) as [Purpose, string][]).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label><span>类型</span><select aria-label="按类型筛选" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value as AssetType | "all")}><option value="all">全部类型</option>{(Object.entries(typeLabels) as [AssetType, string][]).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      </div>

      {visibleBindings.length ? <div className="project-dataset-table" role="table" aria-label="项目数据集清单">
        <div className="project-dataset-head" role="row"><span role="columnheader">数据集</span><span role="columnheader">版本</span><span role="columnheader">用途</span><span role="columnheader">状态</span><span role="columnheader">来源</span><span role="columnheader">操作</span></div>
        {visibleBindings.map((binding) => <article key={binding.id} role="row">
          <div className="project-dataset-identity" role="cell"><div className="binding-card-icon">{assetIcon(binding.asset_type)}</div><div><strong>{bindingName(binding)}</strong>{binding.alias && binding.asset?.name && <small>全局名称：{binding.asset.name}</small>}<span>{typeLabels[binding.asset_type]}</span><div>{(binding.asset?.tags || []).slice(0, 4).map((tag) => <i key={tag}>{tag}</i>)}</div></div></div>
          <div role="cell"><span className="mobile-cell-label">版本</span><b>V{binding.asset?.version || "—"}</b><small>{binding.asset_version_id.slice(0, 12)}</small></div>
          <div role="cell"><span className="mobile-cell-label">用途</span><i className={`purpose-pill ${binding.purpose}`}>{purposeLabels[binding.purpose]}</i></div>
          <div role="cell"><span className="mobile-cell-label">状态</span><StatusBadge status={binding.asset?.status || "ready"} /></div>
          <div role="cell"><span className="mobile-cell-label">来源</span><span>{sourceLabel(binding)}</span></div>
          <div className="project-dataset-actions" role="cell"><button className="button ghost small" onClick={() => openEditor(binding)}><Pencil size={14} />编辑</button><button className="button danger small" onClick={() => openRemove(binding)}><Trash2 size={14} />移除</button></div>
        </article>)}
      </div> : <EmptyState title={(bindings.data || []).length ? "没有匹配的项目数据集" : "项目还没有数据集"} description={(bindings.data || []).length ? "调整搜索词或筛选条件。" : "从全局数据中心选择一个已有版本加入项目。"} action={!(bindings.data || []).length ? <button className="button primary" onClick={openPicker}><Plus size={16} />选择已有数据集</button> : undefined} />}
      <ErrorMessage error={bindings.error || remove.error} />
      <p className="project-dataset-boundary">从项目移除只会解除项目使用关系，不会删除全局数据；已有训练、制品或任务引用时系统会阻止移除。</p>
    </section>

    {pickerOpen && <Modal title="添加已有数据集" description="从全局数据中心选择一个已经存在的不可变版本。" onClose={() => setPickerOpen(false)} wide>
      <div className="binding-picker-steps">
        <section>
          <header><b>1</b><div><strong>选择项目用途</strong><span>用途只描述当前项目如何使用这个版本</span></div></header>
          <div className="picker-purpose-grid">{(Object.entries(purposeLabels) as [Purpose, string][]).map(([value, label]) => <button key={value} className={pickerPurpose === value ? "active" : ""} onClick={() => setPickerPurpose(value)}><strong>{label}</strong></button>)}</div>
        </section>
        <section>
          <header><b>2</b><div><strong>选择已有版本</strong><span>按名称、标签、来源或类型缩小范围</span></div></header>
          <div className="binding-picker-toolbar"><div className="search-box"><Search size={16} /><input aria-label="搜索全局数据集" value={catalogSearch} onChange={(event) => setCatalogSearch(event.target.value)} placeholder="搜索名称、标签或来源" /></div><div className="asset-type-chips"><button className={catalogType === "all" ? "active" : ""} onClick={() => setCatalogType("all")}>全部</button>{(Object.entries(typeLabels) as [AssetType, string][]).map(([value, label]) => <button className={catalogType === value ? "active" : ""} key={value} onClick={() => setCatalogType(value)}>{label}</button>)}</div></div>
          <div className="binding-asset-list">{visibleAssets.map((asset) => { const alreadyBound = boundRefs.has(`${asset.asset_type}:${asset.asset_version_id}:${pickerPurpose}`); return <article key={`${asset.asset_type}:${asset.asset_version_id}`}><div className="binding-asset-icon">{assetIcon(asset.asset_type)}</div><div><span>{typeLabels[asset.asset_type]} · V{asset.version}</span><strong>{asset.name}</strong><small>{asset.detail} · {asset.origin || "全局数据中心"}</small><div className="tag-list">{asset.tags.slice(0, 5).map((tag) => <i key={tag}>{tag}</i>)}</div></div><StatusBadge status={asset.status} /><button className="button primary small" aria-label={`${alreadyBound ? "已加入" : "加入项目"} ${asset.name} V${asset.version}`} disabled={alreadyBound || bind.isPending} onClick={() => bind.mutate(asset)}>{alreadyBound ? "已加入" : "加入项目"}</button></article>; })}</div>
          {!catalog.isLoading && !visibleAssets.length && <EmptyState title="没有匹配的已有版本" description="调整筛选条件，或者前往全局数据中心创建和管理数据集。" />}
          <ErrorMessage error={catalog.error || bind.error} />
        </section>
      </div>
      <div className="form-actions"><button className="button secondary" onClick={() => setPickerOpen(false)}>取消</button><button className="button ghost" onClick={() => { setPickerOpen(false); navigate("/data"); }}>打开全局数据中心<ArrowRight size={14} /></button></div>
    </Modal>}

    {editBinding && <Modal title="编辑项目数据集" description="这里只修改当前项目的用途和显示名称，不会改动全局数据。" onClose={() => setEditBinding(null)}>
      <SubmitForm onSubmit={submitEdit}>
        <Field label="项目别名" hint={`留空时显示全局名称：${editBinding.asset?.name || bindingName(editBinding)}`}><input name="alias" defaultValue={editBinding.alias} maxLength={180} placeholder="可选" /></Field>
        <Field label="项目用途"><select name="purpose" defaultValue={editBinding.purpose}>{(Object.entries(purposeLabels) as [Purpose, string][]).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></Field>
        <ErrorMessage error={update.error} />
        <FormActions submitting={update.isPending} onCancel={() => setEditBinding(null)} />
      </SubmitForm>
    </Modal>}

    {removeBinding && <Modal title="从项目移除数据集" description="全局数据和历史版本不会被删除。" onClose={() => setRemoveBinding(null)}>
      <div className="project-dataset-remove-copy"><strong>{bindingName(removeBinding)} · V{removeBinding.asset?.version || "—"}</strong><p>移除后，本项目不能再用这个关系创建新的训练或评测配置。已有下游引用时，系统会拒绝本次操作并说明阻塞项。</p></div>
      <ErrorMessage error={remove.error} />
      <div className="form-actions"><button className="button secondary" onClick={() => setRemoveBinding(null)}>取消</button><button className="button danger" disabled={remove.isPending} onClick={() => remove.mutate(removeBinding.id)}>{remove.isPending ? "正在移除…" : "确认从项目移除"}</button></div>
    </Modal>}
  </div>;
}
