import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Layers3, MessageSquareText, Pencil, Plus, Search, SlidersHorizontal, Trash2 } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { api, formatBytes, formatDate, formatNumber, patch, post } from "../api";
import type {
  Dataset,
  SupervisionDataset,
  SupervisionDatasetVersion,
  SupervisionMixture,
} from "../types";
import { Alert, EmptyState, ErrorMessage, Field, FormActions, Modal, Panel, StatusBadge } from "./ui";

export function PostTrainingDataPanel({ projectId, datasets, globalMode = false }: { projectId: string; datasets: Dataset[]; globalMode?: boolean }) {
  const queryClient = useQueryClient();
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [mixtureOpen, setMixtureOpen] = useState(false);
  const [mixtureToVersion, setMixtureToVersion] = useState<SupervisionMixture | null>(null);
  const [selectedMixtureId, setSelectedMixtureId] = useState<string | null>(null);
  const supervisionDatasets = useQuery({
    queryKey: ["supervision-datasets", projectId],
    queryFn: () => api<SupervisionDataset[]>(globalMode ? "/catalog/supervision-datasets" : `/projects/${projectId}/supervision-datasets`),
    refetchInterval: (query) => (query.state.data || []).some((item) => item.versions.some((version) => ["queued", "assigned", "running"].includes(version.status))) ? 1500 : false,
  });
  const mixtures = useQuery({ queryKey: ["supervision-mixtures", projectId, globalMode], queryFn: () => api<SupervisionMixture[]>(globalMode ? "/catalog/supervision-mixtures" : `/projects/${projectId}/supervision-mixtures`) });
  const corpora = useQuery({ queryKey: ["supervision-corpora", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/supervision-corpora`), enabled: !globalMode, refetchInterval: 2000 });
  const views = useQuery({ queryKey: ["supervision-views", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/supervision-training-views`), enabled: !globalMode, refetchInterval: 2000 });
  const versions = (supervisionDatasets.data || []).flatMap((item) => item.versions);
  const selectedMixture = (mixtures.data || []).find((item) => item.id === selectedMixtureId);
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["supervision-datasets", projectId] });
    queryClient.invalidateQueries({ queryKey: ["supervision-mixtures", projectId] });
  };
  return <section className="tokenized-workspace">
    <section className="metrics-grid four"><Metric label="监督数据版本" value={versions.length} /><Metric label="可用记录" value={versions.reduce((total, item) => total + Number(item.accepted_record_count || 0), 0)} /><Metric label="数据混合" value={mixtures.data?.length || 0} /><Metric label={globalMode ? "管理范围" : "Corpus / View"} value={globalMode ? "系统全局" : (corpora.data?.length || 0) + (views.data?.length || 0)} /></section>
    <Alert severity="info" title="后训练数据是平台一等制品" detail="Conversation 与 preference 映射、拒绝原因、Chat Template、Tokenizer、loss span、偏好对和读取顺序都以不可变版本记录；质量提示不会阻止继续。" />
    <div className="form-actions"><button type="button" className="button secondary" onClick={() => setDatasetOpen(true)}><MessageSquareText size={15} />创建监督映射</button><button type="button" className="button primary" disabled={!versions.some((item) => item.status === "ready")} onClick={() => { setMixtureToVersion(null); setMixtureOpen(true); }}><Layers3 size={15} />创建后训练混合</button></div>
    <Panel className="table-panel" title="Conversation / Preference 数据" description="映射现有不可变 DatasetVersion，不复制原始数据。">
      {versions.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>名称</th><th>类型</th><th>来源</th><th>接受 / 拒绝</th><th>状态</th><th>报告</th></tr></thead><tbody>{(supervisionDatasets.data || []).flatMap((logical) => logical.versions.map((version) => <tr key={version.id}><td><strong>{logical.name}</strong><span>V{version.version} · {version.identity_hash.slice(0, 10)}</span></td><td>{version.kind}</td><td>{version.dataset_name || version.dataset_version_id}<span>V{version.dataset_version || "—"}</span></td><td><strong>{formatNumber(version.accepted_record_count)} / {formatNumber(version.rejected_record_count)}</strong></td><td><StatusBadge status={version.status} /></td><td><details className="collapsed-config"><summary>查看预览与拒绝原因</summary><pre className="json-preview">{JSON.stringify({ report: version.validation_report, preview: version.sample_preview, mapping: version.mapping_config }, null, 2)}</pre></details></td></tr>))}</tbody></table></div> : <EmptyState title="还没有监督数据映射" description="从已有 DatasetVersion 创建 conversation 或 preference 映射。" />}
    </Panel>
    <Panel className="table-panel" title="后训练数据混合" description="同一混合只包含同一种语义类型，并冻结权重与 seed。">
      {(mixtures.data || []).length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>名称</th><th>版本 / 类型</th><th>来源组合</th><th>身份</th><th>操作</th></tr></thead><tbody>{(mixtures.data || []).map((item) => { const version = item.versions.find((candidate) => candidate.id === item.current_version_id) || item.versions[0]; const summary = version?.entries.slice(0, 3).map((entry) => `${entry.supervision_name || entry.dataset_name || entry.supervision_dataset_version_id?.slice(0, 8) || "未知来源"} ${((entry.normalized_weight || 0) * 100).toFixed(1)}%`).join(" · "); return <tr key={item.id} tabIndex={0} aria-label={`查看后训练混合 ${item.name}`} onClick={() => setSelectedMixtureId(item.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedMixtureId(item.id); } }}><td><strong>{item.name}</strong><span>{item.description || "—"}</span></td><td><strong>V{version?.version || "—"}</strong><span>{version?.kind || "—"}</span></td><td><strong>{version?.entries.length || 0} 个来源</strong><span>{summary || "—"}</span></td><td>{version?.identity_hash.slice(0, 12) || "—"}</td><td><button type="button" className="button secondary small" onClick={(event) => { event.stopPropagation(); setSelectedMixtureId(item.id); }}>查看详情</button></td></tr>; })}</tbody></table></div> : <EmptyState title="还没有后训练数据混合" description="扫描完成后可按权重组合多个 conversation 或 preference 版本。" />}
    </Panel>
    {!globalMode && <Panel className="table-panel" title="已准备的监督 Corpus 与训练 View" description="训练计划的“准备数据”会自动复用或构建。">
      {(corpora.data || []).length || (views.data || []).length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>制品</th><th>目标</th><th>规模</th><th>状态</th><th>空间</th></tr></thead><tbody>{(corpora.data || []).map((item) => <tr key={item.id}><td><strong>Corpus</strong><span>{item.format}</span></td><td>{item.objective_kind}</td><td>{item.objective_kind === "sft" ? `${formatNumber(item.example_count)} examples` : `${formatNumber(item.pair_count)} pairs`}<span>{formatNumber(item.supervised_token_count)} supervised tokens</span></td><td><StatusBadge status={item.status} /></td><td>{formatBytes(item.size_bytes)}</td></tr>)}{(views.data || []).map((item) => <tr key={item.id}><td><strong>Training View</strong><span>{item.format}</span></td><td>{item.objective_kind}</td><td>{item.objective_kind === "sft" ? `${formatNumber(item.example_count)} examples` : `${formatNumber(item.pair_count)} pairs`}<span>{formatNumber(item.budget_value)} {item.budget_unit}</span></td><td><StatusBadge status={item.status} /></td><td>{formatBytes(item.size_bytes)}</td></tr>)}</tbody></table></div> : <EmptyState title="还没有监督训练制品" description="在训练计划绑定后训练混合与 Chat Template，再执行准备数据。" />}
    </Panel>}
    {datasetOpen && <SupervisionDatasetModal projectId={projectId} datasets={datasets} globalMode={globalMode} onClose={() => setDatasetOpen(false)} onCreated={() => { refresh(); setDatasetOpen(false); }} />}
    {mixtureOpen && <SupervisionMixtureModal projectId={projectId} datasets={supervisionDatasets.data || []} mixture={mixtureToVersion || undefined} globalMode={globalMode} onClose={() => { setMixtureOpen(false); setMixtureToVersion(null); }} onCreated={() => { refresh(); setMixtureOpen(false); setMixtureToVersion(null); }} />}
    {selectedMixture && <SupervisionMixtureDrawer mixture={selectedMixture} onClose={() => setSelectedMixtureId(null)} onRecompose={(item) => { setMixtureToVersion(item); setMixtureOpen(true); }} onChanged={refresh} onDeleted={() => { refresh(); setSelectedMixtureId(null); }} />}
  </section>;
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return <article className="metric-card"><span>{label}</span><strong>{String(value)}</strong></article>;
}

function SupervisionMixtureDrawer({ mixture, onClose, onRecompose, onChanged, onDeleted }: { mixture: SupervisionMixture; onClose: () => void; onRecompose: (mixture: SupervisionMixture) => void; onChanged: () => void; onDeleted: () => void }) {
  const currentVersion = mixture.versions.find((candidate) => candidate.id === mixture.current_version_id) || mixture.versions[0];
  const [selectedVersionId, setSelectedVersionId] = useState(currentVersion?.id || "");
  const [metadataOpen, setMetadataOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const version = mixture.versions.find((candidate) => candidate.id === selectedVersionId) || currentVersion;
  if (!version) return null;
  const acceptedRecords = version.entries.reduce((sum, entry) => sum + Number(entry.accepted_record_count || 0), 0);
  return <aside className="detail-drawer mixture-drawer supervision-mixture-drawer">
    <header><button type="button" className="icon-button" aria-label="关闭后训练混合详情" onClick={onClose}><ArrowRight size={19} /></button><div className="drawer-title"><span className="eyebrow">SUPERVISION MIXTURE</span><h2>{mixture.name}</h2><p>{mixture.description || "组合 conversation 或 preference 监督数据版本。"}</p></div><div className="drawer-header-actions"><button type="button" className="button secondary small" onClick={() => setMetadataOpen(true)}><Pencil size={13} />编辑信息</button><button type="button" className="button secondary small" onClick={() => onRecompose(mixture)}><Layers3 size={13} />重新组合</button><button type="button" className="button danger small" onClick={() => setDeleteOpen(true)}><Trash2 size={13} />删除</button></div></header>
    <div className="drawer-content">
      <div className="mixture-version-controls"><Field label="查看版本"><select value={version.id} onChange={(event) => setSelectedVersionId(event.target.value)}>{mixture.versions.map((item) => <option key={item.id} value={item.id}>V{item.version} · {item.kind} · {item.status}</option>)}</select></Field><div className="selection-estimate"><span>当前版本身份</span><strong>{version.identity_hash.slice(0, 12)}</strong><small>{formatDate(version.created_at)}</small></div></div>
      <div className="drawer-metrics"><div><span>版本</span><strong>V{version.version}</strong></div><div><span>语义类型</span><strong>{version.kind}</strong></div><div><span>来源数量</span><strong>{formatNumber(version.entries.length)}</strong></div><div><span>接受记录</span><strong>{formatNumber(acceptedRecords)}</strong></div><div><span>固定 Seed</span><strong>{version.seed}</strong></div><div><span>采样策略</span><strong>weighted deficit V1</strong></div></div>
      <Panel title="来源与权重" description="这些不可变监督映射按规范化比例进入确定性采样顺序。"><div className="supervision-mixture-version-table"><header><span>监督映射</span><span>原始数据</span><span>接受记录</span><span>权重 / 比例</span></header>{version.entries.map((entry, index) => <div key={entry.supervision_dataset_version_id || index}><strong>{entry.supervision_name || entry.supervision_dataset_version_id?.slice(0, 8) || "未知来源"}<small>V{entry.supervision_version || "—"}</small></strong><span>{entry.dataset_name || "—"}<small>Dataset V{entry.dataset_version || "—"}</small></span><span>{formatNumber(entry.accepted_record_count)}</span><span>{entry.weight}<small>{((entry.normalized_weight || 0) * 100).toFixed(1)}%</small></span></div>)}</div></Panel>
      <Panel title="版本历史" description="重新组合会创建新版本，不覆盖之前的数据配方。"><div className="version-timeline">{mixture.versions.map((item) => <article key={item.id}><i /><div><span>V{item.version} · {item.kind}</span><strong>{item.entries.length} 个来源 · seed {item.seed}</strong><small>{formatDate(item.created_at)} · {item.identity_hash.slice(0, 12)}</small></div><StatusBadge status={item.status} /></article>)}</div></Panel>
    </div>
    {metadataOpen && <SupervisionMixtureMetadataModal mixture={mixture} onClose={() => setMetadataOpen(false)} onSaved={() => { setMetadataOpen(false); onChanged(); }} />}
    {deleteOpen && <SupervisionMixtureDeleteModal mixture={mixture} onClose={() => setDeleteOpen(false)} onDeleted={onDeleted} />}
  </aside>;
}

function SupervisionMixtureMetadataModal({ mixture, onClose, onSaved }: { mixture: SupervisionMixture; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(mixture.name);
  const [description, setDescription] = useState(mixture.description || "");
  const save = useMutation({ mutationFn: () => patch(`/supervision-mixtures/${mixture.id}`, { name, description }), onSuccess: onSaved });
  return <Modal title="编辑后训练混合信息" description="名称和说明是可编辑元数据；来源、权重和 seed 通过不可变新版本修改。" onClose={onClose}><form className="form-grid" onSubmit={(event: FormEvent) => { event.preventDefault(); save.mutate(); }}><Field label="名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="说明"><textarea value={description} onChange={(event) => setDescription(event.target.value)} /></Field><ErrorMessage error={save.error} /><FormActions submitting={save.isPending} disabled={!name.trim()} onCancel={onClose} submitLabel="保存信息" /></form></Modal>;
}

function SupervisionMixtureDeleteModal({ mixture, onClose, onDeleted }: { mixture: SupervisionMixture; onClose: () => void; onDeleted: () => void }) {
  const remove = useMutation({ mutationFn: () => api<void>(`/supervision-mixtures/${mixture.id}`, { method: "DELETE" }), onSuccess: onDeleted });
  return <Modal title={`删除 ${mixture.name}`} description="删除该混合及其全部配方版本；监督数据映射和原始数据不会被删除。" onClose={onClose}><div className="form-grid"><Alert severity="warning" title="确认删除这个后训练混合？" detail="若 Corpus、训练计划、任务或训练运行正在引用它，平台会返回具体引用并保留数据。" /><ErrorMessage error={remove.error} /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button type="button" className="button danger" disabled={remove.isPending} onClick={() => remove.mutate()}>{remove.isPending ? "删除中…" : "确认删除"}</button></div></div></Modal>;
}

function SupervisionDatasetModal({ projectId, datasets, globalMode = false, onClose, onCreated }: { projectId: string; datasets: Dataset[]; globalMode?: boolean; onClose: () => void; onCreated: () => void }) {
  const readyVersions: any[] = datasets.flatMap((dataset) => (dataset.versions || []).filter((version) => version.state === "ready" && version.artifact_id && version.record_manifest_artifact_id).map((version) => ({ ...(version as any), dataset_name: dataset.name })));
  const [name, setName] = useState("指令微调数据");
  const [kind, setKind] = useState<"conversation" | "preference">("conversation");
  const [datasetVersionId, setDatasetVersionId] = useState(String(readyVersions[0]?.id || ""));
  const [messagesField, setMessagesField] = useState("messages");
  const [promptField, setPromptField] = useState("prompt");
  const [chosenField, setChosenField] = useState("chosen");
  const [rejectedField, setRejectedField] = useState("rejected");
  const [exactDeduplicate, setExactDeduplicate] = useState(true);
  const create = useMutation({ mutationFn: () => post(globalMode ? "/catalog/supervision-datasets" : `/projects/${projectId}/supervision-datasets`, { name, description: "", dataset_version_id: datasetVersionId, kind, mapping: kind === "conversation" ? { messages_path: messagesField, role_field: "role", content_field: "content", role_map: {} } : { prompt_path: promptField, chosen_path: chosenField, rejected_path: rejectedField, role_field: "role", content_field: "content", role_map: {} }, filters: { exact_deduplicate: exactDeduplicate }, start_scan: true, launcher_id: "builtin.local-process", resource_request: { gpus: 0, cpus: 4 } }), onSuccess: onCreated });
  return <Modal title="创建监督数据映射" description="扫描会规范化记录、统计拒绝原因并保存少量预览；提示不需要确认或放行。" onClose={onClose} wide><form className="form-grid two-columns" onSubmit={(event: FormEvent) => { event.preventDefault(); create.mutate(); }}><Field label="名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="语义类型"><select value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}><option value="conversation">Conversation（SFT）</option><option value="preference">Preference（DPO）</option></select></Field><Field label="来源 DatasetVersion"><select required value={datasetVersionId} onChange={(event) => setDatasetVersionId(event.target.value)}><option value="">请选择已就绪版本</option>{readyVersions.map((item) => <option key={item.id} value={item.id}>{item.dataset_name} · V{item.version}</option>)}</select></Field>{kind === "conversation" ? <Field label="Messages 字段路径"><input value={messagesField} onChange={(event) => setMessagesField(event.target.value)} /></Field> : <><Field label="Prompt 字段路径"><input value={promptField} onChange={(event) => setPromptField(event.target.value)} /></Field><Field label="Chosen 字段路径"><input value={chosenField} onChange={(event) => setChosenField(event.target.value)} /></Field><Field label="Rejected 字段路径"><input value={rejectedField} onChange={(event) => setRejectedField(event.target.value)} /></Field></>}<label className="check-field"><input type="checkbox" checked={exactDeduplicate} onChange={(event) => setExactDeduplicate(event.target.checked)} /><span>精确去重并记录重复数</span></label><div className="span-two"><ErrorMessage error={create.error} /></div><div className="span-two"><FormActions submitting={create.isPending} disabled={!datasetVersionId} onCancel={onClose} submitLabel="创建并扫描" /></div></form></Modal>;
}

function SupervisionMixtureModal({ projectId, datasets, mixture, globalMode = false, onClose, onCreated }: { projectId: string; datasets: SupervisionDataset[]; mixture?: SupervisionMixture; globalMode?: boolean; onClose: () => void; onCreated: () => void }) {
  const ready = useMemo(() => datasets.flatMap((logical) => logical.versions.filter((version) => version.status === "ready").map((version) => ({ ...version, logical_name: logical.name, logical_description: logical.description }))), [datasets]);
  const baseVersion = mixture?.versions.find((version) => version.id === mixture.current_version_id) || mixture?.versions[0];
  const initialEntries = baseVersion?.entries || [];
  const [name, setName] = useState(mixture?.name || "后训练数据混合");
  const [description, setDescription] = useState(mixture?.description || "");
  const [kind, setKind] = useState<"conversation" | "preference">(baseVersion?.kind || ready[0]?.kind || "conversation");
  const [seed, setSeed] = useState(baseVersion?.seed ?? 42);
  const [selected, setSelected] = useState<Record<string, boolean>>(() => Object.fromEntries(initialEntries.map((entry) => [entry.supervision_dataset_version_id, true])));
  const [weights, setWeights] = useState<Record<string, number>>(() => Object.fromEntries(initialEntries.map((entry) => [entry.supervision_dataset_version_id, Number(entry.weight || 1)])));
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [minimumRecords, setMinimumRecords] = useState("");
  const [onlySelected, setOnlySelected] = useState(false);
  const compatible = ready.filter((item) => item.kind === kind);
  const visible = compatible.filter((item) => {
    const haystack = `${item.logical_name} ${item.logical_description} ${item.dataset_name || ""} ${item.dataset_version || ""}`.toLowerCase();
    return haystack.includes(search.trim().toLowerCase())
      && (!minimumRecords || Number(item.accepted_record_count || 0) >= Number(minimumRecords))
      && (!onlySelected || selected[item.id]);
  });
  const selectedCandidates = compatible.filter((item) => selected[item.id]);
  const totalWeight = selectedCandidates.reduce((sum, item) => sum + Math.max(0, Number(weights[item.id] || 0)), 0);
  const invalidWeights = selectedCandidates.some((item) => !Number.isInteger(weights[item.id]) || Number(weights[item.id]) < 1);
  const selectedRecords = selectedCandidates.reduce((sum, item) => sum + Number(item.accepted_record_count || 0), 0);
  const activeFilterCount = Number(Boolean(search.trim())) + Number(Boolean(minimumRecords)) + Number(onlySelected);
  const create = useMutation({
    mutationFn: () => {
      const payload = {
        entries: selectedCandidates.map((item) => ({ supervision_dataset_version_id: item.id, weight: Number(weights[item.id]) })),
        seed,
      };
      return post(mixture ? (globalMode ? `/catalog/supervision-mixtures/${mixture.id}/versions` : `/supervision-mixtures/${mixture.id}/versions`) : (globalMode ? "/catalog/supervision-mixtures" : `/projects/${projectId}/supervision-mixtures`), mixture ? payload : { ...payload, name, description });
    },
    onSuccess: onCreated,
  });
  function toggle(versionId: string, checked: boolean) {
    setSelected((current) => ({ ...current, [versionId]: checked }));
    if (checked) setWeights((current) => ({ ...current, [versionId]: current[versionId] || 1 }));
  }
  function selectVisible() {
    setSelected((current) => ({ ...current, ...Object.fromEntries(visible.map((item) => [item.id, true])) }));
    setWeights((current) => ({ ...current, ...Object.fromEntries(visible.map((item) => [item.id, current[item.id] || 1])) }));
  }
  function clearSelection() {
    setSelected({});
    setOnlySelected(false);
  }
  function changeKind(next: "conversation" | "preference") {
    setKind(next);
    setSelected({});
    setWeights({});
    setOnlySelected(false);
  }
  return <Modal title={mixture ? `重新组合 ${mixture.name}` : "创建后训练混合"} description={mixture ? `基于 V${baseVersion?.version || "—"} 创建不可变新版本，旧版本保持不变。` : "筛选并组合多个同类型监督数据版本，逐来源设置采样权重。"} onClose={onClose} wide><form onSubmit={(event: FormEvent) => { event.preventDefault(); create.mutate(); }}>
    <div className="form-grid two-columns">{!mixture && <><Field label="名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="说明"><input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="用途、数据选择依据或版本说明" /></Field></>}<Field label="语义类型" hint="同一个混合只能包含同一种监督语义"><select value={kind} disabled={Boolean(mixture)} onChange={(event) => changeKind(event.target.value as typeof kind)}><option value="conversation">Conversation / SFT</option><option value="preference">Preference / DPO</option></select></Field><Field label="固定 Seed" hint="进入不可变混合身份和确定性读取顺序"><input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field></div>
    <div className="mixture-source-heading"><div><strong>组合监督数据来源</strong><span>勾选任意多个版本并调整权重；不会自动替你选择来源。</span></div><button type="button" className="button secondary small" aria-expanded={filtersOpen} onClick={() => setFiltersOpen((current) => !current)}><SlidersHorizontal size={14} />筛选来源{activeFilterCount ? `（${activeFilterCount}）` : ""}</button></div>
    {filtersOpen && <div className="supervision-mixture-filters"><div className="search-box"><Search size={15} /><input aria-label="搜索后训练混合来源" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索监督映射或原始数据名称" /></div><label><span>最少接受记录</span><input aria-label="最少接受记录" type="number" min="0" value={minimumRecords} onChange={(event) => setMinimumRecords(event.target.value)} placeholder="不限" /></label><label className="check-field"><input type="checkbox" checked={onlySelected} onChange={(event) => setOnlySelected(event.target.checked)} /><span>只看已选来源</span></label><button type="button" className="button ghost small" onClick={() => { setSearch(""); setMinimumRecords(""); setOnlySelected(false); }}>重置筛选</button></div>}
    <div className="supervision-mixture-toolbar"><span>显示 {visible.length} / {compatible.length} 个候选 · 已选 {selectedCandidates.length} 个，共 {formatNumber(selectedRecords)} 条可用记录</span><div><button type="button" className="button secondary small" disabled={!visible.length} onClick={selectVisible}>选择筛选结果</button><button type="button" className="button ghost small" disabled={!selectedCandidates.length} onClick={clearSelection}>清空选择</button></div></div>
    <div className="weight-editor supervision-weight-editor"><header><span>选择 / 监督映射版本</span><span>原始数据版本</span><span>接受 / 拒绝</span><span>权重 / 比例</span></header>{visible.map((item) => { const weight = Number(weights[item.id] || 0); const ratio = totalWeight && selected[item.id] ? weight / totalWeight : 0; return <div className={selected[item.id] ? "selected" : ""} key={item.id}><label className="mixture-source-choice"><input aria-label={`${item.logical_name} V${item.version} 来源选择`} type="checkbox" checked={Boolean(selected[item.id])} onChange={(event) => toggle(item.id, event.target.checked)} /><strong>{item.logical_name}<small>监督映射 V{item.version} · {item.identity_hash.slice(0, 10)}</small></strong></label><span><strong>{item.dataset_name || item.dataset_version_id}</strong><small>Dataset V{item.dataset_version || "—"}</small></span><span><strong>{formatNumber(item.accepted_record_count)} / {formatNumber(item.rejected_record_count)}</strong><small>{item.kind}</small></span><div className="supervision-weight-cell"><input aria-label={`${item.logical_name} V${item.version} 采样权重`} type="number" min="1" max="1000000" step="1" disabled={!selected[item.id]} value={selected[item.id] ? weight || 1 : 0} onChange={(event) => setWeights((current) => ({ ...current, [item.id]: Number(event.target.value) }))} /><b>{(ratio * 100).toFixed(1)}%</b></div></div>; })}</div>
    {!compatible.length && <Alert severity="warning" title={`没有已就绪的 ${kind} 监督数据版本`} detail="先创建并完成监督数据映射扫描，再回来组合。" />}{Boolean(compatible.length) && !visible.length && <Alert severity="info" title="没有匹配当前筛选条件的来源" detail="修改筛选条件或重置筛选即可继续。" />}
    <div className="supervision-mixture-summary"><span><b>{selectedCandidates.length}</b> 个来源</span><span><b>{formatNumber(selectedRecords)}</b> 条可用记录</span><span><b>{formatNumber(totalWeight)}</b> 总权重</span></div>
    <ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!selectedCandidates.length || invalidWeights || totalWeight <= 0 || !Number.isInteger(seed)} onCancel={onClose} submitLabel={mixture ? "创建混合新版本" : "保存后训练混合"} />
  </form></Modal>;
}
