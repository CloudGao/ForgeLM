import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, FileText, PackageCheck, Play, Plug, Plus, Rocket, Send } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { PackageSuiteManager } from "../components/PackageSuiteManager";
import { api, formatDate, post } from "../api";
import { EmptyState, ErrorMessage, ExecutionFields, executionPayload, Field, FormActions, Modal, PageHeader, Panel, StatusBadge } from "../components/ui";

type ReleaseTab = "quantization" | "builds" | "releases" | "plugins";

function schemaDefaults(schema: any): Record<string, any> {
  const result: Record<string, any> = {};
  for (const [key, property] of Object.entries<any>(schema?.properties || {})) {
    if (property.default !== undefined) result[key] = property.default;
  }
  return result;
}

function pluginLabel(plugin: any) {
  if (!plugin) return "未知插件";
  const framework = plugin.capabilities?.framework;
  if (framework === "forgelm-microlm") return "MicroLM W8A16";
  if (framework === "llm-compressor") return "LLM Compressor";
  if (plugin.plugin_id === "plugin.forgelm.native-exporter") return "ForgeLM Native Exporter";
  if (plugin.plugin_id === "plugin.local-directory.publisher") return "Local Directory Publisher";
  return plugin.plugin_id;
}

function sourceFormat(model: any) {
  return model?.metadata_json?.inference?.format || model?.metadata?.inference?.format || "unknown";
}

export function Releases() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get("tab") as ReleaseTab;
  const tab: ReleaseTab = ["quantization", "builds", "releases", "plugins"].includes(requestedTab) ? requestedTab : "quantization";
  const setTab = (nextTab: ReleaseTab) => setSearchParams((previous) => {
    const next = new URLSearchParams(previous); next.set("tab", nextTab); return next;
  });
  const [legacyOpen, setLegacyOpen] = useState(false);
  const [quantJobOpen, setQuantJobOpen] = useState(false);
  const [releaseJobOpen, setReleaseJobOpen] = useState(false);
  const [buildJob, setBuildJob] = useState<any>(null);
  const [deliveryRelease, setDeliveryRelease] = useState<any>(null);

  const models = useQuery({ queryKey: ["models", projectId], queryFn: () => api<any>(`/projects/${projectId}/models`) });
  const plugins = useQuery({ queryKey: ["plugins"], queryFn: () => api<any[]>("/plugins") });
  const bindings = useQuery({ queryKey: ["plugin-bindings", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/plugin-bindings`) });
  const quantJobs = useQuery({ queryKey: ["quantization-jobs", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/quantization-jobs`), refetchInterval: 2500 });
  const quantizations = useQuery({ queryKey: ["quantizations", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/quantizations`), refetchInterval: 2500 });
  const releaseJobs = useQuery({ queryKey: ["release-jobs", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/release-jobs`), refetchInterval: 2500 });
  const releases = useQuery({ queryKey: ["releases", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/releases`), refetchInterval: 2500 });

  const pluginByVersion = useMemo(() => new Map((plugins.data || []).map((plugin) => [plugin.id, plugin])), [plugins.data]);
  const enabledBindings = (bindings.data || []).filter((item) => item.enabled);
  const bindingsOfKind = (kind: string) => enabledBindings.filter((binding) => pluginByVersion.get(binding.plugin_version_id)?.kind === kind);
  const quantBindings = bindingsOfKind("quantization");
  const exporterBindings = bindingsOfKind("release_exporter");
  const publisherBindings = bindingsOfKind("release_publisher");
  const catalogCount = new Set((plugins.data || []).filter((plugin) => plugin.kind === "plugin_suite").map((plugin) => plugin.plugin_id)).size;

  const refresh = () => {
    for (const key of [["plugin-installations"], ["plugin-bindings", projectId], ["quantization-jobs", projectId], ["quantizations", projectId], ["release-jobs", projectId], ["releases", projectId]]) queryClient.invalidateQueries({ queryKey: key });
  };
  const runQuantization = useMutation({ mutationFn: (jobId: string) => post(`/projects/${projectId}/quantization-jobs/${jobId}/runs`), onSuccess: refresh });
  const runDelivery = useMutation({ mutationFn: ({ releaseId, jobId }: any) => post(`/projects/${projectId}/releases/${releaseId}/delivery-jobs/${jobId}/runs`), onSuccess: refresh });

  return <>
    <PageHeader eyebrow="QUANTIZE & RELEASE" title="量化与发布" description="量化、格式构建和目标交付使用精确版本插件；配置可复用，每次执行冻结独立运行实例。" />
    <div className="segmented-tabs release-tabs" role="tablist" aria-label="量化与发布视图">
      <button className={tab === "quantization" ? "active" : ""} onClick={() => setTab("quantization")}>量化任务 <b>{quantJobs.data?.length || 0}</b></button>
      <button className={tab === "builds" ? "active" : ""} onClick={() => setTab("builds")}>发布构建 <b>{releaseJobs.data?.length || 0}</b></button>
      <button className={tab === "releases" ? "active" : ""} onClick={() => setTab("releases")}>正式版本 <b>{releases.data?.length || 0}</b></button>
      <button className={tab === "plugins" ? "active" : ""} onClick={() => setTab("plugins")}>插件管理 <b>{catalogCount}</b></button>
    </div>

    {tab === "quantization" && <div className="content-grid release-grid">
      <Panel title="量化任务" description="任务保存模型、插件版本和参数，可以重复手动运行。" actions={<><button className="button secondary" onClick={() => setLegacyOpen(true)}><Box size={15} />高级用户命令</button><button className="button primary" disabled={!quantBindings.length} onClick={() => setQuantJobOpen(true)}><Plus size={15} />新建任务</button></>}>
        {quantJobs.data?.length ? <div className="quant-job-list">{quantJobs.data.map((job) => <article key={job.id}><div className="quant-format"><strong>{job.config.algorithm}</strong><span>{job.config.scheme}</span></div><div><strong>{job.name}</strong><span>{models.data?.models?.find((model: any) => model.id === job.model_version_id)?.name || job.model_version_id.slice(0, 8)} · {pluginLabel(pluginByVersion.get(job.plugin_version_id))} · 已运行 {job.runs.length} 次</span></div><StatusBadge status={job.enabled ? "ready" : "disabled"} /><button className="button secondary" disabled={!job.enabled || runQuantization.isPending} onClick={() => runQuantization.mutate(job.id)}><Play size={14} />运行</button></article>)}</div> : <EmptyState title="还没有量化任务" description={quantBindings.length ? "新建任务后可以反复运行，每次都会冻结本次参数。" : "先在插件管理安装并为项目启用量化插件。"} />}
        <ErrorMessage error={runQuantization.error} />
      </Panel>
      <Panel title="运行实例" description="每次运行都有自己的编号、参数快照、Task 和输出制品。">{quantizations.data?.length ? <div className="quant-list">{quantizations.data.map((item) => <article key={item.id}><div className="quant-format"><strong>{item.method}</strong><span>{item.format}</span></div><div><strong>{item.run_number ? `运行 #${item.run_number}` : "兼容运行"}</strong><span>{formatDate(item.created_at)}</span></div><div className="quant-delta"><span>插件 {item.plugin_version_id ? item.plugin_version_id.slice(0, 8) : "用户命令"}</span><span>制品 {item.artifact_id ? item.artifact_id.slice(0, 8) : "—"}</span></div><StatusBadge status={item.status} /></article>)}</div> : <EmptyState title="还没有运行实例" description="从量化任务点击运行，或使用高级用户命令。" />}</Panel>
    </div>}

    {tab === "builds" && <div className="content-grid release-grid">
      <Panel title="发布构建任务" description="格式适配任务固定精确源制品、Exporter 版本和参数；版本号在每次运行时输入。" actions={<button className="button primary" disabled={!exporterBindings.length} onClick={() => setReleaseJobOpen(true)}><Plus size={15} />新建构建任务</button>}>
        {releaseJobs.data?.length ? <div className="quant-job-list release-job-list">{releaseJobs.data.map((job) => <article key={job.id}><div className="release-mark"><PackageCheck size={18} /></div><div><strong>{job.name}</strong><span>{job.source_kind === "quantization_run" ? "量化运行" : "模型版本"} {job.source_id.slice(0, 8)} · {pluginLabel(job.exporter_plugin_version)} → {job.exporter_plugin_version?.capabilities?.output_formats?.join(", ")}</span></div><StatusBadge status={job.enabled ? "ready" : "disabled"} /><button className="button secondary" disabled={!job.enabled} onClick={() => setBuildJob(job)}><Play size={14} />运行构建</button></article>)}</div> : <EmptyState title="还没有发布构建任务" description={exporterBindings.length ? "创建任务并选择一个精确模型或成功量化运行。" : "先在插件管理安装并启用格式适配插件。"} />}
      </Panel>
      <Panel title="构建运行" description="构建成功后才会创建不可变正式版本；失败运行不会伪装成 Release。">
        {releaseJobs.data?.some((job) => job.runs?.length) ? <div className="quant-list">{releaseJobs.data?.flatMap((job) => (job.runs || []).map((run: any) => <article key={run.id}><div className="quant-format"><strong>#{run.run_number}</strong><span>BUILD</span></div><div><strong>{run.release_name} {run.release_version}</strong><span>{job.name} · {formatDate(run.created_at)}</span></div><div className="quant-delta"><span>源 {run.source_snapshot?.format || "—"}</span><span>制品 {run.runtime_artifact_id?.slice(0, 8) || "—"}</span></div><StatusBadge status={run.status} /></article>))}</div> : <EmptyState title="还没有构建运行" description="点击某个构建任务的“运行构建”。" />}
      </Panel>
    </div>}

    {tab === "releases" && <Panel title="正式版本" description="Release 是成功构建的不可变运行时版本；每个版本可以配置多个独立交付目标。">
      {releases.data?.length ? <div className="release-version-stack">{releases.data.map((release) => <article className="release-version-card" key={release.id}>
        <header><div className="release-mark"><Rocket size={20} /></div><div><span>{release.version}</span><strong>{release.name}</strong><small>{release.release_kind === "runtime" ? release.runtime_snapshot?.format : "仅 Manifest 的旧版发布"} · {formatDate(release.created_at)}</small></div><StatusBadge status={release.release_kind === "runtime" ? release.gate_result?.status || release.status : "legacy"} />{release.release_kind === "runtime" && <button className="button secondary" disabled={!publisherBindings.length} onClick={() => setDeliveryRelease(release)}><Plus size={14} />添加交付目标</button>}</header>
        {release.release_kind === "legacy_manifest_only" ? <p className="release-warning">该历史记录没有真实运行时制品，不能执行目标交付。</p> : <div className="delivery-stack">{release.delivery_jobs?.length ? release.delivery_jobs.map((job: any) => { const latest = job.runs?.[0]; return <div className="delivery-row" key={job.id}><div><strong><Send size={14} />{job.name}</strong><span>{pluginLabel(job.publisher_plugin_version)} · 已运行 {job.runs.length} 次</span>{latest?.result?.immutable_locator && <a href={latest.result.uri} target="_blank" rel="noreferrer">{latest.result.immutable_locator}</a>}</div>{latest ? <StatusBadge status={latest.status} /> : <StatusBadge status="ready" />}<button className="button secondary" disabled={!job.enabled || runDelivery.isPending} onClick={() => runDelivery.mutate({ releaseId: release.id, jobId: job.id })}><Play size={14} />{latest ? "再次交付" : "交付"}</button></div>; }) : <EmptyState title="尚未配置交付目标" description="添加 Local、Hugging Face 或其他兼容 Publisher；不同目标独立重试。" />}</div>}
      </article>)}</div> : <EmptyState title="还没有正式版本" description="先在“发布构建”中运行一个格式适配任务。" />}
      <ErrorMessage error={runDelivery.error} />
    </Panel>}

    {tab === "plugins" && <PackageSuiteManager projectId={projectId} />}

    {quantJobOpen && <QuantizationJobModal projectId={projectId} models={models.data?.models || []} bindings={quantBindings} plugins={plugins.data || []} onClose={() => setQuantJobOpen(false)} onCreated={() => { refresh(); setQuantJobOpen(false); }} />}
    {legacyOpen && <LegacyQuantizeModal projectId={projectId} models={models.data?.models || []} onClose={() => setLegacyOpen(false)} onCreated={() => { refresh(); setLegacyOpen(false); }} />}
    {releaseJobOpen && <ReleaseJobModal projectId={projectId} models={models.data?.models || []} quantizations={(quantizations.data || []).filter((item) => item.status === "succeeded" && item.artifact_id)} bindings={exporterBindings} plugins={plugins.data || []} onClose={() => setReleaseJobOpen(false)} onCreated={() => { refresh(); setReleaseJobOpen(false); }} />}
    {buildJob && <ReleaseBuildRunModal projectId={projectId} job={buildJob} onClose={() => setBuildJob(null)} onCreated={() => { refresh(); setBuildJob(null); }} />}
    {deliveryRelease && <DeliveryJobModal projectId={projectId} release={deliveryRelease} bindings={publisherBindings} plugins={plugins.data || []} onClose={() => setDeliveryRelease(null)} onCreated={() => { refresh(); setDeliveryRelease(null); }} />}
  </>;
}

function SchemaConfigFields({ schema, value, onChange }: { schema: any; value: Record<string, any>; onChange: (value: Record<string, any>) => void }) {
  const properties = Object.entries<any>(schema?.properties || {});
  if (!properties.length) return <div className="lab-check"><Plug size={17} /><span><strong>无需额外参数</strong>该插件版本没有用户可配置字段。</span></div>;
  const update = (key: string, next: any) => onChange({ ...value, [key]: next });
  return <>{properties.map(([key, property]) => {
    const label = property.title || key;
    if (property.type === "boolean") return <Field key={key} label={label}><label className="checkbox-field"><input type="checkbox" checked={Boolean(value[key])} onChange={(event) => update(key, event.target.checked)} />{property.description || "启用"}</label></Field>;
    if (property.enum) return <Field key={key} label={label}><select value={value[key] ?? property.enum[0]} onChange={(event) => update(key, event.target.value)}>{property.enum.map((option: any) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}</select></Field>;
    if (property.type === "integer" || property.type === "number") return <Field key={key} label={label}><input type="number" value={value[key] ?? ""} min={property.minimum} max={property.maximum} onChange={(event) => update(key, property.type === "integer" ? Number.parseInt(event.target.value || "0", 10) : Number(event.target.value || 0))} /></Field>;
    if (property.type === "array") return <Field key={key} label={`${label}（逗号分隔）`}><input value={(value[key] || []).join(", ")} onChange={(event) => update(key, event.target.value.split(",").map((item) => item.trim()).filter(Boolean))} /></Field>;
    return <Field key={key} label={label}><input value={value[key] ?? ""} minLength={property.minLength} maxLength={property.maxLength} onChange={(event) => update(key, event.target.value)} /></Field>;
  })}</>;
}

function QuantizationJobModal({ projectId, models, bindings, plugins, onClose, onCreated }: { projectId: string; models: any[]; bindings: any[]; plugins: any[]; onClose: () => void; onCreated: () => void }) {
  const [pluginVersionId, setPluginVersionId] = useState(bindings[0]?.plugin_version_id || "");
  const selectedPlugin = plugins.find((item) => item.id === pluginVersionId);
  const selectedBinding = bindings.find((item) => item.plugin_version_id === pluginVersionId);
  const [config, setConfig] = useState<Record<string, any>>({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) });
  useEffect(() => setConfig({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) }), [pluginVersionId]);
  const compatibleModels = models.filter((model) => model.artifact_id && (selectedPlugin?.capabilities?.input_formats || []).includes(sourceFormat(model)));
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/quantization-jobs`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); const samples = String(form.get("samples") || "").split("\n").map((item) => item.trim()).filter(Boolean); create.mutate({ name: form.get("name"), description: form.get("description"), model_version_id: form.get("model_version_id"), plugin_version_id: pluginVersionId, config: { ...config, ...(config.algorithm === "GPTQ" ? { num_calibration_samples: samples.length } : {}) }, calibration_data_ref: config.algorithm === "GPTQ" ? { samples } : {}, resource_request: { gpus: Number(form.get("gpus") || 0) }, launcher_id: "builtin.local-process", enabled: true }); }
  return <Modal title="新建量化任务" description="参数由插件 Schema 生成并属于这条任务；历史运行冻结实际值。" onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns"><Field label="任务名称"><input name="name" required /></Field><Field label="量化插件"><select value={pluginVersionId} onChange={(event) => setPluginVersionId(event.target.value)}>{bindings.map((binding) => { const plugin = plugins.find((item) => item.id === binding.plugin_version_id); return <option key={binding.id} value={binding.plugin_version_id}>{pluginLabel(plugin)} · {plugin?.version}</option>; })}</select></Field><Field label="源模型"><select name="model_version_id" required>{compatibleModels.map((model) => <option key={model.id} value={model.id}>{model.name} v{model.version} · {sourceFormat(model)}</option>)}</select></Field><Field label="GPU 数"><input name="gpus" type="number" min="0" defaultValue="0" /></Field><SchemaConfigFields schema={selectedPlugin?.config_schema} value={config} onChange={setConfig} /><Field label="说明"><input name="description" /></Field>{config.algorithm === "GPTQ" && <Field label="校准文本（每行一条）"><textarea name="samples" required /></Field>}</div><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="保存任务" /></form></Modal>;
}

function LegacyQuantizeModal({ projectId, models, onClose, onCreated }: { projectId: string; models: any[]; onClose: () => void; onCreated: () => void }) {
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/quantizations`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); create.mutate({ model_version_id: form.get("model_version_id"), method: form.get("method"), format: form.get("format"), calibration_data_ref: { type: "project_default" }, ...executionPayload(form) }); }
  return <Modal title="高级用户命令量化" description="兼容入口：任务调用你配置的真实命令，不享受插件参数 Schema 与版本管理。" onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns"><Field label="源模型"><select name="model_version_id" required>{models.map((model) => <option key={model.id} value={model.id}>{model.name} v{model.version}</option>)}</select></Field><Field label="量化方法"><input name="method" defaultValue="Q4" /></Field><Field label="导出格式"><input name="format" defaultValue="GGUF" /></Field><ExecutionFields adapterId="builtin.command.quantization" defaultGpus={1} /></div><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="启动量化" /></form></Modal>;
}

function ReleaseJobModal({ projectId, models, quantizations, bindings, plugins, onClose, onCreated }: { projectId: string; models: any[]; quantizations: any[]; bindings: any[]; plugins: any[]; onClose: () => void; onCreated: () => void }) {
  const [sourceKind, setSourceKind] = useState("model_version");
  const [pluginVersionId, setPluginVersionId] = useState(bindings[0]?.plugin_version_id || "");
  const [gateMode, setGateMode] = useState("bypassed");
  const selectedPlugin = plugins.find((item) => item.id === pluginVersionId);
  const selectedBinding = bindings.find((item) => item.plugin_version_id === pluginVersionId);
  const [config, setConfig] = useState<Record<string, any>>({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) });
  useEffect(() => setConfig({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) }), [pluginVersionId]);
  const create = useMutation({ mutationFn: async (data: any) => { await post(`/projects/${projectId}/release-jobs/preflight`, data); return post(`/projects/${projectId}/release-jobs`, data); }, onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); const gatePolicy = gateMode === "required" ? { mode: "required", gate_result_id: form.get("gate_result_id") } : { mode: "bypassed", reason: form.get("gate_reason") }; create.mutate({ name: form.get("name"), description: form.get("description"), source_kind: sourceKind, source_id: form.get("source_id"), exporter_plugin_version_id: pluginVersionId, config, gate_policy: gatePolicy, model_card_template: form.get("model_card"), license_policy: { status: form.get("license_status") }, resource_request: { gpus: Number(form.get("gpus") || 0) }, launcher_id: "builtin.local-process", enabled: true }); }
  return <Modal title="新建发布构建任务" description="选择精确源制品和格式适配插件；保存前会执行 Worker、格式和制品可见性预检。" onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns"><Field label="任务名称"><input name="name" required /></Field><Field label="源类型"><select value={sourceKind} onChange={(event) => setSourceKind(event.target.value)}><option value="model_version">模型版本</option><option value="quantization_run">成功量化运行</option></select></Field><Field label="精确源制品"><select name="source_id" required>{sourceKind === "model_version" ? models.filter((model) => model.artifact_id).map((model) => <option key={model.id} value={model.id}>{model.name} v{model.version} · {sourceFormat(model)}</option>) : quantizations.map((run) => <option key={run.id} value={run.id}>运行 #{run.run_number || "—"} · {run.format}</option>)}</select></Field><Field label="格式适配插件"><select value={pluginVersionId} onChange={(event) => setPluginVersionId(event.target.value)}>{bindings.map((binding) => { const plugin = plugins.find((item) => item.id === binding.plugin_version_id); return <option key={binding.id} value={binding.plugin_version_id}>{pluginLabel(plugin)} · {plugin?.version} → {plugin?.capabilities?.output_formats?.join(", ")}</option>; })}</select></Field><SchemaConfigFields schema={selectedPlugin?.config_schema} value={config} onChange={setConfig} /><Field label="GPU 数"><input name="gpus" type="number" min="0" defaultValue="0" /></Field><Field label="Gate 决策"><select value={gateMode} onChange={(event) => setGateMode(event.target.value)}><option value="bypassed">显式跳过并记录原因</option><option value="required">引用已通过的 GateResult</option></select></Field>{gateMode === "bypassed" ? <Field label="跳过原因"><textarea name="gate_reason" required /></Field> : <Field label="GateResult ID"><input name="gate_result_id" required /></Field>}<Field label="License 状态"><select name="license_status"><option value="reviewed">已审核</option><option value="restricted">受限</option><option value="unknown">未知</option></select></Field><Field label="说明"><input name="description" /></Field><Field label="模型卡模板"><textarea name="model_card" required /></Field><div className="preflight-list compact"><div className="lab-check"><FileText size={17} /><span><strong>运行时快照</strong>源哈希、插件精确版本、有效参数与 Gate 会在每次运行冻结。</span></div></div></div><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="预检并保存" /></form></Modal>;
}

function ReleaseBuildRunModal({ projectId, job, onClose, onCreated }: { projectId: string; job: any; onClose: () => void; onCreated: () => void }) {
  const run = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/release-jobs/${job.id}/runs`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); run.mutate({ release_name: form.get("release_name"), release_version: form.get("release_version") }); }
  return <Modal title="运行发布构建" description={`将冻结任务“${job.name}”的当前配置。构建成功后才创建正式版本。`} onClose={onClose}><form onSubmit={submit}><div className="form-grid"><Field label="发布名称"><input name="release_name" required /></Field><Field label="新版本号"><input name="release_version" required placeholder="v1.0.0" /></Field><div className="lab-check"><PackageCheck size={17} /><span><strong>{pluginLabel(job.exporter_plugin_version)}</strong>{job.source_id} → {job.exporter_plugin_version?.capabilities?.output_formats?.join(", ")}</span></div></div><ErrorMessage error={run.error} /><FormActions submitting={run.isPending} onCancel={onClose} submitLabel="创建构建运行" /></form></Modal>;
}

function DeliveryJobModal({ projectId, release, bindings, plugins, onClose, onCreated }: { projectId: string; release: any; bindings: any[]; plugins: any[]; onClose: () => void; onCreated: () => void }) {
  const compatible = bindings.filter((binding) => (plugins.find((item) => item.id === binding.plugin_version_id)?.capabilities?.input_formats || []).includes(release.runtime_snapshot?.format));
  const [pluginVersionId, setPluginVersionId] = useState(compatible[0]?.plugin_version_id || "");
  const selectedPlugin = plugins.find((item) => item.id === pluginVersionId);
  const selectedBinding = compatible.find((item) => item.plugin_version_id === pluginVersionId);
  const [config, setConfig] = useState<Record<string, any>>({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) });
  useEffect(() => setConfig({ ...schemaDefaults(selectedPlugin?.config_schema), ...(selectedBinding?.default_config || {}) }), [pluginVersionId]);
  const create = useMutation({ mutationFn: async (data: any) => { await post(`/projects/${projectId}/releases/${release.id}/delivery-preflight`, data); const job = await post<any>(`/projects/${projectId}/releases/${release.id}/delivery-jobs`, data); return post(`/projects/${projectId}/releases/${release.id}/delivery-jobs/${job.id}/runs`); }, onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); const form = new FormData(event.currentTarget); create.mutate({ name: form.get("name"), publisher_plugin_version_id: pluginVersionId, config, resource_request: { gpus: 0 }, launcher_id: "builtin.local-process", enabled: true }); }
  return <Modal title="添加目标交付" description={`${release.name} ${release.version} · ${release.runtime_snapshot?.format}。保存后创建独立交付运行。`} onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns"><Field label="目标名称"><input name="name" required /></Field><Field label="目标交付插件"><select value={pluginVersionId} onChange={(event) => setPluginVersionId(event.target.value)}>{compatible.map((binding) => { const plugin = plugins.find((item) => item.id === binding.plugin_version_id); return <option key={binding.id} value={binding.plugin_version_id}>{pluginLabel(plugin)} · {plugin?.version}</option>; })}</select></Field><SchemaConfigFields schema={selectedPlugin?.config_schema} value={config} onChange={setConfig} /><div className="lab-check"><Send size={17} /><span><strong>独立交付运行</strong>失败或重试不会重新执行格式构建，也不会改变 Release 状态。</span></div></div>{!compatible.length && <ErrorMessage error={new Error("没有已启用且兼容当前运行时格式的 Publisher")} />}<ErrorMessage error={create.error} /><FormActions submitting={create.isPending} onCancel={onClose} submitLabel="保存并交付" /></form></Modal>;
}
