import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Package, Search, Settings2 } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, formatDate, post } from "../api";
import { TaskDetail } from "./TaskDetail";
import { EmptyState, ErrorMessage, Field, FormActions, Modal, Panel, StatusBadge } from "./ui";
import { PluginConfigFields } from "./PluginConfigFields";
import { compareVersions, installBlocker, installedVersion, isInstallable, pluginKindLabels, pluginLabel, schemaDefaults } from "./pluginManagement";
import type { PluginBinding, PluginInstallation, PluginVersion, PluginWorker } from "./pluginManagement";
import "./PluginManager.css";

type InstallResult = { worker: PluginWorker; taskId?: string; error?: string };
type Operation = { id: string; kind: string; title: string; project_id: string; observed_state: string; created_at: string; payload?: Record<string, any> };
type Tab = "catalog" | "workers" | "operations";
const terminalStates = new Set(["succeeded", "failed", "cancelled", "lost"]);

export function PluginManager({ projectId: fixedProjectId, componentId, operationsOnly }: { projectId?: string; componentId?: string; operationsOnly?: boolean }) {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const projectId = fixedProjectId || params.get("project") || "";
  const rawTab = params.get("pluginTab");
  const tab: Tab = operationsOnly ? "operations" : componentId ? "catalog" : rawTab === "workers" || rawTab === "operations" ? rawTab : "catalog";
  const selectedId = componentId || params.get("plugin") || "";
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState("");
  const [source, setSource] = useState("installable");
  const [workerFilter, setWorkerFilter] = useState("");
  const [page, setPage] = useState(0);
  const [installTarget, setInstallTarget] = useState<{ plugin: PluginVersion; workerId?: string } | null>(null);
  const [configTarget, setConfigTarget] = useState<PluginVersion | null>(null);
  const [taskId, setTaskId] = useState("");
  const plugins = useQuery({ queryKey: ["plugins"], queryFn: () => api<PluginVersion[]>("/plugins") });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<PluginWorker[]>("/workers"), refetchInterval: 5000 });
  const installations = useQuery({ queryKey: ["plugin-installations"], queryFn: () => api<PluginInstallation[]>("/plugin-installations"), refetchInterval: 2500 });
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Array<{ id: string; name: string; archived?: boolean }>>("/projects"), enabled: !fixedProjectId });
  const bindings = useQuery({ queryKey: ["plugin-bindings", projectId], queryFn: () => api<PluginBinding[]>(`/projects/${projectId}/plugin-bindings`), enabled: Boolean(projectId) });
  const operations = useQuery({ queryKey: ["plugin-operations", projectId, page], queryFn: async () => {
    const rows = await api<Operation[]>(`/tasks?kind=plugin_install&include_events=false&limit=31&offset=${page * 30}${projectId ? `&project_id=${encodeURIComponent(projectId)}` : ""}`);
    if (rows.some((row) => row.kind !== "plugin_install")) throw new Error("控制面尚未加载插件任务过滤更新，请重启控制面后查看操作记录。");
    return rows;
  }, enabled: tab === "operations", refetchInterval: 2500 });
  const navigate = (changes: Record<string, string>) => setParams((previous) => {
    const next = new URLSearchParams(previous);
    Object.entries(changes).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key));
    return next;
  });
  const refresh = () => {
    for (const key of [["plugin-installations"], ["plugin-bindings"], ["plugin-operations"], ["tasks"], ["quantization-jobs"], ["release-jobs"], ["releases"]]) void client.invalidateQueries({ queryKey: key });
  };
  const cancel = useMutation({ mutationFn: (id: string) => post(`/tasks/${id}/control`, { action: "cancel", reason: "用户在插件管理中取消安装任务" }), onSuccess: refresh });
  const all = plugins.data || [];
  const grouped = new Map<string, PluginVersion[]>();
  for (const plugin of all) grouped.set(plugin.plugin_id, [...(grouped.get(plugin.plugin_id) || []), plugin]);
  const groups = [...grouped.values()].map((versions) => versions.sort((a, b) => compareVersions(b.version, a.version)));
  const visible = groups.filter(([plugin]) => (!kind || plugin.kind === kind) && (source === "all" || (source === "installable" ? isInstallable(plugin) : !isInstallable(plugin))) && `${pluginLabel(plugin)} ${plugin.plugin_id} ${pluginKindLabels[plugin.kind] || plugin.kind}`.toLowerCase().includes(search.toLowerCase()));
  const selected = all.find((plugin) => plugin.id === selectedId);
  const selectedBinding = (bindings.data || []).find((binding) => binding.plugin_id === selected?.plugin_id);
  const installationFor = (plugin: PluginVersion, workerId: string) => (installations.data || []).find((row) => row.worker_id === workerId && row.plugin_id === plugin.plugin_id);
  const installedWorkers = (plugin: PluginVersion) => (installations.data || []).filter((row) => installedVersion(row, plugin.id));
  const projectReady = Boolean(projectId) && bindings.isSuccess;
  const runtimeReady = workers.isSuccess && installations.isSuccess;

  return <div className="plugin-manager">
    {!componentId && !operationsOnly && <div className="plugin-toolbar">
      <div className="plugin-tabs" role="tablist" aria-label="插件管理视图">{([["catalog", "插件目录"], ["workers", "Worker 安装"], ["operations", "操作记录"]] as const).map(([value, label]) => <button key={value} role="tab" aria-selected={tab === value} className={tab === value ? "active" : ""} onClick={() => navigate({ pluginTab: value, plugin: "" })}>{label}</button>)}</div>
      {!fixedProjectId && <Field label="项目上下文" hint="安装任务归属所选项目；安装不会自动为项目启用插件。"><select value={projectId} onChange={(event) => { setPage(0); setConfigTarget(null); setInstallTarget(null); navigate({ project: event.target.value }); }}><option value="">查看全部 / 选择项目后操作</option>{(projects.data || []).filter((project) => !project.archived || project.id === projectId).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></Field>}
    </div>}
    <ErrorMessage error={plugins.error || workers.error || installations.error || bindings.error || projects.error} />
    {plugins.isPending && <p role="status">正在读取插件目录…</p>}

    {tab === "catalog" && !selected && <>
      <div className="plugin-filters"><Field label="搜索插件"><div className="plugin-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="名称、标识或用途" /></div></Field><Field label="用途"><select value={kind} onChange={(event) => setKind(event.target.value)}><option value="">全部用途</option>{[...new Set(all.map((plugin) => plugin.kind))].sort().map((value) => <option key={value} value={value}>{pluginKindLabels[value] || value}</option>)}</select></Field><Field label="安装方式"><select value={source} onChange={(event) => setSource(event.target.value)}><option value="installable">可安装插件</option><option value="all">全部目录</option><option value="bundled">随平台提供 / 仅登记</option></select></Field></div>
      <div className="plugin-catalog-grid">{visible.map((versions) => {
        const plugin = versions.find((version) => version.enabled !== false) || versions[0];
        const binding = (bindings.data || []).find((item) => item.plugin_id === plugin.plugin_id);
        const installedCount = new Set((installations.data || []).filter((row) => row.plugin_id === plugin.plugin_id && row.installed_versions?.length).map((row) => row.worker_id)).size;
        return <article className="plugin-catalog-card" key={plugin.plugin_id}>
          <div className="plugin-card-heading"><span className="plugin-package-icon"><Package size={22} /></span><span>{pluginKindLabels[plugin.kind] || plugin.kind}</span></div>
          <h3><button onClick={() => navigate({ plugin: plugin.id })}>{pluginLabel(plugin)}</button></h3><code>{plugin.plugin_id}</code>
          <p>{isInstallable(plugin) ? "受控插件目录" : plugin.plugin_id.startsWith("builtin.") ? "随平台提供，依赖实际 Worker 能力" : "仅登记，未提供通用安装能力"} · {plugin.version} · {versions.length} 个版本</p>
          <dl><div><dt>Worker 安装</dt><dd>{installations.isSuccess ? `${installedCount} 台` : "读取中"}</dd></div><div><dt>项目启用</dt><dd>{!projectId ? "未选择项目" : !bindings.isSuccess ? "读取中" : binding?.enabled ? `已启用 ${all.find((item) => item.id === binding.plugin_version_id)?.version || "历史版本"}` : "未启用"}</dd></div></dl>
          <button className="button secondary" onClick={() => navigate({ plugin: plugin.id })}>查看版本与配置</button>
        </article>;
      })}</div>
      {plugins.isSuccess && !visible.length && <EmptyState title={all.length ? "没有符合条件的插件" : "目录中没有插件"} description={all.length ? "调整用途、安装方式或搜索条件。" : "请检查控制面是否已登记内置目录。"} />}
      <p className="plugin-boundary">自定义包导入、环境修复和卸载尚未接入。训练与推理的内置适配器可在“全部目录”查看；目录登记不代表已支持通用 HF 微调。</p>
    </>}

    {tab === "catalog" && selected && <>
      {!componentId && <button className="button ghost" onClick={() => navigate({ plugin: "" })}>← 返回插件目录</button>}
      <Panel title={pluginLabel(selected)} description={`${selected.plugin_id} · ${pluginKindLabels[selected.kind] || selected.kind}`}>
        <div className="plugin-detail-content">
          <div className="plugin-filters"><Field label="插件版本"><select disabled={Boolean(componentId)} value={selected.id} onChange={(event) => navigate({ plugin: event.target.value })}>{(grouped.get(selected.plugin_id) || []).map((version) => <option key={version.id} value={version.id}>{version.version}{version.enabled === false ? " · 已停用" : ""}</option>)}</select></Field><div><StatusBadge status={selected.enabled === false ? "disabled" : "available"} /><p>{selected.support_level || "未声明支持级别"} · 协议 {selected.protocol_version ?? "历史未记录"}</p></div></div>
          <dl className="plugin-facts"><div><dt>安装方式</dt><dd>{isInstallable(selected) ? selected.runtime?.isolation === "venv" ? "Worker 独立 Python 环境" : "Worker 运行时子进程" : "随平台提供或仅登记"}</dd></div><div><dt>已安装 Worker</dt><dd>{installations.isSuccess ? `${installedWorkers(selected).length} 台（精确版本）` : "读取中"}</dd></div><div><dt>Python 要求</dt><dd>{selected.runtime?.python || "未单独声明"}</dd></div><div><dt>框架</dt><dd>{selected.capabilities?.framework || "由插件定义"} {selected.capabilities?.framework_version || ""}</dd></div></dl>
          <div className="plugin-formats"><div><h4>输入格式</h4><p>{(selected.capabilities?.input_formats || []).join("、") || "该协议未声明类型化输入"}</p></div><div><h4>输出格式</h4><p>{(selected.capabilities?.output_formats || []).join("、") || "该协议未声明类型化输出"}</p></div></div>
          <div className="plugin-action-row">{!componentId && <button className="button primary" disabled={!isInstallable(selected) || selected.enabled === false || !projectReady || !runtimeReady} onClick={() => setInstallTarget({ plugin: selected })}><Download size={16} />安装到 Worker</button>}<button className="button secondary" disabled={!isInstallable(selected) || selected.enabled === false || !projectReady} onClick={() => setConfigTarget(selected)}><Settings2 size={16} />项目启用与配置</button></div>
          {!projectId && <p>选择项目上下文后可以安装和配置。</p>}
          {selectedBinding && <p>项目当前绑定：{all.find((item) => item.id === selectedBinding.plugin_version_id)?.version || selectedBinding.plugin_version_id} · {selectedBinding.enabled ? "已启用" : "已停用"}。安装新版本不会自动改变此绑定。</p>}
          {!isInstallable(selected) && <p>此记录不提供独立安装和项目插件绑定；实际能力由 Worker 与对应业务入口决定。</p>}
          <VersionComparison plugin={selected} versions={grouped.get(selected.plugin_id) || []} />
          <details><summary>参数契约与运行依赖</summary><pre>{JSON.stringify({ config_schema: selected.config_schema || {}, runtime: selected.runtime || {}, capabilities: selected.capabilities || {}, source: selected.source || {} }, null, 2)}</pre></details>
        </div>
      </Panel>
    </>}
    {tab === "catalog" && selectedId && !selected && plugins.isSuccess && <p role="alert">该插件版本不在当前目录中，历史运行仍可从任务详情查看。</p>}

    {tab === "workers" && <Panel title="Worker 安装矩阵" description="每格对应精确插件版本与 Worker。离线时显示最后已知事实，不代表当前可执行。">
      <div className="plugin-filters"><Field label="目标 Worker"><select value={workerFilter} onChange={(event) => setWorkerFilter(event.target.value)}><option value="">全部 Worker</option>{(workers.data || []).map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}</select></Field></div>
      {workers.isPending && <p role="status">正在读取 Worker…</p>}
      {workers.isSuccess && !(workers.data || []).length ? <EmptyState title="尚未连接 Worker" description="先在执行环境中连接 Worker，再安装插件。" /> : <div className="plugin-matrix-scroll"><table className="data-table plugin-matrix"><thead><tr><th>插件 / 版本</th>{(workers.data || []).filter((worker) => !workerFilter || worker.id === workerFilter).map((worker) => <th key={worker.id}>{worker.name}<br /><StatusBadge status={worker.status} /></th>)}</tr></thead><tbody>{all.filter(isInstallable).map((plugin) => <tr key={plugin.id}><th><button className="button ghost" onClick={() => navigate({ pluginTab: "catalog", plugin: plugin.id })}>{pluginLabel(plugin)}</button><br />{plugin.version}</th>{(workers.data || []).filter((worker) => !workerFilter || worker.id === workerFilter).map((worker) => {
        const row = installationFor(plugin, worker.id);
        const installed = installedVersion(row, plugin.id);
        const blocker = installBlocker(plugin, worker, row);
        const pending = row?.target_plugin_version?.id === plugin.id && ["installing", "updating"].includes(row?.state || "");
        const failed = row?.target_plugin_version?.id === plugin.id && row?.state === "failed";
        return <td key={worker.id}><strong>{!installations.isSuccess ? "读取中" : installed ? "已安装" : pending ? "安装中" : failed ? "安装失败" : "未安装"}</strong>{installed && <small>最后验证 {formatDate(row?.last_verified_at)}</small>}{blocker && <small>{blocker}</small>}{failed && <small>{row?.last_error?.message || "查看日志了解失败原因"}</small>}{!installed && <button className="button secondary" disabled={Boolean(blocker) || !projectReady || !runtimeReady} onClick={() => setInstallTarget({ plugin, workerId: worker.id })}>安装 {plugin.version}</button>}{row?.task_id && <button className="button ghost" onClick={() => setTaskId(row.task_id!)}>最近安装日志</button>}</td>;
      })}</tr>)}</tbody></table></div>}
    </Panel>}

    {tab === "operations" && <Panel title="插件操作记录" description="保存安装和更新的真实任务。刷新页面后仍可查看进度、日志与失败原因。">
      <ErrorMessage error={operations.error || cancel.error} />
      {operations.isPending && <p role="status">正在读取操作记录…</p>}
      <div className="plugin-operations">{(operations.data || []).slice(0, 30).map((task) => <article key={task.id}><div><strong>{task.title}</strong><small>{task.payload?.plugin_suite ? `套件 ${task.payload.plugin_suite.suite_id} ${task.payload.plugin_suite.version} · ${task.payload.plugin_suite.component_key}` : "独立组件 / 历史记录"}</small><small>{formatDate(task.created_at)} · {(workers.data || []).find((worker) => worker.id === task.payload?.resource_request?.worker_selector?.worker_id)?.name || task.payload?.resource_request?.worker_selector?.worker_id || "历史未记录 Worker"}</small><small>项目：{(projects.data || []).find((project) => project.id === task.project_id)?.name || task.project_id}</small></div><StatusBadge status={task.observed_state} /><button className="button secondary" onClick={() => setTaskId(task.id)}>查看日志</button>{!terminalStates.has(task.observed_state) && <button className="button ghost" disabled={cancel.isPending} onClick={() => cancel.mutate(task.id)}>取消安装</button>}</article>)}</div>
      {operations.isSuccess && !operations.data.length && <EmptyState title="暂无插件操作" description="在目录中选择版本并安装后，任务会出现在这里。" />}
      <div className="plugin-action-row"><button className="button secondary" disabled={page === 0 || operations.isPending} onClick={() => setPage(page - 1)}>上一页</button><span>第 {page + 1} 页</span><button className="button secondary" disabled={(operations.data?.length || 0) <= 30 || operations.isPending} onClick={() => setPage(page + 1)}>下一页</button></div>
    </Panel>}
    {installTarget && <InstallDialog key={`${projectId}:${installTarget.plugin.id}`} projectId={projectId} plugin={installTarget.plugin} workers={workers.data || []} installations={installations.data || []} initialWorkerId={installTarget.workerId} onClose={() => setInstallTarget(null)} onRefresh={refresh} onTask={setTaskId} />}
    {configTarget && <BindingDialog key={`${projectId}:${configTarget.id}`} projectId={projectId} plugin={configTarget} binding={(bindings.data || []).find((binding) => binding.plugin_id === configTarget.plugin_id)} onClose={() => setConfigTarget(null)} onSaved={refresh} />}
    {taskId && <TaskDetail taskId={taskId} onClose={() => setTaskId("")} />}
    {!componentId && !operationsOnly && !fixedProjectId && projectId && <Link className="button ghost" to={`/projects/${projectId}/releases?tab=plugins`}>返回项目插件入口</Link>}
  </div>;
}

function VersionComparison({ plugin, versions }: { plugin: PluginVersion; versions: PluginVersion[] }) {
  const [otherId, setOtherId] = useState("");
  const other = versions.find((version) => version.id === otherId && version.id !== plugin.id);
  return versions.length < 2 ? null : <details><summary>比较版本变化</summary><Field label="对比版本"><select value={other?.id || ""} onChange={(event) => setOtherId(event.target.value)}><option value="">选择其他版本</option>{versions.filter((version) => version.id !== plugin.id).map((version) => <option key={version.id} value={version.id}>{version.version}</option>)}</select></Field>{other && <div className="plugin-matrix-scroll"><table className="data-table"><thead><tr><th>内容</th><th>{other.version}</th><th>{plugin.version}</th></tr></thead><tbody>{([['参数契约', 'config_schema'], ['能力', 'capabilities'], ['运行依赖', 'runtime']] as const).map(([label, key]) => <tr key={key}><th>{label}{JSON.stringify(other[key]) === JSON.stringify(plugin[key]) ? "（相同）" : "（有变化）"}</th><td><pre>{JSON.stringify(other[key] || {}, null, 2)}</pre></td><td><pre>{JSON.stringify(plugin[key] || {}, null, 2)}</pre></td></tr>)}</tbody></table></div>}</details>;
}

function InstallDialog({ projectId, plugin, workers, installations, initialWorkerId, onClose, onRefresh, onTask }: {
  projectId: string; plugin: PluginVersion; workers: PluginWorker[]; installations: PluginInstallation[];
  initialWorkerId?: string; onClose: () => void; onRefresh: () => void; onTask: (id: string) => void;
}) {
  const [selected, setSelected] = useState<string[]>(initialWorkerId ? [initialWorkerId] : []);
  const [results, setResults] = useState<InstallResult[]>([]);
  const rowFor = (worker: PluginWorker) => installations.find((row) => row.worker_id === worker.id && row.plugin_id === plugin.plugin_id);
  const candidates = workers.filter((worker) => selected.includes(worker.id) && !installBlocker(plugin, worker, rowFor(worker)) && !results.some((result) => result.worker.id === worker.id && result.taskId));
  const submit = useMutation({ mutationFn: async () => {
    const next = await Promise.all(candidates.map(async (worker): Promise<InstallResult> => {
      try {
        const row = rowFor(worker);
        const updating = row?.current_plugin_version && compareVersions(plugin.version, row.current_plugin_version.version) > 0;
        const response = updating
          ? await post<{ task: { id: string } }>(`/plugin-installations/${row!.id}/update`, { project_id: projectId, target_plugin_version_id: plugin.id })
          : await post<{ task: { id: string } }>(`/projects/${projectId}/plugin-installations`, { worker_id: worker.id, plugin_version_id: plugin.id });
        return { worker, taskId: response.task.id };
      } catch (error) { return { worker, error: error instanceof Error ? error.message : "提交失败" }; }
    }));
    setResults((previous) => [...previous.filter((result) => !next.some((item) => item.worker.id === result.worker.id)), ...next]);
  }, onSettled: onRefresh });
  return <Modal title={`安装 ${pluginLabel(plugin)} · ${plugin.version}`} description="为每台 Worker 创建独立任务，保留已安装的旧版本。" onClose={onClose} wide><form onSubmit={(event) => { event.preventDefault(); submit.mutate(); }}>
    <div className="plugin-install-plan"><p>Python：{plugin.runtime?.python || "由运行时决定"} · {plugin.runtime?.isolation || "未声明隔离方式"}</p><p>固定依赖：{(plugin.runtime?.requirements || []).join("、") || "使用随 Worker 发布的运行时"}</p><p>下载量与磁盘需求：当前目录未提供预估，Worker 安装时检查。宿主驱动不会自动更新。</p></div>
    <fieldset className="plugin-worker-picker"><legend>选择目标 Worker</legend>{workers.map((worker) => {
      const blocker = installBlocker(plugin, worker, rowFor(worker));
      const submitted = results.some((result) => result.worker.id === worker.id && result.taskId);
      return <label key={worker.id}><input type="checkbox" checked={selected.includes(worker.id)} disabled={Boolean(blocker) || submitted || submit.isPending} onChange={(event) => setSelected((previous) => event.target.checked ? [...previous, worker.id] : previous.filter((id) => id !== worker.id))} /><span>{worker.name}<small>{blocker || (submitted ? "任务已提交" : "可提交；依赖与运行时将在安装时验证")}</small></span><StatusBadge status={worker.status} /></label>;
    })}{!workers.length && <p>没有可选 Worker，请先连接执行环境。</p>}</fieldset>
    {results.length > 0 && <div aria-live="polite" className="plugin-install-results">{results.map((result) => <div key={result.worker.id}><strong>{result.worker.name}</strong>{result.taskId ? <><span>任务已提交，等待实际安装结果</span><button type="button" className="button ghost" onClick={() => onTask(result.taskId!)}>查看任务</button></> : <span role="alert">提交失败：{result.error}</span>}</div>)}</div>}
    <ErrorMessage error={submit.error} /><div className="plugin-action-row"><button type="button" className="button secondary" onClick={onClose}>关闭</button><button className="button primary" disabled={!projectId || !candidates.length || submit.isPending}>{submit.isPending ? "提交中…" : results.length ? "提交剩余目标" : `安装到 ${candidates.length} 台 Worker`}</button></div>
  </form></Modal>;
}

function BindingDialog({ projectId, plugin, binding, onClose, onSaved }: {
  projectId: string; plugin: PluginVersion; binding?: PluginBinding; onClose: () => void; onSaved: () => void;
}) {
  const [config, setConfig] = useState<Record<string, any>>(() => binding ? structuredClone(binding.default_config || {}) : schemaDefaults(plugin.config_schema));
  const [enabled, setEnabled] = useState(binding?.enabled ?? true);
  const [invalid, setInvalid] = useState<Record<string, boolean>>({});
  const save = useMutation({ mutationFn: () => api(`/projects/${projectId}/plugin-bindings/${encodeURIComponent(plugin.plugin_id)}`, { method: "PUT", body: JSON.stringify({ plugin_version_id: plugin.id, enabled, default_config: config }) }), onSuccess: () => { onSaved(); onClose(); } });
  return <Modal title={`项目配置 · ${pluginLabel(plugin)}`} description={`保存后绑定精确版本 ${plugin.version}。参数由该版本契约校验。`} onClose={onClose} wide><form onSubmit={(event) => { event.preventDefault(); save.mutate(); }}>
    <p>绑定变更影响之后启动的量化/发布任务；已创建运行的版本与参数快照保留。停用不会卸载环境或取消正在执行的任务。</p>
    {binding && binding.plugin_version_id !== plugin.id && <p role="status">本次将切换项目绑定版本。已保留原默认参数，请根据新契约调整；不兼容字段会阻止保存。</p>}
    <Field label="项目启用"><label className="checkbox-field"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />允许项目使用此插件</label></Field>
    <PluginConfigFields schema={plugin.config_schema || {}} value={config} onChange={setConfig} onValidity={(key, valid) => setInvalid((previous) => ({ ...previous, [key]: !valid }))} />
    <details><summary>当前完整参数</summary><pre>{JSON.stringify(config, null, 2)}</pre></details>
    <ErrorMessage error={save.error} />
    {Object.values(invalid).some(Boolean) && <p role="alert">请先修正 JSON 字段，再保存配置。</p>}
    <FormActions submitting={save.isPending} disabled={Object.values(invalid).some(Boolean)} onCancel={onClose} submitLabel="保存项目配置" />
  </form></Modal>;
}
