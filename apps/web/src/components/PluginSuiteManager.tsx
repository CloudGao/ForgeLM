import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, post } from "../api";
import { PluginManager } from "./PluginManager";
import { TaskDetail } from "./TaskDetail";
import { EmptyState, ErrorMessage, Field, Modal, Panel, StatusBadge } from "./ui";
import type { PluginVersion, PluginWorker } from "./pluginManagement";
import "./PluginManager.css";

export type SuiteComponent = { key: string; label: string; node_kind: string; plugin_id: string; plugin_version_id: string; version: string; deployment: string; note: string; plugin: PluginVersion | null };
export type PluginSuite = { id: string; suite_id: string; name: string; version: string; description: string; enabled: boolean; architectures: string[]; manifest_hash: string; components: SuiteComponent[] };
type Deployment = { suite_version_id: string; worker_id: string; worker_name: string; worker_status: string; state: string; ready_count: number; total_count: number; project_configuration_required: boolean; components: Array<{ key: string; state: string; task_id?: string; last_verified_at?: string }> };
const positions: Record<string, string> = { training: "训练节点", inference: "模型加载 / 推理", evaluation: "评测节点", quantization: "量化节点", export: "导出节点", delivery: "交付节点", model_import: "权重获取", data_import: "数据导入", transform: "数据转换", clean: "数据清洗", tokenizer: "Tokenizer 节点", preparation: "监督数据准备" };
const states: Record<string, string> = { installed: "已安装精确版本", provided: "Worker 已提供", project_required: "需项目代码配置", missing: "未具备", failed: "安装失败", installing: "安装中", unavailable: "组件版本不可用" };
const modes: Record<string, string> = { installable: "独立安装", bundled: "随 Worker 提供", project_code: "项目代码与配置" };
function destination(kind: string, projectId: string) {
  if (["quantization", "export", "delivery"].includes(kind)) return projectId ? `/projects/${projectId}/releases?tab=${kind === "quantization" ? "quantization" : kind === "export" ? "builds" : "releases"}` : "";
  if (["training", "inference", "evaluation", "tokenizer", "preparation"].includes(kind)) return projectId ? `/projects/${projectId}/${kind === "inference" ? "playground" : kind === "evaluation" ? "evaluations" : kind === "tokenizer" ? "tokenizers" : "training"}` : "";
  return kind === "model_import" ? "/weights" : "/data";
}

export function PluginSuiteManager({ projectId: fixedProjectId }: { projectId?: string }) {
  const [params, setParams] = useSearchParams();
  const client = useQueryClient();
  const projectId = fixedProjectId || params.get("project") || "";
  const tab = ["capabilities", "workers", "operations"].includes(params.get("pluginTab") || "") ? params.get("pluginTab")! : "catalog";
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("");
  const [workerId, setWorkerId] = useState("");
  const [target, setTarget] = useState<{ suite: PluginSuite; workerId?: string } | null>(null);
  const [taskId, setTaskId] = useState("");
  const suites = useQuery({ queryKey: ["plugin-suites"], queryFn: () => api<PluginSuite[]>("/plugin-suites") });
  const deployments = useQuery({ queryKey: ["plugin-suite-deployments"], queryFn: () => api<Deployment[]>("/plugin-suite-deployments"), refetchInterval: 3000 });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<PluginWorker[]>("/workers"), refetchInterval: 5000 });
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Array<{ id: string; name: string; archived?: boolean }>>("/projects"), enabled: !fixedProjectId });
  const navigate = (values: Record<string, string>) => setParams((previous) => { const next = new URLSearchParams(previous); Object.entries(values).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key)); return next; });
  const selected = (suites.data || []).find((suite) => suite.id === params.get("suite")) || (params.get("plugin") ? (suites.data || []).find((suite) => suite.components.some((component) => component.plugin_version_id === params.get("plugin"))) : undefined);
  const component = selected?.components.find((member) => member.key === params.get("component") || member.plugin_version_id === params.get("plugin"));
  const visible = (suites.data || []).filter((suite) => `${suite.name} ${suite.description} ${suite.architectures.join(" ")}`.toLowerCase().includes(search.toLowerCase()) && (!position || suite.components.some((member) => member.node_kind === position)));
  const deploymentFor = (suite: PluginSuite, id: string) => (deployments.data || []).find((row) => row.suite_version_id === suite.id && row.worker_id === id);
  const refresh = () => { for (const key of [["plugin-suite-deployments"], ["plugin-installations"], ["plugin-operations"]]) void client.invalidateQueries({ queryKey: key }); };
  const openSuite = (suite: PluginSuite, key = "") => navigate({ pluginTab: "catalog", suite: suite.id, component: key, plugin: "" });

  return <div className="plugin-manager suite-manager">
    <div className="plugin-toolbar"><div className="plugin-tabs" role="tablist" aria-label="套件管理视图">{[["catalog", "套件目录"], ["capabilities", "节点能力"], ["workers", "Worker 安装"], ["operations", "操作记录"]].map(([value, label]) => <button role="tab" aria-selected={tab === value} className={tab === value ? "active" : ""} key={value} onClick={() => navigate({ pluginTab: value, suite: "", component: "", plugin: "" })}>{label}</button>)}</div>
      {!fixedProjectId && <Field label="操作项目" hint="用于项目配置与安装任务归属"><select value={projectId} onChange={(event) => { setTarget(null); navigate({ project: event.target.value }); }}><option value="">请选择项目</option>{(projects.data || []).filter((project) => !project.archived || project.id === projectId).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></Field>}
    </div>
    <ErrorMessage error={suites.error || deployments.error || workers.error || projects.error} />
    {suites.isPending && <p role="status">正在加载插件套件…</p>}
    {tab === "catalog" && !selected && <>
      <p className="suite-intro">一个套件提供一整套模型或数据流程能力。先选 MicroLM 等套件，再查看内部组件、适用节点与 Worker 部署。</p>
      <div className="plugin-filters"><Field label="搜索套件"><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="套件名称或架构" /></Field><Field label="适用节点"><select value={position} onChange={(event) => setPosition(event.target.value)}><option value="">全部节点能力</option>{Object.entries(positions).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></Field></div>
      <div className="plugin-catalog-grid">{visible.map((suite) => <article className="plugin-catalog-card suite-card" key={suite.id}><div className="suite-card-heading"><span className="suite-monogram">{suite.name.slice(0, 1)}</span><span>套件 {suite.version}</span></div><h3><button onClick={() => openSuite(suite)}>{suite.name}</button></h3><p>{suite.description}</p><div className="suite-capability-tags">{suite.components.map((member) => <button key={member.key} onClick={() => openSuite(suite, member.key)}>{positions[member.node_kind] || member.node_kind}</button>)}</div><dl><div><dt>包含组件</dt><dd>{suite.components.length} 项能力</dd></div><div><dt>Worker 部署</dt><dd>{deployments.isSuccess ? `${(deployments.data || []).filter((row) => row.suite_version_id === suite.id && row.ready_count > 0).length} 台已有部分或全部能力` : "读取中"}</dd></div></dl><button className="button secondary" onClick={() => openSuite(suite)}>管理套件</button></article>)}</div>
      {suites.isSuccess && !visible.length && <EmptyState title="没有匹配的插件套件" description="调整筛选条件；目录只展示实际登记的套件。" />}
    </>}
    {tab === "catalog" && selected && <>
      <button className="button ghost" onClick={() => navigate({ suite: "", component: "", plugin: "" })}>← 返回套件目录</button>
      <Panel title={`${selected.name} · ${selected.version}`} description={selected.description}><div className="plugin-detail-content"><div className="plugin-action-row"><Field label="套件版本"><select value={selected.id} onChange={(event) => navigate({ suite: event.target.value, component: "", plugin: "" })}>{(suites.data || []).filter((suite) => suite.suite_id === selected.suite_id).map((suite) => <option value={suite.id} key={suite.id}>{suite.version}</option>)}</select></Field><button className="button primary" disabled={!projectId || !selected.enabled || !workers.isSuccess || !selected.components.some((member) => member.deployment === "installable")} onClick={() => setTarget({ suite: selected })}>安装套件 / 按需安装</button></div>{!projectId && <p>选择操作项目后可以安装和配置组件。</p>}<p>套件版本固定以下组件组合。共享组件在同一 Worker 复用；内置能力依据 Worker 上报，训练还需项目的模型代码和输入配置。</p>
        <div className="suite-components">{selected.components.map((member) => <article key={member.key}><div><span>{positions[member.node_kind] || member.node_kind}</span><h3>{member.label}</h3><small>{member.plugin_id} · {member.version}</small>{member.note && <p>{member.note}</p>}</div><span>{modes[member.deployment]}</span><div className="suite-component-actions"><button className="button secondary" onClick={() => openSuite(selected, member.key)}>组件详情{member.deployment === "installable" ? "与配置" : ""}</button>{destination(member.node_kind, projectId) && <Link className="button ghost" to={destination(member.node_kind, projectId)}>前往配置节点</Link>}</div></article>)}</div>
        <details><summary>固定版本清单</summary><p>清单 SHA-256：{selected.manifest_hash}</p><pre>{JSON.stringify(selected.components.map(({ plugin: _plugin, ...member }) => member), null, 2)}</pre></details>
      </div></Panel>
      {component?.plugin && <Panel title={`套件组件：${component.label}`} description="使用套件固定版本，项目参数属于该组件。"><PluginManager key={`${selected.id}:${component.key}:${projectId}`} projectId={projectId} componentId={component.plugin_version_id} /></Panel>}
    </>}
    {tab === "capabilities" && <Panel title="节点可以使用哪些套件" description="节点是使用位置，套件提供实现，组件承担具体功能。每项均来自套件固定清单。"><div className="suite-capability-list">{Object.entries(positions).map(([kind, label]) => {
      const choices = (suites.data || []).flatMap((suite) => suite.components.filter((member) => member.node_kind === kind).map((member) => ({ suite, member })));
      return choices.length > 0 && <section key={kind}><h3>{label}</h3>{choices.map(({ suite, member }) => <button className="button secondary" key={`${suite.id}:${member.key}`} onClick={() => openSuite(suite, member.key)}>{suite.name} {suite.version} · {member.label}</button>)}</section>;
    })}</div></Panel>}
    {tab === "workers" && <Panel title="套件在 Worker 上的部署" description="以套件为单位查看，展开后检查组件。环境具备不代表模型权重、项目代码或运行参数已准备完成。"><div className="plugin-filters"><Field label="目标 Worker"><select value={workerId} onChange={(event) => setWorkerId(event.target.value)}><option value="">全部 Worker</option>{(workers.data || []).map((worker) => <option value={worker.id} key={worker.id}>{worker.name}</option>)}</select></Field></div>
      <div className="plugin-matrix-scroll"><table className="data-table plugin-matrix"><thead><tr><th>插件套件 / 版本</th>{(workers.data || []).filter((worker) => !workerId || worker.id === workerId).map((worker) => <th key={worker.id}>{worker.name}<br /><StatusBadge status={worker.status} /></th>)}</tr></thead><tbody>{(suites.data || []).map((suite) => <tr key={suite.id}><th><button className="button ghost" onClick={() => openSuite(suite)}>{suite.name}</button><br />{suite.version}</th>{(workers.data || []).filter((worker) => !workerId || worker.id === workerId).map((worker) => {
        const row = deploymentFor(suite, worker.id);
        return <td key={worker.id}><strong>{!row ? "读取中" : row.state === "ready" ? "环境条件已具备" : row.state === "partial" ? "部分部署" : "尚未具备"}</strong><small>{row ? `${row.ready_count} / ${row.total_count} 项环境条件` : ""}</small>{worker.status !== "online" && <small>Worker 离线，展示最后已知状态</small>}{row?.project_configuration_required && <small>训练仍需项目代码配置</small>}<details><summary>展开 {suite.components.length} 个组件</summary>{suite.components.map((member) => {
          const fact = row?.components.find((item) => item.key === member.key);
          return <div className="suite-deployment-component" key={member.key}><span>{member.label}</span><small>{fact ? states[fact.state] : "读取中"}</small>{fact?.last_verified_at && <small>最近验证 {fact.last_verified_at}</small>}{fact?.task_id && <button className="button ghost" onClick={() => setTaskId(fact.task_id!)}>组件安装日志</button>}</div>;
        })}</details><button className="button secondary" disabled={!projectId || !suite.enabled || worker.status !== "online" || !worker.capabilities?.includes("builtin.plugin-manager") || !suite.components.some((member) => member.deployment === "installable")} onClick={() => setTarget({ suite, workerId: worker.id })}>安装 / 补齐组件</button></td>;
      })}</tr>)}</tbody></table></div>{workers.isSuccess && !workers.data.length && <EmptyState title="尚未连接 Worker" description="请先配置执行环境。" />}</Panel>}
    {tab === "operations" && <><p>操作记录按组件保存；新套件安装任务记录精确套件来源，共享组件复用原任务。</p><PluginManager projectId={projectId} operationsOnly /></>}
    {target && <SuiteInstallDialog key={`${target.suite.id}:${projectId}`} suite={target.suite} workers={workers.data || []} deployments={deployments.data || []} initialWorkerId={target.workerId} projectId={projectId} onClose={() => setTarget(null)} onRefresh={refresh} onTask={setTaskId} />}
    {taskId && <TaskDetail taskId={taskId} onClose={() => setTaskId("")} />}
  </div>;
}

function SuiteInstallDialog({ suite, workers, deployments, initialWorkerId, projectId, onClose, onRefresh, onTask }: { suite: PluginSuite; workers: PluginWorker[]; deployments: Deployment[]; initialWorkerId?: string; projectId: string; onClose: () => void; onRefresh: () => void; onTask: (id: string) => void }) {
  const installable = suite.components.filter((member) => member.deployment === "installable");
  const [keys, setKeys] = useState(installable.map((member) => member.key));
  const [workerIds, setWorkerIds] = useState(initialWorkerId ? [initialWorkerId] : []);
  const [results, setResults] = useState<Array<{ workerId: string; key: string; status: string; task_id?: string; error?: { message?: string } }>>([]);
  const submit = useMutation({ mutationFn: async () => {
    const next = await Promise.all(workerIds.map(async (id) => {
      try { const response = await post<{ results: Array<{ key: string; status: string; task_id?: string; error?: { message?: string } }> }>(`/projects/${projectId}/plugin-suites/${suite.id}/installations`, { worker_id: id, component_keys: keys }); return response.results.map((result) => ({ ...result, workerId: id })); }
      catch (error) { return [{ workerId: id, key: "request", status: "failed", error: { message: error instanceof Error ? error.message : "提交失败" } }]; }
    })); setResults(next.flat());
  }, onSettled: onRefresh });
  const change = (values: string[], value: string, checked: boolean) => checked ? [...new Set([...values, value])] : values.filter((item) => item !== value);
  return <Modal title={`安装套件 ${suite.name} · ${suite.version}`} description="选择要部署的组件，已有精确版本自动复用。每项显示真实任务状态。" onClose={onClose} wide><form onSubmit={(event) => { event.preventDefault(); if (keys.length && workerIds.length && !submit.isPending) submit.mutate(); }}>
    <fieldset className="plugin-worker-picker"><legend>套件内可安装组件</legend><button type="button" className="button ghost" disabled={submit.isPending} onClick={() => setKeys(installable.map((member) => member.key))}>选择全部可安装组件</button>{installable.map((member) => <label key={member.key}><input type="checkbox" checked={keys.includes(member.key)} disabled={submit.isPending} onChange={(event) => setKeys(change(keys, member.key, event.target.checked))} /><span>{member.label}<small>{member.plugin_id} · {member.version}</small></span></label>)}</fieldset>
    <p>其余能力随 Worker 提供或依赖项目代码；此操作不下载模型权重，也不自动配置训练。缺少内置能力时请更新对应 Worker 环境。</p>
    <fieldset className="plugin-worker-picker"><legend>目标 Worker</legend>{workers.map((worker) => <label key={worker.id}><input type="checkbox" checked={workerIds.includes(worker.id)} disabled={submit.isPending || worker.status !== "online" || !worker.capabilities?.includes("builtin.plugin-manager")} onChange={(event) => setWorkerIds(change(workerIds, worker.id, event.target.checked))} /><span>{worker.name}<small>{worker.status !== "online" ? "Worker 离线" : !worker.capabilities?.includes("builtin.plugin-manager") ? "缺少插件安装能力" : `${deployments.find((row) => row.suite_version_id === suite.id && row.worker_id === worker.id)?.ready_count || 0} 项环境条件已具备`}</small></span></label>)}</fieldset>
    <div aria-live="polite">{results.map((result) => <div className="suite-install-result" key={`${result.workerId}:${result.key}`}><strong>{workers.find((worker) => worker.id === result.workerId)?.name} · {suite.components.find((member) => member.key === result.key)?.label || "请求"}</strong><span>{result.status === "queued" ? "任务已提交" : result.status === "in_progress" ? "复用进行中的安装任务" : result.status === "installed" ? "已安装，直接复用" : `失败：${result.error?.message || "请重试"}`}</span>{result.task_id && <button type="button" className="button ghost" onClick={() => onTask(result.task_id!)}>查看任务</button>}</div>)}</div>
    <ErrorMessage error={submit.error} /><div className="plugin-action-row"><button type="button" className="button secondary" onClick={onClose}>关闭</button><button type="submit" className="button primary" disabled={!keys.length || !workerIds.length || submit.isPending}>{submit.isPending ? "提交中…" : "提交套件安装"}</button></div>
  </form></Modal>;
}
