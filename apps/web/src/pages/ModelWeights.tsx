import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, PackageOpen, Plus, Trash2, Unplug } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { api, formatBytes, formatDate, post } from "../api";
import { EmptyState, ErrorMessage, Field, FormActions, Modal, PageHeader, Panel, Progress, StatusBadge } from "../components/ui";
import type { Project } from "../types";

const terminal = new Set(["succeeded", "failed", "cancelled", "lost"]);
const activeDownloadStates = new Set(["queued", "assigned", "starting", "running", "paused"]);

function WeightDownloadStatus({ item }: { item: any }) {
  const active = activeDownloadStates.has(item.state);
  const task = useTask(active ? item.task_id : null);
  const state = task.data?.observed_state || item.state;
  const status = state === "running"
    ? task.data?.current_phase === "verifying" ? "verifying" : "downloading"
    : state;
  const progress = task.data?.progress;
  const hasProgress = typeof progress === "number" && Number.isFinite(progress);
  return <div>
    <StatusBadge status={status} />
    {active && !terminal.has(state) && (hasProgress ? <>
      <span>总进度 {Math.round(Math.max(0, Math.min(1, progress)) * 100)}%</span>
      <Progress value={progress} />
    </> : <span>{task.isError ? "进度暂不可用" : "等待进度上报"}</span>)}
  </div>;
}

function useTask(taskId: string | null) {
  return useQuery({
    queryKey: ["task", taskId],
    queryFn: () => api<any>(`/tasks/${taskId}`),
    enabled: Boolean(taskId),
    refetchInterval: (query) => terminal.has(query.state.data?.observed_state) ? false : 1000,
  });
}

function recommendedFiles(files: any[]) {
  const selected = files.map((item) => String(item.path || "")).filter((path) => {
    const name = path.split("/").pop()?.toLowerCase() || "";
    return [".safetensors", ".bin", ".model", ".tiktoken", ".txt"].some((suffix) => name.endsWith(suffix))
      || name.endsWith(".json")
      || name === "chat_template.jinja";
  });
  return selected.length ? selected : files.map((item) => String(item.path || "")).filter(Boolean);
}

export function ModelWeights() {
  const queryClient = useQueryClient();
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [scanTaskId, setScanTaskId] = useState<string | null>(null);
  const [planTaskId, setPlanTaskId] = useState<string | null>(null);
  const [draft, setDraft] = useState({ name: "", repo_id: "", revision: "main", hf_endpoint: "", parameter_count: 0 });
  const [useWeight, setUseWeight] = useState<any>(null);
  const weights = useQuery({ queryKey: ["model-weights"], queryFn: () => api<any>("/model-weights"), refetchInterval: 3000 });
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
  const scanTask = useTask(scanTaskId);
  const planTask = useTask(planTaskId);
  const files = scanTask.data?.result?.files || [];
  const selectedPaths = useMemo(() => recommendedFiles(files), [files]);

  const scan = useMutation({
    mutationFn: () => post<any>("/model-weights/huggingface/scan", { repo_id: draft.repo_id, revision: draft.revision, hf_endpoint: draft.hf_endpoint || null, resource_request: { gpus: 0 } }),
    onSuccess: (task) => { setScanTaskId(task.id); setPlanTaskId(null); },
  });
  const plan = useMutation({
    mutationFn: () => post<any>("/model-weights/huggingface/plans", { scan_task_id: scanTaskId, selected_paths: selectedPaths }),
    onSuccess: (task) => setPlanTaskId(task.id),
  });
  const create = useMutation({
    mutationFn: () => post<any>("/model-weights", { name: draft.name || draft.repo_id.split("/").pop(), plan_task_id: planTaskId, parameter_count: draft.parameter_count }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["model-weights"] }); setDownloadOpen(false); setScanTaskId(null); setPlanTaskId(null); },
  });
  const use = useMutation({
    mutationFn: (data: any) => post(`/model-weights/${useWeight.id}/project-models`, data),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["model-weights"] }); setUseWeight(null); },
  });
  const redownload = useMutation({ mutationFn: (id: string) => post(`/model-weights/${id}/download`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-weights"] }) });
  const evict = useMutation({ mutationFn: ({ versionId, locationId }: any) => api(`/model-weights/${versionId}/locations/${locationId}`, { method: "DELETE" }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-weights"] }) });
  const remove = useMutation({ mutationFn: (id: string) => api(`/model-weights/${id}`, { method: "DELETE" }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["model-weights"] }) });
  const items = weights.data?.items || [];

  function submitUse(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    use.mutate({ project_id: form.get("project_id"), name: String(form.get("name") || "").trim() || null, is_candidate: true });
  }

  return <main className="system-page model-weight-page">
    <PageHeader eyebrow="GLOBAL MODEL ASSETS" title="模型权重" description="统一下载、复用和清理 Worker 上的不可变外部模型权重；项目只保存引用。" actions={<button className="button primary" onClick={() => setDownloadOpen(true)}><Plus size={16} />下载权重</button>} />
    <section className="metrics-grid four">
      <article className="metric-card"><span>权重版本</span><strong>{items.length}</strong></article>
      <article className="metric-card"><span>可用版本</span><strong>{items.filter((item: any) => item.state === "ready" && item.locations?.length).length}</strong></article>
      <article className="metric-card"><span>Worker 副本</span><strong>{items.reduce((sum: number, item: any) => sum + (item.locations?.length || 0), 0)}</strong></article>
      <article className="metric-card"><span>项目引用</span><strong>{items.reduce((sum: number, item: any) => sum + Number(item.project_model_count || 0), 0)}</strong></article>
    </section>
    <Panel title="全局权重库" description="清理副本只释放 Worker 磁盘；删除登记需要先移除全部项目引用与物理副本。">
      {items.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>模型 / 版本</th><th>来源</th><th>状态</th><th>大小</th><th>位置</th><th>项目</th><th>操作</th></tr></thead><tbody>{items.map((item: any) => <tr key={item.id}>
        <td><strong>{item.asset?.name || "未命名权重"}</strong><span>V{item.version} · {item.identity_hash?.slice(0, 12)}</span></td>
        <td><strong>{item.asset?.repository_id}</strong><span>{item.resolved_commit?.slice(0, 12)} · {formatDate(item.created_at)}</span></td>
        <td><WeightDownloadStatus item={item} /></td>
        <td>{formatBytes(item.artifact?.size_bytes)}</td>
        <td>{item.locations?.length ? item.locations.map((location: any) => <div className="weight-location" key={location.id}><span>{location.worker_id?.slice(0, 8) || "共享"}</span><button className="icon-button danger-icon" title="清理此副本" disabled={evict.isPending || location.availability_state === "deleting"} onClick={() => confirm("确认从该 Worker 清理物理副本？") && evict.mutate({ versionId: item.id, locationId: location.id })}><Unplug size={14} /></button></div>) : "无副本"}</td>
        <td>{item.project_model_count || 0}</td>
        <td><div className="weight-actions">{item.state === "ready" && item.locations?.length ? <button className="button small" onClick={() => setUseWeight(item)}>加入项目</button> : <button className="button small secondary" disabled={redownload.isPending || activeDownloadStates.has(item.state)} onClick={() => redownload.mutate(item.id)}><Download size={14} />下载</button>}<button className="icon-button danger-icon" title="删除权重登记" disabled={remove.isPending} onClick={() => confirm("确认删除权重登记？必须已无项目引用和物理副本。") && remove.mutate(item.id)}><Trash2 size={15} /></button></div></td>
      </tr>)}</tbody></table></div> : <EmptyState title="还没有模型权重" description="从 Hugging Face 解析并选择一个不可变模型快照；下载由选定 Worker 执行。" action={<button className="button primary" onClick={() => setDownloadOpen(true)}><PackageOpen size={16} />添加首个权重</button>} />}
    </Panel>
    {downloadOpen && <Modal title="下载 Hugging Face 模型" description="先解析仓库并冻结文件列表，再在同一 Worker 预检磁盘与缓存，最后创建全局版本。" onClose={() => setDownloadOpen(false)} wide>
      <div className="form-grid two-columns">
        <Field label="显示名称"><input value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} placeholder="例如 Qwen3.5 0.8B" /></Field>
        <Field label="Repo ID"><input value={draft.repo_id} onChange={(event) => setDraft({ ...draft, repo_id: event.target.value })} placeholder="Qwen/Qwen3.5-0.8B" /></Field>
        <Field label="Revision"><input value={draft.revision} onChange={(event) => setDraft({ ...draft, revision: event.target.value })} /></Field>
        <Field label="HF Endpoint（可选）"><input value={draft.hf_endpoint} onChange={(event) => setDraft({ ...draft, hf_endpoint: event.target.value })} placeholder="https://huggingface.co" /></Field>
        <Field label="参数量（可选）"><input type="number" min="0" value={draft.parameter_count} onChange={(event) => setDraft({ ...draft, parameter_count: Number(event.target.value) })} /></Field>
      </div>
      <div className="weight-wizard-actions"><button className="button secondary" disabled={!draft.repo_id.includes("/") || scan.isPending} onClick={() => scan.mutate()}>{scan.isPending ? "正在提交…" : "1. 解析仓库"}</button><button className="button secondary" disabled={scanTask.data?.observed_state !== "succeeded" || !selectedPaths.length || plan.isPending} onClick={() => plan.mutate()}>{plan.isPending ? "正在提交…" : `2. 预检 ${selectedPaths.length} 个文件`}</button><button className="button primary" disabled={planTask.data?.observed_state !== "succeeded" || !planTask.data?.result?.disk_space_sufficient || create.isPending} onClick={() => create.mutate()}>{create.isPending ? "正在创建…" : "3. 创建并下载"}</button></div>
      {scanTaskId && <section className="weight-task-state"><StatusBadge status={scanTask.data?.observed_state} /><span>仓库解析 · {files.length || 0} 文件 · {formatBytes(scanTask.data?.result?.total_size_bytes)}</span>{scanTask.data && !terminal.has(scanTask.data.observed_state) && <Progress value={scanTask.data.progress || 0} />}</section>}
      {planTaskId && <section className="weight-task-state"><StatusBadge status={planTask.data?.observed_state} /><span>下载预检 · 需下载 {formatBytes(planTask.data?.result?.download_bytes)} / 已缓存 {formatBytes(planTask.data?.result?.cached_bytes)}</span>{planTask.data && !terminal.has(planTask.data.observed_state) && <Progress value={planTask.data.progress || 0} />}</section>}
      <ErrorMessage error={scan.error || plan.error || create.error} />
    </Modal>}
    {useWeight && <Modal title="加入项目" description={`${useWeight.asset?.name} V${useWeight.version} 将作为项目模型引用，不复制权重。`} onClose={() => setUseWeight(null)}><form className="form-grid" onSubmit={submitUse}><Field label="项目"><select name="project_id" required>{(projects.data || []).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></Field><Field label="项目内模型名称"><input name="name" defaultValue={useWeight.asset?.name} /></Field><ErrorMessage error={use.error} /><FormActions submitting={use.isPending} onCancel={() => setUseWeight(null)} submitLabel="加入项目" /></form></Modal>}
  </main>;
}
