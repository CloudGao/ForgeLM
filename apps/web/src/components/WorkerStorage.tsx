import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, post, formatBytes, formatDate } from "../api";
import { ErrorMessage, Field, Panel, StatusBadge } from "./ui";

export function WorkerStorage() {
  const client = useQueryClient();
  const [projectId, setProjectId] = useState(() => new URLSearchParams(window.location.search).get("project") || "");
  const [workerId, setWorkerId] = useState(() => new URLSearchParams(window.location.search).get("worker") || "");
  const [selected, setSelected] = useState<string[]>([]);
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<any[]>("/projects") });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<any[]>("/workers") });
  const jobs = useQuery({ queryKey: ["worker-storage-jobs"], queryFn: () => api<any[]>("/storage/worker-jobs"), refetchInterval: 5000 });
  const policy = useQuery({ queryKey: ["storage-policy", projectId], enabled: !!projectId,
    queryFn: () => api<any>(`/storage/projects/${projectId}/policy`) });
  const relevant = jobs.data?.filter((j) => j.project_id === projectId && j.payload.resource_request.worker_selector.worker_id === workerId) || [];
  const scan = relevant.find((j) => j.observed_state === "succeeded" && j.result?.storage_action === "scan");
  const busy = relevant.some((j) => ["queued", "assigned", "starting", "running", "stopping"].includes(j.observed_state));
  const run = useMutation({ mutationFn: (action: string) => post(`/storage/projects/${projectId}/worker-jobs`, {
    worker_id: workerId, action, scan_task_id: scan?.id, item_ids: action === "cleanup" ? selected : [],
  }), onSuccess: () => { setSelected([]); client.invalidateQueries({ queryKey: ["worker-storage-jobs"] }); } });
  const save = useMutation({ mutationFn: (changes: Record<string, unknown>) => api(`/storage/projects/${projectId}/policy`, {
    method: "PUT", body: JSON.stringify({ ...policy.data, ...changes }),
  }), onSuccess: () => client.invalidateQueries({ queryKey: ["storage-policy", projectId] }) });
  return <Panel title="Worker 存储与保留策略" description="由 Worker 执行真实盘点与回收任务。清理只接受盘点中的明确候选，执行时重新校验；原始数据、Tokenizer 和 checkpoint 默认保留。">
    <div className="form-grid">
      <Field label="项目"><select aria-label="存储维护项目" value={projectId} onChange={(e) => { setProjectId(e.target.value); setSelected([]); }}><option value="">选择项目</option>{projects.data?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select></Field>
      <Field label="Worker"><select aria-label="存储维护 Worker" value={workerId} onChange={(e) => { setWorkerId(e.target.value); setSelected([]); }}><option value="">选择 Worker</option>{workers.data?.map((w) => <option key={w.id} value={w.id} disabled={!w.capabilities?.includes("builtin.storage-maintenance")}>{w.name}{!w.capabilities?.includes("builtin.storage-maintenance") ? "（需要升级）" : ""}</option>)}</select></Field>
      {policy.data && <>
        <Field label="计算缓存"><select aria-label="计算缓存保留策略" value={policy.data.cache_cleanup} disabled={save.isPending} onChange={(e) => save.mutate({ cache_cleanup: e.target.value })}><option value="auto">自动回收</option><option value="retain">保留</option></select></Field>
        <Field label="中间数据制品"><select aria-label="中间制品保留策略" value={policy.data.intermediate_retention} disabled={save.isPending} onChange={(e) => save.mutate({ intermediate_retention: e.target.value })}><option value="after_consumers">下游完成后回收可恢复副本</option><option value="retain">长期保留</option></select></Field>
        <Field label="回收宽限期（小时）"><input type="number" min="0" max="8760" defaultValue={policy.data.grace_hours} key={projectId + policy.data.grace_hours} onBlur={(e) => { const n = Number(e.target.value); if (n !== policy.data.grace_hours && Number.isInteger(n) && n >= 0 && n <= 8760) save.mutate({ grace_hours: n }); }} /></Field>
      </>}
    </div>
    <div className="form-actions"><button className="button primary" disabled={!projectId || !workerId || busy || run.isPending} onClick={() => run.mutate("scan")}>盘点 Worker</button><button className="button danger" disabled={!selected.length || busy || run.isPending} onClick={() => run.mutate("cleanup")}>回收选中项</button></div>
    {scan && <>
      <p>最近盘点：{formatDate(scan.ended_at)} · 可用 {formatBytes(scan.result.free_bytes)} / {formatBytes(scan.result.total_bytes)}</p>
      <div className="data-table-wrap"><table className="data-table"><thead><tr><th>选择</th><th>类别 / 路径</th><th>占用</th><th>回收依据</th></tr></thead><tbody>{scan.result.items?.map((item: any) => <tr key={item.id || item.path}><td><input type="checkbox" aria-label={`回收 ${item.path}`} disabled={!item.reclaimable} checked={selected.includes(item.id)} onChange={(e) => setSelected(e.target.checked ? [...selected, item.id] : selected.filter((id) => id !== item.id))} /></td><td>{item.category}<br /><span className="storage-path">{item.path}</span></td><td>{formatBytes(item.size_bytes)}</td><td>{item.reason}</td></tr>)}</tbody></table></div>
      <details><summary>工作区目录总占用（包含受保护和未知内容）</summary>{scan.result.categories?.map((c: any) => <p key={c.path}>{c.path} · {c.error || formatBytes(c.size_bytes)}</p>)}</details>
    </>}
    <div>{relevant.slice(0, 5).map((job) => <p key={job.id}>{job.title} · <StatusBadge status={job.observed_state} /> · {job.result?.storage_action === "cleanup" ? `已删除文件 ${formatBytes(job.result.reclaimed_bytes)}；磁盘可用增加 ${formatBytes(Math.max(job.result.disk_free_delta_bytes || 0, 0))}` : job.current_phase}{job.error?.message && ` · ${job.error.message}`}</p>)}</div>
    <ErrorMessage error={projects.error || workers.error || jobs.error || policy.error || run.error || save.error} />
  </Panel>;
}
