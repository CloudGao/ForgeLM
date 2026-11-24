import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { ErrorMessage, Field } from "./ui";

type Snapshot = { id: string; attempt_id: string; phase: string; digest: string; snapshot: Record<string, any> };

export function configurationDiff(left: any, right: any, path = ""): { path: string; before: any; after: any }[] {
  if (JSON.stringify(left) === JSON.stringify(right)) return [];
  if (left && right && typeof left === "object" && typeof right === "object" && !Array.isArray(left) && !Array.isArray(right)) {
    return [...new Set([...Object.keys(left), ...Object.keys(right)])].sort().flatMap(key => configurationDiff(left[key], right[key], path ? `${path}.${key}` : key));
  }
  return [{ path, before: left, after: right }];
}

export function ExecutionProvenance({ taskId, attemptId, projectId }: { taskId: string; attemptId?: string; projectId?: string | null }) {
  const records = useQuery({ queryKey: ["execution-snapshots", taskId], queryFn: () => api<Snapshot[]>(`/tasks/${taskId}/execution-snapshots`), refetchInterval: 5000 });
  const [compare, setCompare] = useState("");
  const [otherTask, setOtherTask] = useState("");
  const history = useQuery({ queryKey: ["training-executions-for-comparison", projectId], queryFn: () => api<{ items: { task_id: string; title: string }[] }>(`/projects/${projectId}/training-executions?limit=200`), enabled: Boolean(projectId) });
  const external = useQuery({ queryKey: ["execution-snapshots", otherTask], queryFn: () => api<Snapshot[]>(`/tasks/${otherTask}/execution-snapshots`), enabled: Boolean(otherTask) });
  const current = records.data?.filter(row => row.attempt_id === attemptId) || [];
  const intent = current.find(row => row.phase === "intent");
  const resolved = current.find(row => row.phase === "resolved");
  const suite = intent?.snapshot.payload?.suite_execution;
  const others = (otherTask ? external.data : records.data)?.filter(row => row.phase === "intent" && row.attempt_id !== attemptId) || [];
  const previous = others.find(row => row.id === compare);
  const changes = previous && intent ? configurationDiff(previous.snapshot.payload, intent.snapshot.payload) : [];
  return <section className="detail-section"><header><div><h3>版本、配置与血缘</h3><p>每次尝试分别保存执行意图、实际回执和完成结果；历史记录不会被后一次替换。</p></div></header>
    <ErrorMessage error={records.error} />
    <ErrorMessage error={external.error || history.error} />
    {!intent && !records.isLoading && <p>此历史尝试没有完整执行快照。</p>}
    {intent && <>
      <p>{suite ? `${suite.suite_id} ${suite.version} · ${suite.component_id} ${intent.snapshot.component?.implementationVersion || ""}` : `原生入口 · ${intent.snapshot.adapter_id}`}</p>
      {suite?.continuation && <p>从任务 {suite.continuation.source_task_id} 的 Checkpoint {suite.continuation.checkpoint_id} 接续 · {suite.continuation.mode}</p>}
      <p>{resolved ? "实际运行回执已保存" : "尚无实际运行回执"}</p>
      {current.filter(row => row.phase.startsWith("interval:")).map(row => <div key={row.id}><p>已提交 Step ({row.snapshot.start_step}, {row.snapshot.end_step}] · 实际调用 {JSON.stringify(row.snapshot.dispatch)}</p>
      </div>)}
      {resolved?.snapshot.backends?.map((backend: any) => <p key={backend.extension}>{backend.extension}：{backend.id} · {backend.distribution} {backend.version} · {backend.dispatch} · 验证 {backend.verification?.status || "未报告"}</p>)}
      {resolved?.snapshot.comparison && <><p>实测范围：优化器更新（不含前后向和数据加载），选中 {resolved.snapshot.comparison.selected}。</p><table><thead><tr><th>实现</th><th>每次更新</th><th>验证 / 不可用原因</th></tr></thead><tbody>{resolved.snapshot.comparison.candidates.map((candidate: any) => <tr key={candidate.id}><td>{candidate.id}</td><td>{candidate.seconds_per_update === undefined ? "—" : `${(candidate.seconds_per_update * 1000).toFixed(3)} ms`}</td><td>{candidate.reason || candidate.verification?.status}</td></tr>)}</tbody></table></>}
      {projectId && <Field label="选择其他训练进行比较"><select value={otherTask} onChange={e => { setOtherTask(e.target.value); setCompare(""); }}><option value="">当前任务的其他尝试</option>{[...new Map((history.data?.items || []).filter(row => row.task_id !== taskId).map(row => [row.task_id, row])).values()].map(row => <option key={row.task_id} value={row.task_id}>{row.title} · {row.task_id.slice(0, 8)}</option>)}</select></Field>}
      {others.length > 0 && <><Field label="与此前尝试比较配置"><select value={compare} onChange={e => setCompare(e.target.value)}><option value="">选择尝试</option>{others.map(row => <option key={row.id} value={row.id}>{row.attempt_id}</option>)}</select></Field>{previous && (changes.length ? <table><thead><tr><th>配置项</th><th>此前</th><th>本次</th></tr></thead><tbody>{changes.map(change => <tr key={change.path}><td>{change.path}</td><td><code>{JSON.stringify(change.before) ?? "未设置"}</code></td><td><code>{JSON.stringify(change.after) ?? "未设置"}</code></td></tr>)}</tbody></table> : <p>冻结配置一致。</p>)}</>}
      {current.map(row => <details key={row.id}><summary>{({ intent: "执行意图", resolved: "实际回执", completed: "完成记录" } as Record<string, string>)[row.phase] || row.phase} · SHA-256 {row.digest.slice(0, 12)}</summary><pre className="json-preview">{JSON.stringify(row.snapshot, null, 2)}</pre></details>)}
    </>}
  </section>;
}
