import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, post } from "../api";
import { ErrorMessage, Field, Panel, StatusBadge } from "./ui";

export function SuiteTrainingHistory({ projectId, onTask }: { projectId: string; onTask: (id: string) => void }) {
  const [version, setVersion] = useState("");
  const [provider, setProvider] = useState("");
  const [providerVersion, setProviderVersion] = useState("");
  const [offset, setOffset] = useState(0);
  const records = useQuery({ queryKey: ["training-executions", projectId, version, provider, providerVersion, offset], queryFn: () => api<{ items: any[]; has_more: boolean }>(`/projects/${projectId}/training-executions?offset=${offset}${version ? `&component_version=${encodeURIComponent(version)}` : ""}${provider ? `&provider_id=${encodeURIComponent(provider)}` : ""}${providerVersion ? `&provider_version=${encodeURIComponent(providerVersion)}` : ""}`), refetchInterval: 5000 });
  const runs = useQuery({ queryKey: ["suite-runs", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/suite-runs`), refetchInterval: 5000 });
  const control = useMutation({ mutationFn: ({ id, action }: { id: string; action: string }) => post(`/tasks/${id}/control`, { action }), onSuccess: () => { void runs.refetch(); void records.refetch(); } });
  return <Panel title="执行记录与版本追踪" description="从目标套件的“配置与运行”选择任意兼容 Checkpoint 接续。暂停后恢复沿用原配置，换版本或数据会创建新训练。">
    <ErrorMessage error={records.error || runs.error || control.error} />
    {(runs.data || []).filter(r => r.payload?.suite_execution?.training && ["queued", "assigned", "starting", "running", "pausing", "paused", "failed", "lost"].includes(r.observed_state)).map(run => <div key={run.id} className="package-actions"><button className="button ghost" onClick={() => onTask(run.id)}>{run.title} · {run.id.slice(0, 8)}</button><StatusBadge status={run.observed_state} />{["queued", "assigned", "starting", "running"].includes(run.observed_state) && <button className="button secondary" disabled={control.isPending} onClick={() => control.mutate({ id: run.id, action: "pause" })}>保存 Checkpoint 后暂停</button>}{run.observed_state === "paused" && <button className="button secondary" disabled={control.isPending} onClick={() => control.mutate({ id: run.id, action: "resume" })}>按原配置恢复</button>}{["failed", "lost"].includes(run.observed_state) && <button className="button secondary" disabled={control.isPending} onClick={() => control.mutate({ id: run.id, action: "retry" })}>按原配置重试</button>}</div>)}
    <Field label="按组件实现版本查找"><input value={version} placeholder="例如 1.1.0" onChange={e => { setVersion(e.target.value); setOffset(0); }} /></Field>
    <Field label="按实际算子查找"><input value={provider} placeholder="例如 flash_attn" onChange={e => { setProvider(e.target.value); setOffset(0); }} /></Field>
    <Field label="算子版本"><input value={providerVersion} placeholder="例如 2.8.3" onChange={e => { setProviderVersion(e.target.value); setOffset(0); }} /></Field>
    {records.data?.items?.map(row => <div key={row.id}><button className="button ghost" onClick={() => onTask(row.task_id)}>{row.title} · Attempt {row.attempt_number}</button><StatusBadge status={row.status} /><span>{row.suite ? `${row.suite.suite_id} ${row.suite.version} / ${row.component?.id} ${row.component?.implementationVersion}` : "原生执行"}</span></div>)}
    <div className="package-actions"><button className="button ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>上一页</button><button className="button ghost" disabled={!records.data?.has_more} onClick={() => setOffset(offset + 50)}>下一页</button></div>
  </Panel>;
}
