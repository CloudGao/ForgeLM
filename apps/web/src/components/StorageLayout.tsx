import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, formatBytes, post } from "../api";
import { ErrorMessage, Field, Panel, StatusBadge } from "./ui";

type Plan = { id: string; status: string; confirmation_phrase: string; summary: { item_count?: number; bytes?: number; blocked?: { source: string; reason: string }[] }; items?: { id: string; path: string; status: string; size_bytes: number; metadata_json: { destination: string }; error?: { message?: string } }[] };
type Layout = { root: string; directories: Record<string, string>; projects: { id: string; name: string; current: string; target: string }[]; plans: Plan[] };

export function StorageLayout() {
  const client = useQueryClient();
  const [planId, setPlanId] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const layout = useQuery({ queryKey: ["storage-layout"], queryFn: () => api<Layout>("/storage/layout"), retry: false });
  const details = useQuery({ queryKey: ["storage-relocation", planId], queryFn: () => api<Plan>(`/storage/cleanup-plans/${planId}`), enabled: !!planId,
    refetchInterval: (query) => query.state.data?.status === "executing" ? 1500 : false });
  const preview = useMutation({ mutationFn: (scope: string) => post<Plan>("/storage/relocations/preview", { scope }),
    onSuccess: (plan) => { setPlanId(plan.id); setConfirmation(""); client.invalidateQueries({ queryKey: ["storage-layout"] }); } });
  const execute = useMutation({ mutationFn: () => post(`/storage/relocations/${planId}/execute`, { confirmation }),
    onSuccess: () => { client.invalidateQueries({ queryKey: ["storage-relocation", planId] }); client.invalidateQueries({ queryKey: ["storage-layout"] }); setConfirmation(""); } });
  const plan = details.data;
  const labels: Record<string, string> = { dataset: "公共数据集", projects: "项目产物与配置", weights: "共享外部权重", platform: "平台基础数据" };
  return <Panel title="目录归属与迁移" description="数据集集中存储，项目通过版本绑定消费。迁移保留资产身份、引用和血缘；暂停任务的恢复目录单独保留。">
    <ErrorMessage error={layout.error || preview.error || details.error || execute.error} />
    {layout.data?.directories && <>
      <dl>{Object.entries(layout.data.directories).map(([key, value]) => <div key={key}><dt>{labels[key] || key}</dt><dd className="storage-path">{value}</dd></div>)}</dl>
      <div className="page-actions">{[["assets", "预览资产迁移"], ["projects", "预览项目目录迁移"], ["platform", "预览平台辅助数据迁移"]].map(([scope, label]) => <button key={scope} className="button secondary" disabled={preview.isPending || plan?.status === "executing"} onClick={() => preview.mutate(scope)}>{label}</button>)}</div>
      <details><summary>项目实际目录</summary>{layout.data.projects.map((p) => <p key={p.id}><strong>{p.name}</strong><br /><span className="storage-path">{p.current || "未配置"} → {p.target}</span></p>)}</details>
      {!!layout.data.plans.length && <Field label="迁移记录"><select aria-label="迁移记录" value={planId} onChange={(e) => { setPlanId(e.target.value); setConfirmation(""); }}><option value="">选择记录</option>{layout.data.plans.map((p) => <option key={p.id} value={p.id}>{p.id} · {p.status}</option>)}</select></Field>}
    </>}
    {plan && <>
      <p><StatusBadge status={plan.status} /> {plan.summary.item_count || 0} 项 · {formatBytes(plan.summary.bytes || 0)}</p>
      {plan.status === "awaiting_cutover" && <p>数据目录已迁移，等待停服切换平台数据库和存储根配置；任务领取与制品写入仍暂停。</p>}
      <div className="data-table-wrap"><table className="data-table"><thead><tr><th>来源 → 目标</th><th>大小</th><th>状态</th></tr></thead><tbody>{plan.items?.map((item) => <tr key={item.id}><td className="storage-path">{item.path}<br />→ {item.metadata_json.destination}</td><td>{formatBytes(item.size_bytes)}</td><td>{item.status}{item.error?.message && <p>{item.error.message}</p>}</td></tr>)}</tbody></table></div>
      {!!plan.summary.blocked?.length && <details><summary>保留原位的 {plan.summary.blocked.length} 项</summary>{plan.summary.blocked.map((item, i) => <p key={i}><span className="storage-path">{item.source}</span><br />{item.reason}</p>)}</details>}
      {["planned", "interrupted"].includes(plan.status) && <><Field label={`输入「${plan.confirmation_phrase}」确认`}><input aria-label="迁移确认" value={confirmation} onChange={(e) => setConfirmation(e.target.value)} /></Field><button className="button primary" disabled={execute.isPending || confirmation !== plan.confirmation_phrase} onClick={() => execute.mutate()}>{plan.status === "interrupted" ? "校验并继续迁移" : "执行已核验的迁移"}</button></>}
    </>}
  </Panel>;
}
