import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, formatBytes, post } from "../api";
import type { ModelVersion } from "../types";
import { ErrorMessage, Field, Modal } from "./ui";

export type ModelDeletionMode = "normal" | "cascade";
type DeletionPlan = {
  id: string; status: string; confirmation_phrase: string; error?: { message?: string };
  summary: { blockers: string[]; estimated_reclaimable_bytes: number; actual_reclaimed_bytes: number; protected_count?: number };
  items: Array<{ id: string; path: string; category: string; status: string; reason: string; size_bytes: number;
    actual_reclaimed_bytes: number; error?: { message?: string }; metadata_json: { worker_name: string } }>;
};

export function ModelDeletionMenu({ onSelect }: { onSelect: (mode: ModelDeletionMode) => void }) {
  return <details className="model-deletion-menu">
    <summary className="button ghost small">更多</summary>
    <div>
      <button className="button ghost small" onClick={() => onSelect("normal")}>删除</button>
      <button className="button ghost small" onClick={() => onSelect("cascade")}>级联删除</button>
    </div>
  </details>;
}

export function ModelDeletionDialog({ model, mode, onClose }: {
  model: ModelVersion; mode: ModelDeletionMode; onClose: () => void;
}) {
  const client = useQueryClient();
  const [confirmation, setConfirmation] = useState("");
  const [instance] = useState(() => Date.now());
  const title = mode === "cascade" ? "级联删除" : "删除";
  const preview = useQuery({
    queryKey: ["model-deletion-preview", model.id, mode, instance],
    queryFn: () => post<DeletionPlan>(`/models/${model.id}/deletion-plans`, { mode }),
    retry: false, staleTime: Infinity, gcTime: 0, refetchOnWindowFocus: false,
  });
  const key = ["model-deletion-plan", model.id, preview.data?.id];
  const planQuery = useQuery({
    queryKey: key,
    queryFn: () => api<DeletionPlan>(`/models/${model.id}/deletion-plans/${preview.data!.id}`),
    enabled: Boolean(preview.data),
    initialData: preview.data,
    refetchInterval: query => ["executing", "scanning"].includes(query.state.data?.status || "") ? 1000 : false,
  });
  const plan = planQuery.data || preview.data;
  const execute = useMutation({
    mutationFn: () => post<DeletionPlan>(`/models/${model.id}/deletion-plans/${plan!.id}/execute`, { confirmation }),
    onSuccess: result => {
      client.setQueryData(key, result);
      client.invalidateQueries({ queryKey: ["models", model.project_id] });
    },
  });
  const finished = plan && ["succeeded", "partial", "failed"].includes(plan.status);
  const blocked = Boolean(plan?.summary.blockers.length);
  return <Modal title={`${title} ${model.name} V${model.version}`} onClose={onClose} wide
    description={mode === "cascade"
      ? "删除版本及其独占模型文件，并回收不再被使用的训练恢复状态和派生产物。"
      : "删除模型版本及其独占模型文件，保留关联训练运行、Checkpoint 和数据。"}>
    <div className="model-deletion-content">
      <p>原始数据、共享文件和活动任务依赖受保护。父子模型、训练指标、评测和血缘记录保留。</p>
      <ErrorMessage error={preview.error || planQuery.error || execute.error} />
      {preview.isFetching && <p role="status">正在核对文件位置和使用者…</p>}
      {plan && <>
        <p role="status">{finished ? `已回收 ${formatBytes(plan.summary.actual_reclaimed_bytes)}`
          : plan.status === "executing" ? "正在删除并回收文件，可以关闭窗口，任务会继续执行。"
          : plan.status === "scanning" ? "正在等待 Worker 核对文件，盘点完成后可确认删除。"
          : `预计回收 ${formatBytes(plan.summary.estimated_reclaimable_bytes)}`}</p>
        {plan.status === "partial" && <p role="alert">版本已删除，部分文件未能回收。请查看原因并重新预览余留文件。</p>}
        {plan.status === "expired" && <p role="alert">预览已过期，请重新预览。</p>}
        {plan.error?.message && <p role="alert">{plan.error.message}</p>}
        {blocked && <div role="alert"><strong>暂时不能删除</strong><ul>{plan.summary.blockers.map(reason => <li key={reason}>{reason}</li>)}</ul></div>}
        <div className="data-table-wrap"><table className="data-table"><thead><tr><th>处理</th><th>位置 / 产物</th><th>占用</th><th>说明</th></tr></thead>
          <tbody>{plan.items.map(item => <tr key={item.id}>
            <td>{{ planned: "将删除", protected: "保留", deleted: "已删除", missing: "文件已不存在", failed: "未完成", inspecting: "盘点中", queued: "等待回收" }[item.status] || item.status}</td>
            <td><strong>{item.metadata_json.worker_name}</strong><div className="model-deletion-path">{item.path}</div></td>
            <td>{formatBytes(item.size_bytes)}</td><td>{item.error?.message || item.reason}</td>
          </tr>)}</tbody></table></div>
        {!plan.items.length && <p>该版本没有可接管的本地文件；本次仅删除版本登记，保留历史记录。</p>}
        {plan.status === "planned" && !blocked && <Field label={`输入“${plan.confirmation_phrase}”确认`}>
          <input value={confirmation} onChange={event => setConfirmation(event.target.value)} autoComplete="off" />
        </Field>}
      </>}
      <div className="form-actions">
        <button className="button secondary" onClick={onClose}>{finished ? "完成" : "关闭"}</button>
        {plan?.status !== "executing" && <button className="button secondary" disabled={preview.isFetching || execute.isPending}
          onClick={() => { setConfirmation(""); execute.reset(); preview.refetch(); }}>重新预览</button>}
        {!finished && <button className="button danger" disabled={!plan || blocked || plan.status !== "planned"
          || confirmation !== plan.confirmation_phrase || execute.isPending || preview.isFetching}
          onClick={() => execute.mutate()}>{execute.isPending ? "正在提交…" : `确认${title}`}</button>}
      </div>
    </div>
  </Modal>;
}
