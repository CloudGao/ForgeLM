import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowLeft, Database, HardDrive, RefreshCw, ShieldCheck, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, formatBytes, formatDate, post } from "../api";
import { Alert, ErrorMessage, Field, MetricCard, Modal, PageHeader, Panel, Progress, StatusBadge } from "../components/ui";
import type { StorageCleanupPlan, StorageOverview } from "../types";
import { WorkerStorage } from "../components/WorkerStorage";
import { StorageLayout } from "../components/StorageLayout";

const categoryLabels: Record<string, string> = {
  durable_artifact: "托管制品",
  dataset_asset: "数据资产",
  run_state: "普通任务运行",
  training_run: "训练运行",
  scratch: "终态任务缓存",
  training_inputs: "终态训练输入副本",
  cancelled_checkpoint: "取消或丢失训练的恢复点",
  intermediate_checkpoint: "训练中间恢复点",
  orphan_artifact: "已删除模型的遗留制品",
  model_weight: "外部模型权重",
  unknown: "未知内容",
  historical_workspace: "历史项目工作区",
};

function activeOverview(data?: StorageOverview) {
  return [data?.current_scan?.status, ...(data?.cleanup_plans || []).map((item) => item.status)].some((value) => ["queued", "running", "executing"].includes(String(value)));
}

export function StorageCenter() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [confirmPlan, setConfirmPlan] = useState<StorageCleanupPlan | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [projectFilter, setProjectFilter] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [resultPlan, setResultPlan] = useState<StorageCleanupPlan | null>(null);
  const overview = useQuery({
    queryKey: ["storage-overview"],
    queryFn: () => api<StorageOverview>("/storage/overview"),
    refetchInterval: (query) => activeOverview(query.state.data) ? 1000 : 15000,
  });
  const scan = overview.data?.latest_successful_scan;
  const cleanupSupported = overview.data?.cleanup_selection_supported === true;
  const summary = scan?.summary || {};
  const cleanupItems = (scan?.items || []).filter((item) => item.item_type === "cleanup_candidate");
  const candidates = cleanupItems.filter((item) => item.reclaimable);
  const visibleItems = cleanupItems.filter((item) => (!projectFilter || item.project_id === projectFilter) && (!categoryFilter || item.category === categoryFilter));
  const selectable = visibleItems.filter((item) => item.reclaimable);
  const selectedItems = candidates.filter((item) => selectedIds.includes(item.id));
  useEffect(() => { setSelectedIds([]); }, [scan?.id]);
  const resultDetails = useMutation({
    mutationFn: (id: string) => api<StorageCleanupPlan>(`/storage/cleanup-plans/${id}`),
    onSuccess: setResultPlan,
  });
  const categoryTotals: Record<string, number> = (summary.projects || []).reduce((totals: Record<string, number>, project: any) => {
    Object.entries(project.categories || {}).forEach(([category, bytes]) => { totals[category] = (totals[category] || 0) + Number(bytes || 0); });
    return totals;
  }, {});
  const startScan = useMutation({
    mutationFn: () => post("/storage/scans", {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["storage-overview"] }),
  });
  const createPlan = useMutation({
    mutationFn: () => {
      if (!cleanupSupported) throw new Error("后台尚未加载新版数据清理，请等待服务更新后重试。");
      return post<StorageCleanupPlan>("/storage/cleanup-plans", { scan_id: scan?.id, item_ids: selectedItems.map((item) => item.id) });
    },
    onSuccess: (plan) => { setConfirmPlan(plan); setConfirmation(""); queryClient.invalidateQueries({ queryKey: ["storage-overview"] }); },
  });
  const executePlan = useMutation({
    mutationFn: (plan: StorageCleanupPlan) => post<StorageCleanupPlan>(`/storage/cleanup-plans/${plan.id}/execute`, { confirmation }),
    onSuccess: () => { setConfirmPlan(null); setConfirmation(""); setSelectedIds([]); queryClient.invalidateQueries({ queryKey: ["storage-overview"] }); },
  });
  const currentScan = overview.data?.current_scan;
  const scanning = ["queued", "running"].includes(currentScan?.status || "");
  const previewDisabledReason = overview.isPending ? "正在读取存储状态…"
    : overview.isError ? "存储状态读取失败，请稍后重试。"
    : !cleanupSupported ? "后台尚未加载新版数据清理。请在活动任务结束后更新后台，再点击“立即扫描”。"
    : scanning ? "正在扫描，请等待扫描完成。"
    : !scan ? "请先点击右上角“立即扫描”。"
    : !selectable.length ? "当前筛选下没有可清理项；可调整筛选或重新扫描。"
    : !selectedItems.length ? "请先勾选下方可清理项，再生成预览。"
    : createPlan.isPending ? "正在生成清理预览…" : "";

  return (
    <div className="execution-page storage-page">
      <PageHeader
        eyebrow="STORAGE GOVERNANCE"
        title="存储中心"
        description="盘点本机与 Worker 工作区，设置保留策略，按明确候选回收空间。"
        actions={<><button className="button secondary" onClick={() => navigate(-1)}><ArrowLeft size={16} />返回</button><button className="button primary" disabled={scanning || startScan.isPending} onClick={() => startScan.mutate()}><RefreshCw size={16} className={scanning ? "spin-icon" : ""} />{scanning ? "正在扫描" : "立即扫描"}</button></>}
      />

      {!scan && !scanning && <Alert severity="warning" title="还没有空间账本" detail="先执行一次真实扫描，平台才会计算分类占用与确认可回收空间。" />}
      {currentScan?.status === "failed" && <Alert severity="error" title="最近扫描失败" detail={currentScan.error?.message || "请检查项目工作区是否可访问"} />}
      {Number(summary.unscanned_project_count || 0) > 0 && <Alert severity="warning" title={`${summary.unscanned_project_count} 个项目未能扫描`} detail="不可访问的项目不会计入可回收空间，也不会进入清理计划。" />}

      <WorkerStorage />
      <StorageLayout />
      <section className="metrics-grid four storage-metrics">
        <MetricCard label="本机已扫描" value={formatBytes(summary.managed_bytes)} detail={`${summary.project_count || 0} 个项目 · 文件逻辑大小`} tone="cyan" />
        <MetricCard label="确认可回收" value={formatBytes(summary.reclaimable_bytes)} detail={`${summary.candidate_count || 0} 个安全候选`} tone="green" />
        <MetricCard label="控制面卷" value={(summary.volumes || []).length || "—"} detail="卷容量按设备去重" tone="violet" />
        <MetricCard label="最近盘点" value={scan?.ended_at ? formatDate(scan.ended_at) : "—"} detail={scan ? "真实目录扫描" : "等待首次扫描"} />
      </section>

      <div className="content-grid storage-grid">
        <Panel title="卷与保留水位" description="任务可用空间已经扣除 max(卷容量 10%, 50 GiB) 的系统保留量。">
          <div className="storage-volume-list">
            {(summary.volumes || []).map((volume: any) => <article key={volume.id}>
              <header><div><HardDrive size={17} /><strong>{volume.root}</strong></div><StatusBadge status={volume.severity} /></header>
              <Progress value={volume.total_bytes ? volume.used_bytes / volume.total_bytes : 0} />
              <dl><div><dt>剩余</dt><dd>{formatBytes(volume.free_bytes)}</dd></div><div><dt>保留</dt><dd>{formatBytes(volume.reserved_bytes)}</dd></div><div><dt>任务可用</dt><dd>{formatBytes(volume.available_for_tasks_bytes)}</dd></div><div><dt>平台占用</dt><dd>{formatBytes(volume.managed_bytes)}</dd></div></dl>
            </article>)}
            {!(summary.volumes || []).length && <p className="quiet-message">扫描后显示控制面可见卷。</p>}
          </div>
        </Panel>
        <Panel title="分类占用" description="未知内容和正式制品不会进入自动清理。">
          <div className="storage-category-list">
            {(Object.entries(categoryTotals) as Array<[string, number]>).sort(([, left], [, right]) => right - left).map(([category, bytes]) => <div key={category}><span>{categoryLabels[category] || category}</span><strong>{formatBytes(bytes)}</strong></div>)}
            {!Object.keys(categoryTotals).length && <p className="quiet-message">暂无分类数据。</p>}
          </div>
        </Panel>
      </div>

      <Panel title="项目空间" description="项目占用只统计控制面能够真实访问的 .llm-platform 目录。" className="table-panel storage-table-panel">
        <div className="data-table-wrap"><table className="data-table"><thead><tr><th>项目</th><th>工作区</th><th>已管理</th><th>可回收</th><th>状态</th></tr></thead><tbody>
          {(summary.projects || []).map((project: any) => <tr key={project.project_id}><td><strong>{project.project_name}</strong></td><td><span className="storage-path">{project.workspace}</span></td><td>{formatBytes(project.managed_bytes)}</td><td>{formatBytes(project.reclaimable_bytes)}</td><td><StatusBadge status={project.scanned ? "ready" : "failed"} /></td></tr>)}
        </tbody></table></div>
      </Panel>

      <Panel title="数据清理" description="选择本机已核验的缓存、中间恢复点或遗留制品。原始数据、最终恢复点与使用中的文件保留；远端目录使用上方 Worker 存储面板。" actions={<button className="button danger" disabled={!!previewDisabledReason} onClick={() => createPlan.mutate()}><Trash2 size={15} />生成清理预览</button>} className="table-panel storage-table-panel">
        <div className="storage-cleanup-filters">
          <Field label="项目筛选"><select aria-label="清理项目" value={projectFilter} onChange={(event) => { setProjectFilter(event.target.value); setSelectedIds([]); }}><option value="">全部项目</option>{(summary.projects || []).map((project: any) => <option key={project.project_id} value={project.project_id}>{project.project_name}</option>)}</select></Field>
          <Field label="类型筛选"><select aria-label="清理类型" value={categoryFilter} onChange={(event) => { setCategoryFilter(event.target.value); setSelectedIds([]); }}><option value="">全部类型</option>{[...new Set(cleanupItems.map((item) => item.category))].map((category) => <option key={category} value={category}>{categoryLabels[category] || category}</option>)}</select></Field>
          <span role="status">已选 {selectedItems.length} 项 · {formatBytes(selectedItems.reduce((total, item) => total + item.size_bytes, 0))}{previewDisabledReason && <small className="muted">{previewDisabledReason}</small>}</span>
        </div>
        {visibleItems.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th><input type="checkbox" aria-label="选择当前可清理项" disabled={!selectable.length} checked={selectable.length > 0 && selectable.every((item) => selectedIds.includes(item.id))} onChange={(event) => setSelectedIds(event.target.checked ? selectable.map((item) => item.id) : [])} /></th><th>类型 / 项目</th><th>路径</th><th>大小</th><th>状态与依据</th><th>删除影响</th></tr></thead><tbody>{visibleItems.map((item) => <tr key={item.id}>
          <td><input type="checkbox" aria-label={`选择 ${item.path}`} disabled={!item.reclaimable} checked={selectedIds.includes(item.id)} onChange={(event) => setSelectedIds((ids) => event.target.checked ? [...ids, item.id] : ids.filter((id) => id !== item.id))} /></td>
          <td><strong>{categoryLabels[item.category] || item.category}</strong><small className="muted">{item.metadata_json?.project_name || ""}</small><small className="muted">{item.metadata_json?.owner_label || ""}</small></td><td><span className="storage-path">{item.path}</span></td><td>{formatBytes(item.size_bytes)}<small className="muted">{item.file_count} 个文件</small></td>
          <td><strong>{item.reclaimable ? "可清理" : "受保护"}</strong><p>{item.reason}</p></td><td>{item.metadata_json?.impact || "保留任务配置与日志"}</td>
        </tr>)}</tbody></table></div> : <div className="storage-safe-empty"><ShieldCheck size={26} /><div><strong>当前筛选下没有清理项</strong><span>先扫描工作区；未知目录和无法验证的文件不会开放清理。</span></div></div>}
      </Panel>

      {(overview.data?.cleanup_plans || []).length > 0 && <Panel title="最近清理计划" description="回收量统计成功删除的文件字节；硬链接和其他任务写入可能使卷可用空间变化不同。清理后可重新扫描。" className="table-panel storage-table-panel"><div className="data-table-wrap"><table className="data-table"><thead><tr><th>创建时间</th><th>状态</th><th>候选</th><th>预计释放</th><th>实际回收文件</th><th>结果</th></tr></thead><tbody>{overview.data!.cleanup_plans.map((plan) => <tr key={plan.id}><td>{formatDate(plan.created_at)}</td><td><StatusBadge status={plan.status} /></td><td>{plan.summary.item_count || 0}</td><td>{formatBytes(plan.summary.estimated_reclaimable_bytes)}</td><td>{formatBytes(plan.summary.actual_reclaimed_bytes)}</td><td><button className="button secondary small" onClick={() => resultDetails.mutate(plan.id)}>查看结果</button></td></tr>)}</tbody></table></div></Panel>}

      {(summary.workers || []).length > 0 && <Panel title="上次本机扫描记录的 Worker 容量" description="历史容量快照；实时目录盘点和回收请使用上方 Worker 存储面板。"><div className="storage-worker-list">{summary.workers.map((worker: any) => <article key={worker.worker_id}><div><Database size={16} /><strong>{worker.worker_name}</strong><StatusBadge status={worker.status} /></div><span>{worker.root || "未配置工作区"}</span><small>{formatBytes(worker.free_bytes)} 可用 / {formatBytes(worker.total_bytes)} · 历史容量快照</small></article>)}</div></Panel>}
      <ErrorMessage error={overview.error || startScan.error || createPlan.error || executePlan.error || resultDetails.error} />

      {resultPlan && <Modal title="清理执行结果" description="逐项结果来自实际执行记录。" onClose={() => setResultPlan(null)} wide><StatusBadge status={resultPlan.status} /><p>已回收文件 {formatBytes(resultPlan.summary.actual_reclaimed_bytes)}</p><div className="cleanup-plan-items">{resultPlan.items.map((item) => <div key={item.id}><StatusBadge status={item.status} /><span>{item.path}<small>{item.error?.message || item.reason}</small></span><strong>{formatBytes(item.actual_reclaimed_bytes)}</strong></div>)}</div></Modal>}

      {confirmPlan && <Modal title="确认执行存储清理" description="清理计划十分钟内有效；执行时会重新检查任务终态、路径边界和内容指纹。" onClose={() => setConfirmPlan(null)} wide>
        <div className="cleanup-confirmation">
          <Alert severity="warning" title={`将清理 ${confirmPlan.summary.item_count} 项，预计释放 ${formatBytes(confirmPlan.summary.estimated_reclaimable_bytes)}`} detail="删除开始后不可恢复；发生变化的候选会自动跳过。" />
          <div className="cleanup-plan-items">{confirmPlan.items.map((item) => <div key={item.id}><AlertTriangle size={15} /><span>{item.path}<small>{item.metadata_json?.impact || item.reason}</small></span><strong>{formatBytes(item.size_bytes)}</strong></div>)}</div>
          <Field label={`输入“${confirmPlan.confirmation_phrase}”以确认`}><input aria-label="清理确认文字" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></Field>
          <ErrorMessage error={executePlan.error} />
          <div className="form-actions"><button className="button secondary" onClick={() => setConfirmPlan(null)}>取消</button><button className="button danger" disabled={confirmation !== confirmPlan.confirmation_phrase || executePlan.isPending} onClick={() => executePlan.mutate(confirmPlan)}>{executePlan.isPending ? "正在提交…" : "执行精确清理"}</button></div>
        </div>
      </Modal>}
    </div>
  );
}
