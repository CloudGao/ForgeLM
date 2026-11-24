import { useQuery } from "@tanstack/react-query";
import { Activity, Cpu, HardDrive, ScrollText } from "lucide-react";
import { useEffect, useState } from "react";
import { api, apiHref, formatDate } from "../api";
import type { Json, Task } from "../types";
import {
  LOG_WINDOW_SIZE,
  MAX_LOG_WINDOW_SIZE,
  GenerationThroughput,
  ResourceUsageChart,
  TerminalLogViewer,
  useTaskEventWindow,
} from "./TaskObservability";
import { Alert, ErrorMessage, Modal, Progress, StatusBadge } from "./ui";
import { TaskTiming } from "./TaskTiming";
import { TrainingPerformanceReport } from "./TrainingPerformanceReport";
import { ExecutionProvenance } from "./ExecutionProvenance";
import { TrainingOptimizationComparison } from "./TrainingOptimizationComparison";

const terminalStates = new Set(["succeeded", "failed", "cancelled", "lost"]);

function empty(value?: Json) {
  return !value || Object.keys(value).length === 0;
}

export function TaskDetail({ taskId, initialAttemptId, onClose }: { taskId: string; initialAttemptId?: string; onClose: () => void }) {
  const task = useQuery<Task>({
    queryKey: ["task-detail", taskId],
    queryFn: () => api<Task>(`/tasks/${taskId}?include_events=false`),
    refetchInterval: (query) => terminalStates.has(query.state.data?.observed_state || "") ? false : 2000,
  });
  const detail = task.data;
  const attempts = detail?.attempts || [];
  const [attemptId, setAttemptId] = useState(initialAttemptId || "");
  const [comparisonTask, setComparisonTask] = useState("");
  useEffect(() => {
    if (!attemptId && attempts[0]?.id) setAttemptId(attempts[0].id);
  }, [attemptId, attempts]);
  const attempt = attempts.find((candidate) => candidate.id === attemptId) || attempts[0];
  const active = !terminalStates.has(detail?.observed_state || "");
  const [logLimit, setLogLimit] = useState(LOG_WINDOW_SIZE);
  useEffect(() => setLogLimit(LOG_WINDOW_SIZE), [attempt?.id]);
  const resources = useTaskEventWindow({ taskId, eventGroup: "resources", attemptId: attempt?.id, limit: 240, enabled: Boolean(detail), live: active });
  const logs = useTaskEventWindow({ taskId, eventGroup: "logs", attemptId: attempt?.id, limit: logLimit, enabled: Boolean(detail), live: active });
  const timeline = useTaskEventWindow({ taskId, eventGroup: "timeline", attemptId: attempt?.id, limit: 100, enabled: Boolean(detail), live: active });
  const resourceEvents = resources.data?.items || [];
  const summaryEvent = resourceEvents.filter((event) => event.event_type === "resource.summary").at(-1)?.payload;
  const summary = !empty(attempt?.resource_summary) ? attempt?.resource_summary : summaryEvent;
  const resourceSamples = resourceEvents.filter((event) => event.event_type === "resource.sample");

  return <Modal title={detail?.title || `任务 ${taskId.slice(0, 8)}`} description={`${detail?.kind || "任务"} · ${detail?.adapter_id || "等待适配器"}`} onClose={onClose} wide>
    <div className="training-detail-body task-detail-body">
      {task.isLoading && <p className="quiet-message">正在读取任务详情…</p>}
      <ErrorMessage error={task.error} />
      {detail && <>
        <div className="detail-summary-line">
          <StatusBadge status={detail.observed_state} />
          <span><Activity size={14} />任务 {detail.id.slice(0, 8)}</span>
          <span>{formatDate(detail.started_at || detail.created_at)} → {formatDate(detail.ended_at || undefined)}</span>
        </div>
        <Progress value={detail.progress} />
        <TaskTiming task={detail} detail />
        <GenerationThroughput events={timeline.data?.items || []} />

        {detail.project_id && summary && <div className="detail-summary-line">{[
          ["task_log_artifact_id", "下载本地日志"],
          ["generation_runtime_artifact_id", "下载调优记录"],
          ["generation_metrics_artifact_id", "下载生成时序"],
          ["generation_validation_artifact_id", "下载中文约束对照"],
        ].map(([key, label]) => summary[key] ? <a key={key} href={apiHref(`/projects/${detail.project_id}/artifacts/${summary[key]}/payload`)} download>{label}</a> : null)}</div>}

        {attempts.length > 1 && <div className="attempt-tabs" aria-label="任务尝试">{attempts.map((candidate) => <button type="button" className={candidate.id === attempt?.id ? "active" : ""} key={candidate.id} onClick={() => setAttemptId(candidate.id)}>Attempt {candidate.attempt_number} · {candidate.status}</button>)}</div>}

        <section className="detail-section">
          <header><div><h3>资源使用</h3><p>CPU 配额利用率按分配核数计算；实际使用核数单独显示，完整原始采样保存在运行指标制品中。</p></div><Cpu size={17} /></header>
          <ErrorMessage error={resources.error} />
          {resourceSamples.length > 0 ? <>
            <ResourceUsageChart events={resourceEvents} />
            {Array.isArray(summary?.warnings) && summary.warnings.length > 0 && <Alert severity="warning" title="资源采集存在边界" detail={summary.warnings.join("；")} />}
          </> : !resources.isLoading && <Alert severity="info" title="未采集运行资源" detail={summary ? "该 attempt 只有资源摘要，没有可绘制的低频时序。" : attempt ? "这是历史任务，或该 Worker 尚未支持 attempt 级资源采集。" : "任务还没有被 Worker 领取。"} />}
        </section>

        <section className="detail-section">
          <header><div><h3>运行日志</h3><p>固定高度终端默认跟随最新行；向上滚动会暂停跟随，可按 300 条加载更早记录。</p></div><ScrollText size={17} /></header>
          <ErrorMessage error={logs.error} />
          {logs.data?.items.length ? <TerminalLogViewer key={attempt?.id || taskId} events={logs.data.items} hasMore={logs.data.has_more && logLimit < MAX_LOG_WINDOW_SIZE} loadingOlder={logs.isFetching && logLimit > LOG_WINDOW_SIZE} onLoadOlder={() => setLogLimit((value) => Math.min(MAX_LOG_WINDOW_SIZE, value + LOG_WINDOW_SIZE))} /> : !logs.isLoading && <Alert severity="info" title="未采集平台日志" detail={attempt ? "该 attempt 没有产生 stdout/stderr 或平台日志。" : "任务尚未开始执行。"} />}
        </section>

        <div className="training-detail-columns">
          <section className="detail-section">
            <header><div><h3>Attempt 与环境</h3><p>实际 Worker、资源分配和启动环境快照。</p></div><HardDrive size={17} /></header>
            {attempt ? <><dl className="detail-list"><div><dt>Attempt</dt><dd>{attempt.attempt_number}</dd></div><div><dt>Worker</dt><dd>{attempt.worker_id || "—"}</dd></div><div><dt>状态</dt><dd>{attempt.status}</dd></div><div><dt>退出码</dt><dd>{attempt.exit_code ?? "—"}</dd></div><div><dt>样本数</dt><dd>{summary?.sample_count ?? "—"}</dd></div><div><dt>指标制品</dt><dd>{attempt.runtime_metrics_artifact_id && detail.project_id ? <a href={apiHref(`/projects/${detail.project_id}/artifacts/${attempt.runtime_metrics_artifact_id}/payload`)} download>下载完整时序</a> : "—"}</dd></div></dl><details><summary>环境快照</summary><pre className="json-preview">{JSON.stringify(attempt.environment_snapshot || {}, null, 2)}</pre></details><details><summary>日志摘要</summary><pre className="json-preview">{JSON.stringify(attempt.log_summary || {}, null, 2)}</pre></details></> : <p className="quiet-message">任务还没有执行尝试。</p>}
          </section>
          <section className="detail-section">
            <header><div><h3>任务事件</h3><p>状态、进度、资源、指标和制品事件。</p></div></header>
            <ErrorMessage error={timeline.error} />
            {timeline.data?.items.length ? <div className="event-timeline">{timeline.data.items.slice().reverse().map((event) => <article key={event.id}><i /><div><strong>{event.event_type}</strong><small>#{event.sequence} · {formatDate(event.created_at)}</small>{!empty(event.payload) && <pre>{JSON.stringify(event.payload, null, 2)}</pre>}</div></article>)}</div> : !timeline.isLoading && <p className="quiet-message">该 attempt 没有普通任务事件。</p>}
          </section>
        </div>

        <ExecutionProvenance taskId={taskId} attemptId={attempt?.id} projectId={detail.project_id} />
        <TrainingPerformanceReport report={detail.result?.performance_report} />
        {detail.kind === "training_performance" && detail.observed_state === "succeeded" && detail.result?.performance_report?.recommendation && <a className="button secondary" href={`/projects/${detail.project_id}/training?comparison=${detail.id}`}>采用推荐组合创建训练草稿</a>}
        {detail.kind === "train" && detail.observed_state === "paused" && detail.payload?.stages?.length === 1 && detail.payload.stages[0].type === "pretrain" && detail.payload.training_data_view_inputs?.length > 0 && <TrainingOptimizationComparison task={detail} onOpen={setComparisonTask} />}
        {comparisonTask && <TaskDetail taskId={comparisonTask} onClose={() => setComparisonTask("")} />}
        {(!empty(detail.error) || !empty(detail.result)) && <section className="detail-section"><header><div><h3>结果与错误</h3><p>结构化任务结果或失败原因。</p></div></header>{!empty(detail.error) && <pre className="json-preview error-json">{JSON.stringify(detail.error, null, 2)}</pre>}{!empty(detail.result) && <pre className="json-preview">{JSON.stringify(detail.result, null, 2)}</pre>}</section>}

        <section className="detail-section collapsed-config"><details><summary>查看冻结任务负载</summary><pre className="json-preview">{JSON.stringify(detail.payload || {}, null, 2)}</pre></details></section>
      </>}
    </div>
  </Modal>;
}
