import { useQuery } from "@tanstack/react-query";
import { Activity, Clock3, Cpu, GitFork, Package, ScrollText, Tag } from "lucide-react";
import { useEffect, useState } from "react";
import { api, formatDate, formatNumber } from "../api";
import type { Json, ModelRegistry, Task, TrainingRun } from "../types";
import {
  LOG_WINDOW_SIZE,
  MAX_LOG_WINDOW_SIZE,
  ResourceUsageChart,
  TerminalLogViewer,
  useTaskEventWindow,
} from "./TaskObservability";
import { TaskTiming } from "./TaskTiming";
import { Alert, ErrorMessage, Modal, Progress, StatusBadge } from "./ui";

const terminalStates = new Set(["succeeded", "failed", "cancelled", "lost"]);

const metricLabels: Record<string, string> = {
  cross_entropy: "Cross entropy",
  grad_norm: "Gradient norm",
  gpu_memory_allocated: "GPU 已分配显存",
  gpu_memory_reserved: "GPU 预留显存",
  learning_rate: "Learning rate",
  loss: "Loss",
  step: "Step",
  step_time: "Step time",
  tokens: "Tokens",
  tokens_per_second: "Tokens/s",
  z_loss: "Z loss",
};

function isEmpty(value?: Json) {
  return !value || Object.keys(value).length === 0;
}

function formatMetric(key: string, value: unknown) {
  if (typeof value !== "number") return String(value ?? "—");
  if (key.startsWith("gpu_memory_")) return `${formatNumber(value / 1024 / 1024)} MiB`;
  if (key === "tokens" || key === "step") return formatNumber(value);
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value);
}

export function TrainingRunDetail({
  run,
  registry,
  registryLoading,
  registryError,
  initialAttemptId,
  onClose,
  onOpenModel,
  onOpenCheckpoint,
}: {
  run: TrainingRun;
  registry?: ModelRegistry;
  registryLoading?: boolean;
  registryError?: unknown;
  initialAttemptId?: string;
  onClose: () => void;
  onOpenModel: (modelId: string) => void;
  onOpenCheckpoint: (checkpointId: string) => void;
}) {
  const task = useQuery<Task>({
    queryKey: ["task-detail", run.task_id],
    queryFn: () => api<Task>(`/tasks/${run.task_id}?include_events=false`),
    enabled: Boolean(run.task_id),
    refetchInterval: (query) => terminalStates.has(query.state.data?.observed_state || run.status) ? false : 2000,
  });
  const detail = task.data;
  const status = detail?.observed_state || run.status;
  const active = !terminalStates.has(status);
  const [selectedAttemptId, setSelectedAttemptId] = useState(initialAttemptId || "");
  useEffect(() => setSelectedAttemptId(initialAttemptId || ""), [run.id, initialAttemptId]);
  const attempt = selectedAttemptId ? detail?.attempts?.find((item) => item.id === selectedAttemptId) : detail?.attempts?.[0];
  const attemptActive = Boolean(attempt && !terminalStates.has(attempt.status));
  const [logLimit, setLogLimit] = useState(LOG_WINDOW_SIZE);
  useEffect(() => setLogLimit(LOG_WINDOW_SIZE), [attempt?.id]);
  const resources = useTaskEventWindow({ taskId: run.task_id, eventGroup: "resources", attemptId: attempt?.id, limit: 240, enabled: Boolean(attempt), live: attemptActive });
  const logs = useTaskEventWindow({ taskId: run.task_id, eventGroup: "logs", attemptId: attempt?.id, limit: logLimit, enabled: Boolean(attempt), live: attemptActive });
  const timeline = useTaskEventWindow({ taskId: run.task_id, eventGroup: "timeline", attemptId: attempt?.id, limit: 100, enabled: Boolean(attempt), live: attemptActive });
  const metrics = run.latest_metrics || {};
  const frozenSchedules = detail?.payload?.learning_rate_schedules ?? run.config_snapshot.learning_rate_schedules;
  const schedules = Array.isArray(frozenSchedules) ? frozenSchedules as Json[] : [];
  const metricEntries = Object.entries(metrics).filter(([key, value]) => key !== "history" && ["string", "number", "boolean"].includes(typeof value));
  const history = Array.isArray(metrics.history) ? metrics.history.slice(-30).reverse() as Json[] : [];
  const resourceEvents = resources.data?.items || [];
  const resourceSamples = resourceEvents.filter((event) => event.event_type === "resource.sample");
  const imported = Boolean(detail?.payload?.history_import);
  const loss = metrics.loss ?? metrics.cross_entropy;
  const outputCheckpoints = (registry?.checkpoints || []).filter((checkpoint) => checkpoint.training_run_id === run.id);
  const outputCheckpointIds = new Set(outputCheckpoints.map((checkpoint) => checkpoint.id));
  const outputModels = (registry?.models || []).filter((model) =>
    Boolean(model.source_checkpoint_id && outputCheckpointIds.has(model.source_checkpoint_id))
    || model.source_training_run?.id === run.id,
  );

  return <Modal title={`训练运行 ${run.id.slice(0, 8)}`} description={run.current_stage || detail?.current_phase || "等待阶段信息"} onClose={onClose} wide>
    <div className="training-detail-body">
      <div className="detail-summary-line">
        <StatusBadge status={status} />
        <span><Activity size={14} />任务 {run.task_id?.slice(0, 8) || "—"}</span>
        {detail?.payload?.training_run_id === run.id && <span>目录 .llm-platform/training-runs/{run.id}</span>}
        <span><Clock3 size={14} />{formatDate(run.started_at || run.created_at)}</span>
      </div>
      <Progress value={detail?.progress ?? (status === "succeeded" ? 1 : 0)} />
      {detail && <TaskTiming task={detail} detail />}

      <section className="detail-section">
        <header><div><h3>当前指标</h3><p>来自整个训练运行的最新累计快照；下方资源、日志和事件可按执行尝试分别查看。</p></div></header>
        <div className="run-detail-metrics">
          <div><span>Step</span><strong>{formatNumber(run.current_step)}</strong></div>
          <div><span>Tokens</span><strong>{formatNumber(run.current_tokens)}</strong></div>
          <div><span>Loss</span><strong>{loss == null ? "—" : formatMetric("loss", loss)}</strong></div>
          <div><span>Tokens/s</span><strong>{formatMetric("tokens_per_second", metrics.tokens_per_second)}</strong></div>
        </div>
        {metricEntries.length > 0 && <dl className="detail-list metric-detail-list">
          {metricEntries.map(([key, value]) => <div key={key}><dt>{metricLabels[key] || key}</dt><dd>{formatMetric(key, value)}</dd></div>)}
        </dl>}
        {history.length > 0 && <div className="metric-history-wrap"><h4>最近指标历史</h4><table className="metric-history"><thead><tr><th>Step</th><th>Tokens</th><th>Loss</th><th>Tokens/s</th></tr></thead><tbody>{history.map((row, index) => <tr key={`${row.step || index}-${index}`}><td>{formatNumber(row.step)}</td><td>{formatNumber(row.tokens)}</td><td>{formatMetric("loss", row.loss ?? row.cross_entropy)}</td><td>{formatMetric("tokens_per_second", row.tokens_per_second)}</td></tr>)}</tbody></table></div>}
      </section>

      {schedules.length > 0 && <section className="detail-section">
        <header><h3>学习率预热</h3></header>
        <dl className="detail-list">{schedules.map((schedule, index) => {
          const resolvedSteps = schedule.resolved_warmup_steps ?? (schedules.length === 1 ? metrics.warmup_steps : null);
          const totalSteps = schedule.total_steps ?? (schedules.length === 1 ? metrics.optimizer_steps : null);
          return <div key={index}><dt>{String(schedule.stage_name || `阶段 ${index + 1}`)}</dt><dd>
            比例 {formatMetric("warmup_ratio", Number(schedule.warmup_ratio || 0) * 100)}%
            {Number(schedule.warmup_steps || 0) > 0 && `，显式步数 ${formatMetric("warmup_steps", Number(schedule.warmup_steps))} 优先`}
            {resolvedSteps != null ? ` · 预热 ${formatMetric("warmup_steps", Number(resolvedSteps))} / 总计 ${formatMetric("optimizer_steps", Number(totalSteps))} 步` : " · 等待引擎确定优化器总步数"}
          </dd></div>;
        })}</dl>
      </section>}

      <section className="detail-section">
        <header><div><h3>训练输出与血缘</h3><p>只展示控制面为本次训练运行实际登记的 Checkpoint 和模型版本。</p></div><GitFork size={17} /></header>
        <ErrorMessage error={registryError} />
        {registryLoading && !registry
          ? <p className="quiet-message">正在读取训练输出…</p>
          : !outputCheckpoints.length && !outputModels.length
          ? <Alert severity="info" title="尚未登记训练输出" detail={active ? "训练产物会在 Worker 完成上传并通过输出校验后显示。" : "该运行没有关联的 Checkpoint 或模型版本；平台不会根据任务名称猜测输出。"} />
          : <div className="training-output-lineage">
            {outputCheckpoints.map((checkpoint) => <article key={checkpoint.id}>
              <div className="training-output-icon"><Tag size={17} /></div>
              <div className="training-output-main">
                <span>训练 Checkpoint</span>
                <strong>Checkpoint {checkpoint.id.slice(0, 8)}</strong>
                <small>Step {formatNumber(checkpoint.step)} · {formatNumber(checkpoint.token_count)} Tokens · 制品 {checkpoint.artifact_id?.slice(0, 8) || "未登记"}</small>
              </div>
              <StatusBadge status={checkpoint.status} />
              <button type="button" className="button secondary small" onClick={() => onOpenCheckpoint(checkpoint.id)}>打开 Checkpoint</button>
            </article>)}
            {outputModels.map((model) => <article key={model.id}>
              <div className="training-output-icon model"><Package size={17} /></div>
              <div className="training-output-main">
                <span>输出模型版本</span>
                <strong>{model.name} V{model.version}</strong>
                <small>{model.kind.toUpperCase()} · {formatNumber(model.parameter_count)} 参数 · 来源 Checkpoint {model.source_checkpoint_id?.slice(0, 8) || "—"}</small>
                <small>父模型：{model.parent_model ? `${model.parent_model.name} V${model.parent_model.version}` : "无"}</small>
              </div>
              <StatusBadge status={model.status} />
              <button type="button" className="button primary small" onClick={() => onOpenModel(model.id)}>打开模型详情</button>
            </article>)}
          </div>}
      </section>

      <section className="detail-section training-observed-attempt">
        <header><div><h3>执行尝试详情</h3><p>选择一次真实执行，查看当次的资源、日志和事件。</p></div></header>
        <label>查看执行尝试<select aria-label="查看执行尝试" value={attempt?.id || selectedAttemptId} disabled={!detail?.attempts?.length} onChange={(event) => setSelectedAttemptId(event.target.value)}>
          {!detail?.attempts?.length && <option value="">尚无执行尝试</option>}
          {selectedAttemptId && detail && !attempt && <option value={selectedAttemptId}>指定执行不存在</option>}
          {detail?.attempts?.map((item) => <option key={item.id} value={item.id}>Attempt {item.attempt_number} · {formatDate(item.started_at || undefined)} · {item.id.slice(0, 8)}</option>)}
        </select></label>
        {selectedAttemptId && detail && !attempt && <p role="alert">该执行尝试不存在或不属于此训练运行。</p>}
        {attempt && <div className="detail-summary-line"><StatusBadge status={attempt.status} /><span>Worker {attempt.worker_id || "—"}</span><span>{attempt.id}</span></div>}
      </section>

      <section className="detail-section">
        <header><div><h3>资源使用</h3><p>当前所选执行尝试的系统、进程和 GPU 资源曲线。</p></div><Cpu size={17} /></header>
        <ErrorMessage error={resources.error} />
        {resourceSamples.length > 0 ? <ResourceUsageChart events={resourceEvents} /> : !resources.isLoading && <Alert severity="info" title="未采集运行资源" detail={attempt ? "该 attempt 没有可绘制的低频资源样本。" : "任务尚未开始执行。"} />}
      </section>

      <section className="detail-section">
        <header><div><h3>运行日志</h3><p>固定高度终端默认跟随最新行；向上滚动会暂停跟随，可按 300 条加载更早记录。</p></div><ScrollText size={17} /></header>
        {(task.isLoading || logs.isLoading) && <p className="quiet-message">正在读取任务日志…</p>}
        <ErrorMessage error={task.error || logs.error} />
        {!task.isLoading && !logs.isLoading && !task.error && !logs.error && !logs.data?.items.length && <Alert severity="info" title="未采集平台日志" detail={imported ? "这是历史导入的训练记录；平台保留了指标和来源映射，但导入清单没有包含原始日志。" : "该任务尚未输出日志；新任务的 stdout/stderr 会在这里持续刷新。"} />}
        {logs.data?.items.length ? <TerminalLogViewer key={attempt?.id || run.task_id} events={logs.data.items} hasMore={logs.data.has_more && logLimit < MAX_LOG_WINDOW_SIZE} loadingOlder={logs.isFetching && logLimit > LOG_WINDOW_SIZE} onLoadOlder={() => setLogLimit((value) => Math.min(MAX_LOG_WINDOW_SIZE, value + LOG_WINDOW_SIZE))} /> : null}
      </section>

      <div className="training-detail-columns">
        <section className="detail-section">
          <header><div><h3>任务尝试</h3><p>每次分配到 Worker 的执行记录。</p></div><Cpu size={17} /></header>
          {(detail?.attempts || []).length ? <div className="attempt-list">{detail!.attempts!.map((attempt) => <article key={attempt.id}><div><strong>Attempt {attempt.attempt_number}</strong><StatusBadge status={attempt.status} /></div><dl className="detail-list"><div><dt>Worker</dt><dd>{attempt.worker_id || "—"}</dd></div><div><dt>Exit code</dt><dd>{attempt.exit_code ?? "—"}</dd></div><div><dt>开始</dt><dd>{formatDate(attempt.started_at || undefined)}</dd></div></dl>{!isEmpty(attempt.error) && <pre className="json-preview error-json">{JSON.stringify(attempt.error, null, 2)}</pre>}</article>)}</div> : <p className="quiet-message">没有任务尝试记录。</p>}
        </section>
        <section className="detail-section">
          <header><div><h3>任务事件</h3><p>状态、进度、指标和制品事件。</p></div></header>
          <ErrorMessage error={timeline.error} />
          {timeline.data?.items.length ? <div className="event-timeline">{timeline.data.items.slice().reverse().map((event) => <article key={event.id}><i /><div><strong>{event.event_type}</strong><small>#{event.sequence} · {formatDate(event.created_at)}</small>{!isEmpty(event.payload) && <pre>{JSON.stringify(event.payload, null, 2)}</pre>}</div></article>)}</div> : !timeline.isLoading && <p className="quiet-message">没有普通任务事件记录。</p>}
        </section>
      </div>

      {(!isEmpty(detail?.error) || !isEmpty(detail?.result)) && <section className="detail-section">
          <header><div><h3>任务结果</h3><p>整个任务最近一次上报的结构化结果或错误；各次尝试的错误分别保留在执行记录中。</p></div></header>
        {!isEmpty(detail?.error) && <pre className="json-preview error-json">{JSON.stringify(detail?.error, null, 2)}</pre>}
        {!isEmpty(detail?.result) && <pre className="json-preview">{JSON.stringify(detail?.result, null, 2)}</pre>}
      </section>}

      <section className="detail-section collapsed-config">
        <details><summary>查看运行配置快照</summary><pre className="json-preview">{JSON.stringify(run.config_snapshot || {}, null, 2)}</pre></details>
      </section>
    </div>
  </Modal>;
}
