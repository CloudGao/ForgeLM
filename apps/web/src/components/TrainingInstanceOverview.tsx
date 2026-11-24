import { Activity, ArrowUpRight, GitBranch } from "lucide-react";
import { formatDate, formatNumber } from "../api";
import type { ProjectPipelineInstance, ProjectPipelineNodeRun, ProjectPipelineRun, TrainingRun } from "../types";
import { executionScopeLabel } from "./TrainingExecutionDialog";
import { EmptyState, StatusBadge } from "./ui";

function reportsTrainingRun(value: unknown, run: TrainingRun): boolean {
  if (!value || typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some((item) => reportsTrainingRun(item, run));
  const record = value as Record<string, unknown>;
  if (record.training_run_id === run.id || (record.id === run.id && (record.task_id === run.task_id || record.kind === "training_run"))) return true;
  return Object.entries(record).some(([key, item]) => !["command", "command_result"].includes(key) && reportsTrainingRun(item, run));
}

export function trainingRunsFor(instance: ProjectPipelineInstance, attempts: ProjectPipelineRun[], runs: TrainingRun[], selectedAttempt?: ProjectPipelineRun) {
  const executions = selectedAttempt ? [selectedAttempt] : attempts;
  const ids = new Set(executions.flatMap((attempt) => [attempt.source_refs?.training_run_id, ...attempt.node_runs.flatMap((node) => [node.task?.payload?.training_run_id, node.config_snapshot?.training_run_id, node.input_snapshot?.training_run_id])]));
  const taskIds = new Set(executions.flatMap((attempt) => attempt.node_runs.map((node) => node.task_id)).filter(Boolean));
  if (!selectedAttempt) ids.add(instance.source_refs?.training_run_id);
  return runs.filter((run) => ids.has(run.id) || taskIds.has(run.task_id) || executions.some((attempt) => attempt.node_runs.some((node) => reportsTrainingRun(node.output_snapshot, run))) || (!selectedAttempt && run.config_snapshot?.revision?.effective_config?.training_instance_id === instance.id));
}

export function TrainingRunSummaries({ runs, onOpen }: { runs: TrainingRun[]; onOpen: (run: TrainingRun) => void }) {
  return <section className="training-overview-section" aria-label="训练指标与产物">
    <header><div><h2>训练指标与产物</h2><p>训练运行的累计指标。打开详情可查看全部执行尝试、日志、配置与输出版本。</p></div></header>
    {runs.length ? <div className="training-run-summaries">{runs.map((run) => <article key={run.id}>
      <div className="training-summary-title"><strong>{run.current_stage || "模型训练"} · {run.id.slice(0, 8)}</strong><StatusBadge status={run.status} /></div>
      <div className="run-detail-metrics"><div><span>Step</span><strong>{formatNumber(run.current_step)}</strong></div><div><span>Tokens</span><strong>{formatNumber(run.current_tokens)}</strong></div><div><span>Loss</span><strong>{run.latest_metrics?.loss ?? run.latest_metrics?.cross_entropy ?? "—"}</strong></div></div>
      <button className="button secondary small" onClick={() => onOpen(run)}><ArrowUpRight size={15} />打开训练详情</button>
    </article>)}</div> : <p className="training-muted">尚无已关联的训练运行。其他节点的执行与产物可从执行记录查看。</p>}
  </section>;
}

export function TrainingInstanceOverview({ instance, attempts, attemptsLoading, runs, runsLoading, onOpenAttempt, onOpenRun, onOpenGraph }: {
  instance: ProjectPipelineInstance; attempts: ProjectPipelineRun[]; attemptsLoading: boolean; runs: TrainingRun[]; runsLoading: boolean;
  onOpenAttempt: (id: string) => void; onOpenRun: (run: TrainingRun) => void; onOpenGraph: () => void;
}) {
  const latest = attempts[0];
  return <div className="training-overview" aria-label="训练实例概览">
    <section className="training-overview-section">
      <header><div><h2>训练实例</h2><p>{instance.description || "这份训练配置已冻结。每次执行分别保留状态、日志和产物。"}</p></div>{latest && <StatusBadge status={latest.status} />}</header>
      <dl className="training-instance-facts"><div><dt>实例 ID</dt><dd>{instance.id}</dd></div><div><dt>创建时间</dt><dd>{formatDate(instance.created_at)}</dd></div><div><dt>配置状态</dt><dd>{instance.locked ? "已冻结 · 修改需复制新实例" : "历史草稿"}</dd></div><div><dt>执行次数</dt><dd>{attemptsLoading ? "读取中…" : attempts.length}</dd></div></dl>
      <button className="button secondary small" onClick={onOpenGraph}><GitBranch size={15} />查看工作流与冻结配置</button>
    </section>
    {runsLoading ? <p role="status">正在读取训练指标与产物…</p> : <TrainingRunSummaries runs={runs} onOpen={onOpenRun} />}
    <section className="training-overview-section" aria-label="实例执行历史">
      <header><div><h2>执行记录</h2><p>每个 Attempt 保留自己的执行范围和结果。</p></div></header>
      {attemptsLoading ? <p>正在读取执行记录…</p> : !attempts.length ? <EmptyState title="尚未执行" description="开始执行后，首次 Attempt 会出现在这里。" /> : <div className="training-execution-history">{attempts.map((attempt, i) => <button key={attempt.id} onClick={() => onOpenAttempt(attempt.id)}>
        <Activity size={18} /><div><strong>Attempt {attempt.context_snapshot?.__training_instance?.attempt_number || attempts.length - i}</strong><p>{executionScopeLabel(attempt)}</p><small>{formatDate(attempt.created_at)} · {attempt.id.slice(0, 8)}</small></div><StatusBadge status={attempt.status} /><span>查看执行详情 →</span>
      </button>)}</div>}
    </section>
  </div>;
}

export function TrainingAttemptOverview({ attempt, runs, onOpenRun, onTask, onOpenGraph }: {
  attempt: ProjectPipelineRun; runs: TrainingRun[]; onOpenRun: (run: TrainingRun) => void; onTask: (node: ProjectPipelineNodeRun) => void; onOpenGraph: () => void;
}) {
  return <div className="training-overview" aria-label="Attempt 执行详情">
    <section className="training-overview-section"><header><div><h2>执行详情</h2><p>{executionScopeLabel(attempt)}</p></div><button className="button secondary small" onClick={onOpenGraph}><GitBranch size={15} />查看执行图与上游血缘</button></header>
      <dl className="training-instance-facts"><div><dt>Attempt ID</dt><dd>{attempt.id}</dd></div><div><dt>所属实例</dt><dd>{attempt.instance_name || attempt.instance_id}</dd></div><div><dt>开始时间</dt><dd>{formatDate(attempt.started_at || undefined)}</dd></div><div><dt>结束时间</dt><dd>{formatDate(attempt.ended_at || undefined)}</dd></div></dl>
    </section>
    <TrainingRunSummaries runs={runs} onOpen={onOpenRun} />
    <section className="training-overview-section"><header><div><h2>节点执行与输出</h2><p>本次执行的内部步骤及其实际记录。</p></div></header>
      <div className="training-execution-history">{attempt.node_runs.map((node) => <article key={node.id}><div><strong>{node.label}</strong><p>{node.input_snapshot?.excluded_from_execution ? "范围外 · 未执行" : node.input_snapshot?.reused_from_node_run_id ? "复用历史结果" : node.error?.message || node.node_id}</p></div><StatusBadge status={node.status} />{node.task_id && <button className="button secondary small" onClick={() => onTask(node)}>资源、日志与产物</button>}</article>)}</div>
    </section>
  </div>;
}
