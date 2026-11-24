import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  Beaker,
  Activity,
  ChevronLeft,
  ChevronRight,
  Clock3,
  ClipboardList,
  GitCompareArrows,
  Plus,
  Play,
  ShieldCheck,
  Target,
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, formatDate, post, postForm } from "../api";
import type { Task } from "../types";
import { TaskTiming } from "../components/TaskTiming";
import {
  EmptyState,
  ErrorMessage,
  ExecutionFields,
  executionPayload,
  Field,
  FormActions,
  Modal,
  PageHeader,
  Panel,
  Progress,
  StatusBadge,
} from "../components/ui";

import { EvaluationDatasetList, evaluationDatasetRows, type EvaluationDatasetRow } from "../components/EvaluationDatasetList";

type EvaluationTab = "results" | "datasets" | "suites";

export function Evaluations() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<EvaluationTab>("results");
  const [launchOpen, setLaunchOpen] = useState(false);
  const [assetOpen, setAssetOpen] = useState<"dataset" | "suite" | null>(null);
  const [suiteOpen, setSuiteOpen] = useState(false);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [benchmarkOpen, setBenchmarkOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<any | null>(null);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const models = useQuery({
    queryKey: ["models", projectId],
    queryFn: () => api<any>(`/projects/${projectId}/models`),
  });
  const catalog = useQuery({
    queryKey: ["benchmark-catalog"],
    queryFn: () => api<any[]>("/benchmarks/catalog"),
  });
  const suitePresets = useQuery({
    queryKey: ["benchmark-suite-presets"],
    queryFn: () => api<any[]>("/benchmarks/suite-presets"),
  });
  const evalDatasets = useQuery({
    queryKey: ["eval-datasets", projectId],
    queryFn: () => api<any[]>(`/projects/${projectId}/evaluation-datasets`),
  });
  const sourceDatasets = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api<any[]>(`/projects/${projectId}/datasets`),
  });
  const suites = useQuery({
    queryKey: ["eval-suites", projectId],
    queryFn: () => api<any[]>(`/projects/${projectId}/evaluation-suites`),
  });
  const runs = useQuery({
    queryKey: ["evaluations", projectId],
    queryFn: () => api<any[]>(`/projects/${projectId}/evaluations`),
    refetchInterval: 2500,
  });
  const datasetRows = useMemo(() => evaluationDatasetRows(catalog.data || [], evalDatasets.data || [], [], false), [catalog.data, evalDatasets.data]);
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["evaluations", projectId] });
    queryClient.invalidateQueries({ queryKey: ["eval-datasets", projectId] });
    queryClient.invalidateQueries({ queryKey: ["eval-suites", projectId] });
  };

  return <>
    <PageHeader
      eyebrow="EVALUATION CENTER"
      title="评测中心"
      description="选择评测集、组织套件并运行模型评测，查看结果和逐题诊断。"
      actions={<>
        <button className="button secondary" disabled={(runs.data?.length || 0) < 2} title={(runs.data?.length || 0) < 2 ? "至少需要两次评测运行" : undefined} onClick={() => setComparisonOpen(true)}><GitCompareArrows size={16} />结果对比</button>
        <button className="button primary" onClick={() => setLaunchOpen(true)}><Play size={16} />发起评测</button>
      </>}
    />
    <div className="segmented-tabs">
      <button className={tab === "results" ? "active" : ""} onClick={() => setTab("results")}><BarChart3 size={16} />评测结果 <b>{runs.data?.length || 0}</b></button>
      <button className={tab === "datasets" ? "active" : ""} onClick={() => setTab("datasets")}><ClipboardList size={16} />评测集 <b>{datasetRows.length}</b></button>
      <button className={tab === "suites" ? "active" : ""} onClick={() => setTab("suites")}><ShieldCheck size={16} />评测套件 <b>{suites.data?.length || 0}</b></button>
    </div>
    {tab === "results" && <Results runs={runs.data || []} onLaunch={() => setLaunchOpen(true)} />}
    {tab === "datasets" && <>
      <Panel title="评测集（Benchmark）" description="内置和导入的评测集统一选择，来源与评分配置随版本保存。" actions={<div className="benchmark-asset-actions"><Link className="button ghost" to="/benchmarks">全局评测集</Link><button className="button secondary" onClick={() => setAssetOpen("dataset")}><Plus size={16} />添加评测集</button><button className="button primary" disabled={!selectedDatasets.length} onClick={() => setSuiteOpen(true)}>用所选项组装套件 ({selectedDatasets.length})</button></div>}>
        <ErrorMessage error={catalog.error || evalDatasets.error} />
      </Panel>
      <EvaluationDatasetList rows={datasetRows} loading={catalog.isLoading || evalDatasets.isLoading} selected={selectedDatasets} onToggle={(key) => setSelectedDatasets((current) => current.includes(key) ? current.filter((value) => value !== key) : [...current, key])} renderActions={(row) => row.kind === "version" ? <Link className="button ghost small" to={`/benchmarks?asset=${encodeURIComponent(row.item.id)}`}>查看评测集</Link> : <Link className="button ghost small" to={`/benchmarks?builtin=${encodeURIComponent(row.item.id)}`}>查看评测集</Link>} />
    </>}
    {tab === "suites" && <SuiteWorkspace
      items={suites.data || []}
      presets={suitePresets.data || []}
      datasets={evalDatasets.data || []}
      error={suitePresets.error}
      onCreate={() => { setSelectedDatasets([]); setSuiteOpen(true); }}
      onApplyPreset={(preset) => { setSelectedPreset(preset); setSelectedBenchmarks(preset.benchmark_ids || []); setBenchmarkOpen(true); }}
    />}
    {launchOpen && <EvaluationModal
      projectId={projectId}
      models={models.data?.models || []}
      suites={suites.data || []}
      onClose={() => setLaunchOpen(false)}
      onCreated={() => { refresh(); setLaunchOpen(false); }}
    />}
    {assetOpen && <EvalAssetModal
      projectId={projectId}
      type={assetOpen}
      datasets={evalDatasets.data || []}
      sourceDatasets={sourceDatasets.data || []}
      onClose={() => setAssetOpen(null)}
      onCreated={() => { refresh(); setAssetOpen(null); }}
    />}
    {suiteOpen && <EvaluationSuiteCreateModal projectId={projectId} rows={datasetRows} initialSelected={selectedDatasets} onClose={() => setSuiteOpen(false)} onCreated={() => { refresh(); setSuiteOpen(false); setSelectedDatasets([]); setTab("suites"); }} />}
    {benchmarkOpen && <BenchmarkSuiteModal
      projectId={projectId}
      selected={selectedBenchmarks}
      catalog={catalog.data || []}
      preset={selectedPreset}
      onClose={() => { setBenchmarkOpen(false); setSelectedPreset(null); }}
      onCreated={() => {
        refresh();
        setBenchmarkOpen(false);
        setSelectedPreset(null);
        setSelectedBenchmarks([]);
        setTab("suites");
      }}
    />}
    {comparisonOpen && <EvaluationComparison runs={runs.data || []} models={models.data?.models || []} onClose={() => setComparisonOpen(false)} />}
  </>;
}

function EvaluationComparison({ runs, models, onClose }: { runs: any[]; models: any[]; onClose: () => void }) {
  const modelName = (id: string) => { const model = models.find((item) => item.id === id); return model ? `${model.name} v${model.version}` : id.slice(0, 8); };
  const dimensions = Array.from(new Set(runs.flatMap((run) => Object.keys(run.summary?.dimensions || {}))));
  return <Modal title="评测结果对比" description="横向比较现有运行；可比性签名不同的结果不应直接判定回归。" onClose={onClose} wide><div className="data-table-wrap"><table className="data-table"><thead><tr><th>运行</th><th>模型</th><th>综合得分</th>{dimensions.map((name) => <th key={name}>{name}</th>)}<th>可比性</th><th>时间</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><strong>{run.id.slice(0, 8)}</strong><span><StatusBadge status={run.status} /></span></td><td>{(run.model_version_ids || []).map(modelName).join(" / ") || "—"}</td><td>{typeof run.summary?.overall === "number" ? `${(run.summary.overall * 100).toFixed(1)}%` : "—"}</td>{dimensions.map((name) => <td key={name}>{typeof run.summary?.dimensions?.[name] === "number" ? `${(run.summary.dimensions[name] * 100).toFixed(1)}%` : "—"}</td>)}<td>{run.compatibility_hash?.slice(0, 10) || "自定义"}</td><td>{formatDate(run.created_at)}</td></tr>)}</tbody></table></div><div className="form-actions"><span /><button className="button primary" onClick={onClose}>关闭</button></div></Modal>;
}

function score(value: unknown, digits = 1) {
  return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "—";
}

function taskScore(value: unknown, metric: unknown, digits = 2) {
  if (typeof value !== "number") return "—";
  const name = String(metric || "").toLowerCase();
  return name.includes("perplexity") || name === "bits_per_byte"
    ? value.toFixed(digits)
    : score(value, digits);
}

function runModels(run: any) {
  return run.config_snapshot?.models || [];
}

function runSuites(run: any) {
  return run.config_snapshot?.suites || [];
}

function runPrimaryModelId(run: any): string {
  return run.config_snapshot?.primary_model_version_id || run.summary?.primary_model_version_id || run.model_version_ids?.[0] || "";
}

const EVALUATION_MODEL_COLORS = ["#55d7c4", "#7aa8ff", "#f4b860", "#cf8cff", "#ff7b91", "#62c3ff"];

function runModelLabel(run: any, modelId: string) {
  const model = runModels(run).find((item: any) => item.id === modelId);
  return model ? `${model.name} v${model.version}` : modelId.slice(0, 8);
}

function runModelIds(run: any): string[] {
  const configured = runModels(run).map((model: any) => String(model.id));
  const reported = Object.keys(run.summary?.models || {});
  return Array.from(new Set([...configured, ...reported, ...(run.model_version_ids || [])]));
}

function Results({ runs, onLaunch }: { runs: any[]; onLaunch: () => void }) {
  const [selectedId, setSelectedId] = useState("");
  if (!runs.length) return <EmptyState title="还没有评测结果" description="选择评测集组成套件，发起第一轮评测。" action={<button className="button primary" onClick={onLaunch}><Play size={16} />发起评测</button>} />;
  const selected = runs.find((run) => run.id === selectedId);
  return <>
    <Panel title="评测轮次" description="一行代表一次冻结的评测运行；打开后查看该轮的目标、执行过程与完整结果。" className="table-panel evaluation-runs-panel">
      <div className="data-table-wrap"><table className="data-table evaluation-runs-table"><thead><tr><th>轮次 / 运行</th><th>评测目标</th><th>状态</th><th>各模型套件加权综合分</th><th>逐项完成</th><th>可比性</th><th>发起时间</th></tr></thead><tbody>{runs.map((run, index) => {
        const ordinal = runs.length - index;
        const models = runModels(run);
        const suites = runSuites(run);
        const taskCount = Object.values(run.summary?.models || {}).reduce((total: number, model: any) => total + Object.keys(model.tasks || {}).length, 0);
        const plannedPerModel = (run.config_snapshot?.benchmarks || []).length || suites.reduce((total: number, suite: any) => total + (suite.datasets?.length || 0), 0);
        const planned = plannedPerModel * Math.max(models.length || run.model_version_ids?.length || 0, 1);
        return <tr key={run.id} tabIndex={0} aria-label={`打开第 ${ordinal} 轮评测`} onClick={() => setSelectedId(run.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedId(run.id); } }}>
          <td><strong>第 {ordinal} 轮</strong><span>{run.id.slice(0, 8)}</span></td>
          <td><strong>{models.map((model: any) => `${model.id === runPrimaryModelId(run) ? "主评 " : "对照 "}${model.name} v${model.version}`).join(" / ") || `${run.model_version_ids?.length || 0} 个模型`}</strong><span>{suites.map((suite: any) => `${suite.name} v${suite.version}`).join(" / ") || `${run.suite_version_ids?.length || 0} 个套件`}</span></td>
          <td><StatusBadge status={run.status} /></td><td><div className="evaluation-run-model-scores">{runModelIds(run).map((modelId) => <span key={modelId} className={modelId === runPrimaryModelId(run) ? "primary" : ""}><b>{modelId === runPrimaryModelId(run) ? "主" : "对照"}</b>{runModelLabel(run, modelId)} <strong>{score(run.summary?.models?.[modelId]?.overall ?? (modelId === runPrimaryModelId(run) ? run.summary?.overall : undefined))}</strong></span>)}</div></td><td>{taskCount} / {planned || taskCount || "—"}</td><td>{run.compatibility_hash ? run.compatibility_hash.slice(0, 10) : "自定义"}</td><td>{formatDate(run.created_at)}</td>
        </tr>;
      })}</tbody></table></div>
    </Panel>
    {selected && <EvaluationRunDetail run={selected} runs={runs} ordinal={runs.length - runs.indexOf(selected)} onClose={() => setSelectedId("")} />}
  </>;
}

function EvaluationRunDetail({ run, runs, ordinal, onClose }: { run: any; runs: any[]; ordinal: number; onClose: () => void }) {
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const task = useQuery<Task>({
    queryKey: ["evaluation-task", run.task_id],
    queryFn: () => api<Task>(`/tasks/${run.task_id}?include_events=false`),
    enabled: Boolean(run.task_id),
    refetchInterval: (query) => ["succeeded", "failed", "cancelled", "lost"].includes(query.state.data?.observed_state || run.status) ? false : 2000,
  });
  const detail = task.data;
  const models = runModels(run);
  const suites = runSuites(run);
  const primaryModelId = runPrimaryModelId(run);
  const modelIds = runModelIds(run);
  const chartModels = modelIds.map((modelId, index) => ({
    id: modelId,
    label: runModelLabel(run, modelId),
    color: EVALUATION_MODEL_COLORS[index % EVALUATION_MODEL_COLORS.length],
  }));
  const dimensionNames = Array.from(new Set(modelIds.flatMap((modelId) => Object.keys(run.summary?.models?.[modelId]?.dimensions || {}))));
  const dimensionChart = dimensionNames.map((name) => ({
    name,
    ...Object.fromEntries(modelIds.map((modelId) => [modelId, typeof run.summary?.models?.[modelId]?.dimensions?.[name] === "number" ? run.summary.models[modelId].dimensions[name] * 100 : null])),
  }));
  const previous = runs.find((candidate) => {
    return candidate.id !== run.id && candidate.created_at < run.created_at && candidate.compatibility_hash && candidate.compatibility_hash === run.compatibility_hash && runPrimaryModelId(candidate) === primaryModelId && typeof candidate.summary?.overall === "number";
  });
  const delta = previous && typeof run.summary?.overall === "number" ? run.summary.overall - previous.summary.overall : null;
  const datasetNames = new Map<string, string>();
  for (const suite of suites) for (const dataset of suite.datasets || []) datasetNames.set(dataset.task_id || dataset.benchmark_config?.task_id, dataset.name || dataset.benchmark_config?.name);
  const itemRows = Object.entries(run.summary?.models || {}).flatMap(([modelId, modelResult]: [string, any]) => Object.entries(modelResult.tasks || {}).map(([taskId, result]: [string, any]) => ({ modelId, taskId, ...result })));
  const primaryResult = run.summary?.models?.[primaryModelId];
  const primaryTasks = Object.values(primaryResult?.tasks || {}) as any[];
  const includedTasks = primaryTasks.filter((item) => item.include_in_overall !== false);
  const diagnosticTasks = primaryTasks.filter((item) => item.include_in_overall === false);
  const totalWeight = primaryResult?.total_weight ?? includedTasks.reduce((total, item) => total + Number(item.weight ?? 1), 0);
  const taskIds = Array.from(new Set(modelIds.flatMap((modelId) => Object.keys(run.summary?.models?.[modelId]?.tasks || {}))));
  const attempt = detail?.attempts?.[0];
  const batchExecution = run.summary?.execution;
  const sampling = run.inference_config?.sampling || run.config_snapshot?.inference_config?.sampling;
  const samplingPlanHashes = Array.from(new Set(Object.values(batchExecution?.sampling?.benchmarks || {}).map((item: any) => item.plan_hash).filter(Boolean)));
  const batchRows = Object.values(batchExecution?.models || {}).flatMap((tasks: any) => Object.values(tasks || {})) as any[];
  const effectiveBatchSizes = Array.from(new Set(batchRows.flatMap((row: any) => row.effective_batch_sizes || (row.effective_batch_size ? [row.effective_batch_size] : [])))).sort((left: any, right: any) => Number(left) - Number(right));
  const oomBackoffs = batchRows.reduce((total: number, row: any) => total + Number(row.oom_backoffs || 0), 0);
  return <><Modal title={`第 ${ordinal} 轮评测 · ${run.id.slice(0, 8)}`} description="本轮冻结目标、实际执行过程与最终结果" onClose={onClose} wide>
    <div className="training-detail-body evaluation-run-detail">
      <div className="detail-summary-line"><StatusBadge status={detail?.observed_state || run.status} /><span><Activity size={14} />任务 {run.task_id?.slice(0, 8) || "—"}</span><span><Clock3 size={14} />{formatDate(detail?.started_at || run.created_at)} → {formatDate(detail?.ended_at || undefined)}</span></div>
      <Progress value={detail?.progress ?? (run.status === "succeeded" ? 1 : 0)} />
      {detail && <TaskTiming task={detail} detail />}
      <section className="detail-section"><header><div><h3>评测目标</h3><p>发起评测时冻结的主评模型、对照模型、套件、抽样身份和评分配置。</p></div><Target size={17} /></header><dl className="detail-list evaluation-goal-list"><div><dt>主评模型</dt><dd>{models.find((model: any) => model.id === primaryModelId) ? `${models.find((model: any) => model.id === primaryModelId).name} v${models.find((model: any) => model.id === primaryModelId).version}` : primaryModelId || "—"}</dd></div><div><dt>对照模型</dt><dd>{models.filter((model: any) => model.id !== primaryModelId).map((model: any) => `${model.name} v${model.version}`).join(" / ") || "无"}</dd></div><div><dt>评测套件</dt><dd>{suites.map((suite: any) => `${suite.name} v${suite.version}`).join(" / ") || (run.suite_version_ids || []).join(" / ")}</dd></div><div><dt>评测项目</dt><dd>{(run.config_snapshot?.benchmarks || []).map((item: any) => item.name || item.task_id).join("、") || `${itemRows.length || "—"} 项`}</dd></div><div><dt>题目范围</dt><dd>{sampling ? `${sampling.percentage}% · Seed ${sampling.seed} · ${sampling.method}` : run.inference_config?.limit ? `历史 limit ${run.inference_config.limit}` : "历史运行未记录"}</dd></div><div><dt>抽样计划</dt><dd>{samplingPlanHashes.length ? samplingPlanHashes.map((hash: any) => String(hash).slice(0, 12)).join(" / ") : sampling ? "运行中生成" : "—"}</dd></div><div><dt>聚合策略</dt><dd>{run.scoring_config?.aggregator || run.config_snapshot?.scoring_config?.aggregator || "weighted_mean"}</dd></div><div><dt>推理配置</dt><dd>{JSON.stringify(run.inference_config || run.config_snapshot?.inference_config || {})}</dd></div></dl></section>
      <section className="detail-section"><header><div><h3>评测过程</h3><p>任务阶段、进度和实际 Worker 执行记录。</p></div><Activity size={17} /></header><div className="run-detail-metrics"><div><span>当前阶段</span><strong>{detail?.current_phase || (run.status === "succeeded" ? "completed" : "等待任务信息")}</strong></div><div><span>执行进度</span><strong>{Math.round((detail?.progress ?? (run.status === "succeeded" ? 1 : 0)) * 100)}%</strong></div><div><span>Attempt</span><strong>{detail?.attempts?.length || 0}</strong></div><div><span>退出码</span><strong>{attempt?.exit_code ?? "—"}</strong></div></div><dl className="detail-list metric-detail-list"><div><dt>Worker</dt><dd>{attempt?.worker_id || "—"}</dd></div><div><dt>执行时间</dt><dd>{detail?.timing_estimate?.actual_duration_seconds ? `${Math.round(detail.timing_estimate.actual_duration_seconds / 60)} 分钟` : "—"}</dd></div><div><dt>Batch 策略</dt><dd>{batchExecution ? `${batchExecution.batch_policy === "auto" ? "自动" : "手动"}（请求 ${batchExecution.requested_batch_size}）` : "历史运行未记录"}</dd></div><div><dt>实际 Batch</dt><dd>{effectiveBatchSizes.length ? effectiveBatchSizes.join(" / ") : batchExecution ? "运行时未返回" : "—"}{oomBackoffs ? ` · OOM 降档 ${oomBackoffs} 次` : ""}</dd></div></dl><ErrorMessage error={task.error || (detail?.error && Object.keys(detail.error).length ? detail.error : null)} /></section>
      <section className="detail-section evaluation-score-section">
        <header><div><h3>各模型综合分与能力对比</h3><p>同轮模型使用同一抽样计划；综合分按套件项目权重聚合，能力图每条线代表一个模型。</p></div><BarChart3 size={17} /></header>
        <div className="evaluation-model-score-grid">{chartModels.map((model) => {
          const result = run.summary?.models?.[model.id];
          const modelDelta = model.id === primaryModelId ? delta : null;
          return <article key={model.id} style={{ "--model-color": model.color } as any}>
            <div><span>{model.id === primaryModelId ? "主评模型" : "对照模型"}</span><i style={{ background: model.color }} /></div>
            <strong>{score(result?.overall ?? (model.id === primaryModelId ? run.summary?.overall : undefined), 2)}</strong>
            <p>{model.label}{model.id === primaryModelId && previous ? ` · 较上一可比轮 ${modelDelta != null && modelDelta >= 0 ? "+" : ""}${modelDelta != null ? (modelDelta * 100).toFixed(2) : "—"} pp` : ""}</p>
          </article>;
        })}</div>
        <div className="evaluation-score-formula">
          <div><strong>套件加权综合分怎么算？</strong><code>Σ（计入项目得分 × 项目权重）÷ Σ项目权重</code></div>
          <p>按评测项目加权，不按题数加权。本轮主评模型计入 {includedTasks.length} 项、诊断 {diagnosticTasks.length} 项，计入权重合计 {Number(totalWeight).toFixed(2)}。标为“诊断项”的结果不会拉高或拉低综合分；历史运行保留当时冻结的项目范围。</p>
        </div>
        {dimensionChart.length ? <div className="chart-box evaluation-comparison-chart"><ResponsiveContainer width="100%" height={320}><LineChart data={dimensionChart} margin={{ top: 18, right: 28, left: -6, bottom: 8 }}><CartesianGrid stroke="#1c2c41" strokeDasharray="3 3" /><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "#aab9ca" }} interval={0} /><YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: "#7890ad" }} tickFormatter={(value) => `${value}%`} /><Tooltip formatter={(value, name) => [`${Number(value || 0).toFixed(2)}%`, chartModels.find((model) => model.id === name)?.label || name]} contentStyle={{ background: "#111d2d", border: "1px solid #243752", borderRadius: 10 }} /><Legend formatter={(value) => chartModels.find((model) => model.id === value)?.label || value} />{chartModels.map((model) => <Line key={model.id} type="monotone" dataKey={model.id} name={model.id} stroke={model.color} strokeWidth={model.id === primaryModelId ? 3 : 2} dot={{ r: model.id === primaryModelId ? 4 : 3, fill: model.color }} activeDot={{ r: 6 }} connectNulls={false} />)}</LineChart></ResponsiveContainer></div> : <p className="quiet-message">该轮尚未产生能力维度结果。</p>}
      </section>
      <section className="detail-section">
        <header><div><h3>逐项模型对比与逐题报告</h3><p>每个项目一行、每个模型一列；点击任一分数查看该模型的题目、回答、参考答案与逐题得分。</p></div><ClipboardList size={17} /></header>
        {taskIds.length ? <div className="data-table-wrap"><table className="data-table evaluation-item-table"><thead><tr><th>评测项目</th><th>指标 / 样本</th><th>聚合</th>{chartModels.map((model) => <th key={model.id}><span className="evaluation-table-model-heading"><i style={{ background: model.color }} />{model.label}{model.id === primaryModelId ? "（主）" : ""}</span></th>)}</tr></thead><tbody>{taskIds.map((taskId) => {
          const results = modelIds.map((modelId) => run.summary?.models?.[modelId]?.tasks?.[taskId]);
          const representative = results.find(Boolean) || {};
          const itemName = datasetNames.get(taskId) || representative.raw_metrics?.alias || taskId;
          const sampleCount = representative.raw_metrics?.sample_len ?? Object.values(representative.raw_metrics?.sample_count || {})[0] ?? "—";
          return <tr key={taskId} aria-label={`${itemName} 模型对比`}><td><strong>{itemName}</strong><span>{representative.benchmark_id || taskId}</span></td><td>{representative.metric || "—"}<span>{sampleCount} 题</span></td><td>{representative.include_in_overall === false ? <span className="diagnostic-badge">诊断项</span> : `权重 ${representative.weight ?? 1}`}</td>{modelIds.map((modelId, index) => { const result = results[index]; return <td key={modelId}>{result ? <button type="button" className="evaluation-score-button" aria-label={`查看 ${itemName} ${runModelLabel(run, modelId)} 逐题结果`} onClick={() => setSelectedItem({ ...result, modelId, taskId, name: itemName })}>{taskScore(result.score, result.metric, 2)}<span>查看逐题</span></button> : "—"}</td>; })}</tr>;
        })}</tbody></table></div> : <p className="quiet-message">该轮尚未产生逐项结果。</p>}
      </section>
      {run.compatibility_hash && <section className="detail-section collapsed-config"><details><summary>查看可比性签名与冻结配置</summary><pre className="json-preview">{JSON.stringify({ compatibility_hash: run.compatibility_hash, benchmark_signature: run.benchmark_signature, config_snapshot: run.config_snapshot }, null, 2)}</pre></details></section>}
    </div>
  </Modal>{selectedItem && <EvaluationItemSamples key={`${selectedItem.modelId}-${selectedItem.taskId}`} run={run} item={selectedItem} modelName={models.find((model: any) => model.id === selectedItem.modelId)?.name || selectedItem.modelId.slice(0, 8)} onClose={() => setSelectedItem(null)} />}</>;
}

function displayAnswer(value: any) {
  if (value == null || value === "") return "—";
  if (typeof value === "object" && !Array.isArray(value) && "label" in value) return `${value.label}. ${typeof value.value === "string" ? value.value : JSON.stringify(value.value)}`;
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function EvaluationItemSamples({ run, item, modelName, onClose }: { run: any; item: any; modelName: string; onClose: () => void }) {
  const pageSize = 20;
  const [offset, setOffset] = useState(0);
  const samples = useQuery<any>({
    queryKey: ["evaluation-samples", run.id, item.modelId, item.taskId, offset],
    queryFn: () => api<any>(`/projects/${run.project_id}/evaluations/${run.id}/samples?model_version_id=${encodeURIComponent(item.modelId)}&task_id=${encodeURIComponent(item.taskId)}&offset=${offset}&limit=${pageSize}`),
  });
  return <Modal title={`${item.name} · 逐题结果`} description={`${modelName} · ${item.metric || "主指标"} · 项目得分 ${taskScore(item.score, item.metric, 2)}`} onClose={onClose} wide>
    <div className="evaluation-sample-detail">
      {samples.isLoading && <p className="quiet-message">正在读取完整评测报告…</p>}
      <ErrorMessage error={samples.error} />
      {samples.data?.source && <div className="evaluation-report-source"><strong>逐题报告已持久化</strong><span>原始 JSONL：{samples.data.source}</span></div>}
      {(samples.data?.items || []).map((sample: any, index: number) => <article className="evaluation-sample-card" key={`${sample.id}-${index}`}>
        <header><div><strong>问题 #{offset + index + 1}</strong><span>{sample.metadata?.subject || sample.metadata?.category || `ID ${sample.id || "—"}`}</span></div><StatusBadge status={sample.status} /><b>{score(sample.score, 2)}</b></header>
        <section><h4>问题</h4><p>{sample.question ? (typeof sample.question === "string" ? sample.question : JSON.stringify(sample.question)) : "报告未提供题目正文"}</p>{sample.choices?.length ? <ol className="evaluation-sample-choices" type="A">{sample.choices.map((choice: any, choiceIndex: number) => <li key={choiceIndex}>{typeof choice === "string" ? choice : JSON.stringify(choice)}</li>)}</ol> : null}</section>
        <div className="evaluation-answer-grid"><section><h4>模型回答</h4><pre>{displayAnswer(sample.model_answer)}</pre></section><section><h4>参考答案</h4><pre>{displayAnswer(sample.reference_answer)}</pre></section></div>
        <footer><span>指标：{sample.metric}</span>{Object.entries(sample.metrics || {}).map(([name, value]) => <span key={name}>{name}: {String(value)}</span>)}</footer>
      </article>)}
      {!samples.isLoading && !samples.error && !samples.data?.items?.length && <p className="quiet-message">该项目没有逐题记录。</p>}
      {samples.data && <div className="evaluation-sample-pagination"><span>第 {offset + 1}–{offset + samples.data.items.length} 条</span><div><button type="button" className="button secondary small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}><ChevronLeft size={14} />上一页</button><button type="button" className="button secondary small" disabled={!samples.data.has_more} onClick={() => setOffset(offset + pageSize)}>下一页<ChevronRight size={14} /></button></div></div>}
    </div>
  </Modal>;
}

function EvalMetric({ label, value }: { label: string; value: any }) {
  return <article className="metric-card"><span>{label}</span><strong>{value}</strong></article>;
}

function EvaluationSuiteCreateModal({ projectId, rows, initialSelected, onClose, onCreated }: { projectId: string; rows: EvaluationDatasetRow[]; initialSelected: string[]; onClose: () => void; onCreated: () => void }) {
  const [selected, setSelected] = useState(initialSelected);
  const chosen = rows.filter((row) => selected.includes(row.key));
  const isBuiltin = (row: EvaluationDatasetRow) => row.kind === "builtin" || Boolean(row.item.benchmark_config?.id);
  const mixed = chosen.some(isBuiltin) && chosen.some((row) => !isBuiltin(row));
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/evaluation-suites`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!chosen.length || mixed) return;
    const form = new FormData(event.currentTarget);
    const gate = String(form.get("gate") || "").trim();
    create.mutate({
      name: form.get("name"),
      datasets: chosen.filter((row) => row.kind === "version").map((row) => ({ dataset_version_id: row.item.id, weight: 1 })),
      benchmark_ids: chosen.filter((row) => row.kind === "builtin").map((row) => row.item.id),
      scoring_config: { aggregator: "weighted_mean" },
      release_gate: gate ? { overall: Number(gate) } : {},
    });
  }
  return <Modal title="新建评测套件" description="选择评测集并冻结精确版本，已有评分配置保持不变。" onClose={onClose} wide><form className="form-grid" onSubmit={submit}>
    <EvaluationDatasetList rows={rows} selected={selected} onToggle={(key) => setSelected((current) => current.includes(key) ? current.filter((value) => value !== key) : [...current, key])} />
    <div className="form-grid two-columns"><Field label="套件名称"><input name="name" required defaultValue={`评测套件 ${new Date().toLocaleDateString()}`} /></Field><Field label="综合分发布门槛（可选）"><input name="gate" type="number" min="0" max="1" step="0.01" placeholder="留空，仅评测" /></Field></div>
    {mixed && <p className="form-error" role="alert">所选评测集使用不同执行器（lm-eval / 用户命令），当前需要分别创建套件。</p>}
    <ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!chosen.length || mixed} onCancel={onClose} submitLabel="创建不可变套件" />
  </form></Modal>;
}

function SuiteWorkspace({ items, presets, datasets, error, onCreate, onApplyPreset }: { items: any[]; presets: any[]; datasets: any[]; error: unknown; onCreate: () => void; onApplyPreset: (preset: any) => void }) {
  const [openSuite, setOpenSuite] = useState<any | null>(null);
  return <>
    <Panel className="benchmark-presets" title="套件模板（尚未创建）" description="模板提供推荐的评测集组合，使用后可检查配置并创建项目套件。">
      <ErrorMessage error={error} />
      {presets.length ? <div className="benchmark-preset-grid">{presets.map((preset) => <article key={preset.id}><div><span>BASE MODEL TEMPLATE</span><strong>{preset.name}</strong><p>{preset.description}</p></div><div className="benchmark-tags"><span>{preset.benchmark_ids?.length || 0} 个评测项目</span><span>0-shot</span><span>Chat Template 关闭</span></div><button type="button" className="button primary small" onClick={() => onApplyPreset(preset)}>从模板创建套件</button></article>)}</div> : <p className="quiet-message">当前没有可用的套件模板。</p>}
    </Panel>
    <Panel title="已创建的评测套件" description="每张卡片都是项目内不可变的任务组合，可直接用于发起评测。" actions={<button type="button" className="button primary" onClick={onCreate}><Plus size={16} />新建空白套件</button>}>
      <AssetGrid items={items} type="suite" onCreate={onCreate} onOpen={setOpenSuite} />
    </Panel>
    {openSuite && <SuiteDetailModal suite={openSuite} datasets={datasets} onClose={() => setOpenSuite(null)} />}
  </>;
}

function SuiteDetailModal({ suite, datasets, onClose }: { suite: any; datasets: any[]; onClose: () => void }) {
  const benchmarkSuite = suite.scoring_config?.asset_type === "benchmark";
  const aggregator = suite.scoring_config?.aggregator === "weighted_mean" ? "加权平均" : (suite.scoring_config?.aggregator || "自定义");
  const overallGate = suite.release_gate?.overall;
  const recommended = suite.scoring_config?.recommended_inference_config || {};
  const datasetById = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  const rows = (suite.datasets || []).map((entry: any) => {
    const dataset = datasetById.get(entry.dataset_version_id) as any;
    const config = dataset?.benchmark_config || entry.benchmark_config || {};
    return {
      id: entry.dataset_version_id || entry.benchmark_id || entry.task_id,
      name: dataset?.name || config.name || entry.name || entry.task_id || "未命名项目",
      taskId: entry.task_id || config.task_id || dataset?.task_type || "—",
      taskType: config.task_type || dataset?.task_type || "—",
      metric: config.primary_metric || entry.primary_metric || dataset?.scoring_config?.primary_metric || "—",
      weight: typeof entry.weight === "number" ? entry.weight : 1,
      fewshot: config.execution?.num_fewshot ?? config.defaults?.num_fewshot ?? "默认",
      include: config.include_in_overall !== false,
      hash: entry.config_hash || config.config_hash || dataset?.compatibility_hash,
    };
  });
  return <Modal title={`${suite.name} · V${suite.version}`} description={`${benchmarkSuite ? "标准套件" : "自定义套件"} · ${rows.length} 个评测项目 · 不可变版本`} onClose={onClose} wide className="evaluation-suite-detail">
    <div className="training-detail-body">
      <section className="detail-section"><header><div><h3>套件配置</h3><p>以下配置会在发起评测时随套件版本冻结。</p></div><ShieldCheck size={17} /></header>
        <dl className="detail-list evaluation-goal-list"><div><dt>套件 ID</dt><dd>{suite.id}</dd></div><div><dt>创建时间</dt><dd>{formatDate(suite.created_at)}</dd></div><div><dt>套件类型</dt><dd>{benchmarkSuite ? "标准 Benchmark 套件" : "自定义评测套件"}</dd></div><div><dt>综合分算法</dt><dd>{aggregator}</dd></div><div><dt>综合分门槛</dt><dd>{typeof overallGate === "number" ? `${(overallGate * 100).toFixed(0)}%` : "未设置"}</dd></div><div><dt>来源模板</dt><dd>{suite.scoring_config?.preset_id || "手动创建"}</dd></div></dl>
      </section>
      <section className="detail-section"><header><div><h3>建议推理配置</h3><p>创建评测时可以检查并调整，实际运行仍以运行快照为准。</p></div><Target size={17} /></header>
        <div className="suite-inference-grid"><div><span>题目比例</span><strong>{typeof recommended.sample_percentage === "number" ? `${recommended.sample_percentage}%` : "未指定"}</strong></div><div><span>抽样 Seed</span><strong>{recommended.sample_seed ?? "未指定"}</strong></div><div><span>Chat Template</span><strong>{recommended.apply_chat_template === true ? "开启" : recommended.apply_chat_template === false ? "关闭" : "未指定"}</strong></div></div>
      </section>
      <section className="detail-section"><header><div><h3>评测项目明细</h3><p>按套件引用的冻结评测集版本展示；权重参与套件综合分计算。</p></div><ClipboardList size={17} /></header>
        <div className="data-table-wrap"><table className="data-table suite-item-table"><thead><tr><th>评测项目</th><th>任务 ID</th><th>类型</th><th>主要指标</th><th>权重</th><th>Few-shot</th><th>计入综合分</th><th>配置身份</th></tr></thead><tbody>{rows.map((row: any) => <tr key={row.id}><td><strong>{row.name}</strong></td><td><code>{row.taskId}</code></td><td>{row.taskType}</td><td>{row.metric}</td><td>{row.weight}</td><td>{row.fewshot}</td><td>{row.include ? "是" : "否（诊断项）"}</td><td><code>{row.hash?.slice(0, 10) || "—"}</code></td></tr>)}</tbody></table></div>
      </section>
    </div>
  </Modal>;
}

function BenchmarkSuiteModal({ projectId, selected, catalog, preset, onClose, onCreated }: { projectId: string; selected: string[]; catalog: any[]; preset?: any | null; onClose: () => void; onCreated: () => void }) {
  const create = useMutation({
    mutationFn: (data: any) => post(`/projects/${projectId}/benchmark-suites`, data),
    onSuccess: onCreated,
  });
  const selectedItems = catalog.filter((item) => selected.includes(item.id));
  const recommendedInference = preset?.recommended_inference_config || preset?.scoring_config?.recommended_inference_config || {};
  const presetMode = preset?.model_type === "instruction" ? "指令模型" : "基模";
  const chatTemplateMode = recommendedInference.apply_chat_template ? "开启" : "关闭";
  const samplePercentage = recommendedInference.sample_percentage ?? 100;
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const gate = String(form.get("gate") || "").trim();
    create.mutate({
      name: form.get("name"),
      benchmark_ids: selected,
      overrides: preset?.overrides || {},
      scoring_config: preset?.scoring_config || { aggregator: "weighted_mean" },
      release_gate: gate ? { overall: Number(gate) } : {},
    });
  }
  return <Modal title="创建评测套件" description="任务定义和默认 few-shot 会冻结为项目内不可变版本。" onClose={onClose} wide><form className="form-grid" onSubmit={submit}>
    {preset ? <div className="benchmark-preset-summary"><strong>{preset.name}</strong><span>{presetMode} · 统一 0-shot · {chatTemplateMode} Chat Template · {samplePercentage}% 样本</span>{preset.notes?.map((note: string) => <small key={note}>{note}</small>)}</div> : null}
    <div className="selected-benchmark-list">{selectedItems.map((item) => { const fewshot = preset?.overrides?.[item.id]?.num_fewshot ?? item.defaults?.num_fewshot; return <article key={item.id}><div><strong>{item.name}</strong><span>{item.task_id} · {item.primary_metric}</span></div><b>{fewshot ?? "默认"}-shot</b></article>; })}</div>
    <div className="form-grid two-columns"><Field label="套件名称"><input name="name" required defaultValue={preset?.name || `标准评测套件 ${new Date().toLocaleDateString()}`} /></Field><Field label="综合分发布门槛（可选）"><input name="gate" type="number" min="0" max="1" step="0.01" placeholder="留空，仅评测" /></Field></div>
    <ErrorMessage error={create.error} />
    <FormActions submitting={create.isPending || !selected.length} onCancel={onClose} submitLabel="创建不可变套件" />
  </form></Modal>;
}

function AssetGrid({ items, type, onCreate, onOpen }: { items: any[]; type: "dataset" | "suite"; onCreate: () => void; onOpen?: (item: any) => void }) {
  return items.length ? <section className="asset-grid">{items.map((item) => {
    const benchmark = type === "dataset" ? item.source_type === "benchmark" : item.scoring_config?.asset_type === "benchmark";
    const aggregator = item.scoring_config?.aggregator === "weighted_mean" ? "加权平均" : (item.scoring_config?.aggregator || "自定义");
    const overallGate = item.release_gate?.overall;
    const card = <Panel title={`${item.name} · V${item.version}`} description={type === "dataset" ? `${benchmark ? item.benchmark_config?.task_id : item.task_type} · ${item.license}` : `${item.datasets?.length || 0} 个评测项目`} actions={type === "suite" ? <span className="suite-card-actions"><span className={`suite-type-badge ${benchmark ? "standard" : "custom"}`}>{benchmark ? "标准套件" : "自定义套件"}</span><ChevronRight size={17} /></span> : <StatusBadge status={benchmark ? "benchmark" : "ready"} />}><div className="asset-facts">{type === "dataset" ? <><span>{benchmark ? "配置" : "样本"} <b>{benchmark ? item.compatibility_hash?.slice(0, 8) : item.sample_count}</b></span><span>能力维度 <b>{item.dimensions?.length || 0}</b></span></> : <><span>综合分算法 <b>{aggregator}</b></span><span>综合分门槛 <b>{typeof overallGate === "number" ? `${(overallGate * 100).toFixed(0)}%` : "未设置"}</b></span></>}</div></Panel>;
    return type === "suite" ? <article key={item.id} className="suite-card-button" role="button" tabIndex={0} aria-label={`查看套件 ${item.name} V${item.version}`} onClick={() => onOpen?.(item)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onOpen?.(item); } }}>{card}</article> : <div key={item.id}>{card}</div>;
  })}</section> : <EmptyState title={type === "dataset" ? "还没有评测集" : "还没有评测套件"} description="选择内置或导入的评测集，创建版本化评测套件。" action={<button className="button primary" onClick={onCreate}><Plus size={16} />新建</button>} />;
}

function evaluationRuntimeLabel(model: any) {
  const source = model.inference_source || {};
  if (source.type === "native") return source.display_name || "项目原生推理";
  if (source.type === "api") return `${source.endpoint_mode === "completions" ? "Completions" : "Chat Completions"} · ${source.endpoint_model || "API"}`;
  if (model.model_weight_version_id || source.type === "external") return "内置 Transformers 推理";
  if (source.type === "pretrained" || source.type === "managed") return "登记的模型推理配置";
  return source.display_name || "模型推理配置不可用";
}

function evaluationRuntimeIssue(model: any) {
  const source = model.inference_source || {};
  if (["missing", "unsupported", "unavailable"].includes(source.type)) return source.reason || "模型没有可用的推理来源";
  if (source.type === "command") return "该项目推理适配器尚未提供 Benchmark bridge";
  return "";
}

function hasEvaluationDatasets(suite: any) {
  return Array.isArray(suite.datasets) && suite.datasets.length > 0;
}

function EvaluationModal({ projectId, models, suites, onClose, onCreated }: { projectId: string; models: any[]; suites: any[]; onClose: () => void; onCreated: () => void }) {
  const [selectedModels, setSelectedModels] = useState<string[]>(models.slice(0, 1).map((item) => item.id));
  const [primaryModelId, setPrimaryModelId] = useState(models[0]?.id || "");
  const [selectedSuites, setSelectedSuites] = useState<string[]>(() => suites.filter(hasEvaluationDatasets).slice(0, 1).map((item) => item.id));
  const chosenSuites = suites.filter((suite) => selectedSuites.includes(suite.id));
  const endpointOnly = selectedModels.length > 0 && selectedModels.every((id) => models.find((model) => model.id === id)?.inference_source?.type === "api");
  const hasRunnableSuite = suites.some(hasEvaluationDatasets);
  const benchmarkMode = chosenSuites.length > 0 && chosenSuites.every((suite) => hasEvaluationDatasets(suite) && suite.datasets.every((entry: any) => entry.benchmark_id));
  const recommendedInference = chosenSuites.find((suite) => suite.scoring_config?.recommended_inference_config)?.scoring_config?.recommended_inference_config || {};
  const start = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/evaluations`, data), onSuccess: onCreated });
  const selectedModelRecords = selectedModels.map((id) => models.find((model) => model.id === id)).filter(Boolean);
  const runtimeIssues = benchmarkMode ? selectedModelRecords.map((model) => ({ model, issue: evaluationRuntimeIssue(model) })).filter((item) => item.issue) : [];
  const chatEndpointSelected = selectedModelRecords.some((model) => model.inference_source?.type === "api" && model.inference_source?.endpoint_mode !== "completions");
  const incompatibleTasks = benchmarkMode && chatEndpointSelected ? chosenSuites.flatMap((suite) => suite.datasets || []).filter((entry: any) => (entry.task_type || entry.benchmark_config?.task_type) !== "generation").map((entry: any) => entry.task_id || entry.benchmark_config?.task_id).filter(Boolean) : [];
  function toggleModel(modelId: string, selected: boolean) {
    const next = selected ? [...selectedModels, modelId] : selectedModels.filter((id) => id !== modelId);
    setSelectedModels(next);
    if (!next.includes(primaryModelId)) setPrimaryModelId(next[0] || "");
  }
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedModels.length || !selectedSuites.length || chosenSuites.some((suite) => !hasEvaluationDatasets(suite))) return;
    const form = new FormData(event.currentTarget);
    const execution = executionPayload(form);
    const batchMode = String(form.get("benchmark_batch_mode") || "auto");
    start.mutate({
      model_version_ids: selectedModels,
      primary_model_version_id: primaryModelId || selectedModels[0],
      suite_version_ids: selectedSuites,
      inference_config: benchmarkMode ? {
        batch_size: batchMode === "auto" ? "auto" : Number(form.get("benchmark_batch_size")),
        device: form.get("benchmark_device") || "cuda",
        sample_percentage: Number(form.get("benchmark_sample_percentage")),
        sample_seed: Number(form.get("benchmark_sample_seed")),
        apply_chat_template: form.get("benchmark_chat_template") === "on",
        seed: "0,1234,1234,1234",
      } : { temperature: 0, max_tokens: 512 },
      scoring_config: { save_samples: true, max_sample_results: 200 },
      ...execution,
    });
  }
  return <Modal title="发起评测" description="模型、套件、推理配置和评测集版本会被冻结。" onClose={onClose} wide><form onSubmit={submit}>
    <div className="selection-columns"><div><h3>选择模型</h3>{models.map((model) => { const selected = selectedModels.includes(model.id); return <div className="selection-row evaluation-model-choice" key={model.id}><label><input aria-label={`选择 ${model.name} v${model.version}`} type="checkbox" checked={selected} onChange={(event) => toggleModel(model.id, event.target.checked)} /><span><b>{model.name} v{model.version}</b><small>{model.inference_source?.type === "api" ? `${model.inference_source.endpoint_mode === "completions" ? "Completions" : "Chat Completions"} · 厂商托管` : `${model.kind} · ${model.status}`}</small></span></label>{selected && <label className="primary-model-choice"><input type="radio" name="primary_model" checked={primaryModelId === model.id} onChange={() => setPrimaryModelId(model.id)} /><span>{primaryModelId === model.id ? "主评" : "设为主评"}</span></label>}</div>; })}</div><div><h3>选择评测套件</h3>{suites.map((suite) => {
      const empty = !hasEvaluationDatasets(suite);
      return <label className={`selection-row${empty ? " disabled" : ""}`} aria-disabled={empty} key={suite.id}><input type="checkbox" disabled={empty} checked={selectedSuites.includes(suite.id)} onChange={(event) => setSelectedSuites(event.target.checked ? [...selectedSuites, suite.id] : selectedSuites.filter((id) => id !== suite.id))} /><span><b>{suite.name} v{suite.version}</b><small>{empty ? "0 个评测集 · 空套件，不能启动；请先选择评测集创建套件" : `${suite.datasets.length} 个评测集 · ${suite.scoring_config?.asset_type === "benchmark" ? "标准 Benchmark" : "自定义"}`}</small></span></label>;
    })}</div></div>
    {!hasRunnableSuite && <div className="form-error" role="alert">请先选择评测集创建套件。</div>}
    {benchmarkMode && <div className="benchmark-runtime"><h3>模型运行时</h3><p>与 Playground 使用同一个冻结推理配置；评测只补充 Batch、Device 和样本范围。</p>{selectedModelRecords.map((model) => <div className="benchmark-runtime-row" key={model.id}><strong>{model.name}</strong><span className="endpoint-runtime-badge">{evaluationRuntimeLabel(model)}</span>{model.inference_source?.endpoint_url && <code>{model.inference_source.endpoint_url}</code>}</div>)}</div>}
    {runtimeIssues.map(({ model, issue }) => <div className="form-error" role="alert" key={model.id}>{model.name}：{issue}</div>)}
    {!!incompatibleTasks.length && <div className="form-error" role="alert">Chat Completions 端点不能执行需要 Log-likelihood 的项目：{Array.from(new Set(incompatibleTasks)).join("、")}</div>}
    {benchmarkMode && <div className="form-grid two-columns benchmark-options"><Field label="Batch 策略" hint={endpointOnly ? "API 模型自动使用单请求批量，服务并发由端点配置控制" : "自动模式根据模型、请求长度和当前可用显存探测，并在 OOM 时降档"}><select aria-label="Batch 策略" name="benchmark_batch_mode" defaultValue="auto"><option value="auto">自动（推荐）</option><option value="manual">手动</option></select></Field><Field label="手动 Batch size" hint="仅手动策略生效；CUDA OOM 时平台按二分序列自动降档"><input aria-label="手动 Batch size" name="benchmark_batch_size" type="number" min="1" max="65536" step="1" defaultValue="1" required /></Field><Field label="Device"><select key={endpointOnly ? "endpoint-only" : "local-model"} name="benchmark_device" defaultValue={endpointOnly ? "cpu" : "cuda"}><option value="cuda">cuda</option><option value="cpu">cpu</option><option value="mps">mps</option></select></Field><Field label="使用题目比例（%）" hint="按固定 Seed 对每个子任务确定性抽样；相同比例和 Seed 会复用同一批题"><input key={`percentage-${selectedSuites.join("-")}`} aria-label="使用题目比例（%）" name="benchmark_sample_percentage" type="number" min="0.01" max="100" step="0.01" defaultValue={recommendedInference.sample_percentage ?? 100} required /></Field><Field label="抽样 Seed" hint="只控制题目选择，不影响模型生成随机数"><input key={`seed-${selectedSuites.join("-")}`} aria-label="抽样 Seed" name="benchmark_sample_seed" type="number" min="0" max="2147483647" step="1" defaultValue={recommendedInference.sample_seed ?? 42} required /></Field><label className="check-field benchmark-checkbox"><input key={`${chatEndpointSelected ? "chat-endpoint" : "other-model"}-${selectedSuites.join("-")}`} name="benchmark_chat_template" type="checkbox" defaultChecked={recommendedInference.apply_chat_template ?? chatEndpointSelected} /><span><b>应用 Chat Template</b><small>基模预设默认关闭；Chat Completions 端点默认启用，其他模型可手动调整</small></span></label></div>}
    {!!selectedSuites.length && <div className="form-grid two-columns"><ExecutionFields key={endpointOnly ? "endpoint-only" : "local-model"} adapterId={benchmarkMode ? "builtin.lm-eval" : "builtin.command.evaluation"} defaultGpus={endpointOnly ? 0 : 1} /></div>}
    <ErrorMessage error={start.error} />
    <FormActions submitting={start.isPending} disabled={!selectedModels.length || !selectedSuites.length || !!runtimeIssues.length || !!incompatibleTasks.length || !primaryModelId} onCancel={onClose} submitLabel="启动评测" />
  </form></Modal>;
}

function EvalAssetModal({ projectId, type, datasets, sourceDatasets, onClose, onCreated }: { projectId: string; type: "dataset" | "suite"; datasets: any[]; sourceDatasets: any[]; onClose: () => void; onCreated: () => void }) {
  const [sourceMode, setSourceMode] = useState<"upload" | "data_version">("upload");
  const sourceVersions = sourceDatasets.flatMap((dataset) => (dataset.versions || [])
    .filter((version: any) => version.artifact_id)
    .map((version: any) => ({ dataset, version })));
  const create = useMutation({
    mutationFn: ({ path, data, form }: { path: string; data?: any; form?: FormData }) => form ? postForm(path, form) : post(path, data),
    onSuccess: onCreated,
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    if (type === "suite") {
      create.mutate({ path: `/projects/${projectId}/evaluation-suites`, data: {
        name: form.get("name"),
        datasets: Array.from(form.getAll("dataset_ids")).map((id) => ({ dataset_version_id: id, weight: 1 })),
        scoring_config: { aggregator: "weighted_mean" },
        release_gate: { overall: Number(form.get("gate")) },
      } });
      return;
    }
    const dimensions = String(form.get("dimensions")).split(/[,，]/).map((value) => value.trim()).filter(Boolean);
    const contaminationTags = String(form.get("contamination_tags")).split(/[,，]/).map((value) => value.trim()).filter(Boolean);
    const schemaConfig = {
      prompt_field: String(form.get("prompt_field") || "").trim(),
      answer_field: String(form.get("answer_field") || "").trim(),
      id_field: String(form.get("id_field") || "").trim(),
      choices_field: String(form.get("choices_field") || "").trim(),
    };
    const scoringConfig = { metric: String(form.get("metric") || "exact_match").trim() };
    if (sourceMode === "upload") {
      form.set("dimensions", JSON.stringify(dimensions));
      form.set("contamination_tags", JSON.stringify(contaminationTags));
      form.set("schema_config", JSON.stringify(schemaConfig));
      form.set("scoring_config", JSON.stringify(scoringConfig));
      create.mutate({ path: `/projects/${projectId}/evaluation-datasets/upload`, form });
    } else {
      create.mutate({ path: `/projects/${projectId}/evaluation-datasets/from-data-version`, data: {
        name: form.get("name"),
        dataset_version_id: form.get("dataset_version_id"),
        task_type: form.get("task_type"),
        license: form.get("license"),
        dimensions,
        contamination_tags: contaminationTags,
        schema_config: schemaConfig,
        scoring_config: scoringConfig,
      } });
    }
  }
  return <Modal title={type === "dataset" ? "添加真实评测集" : "新建评测套件"} description={type === "dataset" ? "上传自建题目，或复用数据中心已冻结的 Hugging Face/HTTP/Git/本地数据版本。" : undefined} onClose={onClose} wide={type === "dataset"}><form className="form-grid" onSubmit={submit}><Field label="名称"><input name="name" required /></Field>{type === "dataset" ? <>
    <div className="source-mode-switch" role="group" aria-label="评测集来源"><button type="button" className={sourceMode === "upload" ? "active" : ""} onClick={() => setSourceMode("upload")}>上传 JSONL / CSV</button><button type="button" className={sourceMode === "data_version" ? "active" : ""} onClick={() => setSourceMode("data_version")}>选择数据中心版本</button></div>
    {sourceMode === "upload" ? <Field label="评测文件" hint="UTF-8，支持 JSONL、NDJSON、CSV、TSV，最大 256 MiB"><input name="file" type="file" accept=".jsonl,.ndjson,.csv,.tsv" required /></Field> : <Field label="数据中心版本" hint="Hugging Face 数据请先在数据中心完成下载和冻结">{sourceVersions.length ? <select name="dataset_version_id" required defaultValue=""><option value="" disabled>选择不可变版本</option>{sourceVersions.map(({ dataset, version }) => <option key={version.id} value={version.id}>{dataset.name} · V{version.version} · {dataset.source_type} · {version.document_count || 0} 条</option>)}</select> : <div className="form-hint-block">还没有带数据快照的版本。<Link to={`/projects/${projectId}/data`}>前往数据中心导入 Hugging Face 或本地数据</Link></div>}</Field>}
    <div className="form-grid two-columns"><Field label="任务类型"><select name="task_type"><option value="generation">generation</option><option value="classification">classification</option><option value="multiple_choice">multiple_choice</option></select></Field><Field label="许可证"><input name="license" defaultValue={sourceMode === "upload" ? "internal" : "unknown"} /></Field><Field label="能力维度"><input name="dimensions" defaultValue="领域知识, 指令遵循" /></Field><Field label="污染标签" hint="公开题库建议填写 public-benchmark"><input name="contamination_tags" defaultValue={sourceMode === "upload" ? "private-eval" : "public-source"} /></Field></div>
    <Panel title="字段映射" description="字段名会随数据版本冻结，平台不会猜测答案或评分语义。"><div className="form-grid two-columns"><Field label="Prompt 字段"><input name="prompt_field" required defaultValue="prompt" /></Field><Field label="答案字段"><input name="answer_field" required defaultValue="answer" /></Field><Field label="样本 ID 字段"><input name="id_field" defaultValue="id" /></Field><Field label="选项字段"><input name="choices_field" defaultValue="choices" /></Field></div></Panel>
    <Field label="主要评分指标"><select name="metric" defaultValue="exact_match"><option value="exact_match">exact_match</option><option value="accuracy">accuracy</option><option value="f1">f1</option><option value="pass@1">pass@1</option><option value="llm_judge">llm_judge</option></select></Field>
    <div className="benchmark-notice"><Beaker size={17} /><span>自定义运行会把本机数据路径写入 <code>LLM_PLATFORM_EVAL_DATASETS</code>；评分仍由你配置的评测命令完成。代码题不会由平台自动执行。</span></div>
  </> : <><Field label="评测集">{datasets.map((dataset) => <label className="check-field" key={dataset.id}><input name="dataset_ids" type="checkbox" value={dataset.id} /><span>{dataset.name} v{dataset.version}</span></label>)}</Field><Field label="综合分发布门槛"><input name="gate" type="number" step="0.01" defaultValue="0.6" /></Field></>}<ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={type === "dataset" && sourceMode === "data_version" && !sourceVersions.length} onCancel={onClose} /></form></Modal>;
}
