import { ReactFlowProvider } from "@xyflow/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Archive, ArrowLeft, Check, Copy, GitBranch, Play, Plus, RotateCcw, Search, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, ApiError, formatDate, formatNumber, post } from "../api";
import { NativeTrainingConfig } from "../components/NativeTrainingConfig";
import { InterleavedGenerationDialog } from "../components/InterleavedGenerationDialog";
import { TrainingCanvas, type TrainingGraphValue } from "../components/TrainingCanvas";
import { TrainingExecutionDialog, executionScopeLabel, type ExecutionSelectionMode } from "../components/TrainingExecutionDialog";
import { TaskDetail } from "../components/TaskDetail";
import { TrainingRunDetail } from "../components/TrainingRunDetail";
import { TrainingAttemptOverview, TrainingInstanceOverview, trainingRunsFor } from "../components/TrainingInstanceOverview";
import { EmptyState, ErrorMessage, PageHeader, Panel, Progress, StatusBadge } from "../components/ui";
import type { Json, ModelRegistry, PipelineNodeCatalogItem, PipelineValidationReport, ProjectPipelineDagNode, ProjectPipelineInstance, ProjectPipelineNodeRun, ProjectPipelineOptions, ProjectPipelineRun, TrainingLineageNode, TrainingRun } from "../types";

type Instance = ProjectPipelineInstance & { cloned_from_instance_id?: string | null; training_instance?: boolean; attempt_count?: number };
const activeStates = new Set(["queued", "running", "cancelling"]);

function freshGraph(): TrainingGraphValue {
  return { name: "", description: "", nodes: [{ id: "training", type: "training", label: "模型训练", description: "配置训练参数与输入版本", position: { x: 200, y: 180 }, config: { execution_mode: "native", training_spec: {} } }], edges: [], variables: {}, viewport: {} };
}

function cloneGraph(instance: Instance): TrainingGraphValue {
  const nodes = structuredClone(instance.nodes);
  for (const node of nodes) { delete node.config.plan_revision_id; delete node.config._native_plan_id; }
  const variables = structuredClone(instance.variables || {});
  delete variables.__training_instance; delete variables.__forgelm; delete variables.__training_history;
  return { name: `${instance.name} · 副本`, description: instance.description, nodes, edges: structuredClone(instance.edges), variables, viewport: instance.viewport || {}, cloned_from_instance_id: instance.id };
}

function graphFromRun(run: TrainingRun): TrainingGraphValue {
  const revision = run.config_snapshot?.revision || {};
  const graph = freshGraph();
  graph.name = `${run.current_stage || "历史训练"} · 副本`;
  graph.nodes[0].config.training_spec = {
    ...revision,
    stages: run.config_snapshot.selected_stages || revision.stages || [],
    engine_config: revision.effective_config?.engine_config || {},
    input_model_ref: run.config_snapshot.resume_checkpoint?.checkpoint_id
      ? { ...revision.input_model_ref, type: "checkpoint_finetune", checkpoint_id: run.config_snapshot.resume_checkpoint.checkpoint_id, data_cursor_policy: "reset" }
      : revision.input_model_ref || {},
  };
  return graph;
}

function inputSummary(instance: Instance) {
  return instance.nodes.flatMap((node) => {
    const spec = node.config?.training_spec;
    if (spec) return (spec.stages || []).map((stage: Json) => `${stage.name || stage.type}${stage.target_tokens ? ` · ${formatNumber(stage.target_tokens)} Tokens` : stage.budget?.value ? ` · ${formatNumber(stage.budget.value)} ${stage.budget.unit}` : ""}`);
    if (node.type === "data") return [`数据 ${String(node.config.asset_version_id || "未绑定").slice(0, 8)}`];
    return [];
  }).join(" · ") || `${instance.node_count} 个节点 · ${instance.edge_count} 条依赖`;
}

export function Training() {
  const { projectId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const base = `/projects/${projectId}/training-instances`;
  const [draft, setDraft] = useState<TrainingGraphValue | null>(null);
  const [report, setReport] = useState<PipelineValidationReport | null>(null);
  const [editingNode, setEditingNode] = useState<ProjectPipelineDagNode | null>(null);
  const [executionSelection, setExecutionSelection] = useState<{ mode: ExecutionSelectionMode; nodeId?: string; sourceId?: string } | null>(null);
  const [search, setSearch] = useState("");
  const [includeArchived, setIncludeArchived] = useState(false);
  const [interleavedOpen, setInterleavedOpen] = useState(false);
  const [taskDetail, setTaskDetail] = useState<{ id: string; attemptId?: string } | null>(null);
  const instanceId = params.get("instance") || "";
  const attemptId = params.get("attempt") || "";
  const legacyRunId = params.get("node_run") || "";
  const oldRunId = params.get("run") || "";
  const resolvedRun = useQuery({ queryKey: ["training-run-instance", projectId, oldRunId], queryFn: () => api<{ instance_id: string }>(`${base}/resolve-run/${oldRunId}`), enabled: Boolean(oldRunId), retry: false });
  useEffect(() => { if (resolvedRun.data) { setParams({ instance: resolvedRun.data.instance_id, ...(params.get("task_attempt") ? { attempt: params.get("task_attempt")!, view: "attempts" } : {}) }, { replace: true }); } }, [resolvedRun.data, setParams]);
  const instances = useQuery({ queryKey: ["training-instances", projectId, includeArchived], queryFn: () => api<Instance[]>(`${base}?include_archived=${includeArchived}`), refetchInterval: draft ? false : 3000 });
  const instance = useQuery({ queryKey: ["training-instance", projectId, instanceId], queryFn: () => api<Instance>(`${base}/${instanceId}`), enabled: Boolean(instanceId) && !draft, refetchInterval: 3000 });
  const attempts = useQuery({ queryKey: ["training-attempts", projectId, instanceId], queryFn: () => api<ProjectPipelineRun[]>(`${base}/${instanceId}/attempts`), enabled: Boolean(instanceId) && !draft, refetchInterval: (q) => q.state.data?.some((r) => activeStates.has(r.status)) ? 2000 : false });
  const catalog = useQuery({ queryKey: ["pipeline-node-catalog", projectId], queryFn: () => api<PipelineNodeCatalogItem[]>(`/projects/${projectId}/pipeline-node-catalog`), staleTime: 60_000 });
  const options = useQuery({ queryKey: ["pipeline-options", projectId], queryFn: () => api<ProjectPipelineOptions>(`/projects/${projectId}/pipeline-options`), enabled: Boolean(draft) });
  const legacyRuns = useQuery({ queryKey: ["runs", projectId], queryFn: () => api<TrainingRun[]>(`/projects/${projectId}/training-runs`), refetchInterval: draft ? false : 5000 });
  const legacyRun = useQuery({ queryKey: ["training-run", legacyRunId], queryFn: () => api<TrainingRun>(`/training-runs/${legacyRunId}`), enabled: Boolean(legacyRunId), refetchInterval: (q) => q.state.data && ["succeeded", "failed", "cancelled", "lost"].includes(q.state.data.status) ? false : 2000 });
  const registry = useQuery({ queryKey: ["model-registry", projectId], queryFn: () => api<ModelRegistry>(`/projects/${projectId}/models?include_archived=true`), enabled: Boolean(legacyRunId) });
  const savedKey = `training-draft:${projectId}`;
  const comparisonId = params.get("comparison") || "";
  const comparisonDraft = useQuery({ queryKey: ["comparison-draft", projectId, comparisonId], queryFn: () => api<Json>(`/projects/${projectId}/training-performance/${comparisonId}/draft`), enabled: Boolean(comparisonId), retry: false });
  useEffect(() => {
    if (comparisonDraft.data && comparisonId) {
      const graph = freshGraph(); graph.name = "采用实测算子组合继续训练";
      graph.nodes[0].config.training_spec = comparisonDraft.data;
      setDraft(graph); setReport(null); setParams({}, { replace: true });
    }
  }, [comparisonDraft.data, comparisonId, setParams]);
  useEffect(() => { if (draft) sessionStorage.setItem(savedKey, JSON.stringify(draft)); }, [draft, savedKey]);

  function invalidate() {
    queryClient.invalidateQueries({ queryKey: ["training-instances", projectId] });
    queryClient.invalidateQueries({ queryKey: ["training-instance", projectId] });
    queryClient.invalidateQueries({ queryKey: ["training-attempts", projectId] });
    queryClient.invalidateQueries({ queryKey: ["tasks", projectId] });
  }
  function openInstance(id: string) { setDraft(null); setParams({ instance: id }); }
  function newDraft(value?: TrainingGraphValue) {
    let next = value;
    if (!next) { try { next = JSON.parse(sessionStorage.getItem(savedKey) || "null"); } catch { /* Start a clean editor if an old draft cannot be parsed. */ } }
    setReport(null); create.reset(); validate.reset(); setDraft(next || freshGraph());
  }
  const create = useMutation({ mutationFn: (value: TrainingGraphValue) => post<Instance>(base, value), onSuccess: (created) => { sessionStorage.removeItem(savedKey); invalidate(); queryClient.setQueryData(["training-instance", projectId, created.id], created); openInstance(created.id); }, onError: (error) => { if (error instanceof ApiError && error.detail?.errors) setReport(error.detail as PipelineValidationReport); } });
  const copyInstance = useMutation({ mutationFn: async (item: Instance) => {
    if (item.instance_kind === "imported" && item.source_refs?.training_run_id) {
      const run = await api<TrainingRun>(`/training-runs/${item.source_refs.training_run_id}`);
      return { ...graphFromRun(run), name: `${item.name} · 副本`, cloned_from_instance_id: item.id };
    }
    return cloneGraph(item);
  }, onSuccess: (graph) => newDraft(graph) });
  const validate = useMutation({ mutationFn: (value: TrainingGraphValue) => post<PipelineValidationReport>(`${base}/validate`, value), onSuccess: setReport });
  const execute = useMutation({ mutationFn: ({ id, retry }: { id: string; retry?: string }) => post<ProjectPipelineRun>(`${base}/${id}/attempts`, { retry_from_attempt_id: retry || null }), onSuccess: (run) => { invalidate(); setParams({ instance: run.instance_id, attempt: run.id, view: "attempts" }); } });
  const cancel = useMutation({ mutationFn: (run: ProjectPipelineRun) => post(`${base}/${run.instance_id}/attempts/${run.id}/cancel`), onSuccess: invalidate });
  const archive = useMutation({ mutationFn: (item: Instance) => api(`${base}/${item.id}/archive`, { method: "PATCH", body: JSON.stringify({ archived: item.status !== "archived" }) }), onSuccess: invalidate });
  const control = useMutation({ mutationFn: ({ taskId, action }: { taskId: string; action: string }) => post(`/tasks/${taskId}/control`, { action, reason: "从训练实例控制节点执行" }), onSuccess: invalidate });
  const selectedInstance = instance.data;
  const latest = attempts.data?.[0];
  const selectedAttempt = attemptId ? attempts.data?.find((a) => a.id === attemptId) : latest;
  const showingAttempts = params.get("view") === "attempts" || Boolean(attemptId);
  const showingOverview = !showingAttempts && params.get("view") === "overview";
  const showingConfig = !showingAttempts && !showingOverview;
  const showingAttemptGraph = params.get("panel") !== "details";
  const associatedRuns = selectedInstance ? trainingRunsFor(selectedInstance, attempts.data || [], legacyRuns.data || [], showingAttempts ? selectedAttempt : undefined) : [];
  const selectedRun = legacyRun.data?.project_id === projectId ? legacyRun.data : undefined;
  function openExecutionSelection(mode: ExecutionSelectionMode, nodeId?: string) {
    setExecutionSelection({ mode, nodeId, sourceId: selectedAttempt && ["succeeded", "failed", "cancelled"].includes(selectedAttempt.status) ? selectedAttempt.id : undefined });
  }
  const executeNode = selectedInstance?.can_run && selectedInstance.instance_kind !== "imported" ? (nodeId: string, mode: ExecutionSelectionMode) => openExecutionSelection(mode, nodeId) : undefined;
  const modal = <>
    {interleavedOpen && selectedInstance && <InterleavedGenerationDialog instance={selectedInstance} sourceTaskId={latest?.node_runs.find(n => n.node_type === "fact_translation")?.task_id || undefined} onClose={() => setInterleavedOpen(false)} onCreated={created => { setInterleavedOpen(false); invalidate(); setParams({ instance: created.id }); }} />}
    {executionSelection && selectedInstance && <TrainingExecutionDialog instance={selectedInstance} attempts={attempts.data || []} initialMode={executionSelection.mode} initialNodeId={executionSelection.nodeId} initialSourceId={executionSelection.sourceId} onClose={() => setExecutionSelection(null)} onStarted={(run) => { setExecutionSelection(null); invalidate(); setParams({ instance: run.instance_id, attempt: run.id, view: "attempts" }); }} />}
    {taskDetail && <TaskDetail taskId={taskDetail.id} initialAttemptId={taskDetail.attemptId} onClose={() => setTaskDetail(null)} />}
    {legacyRunId && <ErrorMessage error={legacyRun.error} />}
    {selectedRun && <TrainingRunDetail key={selectedRun.id} run={selectedRun} initialAttemptId={params.get("task_attempt") || undefined} registry={registry.data} registryLoading={registry.isPending} registryError={registry.error} onClose={() => { const next = new URLSearchParams(params); next.delete("node_run"); next.delete("task_attempt"); setParams(next, { replace: true }); }} onOpenModel={(id) => navigate(`/projects/${projectId}/models?model=${id}`)} onOpenCheckpoint={(id) => navigate(`/projects/${projectId}/models?tab=checkpoints&checkpoint=${id}`)} />}
  </>;
  function openTrainingRun(id: string, taskAttemptId?: string) { const next = new URLSearchParams(params); next.set("node_run", id); next.delete("task_attempt"); if (taskAttemptId) next.set("task_attempt", taskAttemptId); setParams(next); }
  function openAssociatedRun(run: TrainingRun) { const node = showingAttempts ? selectedAttempt?.node_runs.find((n) => n.task_id === run.task_id) : undefined; openTrainingRun(run.id, node?.input_snapshot?.task_attempt_id); }
  function openAttempt(id: string, panel?: string) { setParams({ instance: instanceId, view: "attempts", attempt: id, ...(panel ? { panel } : {}) }); }
  function openHistory(node: TrainingLineageNode) {
    const source = node.source;
    if (source.instance_id) setParams({ instance: source.instance_id, ...(source.attempt_id ? { view: "attempts", attempt: source.attempt_id } : {}) });
    else if (source.training_run_id) setParams({ run: source.training_run_id });
    else if (source.task_id) setTaskDetail({ id: source.task_id, attemptId: source.task_attempt_id || undefined });
  }
  function openTask(node: ProjectPipelineNodeRun) { if (node.task_id) setTaskDetail({ id: node.task_id, attemptId: node.input_snapshot?.task_attempt_id || undefined }); }

  if (draft) return <div className="pipeline-workspace training-workspace">
    <header className="pipeline-workspace-header"><button className="icon-button" aria-label="返回训练列表" onClick={() => setDraft(null)}><ArrowLeft size={19} /></button><div className="pipeline-workspace-title"><span>{draft.cloned_from_instance_id ? "复制训练 · 编辑副本" : "创建训练"}</span><div><h1>{draft.name || "新的训练实例"}</h1><b>编辑草稿</b></div></div><div className="pipeline-workspace-actions"><button className="button secondary" disabled={!draft.name.trim() || validate.isPending || create.isPending} onClick={() => validate.mutate(draft)}><Check size={16} />检查配置</button><button className="button primary" disabled={!draft.name.trim() || create.isPending} onClick={() => create.mutate(draft)}><Plus size={16} />{create.isPending ? "正在创建…" : "创建并冻结实例"}</button></div></header>
    {catalog.isPending ? <Panel>正在加载节点库…</Panel> : <ReactFlowProvider><TrainingCanvas value={draft} catalog={catalog.data || []} options={options.data} onChange={(value) => { setDraft(value); setReport(null); }} onConfigureTraining={setEditingNode} /></ReactFlowProvider>}
    {report && <div className="pipeline-validation-drawer" role="status"><header><strong>{report.valid ? "配置检查通过" : `${report.errors.length} 项需要修正`}</strong></header>{[...report.errors, ...report.warnings].map((item, i) => <p key={i}>{item.node_id ? `${item.node_id}：` : ""}{item.message}</p>)}</div>}
    <ErrorMessage error={create.error || validate.error || catalog.error || options.error} />
    {editingNode && <NativeTrainingConfig projectId={projectId} spec={editingNode.config.training_spec || {}} onClose={() => setEditingNode(null)} onSave={(spec) => { setDraft({ ...draft, nodes: draft.nodes.map((n) => n.id === editingNode.id ? { ...n, config: { ...n.config, execution_mode: "native", training_spec: spec } } : n) }); setEditingNode(null); setReport(null); }} />}
  </div>;

  if (instanceId) return <div className="pipeline-workspace training-workspace">
    <header className="pipeline-workspace-header"><button className="icon-button" aria-label="返回训练列表" onClick={() => setParams({})}><ArrowLeft size={19} /></button><div className="pipeline-workspace-title"><span>训练实例 · {selectedInstance?.id.slice(0, 8)}</span><div><h1>{selectedInstance?.name || "读取实例…"}</h1>{selectedInstance && <StatusBadge status={selectedInstance.status} />}</div></div>{selectedInstance && <div className="pipeline-workspace-actions">
      <button className="button secondary" onClick={() => copyInstance.mutate(selectedInstance)}><Copy size={15} />复制并修改</button>
      {selectedInstance.nodes.length === 2 && selectedInstance.nodes.some(n => n.type === "fact_translation") && selectedInstance.nodes.some(n => n.type === "knowledge") && <button className="button secondary" onClick={() => setInterleavedOpen(true)}>分批交错</button>}
      <button className="button secondary" disabled={archive.isPending || Boolean(latest && activeStates.has(latest.status))} onClick={() => archive.mutate(selectedInstance)}><Archive size={15} />{selectedInstance.status === "archived" ? "恢复实例" : "归档"}</button>
      {selectedInstance.can_restore_execution && selectedInstance.status !== "archived" && selectedInstance.source_refs?.task_id && <button className="button primary" disabled={control.isPending} onClick={() => control.mutate({ taskId: selectedInstance.source_refs!.task_id!, action: latest?.status === "paused" || latest?.status === "cancelled" ? "resume" : "retry" })}><RotateCcw size={15} />恢复执行</button>}
      {selectedInstance.instance_kind !== "imported" && <button className="button primary" disabled={!selectedInstance.can_run || execute.isPending} onClick={() => execute.mutate({ id: instanceId })}><Play size={15} />{latest ? "从头执行" : "开始执行"}</button>}
      {selectedInstance.instance_kind !== "imported" && <button className="button secondary" disabled={!selectedInstance.can_run || attempts.isPending} onClick={() => openExecutionSelection("selected")}><GitBranch size={15} />选择节点执行</button>}
      {latest && ["failed", "cancelled"].includes(latest.status) && selectedInstance.status === "ready" && selectedInstance.instance_kind !== "imported" && <button className="button primary" disabled={execute.isPending} onClick={() => execute.mutate({ id: instanceId, retry: latest.id })}><RotateCcw size={15} />重试并恢复</button>}
    </div>}</header>
    <div className="pipeline-tabs" role="tablist" aria-label="训练实例视图"><button role="tab" aria-selected={showingOverview} className={showingOverview ? "active" : ""} onClick={() => setParams({ instance: instanceId, view: "overview" })}><Activity size={16} />实例概览</button><button role="tab" aria-selected={showingAttempts} className={showingAttempts ? "active" : ""} onClick={() => latest ? openAttempt(latest.id) : setParams({ instance: instanceId, view: "attempts" })}><Activity size={16} />执行记录 <span>{attempts.data?.length || 0}</span></button><button role="tab" aria-selected={showingConfig} className={showingConfig ? "active" : ""} onClick={() => setParams({ instance: instanceId, view: "config" })}><GitBranch size={16} />工作流与配置</button></div>
    <ErrorMessage error={copyInstance.error || instance.error || attempts.error || catalog.error || execute.error || cancel.error || archive.error || control.error} />
    {!selectedInstance && instance.isPending && <Panel>正在读取实例…</Panel>}
    <ErrorMessage error={legacyRuns.error} />
    {selectedInstance && showingOverview && <TrainingInstanceOverview instance={selectedInstance} attempts={attempts.data || []} attemptsLoading={attempts.isPending} runs={associatedRuns} runsLoading={legacyRuns.isPending} onOpenAttempt={openAttempt} onOpenRun={openAssociatedRun} onOpenGraph={() => setParams({ instance: instanceId, view: "config" })} />}
    {selectedInstance && showingConfig && <ReactFlowProvider><TrainingCanvas key={instanceId} value={selectedInstance} catalog={catalog.data || []} readOnly onExecuteNode={executeNode} onTask={openTask} onTrainingRun={openTrainingRun} onOpenHistory={openHistory} /></ReactFlowProvider>}
    {selectedInstance && showingAttempts && <>
      <div className="training-attempt-strip">{attempts.isPending && <p>正在读取执行记录…</p>}{!attempts.isPending && !attempts.data?.length && <EmptyState title="尚未执行" description="配置已冻结，点击开始执行创建首次 Attempt。" />}{attempts.data?.map((a, i, all) => <button key={a.id} className={`training-attempt-chip ${a.id === selectedAttempt?.id ? "selected" : ""}`} onClick={() => openAttempt(a.id, showingAttemptGraph ? "graph" : "details")}><strong>Attempt {a.context_snapshot?.__training_instance?.attempt_number || all.length - i}</strong><StatusBadge status={a.status} /><small>{formatDate(a.created_at)}</small><span>{executionScopeLabel(a)}</span></button>)}</div>
      {attemptId && !selectedAttempt && !attempts.isPending && <p role="alert">该执行记录不存在或不属于当前实例。</p>}
      {selectedAttempt && <>
        <div className="training-attempt-status"><StatusBadge status={selectedAttempt.status} /><Progress value={selectedAttempt.node_summary?.progress || 0} /><span>{selectedAttempt.node_summary?.completed || 0}/{selectedAttempt.node_summary?.total || 0} 节点</span>{activeStates.has(selectedAttempt.status) && <button className="button danger small" disabled={cancel.isPending || selectedAttempt.status === "cancelling"} onClick={() => selectedAttempt.run_kind === "imported" && selectedInstance.source_refs?.task_id ? control.mutate({ taskId: selectedInstance.source_refs.task_id, action: "cancel" }) : cancel.mutate(selectedAttempt)}><Square size={14} />{selectedAttempt.status === "cancelling" ? "正在停止" : "停止执行"}</button>}</div>
        <p className="training-muted">{executionScopeLabel(selectedAttempt)}</p>
        {selectedAttempt.error?.message && <p className="form-error" role="alert">{selectedAttempt.error.message}</p>}
        {selectedAttempt.node_runs.filter((n) => n.task && (n.status === "paused" || (n.task.payload?.command?.control?.safe_pause?.protocol === "forgelm-command-control-file-v1" && n.task.observed_state === "running"))).map((n) => <div key={n.id} className="training-node-controls"><span>{n.label}</span><button className="button secondary small" disabled={control.isPending} onClick={() => control.mutate({ taskId: n.task_id!, action: n.status === "paused" ? "resume" : "pause" })}>{n.status === "paused" ? "继续执行" : "安全暂停"}</button></div>)}
        <div className="pipeline-tabs" role="tablist" aria-label="Attempt 视图"><button role="tab" aria-selected={!showingAttemptGraph} className={!showingAttemptGraph ? "active" : ""} onClick={() => openAttempt(selectedAttempt.id, "details")}>执行详情</button><button role="tab" aria-selected={showingAttemptGraph} className={showingAttemptGraph ? "active" : ""} onClick={() => openAttempt(selectedAttempt.id, "graph")}>执行图与血缘</button></div>
        {showingAttemptGraph ? <ReactFlowProvider><TrainingCanvas key={selectedAttempt.id} value={selectedInstance} catalog={catalog.data || []} readOnly attempt={selectedAttempt} onExecuteNode={executeNode} onTask={openTask} onTrainingRun={openTrainingRun} onOpenHistory={openHistory} /></ReactFlowProvider> : <TrainingAttemptOverview attempt={selectedAttempt} runs={associatedRuns} onOpenRun={openAssociatedRun} onTask={openTask} onOpenGraph={() => openAttempt(selectedAttempt.id, "graph")} />}
      </>}
    </>}
    {modal}
  </div>;

  if (oldRunId) return <Panel><p>正在打开训练实例…</p><ErrorMessage error={resolvedRun.error} /></Panel>;
  const entries = (instances.data || []).filter((item) => `${item.name} ${inputSummary(item)} ${item.id} ${item.description || ""}`.toLowerCase().includes(search.toLowerCase())).sort((a, b) => b.created_at.localeCompare(a.created_at));
  return <div className="training-page">
    <PageHeader eyebrow="TRAINING" title="训练" description="冻结一次实验的完整配置，跟踪每次执行、Checkpoint 与模型产物。" actions={<button className="button primary" onClick={() => newDraft()}><Plus size={17} />新建训练</button>} />
    <div className="training-list-toolbar"><label className="training-search"><Search size={16} /><input aria-label="搜索训练实例" placeholder="搜索训练名称、阶段或 ID" value={search} onChange={(event) => setSearch(event.target.value)} /></label><label className="training-archive-filter"><input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />显示已归档</label><span>{entries.length} 条训练记录</span></div>
    <ErrorMessage error={comparisonDraft.error || copyInstance.error || instances.error || legacyRuns.error || execute.error} />
    {instances.isPending ? <Panel>正在读取训练实例…</Panel> : !entries.length ? <EmptyState title={search ? "没有匹配的训练实例" : "创建第一次训练实验"} description="配置训练节点与依赖，创建后保留完整不可变快照。后续修改通过复制实例完成。" /> : <section className="training-instance-list" aria-label="训练实例列表"><header><span>训练实例 · 关键输入</span><span>最近执行 · 指标与产物</span><span>创建时间</span><span /></header>{entries.map((item) => {
      const metrics = item.metrics_summary;
      return <article key={item.id} className="training-instance-row"><button className="training-instance-open" onClick={() => openInstance(item.id)}><div className="pipeline-definition-icon"><Activity size={18} /></div><div><strong>{item.name}</strong><p>{inputSummary(item)}</p><small>{item.status === "archived" ? "已归档" : item.locked ? "配置已冻结" : "历史编辑草稿"}{item.cloned_from_instance_id ? ` · 复制自 ${item.cloned_from_instance_id.slice(0, 8)}` : ""}</small></div></button><div>{item.latest_run ? <><StatusBadge status={item.latest_run.status} /><small>{item.run_count} 次执行</small></> : <span className="training-muted">尚未执行</span>}{metrics && <small>Step {formatNumber(metrics.step || 0)} · Loss {metrics.loss ?? "—"}</small>}{Boolean(item.output_summary?.checkpoints) && <small>{item.output_summary!.checkpoints} 个 Checkpoint</small>}</div><time>{formatDate(item.created_at)}</time><div className="training-row-actions"><button className="icon-button" aria-label={`复制 ${item.name}`} onClick={() => copyInstance.mutate(item)}><Copy size={16} /></button><button className="button ghost small" onClick={() => openInstance(item.id)}>查看详情</button></div></article>;
    })}</section>}
    {modal}
  </div>;
}
