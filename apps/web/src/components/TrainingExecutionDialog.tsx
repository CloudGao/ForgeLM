import { useMutation, useQuery } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useState } from "react";
import { formatDate, post } from "../api";
import type { Json, ProjectPipelineInstance, ProjectPipelineRun } from "../types";
import { ErrorMessage, Field, Modal, StatusBadge } from "./ui";

export type ExecutionSelectionMode = "from_node" | "selected";
type ExecutionPreview = {
  valid: boolean;
  selected_node_ids: string[];
  reused_node_ids: string[];
  excluded_node_ids: string[];
  reused_nodes: { node_id: string; label: string; status: string; source_attempt_id: string; outputs: Json }[];
  errors: { code: string; message: string; node_id?: string }[];
  warnings: string[];
};

export function executionScopeLabel(run: ProjectPipelineRun) {
  const scope = run.context_snapshot?.__training_instance?.execution_scope;
  const count = scope?.selected_node_ids?.length || 0;
  const range = scope?.selection_mode === "from_node" ? `从节点继续 · ${count} 个节点`
    : scope?.selection_mode === "selected" ? `选定执行 · ${count} 个节点` : "全量执行";
  return run.trigger_kind === "retry" ? `恢复执行 · ${range}` : range;
}

export function TrainingExecutionDialog({ instance, attempts, initialMode, initialNodeId, initialSourceId, onClose, onStarted }: {
  instance: ProjectPipelineInstance;
  attempts: ProjectPipelineRun[];
  initialMode: ExecutionSelectionMode;
  initialNodeId?: string;
  initialSourceId?: string;
  onClose: () => void;
  onStarted: (run: ProjectPipelineRun) => void;
}) {
  const sources = attempts.filter((run) => ["succeeded", "failed", "cancelled"].includes(run.status) && run.run_kind !== "imported");
  const [mode, setMode] = useState(initialMode);
  const [selected, setSelected] = useState<string[]>(initialNodeId ? [initialNodeId] : []);
  const [startNode, setStartNode] = useState(initialNodeId || instance.nodes[0]?.id || "");
  const [sourceId, setSourceId] = useState(initialSourceId || sources[0]?.id || "");
  const nodeIds = mode === "from_node" ? [startNode].filter(Boolean) : selected;
  const request = { selection_mode: mode, node_ids: nodeIds, source_attempt_id: sourceId || null };
  const base = `/projects/${instance.project_id}/training-instances/${instance.id}/attempts`;
  const preview = useQuery({
    queryKey: ["training-execution-preview", instance.id, mode, nodeIds, sourceId],
    queryFn: () => post<ExecutionPreview>(`${base}/preview`, request),
    enabled: nodeIds.length > 0,
    retry: false,
    staleTime: 0,
  });
  const execute = useMutation({ mutationFn: () => post<ProjectPipelineRun>(base, request), onSuccess: onStarted });
  const labels = new Map(instance.nodes.map((node) => [node.id, node.label]));
  const plan = nodeIds.length ? preview.data : undefined;
  return <Modal title="选择执行范围" description="在当前训练实例下创建新的 Attempt，配置保持冻结。" onClose={onClose} wide>
    <div className="training-execution-dialog">
      <div className="form-grid two-columns">
        <Field label="执行方式"><select aria-label="执行方式" value={mode} disabled={execute.isPending} onChange={(event) => setMode(event.target.value as ExecutionSelectionMode)}><option value="from_node">从指定节点向后执行</option><option value="selected">只运行选中节点</option></select></Field>
        <Field label="上游结果来源" hint="范围外依赖复用该次执行的冻结输出；没有依赖时可不选。"><select aria-label="上游结果来源" value={sourceId} disabled={execute.isPending} onChange={(event) => setSourceId(event.target.value)}><option value="">不使用历史输出</option>{sources.map((run) => <option key={run.id} value={run.id}>Attempt {run.context_snapshot?.__training_instance?.attempt_number || attempts.length - attempts.indexOf(run)} · {formatDate(run.created_at)} · {run.id.slice(0, 8)}</option>)}</select></Field>
      </div>
      {mode === "from_node" ? <Field label="起始节点" hint="包含该节点和所有可达的下游节点。"><select aria-label="起始节点" value={startNode} disabled={execute.isPending} onChange={(event) => setStartNode(event.target.value)}>{instance.nodes.map((node) => <option key={node.id} value={node.id}>{node.label} · {node.id}</option>)}</select></Field>
        : <fieldset className="training-node-picker"><legend>勾选要运行的节点</legend>{instance.nodes.map((node) => <label key={node.id}><input type="checkbox" checked={selected.includes(node.id)} disabled={execute.isPending} onChange={(event) => setSelected(event.target.checked ? [...selected, node.id] : selected.filter((id) => id !== node.id))} /><span><strong>{node.label}</strong><small>{node.id}</small></span></label>)}</fieldset>}
      {!nodeIds.length && <p className="training-muted">请至少选择一个节点。</p>}
      {nodeIds.length > 0 && preview.isFetching && <p role="status">正在检查执行范围和上游结果…</p>}
      <ErrorMessage error={preview.error || execute.error} />
      {plan && <section className="training-execution-preview" aria-label="执行范围预览">
        <h3>本次执行 {plan.selected_node_ids.length} 个节点</h3>
        <p>{plan.selected_node_ids.map((id) => labels.get(id) || id).join("、")}</p>
        {plan.reused_nodes.length > 0 && <><h4>复用的上游结果</h4>{plan.reused_nodes.map((node) => <details key={node.node_id}><summary>{node.label} <StatusBadge status={node.status} /> · 来源 {node.source_attempt_id.slice(0, 8)}</summary><pre>{JSON.stringify(node.outputs, null, 2)}</pre></details>)}</>}
        <p className="training-muted">{plan.excluded_node_ids.length} 个节点不参与本次执行。范围内节点将重新执行；恢复失败进度请使用执行记录中的“重试并恢复”。</p>
        {plan.warnings.map((warning, index) => <p key={index} className="training-muted">{warning}</p>)}
        {plan.errors.map((error, index) => <p className="form-error" role="alert" key={index}>{error.message}</p>)}
      </section>}
      <div className="form-actions"><button className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={!instance.can_run || !plan?.valid || preview.isFetching || Boolean(preview.error) || execute.isPending} onClick={() => execute.mutate()}><Play size={15} />{execute.isPending ? "正在创建…" : "创建 Attempt 并执行"}</button></div>
    </div>
  </Modal>;
}
