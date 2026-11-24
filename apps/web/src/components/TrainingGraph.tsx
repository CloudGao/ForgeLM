import { Handle, MarkerType, Position, type Edge, type Node, type NodeProps } from "@xyflow/react";
import { useState } from "react";
import { DataGenerationFields } from "./DataGenerationFields";
import { SuiteCanvasFields } from "./SuiteCanvasFields";
import "@xyflow/react/dist/style.css";
import { Blend, Bot, Braces, CircleDotDashed, Database, GitBranch, GitMerge, Rocket, ScanSearch, Settings2, Shuffle, Sparkles, TestTubeDiagonal, Trash2, WandSparkles } from "lucide-react";
import { Field, StatusBadge } from "./ui";
import { DisplayHandles, type DisplayPort } from "./TrainingGraphLayout";
import type { Json, LineageAssetRelation, LifecycleVersionRef, PipelineNodeCatalogItem, PipelineNodeKind, ProjectPipelineDagEdge, ProjectPipelineDagNode, ProjectPipelineNodeRun, ProjectPipelineOptions, ProjectPipelineRun } from "../types";

export type PipelineNodeData = {
  kind: PipelineNodeKind;
  label: string;
  description: string;
  config: Json;
  catalog?: PipelineNodeCatalogItem;
  runStatus?: string;
  versionRefs?: LifecycleVersionRef[];
  lineageDetail?: string;
  executionLabel?: string;
  contextInput?: boolean;
  displayPorts?: DisplayPort[];
};

type PipelineNodePresentation = {
  versions: LifecycleVersionRef[];
  detail?: string;
};

export type FlowNode = Node<PipelineNodeData, "pipelineNode">;

export const nodeIcons: Record<PipelineNodeKind, typeof Database> = {
  translation: WandSparkles,
  fact_translation: WandSparkles,
  knowledge: Sparkles,
  selection: ScanSearch,
  preparation: Settings2,
  data: Database,
  transform: WandSparkles,
  clean: Sparkles,
  mixture: Blend,
  tokenizer: Braces,
  training: Bot,
  evaluation: TestTubeDiagonal,
  check: ScanSearch,
  custom: Settings2,
  branch: GitBranch,
  join: GitMerge,
  quantization: Shuffle,
  release: Rocket,
};

const taskNodeKinds = new Set<PipelineNodeKind>([
  "transform",
  "clean",
  "mixture",
  "tokenizer",
  "training",
  "evaluation",
  "custom",
  "quantization",
  "release",
]);

export const defaultAdapters: Partial<Record<PipelineNodeKind, string>> = {
  transform: "builtin.command.data",
  clean: "builtin.command.data",
  mixture: "builtin.command.data",
  tokenizer: "builtin.command.tokenizer",
  training: "builtin.command.training",
  evaluation: "builtin.command.evaluation",
  custom: "builtin.command.data",
  quantization: "builtin.command.quantization",
  release: "builtin.command.data",
};

const versionKindLabels: Record<string, string> = {
  dataset_version: "数据版本",
  mixture_version: "混合版本",
  supervision_dataset_version: "监督数据版本",
  supervision_mixture_version: "监督混合版本",
  training_data_view_version: "Training View",
  tokenizer_version: "Tokenizer",
  code_version: "代码版本",
  plan_revision: "训练计划",
  training_run: "训练运行",
  checkpoint: "Checkpoint",
  model_version: "模型版本",
  evaluation_run: "评测运行",
  quantization_run: "量化运行",
  release: "发布版本",
};

function normalizedVersionRefs(value: unknown): LifecycleVersionRef[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const candidate = item as Partial<LifecycleVersionRef>;
    if (!candidate.id || !candidate.label) return [];
    return [{
      kind: String(candidate.kind || "version"),
      id: String(candidate.id),
      label: String(candidate.label),
      ...(candidate.version !== undefined ? { version: candidate.version } : {}),
      ...(candidate.status ? { status: String(candidate.status) } : {}),
      ...(candidate.metadata ? { metadata: candidate.metadata } : {}),
    }];
  });
}

function lineageVersionRef(item: LineageAssetRelation): LifecycleVersionRef {
  const version = item.version ? String(item.version) : "";
  const marker = item.asset_type === "plan_revision" ? `R${version}` : `V${version}`;
  const normalizedLabel = item.label.replaceAll(" ", "").toLowerCase();
  const label = version && item.asset_type !== "checkpoint" && !normalizedLabel.includes(marker.toLowerCase())
    ? `${item.label} · ${marker}`
    : item.label;
  return {
    kind: item.asset_type,
    id: item.asset_id,
    label,
    ...(version ? { version } : {}),
    ...(item.status ? { status: item.status } : {}),
    ...(item.metadata ? { metadata: item.metadata } : {}),
  };
}

export function runPresentations(run?: ProjectPipelineRun): Map<string, PipelineNodePresentation> {
  const result = new Map<string, PipelineNodePresentation>();
  for (const node of run?.node_runs || []) {
    const stored = node.lineage
      ? [...node.lineage.consumed_assets, ...node.lineage.produced_assets].map(lineageVersionRef)
      : [];
    const versions = stored.length ? stored : normalizedVersionRefs(node.output_snapshot?.versions);
    const detail = node.lineage
      ? `已保存 ${node.lineage.consumed_assets.length} 个消费关系与 ${node.lineage.produced_assets.length} 个产出关系`
      : typeof node.output_snapshot?.detail === "string" ? node.output_snapshot.detail : undefined;
    if (versions.length || detail) result.set(node.node_id, { versions, detail });
  }
  return result;
}

function executionSummary(config: Json | null | undefined): string | undefined {
  if (!config) return undefined;
  const command = config.command || {};
  const parts = [config.adapter_id, command.executable].filter(Boolean).map(String);
  if (config.recipe_version_id) parts.push(`配方 ${String(config.recipe_version_id).slice(0, 8)}`);
  if (config.code_version_id) parts.push(`代码 ${String(config.code_version_id).slice(0, 8)}`);
  return parts.length ? parts.join(" · ") : undefined;
}

export function toFlowNodes(nodes: ProjectPipelineDagNode[], catalog: PipelineNodeCatalogItem[], statuses?: Map<string, string>, presentations?: Map<string, PipelineNodePresentation>): FlowNode[] {
  const catalogMap = new Map(catalog.map((item) => [item.type, item]));
  return nodes.map((node) => ({
    id: node.id,
    type: "pipelineNode",
    position: node.position,
    data: {
      kind: node.type,
      label: node.label,
      description: node.description || "",
      config: node.config || {},
      catalog: catalogMap.get(node.type),
      runStatus: statuses?.get(node.id),
      versionRefs: presentations?.get(node.id)?.versions || [],
      lineageDetail: presentations?.get(node.id)?.detail,
      executionLabel: executionSummary(node.config),
    },
  }));
}

export function toFlowEdges(edges: ProjectPipelineDagEdge[]): Edge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.source_handle || undefined,
    targetHandle: edge.target_handle || undefined,
    label: edge.label || undefined,
    markerEnd: { type: MarkerType.ArrowClosed },
    className: "pipeline-flow-edge",
  }));
}

export function fromFlowNodes(nodes: FlowNode[]): ProjectPipelineDagNode[] {
  return nodes.map((node) => ({
    id: node.id,
    type: node.data.kind,
    position: { x: node.position.x, y: node.position.y },
    label: node.data.label,
    description: node.data.description,
    config: node.data.config || {},
  }));
}

export function fromFlowEdges(edges: Edge[]): ProjectPipelineDagEdge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    source_handle: edge.sourceHandle || null,
    target_handle: edge.targetHandle || null,
    label: typeof edge.label === "string" ? edge.label : "",
  }));
}

function PipelineGraphNode({ data, selected }: NodeProps<FlowNode>) {
  const Icon = nodeIcons[data.kind] || CircleDotDashed;
  const catalog = data.catalog;
  const inputs = catalog?.inputs || (data.kind === "data" ? [] : [{ id: "in" }]);
  const outputs = catalog?.outputs || [{ id: "out" }];
  const color = catalog?.color || "#64748b";
  return (
    <>
    {data.displayPorts && <DisplayHandles ports={data.displayPorts} />}
    <article
      className={`pipeline-graph-node ${selected ? "is-selected" : ""} ${data.runStatus ? `is-${data.runStatus}` : ""}`}
      style={{ "--node-accent": color } as React.CSSProperties}
      aria-label={`${data.label} · ${catalog?.label || data.kind}`}
    >
      {!data.displayPorts && data.contextInput && <Handle id="context-in" type="target" position={Position.Left} isConnectable={false} style={{ opacity: 0 }} />}
      {!data.displayPorts && inputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="target"
          position={Position.Left}
          style={{ top: `${((index + 1) / (inputs.length + 1)) * 100}%` }}
          className="pipeline-handle"
        />
      ))}
      <div className="pipeline-node-icon"><Icon size={18} /></div>
      <div className="pipeline-node-copy">
        <span>{catalog?.label || data.kind}</span>
        <strong>{data.label}</strong>
        {data.description && !data.versionRefs?.length && <small>{data.description}</small>}
      </div>
      {data.runStatus && <StatusBadge status={data.runStatus} />}
      {Boolean(data.versionRefs?.length) && <div className="pipeline-node-lineage" aria-label={`${data.label} 的精确版本`}>
        {data.versionRefs!.slice(0, 3).map((version) => <div className="pipeline-node-version" key={`${version.kind}:${version.id}`}>
          <strong>{version.label}</strong>
          <small>{versionKindLabels[version.kind] || version.kind}{version.status ? ` · ${version.status}` : ""}</small>
        </div>)}
        {data.versionRefs!.length > 3 && <small className="pipeline-node-version-more">另有 {data.versionRefs!.length - 3} 个版本</small>}
      </div>}
      {data.executionLabel && <div className="pipeline-node-execution"><Braces size={12} /><span>{data.executionLabel}</span></div>}
      {!data.displayPorts && outputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="source"
          position={Position.Right}
          style={{ top: `${((index + 1) / (outputs.length + 1)) * 100}%` }}
          className="pipeline-handle"
        />
      ))}
    </article>
    </>
  );
}

export function LineageVersionList({ versions, detail, title = "绑定与产出版本" }: { versions: LifecycleVersionRef[]; detail?: string; title?: string }) {
  if (!versions.length && !detail) return null;
  return <section className="pipeline-lineage-facts"><header><strong>{title}</strong>{detail && <span>{detail}</span>}</header>{versions.length ? <div>{versions.map((version) => <article key={`${version.kind}:${version.id}`}><div><strong>{version.label}</strong><small>{versionKindLabels[version.kind] || version.kind}</small></div>{version.status && <StatusBadge status={version.status} />}</article>)}</div> : <p>该节点没有登记结构化版本产出。</p>}</section>;
}

export function LineageExecutionFacts({ node }: { node: ProjectPipelineNodeRun }) {
  const lineage = node.lineage;
  if (!lineage) return null;
  const refs = (items: typeof lineage.consumed_assets): LifecycleVersionRef[] => items.map(lineageVersionRef);
  const identity = lineage.execution_identity || {};
  const identityParts = [
    identity.adapter_id,
    identity.launcher_id,
    identity.code_version_id ? `代码 ${String(identity.code_version_id).slice(0, 8)}` : "",
    identity.plan_revision_id ? `计划 ${String(identity.plan_revision_id).slice(0, 8)}` : "",
    identity.training_run_id ? `训练运行 ${String(identity.training_run_id).slice(0, 8)}` : "",
  ].filter(Boolean).map(String);
  return <section className="pipeline-lineage-ledger"><header><div><strong>已持久化血缘</strong><span>不可变事实 · {lineage.content_hash.slice(0, 10)}</span></div><StatusBadge status="registered" /></header>{identityParts.length > 0 && <p className="pipeline-lineage-identity"><Braces size={13} />{identityParts.join(" · ")}</p>}<LineageVersionList versions={refs(lineage.consumed_assets)} title="实际消费" /><LineageVersionList versions={refs(lineage.produced_assets)} title="实际产出" />{!lineage.consumed_assets.length && !lineage.produced_assets.length && <p>已保存执行身份；该节点没有登记结构化资产关系。</p>}</section>;
}

export const nodeTypes = { pipelineNode: PipelineGraphNode };

export function ReadOnlyNodeInspector({ node }: { node: FlowNode }) {
  const Icon = nodeIcons[node.data.kind];
  return <><header><div><span>冻结节点</span><h2><Icon size={17} />{node.data.catalog?.label || node.data.kind}</h2></div></header><div className="pipeline-inspector-body"><dl className="pipeline-run-node-facts"><div><dt>节点名称</dt><dd>{node.data.label}</dd></div><div><dt>稳定 ID</dt><dd>{node.id}</dd></div>{node.data.executionLabel && <div><dt>执行身份</dt><dd>{node.data.executionLabel}</dd></div>}</dl><LineageVersionList versions={node.data.versionRefs || []} detail={node.data.lineageDetail} />{!node.data.versionRefs?.length && node.data.runStatus === "succeeded" && <div className="pipeline-inspector-note"><CircleDotDashed size={16} /><p>该节点已完成，但没有登记结构化版本产出；请查看任务输出契约。</p></div>}<details><summary>冻结配置</summary><pre>{JSON.stringify(node.data.config || {}, null, 2)}</pre></details></div></>;
}

export function NodeInspector({ node, options, onChange, onDelete }: { node: FlowNode; options?: ProjectPipelineOptions; onChange: (patch: Partial<PipelineNodeData>) => void; onDelete: () => void }) {
  const Icon = nodeIcons[node.data.kind];
  const config = node.data.config || {};
  function updateConfig(patch: Json) { onChange({ config: { ...config, ...patch } }); }
  function parseValue(value: string) { try { return JSON.parse(value); } catch { return value; } }
  return (
    <>
      <header><div><span>节点配置</span><h2><Icon size={17} />{node.data.catalog?.label || node.data.kind}</h2></div><button className="icon-button danger-icon" aria-label="删除节点" onClick={onDelete}><Trash2 size={16} /></button></header>
      <div className="pipeline-inspector-body">
        <Field label="节点名称"><input value={node.data.label} onChange={(event) => onChange({ label: event.target.value })} /></Field>
        <Field label="说明"><textarea value={node.data.description} onChange={(event) => onChange({ description: event.target.value })} /></Field>
        <div className="pipeline-node-id"><span>稳定节点 ID</span><code>{node.id}</code></div>
        {config.execution_mode === "suite" && <SuiteCanvasFields config={config} onChange={updateConfig} />}
        {!["translation", "fact_translation", "knowledge"].includes(node.data.kind) && !!node.data.catalog?.native_operations?.length && <Field label="节点操作"><select value={config.native_operation || ""} onChange={(event) => onChange({ config: event.target.value ? { execution_mode: "native", native_operation: event.target.value, parameters: {}, input_bindings: {} } : { execution_mode: "command" } })}><option value="">{node.data.kind === "preparation" ? "请选择内置操作" : "自定义命令"}</option>{node.data.catalog.native_operations.map((op) => <option key={op.id} value={op.id}>{op.label}</option>)}</select></Field>}
        {["translation", "fact_translation", "knowledge"].includes(node.data.kind) && <DataGenerationFields kind={node.data.kind} config={config} options={options} onChange={updateConfig} />}
        {config.native_operation && !["translation", "fact_translation", "knowledge"].includes(node.data.kind) && <NativeDomainFields key={`${node.id}:${config.native_operation}`} config={config} catalog={node.data.catalog} onChange={updateConfig} />}
        {config.native_operation && <Field label="输出保留策略" hint="自动管理继承项目策略；下游结束后仍会检查读取租约、宽限期和可靠恢复副本。"><select value={config.output_retention || "auto"} onChange={(e) => updateConfig({ output_retention: e.target.value })}><option value="auto">自动管理</option><option value="after_consumers">下游完成后回收可恢复副本</option><option value="retain">长期保留</option></select></Field>}
        {node.data.kind === "data" && (
          <Field label="精确数据版本" hint="只列出项目已经绑定或拥有的数据版本">
            <select value={config.asset_type && config.asset_version_id ? `${config.asset_type}:${config.asset_version_id}` : ""} onChange={(event) => { const separator = event.target.value.indexOf(":"); updateConfig(separator > 0 ? { asset_type: event.target.value.slice(0, separator), asset_version_id: event.target.value.slice(separator + 1) } : { asset_type: "", asset_version_id: "" }); }}>
              <option value="">请选择数据版本</option>
              {(options?.data_versions || []).map((item) => <option key={`${item.asset_type}:${item.asset_version_id}`} value={`${item.asset_type}:${item.asset_version_id}`}>{item.label} · {item.status}</option>)}
            </select>
          </Field>
        )}
        {node.data.kind === "training" && <Field label="训练执行方式"><select value={config.execution_mode || "command"} onChange={(event) => updateConfig({ execution_mode: event.target.value })}><option value="native">结构化训练</option><option value="command">自定义命令</option></select></Field>}
        {taskNodeKinds.has(node.data.kind) && config.execution_mode !== "native" && config.execution_mode !== "suite" && <><Field label="代码版本"><select value={config.code_version_id || ""} onChange={(event) => updateConfig({ code_version_id: event.target.value || null })}><option value="">使用完整内联命令</option>{options?.code_versions?.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></Field><CommandNodeFields kind={node.data.kind} config={config} onChange={updateConfig} /><Field label="输出制品" hint="每行 kind=相对路径，执行后校验并登记内容版本"><textarea value={(config.command?.outputs || []).map((output: Json) => `${output.kind}=${output.path}`).join("\n")} onChange={(event) => updateConfig({ command: { ...config.command, outputs: event.target.value.split("\n").filter(Boolean).map((line) => { const i = line.indexOf("="); return { kind: i > 0 ? line.slice(0, i).trim() : "artifact", path: i > 0 ? line.slice(i + 1).trim() : line.trim() }; }) } })} /></Field></>}
        {["check", "branch"].includes(node.data.kind) && (
          <>
            <Field label="取值路径" hint="例如 upstream.check.passed"><input value={config.path || ""} onChange={(event) => updateConfig({ path: event.target.value })} placeholder="upstream.node-id.field" /></Field>
            <Field label="比较方式"><select value={config.operator || "truthy"} onChange={(event) => updateConfig({ operator: event.target.value })}><option value="truthy">为真</option><option value="falsy">为假</option><option value="equals">等于</option><option value="not_equals">不等于</option><option value="contains">包含</option><option value="gt">大于</option><option value="gte">大于等于</option><option value="lt">小于</option><option value="lte">小于等于</option></select></Field>
            {!(["truthy", "falsy"].includes(config.operator || "truthy")) && <Field label="比较值" hint="支持 JSON 数字、布尔和字符串"><input value={typeof config.value === "string" ? config.value : JSON.stringify(config.value ?? "")} onChange={(event) => updateConfig({ value: parseValue(event.target.value) })} /></Field>}
          </>
        )}
        {node.data.kind === "check" && <><Field label="未通过时"><select value={config.failure_policy || "continue"} onChange={(event) => updateConfig({ failure_policy: event.target.value })}><option value="continue">输出结果，继续交给分支处理</option><option value="fail">结束本次执行</option></select></Field><Field label="严重级别"><select value={config.severity || "warning"} onChange={(event) => updateConfig({ severity: event.target.value })}><option value="warning">警告</option><option value="error">错误</option><option value="info">信息</option></select></Field></>}
        {node.data.kind === "branch" && <div className="pipeline-branch-help"><GitBranch size={16} /><p>右侧三个出口分别是通过、未通过和默认。至少连接其中两个出口才能创建实例。</p></div>}
        {node.data.kind === "join" && <div className="pipeline-inspector-note"><GitMerge size={16} /><p>汇合节点只等待实际被分支选择的上游路径，未选择路径会自动跳过。</p></div>}
      </div>
    </>
  );
}

function NativeDomainFields({ config, catalog, onChange }: { config: Json; catalog?: PipelineNodeCatalogItem; onChange: (patch: Json) => void }) {
  const [parameters, setParameters] = useState(JSON.stringify(config.parameters || {}, null, 2));
  const [bindings, setBindings] = useState(JSON.stringify(config.input_bindings || {}, null, 2));
  const [error, setError] = useState("");
  const operation = catalog?.native_operations?.find((op) => op.id === config.native_operation);
  function save(key: string, value: string) {
    try { const parsed = JSON.parse(value); if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("请输入 JSON 对象"); onChange({ [key]: parsed }); setError(""); }
    catch (e) { setError(e instanceof Error ? e.message : "JSON 格式错误"); }
  }
  return <><Field label="领域参数" hint="使用系统领域 API 的结构化参数；资源请求可指定 AutoDL Worker。"><textarea rows={14} value={parameters} onChange={(e) => { setParameters(e.target.value); save("parameters", e.target.value); }} /></Field><Field label="上游输出绑定" hint={'目标参数路径 → {"node_id":"上游节点 ID","output_path":"输出字段"}；对应节点必须连线。'}><textarea rows={8} value={bindings} onChange={(e) => { setBindings(e.target.value); save("input_bindings", e.target.value); }} /></Field>{error && <p role="alert">{error}</p>}<p>输出字段：{operation?.output_key}</p><details><summary>参数契约</summary><pre>{JSON.stringify(operation?.input_schema || {}, null, 2)}</pre></details></>;
}

function CommandNodeFields({ kind, config, onChange }: { kind: PipelineNodeKind; config: Json; onChange: (patch: Json) => void }) {
  const command = config.command || {};
  const resources = config.resource_request || {};
  const updateCommand = (patch: Json) => onChange({ command: { ...command, ...patch } });
  return <><Field label="执行适配器"><select value={config.adapter_id || defaultAdapters[kind] || "builtin.command.data"} onChange={(event) => onChange({ adapter_id: event.target.value })}><option value="builtin.command.data">通用数据命令</option><option value="builtin.command.tokenizer">Tokenizer 命令</option><option value="builtin.command.training">训练命令</option><option value="builtin.command.evaluation">评测命令</option><option value="builtin.command.quantization">量化命令</option></select></Field><Field label="可执行文件" hint="无 Shell 参数化执行；可用 __worker_python__ 指向 Worker Python"><input value={command.executable || ""} onChange={(event) => updateCommand({ executable: event.target.value })} placeholder="python / torchrun / 可执行文件路径" /></Field><Field label="命令参数" hint="每行一个参数"><textarea value={(command.args || []).join("\n")} onChange={(event) => updateCommand({ args: event.target.value.split("\n").filter((item) => item.length) })} placeholder="-m&#10;my_training.entrypoint" /></Field><Field label="工作目录（可选）"><input value={command.cwd || ""} onChange={(event) => updateCommand({ cwd: event.target.value || null })} /></Field><div className="form-grid two-columns compact"><Field label="CPU"><input type="number" min="0" value={resources.cpus ?? 1} onChange={(event) => onChange({ resource_request: { ...resources, cpus: Number(event.target.value) } })} /></Field><Field label="GPU"><input type="number" min="0" value={resources.gpus ?? 0} onChange={(event) => onChange({ resource_request: { ...resources, gpus: Number(event.target.value) } })} /></Field></div></>;
}
