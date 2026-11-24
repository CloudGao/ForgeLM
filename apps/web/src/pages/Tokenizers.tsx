import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowRight, Braces, Database, FlaskConical, LockKeyhole, Pencil, Plus, RefreshCw, RotateCcw, Settings2, ShieldCheck, Star } from "lucide-react";
import { FormEvent, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, formatDate, post } from "../api";
import { Alert, EmptyState, ErrorMessage, ExecutionFields, executionPayload, Field, FormActions, Modal, PageHeader, Panel, StatusBadge } from "../components/ui";
import type { ChatTemplate, Dataset, Mixture, Task } from "../types";

interface TokenizedDataset {
  id: string;
  status: string;
  task_id?: string | null;
  artifact_id?: string | null;
  dataset_version_id: string;
  dataset_name?: string | null;
  dataset_version?: number | null;
  dataset_token_estimate?: number | null;
  tokenizer_version_id: string;
  tokenizer_name?: string | null;
  tokenizer_version?: number | null;
  selection_config: Record<string, any>;
  encoding_config: Record<string, any>;
  document_count: number;
  token_count: number;
  shard_count: number;
  failure_reason?: string | null;
  format?: string;
  identity_hash?: string;
  created_at: string;
}

const TOKENIZER_GATE_VERSION = "tokenizer-quality-gate-v1";
type TokenizerGateDefinition = { metric: string; label: string; group: string; operator: "gte" | "lte"; symbol: "≥" | "≤"; defaultEnabled: boolean; defaultThreshold: number; min: number; max: number; step: number; hint: string; custom?: boolean };
const TOKENIZER_GATE_DEFINITIONS: TokenizerGateDefinition[] = [
  { metric: "round_trip", label: "Round-trip 成功率", group: "可逆性与格式", operator: "gte", symbol: "≥", defaultEnabled: true, defaultThreshold: 1, min: 0, max: 1, step: 0.001, hint: "编码后解码与规范化文本完全一致的文档比例" },
  { metric: "newline_round_trip", label: "换行保真率", group: "可逆性与格式", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 1, min: 0, max: 1, step: 0.001, hint: "含换行评测文本完整往返的比例；没有该类样本时实测值为空" },
  { metric: "indentation_round_trip", label: "缩进保真率", group: "可逆性与格式", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 1, min: 0, max: 1, step: 0.001, hint: "含行首空格或 Tab 文本完整往返的比例" },
  { metric: "special_token_integrity", label: "特殊 Token 完整性", group: "可逆性与格式", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 1, min: 0, max: 1, step: 0.001, hint: "配置的特殊 Token 均存在且能独立编码为对应 ID 的比例" },
  { metric: "unk_rate", label: "Unknown Token 比例", group: "未知词与词表", operator: "lte", symbol: "≤", defaultEnabled: true, defaultThreshold: 0, min: 0, max: 1, step: 0.0001, hint: "评测样本中 unknown token 占全部 token 的比例" },
  { metric: "byte_fallback_rate", label: "Byte fallback 比例", group: "未知词与词表", operator: "lte", symbol: "≤", defaultEnabled: false, defaultThreshold: 0, min: 0, max: 1, step: 0.0001, hint: "评测输出中 <0xNN> Byte fallback Token 的比例" },
  { metric: "vocab_garbage_rate", label: "垃圾词表比例", group: "未知词与词表", operator: "lte", symbol: "≤", defaultEnabled: false, defaultThreshold: 0, min: 0, max: 1, step: 0.0001, hint: "空 Token、替换字符、控制字符或异常超长 Token 的比例" },
  { metric: "zh_chars_per_token", label: "中文压缩率", group: "压缩效率", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 1.5, min: 0, max: 100, step: 0.01, hint: "平均每个 Token 覆盖的中文字符数" },
  { metric: "en_chars_per_token", label: "英文压缩率", group: "压缩效率", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 3, min: 0, max: 100, step: 0.01, hint: "平均每个 Token 覆盖的英文字符数" },
  { metric: "code_chars_per_token", label: "代码压缩率", group: "压缩效率", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 2, min: 0, max: 100, step: 0.01, hint: "平均每个 Token 覆盖的代码字符数" },
  { metric: "math_chars_per_token", label: "数学压缩率", group: "压缩效率", operator: "gte", symbol: "≥", defaultEnabled: false, defaultThreshold: 1.5, min: 0, max: 100, step: 0.01, hint: "含数字或数学运算符样本中平均每个 Token 覆盖的字符数" },
];

function tokenizerGateDefinitions(config: Record<string, any> = {}) {
  const storedRules = config.gate_policy?.rules || {};
  const known = new Set(TOKENIZER_GATE_DEFINITIONS.map((definition) => definition.metric));
  const historical = Object.entries(storedRules).filter(([metric]) => !known.has(metric)).map(([metric, value]) => { const rule = value as Record<string, any>; const operator = rule.operator === "lte" ? "lte" : "gte"; return { metric, label: String(rule.label || metric), group: "历史 / 自定义指标", operator, symbol: operator === "lte" ? "≤" : "≥", defaultEnabled: Boolean(rule.enabled), defaultThreshold: Number(rule.threshold || 0), min: -1_000_000_000_000, max: 1_000_000_000_000, step: 0.0001, hint: "历史版本保留的指标规则；内置 Worker 未上报时需由自定义命令提供", custom: true } as TokenizerGateDefinition; });
  return [...TOKENIZER_GATE_DEFINITIONS, ...historical];
}

function tokenizerGatePolicy(config: Record<string, any> = {}) {
  const stored = config.gate_policy || {};
  const rules = stored.rules || {};
  return {
    version: TOKENIZER_GATE_VERSION,
    rules: Object.fromEntries(tokenizerGateDefinitions(config).map((definition) => [definition.metric, {
      enabled: rules[definition.metric]?.enabled ?? definition.defaultEnabled,
      operator: definition.operator,
      threshold: rules[definition.metric]?.threshold ?? definition.defaultThreshold,
      ...(definition.custom ? { label: rules[definition.metric]?.label || definition.label } : {}),
    }])),
  };
}

function gateValue(metric: string, value: unknown) {
  if (value == null || Number.isNaN(Number(value))) return "等待实测值";
  return ["round_trip", "newline_round_trip", "indentation_round_trip", "special_token_integrity", "unk_rate", "byte_fallback_rate", "vocab_garbage_rate"].includes(metric) ? `${(Number(value) * 100).toFixed(2).replace(/\.00$/, "")}%` : Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function gateReason(reason: string | undefined) {
  return ({ disabled: "未启用", missing_metric: "执行结果缺少该指标", threshold_not_met: "低于设定下限", threshold_exceeded: "超过设定上限", passed: "达到阈值" } as Record<string, string>)[reason || ""] || "等待任务完成";
}

function tokenizerStatus(tokenizer: any, task?: Task) {
  return task?.observed_state === "succeeded" ? tokenizer.status : task?.observed_state || tokenizer.status;
}

interface TrainingDataView {
  id: string;
  status: string;
  task_id?: string | null;
  mixture_name?: string | null;
  mixture_version?: number | null;
  tokenizer_name?: string | null;
  sequence_length: number;
  requested_tokens: number;
  token_count: number;
  sequence_count: number;
  config_snapshot: Record<string, any>;
  warnings?: any[];
  failure_reason?: string | null;
  created_at: string;
}

export function Tokenizers() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [selectedTokenizerId, setSelectedTokenizerId] = useState<string | null>(null);
  const [editingTokenizerId, setEditingTokenizerId] = useState<string | null>(null);
  const [editingTaskSnapshot, setEditingTaskSnapshot] = useState<Task | undefined>();
  const [templateOpen, setTemplateOpen] = useState(false);
  const tokenizers = useQuery({ queryKey: ["tokenizers", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/tokenizers`), refetchInterval: 3000 });
  const datasets = useQuery({ queryKey: ["datasets", projectId], queryFn: () => api<Dataset[]>(`/projects/${projectId}/datasets`) });
  const mixtures = useQuery({ queryKey: ["mixtures", projectId], queryFn: () => api<Mixture[]>(`/projects/${projectId}/mixtures`) });
  const chatTemplates = useQuery({ queryKey: ["chat-templates", projectId], queryFn: () => api<ChatTemplate[]>(`/projects/${projectId}/chat-templates`) });
  const tasks = useQuery({ queryKey: ["tasks", projectId], queryFn: () => api<Task[]>(`/tasks?project_id=${projectId}`), refetchInterval: 2000 });
  const taskMap = new Map((tasks.data || []).map((task) => [task.id, task]));
  const selectedTokenizer = (tokenizers.data || []).find((item) => item.id === selectedTokenizerId);
  const selectedTask = selectedTokenizer?.task_id ? taskMap.get(selectedTokenizer.task_id) : undefined;
  const editingTokenizer = (tokenizers.data || []).find((item) => item.id === editingTokenizerId);
  const editingTask = editingTokenizer?.task_id ? taskMap.get(editingTokenizer.task_id) : undefined;
  const action = useMutation({ mutationFn: ({ id, action }: { id: string; action: string }) => post(`/tokenizers/${id}/${action}`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tokenizers", projectId] }) });
  const closeCreated = () => { queryClient.invalidateQueries({ queryKey: ["tokenizers", projectId] }); setCreateOpen(false); };
  const closeEdited = () => { queryClient.invalidateQueries({ queryKey: ["tokenizers", projectId] }); queryClient.invalidateQueries({ queryKey: ["tasks", projectId] }); setEditingTokenizerId(null); setEditingTaskSnapshot(undefined); };

  return <>
    <PageHeader eyebrow="TOKENIZER LAB" title="Tokenizer 实验室" description="训练并锁定 Tokenizer，同时维护原生后训练使用的不可变 Chat Template。训练数据制品由训练中心自动编排，并在数据中心查看。" actions={<><button className="button secondary" onClick={() => setTemplateOpen(true)}><Braces size={17} />新建 Chat Template</button><button className="button primary" onClick={() => setCreateOpen(true)}><Plus size={17} />新建实验</button></>} />
    <section className="metrics-grid four"><Metric label="实验总数" value={tokenizers.data?.length || 0} /><Metric label="已通过门禁" value={(tokenizers.data || []).filter((item) => item.status === "passed").length} /><Metric label="已锁定版本" value={(tokenizers.data || []).filter((item) => item.locked).length} /><Metric label="项目默认" value={(tokenizers.data || []).find((item) => item.is_default)?.name || "未设置"} /></section>
      {(tokenizers.data || []).length ? <section className="tokenizer-grid">{(tokenizers.data || []).map((tokenizer) => { const task = taskMap.get(tokenizer.task_id); const metrics = tokenizer.metrics || {}; const gateResults = task?.result?.gate_results || []; const enabledGates = gateResults.filter((item: any) => item.enabled); const passedGates = enabledGates.filter((item: any) => item.passed).length; return <div key={tokenizer.id} className="interactive-tokenizer" role="button" tabIndex={0} aria-label={`查看 Tokenizer 实验 ${tokenizer.name} v${tokenizer.version}`} onClick={() => setSelectedTokenizerId(tokenizer.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedTokenizerId(tokenizer.id); } }}><Panel className={tokenizer.is_default ? "featured-card" : ""}>
      <div className="tokenizer-card-head"><div className="tokenizer-symbol"><FlaskConical size={22} /></div><div><span>{tokenizer.algorithm} · {tokenizer.vocab_size.toLocaleString()} 词表</span><h2>{tokenizer.name} <small>v{tokenizer.version}</small></h2></div><StatusBadge status={tokenizerStatus(tokenizer, task)} /></div>
      <div className="tokenizer-metrics"><div><span>中文压缩</span><strong>{metrics.zh_chars_per_token ?? "—"}</strong></div><div><span>英文压缩</span><strong>{metrics.en_chars_per_token ?? "—"}</strong></div><div><span>代码压缩</span><strong>{metrics.code_chars_per_token ?? "—"}</strong></div><div><span>Round-trip</span><strong>{metrics.round_trip != null ? `${metrics.round_trip * 100}%` : "—"}</strong></div></div>
      <div className="tokenizer-checks"><span><ShieldCheck size={14} />{enabledGates.length ? `门禁 ${passedGates}/${enabledGates.length}` : "门禁等待判定"}</span><span>制品 {tokenizer.artifact_id ? tokenizer.artifact_id.slice(0, 8) : "待生成"}</span></div>
      <footer>{tokenizer.is_default ? <span className="default-label"><Star size={15} />项目默认</span> : <span className="muted">{tokenizer.locked ? "版本已锁定" : "尚未锁定"}</span>}<div>{!tokenizer.locked && tokenizer.status === "passed" && <button className="button small secondary" onClick={(event) => { event.stopPropagation(); action.mutate({ id: tokenizer.id, action: "lock" }); }}><LockKeyhole size={15} />锁定</button>}{tokenizer.status === "passed" && !tokenizer.is_default && <button className="button small" onClick={(event) => { event.stopPropagation(); action.mutate({ id: tokenizer.id, action: "set-default" }); }}><Star size={15} />设为默认</button>}</div></footer>
      </Panel></div>; })}</section> : <EmptyState title="还没有 Tokenizer 实验" description="先准备数据版本，再绑定真实 Tokenizer 命令。" action={<button className="button primary" onClick={() => setCreateOpen(true)}><Plus size={16} />新建实验</button>} />}
    <ChatTemplatePanel templates={chatTemplates.data || []} tokenizers={tokenizers.data || []} />
    {createOpen && <TokenizerModal projectId={projectId} datasets={datasets.data || []} mixtures={mixtures.data || []} onClose={() => setCreateOpen(false)} onCreated={closeCreated} />}
    {selectedTokenizer && <TokenizerDetail tokenizer={selectedTokenizer} task={selectedTask} onEdit={(task) => { setEditingTaskSnapshot(task); setEditingTokenizerId(selectedTokenizer.id); setSelectedTokenizerId(null); }} onClose={() => setSelectedTokenizerId(null)} />}
    {editingTokenizer && <TokenizerModal projectId={projectId} datasets={datasets.data || []} mixtures={mixtures.data || []} initial={editingTokenizer} initialTask={editingTaskSnapshot || editingTask} onClose={() => { setEditingTokenizerId(null); setEditingTaskSnapshot(undefined); }} onCreated={closeEdited} />}
    {templateOpen && <ChatTemplateModal projectId={projectId} onClose={() => setTemplateOpen(false)} onCreated={() => { queryClient.invalidateQueries({ queryKey: ["chat-templates", projectId] }); setTemplateOpen(false); }} />}
  </>;
}

const DEFAULT_CHAT_TEMPLATE = `{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' }}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}`;

function ChatTemplatePanel({ templates, tokenizers }: { templates: ChatTemplate[]; tokenizers: any[] }) {
  const [report, setReport] = useState<Record<string, any> | null>(null);
  const validate = useMutation({ mutationFn: ({ versionId, tokenizerId }: { versionId: string; tokenizerId: string }) => post<Record<string, any>>(`/chat-template-versions/${versionId}/validate`, { tokenizer_version_id: tokenizerId || null, messages: [{ role: "system", content: "You are helpful." }, { role: "user", content: "Hello" }, { role: "assistant", content: "Hi!" }] }), onSuccess: setReport });
  return <Panel title="Chat Template" description="每次编辑生成不可变版本；SFT 的 assistant-only mask 与 DPO prompt/completion 边界都由这里定义。">
    {templates.length ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>模板</th><th>当前版本</th><th>Assistant mask</th><th>哈希</th><th>试渲染</th></tr></thead><tbody>{templates.map((template) => { const version = template.versions.find((item) => item.id === template.current_version_id) || template.versions[0]; return <tr key={template.id}><td><strong>{template.name}</strong><span>{template.description || "—"}</span></td><td>V{version?.version || "—"}</td><td><StatusBadge status={version?.validation_report?.assistant_mask_capable ? "ready" : "warning"} /></td><td>{version?.content_hash?.slice(0, 12) || "—"}</td><td><button type="button" className="button ghost small" disabled={!version || validate.isPending} onClick={() => validate.mutate({ versionId: version.id, tokenizerId: tokenizers.find((item) => item.locked)?.id || "" })}><Braces size={13} />验证</button></td></tr>; })}</tbody></table></div> : <EmptyState title="还没有 Chat Template" description="创建模板后，SFT/DPO 计划可以直接绑定其不可变版本。" />}
    {report && <details open className="collapsed-config"><summary>最近一次模板验证</summary><pre className="json-preview">{JSON.stringify(report, null, 2)}</pre></details>}
    <ErrorMessage error={validate.error} />
  </Panel>;
}

function ChatTemplateModal({ projectId, onClose, onCreated }: { projectId: string; onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("默认对话模板");
  const [description, setDescription] = useState("用于 SFT 与 DPO 的角色标记和 assistant loss span");
  const [body, setBody] = useState(DEFAULT_CHAT_TEMPLATE);
  const create = useMutation({ mutationFn: () => post(`/projects/${projectId}/chat-templates`, { name, description, template_body: body, role_schema: { roles: ["system", "user", "assistant"] }, required_special_tokens: ["<|system|>", "<|user|>", "<|assistant|>"], assistant_mask_mode: "generation_spans_v1" }), onSuccess: onCreated });
  return <Modal title="新建 Chat Template" description="模板采用 Hugging Face/Jinja chat_template 语法；generation 区域定义 assistant-only loss。" onClose={onClose} wide><form className="form-grid" onSubmit={(event) => { event.preventDefault(); create.mutate(); }}><Field label="模板名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field><Field label="说明"><input value={description} onChange={(event) => setDescription(event.target.value)} /></Field><Field label="Jinja 模板" hint="必须用 generation/endgeneration 包围 assistant 内容"><textarea rows={12} required value={body} onChange={(event) => setBody(event.target.value)} /></Field><ErrorMessage error={create.error} /><FormActions onCancel={onClose} submitLabel={create.isPending ? "创建中…" : "创建不可变版本"} disabled={create.isPending} /></form></Modal>;
}

function Metric({ label, value }: { label: string; value: unknown }) { return <article className="metric-card"><span>{label}</span><strong>{String(value)}</strong></article>; }

function TokenizedDatasetModal({ projectId, tokenizers, datasets, initial, initialTask, onClose, onCreated }: { projectId: string; tokenizers: any[]; datasets: Dataset[]; initial?: TokenizedDataset; initialTask?: Task; onClose: () => void; onCreated: () => void }) {
  const eligibleTokenizers = tokenizers.filter((item) => item.locked && item.artifact_id);
  const eligibleDatasets = datasets.filter((item) => item.current_version?.state === "ready" && item.current_version?.train_ready && item.current_version?.artifact_id && item.current_version?.record_manifest_artifact_id);
  const hasInitialTokenizer = eligibleTokenizers.some((item) => item.id === initial?.tokenizer_version_id);
  const hasInitialDataset = eligibleDatasets.some((item) => item.current_version?.id === initial?.dataset_version_id);
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/tokenized-corpora`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const selectionMode = String(form.get("selection_mode"));
    create.mutate({
      dataset_version_id: form.get("dataset_version_id"),
      tokenizer_version_id: form.get("tokenizer_version_id"),
      selection: {
        mode: selectionMode,
        estimated_token_limit: selectionMode === "estimated_token_limit" ? Number(form.get("estimated_token_limit")) : null,
        ...(selectionMode === "mixture" ? { mixture_version_id: selection.mixture_version_id } : {}),
      },
      encoding: {
        eos_policy: form.get("eos_policy"),
        empty_document_policy: form.get("empty_document_policy"),
        token_dtype: form.get("token_dtype"),
        shard_token_limit: Number(form.get("shard_token_limit")),
      },
      ...executionPayload(form),
    });
  }
  const ready = (eligibleTokenizers.length > 0 || Boolean(initial)) && (eligibleDatasets.length > 0 || Boolean(initial));
  const selection = initial?.selection_config || initialTask?.payload?.selection || {};
  const encoding = initial?.encoding_config || initialTask?.payload?.encoding || {};
  return <Modal title={initial ? "查看并派生 Tokenized Corpus 配置" : "构建 Tokenized Corpus"} description="所有会改变编码结果的策略都在这里公开配置并进入不可变身份；实际 Token 数只记录结果，不与估算目标比较。" onClose={onClose} wide><form onSubmit={submit}>
    {initial && <div className="tokenized-edit-context"><StatusBadge status={initial.status} /><span>源版本 {initial.id.slice(0, 8)}</span><span>实际 {Number(initial.token_count).toLocaleString()} Tokens</span><strong>旧版本不会被覆盖</strong></div>}
    <div className="form-grid two-columns">
    <Field label="数据集版本"><select name="dataset_version_id" required disabled={!eligibleDatasets.length && !initial} defaultValue={initial?.dataset_version_id || ""}><option value="">请选择可训练版本</option>{initial && !hasInitialDataset && <option value={initial.dataset_version_id}>{initial.dataset_name || initial.dataset_version_id} · V{initial.dataset_version || "—"}（历史绑定）</option>}{eligibleDatasets.map((item) => <option key={item.current_version!.id} value={item.current_version!.id}>{item.name} · V{item.current_version!.version} · 估算 {Number(item.current_version!.token_estimate || 0).toLocaleString()}</option>)}</select></Field>
    <Field label="锁定 Tokenizer"><select name="tokenizer_version_id" required disabled={!eligibleTokenizers.length && !initial} defaultValue={initial?.tokenizer_version_id || ""}><option value="">请选择带制品的锁定版本</option>{initial && !hasInitialTokenizer && <option value={initial.tokenizer_version_id}>{initial.tokenizer_name || initial.tokenizer_version_id} · V{initial.tokenizer_version || "—"}（历史绑定）</option>}{eligibleTokenizers.map((item) => <option key={item.id} value={item.id}>{item.name} · V{item.version} · {Number(item.vocab_size).toLocaleString()}</option>)}</select></Field>
    <Field label="输入选择策略" hint="这是输入范围，不是输出成功门槛。"><select name="selection_mode" defaultValue={selection.mode || "all"}><option value="all">全部文档</option><option value="estimated_token_limit">按导入估算截取输入</option>{selection.mixture_version_id && <option value="mixture">沿用当前混合的筛选结果</option>}</select></Field>
    <Field label="输入估算 Token 上限" hint="仅在估算截取模式生效；真实输出可能高于或低于该估算。"><input name="estimated_token_limit" type="number" min="1" defaultValue={selection.estimated_token_limit || 100000000} /></Field>
    <Field label="EOS 策略"><select name="eos_policy" defaultValue={encoding.eos_policy || "append_if_available"}><option value="append_if_available">可用时追加；缺失则警告并继续</option><option value="append_required">必须追加；启动前缺失即报配置错误</option><option value="none">不追加 EOS</option></select></Field>
    <Field label="空文档策略"><select name="empty_document_policy" defaultValue={encoding.empty_document_policy || "skip"}><option value="skip">跳过并记录 warning</option><option value="eos_only">有 EOS 时写入仅 EOS 文档</option></select></Field>
    <Field label="Token dtype"><select name="token_dtype" defaultValue={encoding.token_dtype || "auto"}><option value="auto">自动（按词表解析为 uint16/uint32）</option><option value="uint16">固定 uint16</option><option value="uint32">固定 uint32</option></select></Field>
    <Field label="每分片 Token 上限" hint="控制单个连续 Token 文件的目标大小；超长单文档不会被拆开。"><input name="shard_token_limit" type="number" min="1048576" defaultValue={encoding.shard_token_limit || 268435456} required /></Field>
    <Field label="制品格式"><input aria-label="制品格式" value="forgelm-token-corpus-v2" readOnly /></Field>
    <Field label="文档索引"><input value="uint64 offset + uint64 length" readOnly /></Field>
    <ExecutionFields adapterId="builtin.tokenize-corpus" gpuDisabled defaults={{ resource_request: initialTask?.payload?.resource_request, command: initialTask?.payload?.command }} />
    {!ready && <div className="span-two"><Alert severity="warning" title="前置条件未满足" detail="需要至少一个带稳定 Record Manifest 的可训练数据集版本，以及一个锁定且带制品的 Tokenizer。" /></div>}
  </div><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!ready} onCancel={onClose} submitLabel={initial ? "按当前配置派生版本" : "创建并开始编码"} /></form></Modal>;
}

function TrainingDataViewModal({ projectId, tokenizers, mixtures, corpora, onClose, onCreated }: { projectId: string; tokenizers: any[]; mixtures: Mixture[]; corpora: TokenizedDataset[]; onClose: () => void; onCreated: () => void }) {
  const [checked, setChecked] = useState<string[]>([]);
  const [preflightResult, setPreflightResult] = useState<any>();
  const [frozenPayload, setFrozenPayload] = useState<any>();
  const [sequenceOrderPolicy, setSequenceOrderPolicy] = useState("block_shuffle_v1");
  const eligibleTokenizers = tokenizers.filter((item) => item.locked && item.artifact_id);
  const eligibleMixtures = mixtures.filter((item) => item.current_version?.status === "ready" && item.current_version?.manifest_artifact_id);
  const readyCorpora = corpora.filter((item) => item.status === "ready" && item.artifact_id);
  const preflight = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/training-data-views/preflight`, data), onSuccess: (result, variables) => { setPreflightResult(result); setFrozenPayload(variables); } });
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/training-data-views`, data), onSuccess: onCreated });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const data = {
      mixture_version_id: form.get("mixture_version_id"),
      tokenizer_version_id: form.get("tokenizer_version_id"),
      corpus_version_ids: checked,
      config: {
        sequence_length: Number(form.get("sequence_length")),
        requested_tokens: Number(form.get("requested_tokens")),
        consumption_policy: form.get("consumption_policy"),
        tail_policy: form.get("tail_policy"),
        padding_token_id: form.get("tail_policy") === "pad" ? Number(form.get("padding_token_id")) : null,
        packing_policy: form.get("packing_policy"),
        sampling_policy: form.get("sampling_policy"),
        sequence_order_policy: sequenceOrderPolicy,
        shuffle_block_sequences: sequenceOrderPolicy === "block_shuffle_v1"
          ? Number(form.get("shuffle_block_sequences"))
          : null,
        seed: Number(form.get("seed")),
      },
      ...executionPayload(form),
    };
    setPreflightResult(undefined);
    setFrozenPayload(undefined);
    preflight.mutate(data);
  }
  const ready = eligibleTokenizers.length > 0 && eligibleMixtures.length > 0 && checked.length > 0;
  return <Modal title="新建 Training Data View" description="训练配额和采样策略只属于这个视图。先用真实 Corpus 容量预检，再按同一份冻结配置创建索引。" onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns">
    <Field label="数据混合版本"><select name="mixture_version_id" required defaultValue=""><option value="">请选择已解析版本</option>{eligibleMixtures.map((item) => <option value={item.current_version!.id} key={item.current_version!.id}>{item.name} · V{item.current_version!.version}</option>)}</select></Field>
    <Field label="锁定 Tokenizer"><select name="tokenizer_version_id" required defaultValue=""><option value="">请选择 Tokenizer</option>{eligibleTokenizers.map((item) => <option value={item.id} key={item.id}>{item.name} · V{item.version}</option>)}</select></Field>
    <div className="span-two"><strong>Corpus 绑定</strong><p className="muted">勾选混合所引用的每个数据集 Corpus。预检会公开列出缺少或不兼容的绑定。</p><div className="attempt-list">{readyCorpora.map((item) => <label className="check-field" key={item.id}><input type="checkbox" checked={checked.includes(item.id)} onChange={(event) => setChecked(event.target.checked ? [...checked, item.id] : checked.filter((id) => id !== item.id))} /><span><b>{item.dataset_name} V{item.dataset_version}</b><small>{item.tokenizer_name} · {Number(item.token_count).toLocaleString()} 实际 Tokens</small></span></label>)}</div></div>
    <Field label="Sequence length"><input name="sequence_length" type="number" min="16" max="65535" defaultValue="2048" required /></Field>
    <Field label="训练请求 Tokens" hint="这是训练采样请求，不是 Corpus 成功门槛。"><input name="requested_tokens" type="number" min="1" defaultValue="12000000000" required /></Field>
    <Field label="数据消费策略"><select name="consumption_policy" defaultValue="unique_only"><option value="unique_only">unique_only · 每个文档只用一次</option><option value="repeat_epochs">repeat_epochs · 确定性循环</option></select></Field>
    <Field label="尾部策略"><select name="tail_policy" defaultValue="drop"><option value="drop">drop · 丢弃不足一个序列的尾部</option><option value="pad">pad · 补齐并记录有效长度</option></select></Field>
    <Field label="Padding Token ID" hint="仅 pad 策略生效；必须由用户明确配置。"><input name="padding_token_id" type="number" min="0" defaultValue="0" /></Field>
    <Field label="随机种子"><input name="seed" type="number" defaultValue="42" /></Field>
    <Field label="Packing 策略"><select name="packing_policy" defaultValue="document_stream_v1"><option value="document_stream_v1">document_stream_v1</option></select></Field>
    <Field label="混合采样策略"><select name="sampling_policy" defaultValue="weighted_deficit_v1"><option value="weighted_deficit_v1">weighted_deficit_v1 · 确定性权重调度</option></select></Field>
    <Field label="序列顺序" hint="可选择低空间的块随机，或准备阶段导出全局随机文件换取训练顺序 I/O。"><select name="sequence_order_policy" value={sequenceOrderPolicy} onChange={(event) => setSequenceOrderPolicy(event.target.value)}><option value="block_shuffle_v1">block_shuffle_v1 · 块间随机、块内顺序</option><option value="materialized_global_shuffle_v1">materialized_global_shuffle_v1 · 全局随机导出</option><option value="sequential_v1">sequential_v1 · 全局顺序</option></select></Field>
    <Field label="每块 Sequences" hint="默认 16；以 2048 上下文计约为 32K Tokens。"><input name="shuffle_block_sequences" type="number" min="1" max="4096" defaultValue="16" required={sequenceOrderPolicy === "block_shuffle_v1"} disabled={sequenceOrderPolicy !== "block_shuffle_v1"} /></Field>
    <Field label="索引格式"><input value="forgelm-training-index-v1" readOnly /></Field>
    <ExecutionFields adapterId="builtin.build-training-index" gpuDisabled />
  </div>
  {preflightResult && <section className="detail-section"><header><div><h3>真实容量预检</h3><p>系统校验与用户策略分开显示。</p></div><StatusBadge status={preflightResult.ready ? "ready" : "blocked"} /></header><dl className="detail-list"><div><dt>请求序列</dt><dd>{Number(preflightResult.requested_sequences).toLocaleString()}</dd></div><div><dt>唯一序列容量</dt><dd>{Number(preflightResult.unique_sequence_capacity).toLocaleString()}</dd></div><div><dt>所需 Epoch</dt><dd>{preflightResult.required_epochs}</dd></div>{Number(preflightResult.storage_estimate?.materialized_sequence_bytes || 0) > 0 && <><div><dt>物化文件估算</dt><dd>{(Number(preflightResult.storage_estimate.materialized_sequence_bytes) / (1024 ** 3)).toFixed(2)} GiB</dd></div><div><dt>Worker 最低可用空间</dt><dd>{(Number(preflightResult.storage_estimate.worker_free_space_required_bytes) / (1024 ** 3)).toFixed(2)} GiB</dd></div></>}<div><dt>配置快照</dt><dd><pre className="json-preview">{JSON.stringify(preflightResult.configuration, null, 2)}</pre></dd></div></dl>{preflightResult.system_validations.map((item: any) => <Alert key={item.id} severity={item.status === "passed" ? "success" : "blocker"} title={`系统校验：${item.id}`} detail={`${item.message}（不可配置，因为它验证制品是否有效；不属于门禁）`} />)}{preflightResult.warnings.map((item: any) => <Alert key={item.code} severity="warning" title={item.code} detail={item.message} />)}</section>}
  <ErrorMessage error={preflight.error || create.error} /><div className="form-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button>{frozenPayload ? <button type="button" className="button primary" disabled={create.isPending} onClick={() => create.mutate(frozenPayload)}>{create.isPending ? "创建中…" : "按此快照创建视图"}</button> : <button type="submit" className="button primary" disabled={!ready || preflight.isPending}>{preflight.isPending ? "预检中…" : "运行真实容量预检"}</button>}</div></form></Modal>;
}

function TokenizerDetail({ tokenizer, task: listTask, onEdit, onClose }: { tokenizer: any; task?: Task; onEdit: (task?: Task) => void; onClose: () => void }) {
  const queryClient = useQueryClient();
  const taskQuery = useQuery({ queryKey: ["task", tokenizer.task_id], queryFn: () => api<Task>(`/tasks/${tokenizer.task_id}`), enabled: Boolean(tokenizer.task_id), refetchInterval: 2000 });
  const task = taskQuery.data || listTask;
  const status = tokenizerStatus(tokenizer, task);
  const retry = useMutation({ mutationFn: () => post(`/tasks/${tokenizer.task_id}/control`, { action: "retry", reason: "用户从 Tokenizer 实验详情重试" }), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["task", tokenizer.task_id] }); queryClient.invalidateQueries({ queryKey: ["tasks", tokenizer.project_id] }); } });
  const error = task?.error && Object.keys(task.error).length ? task.error : undefined;
  const metrics = tokenizer.metrics || {};
  const frozenGatePolicy = task?.result?.gate_policy || tokenizerGatePolicy(tokenizer.config || {});
  const gateConfig = { ...(tokenizer.config || {}), gate_policy: frozenGatePolicy };
  const gateDefinitions = tokenizerGateDefinitions(gateConfig);
  const completedGateResults = new Map((task?.result?.gate_results || []).map((item: any) => [item.metric, item]));
  const gateResults = gateDefinitions.map((definition) => completedGateResults.get(definition.metric) || ({ metric: definition.metric, label: definition.label, ...frozenGatePolicy.rules[definition.metric], actual: metrics[definition.metric], passed: undefined, reason: metrics[definition.metric] == null ? undefined : "等待任务完成" }));
  const enabledGateResults = gateResults.filter((item: any) => item.enabled);
  const gateStatus = task?.result?.gate_passed === true ? "passed" : task?.result?.gate_passed === false ? "failed_gate" : "queued";
  return <Modal title={`${tokenizer.name} v${tokenizer.version}`} description="Tokenizer 实验详情、执行状态与可复现配置。" onClose={onClose} wide><div className="training-detail-body">
    <div className="detail-summary-line"><StatusBadge status={status} /><span><FlaskConical size={14} />{tokenizer.algorithm} · {tokenizer.vocab_size.toLocaleString()} 词表</span><span>{tokenizer.locked ? "已锁定" : "未锁定"}</span>{tokenizer.is_default && <span><Star size={14} />项目默认</span>}</div>
    {error && <Alert severity="blocker" title={status === "lost" ? "执行 Worker 已失联" : "实验执行失败"} detail={String(error.message || error.detail || JSON.stringify(error))} />}
    <section className="detail-section tokenizer-gate-evidence"><header><div><h3>质量门禁</h3><p>以下规则来自当前不可变版本；实测值和结论由控制面在任务完成时冻结。</p></div><StatusBadge status={gateStatus} /></header><div className="tokenizer-gate-evidence-grid">{gateResults.map((item: any) => <article key={item.metric} className={!item.enabled ? "disabled" : item.passed === true ? "passed" : item.passed === false ? "failed" : "pending"}><div><span>{item.label}</span><StatusBadge status={!item.enabled ? "disabled" : item.passed === true ? "passed" : item.passed === false ? "failed_gate" : "queued"} /></div><strong>{item.operator === "lte" ? "≤" : "≥"} {gateValue(item.metric, item.threshold)}</strong><dl><div><dt>实测值</dt><dd>{gateValue(item.metric, item.actual)}</dd></div><div><dt>判定</dt><dd>{gateReason(item.reason)}</dd></div></dl></article>)}</div><p className="tokenizer-gate-summary">已启用 {enabledGateResults.length} 条规则。修改门禁会创建新的 Tokenizer 版本，不会重算历史结论。</p></section>
    <section className="detail-section"><header><div><h3>实验结果</h3><p>真实引擎产出的压缩率、往返验证与制品记录。</p></div><Activity size={17} /></header><div className="run-detail-metrics"><div><span>中文压缩</span><strong>{metrics.zh_chars_per_token ?? "—"}</strong></div><div><span>英文压缩</span><strong>{metrics.en_chars_per_token ?? "—"}</strong></div><div><span>代码压缩</span><strong>{metrics.code_chars_per_token ?? "—"}</strong></div><div><span>Round-trip</span><strong>{metrics.round_trip != null ? `${metrics.round_trip * 100}%` : "—"}</strong></div></div><dl className="detail-list"><div><dt>Tokenizer 制品</dt><dd>{tokenizer.artifact_id || "尚未生成"}</dd></div><div><dt>数据输入</dt><dd>{tokenizer.data_ref_type} · {tokenizer.data_ref_id}</dd></div><div><dt>创建时间</dt><dd>{formatDate(tokenizer.created_at)}</dd></div></dl></section>
    <section className="detail-section"><header><div><h3>执行任务</h3><p>任务分配、进度以及最近一次执行尝试。</p></div><Activity size={17} /></header>{task ? <><dl className="detail-list"><div><dt>任务 ID</dt><dd>{task.id}</dd></div><div><dt>执行适配器</dt><dd>{task.adapter_id || "—"}</dd></div><div><dt>当前阶段</dt><dd>{task.current_phase || "—"}</dd></div><div><dt>进度</dt><dd>{Math.round((task.progress || 0) * 100)}%</dd></div><div><dt>开始 / 结束</dt><dd>{formatDate(task.started_at || undefined)} / {formatDate(task.ended_at || undefined)}</dd></div></dl>{task.attempts?.length ? <div className="attempt-list">{task.attempts.map((attempt) => <article key={attempt.id}><div><strong>尝试 #{attempt.attempt_number}</strong><StatusBadge status={attempt.status} /></div><dl className="detail-list"><div><dt>Worker</dt><dd>{attempt.worker_id || "未分配"}</dd></div><div><dt>退出码</dt><dd>{attempt.exit_code ?? "—"}</dd></div></dl></article>)}</div> : null}</> : <p>{tokenizer.task_id ? "正在读取任务详情…" : "该实验尚未启动执行任务。"}</p>}</section>
    <details className="detail-section collapsed-config"><summary><Settings2 size={14} /> 查看实验配置</summary><pre className="json-preview">{JSON.stringify(tokenizer.config || {}, null, 2)}</pre></details>
    <ErrorMessage error={taskQuery.error || retry.error} /><div className="form-actions"><button className="button secondary" onClick={() => onEdit(task)}><Pencil size={15} />编辑配置</button><span />{(status === "failed" || status === "lost") && tokenizer.task_id && <button className="button secondary" disabled={retry.isPending} onClick={() => retry.mutate()}><RotateCcw size={15} />{retry.isPending ? "正在重试…" : "重试任务"}</button>}<button className="button primary" onClick={onClose}>关闭</button></div>
  </div></Modal>;
}

function TokenizerModal({ projectId, datasets, mixtures, initial, initialTask, onClose, onCreated }: { projectId: string; datasets: Dataset[]; mixtures: Mixture[]; initial?: any; initialTask?: Task; onClose: () => void; onCreated: () => void }) {
  const navigate = useNavigate();
  const initialConfig = initial?.config || {};
  const [refType, setRefType] = useState<"mixture_version" | "dataset_version">(initial?.data_ref_type || "mixture_version");
  const [dataRefId, setDataRefId] = useState(initial?.data_ref_id || "");
  const [adapterId, setAdapterId] = useState(initialTask?.adapter_id || "builtin.hf-tokenizer");
  const [algorithm, setAlgorithm] = useState(initial?.algorithm || "BPE");
  const diskBpe = adapterId === "builtin.disk-bpe-tokenizer";
  const builtinCpu = adapterId === "builtin.hf-tokenizer" || diskBpe;
  const create = useMutation({ mutationFn: (data: any) => post(`/projects/${projectId}/tokenizers`, data), onSuccess: onCreated });
  const eligibleDatasets = datasets.filter((item) => item.current_version?.state === "ready" && item.current_version?.train_ready && item.current_version?.artifact_id && item.current_version?.record_manifest_artifact_id);
  const eligibleMixtures = mixtures.filter((item) => item.current_version?.status === "ready" && item.current_version?.manifest_artifact_id);
  const availableRefs = refType === "mixture_version" ? eligibleMixtures.map((item) => ({ id: item.current_version!.id, label: `${item.name} · V${item.current_version!.version}` })) : eligibleDatasets.map((item) => ({ id: item.current_version!.id, label: `${item.name} · V${item.current_version!.version}` }));
  const refs = initial?.data_ref_type === refType && initial.data_ref_id && !availableRefs.some((ref) => ref.id === initial.data_ref_id) ? [{ id: initial.data_ref_id, label: `当前绑定版本 · ${initial.data_ref_id.slice(0, 8)}` }, ...availableRefs] : availableRefs;
  const repairableDataset = datasets.find((item) => item.current_version?.artifact_id && !item.current_version?.record_manifest_artifact_id);
  const goToData = (query: string) => { onClose(); navigate(`/projects/${projectId}/data?${query}`); };
  const gateDefinitions = tokenizerGateDefinitions(initialConfig);
  const initialGatePolicy = tokenizerGatePolicy(initialConfig);
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const gatePolicy = {
      version: TOKENIZER_GATE_VERSION,
      rules: Object.fromEntries(gateDefinitions.map((definition) => [definition.metric, {
        enabled: form.get(`gate_${definition.metric}_enabled`) === "on",
        operator: definition.operator,
        threshold: Number(form.get(`gate_${definition.metric}_threshold`)),
        ...(definition.custom ? { label: definition.label } : {}),
      }])),
    };
    create.mutate({ name: form.get("name"), algorithm, vocab_size: Number(form.get("vocab_size")), data_ref_type: refType, data_ref_id: form.get("data_ref_id"), config: { sample_tokens: Number(form.get("sample_tokens")), normalization: form.get("normalization"), byte_fallback: form.get("byte_fallback") === "on", special_tokens: String(form.get("special_tokens") || "").split("\n").map((value) => value.trim()).filter(Boolean), gate_policy: gatePolicy, ...(diskBpe ? { min_word_frequency: Number(form.get("min_word_frequency")), max_unique_words: Number(form.get("max_unique_words")), max_candidate_characters: Number(form.get("max_candidate_characters")), max_word_length: Number(form.get("max_word_length")), merge_occurrence_budget: Number(form.get("merge_occurrence_budget")) } : {}) }, start: true, ...executionPayload(form) });
  }
  return <Modal title={initial ? `编辑 ${initial.name} v${initial.version} 配置` : "新建 Tokenizer 实验"} description={initial ? "保存会创建新的不可变版本并启动，当前版本和历史任务不会被覆盖。" : "可选择直接 HF 训练、磁盘外部聚合 BPE，或项目自定义命令。"} onClose={onClose} wide><form onSubmit={submit}><div className="form-grid two-columns">
    <Field label="实验名称" hint={initial ? "版本名称保持不变" : undefined}><input name="name" required defaultValue={initial?.name || ""} readOnly={Boolean(initial)} /></Field><Field label="算法" hint={diskBpe ? "磁盘外部聚合后端当前只支持 BPE" : undefined}><select name="algorithm" value={algorithm} disabled={diskBpe} onChange={(event) => setAlgorithm(event.target.value)}><option>BPE</option><option>Unigram</option><option>WordPiece</option></select></Field>
    <Field label="输入类型"><select className="tokenizer-ref-type-select" value={refType} onChange={(event) => { setRefType(event.target.value as typeof refType); setDataRefId(""); }}><option value="mixture_version">数据混合版本</option><option value="dataset_version">单个数据集版本</option></select></Field><Field label="训练数据" hint={refs.length ? `${refs.length} 个可用版本，全部展开显示` : undefined}><select aria-label="训练数据" className="tokenizer-data-ref-list" name="data_ref_id" size={Math.min(Math.max(refs.length + 1, 2), 6)} required disabled={!refs.length} value={dataRefId} onChange={(event) => setDataRefId(event.target.value)}><option value="">{refs.length ? "请选择训练数据" : "暂无可用版本"}</option>{refs.map((ref) => <option key={ref.id} value={ref.id}>{ref.label}</option>)}</select></Field>
    <Field label="词表大小"><input name="vocab_size" type="number" defaultValue={initial?.vocab_size || 32000} min="256" /></Field><Field label="训练语料抽样 Tokens" hint={diskBpe ? "全量扫描该目标；Token 数不作为内存上限，合并工作集由下方参数约束" : adapterId === "builtin.hf-tokenizer" ? "正文按有界批次流式送入；内存主要随不同预分词片段数量增长，不按总 Tokens 线性增长" : undefined}><input name="sample_tokens" type="number" defaultValue={initialConfig.sample_tokens || 100000000} min="1" /></Field><Field label="Normalization"><select name="normalization" defaultValue={initialConfig.normalization || "NFKC"}><option>NFKC</option><option>NFC</option><option>None</option></select></Field><Field label="特殊 Tokens" hint="每行一个；第一项作为 unknown token"><textarea name="special_tokens" defaultValue={(initialConfig.special_tokens || ["<unk>", "<s>", "</s>"]).join("\n")} /></Field>
    <Field label="执行方式"><select value={adapterId} onChange={(event) => { const next = event.target.value; setAdapterId(next); if (next === "builtin.disk-bpe-tokenizer") setAlgorithm("BPE"); }}><option value="builtin.hf-tokenizer">内置 Hugging Face Tokenizers（CPU）</option><option value="builtin.disk-bpe-tokenizer">磁盘外部聚合 BPE（大规模语料，CPU）</option><option value="builtin.command.tokenizer">自定义命令</option></select></Field><ExecutionFields adapterId={adapterId} gpuDisabled={builtinCpu} defaults={{ resource_request: initialTask?.payload?.resource_request, command: initialTask?.payload?.command }} />
    {diskBpe && <><Field label="最低词频" hint="全量统计后，低于该频次的词形不进入合并"><input name="min_word_frequency" type="number" min="1" max="1000000" defaultValue={initialConfig.min_word_frequency ?? 2} /></Field><Field label="最大候选词形数" hint="BPE 合并阶段的不同词形硬上限"><input name="max_unique_words" type="number" min="1000" max="1000000" defaultValue={initialConfig.max_unique_words ?? 250000} /></Field><Field label="候选字符预算" hint="所有候选词形字符总数的硬上限"><input name="max_candidate_characters" type="number" min="100000" max="16000000" defaultValue={initialConfig.max_candidate_characters ?? 4000000} /></Field><Field label="单词形最大长度"><input name="max_word_length" type="number" min="1" max="2048" defaultValue={initialConfig.max_word_length ?? 128} /></Field><Field label="合并重放预算" hint="必须不小于最大候选词形数；限制送入 HF Trainer 的总次数"><input name="merge_occurrence_budget" type="number" min="1000" max="20000000" defaultValue={initialConfig.merge_occurrence_budget ?? 2000000} /></Field></>}
    <label className="check-field"><input type="checkbox" name="byte_fallback" defaultChecked={initialConfig.byte_fallback ?? true} disabled={adapterId === "builtin.hf-tokenizer" && algorithm === "WordPiece"} /><span><b>启用 Byte fallback</b>{adapterId === "builtin.hf-tokenizer" && algorithm === "WordPiece" && <small>内置 WordPiece 不支持，已关闭</small>}</span></label>
    <section className="tokenizer-gate-editor span-two" aria-label="Tokenizer 质量门禁配置"><header><div><span className="eyebrow">QUALITY GATES · {gateDefinitions.length} RULES</span><h3>全部质量门禁</h3><p>所有规则都在这里展示和配置；历史规则不会因为当前目录变化而消失。</p></div><ShieldCheck size={20} /></header><div className="tokenizer-gate-groups">{Array.from(new Set(gateDefinitions.map((definition) => definition.group))).map((group) => <section key={group}><div className="tokenizer-gate-group-heading"><strong>{group}</strong><span>{gateDefinitions.filter((definition) => definition.group === group).length} 条规则</span></div><div className="tokenizer-gate-list">{gateDefinitions.filter((definition) => definition.group === group).map((definition) => { const rule = initialGatePolicy.rules[definition.metric]; return <article key={definition.metric}><label className="tokenizer-gate-toggle"><input type="checkbox" name={`gate_${definition.metric}_enabled`} defaultChecked={rule.enabled} /><span><b>{definition.label}</b><small>{definition.hint}</small></span></label><div className="tokenizer-gate-threshold"><span>{definition.symbol}</span><Field label={`${definition.label}阈值`}><input name={`gate_${definition.metric}_threshold`} aria-label={`${definition.label}阈值`} type="number" min={definition.min} max={definition.max} step={definition.step} defaultValue={rule.threshold} required /></Field></div></article>; })}</div></section>)}</div></section>
    {!refs.length && <div className="span-two"><Alert severity="warning" title={refType === "mixture_version" ? "没有可用的数据混合版本" : "没有可训练的数据集版本"} detail={refType === "mixture_version" ? "数据混合必须解析完成并生成 Manifest；也可以先改用已有的单个数据集。" : "数据集必须分析完成、可训练，并生成 Record Manifest。Parquet 数据可通过重新分析补齐。"} /><div className="form-actions">{refType === "mixture_version" && eligibleDatasets.length > 0 && <button type="button" className="button secondary" onClick={() => setRefType("dataset_version")}><Database size={15} />改用单个数据集</button>}{refType === "dataset_version" && repairableDataset && <button type="button" className="button primary" onClick={() => goToData(`tab=datasets&dataset=${repairableDataset.id}&action=reanalyze`)}><RefreshCw size={15} />重新分析 {repairableDataset.name}</button>}<button type="button" className="button secondary" onClick={() => goToData(refType === "mixture_version" ? "tab=mixtures" : "tab=datasets")}><ArrowRight size={15} />{refType === "mixture_version" ? "前往数据混合" : "前往数据中心"}</button></div></div>}
  </div><ErrorMessage error={create.error} /><FormActions submitting={create.isPending} disabled={!refs.length} onCancel={onClose} submitLabel={initial ? "创建新版本并启动" : "创建并启动"} /></form></Modal>;
}
