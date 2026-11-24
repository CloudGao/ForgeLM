import { Plus, Trash2 } from "lucide-react";
import type { Json } from "../types";

type GatePhase = "input" | "output";

const reservedSystemRuleIds = new Set([
  "data_mixture_ready", "tokenizer_locked", "tokenizer_model_compatible",
  "validation_input_ready", "resume_checkpoint_compatible",
  "worker_capability_available", "tokenized_dataset_integrity",
  "finite_training_state", "output_artifact_integrity", "training_plan_valid",
  "training_data_view_integrity", "training_data_view_selection",
  "training_data_view_capacity", "code_version_ready",
  "engine_configuration_valid", "task_input_materialization",
  "runtime_training_invariants",
]);

function editableRule(raw: Json): Json {
  return {
    ...raw,
    id: String(raw.id || `user_rule_${Date.now()}`),
    label: String(raw.label || raw.id || "自定义检查"),
    kind: String(raw.kind || "named_check"),
    source: raw.source === "template" ? "template" : "user",
    locked: false,
    enabled: raw.enabled !== false,
    severity: String(raw.severity || "blocker"),
    failure_action: String(raw.failure_action || "stop"),
    operator: String(raw.operator || "exists"),
    metric: String(raw.metric || ""),
    value: raw.value ?? "",
  };
}

function editableRules(configured: unknown): Json[] {
  return (Array.isArray(configured) ? configured : [])
    .filter((rule) => rule && rule.source !== "system" && !reservedSystemRuleIds.has(String(rule.id || "")))
    .map(editableRule);
}

export function normalizeStageGatesForEditor(stage: Json): Json {
  const legacyGate = stage.gate && typeof stage.gate === "object" ? stage.gate : {};
  const policy = stage.gate_policy && typeof stage.gate_policy === "object" ? stage.gate_policy : {};
  const legacyRules = Array.isArray(legacyGate.checks)
    ? legacyGate.checks.map((check: unknown) => editableRule({ id: String(check), label: String(check).replaceAll("_", " "), source: "template" }))
    : [];
  const { gate: _legacy, ...base } = stage;
  return {
    ...base,
    gate_policy: {
      ...policy,
      schema_version: 2,
      transition: policy.transition || (["manual_review", "final_acceptance"].includes(String(legacyGate.release)) ? "manual" : "automatic"),
      data_consumption: policy.data_consumption || "unique_only",
      input_rules: editableRules(policy.input_rules),
      output_rules: editableRules(policy.output_rules ?? legacyRules),
    },
  };
}

function rulesKey(phase: GatePhase) {
  return `${phase}_rules`;
}

export function StageGateEditor({ stage, stageIndex, onChange }: { stage: Json; stageIndex: number; onChange: (stage: Json) => void }) {
  const normalized = normalizeStageGatesForEditor(stage);
  const policy = normalized.gate_policy;

  function updatePolicy(key: string, value: unknown) {
    onChange({ ...normalized, gate_policy: { ...policy, [key]: value } });
  }

  function updateRule(phase: GatePhase, ruleIndex: number, key: string, value: unknown) {
    const keyName = rulesKey(phase);
    const rules = [...policy[keyName]];
    rules[ruleIndex] = { ...rules[ruleIndex], [key]: value };
    updatePolicy(keyName, rules);
  }

  function removeRule(phase: GatePhase, ruleIndex: number) {
    const keyName = rulesKey(phase);
    updatePolicy(keyName, policy[keyName].filter((_: Json, index: number) => index !== ruleIndex));
  }

  function addRule(phase: GatePhase) {
    const keyName = rulesKey(phase);
    updatePolicy(keyName, [...policy[keyName], editableRule({
      id: `user_${phase}_${Date.now()}`,
      label: phase === "input" ? "新的运行前记录" : "新的运行后记录",
    })]);
  }

  function renderRules(phase: GatePhase) {
    const keyName = rulesKey(phase);
    const phaseLabel = phase === "input" ? "输入" : "输出";
    return <section className="gate-rule-section">
      <header><div><strong>{phaseLabel}记录</strong><span>可选的观察与审计规则，不影响训练提交。</span></div><button type="button" className="button secondary small" onClick={() => addRule(phase)}><Plus size={13} />添加记录</button></header>
      <div className="gate-rule-list">{policy[keyName].map((rule: Json, ruleIndex: number) => <div className="gate-rule-row" key={`${rule.id}-${ruleIndex}`}>
        <div className="gate-rule-meta">
          <label className="gate-checkbox"><input type="checkbox" checked={rule.enabled !== false} onChange={(event) => updateRule(phase, ruleIndex, "enabled", event.target.checked)} />启用</label>
          <span className="gate-rule-origin">{rule.source === "template" ? "模板规则" : "用户规则"}</span>
        </div>
        <label className="gate-rule-label"><span>规则名称</span><input aria-label={`阶段 ${stageIndex + 1} ${phaseLabel}规则 ${ruleIndex + 1} 名称`} value={rule.label} onChange={(event) => updateRule(phase, ruleIndex, "label", event.target.value)} /></label>
        <label className="gate-rule-label"><span>规则 ID</span><input value={rule.id} onChange={(event) => updateRule(phase, ruleIndex, "id", event.target.value)} /></label>
        <div className="gate-rule-settings">
          <label><span>类型</span><select value={rule.kind} onChange={(event) => updateRule(phase, ruleIndex, "kind", event.target.value)}><option value="named_check">命名检查</option><option value="metric_threshold">指标阈值</option><option value="artifact_required">要求制品</option><option value="manual_approval">人工批准</option></select></label>
          <label><span>比较符</span><select value={rule.operator} onChange={(event) => updateRule(phase, ruleIndex, "operator", event.target.value)}><option value="exists">存在</option><option value="gte">≥</option><option value="gt">&gt;</option><option value="lte">≤</option><option value="lt">&lt;</option><option value="eq">=</option><option value="ne">≠</option></select></label>
          <label><span>指标/制品键</span><input value={rule.metric} onChange={(event) => updateRule(phase, ruleIndex, "metric", event.target.value)} /></label>
          <label><span>比较值</span><input value={String(rule.value ?? "")} onChange={(event) => updateRule(phase, ruleIndex, "value", event.target.value)} /></label>
          <label><span>显示级别</span><select aria-label={`阶段 ${stageIndex + 1} ${phaseLabel}规则 ${ruleIndex + 1} 严重性`} value={rule.severity} onChange={(event) => updateRule(phase, ruleIndex, "severity", event.target.value)}><option value="blocker">错误</option><option value="warning">警告</option><option value="info">信息</option></select></label>
          <label><span>记录方式</span><select aria-label={`阶段 ${stageIndex + 1} ${phaseLabel}规则 ${ruleIndex + 1} 失败动作`} value={rule.failure_action} onChange={(event) => updateRule(phase, ruleIndex, "failure_action", event.target.value)}><option value="stop">标记需处理</option><option value="notify_and_continue">通知</option><option value="record_only">仅记录</option></select></label>
        </div>
        <button type="button" className="icon-button gate-rule-delete" title="删除此规则" aria-label={`删除 ${rule.label}`} onClick={() => removeRule(phase, ruleIndex)}><Trash2 size={14} /></button>
      </div>)}</div>
    </section>;
  }

  return <details className="stage-gate-editor">
    <summary>高级阶段记录（可选） · 输入 {policy.input_rules.length} / 输出 {policy.output_rules.length} 条</summary>
    <p className="muted">这里的设置用于观察和审计，不会阻止提交训练。默认留空即可。</p>
    <div className="stage-gate-policy-grid">
      <label><span>阶段记录偏好</span><select aria-label={`阶段 ${stageIndex + 1} 转换方式`} value={policy.transition} onChange={(event) => updatePolicy("transition", event.target.value)}><option value="automatic">自动继续</option><option value="manual">标记为需人工复核</option></select></label>
      <label><span>数据消费策略</span><select aria-label={`阶段 ${stageIndex + 1} 数据消费策略`} value={policy.data_consumption} onChange={(event) => updatePolicy("data_consumption", event.target.value)}><option value="unique_only">唯一消费（不足时给出提示）</option><option value="repeat_epochs">按冻结顺序循环多个 Epoch</option></select></label>
      <label><span>评估间隔 Tokens</span><input aria-label={`阶段 ${stageIndex + 1} 评估间隔 Tokens`} type="number" min="0" value={Number(normalized.evaluation_interval_tokens || 0)} onChange={(event) => onChange({ ...normalized, evaluation_interval_tokens: Number(event.target.value) })} /></label>
      <label><span>Checkpoint 间隔 Tokens</span><input aria-label={`阶段 ${stageIndex + 1} Checkpoint 间隔 Tokens`} type="number" min="0" value={Number(normalized.checkpoint_interval_tokens || 0)} onChange={(event) => onChange({ ...normalized, checkpoint_interval_tokens: Number(event.target.value) })} /></label>
      <label className="gate-checkbox"><input type="checkbox" checked={normalized.evaluation_at_end !== false} onChange={(event) => onChange({ ...normalized, evaluation_at_end: event.target.checked })} />阶段结束时评估</label>
      <label className="gate-checkbox"><input type="checkbox" checked={normalized.checkpoint_at_end !== false} onChange={(event) => onChange({ ...normalized, checkpoint_at_end: event.target.checked })} />阶段结束时保存 Checkpoint</label>
    </div>
    {renderRules("input")}
    {renderRules("output")}
  </details>;
}
