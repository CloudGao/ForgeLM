import type { FormEvent, ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";

export function StatusBadge({ status }: { status?: string }) {
  const normalized = status || "unknown";
  const labels: Record<string, string> = {
    succeeded: "已完成",
    passed: "已通过",
    ready: "就绪",
    official: "正式",
    running: "运行中",
    downloading: "下载中",
    verifying: "校验中",
    online: "在线",
    offline: "离线",
    installing: "安装中",
    updating: "更新中",
    in_progress: "进行中",
    assigned: "已分配",
    starting: "启动中",
    injecting: "注入中",
    queued: "排队中",
    paused: "已暂停",
    waiting: "待就绪",
    warning: "有警告",
    blocked: "已阻断",
    failed: "失败",
    failed_gate: "门禁未通过",
    disabled: "未启用",
    cancelled: "已取消",
    skipped: "已跳过",
    stopped: "已停止",
    candidate: "候选",
    draft: "草稿",
    registered: "已登记",
    available: "可用",
    archived: "已归档",
    deleted: "已删除",
    needs_review: "需检查",
    published: "已发布",
    benchmark: "Benchmark",
    not_started: "未开始",
    lost: "已失联",
    healthy: "健康",
    critical: "空间紧张",
    planned: "待确认",
    executing: "清理中",
    partial: "部分完成",
    expired: "已过期",
  };
  const positive = ["succeeded", "passed", "ready", "official", "published", "online", "benchmark", "healthy"];
  const active = ["running", "downloading", "verifying", "in_progress", "assigned", "starting", "injecting", "queued", "pausing", "executing"];
  const negative = ["failed", "failed_gate", "blocked", "cancelled", "stopped", "lost", "offline", "critical"];
  const tone = positive.includes(normalized) ? "positive" : active.includes(normalized) ? "active" : negative.includes(normalized) ? "negative" : "neutral";
  return <span className={`status status-${tone}`}><i />{labels[normalized] || normalized}</span>;
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function MetricCard({ label, value, detail, tone = "default" }: { label: string; value: ReactNode; detail?: ReactNode; tone?: string }) {
  return (
    <article className={`metric-card metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

export function Panel({ title, description, actions, children, className = "" }: { title?: string; description?: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-header">
          <div>{title && <h2>{title}</h2>}{description && <p>{description}</p>}</div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Modal({ title, description, children, onClose, wide = false, className = "" }: { title: string; description?: string; children: ReactNode; onClose: () => void; wide?: boolean; className?: string }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={`modal ${wide ? "modal-wide" : ""} ${className}`.trim()} role="dialog" aria-modal="true" aria-label={title}>
        <header><div><h2>{title}</h2>{description && <p>{description}</p>}</div><button className="icon-button" onClick={onClose} aria-label="关闭"><X size={18} /></button></header>
        {children}
      </section>
    </div>
  );
}

export function FormActions({ submitting, disabled, onCancel, submitLabel = "保存" }: { submitting?: boolean; disabled?: boolean; onCancel: () => void; submitLabel?: string }) {
  return <div className="form-actions"><button type="button" className="button secondary" onClick={onCancel}>取消</button><button type="submit" className="button primary" disabled={submitting || disabled}>{submitting ? "处理中…" : submitLabel}</button></div>;
}

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return <div className="empty-state"><div className="empty-mark">◇</div><h3>{title}</h3><p>{description}</p>{action}</div>;
}

export function Alert({ severity, title, detail }: { severity: string; title: string; detail?: string }) {
  const Icon = severity === "blocker" || severity === "warning" ? AlertTriangle : severity === "success" ? CheckCircle2 : Info;
  return <div className={`alert alert-${severity}`}><Icon size={17} /><div><strong>{title}</strong>{detail && <span>{detail}</span>}</div></div>;
}

export function Progress({ value }: { value: number }) {
  const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
  return <div className="progress" aria-label={`进度 ${percent}%`}><span style={{ width: `${percent}%` }} /></div>;
}

export function SubmitForm({ onSubmit, children, className = "form-grid" }: { onSubmit: (event: FormEvent<HTMLFormElement>) => void; children: ReactNode; className?: string }) {
  return <form className={className} onSubmit={onSubmit}>{children}</form>;
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}

export function ExecutionFields({ adapterId, defaultGpus = 0, gpuDisabled = false, defaults }: { adapterId: string; defaultGpus?: number; gpuDisabled?: boolean; defaults?: { resource_request?: Record<string, any>; command?: Record<string, any> } }) {
  const commandAdapter = adapterId.startsWith("builtin.command.");
  const resourceRequest = defaults?.resource_request || {};
  const command = defaults?.command || {};
  const workerLabels = Object.entries(resourceRequest.worker_selector?.labels || {}).map(([key, value]) => `${key}=${value}`).join(", ");
  const commandOutputs = (command.outputs || []).map((output: any) => `${output.kind || "task_output"}=${output.path || ""}`).join("\n");
  return <>
    <input type="hidden" name="adapter_id" value={adapterId} />
    <Field label="执行适配器"><input value={adapterId} readOnly /></Field>
    <Field label="GPU 数量" hint={gpuDisabled ? "该内置适配器仅使用 CPU，不会占用 GPU" : undefined}><input name="execution_gpus" type="number" min="0" defaultValue={gpuDisabled ? 0 : (resourceRequest.gpus ?? defaultGpus)} disabled={gpuDisabled} /></Field>
    <Field label="Worker 标签" hint="逗号或换行分隔 key=value；留空表示任意匹配 Worker"><input name="execution_worker_labels" defaultValue={workerLabels} placeholder="environment=wsl, project=example-model" /></Field>
    {commandAdapter && <>
      <Field label="可执行文件"><input name="execution_executable" required defaultValue={command.executable || ""} placeholder="python 或完整可执行文件路径" /></Field>
      <Field label="命令执行目录（可选）"><input name="execution_cwd" defaultValue={command.cwd || ""} placeholder="留空使用平台分配的运行目录" /></Field>
      <Field label="命令参数" hint="每行一个参数"><textarea name="execution_args" defaultValue={(command.args || []).join("\n")} /></Field>
      <Field label="输出制品" hint="每行 kind=路径；目录自动打包"><textarea name="execution_outputs" defaultValue={commandOutputs} /></Field>
    </>}
  </>;
}

export function parseWorkerLabels(value: FormDataEntryValue | null) {
  const labels: Record<string, string> = {};
  String(value || "").split(/[\n,]/).map((item) => item.trim()).filter(Boolean).forEach((item) => {
    const separator = item.indexOf("=");
    if (separator > 0 && item.slice(separator + 1).trim()) labels[item.slice(0, separator).trim()] = item.slice(separator + 1).trim();
  });
  return labels;
}

export function executionPayload(form: FormData) {
  const adapterId = String(form.get("adapter_id") || "");
  const args = String(form.get("execution_args") || "").split("\n").map((value) => value.trim()).filter(Boolean);
  const outputs = String(form.get("execution_outputs") || "").split("\n").map((value) => value.trim()).filter(Boolean).map((value) => {
    const separator = value.indexOf("=");
    return separator > 0 ? { kind: value.slice(0, separator).trim(), path: value.slice(separator + 1).trim() } : { kind: "task_output", path: value };
  });
  const workerLabels = parseWorkerLabels(form.get("execution_worker_labels"));
  return {
    adapter_id: adapterId || undefined,
    launcher_id: "builtin.local-process",
    resource_request: { gpus: Number(form.get("execution_gpus") || 0), ...(Object.keys(workerLabels).length ? { worker_selector: { labels: workerLabels } } : {}) },
    command: adapterId.startsWith("builtin.command.") ? { executable: form.get("execution_executable"), cwd: form.get("execution_cwd") || null, args, env: {}, outputs } : {},
  };
}

export function ErrorMessage({ error }: { error: unknown }) {
  if (!error) return null;
  return <div className="form-error">{error instanceof Error ? error.message : "操作失败，请重试"}</div>;
}
