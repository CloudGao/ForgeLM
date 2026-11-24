import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Braces, Check, Edit3, GitCommit, History } from "lucide-react";
import { useState } from "react";
import { formatDate, post, postForm } from "../api";
import type {
  CodeVersion,
  ChatTemplate,
  Json,
  Mixture,
  ModelSpecification,
  PlanRevision,
  SupervisionMixture,
  TrainingPlan,
  TrainingValidationSet,
  Worker,
} from "../types";
import {
  TrainingPlanConfigForm,
  type TrainingPlanPayload,
} from "./TrainingPlanConfigForm";
import { Modal, StatusBadge } from "./ui";

function pretty(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2);
}

function revisionEngineConfig(revision: PlanRevision): Json {
  const config = revision.effective_config?.engine_config;
  return config && typeof config === "object" && !Array.isArray(config) ? config : {};
}

interface DetailProps {
  plan: TrainingPlan;
  tokenizers: Json[];
  mixtures: Mixture[];
  supervisionMixtures: SupervisionMixture[];
  chatTemplates: ChatTemplate[];
  codeVersions: Array<CodeVersion & { source_name: string }>;
  modelSpecifications: ModelSpecification[];
  validationSets: TrainingValidationSet[];
  workers: Worker[];
  onClose: () => void;
}

export function TrainingPlanDetail({
  plan,
  tokenizers,
  mixtures,
  supervisionMixtures,
  chatTemplates,
  codeVersions,
  modelSpecifications,
  validationSets,
  workers,
  onClose,
}: DetailProps) {
  const queryClient = useQueryClient();
  const [selectedRevisionId, setSelectedRevisionId] = useState(
    plan.current_revision?.id || plan.revisions[0]?.id || "",
  );
  const [editing, setEditing] = useState(false);

  const revision =
    plan.revisions.find((item) => item.id === selectedRevisionId) || plan.current_revision;
  if (!revision) {
    return <Modal title={plan.name} description="训练计划尚无修订版" onClose={onClose}><p className="quiet-message">无法查看训练配置。</p></Modal>;
  }

  function selectRevision(id: string) {
    setSelectedRevisionId(id);
    setEditing(false);
  }

  async function uploadValidation(name: string, file: File) {
    const form = new FormData();
    form.set("name", name);
    form.set("file", file);
    const created = await postForm<TrainingValidationSet>(
      `/projects/${plan.project_id}/training-validation-sets/upload`,
      form,
    );
    await queryClient.invalidateQueries({
      queryKey: ["training-validation-sets", plan.project_id],
    });
    return created;
  }

  return <Modal title={plan.name} description={plan.description || "训练计划详情"} onClose={onClose} wide className="training-plan-modal">
    <div className="training-detail-body plan-detail-body">
      <div className="detail-summary-line">
        <StatusBadge status={plan.status} />
        <span><GitCommit size={14} />当前 Revision {plan.current_revision?.revision || "—"}</span>
        <span>计划 {plan.id.slice(0, 8)}</span>
      </div>
      <div className={`plan-detail-layout ${editing ? "is-editing" : ""}`}>
        <aside className="revision-sidebar">
          <header><History size={15} /><strong>修订历史</strong></header>
          <div>{plan.revisions.map((item) => <button type="button" key={item.id} className={item.id === revision.id ? "active" : ""} onClick={() => selectRevision(item.id)}><i>{item.id === plan.current_revision?.id ? <Check size={12} /> : item.revision}</i><span><b>Revision {item.revision}</b><small>{formatDate(item.created_at)}{item.locked ? " · 已锁定" : ""}</small></span></button>)}</div>
        </aside>
        <main className="revision-content">
          <div className="revision-heading"><div><span>REVISION {revision.revision}</span><h3>{revision.id === plan.current_revision?.id ? "当前执行配置" : "历史执行配置"}</h3><p>{revision.locked ? "该修订版已用于训练并锁定。" : "该修订版尚未锁定；启动时会冻结。"}</p></div><button className="button primary small" onClick={() => setEditing(!editing)}><Edit3 size={14} />{editing ? "取消编辑" : "基于此版本编辑"}</button></div>
          {editing ? <PlanRevisionEditor
            plan={plan}
            revision={revision}
            tokenizers={tokenizers}
            mixtures={mixtures}
            supervisionMixtures={supervisionMixtures}
            chatTemplates={chatTemplates}
            codeVersions={codeVersions}
            modelSpecifications={modelSpecifications}
            validationSets={validationSets}
            workers={workers}
            onUploadValidation={uploadValidation}
            onCancel={() => setEditing(false)}
            onCreated={(created) => {
              queryClient.setQueryData<TrainingPlan[]>(["plans", plan.project_id], (current) =>
                current?.map((item) => item.id === plan.id
                  ? { ...item, current_revision_id: created.id, current_revision: created, revisions: [created, ...item.revisions] }
                  : item),
              );
              setSelectedRevisionId(created.id);
              setEditing(false);
              queryClient.invalidateQueries({ queryKey: ["plans", plan.project_id] });
            }}
          /> : <RevisionView revision={revision} modelSpecifications={modelSpecifications} validationSets={validationSets} mixtures={mixtures} workers={workers} />}
        </main>
      </div>
    </div>
  </Modal>;
}

function RevisionView({ revision, modelSpecifications, validationSets, mixtures, workers }: { revision: PlanRevision; modelSpecifications: ModelSpecification[]; validationSets: TrainingValidationSet[]; mixtures: Mixture[]; workers: Worker[] }) {
  const engineConfig = revisionEngineConfig(revision);
  const validationSet = validationSets.find((item) => item.id === revision.validation_set_id);
  const validationMixture = mixtures.find((item) => item.current_version?.id === revision.validation_mixture_version_id);
  const worker = workers.find((item) => item.id === revision.resource_request?.worker_selector?.worker_id);
  const modelSpecification = modelSpecifications.find((item) => item.revisions.some((candidate) => candidate.id === revision.model_specification_revision_id));
  const modelSpecificationRevision = modelSpecification?.revisions.find((item) => item.id === revision.model_specification_revision_id);
  const args = Array.isArray(engineConfig.args) ? engineConfig.args.map(String) : [];
  const completeConfig = {
    stages: revision.stages,
    input_model_ref: revision.input_model_ref,
    model_specification_revision_id: revision.model_specification_revision_id,
    tokenizer_version_id: revision.tokenizer_version_id,
    validation_set_id: revision.validation_set_id,
    validation_mixture_version_id: revision.validation_mixture_version_id,
    code_version_id: revision.code_version_id,
    engine_plugin_id: revision.engine_plugin_id,
    launcher_plugin_id: revision.launcher_plugin_id,
    resource_request: revision.resource_request,
    engine_config: engineConfig,
  };
  return <div className="revision-view">
    <div className="revision-facts">
      <div><span>Engine</span><strong>{revision.engine_plugin_id === "builtin.command.training" ? "内置命令训练引擎" : revision.engine_plugin_id}</strong></div>
      <div><span>Launcher</span><strong>{revision.launcher_plugin_id === "builtin.local-process" ? "本地进程" : revision.launcher_plugin_id}</strong></div>
      <div><span>Tokenizer</span><strong>{revision.tokenizer_version_id?.slice(0, 8) || "未设置"}</strong></div>
      <div><span>模型规格</span><strong>{modelSpecification ? `${modelSpecification.name} · V${modelSpecificationRevision?.revision}` : "历史自由配置"}</strong></div>
      <div><span>训练验证集</span><strong>{validationSet ? `${validationSet.name} · V${validationSet.version}` : validationMixture ? `${validationMixture.name} · V${validationMixture.current_version?.version}` : "未绑定"}</strong></div>
      <div><span>训练代码版本</span><strong>{revision.code_version_id?.slice(0, 8) || "未绑定"}</strong></div>
      <div><span>执行 Worker</span><strong>{worker?.name || "自动选择"}</strong></div>
      <div><span>GPU / CPU</span><strong>{revision.resource_request?.gpus ?? 0} / {revision.resource_request?.cpus ?? "—"}</strong></div>
    </div>
    <section className="detail-section">
      <header><div><h3>训练阶段</h3><p>阶段、数据版本与门禁 token 目标均为结构化配置。</p></div></header>
      <div className="revision-stage-list">{revision.stages.map((stage, index) => { const policy = stage.gate_policy || {}; const inputRules = Array.isArray(policy.input_rules) ? policy.input_rules : []; const outputRules = Array.isArray(policy.output_rules) ? policy.output_rules : []; return <article key={`${stage.name || "stage"}-${index}`}><i>{index}</i><div><strong>{stage.name || `Stage ${index}`}</strong><span>{stage.type || stage.workflow || stage.mode || "自定义阶段"} · {policy.transition === "manual" ? "标记人工复核" : "自动继续"} · {policy.data_consumption === "repeat_epochs" ? "允许多 Epoch" : "唯一序列"}</span><details className="revision-stage-gates"><summary>查看可选阶段记录（输入 {inputRules.length} / 输出 {outputRules.length}）</summary><div><section><b>输入记录</b>{inputRules.map((rule: any) => <p key={rule.id}>{rule.source === "template" ? "模板规则" : "用户规则"} · {rule.enabled === false ? "已停用" : "已启用"} · {rule.label} · {rule.operator}{rule.value !== "" && rule.value != null ? ` ${rule.value}` : ""}</p>)}</section><section><b>输出记录</b>{outputRules.map((rule: any) => <p key={rule.id}>{rule.source === "template" ? "模板规则" : "用户规则"} · {rule.enabled === false ? "已停用" : "已启用"} · {rule.label} · {rule.severity} · {rule.failure_action}</p>)}</section></div></details></div><small>{stage.target_tokens ? `${stage.target_tokens} tokens` : stage.target_steps ? `${stage.target_steps} steps` : ""}</small></article>; })}</div>
    </section>
    <div className="training-detail-columns">
      <section className="detail-section"><header><div><h3>输入与资源</h3><p>平台实体和 Worker 资源约束。</p></div></header><dl className="detail-list"><div><dt>起始模型</dt><dd>{!Object.keys(revision.input_model_ref || {}).length || revision.input_model_ref?.type === "random_init" ? "按模型配置随机初始化" : "已绑定不可变引用"}</dd></div><div><dt>精度</dt><dd>{revision.resource_request?.precision || "未设置"}</dd></div><div><dt>验证样本</dt><dd>{validationSet ? `${validationSet.document_count} 文档 / 约 ${validationSet.estimated_tokens} tokens` : validationMixture ? `数据混合目标 ${validationMixture.current_version?.target_tokens || 0} tokens` : "—"}</dd></div></dl></section>
      <section className="detail-section"><header><div><h3>运行命令</h3><p>运行时路径由 Worker 在任务执行前解析。</p></div><Braces size={17} /></header><dl className="detail-list"><div><dt>Python</dt><dd>{engineConfig.executable === "__worker_python__" ? "Worker Python（自动）" : String(engineConfig.executable || "未设置")}</dd></div><div><dt>工作目录</dt><dd>{engineConfig.cwd || "代码根目录（自动）"}</dd></div><div><dt>参数</dt><dd>{args.length ? args.join(" ") : "—"}</dd></div></dl></section>
    </div>
    <details className="collapsed-config"><summary>查看该 Revision 的完整 JSON（只读）</summary><pre className="json-preview">{pretty(completeConfig)}</pre></details>
  </div>;
}

interface EditorProps {
  plan: TrainingPlan;
  revision: PlanRevision;
  tokenizers: Json[];
  mixtures: Mixture[];
  supervisionMixtures: SupervisionMixture[];
  chatTemplates: ChatTemplate[];
  codeVersions: Array<CodeVersion & { source_name: string }>;
  modelSpecifications: ModelSpecification[];
  validationSets: TrainingValidationSet[];
  workers: Worker[];
  onUploadValidation: (name: string, file: File) => Promise<TrainingValidationSet>;
  onCancel: () => void;
  onCreated: (revision: PlanRevision) => void;
}

function PlanRevisionEditor({
  plan,
  revision,
  tokenizers,
  mixtures,
  supervisionMixtures,
  chatTemplates,
  codeVersions,
  modelSpecifications,
  validationSets,
  workers,
  onUploadValidation,
  onCancel,
  onCreated,
}: EditorProps) {
  const create = useMutation({
    mutationFn: (payload: TrainingPlanPayload) =>
      post<PlanRevision>(`/training-plans/${plan.id}/revisions`, payload),
    onSuccess: onCreated,
  });
  const nextRevision = Math.max(...plan.revisions.map((item) => item.revision), 0) + 1;
  return <TrainingPlanConfigForm
    mode="revision"
    initial={revision}
    defaultStages={revision.stages}
    tokenizers={tokenizers}
    mixtures={mixtures}
    supervisionMixtures={supervisionMixtures}
    chatTemplates={chatTemplates}
    codeVersions={codeVersions}
    modelSpecifications={modelSpecifications}
    validationSets={validationSets}
    workers={workers}
    submitting={create.isPending}
    error={create.error}
    submitLabel={`保存为 Revision ${nextRevision}`}
    onCancel={onCancel}
    onSubmit={(payload) => create.mutate(payload)}
    onUploadValidation={onUploadValidation}
  />;
}
