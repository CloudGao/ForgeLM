import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, postForm } from "../api";
import type { ChatTemplate, CodeSource, Json, Mixture, ModelRegistry, ModelSpecification, PlanRevision, SupervisionMixture, TrainingValidationSet, Worker } from "../types";
import { TrainingPlanConfigForm, type TrainingPlanPayload } from "./TrainingPlanConfigForm";
import { ErrorMessage, Field, Modal } from "./ui";

const defaultStages = [{ name: "模型训练", type: "pretrain", target_tokens: 10_000_000, sequence_length: 4096, global_batch_size: 64, learning_rate: 0.0003, scheduler: "cosine_with_min_lr", warmup_ratio: 0.015, min_learning_rate: 0.00003, evaluation_interval_tokens: 0, checkpoint_interval_tokens: 10_000_000, evaluation_at_end: true, checkpoint_at_end: true }];

export function NativeTrainingConfig({ projectId, spec, onSave, onClose }: { projectId: string; spec: Json; onSave: (spec: TrainingPlanPayload) => void; onClose: () => void }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const fullState = spec.input_model_ref?.type === "full_state_continuation";
  const [checkpointId, setCheckpointId] = useState(fullState ? "__full_state__" : String(spec.input_model_ref?.checkpoint_id || ""));
  const [continuationMode, setContinuationMode] = useState(String(spec.input_model_ref?.mode || "weights"));
  const [keepOptimizer, setKeepOptimizer] = useState(Boolean(spec.input_model_ref?.keep_optimizer));
  const options = useQuery({ queryKey: ["training-node-options", projectId], queryFn: async () => {
    const [tokenizers, mixtures, supervisionMixtures, chatTemplates, sources, modelSpecifications, validationSets, workers, registry] = await Promise.all([
      api<Json[]>(`/projects/${projectId}/tokenizers`), api<Mixture[]>(`/projects/${projectId}/mixtures`), api<SupervisionMixture[]>(`/projects/${projectId}/supervision-mixtures`), api<ChatTemplate[]>(`/projects/${projectId}/chat-templates`), api<CodeSource[]>(`/projects/${projectId}/code-sources`), api<ModelSpecification[]>(`/projects/${projectId}/model-specifications`), api<TrainingValidationSet[]>(`/projects/${projectId}/training-validation-sets`), api<Worker[]>("/workers"), api<ModelRegistry>(`/projects/${projectId}/models?include_archived=true`),
    ]);
    return { tokenizers, mixtures, supervisionMixtures, chatTemplates, codeVersions: sources.flatMap((source) => source.versions.filter((v) => v.state === "ready").map((v) => ({ ...v, source_name: source.name }))), modelSpecifications, validationSets, workers, registry };
  } });
  const initial = { ...spec, effective_config: { engine_config: spec.engine_config || {} } } as unknown as PlanRevision;
  async function uploadValidation(name: string, file: File) {
    const form = new FormData(); form.set("name", name); form.set("file", file);
    const created = await postForm<TrainingValidationSet>(`/projects/${projectId}/training-validation-sets/upload`, form);
    await queryClient.invalidateQueries({ queryKey: ["training-node-options", projectId] });
    return created;
  }
  return <Modal title="训练节点参数" description="配置训练阶段、精确数据与代码版本、超参数和资源要求。保存到当前编辑草稿，创建实例时统一冻结。" onClose={onClose} wide className="training-plan-modal">
    <ErrorMessage error={options.error} />
    {options.isPending && <p>正在读取项目训练资源…</p>}
    {options.data && <><Field label="初始 Checkpoint" hint="完整状态接续保留优化器、学习率计划和数据位置；普通 Checkpoint 微调重置优化器。"><select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)}>{fullState && <option value="__full_state__">完整状态接续（性能比较节点）</option>}<option value="">使用模型规格的初始化配置</option>{options.data.registry.checkpoints.filter((c) => c.status === "available" && c.artifact_id).map((c) => <option key={c.id} value={c.id}>Step {c.step} · {c.id.slice(0, 8)}</option>)}</select></Field>
      {checkpointId && checkpointId !== "__full_state__" && <><Field label="继承训练状态" hint="完整恢复要求配置、数据和 Tokenizer 相同；配置变更接续会重置调度器、随机状态和数据游标。目标引擎及 Checkpoint 必须支持所选状态。"><select value={continuationMode} onChange={e => { setContinuationMode(e.target.value); setKeepOptimizer(false); }}><option value="weights">仅继承权重</option><option value="configured">更换数据 / 训练配置</option><option value="full">完整恢复（允许兼容引擎版本）</option></select></Field>{continuationMode === "configured" && <label><input type="checkbox" checked={keepOptimizer} onChange={e => setKeepOptimizer(e.target.checked)} />保留兼容 Adam 动量</label>}</>}
      <TrainingPlanConfigForm mode="node" initial={initial} defaultStages={defaultStages} {...options.data} submitLabel="保存节点配置" onCancel={onClose} onSubmit={(payload) => onSave({ ...payload, input_model_ref: checkpointId === "__full_state__" ? spec.input_model_ref : checkpointId ? continuationMode === "weights" ? { type: "checkpoint_finetune", checkpoint_id: checkpointId, data_cursor_policy: "reset" } : { type: "checkpoint_continuation", checkpoint_id: checkpointId, mode: continuationMode, keep_optimizer: continuationMode === "configured" && keepOptimizer } : { type: "random_init" } })} onImportCode={() => { onClose(); navigate(`/projects/${projectId}/overview?import=code`); }} onUploadValidation={uploadValidation} />
    </>}
  </Modal>;
}
