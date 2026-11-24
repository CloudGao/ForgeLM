import { Plus, Save, Trash2, Upload } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { formatNumber } from "../api";
import type {
  CodeVersion,
  ChatTemplate,
  Json,
  Mixture,
  ModelSpecification,
  PlanRevision,
  SupervisionMixture,
  TrainingValidationSet,
  Worker,
} from "../types";
import { normalizeStageGatesForEditor, StageGateEditor } from "./StageGateEditor";
import { ErrorMessage, Field, FormActions } from "./ui";
import { TrainingOptimizationEditor } from "./TrainingOptimizationEditor";

export interface TrainingPlanPayload extends Json {
  name?: string;
  description?: string;
  stages: Json[];
  input_model_ref: Json;
  model_specification_revision_id: string | null;
  tokenizer_version_id: string | null;
  validation_set_id: string | null;
  validation_mixture_version_id: string | null;
  code_version_id: string | null;
  engine_plugin_id: string;
  launcher_plugin_id: string;
  resource_request: Json;
  engine_config: Json;
}

interface Props {
  mode: "create" | "revision" | "node";
  initial?: PlanRevision;
  defaultStages: Json[];
  tokenizers: Json[];
  mixtures: Mixture[];
  supervisionMixtures?: SupervisionMixture[];
  chatTemplates?: ChatTemplate[];
  codeVersions: Array<CodeVersion & { source_name: string }>;
  modelSpecifications: ModelSpecification[];
  validationSets: TrainingValidationSet[];
  workers: Worker[];
  submitting?: boolean;
  error?: unknown;
  submitLabel: string;
  onCancel: () => void;
  onSubmit: (payload: TrainingPlanPayload) => void;
  onImportCode?: () => void;
  onUploadValidation?: (name: string, file: File) => Promise<TrainingValidationSet>;
}

function engineConfig(revision?: PlanRevision): Json {
  const value = revision?.effective_config?.engine_config;
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function outputLines(outputs: unknown): string {
  if (!Array.isArray(outputs)) return "";
  return outputs
    .map((output) => `${output?.kind || "checkpoint"}=${output?.path || ""}`)
    .join("\n");
}

function parseOutputs(value: string): Json[] {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const separator = line.indexOf("=");
      return separator > 0
        ? { kind: line.slice(0, separator).trim(), path: line.slice(separator + 1).trim() }
        : { kind: "checkpoint", path: line };
    });
}

function codeEntrypoint(
  codeVersions: Array<CodeVersion & { source_name: string }>,
  codeVersionId: string,
) {
  return codeVersions.find((version) => version.id === codeVersionId)?.entrypoints?.training || "";
}

const defaultDataPreparation: Json = {
  corpus: {
    selection: { mode: "all", estimated_token_limit: null },
    encoding: {
      eos_policy: "append_if_available",
      empty_document_policy: "skip",
      token_dtype: "auto",
      shard_token_limit: 268_435_456,
    },
  },
  view: {
    consumption_policy: "unique_only",
    tail_policy: "drop",
    padding_token_id: null,
    packing_policy: "document_stream_v1",
    sampling_policy: "weighted_deficit_v1",
    sequence_order_policy: "block_shuffle_v1",
    shuffle_block_sequences: 16,
    seed: 42,
  },
  execution: {
    launcher_id: "builtin.local-process",
    resource_request: { gpus: 0, cpus: 4 },
  },
};

function normalizeDataPreparation(stage: Json): Json {
  const current = (stage.data_preparation as Json | undefined) || {};
  const corpus = (current.corpus as Json | undefined) || {};
  const execution = (current.execution as Json | undefined) || {};
  return {
    ...defaultDataPreparation,
    ...current,
    corpus: {
      ...(defaultDataPreparation.corpus as Json),
      ...corpus,
      selection: {
        ...((defaultDataPreparation.corpus as Json).selection as Json),
        ...((corpus.selection as Json | undefined) || {}),
      },
      encoding: {
        ...((defaultDataPreparation.corpus as Json).encoding as Json),
        ...((corpus.encoding as Json | undefined) || {}),
      },
    },
    view: {
      ...(defaultDataPreparation.view as Json),
      ...(((current.view as Json | undefined) || {})),
    },
    execution: {
      ...(defaultDataPreparation.execution as Json),
      ...execution,
      resource_request: {
        ...((defaultDataPreparation.execution as Json).resource_request as Json),
        ...(((execution.resource_request as Json | undefined) || {})),
      },
    },
  };
}

export function TrainingPlanConfigForm({
  mode,
  initial,
  defaultStages,
  tokenizers,
  mixtures,
  supervisionMixtures = [],
  chatTemplates = [],
  codeVersions,
  modelSpecifications,
  validationSets,
  workers,
  submitting,
  error,
  submitLabel,
  onCancel,
  onSubmit,
  onImportCode,
  onUploadValidation,
}: Props) {
  const initialEngine = engineConfig(initial);
  const readyMixtures = mixtures.filter(
    (mixture) => mixture.current_version?.status === "ready" && mixture.current_version?.manifest_artifact_id,
  );
  const readySupervisionMixtures = supervisionMixtures.flatMap((mixture) =>
    mixture.versions
      .filter((version) => version.status === "ready")
      .map((version) => ({ ...version, mixture_name: mixture.name })),
  );
  const chatTemplateVersions = chatTemplates.flatMap((template) =>
    template.versions.map((version) => ({ ...version, template_name: template.name })),
  );
  const initialModelSpecificationRevision = modelSpecifications
    .flatMap((item) => item.revisions)
    .find((revision) => revision.id === (
      initial?.model_specification_revision_id
      || modelSpecifications[0]?.current_revision?.id
    ));
  const initialCodeVersionId = initial?.code_version_id
    || codeVersions[0]?.id
    || "";
  const initialValidationSetId = initial?.validation_set_id || "";
  const initialValidationMixtureVersionId = initial?.validation_mixture_version_id
    || (mode === "create"
      ? readyMixtures.find((mixture) => /验证|測試|测试|validation|eval/i.test(mixture.name))?.current_version?.id || ""
      : "");
  const randomInit = !initial?.input_model_ref
    || !Object.keys(initial.input_model_ref).length
    || initial.input_model_ref.type === "random_init";
  const initialArgs = Array.isArray(initialEngine.args)
    ? initialEngine.args.map(String)
    : [
        codeEntrypoint(codeVersions, initialCodeVersionId),
        ...(initialValidationSetId ? ["--validation-file", "__validation_file__"] : []),
      ].filter(Boolean);
  const [name, setName] = useState("主训练计划");
  const [description, setDescription] = useState("预训练到后训练的阶段化流程");
  const [stages, setStages] = useState<Json[]>(
    (initial?.stages?.length ? initial.stages : defaultStages).map((stage) => normalizeStageGatesForEditor({
      ...stage,
      data_ref_id: stage.data_ref_id || readyMixtures[0]?.current_version?.id || "",
      data_preparation: normalizeDataPreparation(stage),
    })),
  );
  const [tokenizerVersionId, setTokenizerVersionId] = useState(
    initial?.tokenizer_version_id || tokenizers.find((item) => item.locked)?.id || "",
  );
  const [validationSetId, setValidationSetId] = useState(initialValidationSetId);
  const [validationMixtureVersionId, setValidationMixtureVersionId] = useState(
    initialValidationMixtureVersionId,
  );
  const [codeVersionId, setCodeVersionId] = useState(initialCodeVersionId);
  const [modelSpecificationRevisionId, setModelSpecificationRevisionId] = useState(
    initialModelSpecificationRevision?.id || "",
  );
  const [enginePluginId, setEnginePluginId] = useState(
    initial?.engine_plugin_id || "builtin.command.training",
  );
  const [launcherPluginId, setLauncherPluginId] = useState(
    initial?.launcher_plugin_id || "builtin.local-process",
  );
  const [gpus, setGpus] = useState(Number(initial?.resource_request?.gpus ?? 1));
  const [cpus, setCpus] = useState(Number(initial?.resource_request?.cpus ?? 8));
  const [precision, setPrecision] = useState(String(initial?.resource_request?.precision || "bf16"));
  const [workerId, setWorkerId] = useState(
    String(initial?.resource_request?.worker_selector?.worker_id || ""),
  );
  const [runtimeMode, setRuntimeMode] = useState(
    initialEngine.executable && initialEngine.executable !== "__worker_python__" ? "custom" : "worker-python",
  );
  const [customExecutable, setCustomExecutable] = useState(
    initialEngine.executable && initialEngine.executable !== "__worker_python__"
      ? String(initialEngine.executable)
      : "",
  );
  const [argsText, setArgsText] = useState(initialArgs.join("\n"));
  const [cwd, setCwd] = useState(String(initialEngine.cwd || ""));
  const [outputsText, setOutputsText] = useState(outputLines(initialEngine.outputs));
  const [uploadedSets, setUploadedSets] = useState<TrainingValidationSet[]>([]);
  const [uploadName, setUploadName] = useState("训练验证集");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<unknown>(null);
  const [uploading, setUploading] = useState(false);

  const allValidationSets = useMemo(() => {
    const byId = new Map(validationSets.map((item) => [item.id, item]));
    uploadedSets.forEach((item) => byId.set(item.id, item));
    return [...byId.values()];
  }, [uploadedSets, validationSets]);

  function updateStage(index: number, key: string, value: unknown) {
    setStages((current) =>
      current.map((stage, stageIndex) =>
        stageIndex === index ? { ...stage, [key]: value } : stage,
      ),
    );
  }

  function updateStageType(index: number, type: string) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const compatibleStage = { ...stage };
      if (type === "dpo") delete compatibleStage.optimizations;
      if (type === "sft" || type === "dpo") {
        const objective = type === "sft"
          ? { schema_version: 1, kind: "sft", loss_scope: "assistant_only", mask_policy: "template_generation_spans_v1" }
          : { schema_version: 1, kind: "dpo", loss_type: "sigmoid", beta: 0.1, label_smoothing: 0 };
        return {
          ...compatibleStage,
          type,
          objective,
          data_ref_id: readySupervisionMixtures.find((item) => item.kind === (type === "sft" ? "conversation" : "preference"))?.id || "",
          chat_template_version_id: chatTemplateVersions[0]?.id || "",
          budget: { unit: type === "sft" ? "supervised_tokens" : "preference_pairs", value: 100_000 },
          target_tokens: type === "sft" ? 100_000 : 0,
          supervision_preparation: {
            corpus: { max_length: Number(stage.sequence_length || 2048), overflow_policy: "drop_overflow_v1", min_completion_tokens: 1, token_dtype: "auto" },
            view: { consumption_policy: "unique_only", packing_policy: type === "sft" ? "no_packing_v1" : "none_v1", sample_order_policy: "global_shuffle_v1", sample_offset: 0, seed: 42 },
            execution: { launcher_id: "builtin.local-process", resource_request: { gpus: 0, cpus: 4 } },
          },
        };
      }
      return {
        ...compatibleStage,
        type,
        data_ref_id: readyMixtures[0]?.current_version?.id || "",
        target_tokens: Number(stage.target_tokens || 10_000_000),
      };
    }));
  }

  function updatePostTraining(index: number, section: "corpus" | "view", key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const preparation = (stage.supervision_preparation as Json | undefined) || {};
      return {
        ...stage,
        supervision_preparation: {
          ...preparation,
          [section]: { ...((preparation[section] as Json | undefined) || {}), [key]: value },
        },
      };
    }));
  }

  function updateObjective(index: number, key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => stageIndex === index
      ? { ...stage, objective: { ...((stage.objective as Json | undefined) || {}), [key]: value } }
      : stage));
  }

  function updateBudget(index: number, key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => stageIndex === index
      ? { ...stage, budget: { ...((stage.budget as Json | undefined) || {}), [key]: value } }
      : stage));
  }

  function replaceStage(index: number, stage: Json) {
    setStages((current) => current.map((item, stageIndex) => stageIndex === index ? stage : item));
  }

  function updatePreparation(index: number, section: "view" | "execution", key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const preparation = normalizeDataPreparation(stage);
      return { ...stage, data_preparation: { ...preparation, [section]: { ...(preparation[section] as Json), [key]: value } } };
    }));
  }

  function updateSequenceOrder(index: number, policy: string) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const preparation = normalizeDataPreparation(stage);
      const view = preparation.view as Json;
      return {
        ...stage,
        data_preparation: {
          ...preparation,
          view: {
            ...view,
            sequence_order_policy: policy,
            shuffle_block_sequences: policy === "block_shuffle_v1"
              ? Number(view.shuffle_block_sequences || 16)
              : null,
          },
        },
      };
    }));
  }

  function updateCorpusPreparation(index: number, section: "selection" | "encoding", key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const preparation = normalizeDataPreparation(stage);
      const corpus = preparation.corpus as Json;
      return { ...stage, data_preparation: { ...preparation, corpus: { ...corpus, [section]: { ...(corpus[section] as Json), [key]: value } } } };
    }));
  }

  function updatePreparationResource(index: number, key: string, value: unknown) {
    setStages((current) => current.map((stage, stageIndex) => {
      if (stageIndex !== index) return stage;
      const preparation = normalizeDataPreparation(stage);
      const execution = preparation.execution as Json;
      return { ...stage, data_preparation: { ...preparation, execution: { ...execution, resource_request: { ...(execution.resource_request as Json), [key]: value } } } };
    }));
  }

  function selectCodeVersion(nextId: string) {
    const previousEntrypoint = codeEntrypoint(codeVersions, codeVersionId);
    const nextEntrypoint = codeEntrypoint(codeVersions, nextId);
    setCodeVersionId(nextId);
    setArgsText((current) => {
      const lines = current.split("\n");
      if (!current.trim()) return nextEntrypoint;
      if (previousEntrypoint && lines[0] === previousEntrypoint) {
        return [nextEntrypoint, ...lines.slice(1)].filter(Boolean).join("\n");
      }
      return current;
    });
  }

  function selectModelSpecification(nextId: string) {
    setModelSpecificationRevisionId(nextId);
  }

  function selectValidationReference(reference: string) {
    const [kind, nextId = ""] = reference.split(":", 2);
    const fileSelected = kind === "file";
    setValidationSetId(fileSelected ? nextId : "");
    setValidationMixtureVersionId(kind === "mixture" ? nextId : "");
    if (kind === "mixture") {
      const currentCode = codeVersions.find((version) => version.id === codeVersionId);
      const compatibleCode = codeVersions.find((version) =>
        (!currentCode || version.source_id === currentCode.source_id)
        && Boolean(version.config_schema?.properties?.validation_mixture_version_id),
      );
      if (compatibleCode && compatibleCode.id !== codeVersionId) {
        selectCodeVersion(compatibleCode.id);
      }
    }
    setArgsText((current) => {
      const hasToken = current.includes("__validation_file__") || current.includes("${VALIDATION_FILE}");
      if (fileSelected && nextId && !hasToken) {
        return `${current.trim()}\n--validation-file\n__validation_file__`.trim();
      }
      if (kind === "mixture") {
        const lines = current.split("\n");
        return lines.filter((line, index) => {
          if (line.trim() === "__validation_file__" || line.trim() === "${VALIDATION_FILE}") return false;
          if (line.trim() === "--validation-file" && [lines[index + 1]?.trim()].some((value) => value === "__validation_file__" || value === "${VALIDATION_FILE}")) return false;
          return true;
        }).join("\n");
      }
      return current;
    });
  }

  function buildPayload(): TrainingPlanPayload {
    const resourceRequest: Json = {
      ...(initial?.resource_request || {}),
      gpus,
      cpus,
      precision,
    };
    if (workerId) resourceRequest.worker_selector = { worker_id: workerId };
    else delete resourceRequest.worker_selector;
    const config = {
      ...initialEngine,
      executable: runtimeMode === "worker-python" ? "__worker_python__" : customExecutable,
      args: argsText.split("\n").map((value) => value.trim()).filter(Boolean),
      cwd: cwd || null,
      env: initialEngine.env || {},
      outputs: parseOutputs(outputsText),
    };
    return {
      ...(mode === "create" ? { name, description } : {}),
      stages: stages.map(({ training_data_view_id: _legacyViewId, ...stage }) => stage),
      input_model_ref: initial?.input_model_ref || { type: "random_init" },
      model_specification_revision_id: modelSpecificationRevisionId || null,
      tokenizer_version_id: tokenizerVersionId || null,
      validation_set_id: validationSetId || null,
      validation_mixture_version_id: validationMixtureVersionId || null,
      code_version_id: codeVersionId || null,
      engine_plugin_id: enginePluginId,
      launcher_plugin_id: launcherPluginId,
      resource_request: resourceRequest,
      engine_config: config,
    };
  }

  const preview = buildPayload();

  async function uploadValidation() {
    if (!onUploadValidation || !uploadFile || !uploadName.trim()) return;
    setUploading(true);
    setUploadError(null);
    try {
      const created = await onUploadValidation(uploadName.trim(), uploadFile);
      setUploadedSets((current) => [created, ...current]);
      selectValidationReference(`file:${created.id}`);
      setUploadFile(null);
    } catch (caught) {
      setUploadError(caught);
    } finally {
      setUploading(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit(buildPayload());
  }

  return <form className="structured-training-form" onSubmit={submit}>
    {mode === "revision" && <div className="revision-notice"><Save size={16} /><div><strong>保存为新的不可变 Revision</strong><p>原修订版和历史运行不会被修改。</p></div></div>}
    <div className="form-grid two-columns">
      {mode === "create" && <>
        <Field label="计划名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field>
        <Field label="说明"><input value={description} onChange={(event) => setDescription(event.target.value)} /></Field>
      </>}
      <Field label="Tokenizer" hint="只显示已锁定、可复现的版本">
        <select aria-label="Tokenizer" required value={tokenizerVersionId} onChange={(event) => setTokenizerVersionId(event.target.value)}>
          <option value="">请选择 Tokenizer</option>
          {tokenizers.filter((item) => item.locked).map((item) => <option value={item.id} key={item.id}>{item.name} · V{item.version}</option>)}
        </select>
      </Field>
      <Field label="模型规格" hint="选择已解析的不可变架构配置；训练启动会冻结完整 JSON 和签名">
        <select aria-label="模型规格" required={Boolean(modelSpecifications.length)} disabled={!modelSpecifications.length} value={modelSpecificationRevisionId} onChange={(event) => selectModelSpecification(event.target.value)}>
          <option value="">{modelSpecifications.length ? "请选择模型规格" : "尚未创建模型规格（沿用代码配置）"}</option>
          {modelSpecifications.map((specification) => <optgroup label={specification.name} key={specification.id}>{specification.revisions.map((revision) => <option value={revision.id} key={revision.id}>V{revision.revision} · {revision.architecture_family_id} · {formatNumber(revision.parameter_summary.total_parameters)} 参数</option>)}</optgroup>)}
        </select>
      </Field>
      <Field label="训练验证集" hint="可直接选择已解析的数据混合；平台会切换到支持该协议的最新同源代码版本">
        <select aria-label="训练验证集" required value={validationSetId ? `file:${validationSetId}` : validationMixtureVersionId ? `mixture:${validationMixtureVersionId}` : ""} onChange={(event) => selectValidationReference(event.target.value)}>
          <option value="">请选择训练验证集</option>
          <optgroup label="数据中心 · 已就绪数据混合">{readyMixtures.map((mixture) => <option value={`mixture:${mixture.current_version!.id}`} key={mixture.current_version!.id}>{mixture.name} · V{mixture.current_version!.version} · 目标 {formatNumber(mixture.current_version!.target_tokens)} tokens</option>)}</optgroup>
          <optgroup label="上传的固定验证文件">{allValidationSets.filter((item) => item.status === "ready").map((item) => <option value={`file:${item.id}`} key={item.id}>{item.name} · V{item.version} · {formatNumber(item.estimated_tokens)} tokens</option>)}</optgroup>
        </select>
      </Field>
      <div className="validation-upload span-two">
        <Field label="上传新的验证集版本" hint="UTF-8 JSONL、NDJSON 或 TXT；同名上传会生成新版本">
          <input aria-label="验证集名称" value={uploadName} onChange={(event) => setUploadName(event.target.value)} />
        </Field>
        <input aria-label="选择验证集文件" type="file" accept=".jsonl,.ndjson,.txt" onChange={(event) => setUploadFile(event.target.files?.[0] || null)} />
        <button type="button" className="button secondary small" disabled={!uploadFile || uploading} onClick={uploadValidation}><Upload size={14} />{uploading ? "上传中…" : "上传并选中"}</button>
      </div>
      {Boolean(uploadError) && <div className="span-two"><ErrorMessage error={uploadError} /></div>}
      <Field label="训练代码版本" hint="只负责训练 Harness；模型实现由上方模型规格独立绑定">
        <select aria-label="训练代码版本" value={codeVersionId} onChange={(event) => selectCodeVersion(event.target.value)}>
          <option value="">不绑定训练代码版本</option>
          {codeVersions.map((version) => <option value={version.id} key={version.id}>{version.source_name} · V{version.version}{version.git_commit ? ` · ${version.git_commit.slice(0, 8)}` : ""}</option>)}
        </select>
      </Field>
      {mode !== "node" && <Field label="起始权重" hint="来源由模型规格冻结；历史计划继续保留原模型引用">
        <select aria-label="起始模型" value={randomInit ? "random" : "bound"} disabled>
          <option value="random">按模型规格初始化</option>
          <option value="bound">保留当前模型引用</option>
        </select>
      </Field>}
      {!codeVersions.length && <div className="training-code-empty span-two" role="status"><div><strong>还没有可绑定的训练代码版本</strong><span>请先导入并冻结训练代码，平台才能自动解析入口。</span></div>{onImportCode && <button type="button" className="button secondary small" onClick={onImportCode}><Upload size={14} />导入训练代码</button>}</div>}
      <Field label="训练引擎"><select aria-label="训练引擎" value={enginePluginId} onChange={(event) => setEnginePluginId(event.target.value)}><option value="builtin.command.training">内置命令训练引擎</option></select></Field>
      <Field label="Launcher"><select aria-label="Launcher" value={launcherPluginId} onChange={(event) => setLauncherPluginId(event.target.value)}><option value="builtin.local-process">本地进程</option></select></Field>
      <Field label="执行 Worker" hint="自动表示由系统选择任意满足资源和能力的在线 Worker"><select aria-label="执行 Worker" value={workerId} onChange={(event) => setWorkerId(event.target.value)}><option value="">自动选择兼容 Worker</option>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name} · {worker.status}</option>)}</select></Field>
      <Field label="精度"><select aria-label="精度" value={precision} onChange={(event) => setPrecision(event.target.value)}><option value="bf16">bf16</option><option value="fp16">fp16</option><option value="fp32">fp32</option></select></Field>
      <Field label="GPU 数量"><input aria-label="GPU 数量" type="number" min="0" value={gpus} onChange={(event) => setGpus(Number(event.target.value))} /></Field>
      <Field label="CPU 数量"><input aria-label="CPU 数量" type="number" min="1" value={cpus} onChange={(event) => setCpus(Number(event.target.value))} /></Field>
    </div>

    <div className="stage-editor"><header><div><h3>训练阶段</h3><p>每个阶段直接声明目标、数据、预算和执行参数；运行提示不会阻止启动。</p></div><button type="button" className="button secondary small" onClick={() => setStages((current) => [...current, normalizeStageGatesForEditor({ name: `Stage ${current.length} 新阶段`, type: "pretrain", target_tokens: 10_000_000, sequence_length: 4096, learning_rate: 0.0003, scheduler: "cosine_with_min_lr", warmup_ratio: 0.015, min_learning_rate: 0.00003, evaluation_interval_tokens: 0, checkpoint_interval_tokens: 0, evaluation_at_end: true, checkpoint_at_end: true, data_ref_id: readyMixtures[0]?.current_version?.id || "", data_preparation: normalizeDataPreparation({}) })])}><Plus size={15} />增加阶段</button></header>
      {stages.map((stage, index) => {
        const isPostTraining = ["sft", "dpo"].includes(String(stage.type));
        const isDpo = stage.type === "dpo";
        const objective = (stage.objective as Json | undefined) || {};
        const budget = (stage.budget as Json | undefined) || {};
        const supervisionPreparation = (stage.supervision_preparation as Json | undefined) || {};
        const supervisionCorpus = (supervisionPreparation.corpus as Json | undefined) || {};
        const supervisionView = (supervisionPreparation.view as Json | undefined) || {};
        const requiredSupervisionKind = isDpo ? "preference" : "conversation";
        const preparation = normalizeDataPreparation(stage);
        const corpus = preparation.corpus as Json;
        const selection = corpus.selection as Json;
        const encoding = corpus.encoding as Json;
        const view = preparation.view as Json;
        const execution = preparation.execution as Json;
        const preparationResource = execution.resource_request as Json;
        return <article key={index} className="stage-card">
        <div className="stage-card-heading">
          <div className="stage-number">{index + 1}</div>
          <div><strong>阶段 {index + 1}</strong><span>{String(stage.name || "未命名阶段")}</span></div>
        </div>
        {stages.length > 1 && <button type="button" className="icon-button stage-delete" title="删除此阶段" aria-label={`删除阶段 ${index + 1}`} onClick={() => setStages((current) => current.filter((_, stageIndex) => stageIndex !== index))}><Trash2 size={16} /></button>}
        <div className="stage-body">
          <div className="stage-fields structured-stage-fields">
            <label className="stage-name-field"><span>阶段名称</span><input aria-label={`阶段 ${index + 1} 名称`} value={String(stage.name || "")} onChange={(event) => updateStage(index, "name", event.target.value)} /></label>
            <label><span>训练类型</span><select aria-label={`阶段 ${index + 1} 类型`} value={String(stage.type || "pretrain")} onChange={(event) => updateStageType(index, event.target.value)}><option value="pretrain">预训练</option><option value="annealing">退火</option><option value="sft">SFT</option><option value="dpo">DPO</option></select></label>
            <label className="stage-data-field"><span>{isPostTraining ? "后训练数据混合" : "训练数据混合"}</span><select aria-label={`阶段 ${index + 1} 数据混合`} required value={String(stage.data_ref_id || "")} onChange={(event) => updateStage(index, "data_ref_id", event.target.value)}><option value="">请选择</option>{isPostTraining ? readySupervisionMixtures.filter((item) => item.kind === requiredSupervisionKind).map((item) => <option value={item.id} key={item.id}>{item.mixture_name} · V{item.version} · {item.kind}</option>) : readyMixtures.map((mixture) => <option value={mixture.current_version!.id} key={mixture.id}>{mixture.name} · V{mixture.current_version!.version}</option>)}</select></label>
            {!isPostTraining && <label><span>目标 Tokens</span><input aria-label={`阶段 ${index + 1} 目标 Tokens`} type="number" min="1" value={Number(stage.target_tokens || 0)} onChange={(event) => updateStage(index, "target_tokens", Number(event.target.value))} /></label>}
            <label><span>Sequence</span><input aria-label={`阶段 ${index + 1} Sequence`} type="number" min="16" max="65535" value={Number(stage.sequence_length || 4096)} onChange={(event) => updateStage(index, "sequence_length", Number(event.target.value))} /></label>
            <label><span>Learning rate</span><input aria-label={`阶段 ${index + 1} Learning rate`} type="number" min="0" step="0.000001" value={Number(stage.learning_rate || 0)} onChange={(event) => updateStage(index, "learning_rate", Number(event.target.value))} /></label>
            {!isPostTraining && <>
              <label><span>LR Scheduler</span><select aria-label={`阶段 ${index + 1} LR Scheduler`} value={String(stage.scheduler || "cosine_with_min_lr")} onChange={(event) => updateStage(index, "scheduler", event.target.value)}><option value="cosine_with_min_lr">Cosine（含最低 LR）</option><option value="cosine">Cosine</option><option value="linear">Linear</option><option value="constant">Constant</option></select></label>
              <label><span>预热比例</span><input aria-label={`阶段 ${index + 1} Warmup ratio`} type="number" min="0" max="0.999" step="0.001" value={Number(stage.warmup_ratio ?? 0.015)} onChange={(event) => updateStage(index, "warmup_ratio", Number(event.target.value))} /><small>按优化器总步数向上取整；例如 0.015 表示 1.5%。</small></label>
              <label><span>Minimum learning rate</span><input aria-label={`阶段 ${index + 1} Minimum learning rate`} type="number" min="0" step="0.000001" disabled={String(stage.scheduler || "cosine_with_min_lr") !== "cosine_with_min_lr"} value={Number(stage.min_learning_rate ?? 0.00003)} onChange={(event) => updateStage(index, "min_learning_rate", Number(event.target.value))} /></label>
            </>}
          </div>
          {isPostTraining && <section className="training-data-config">
            <header><div><strong>{isDpo ? "DPO 偏好优化" : "SFT 监督微调"}</strong><span>目标、模板、预算和监督数据制品会一起冻结到运行快照。</span></div></header>
            <div className="form-subsection"><h4>目标与预算</h4><div className="stage-fields structured-stage-fields">
              <label><span>Chat Template</span><select aria-label={`阶段 ${index + 1} Chat Template`} required value={String(stage.chat_template_version_id || "")} onChange={(event) => updateStage(index, "chat_template_version_id", event.target.value)}><option value="">请选择模板版本</option>{chatTemplateVersions.map((item) => <option value={item.id} key={item.id}>{item.template_name} · V{item.version}</option>)}</select></label>
              <label><span>预算单位</span><select aria-label={`阶段 ${index + 1} 预算单位`} value={String(budget.unit || (isDpo ? "preference_pairs" : "supervised_tokens"))} onChange={(event) => updateBudget(index, "unit", event.target.value)}>{isDpo ? <option value="preference_pairs">偏好对</option> : <><option value="supervised_tokens">监督 Tokens</option><option value="examples">样本</option></>}<option value="steps">优化器 Steps</option></select></label>
              <label><span>预算</span><input aria-label={`阶段 ${index + 1} 后训练预算`} type="number" min="1" value={Number(budget.value || 0)} onChange={(event) => { const value = Number(event.target.value); updateBudget(index, "value", value); if (!isDpo) updateStage(index, "target_tokens", value); }} /></label>
              {isDpo && <><label><span>Beta</span><input aria-label={`阶段 ${index + 1} DPO Beta`} type="number" min="0.000001" step="0.01" value={Number(objective.beta || 0.1)} onChange={(event) => updateObjective(index, "beta", Number(event.target.value))} /></label><label><span>Label smoothing</span><input aria-label={`阶段 ${index + 1} DPO Label smoothing`} type="number" min="0" max="0.49" step="0.01" value={Number(objective.label_smoothing || 0)} onChange={(event) => updateObjective(index, "label_smoothing", Number(event.target.value))} /></label></>}
            </div></div>
            <div className="form-subsection"><h4>监督 Corpus</h4><div className="stage-fields structured-stage-fields">
              <label><span>超长样本</span><select aria-label={`阶段 ${index + 1} 超长样本策略`} value={String(supervisionCorpus.overflow_policy || "drop_overflow_v1")} onChange={(event) => updatePostTraining(index, "corpus", "overflow_policy", event.target.value)}><option value="drop_overflow_v1">丢弃并记录</option><option value="conversation_tail_v1">保留最近对话轮次</option><option value="prompt_completion_v1">优先保留 completion</option></select></label>
              <label><span>最短 Completion</span><input aria-label={`阶段 ${index + 1} 最短 Completion`} type="number" min="1" value={Number(supervisionCorpus.min_completion_tokens || 1)} onChange={(event) => updatePostTraining(index, "corpus", "min_completion_tokens", Number(event.target.value))} /></label>
              <label><span>数据不足时</span><select aria-label={`阶段 ${index + 1} 后训练消费策略`} value={String(supervisionView.consumption_policy || "unique_only")} onChange={(event) => updatePostTraining(index, "view", "consumption_policy", event.target.value)}><option value="unique_only">只用唯一样本</option><option value="repeat_epochs">允许重复 Epoch</option></select></label>
              <label><span>Packing</span><select aria-label={`阶段 ${index + 1} 后训练 Packing`} value={String(supervisionView.packing_policy || (isDpo ? "none_v1" : "no_packing_v1"))} onChange={(event) => updatePostTraining(index, "view", "packing_policy", event.target.value)}>{isDpo ? <option value="none_v1">偏好对不 Packing</option> : <><option value="no_packing_v1">单样本序列</option><option value="isolated_example_packing_v1">隔离样本 Packing</option></>}</select></label>
              <label className="check-field"><input aria-label={`阶段 ${index + 1} 全局随机打散`} type="checkbox" checked={String(supervisionView.sample_order_policy || "sequential_v1") === "global_shuffle_v1"} onChange={(event) => updatePostTraining(index, "view", "sample_order_policy", event.target.checked ? "global_shuffle_v1" : "sequential_v1")} /><span><b>全局随机打散</b><small>以完整对话或偏好对为单位冻结排列</small></span></label>
              <label><span>打散 Seed</span><input aria-label={`阶段 ${index + 1} 后训练打散 Seed`} type="number" value={Number(supervisionView.seed ?? 42)} onChange={(event) => updatePostTraining(index, "view", "seed", Number(event.target.value))} /></label>
            </div></div>
            <details className="collapsed-config"><summary>查看原生后训练配置</summary><pre className="json-preview">{JSON.stringify({ objective, supervision_mixture_version_id: stage.data_ref_id, chat_template_version_id: stage.chat_template_version_id, budget, supervision_preparation: supervisionPreparation }, null, 2)}</pre></details>
          </section>}
          {!isPostTraining && <section className="training-data-config">
            <header><div><strong>训练数据配置</strong><span>系统按此配置自动复用或构建编码数据和读取索引；最终值全部公开。混合含筛选规则时，以冻结的入选文档为准，优先复用已有编码语料；下方范围截取仅用于未筛选来源。</span></div></header>
            <div className="form-subsection"><h4>训练语义</h4><div className="stage-fields structured-stage-fields">
              <label><span>数据不足时</span><select aria-label={`阶段 ${index + 1} 数据不足策略`} value={String(view.consumption_policy)} onChange={(event) => updatePreparation(index, "view", "consumption_policy", event.target.value)}><option value="unique_only">只用唯一数据；不足时由我调整目标</option><option value="repeat_epochs">允许按 Epoch 重复直到目标</option></select></label>
              <label><span>序列尾部</span><select aria-label={`阶段 ${index + 1} 尾部策略`} value={String(view.tail_policy)} onChange={(event) => updatePreparation(index, "view", "tail_policy", event.target.value)}><option value="drop">丢弃不足一个序列的尾部</option><option value="pad">Padding 保留尾部</option></select></label>
              <label><span>Padding Token ID</span><input aria-label={`阶段 ${index + 1} Padding Token ID`} type="number" min="0" required={view.tail_policy === "pad"} disabled={view.tail_policy !== "pad"} value={view.padding_token_id == null ? "" : Number(view.padding_token_id)} placeholder="pad 时必须明确填写" onChange={(event) => updatePreparation(index, "view", "padding_token_id", event.target.value === "" ? null : Number(event.target.value))} /></label>
              <label><span>Packing</span><select aria-label={`阶段 ${index + 1} Packing 策略`} value={String(view.packing_policy)} onChange={(event) => updatePreparation(index, "view", "packing_policy", event.target.value)}><option value="document_stream_v1">文档连续流 V1</option></select></label>
              <label><span>混合采样</span><select aria-label={`阶段 ${index + 1} 采样策略`} value={String(view.sampling_policy)} onChange={(event) => updatePreparation(index, "view", "sampling_policy", event.target.value)}><option value="weighted_deficit_v1">加权赤字调度 V1</option></select></label>
              <label><span>序列顺序</span><select aria-label={`阶段 ${index + 1} 序列顺序`} value={String(view.sequence_order_policy || "block_shuffle_v1")} onChange={(event) => updateSequenceOrder(index, event.target.value)}><option value="block_shuffle_v1">块间随机、块内顺序（低空间）</option><option value="materialized_global_shuffle_v1">全局随机导出、训练顺序读（高空间）</option><option value="sequential_v1">全局顺序读取（兼容）</option></select></label>
              <label><span>每块 Sequences</span><input aria-label={`阶段 ${index + 1} 随机块大小`} type="number" min="1" max="4096" required={view.sequence_order_policy === "block_shuffle_v1"} disabled={view.sequence_order_policy !== "block_shuffle_v1"} value={view.shuffle_block_sequences == null ? "" : Number(view.shuffle_block_sequences)} onChange={(event) => updatePreparation(index, "view", "shuffle_block_sequences", Number(event.target.value))} /></label>
              <label><span>采样 Seed</span><input aria-label={`阶段 ${index + 1} 采样 Seed`} type="number" value={Number(view.seed)} onChange={(event) => updatePreparation(index, "view", "seed", Number(event.target.value))} /></label>
            </div></div>
            <div className="form-subsection"><h4>编码语义</h4><div className="stage-fields structured-stage-fields">
              <label><span>Corpus 范围</span><select aria-label={`阶段 ${index + 1} Corpus 范围`} value={String(selection.mode)} onChange={(event) => updateCorpusPreparation(index, "selection", "mode", event.target.value)}><option value="all">编码全部文档</option><option value="estimated_token_limit">按估算 Token 上限截取</option></select></label>
              <label><span>估算 Token 上限</span><input aria-label={`阶段 ${index + 1} Corpus Token 上限`} type="number" min="1" disabled={selection.mode !== "estimated_token_limit"} value={selection.estimated_token_limit == null ? "" : Number(selection.estimated_token_limit)} onChange={(event) => updateCorpusPreparation(index, "selection", "estimated_token_limit", event.target.value === "" ? null : Number(event.target.value))} /></label>
              <label><span>EOS</span><select aria-label={`阶段 ${index + 1} EOS 策略`} value={String(encoding.eos_policy)} onChange={(event) => updateCorpusPreparation(index, "encoding", "eos_policy", event.target.value)}><option value="append_if_available">有 EOS 时追加</option><option value="append_required">必须追加 EOS</option><option value="none">不追加</option></select></label>
              <label><span>空文档</span><select aria-label={`阶段 ${index + 1} 空文档策略`} value={String(encoding.empty_document_policy)} onChange={(event) => updateCorpusPreparation(index, "encoding", "empty_document_policy", event.target.value)}><option value="skip">跳过</option><option value="eos_only">写入单个 EOS</option></select></label>
              <label><span>Token dtype</span><select aria-label={`阶段 ${index + 1} Token dtype`} value={String(encoding.token_dtype)} onChange={(event) => updateCorpusPreparation(index, "encoding", "token_dtype", event.target.value)}><option value="auto">自动（按词表）</option><option value="uint16">uint16</option><option value="uint32">uint32</option></select></label>
              <label><span>每分片 Token 上限</span><input aria-label={`阶段 ${index + 1} 分片 Token 上限`} type="number" min="1048576" value={Number(encoding.shard_token_limit)} onChange={(event) => updateCorpusPreparation(index, "encoding", "shard_token_limit", Number(event.target.value))} /></label>
            </div></div>
            <div className="form-subsection"><h4>准备任务执行</h4><div className="stage-fields structured-stage-fields">
              <label><span>Launcher</span><select aria-label={`阶段 ${index + 1} 数据准备 Launcher`} value={String(execution.launcher_id)} onChange={(event) => updatePreparation(index, "execution", "launcher_id", event.target.value)}><option value="builtin.local-process">本地进程</option></select></label>
              <label><span>CPU 数量</span><input aria-label={`阶段 ${index + 1} 数据准备 CPU`} type="number" min="1" value={Number(preparationResource.cpus || 4)} onChange={(event) => updatePreparationResource(index, "cpus", Number(event.target.value))} /></label>
              <label><span>执行 Worker</span><select aria-label={`阶段 ${index + 1} 数据准备 Worker`} value={String((preparationResource.worker_selector as Json | undefined)?.worker_id || "")} onChange={(event) => updatePreparationResource(index, "worker_selector", event.target.value ? { worker_id: event.target.value } : undefined)}><option value="">自动选择可访问数据的 Worker</option>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name} · {worker.status}</option>)}</select></label>
              <label><span>GPU 数量</span><input readOnly aria-label={`阶段 ${index + 1} 数据准备 GPU`} type="number" value="0" /></label>
            </div></div>
            <details className="collapsed-config"><summary>查看本阶段有效数据配置</summary><pre className="json-preview">{JSON.stringify({ mixture_version_id: stage.data_ref_id, tokenizer_version_id: tokenizerVersionId, sequence_length: stage.sequence_length, requested_tokens: stage.target_tokens, data_preparation: preparation }, null, 2)}</pre></details>
          </section>}
          <StageGateEditor stage={stage} stageIndex={index} onChange={(nextStage) => replaceStage(index, nextStage)} />
          {!isDpo && <TrainingOptimizationEditor value={stage.optimizations} onChange={(value) => { const next = { ...stage }; if (value) next.optimizations = value; else delete next.optimizations; replaceStage(index, next); }} />}
        </div>
      </article>;})}
    </div>

    <details className="advanced-training-config"><summary>高级命令配置</summary><div className="form-grid two-columns">
      <Field label="Python 运行时"><select aria-label="Python 运行时" value={runtimeMode} onChange={(event) => setRuntimeMode(event.target.value)}><option value="worker-python">自动使用 Worker Python</option><option value="custom">自定义可执行文件</option></select></Field>
      {runtimeMode === "custom" ? <Field label="自定义可执行文件"><input required value={customExecutable} onChange={(event) => setCustomExecutable(event.target.value)} /></Field> : <Field label="解析结果"><input readOnly value="由 Worker 解析为自身 Python 环境" /></Field>}
      <Field label="代码内执行目录" hint="留空时使用代码版本根目录"><input value={cwd} onChange={(event) => setCwd(event.target.value)} placeholder="例如 src/training" /></Field>
      <span />
      <Field label="命令参数" hint="每行一个参数；代码入口和验证文件占位符由平台预填"><textarea value={argsText} onChange={(event) => setArgsText(event.target.value)} /></Field>
      <Field label="输出制品" hint="每行 kind=相对路径"><textarea value={outputsText} onChange={(event) => setOutputsText(event.target.value)} placeholder={'checkpoint=outputs/checkpoint-final\ntraining_log=outputs/train.jsonl'} /></Field>
    </div></details>

    <details className="collapsed-config"><summary>查看完整 JSON（只读）</summary><pre className="json-preview">{JSON.stringify(preview, null, 2)}</pre></details>
    {Boolean(error) && <ErrorMessage error={error} />}
    <FormActions submitting={submitting} onCancel={onCancel} submitLabel={submitLabel} />
  </form>;
}
