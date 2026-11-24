import { useMutation } from "@tanstack/react-query";
import { Braces, Code2, Play, Save, SlidersHorizontal } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { formatNumber, post } from "../api";
import type {
  CodeVersion,
  Json,
  ModelArchitecture,
  ModelArchitectureField,
  ModelSpecification,
  ModelSpecificationRevision,
} from "../types";
import { Alert, ErrorMessage, Field, FormActions } from "./ui";

interface Props {
  projectId: string;
  architectures: ModelArchitecture[];
  codeVersions: Array<CodeVersion & { source_name: string }>;
  specification?: ModelSpecification;
  baseRevision?: ModelSpecificationRevision;
  onCancel: () => void;
  onSaved: (specification: ModelSpecification) => void;
}

function defaults(fields: ModelArchitectureField[]): Json {
  return Object.fromEntries(fields.map((field) => [field.key, field.default]));
}

function schemaFields(version?: CodeVersion): ModelArchitectureField[] {
  const properties = version?.config_schema?.properties;
  if (!properties || typeof properties !== "object" || Array.isArray(properties)) return [];
  const reserved = new Set(["architecture_name", "vocab_size", "total_parameters", "active_parameters", "trainable_parameters"]);
  return Object.entries(properties).flatMap(([key, rawValue]) => {
    if (reserved.has(key) || !rawValue || typeof rawValue !== "object" || Array.isArray(rawValue)) return [];
    const raw = rawValue as Json;
    const enumValues = Array.isArray(raw.enum) ? raw.enum : [];
    const rawType = String(raw.type || "string");
    const type: ModelArchitectureField["type"] = enumValues.length
      ? "select"
      : rawType === "integer" ? "integer"
      : rawType === "number" ? "number"
      : rawType === "boolean" ? "boolean"
      : rawType === "object" ? "json"
      : rawType === "array" && raw.items?.type === "integer" ? "integer_list"
      : "text";
    const fallback = type === "boolean" ? false : type === "json" ? {} : type === "integer_list" ? [] : type === "integer" || type === "number" ? 0 : "";
    return [{
      key,
      label: String(raw.title || key),
      group: String(raw["x-group"] || "自定义模型参数"),
      type,
      default: raw.default ?? fallback,
      minimum: typeof raw.minimum === "number" ? raw.minimum : undefined,
      maximum: typeof raw.maximum === "number" ? raw.maximum : undefined,
      step: typeof raw.multipleOf === "number" ? raw.multipleOf : undefined,
      hint: String(raw.description || ""),
      advanced: Boolean(raw["x-advanced"]),
      options: enumValues.map((value) => ({ value: String(value), label: String(value) })),
    }];
  });
}

function isVisible(field: ModelArchitectureField, config: Json) {
  if (!field.visible_when) return true;
  const current = config[field.visible_when.field];
  if (field.visible_when.in) return field.visible_when.in.includes(current);
  return current === field.visible_when.equals;
}

export function ModelSpecificationEditor({
  projectId,
  architectures,
  codeVersions,
  specification,
  baseRevision,
  onCancel,
  onSaved,
}: Props) {
  const initialArchitecture = architectures.find((item) => item.id === baseRevision?.architecture_family_id)
    || architectures.find((item) => item.id === "llama_dense")
    || architectures[0];
  const [name, setName] = useState(specification?.name || "我的模型规格");
  const [description, setDescription] = useState(specification?.description || "可执行、可复现的模型定义");
  const [architectureId, setArchitectureId] = useState(initialArchitecture?.id || "");
  const [codeVersionId, setCodeVersionId] = useState(baseRevision?.code_version_id || "");
  const initialCodeVersion = codeVersions.find((item) => item.id === baseRevision?.code_version_id);
  const initialFields = [...(initialArchitecture?.fields || []), ...schemaFields(initialCodeVersion)];
  const [config, setConfig] = useState<Json>({ ...defaults(initialFields), ...(baseRevision?.user_config || {}) });
  const [initializationType, setInitializationType] = useState(String(baseRevision?.initialization?.type || "random"));
  const [initializationReference, setInitializationReference] = useState(String(baseRevision?.initialization?.reference || ""));
  const [previewMode, setPreviewMode] = useState<"json" | "code">("json");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const architecture = architectures.find((item) => item.id === architectureId);
  const selectedCodeVersion = codeVersions.find((item) => item.id === codeVersionId);
  const customSchemaFields = architecture?.source_type === "custom_code" ? schemaFields(selectedCodeVersion) : [];
  const fields = useMemo(
    () => [...(architecture?.fields || []), ...customSchemaFields],
    [architecture, selectedCodeVersion],
  );
  const visibleFields = fields.filter((field) => isVisible(field, config) && (showAdvanced || !field.advanced));
  const groups = [...new Set(visibleFields.map((field) => field.group))];
  const categories = [...new Set(architectures.map((item) => item.category || "其他实现"))];
  const compatibleModelCode = codeVersions.filter((version) => Boolean(version.entrypoints?.model_factory) && Boolean(version.config_schema?.properties));

  const resolvePayload = {
    source_type: architecture?.source_type || "standard",
    architecture_family_id: architectureId,
    preset_id: null,
    code_version_id: architecture?.source_type === "custom_code" ? codeVersionId || null : null,
    config,
    initialization: {
      type: initializationType,
      ...(initializationType !== "random" ? { reference: initializationReference } : {}),
    },
  };
  const inputKey = JSON.stringify(resolvePayload);
  const preview = useMutation({
    mutationFn: async () => ({
      inputKey,
      result: await post<any>(`/projects/${projectId}/model-specifications/resolve`, resolvePayload),
    }),
  });
  const save = useMutation({
    mutationFn: () => specification
      ? post<ModelSpecification>(`/model-specifications/${specification.id}/revisions`, resolvePayload)
      : post<ModelSpecification>(`/projects/${projectId}/model-specifications`, { name, description, ...resolvePayload }),
    onSuccess: onSaved,
  });
  const result = preview.data?.inputKey === inputKey ? preview.data.result : undefined;

  function selectArchitecture(nextId: string) {
    const next = architectures.find((item) => item.id === nextId);
    setArchitectureId(nextId);
    setCodeVersionId("");
    setConfig(defaults(next?.fields || []));
    preview.reset();
  }

  function selectModelCode(nextId: string) {
    const nextVersion = codeVersions.find((item) => item.id === nextId);
    setCodeVersionId(nextId);
    setConfig({ ...defaults(architecture?.fields || []), ...defaults(schemaFields(nextVersion)) });
    preview.reset();
  }

  function updateField(key: string, value: unknown) {
    setConfig((current) => ({ ...current, [key]: value }));
    preview.reset();
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    save.mutate();
  }

  return <form className="model-spec-form" onSubmit={submit}>
    {specification && <div className="revision-notice"><Save size={16} /><div><strong>保存为新的不可变 Revision</strong><p>当前 V{baseRevision?.revision} 和已关联训练不会被修改。</p></div></div>}
    {!specification && <div className="form-grid two-columns">
      <Field label="规格名称"><input required value={name} onChange={(event) => setName(event.target.value)} /></Field>
      <Field label="说明"><input value={description} onChange={(event) => setDescription(event.target.value)} /></Field>
    </div>}
    <section className="model-spec-section">
      <header><div><span>01</span><div><h3>选择可执行实现</h3><p>架构族决定真实 Config 类、模型构造器和可编辑的关键结构字段。</p></div></div></header>
      <div className="form-grid two-columns">
        <Field label="架构实现族" hint={architecture?.description}>
          <select aria-label="架构实现族" value={architectureId} onChange={(event) => selectArchitecture(event.target.value)}>
            {categories.map((category) => <optgroup label={category} key={category}>{architectures.filter((item) => (item.category || "其他实现") === category).map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</optgroup>)}
          </select>
        </Field>
        <div className="model-implementation-summary">
          <span>实现绑定</span>
          <strong>{String(architecture?.implementation?.config_class || architecture?.implementation?.kind || "—")}</strong>
          <small>{architecture?.starting_point?.description}</small>
        </div>
        {architecture?.source_type === "custom_code" && <Field label="模型实现代码版本" hint="必须声明 model_factory 入口和模型 config_schema；训练代码在训练计划中另选。"><select aria-label="模型实现代码版本" required value={codeVersionId} onChange={(event) => selectModelCode(event.target.value)}><option value="">请选择兼容代码版本</option>{compatibleModelCode.map((version) => <option value={version.id} key={version.id}>{version.source_name} · V{version.version}{version.git_commit ? ` · ${version.git_commit.slice(0, 8)}` : ""}</option>)}</select></Field>}
      </div>
      {architecture?.source_type === "custom_code" && !compatibleModelCode.length && <Alert severity="warning" title="没有可用的模型实现代码" detail="请在项目总览导入代码，并填写 model_factory 入口与模型配置 Schema。" />}
      <div className="model-capability-list">{(architecture?.capabilities || []).map((capability) => <span key={capability}>{capability}</span>)}</div>
    </section>
    <section className="model-spec-section">
      <header><div><span>02</span><div><h3>配置真实架构字段</h3><p>基础区显示结构决策；Dropout、Bias、初始化和扩展项收在高级字段。</p></div></div><button className="button secondary small" type="button" onClick={() => setShowAdvanced((value) => !value)}><SlidersHorizontal size={14} />{showAdvanced ? "收起高级字段" : "显示高级字段"}</button></header>
      {groups.map((group) => <div className="model-field-group" key={group}><h4>{group}</h4><div className="form-grid three-columns">{visibleFields.filter((field) => field.group === group).map((field) => <FieldControl field={field} value={config[field.key] ?? field.default} onChange={(value) => updateField(field.key, value)} key={field.key} />)}</div></div>)}
    </section>
    <section className="model-spec-section">
      <header><div><span>03</span><div><h3>选择初始化来源</h3><p>模型结构和初始权重分开冻结。</p></div></div></header>
      <div className="form-grid two-columns"><Field label="初始化方式"><select aria-label="初始化方式" value={initializationType} onChange={(event) => { setInitializationType(event.target.value); preview.reset(); }}><option value="random">随机初始化</option><option value="pretrained">预训练模型标识</option><option value="model_version">项目模型版本 ID</option><option value="checkpoint">Checkpoint ID</option></select></Field>{initializationType !== "random" && <Field label="来源引用"><input required value={initializationReference} onChange={(event) => { setInitializationReference(event.target.value); preview.reset(); }} placeholder="仓库标识或不可变 ID" /></Field>}</div>
    </section>
    <section className="model-spec-section preview-section">
      <header><div><span>04</span><div><h3>解析与运行契约</h3><p>JSON、代码、参数统计和 Worker 实现绑定来自同一服务端解析结果。</p></div></div><button type="button" className="button secondary" disabled={preview.isPending || (architecture?.source_type === "custom_code" && !codeVersionId)} onClick={() => preview.mutate()}><Play size={15} />{preview.isPending ? "解析中…" : "生成运行预览"}</button></header>
      {result && <ResolvedPreview result={result} mode={previewMode} onMode={setPreviewMode} />}
      <ErrorMessage error={preview.error} />
    </section>
    <ErrorMessage error={save.error} />
    <FormActions submitting={save.isPending} disabled={!result || result.status !== "ready"} onCancel={onCancel} submitLabel={specification ? "保存新 Revision" : "保存模型规格"} />
  </form>;
}

function JsonFieldControl({ field, value, onChange }: { field: ModelArchitectureField; value: any; onChange: (value: any) => void }) {
  const serialized = JSON.stringify(value ?? field.default, null, 2);
  const [text, setText] = useState(serialized);
  const [invalid, setInvalid] = useState(false);
  useEffect(() => { setText(serialized); setInvalid(false); }, [serialized]);
  return <Field label={field.label} hint={invalid ? "JSON 格式尚未完成" : field.hint}>
    <textarea
      aria-label={field.label}
      className={invalid ? "invalid" : ""}
      value={text}
      onChange={(event) => {
        const next = event.target.value;
        setText(next);
        try {
          onChange(JSON.parse(next));
          setInvalid(false);
        } catch {
          setInvalid(true);
        }
      }}
    />
  </Field>;
}

function FieldControl({ field, value, onChange }: { field: ModelArchitectureField; value: any; onChange: (value: any) => void }) {
  if (field.type === "boolean") return <Field label={field.label} hint={field.hint}><select aria-label={field.label} value={String(Boolean(value))} onChange={(event) => onChange(event.target.value === "true")}><option value="false">关闭</option><option value="true">开启</option></select></Field>;
  if (field.type === "select") return <Field label={field.label} hint={field.hint}><select aria-label={field.label} value={String(value ?? field.default)} onChange={(event) => onChange(event.target.value)}>{field.options?.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select></Field>;
  if (field.type === "json" || field.type === "integer_list") return <JsonFieldControl field={field} value={value} onChange={onChange} />;
  return <Field label={field.label} hint={field.hint}><input aria-label={field.label} type={field.type === "text" ? "text" : "number"} min={field.minimum} max={field.maximum} step={field.step} value={String(value ?? field.default)} onChange={(event) => onChange(field.type === "text" ? event.target.value : field.type === "integer" ? Number.parseInt(event.target.value || "0", 10) : Number(event.target.value))} /></Field>;
}

function ResolvedPreview({ result, mode, onMode }: { result: any; mode: "json" | "code"; onMode: (mode: "json" | "code") => void }) {
  const parameters = result.parameter_summary || {};
  const method = parameters.method === "user_declared" ? "用户声明（未验证）" : parameters.verified ? "架构精确公式" : "架构估算，运行时复核";
  return <div className="model-resolve-preview">
    {(result.issues || []).map((issue: any) => <Alert severity={issue.severity} title={issue.message} detail={issue.code} key={`${issue.code}-${issue.field || ""}`} />)}
    {result.status === "ready" && <>
      <div className="model-parameter-cards"><div><span>总参数量</span><strong>{formatNumber(parameters.total_parameters)}</strong></div><div><span>每 Token 激活</span><strong>{formatNumber(parameters.active_parameters_per_token)}</strong></div><div><span>可训练参数</span><strong>{formatNumber(parameters.trainable_parameters)}</strong></div><div><span>统计依据</span><strong>{method}</strong></div></div>
      <div className="preview-tabs"><button type="button" className={mode === "json" ? "active" : ""} onClick={() => onMode("json")}><Braces size={14} />运行 JSON</button><button type="button" className={mode === "code" ? "active" : ""} onClick={() => onMode("code")}><Code2 size={14} />实例化代码</button></div>
      <pre className="json-preview model-code-preview">{mode === "json" ? JSON.stringify(result.generated_json, null, 2) : result.generated_code}</pre>
    </>}
  </div>;
}
