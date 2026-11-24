import { useEffect, useState } from "react";
import { Field } from "./ui";

function JsonField({ label, value, onChange, onValidity }: { label: string; value: unknown; onChange: (value: any) => void; onValidity: (valid: boolean) => void }) {
  const [text, setText] = useState(JSON.stringify(value, null, 2) ?? "");
  const [error, setError] = useState("");
  useEffect(() => { setText(JSON.stringify(value, null, 2) ?? ""); setError(""); }, [value]);
  return <Field label={label} hint={error || "填写 JSON；字段类型与条件由服务端校验"}><textarea aria-label={label} value={text} aria-invalid={Boolean(error)} onChange={(event) => {
    const next = event.target.value; setText(next);
    try { const parsed = JSON.parse(next); setError(""); onValidity(true); onChange(parsed); }
    catch { setError("JSON 格式不完整"); onValidity(false); }
  }} /></Field>;
}

export function PluginConfigFields({ schema, value, onChange, onValidity }: {
  schema: Record<string, any>; value: Record<string, any>;
  onChange: (next: Record<string, any>) => void; onValidity: (key: string, valid: boolean) => void;
}) {
  const properties = schema.properties || {};
  const keys = [...new Set([...Object.keys(properties), ...Object.keys(value)])];
  return <div className="form-grid two-columns">{keys.map((key) => {
    const property = properties[key] || {};
    const label = property.title || key;
    const update = (next: any) => onChange({ ...value, [key]: next });
    const remove = () => { const next = { ...value }; delete next[key]; onChange(next); onValidity(key, true); };
    const required = (schema.required || []).includes(key);
    let field;
    if (property.enum) field = <Field label={label} hint={property.description}><select value={value[key] === undefined ? "" : JSON.stringify(value[key])} onChange={(event) => event.target.value ? update(JSON.parse(event.target.value)) : remove()} required={required}><option value="">未设置</option>{property.enum.map((option: unknown) => <option key={JSON.stringify(option)} value={JSON.stringify(option)}>{String(option)}</option>)}</select></Field>;
    else if (property.type === "boolean") field = <Field label={label} hint={property.description}><select value={value[key] === undefined ? "" : String(value[key])} onChange={(event) => event.target.value ? update(event.target.value === "true") : remove()} required={required}><option value="">未设置</option><option value="true">开启</option><option value="false">关闭</option></select></Field>;
    else if (["number", "integer"].includes(property.type)) field = <Field label={label} hint={property.description}><input type="number" step={property.type === "integer" ? 1 : "any"} min={property.minimum} max={property.maximum} required={required} value={value[key] ?? ""} onChange={(event) => event.target.value === "" ? remove() : update(Number(event.target.value))} /></Field>;
    else if (property.type === "string") field = <Field label={label} hint={property.description}><input value={value[key] ?? ""} required={required} minLength={property.minLength} maxLength={property.maxLength} onChange={(event) => update(event.target.value)} /></Field>;
    else field = <JsonField label={label} value={value[key]} onChange={update} onValidity={(valid) => onValidity(key, valid)} />;
    return <div key={key}>{field}{!required && Object.hasOwn(value, key) && <button type="button" className="button ghost" onClick={remove}>移除 {label}</button>}</div>;
  })}{!keys.length && <p>该版本无需配置默认参数。</p>}</div>;
}
