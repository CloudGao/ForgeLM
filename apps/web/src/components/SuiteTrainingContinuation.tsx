import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { ErrorMessage, Field } from "./ui";

export type ContinuationSelection = { checkpoint_id: string; mode: string; keep_optimizer: boolean };
type Checkpoint = { id: string; task_id: string; attempt_id: string; step: number; manifest: Record<string, any> };

export function SuiteTrainingContinuation({ projectId, value, onChange, onUseSource }: { projectId: string; value: ContinuationSelection; onChange: (value: ContinuationSelection) => void; onUseSource?: (config: Record<string, any>, inputs: Record<string, string>) => void }) {
  const checkpoints = useQuery({ queryKey: ["training-state-checkpoints", projectId], queryFn: () => api<Checkpoint[]>(`/projects/${projectId}/training-state-checkpoints`), refetchInterval: 5000 });
  const selected = checkpoints.data?.find(c => c.id === value.checkpoint_id);
  const source = useQuery({ queryKey: ["continuation-source", selected?.task_id], queryFn: () => api<{ payload: { suite_execution: { config: Record<string, any>; inputs: Record<string, { artifact_id: string }> } } }>(`/tasks/${selected!.task_id}?include_events=false`), enabled: Boolean(selected && onUseSource) });
  return <fieldset><legend>训练起点</legend><ErrorMessage error={checkpoints.error} />
    <Field label="从 Checkpoint 继续训练"><select value={value.checkpoint_id} onChange={e => onChange({ ...value, checkpoint_id: e.target.value })}><option value="">从所选模型开始新训练</option>{checkpoints.data?.map(c => <option key={c.id} value={c.id}>Step {c.step} · {c.task_id.slice(0, 8)} / {c.attempt_id.slice(0, 8)}</option>)}</select></Field>
    {selected && <>
      <ErrorMessage error={source.error} />
      {onUseSource && <button type="button" className="button secondary" disabled={!source.data?.payload?.suite_execution} onClick={() => { const spec = source.data!.payload.suite_execution; onUseSource(spec.config, Object.fromEntries(Object.entries(spec.inputs).map(([name, ref]) => [name, ref.artifact_id]))); }}>沿用源训练配置与输入</button>}
      <Field label="状态继承方式"><select value={value.mode} onChange={e => onChange({ ...value, mode: e.target.value, keep_optimizer: false })}><option value="weights">仅继承权重</option><option value="configured">配置变更接续</option><option value="full">完整状态接续</option></select></Field>
      {value.mode === "configured" && <label><input type="checkbox" checked={value.keep_optimizer} onChange={e => onChange({ ...value, keep_optimizer: e.target.checked })} />保留兼容的 AdamW 动量；使用本次学习率</label>}
      <p>{value.mode === "full" ? "继承权重、优化器、学习率进度、随机状态和数据游标。配置与输入必须一致，目标组件会检查状态兼容性。" : `继承权重${value.keep_optimizer ? "和兼容优化器状态" : ""}；重置学习率进度、随机状态和数据游标。可更换数据与优化方式。`}</p>
      <p>本次创建独立训练记录。模型输入用于校验目标架构，实际初始权重来自此 Checkpoint。</p>
      <details><summary>源 Checkpoint 配置与状态</summary><pre className="json-preview">{JSON.stringify(selected.manifest, null, 2)}</pre></details>
    </>}
  </fieldset>;
}
