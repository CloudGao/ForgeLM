import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, post } from "../api";
import type { CodeSource, Json, Task } from "../types";
import { ErrorMessage, Field } from "./ui";
import { TrainingOptimizationEditor } from "./TrainingOptimizationEditor";

export function TrainingOptimizationComparison({ task, onOpen }: { task: Task; onOpen: (id: string) => void }) {
  const [version, setVersion] = useState("");
  const [profile, setProfile] = useState<Json | undefined>({ partial_rope: "flash_attn", linear_cross_entropy: "fla" });
  const [batch, setBatch] = useState(Number(task.payload?.stages?.[0]?.micro_batch_size || 1));
  const sources = useQuery({ queryKey: ["optimization-engines", task.project_id], queryFn: () => api<CodeSource[]>(`/projects/${task.project_id}/code-sources`) });
  const versions = (sources.data || []).flatMap(source => source.versions.filter(v => v.state === "ready" && v.training_contract?.performance_probe === "forgelm.training-performance.v1" && v.training_contract?.optimization_profiles === "forgelm.microlm-optimizations.v1").map(v => ({ ...v, source: source.name })));
  const run = useMutation({ mutationFn: () => post<{ task_id: string }>(`/projects/${task.project_id}/training-performance`, {
    source_task_id: task.id, engine_version_id: version, search_mode: "manual", warmup_steps: 2, measurement_steps: 6,
    candidates: [{ micro_batch_size: batch, activation_checkpointing: task.payload?.stages?.[0]?.activation_checkpointing ?? true,
      checkpoint_every_n_layers: task.payload?.stages?.[0]?.checkpoint_every_n_layers || 1, optimizations: profile }],
  }), onSuccess: result => onOpen(result.task_id) });
  return <section className="detail-section"><h3>比较训练算子组合</h3><p>从本次安全暂停的完整状态比较原组合和候选，保持数据、有效批量和优化器预算。结果保留 loss、梯度、吞吐与显存证据。</p>
    <ErrorMessage error={sources.error || run.error} />
    <Field label="比较引擎版本"><select value={version} onChange={e => setVersion(e.target.value)}><option value="">选择支持算子比较的版本</option>{versions.map(v => <option key={v.id} value={v.id}>{v.source} · V{v.version}</option>)}</select></Field>
    <Field label="候选微批量"><input type="number" min="1" max="64" value={batch} onChange={e => setBatch(Number(e.target.value))} /></Field>
    <TrainingOptimizationEditor value={profile} onChange={setProfile} />
    <button className="button secondary" disabled={!version || !profile || run.isPending || !Number.isInteger(batch) || batch < 1 || batch > 64} onClick={() => run.mutate()}>创建完整训练比较</button>
  </section>;
}
