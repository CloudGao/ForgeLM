import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { post } from "../api";
import type { Json, ProjectPipelineInstance } from "../types";
import { ErrorMessage, Field, Modal } from "./ui";

export function InterleavedGenerationDialog({ instance, sourceTaskId, onClose, onCreated }: {
  instance: ProjectPipelineInstance; sourceTaskId?: string; onClose: () => void;
  onCreated: (instance: ProjectPipelineInstance) => void;
}) {
  const [name, setName] = useState(`${instance.name.slice(0, 100)} · 分批交错`);
  const [batchRecords, setBatchRecords] = useState(100000);
  const [reuse, setReuse] = useState(Boolean(sourceTaskId));
  const [summary, setSummary] = useState<Json | null>(null);
  const body = { name, batch_records: batchRecords, checkpoint_source_task_id: reuse ? sourceTaskId : null };
  const base = `/projects/${instance.project_id}/training-instances/${instance.id}`;
  const preview = useMutation({ mutationFn: () => post<{ summary: Json }>(`${base}/interleaved-preview`, body), onSuccess: value => setSummary(value.summary) });
  const create = useMutation({ mutationFn: () => post<ProjectPipelineInstance>(`${base}/interleaved`, body), onSuccess: value => onCreated(value) });
  return <Modal title="三元组翻译与正文分批交错" onClose={onClose}>
    <p>每批先提交中文三元组，再生成该批正文。沿用当前两个节点的模型与生成参数；每批保留独立版本和执行记录。</p>
    <Field label="新实例名称"><input value={name} onChange={e => { setName(e.target.value); setSummary(null); }} /></Field>
    <Field label="每批源记录数" hint="较大的批次减少模型加载次数；交错可提前产出正文，不保证缩短总计算时间。"><input type="number" min={1} max={1000000} value={batchRecords} onChange={e => { setBatchRecords(Number(e.target.value)); setSummary(null); }} /></Field>
    {sourceTaskId && <label><input type="checkbox" checked={reuse} onChange={e => { setReuse(e.target.checked); setSummary(null); }} />接续当前翻译任务已有进度</label>}
    {reuse && <p>创建前请通过任务详情安全暂停来源翻译任务。平台会验证断点身份，复用已完成和已跳过记录。</p>}
    {summary && <p role="status">共 {summary.source_records.toLocaleString()} 条源记录，分为 {summary.batch_count} 批；每批翻译后生成正文，末尾汇集全部批次。</p>}
    <ErrorMessage error={preview.error || create.error} />
    <div className="form-actions"><button className="button secondary" onClick={onClose}>取消</button><button className="button secondary" disabled={preview.isPending || create.isPending} onClick={() => preview.mutate()}>预览批次</button><button className="button primary" disabled={!summary || preview.isPending || create.isPending || !name.trim()} onClick={() => create.mutate()}>创建交错实例</button></div>
  </Modal>;
}
