import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, post } from "../api";
import { ErrorMessage, Field, StatusBadge } from "./ui";

type ServiceRequest = { id: string; status: string; chunks: { event: string; data: { text?: string } }[]; error?: { message?: string } };

export function SuiteServiceConsole({ projectId }: { projectId: string }) {
  const [task, setTask] = useState("");
  const [prompt, setPrompt] = useState("");
  const [request, setRequest] = useState("");
  const services = useQuery({ queryKey: ["suite-runs", projectId], queryFn: () => api<{ id: string; kind: string; title: string; observed_state: string }[]>(`/projects/${projectId}/suite-runs`), refetchInterval: 3000 });
  const status = useQuery({ queryKey: ["suite-service-request", request], queryFn: () => api<ServiceRequest>(`/suite-service-requests/${request}`), enabled: !!request, refetchInterval: query => ["queued", "running"].includes(query.state.data?.status || "queued") ? 500 : false });
  const generate = useMutation({ mutationFn: () => post<ServiceRequest>(`/suite-services/${task}/generate`, { prompt, max_tokens: 256, temperature: 0.7 }), onSuccess: data => setRequest(data.id) });
  const cancel = useMutation({ mutationFn: () => post(`/suite-service-requests/${request}/cancel`), onSuccess: () => void status.refetch() });
  const active = ["queued", "running"].includes(status.data?.status || "");
  return <section className="suite-service-console"><h3>推理服务调试</h3><ErrorMessage error={generate.error || status.error || cancel.error} /><Field label="运行中的套件服务"><select value={task} onChange={e => setTask(e.target.value)}><option value="">选择服务</option>{services.data?.filter(t => t.kind === "suite_service" && t.observed_state === "running").map(t => <option key={t.id} value={t.id}>{t.title} · {t.id}</option>)}</select></Field><Field label="输入提示词"><textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={3} /></Field><div className="package-actions"><button className="button" disabled={!task || !prompt.trim() || active || generate.isPending} onClick={() => generate.mutate()}>生成</button>{active && <button className="button secondary" disabled={cancel.isPending} onClick={() => cancel.mutate()}>取消本次生成</button>}</div>{status.data && <><StatusBadge status={status.data.status} /><pre style={{ whiteSpace: "pre-wrap" }}>{status.data.chunks.filter(c => c.event === "chunk").map(c => c.data.text || "").join("")}</pre>{status.data.error?.message && <p role="alert">{status.data.error.message}</p>}</>}<p>停止服务、释放 Worker 资源请使用对应任务的取消操作。</p></section>;
}
