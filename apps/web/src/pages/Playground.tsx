import { InferenceOutput } from "../components/InferenceOutput";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Braces, Clock3, Gauge, GitCompareArrows, History, MessageSquarePlus, Save, Send, SlidersHorizontal, Sparkles, User } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api, formatDate, post } from "../api";
import { Alert, EmptyState, ErrorMessage, Modal, PageHeader, Panel, StatusBadge } from "../components/ui";

function sessionPreview(session: any): string {
  const firstPrompt = session?.records?.[0]?.prompt?.trim();
  if (firstPrompt) return firstPrompt.length > 72 ? `${firstPrompt.slice(0, 72)}…` : firstPrompt;
  return "空会话";
}

function sessionModelLabel(session: any, models: any[] | undefined): string {
  const modelId = session?.model_version_ids?.[0];
  const model = models?.find((item) => item.id === modelId);
  if (model) return `${model.name} v${model.version}`;
  return modelId ? `模型 ${String(modelId).slice(0, 8)}` : "未绑定模型";
}

const TRANSFORMERS_ADAPTER = "builtin.transformers-inference";

function modelSource(model: any): string | null {
  const metadata = model?.metadata_json || {};
  const source = model?.inference_source;
  if (source?.type === "unavailable" || source?.type === "unsupported") return null;
  if (["command", "native"].includes(source?.type)) return `${source.display_name || "项目原生推理"} · 制品 ${model.artifact_id?.slice(0, 8)}`;
  return metadata?.inference?.pretrained || metadata?.pretrained || metadata?.huggingface_id || (model?.artifact_id ? `制品 ${model.artifact_id.slice(0, 8)}` : null);
}

function workerCanRunModel(worker: any, model: any, adapterId: string, requestedGpus = 0): boolean {
  const capabilities = new Set<string>(worker.capabilities || []);
  if (worker.status !== "online" || (!capabilities.has("*") && (!capabilities.has(adapterId) || !capabilities.has("builtin.local-process")))) return false;
  const source = model?.inference_source;
  const profileRequest = source?.adapter_id === adapterId ? source?.resource_request || {} : {};
  const selector = profileRequest.worker_selector || {};
  const requiredGpus = Math.max(Number(profileRequest.gpus || 0), requestedGpus);
  if (requiredGpus > Number(worker.resources?.gpus?.length || 0)) return false;
  if (Number(profileRequest.cpus || 0) > Number(worker.resources?.cpu_count || 0)) return false;
  if (selector.worker_id && selector.worker_id !== worker.id) return false;
  if (selector.worker_name && selector.worker_name !== worker.name) return false;
  if (selector.worker_ids?.length && !selector.worker_ids.includes(worker.id)) return false;
  if (Object.entries(selector.labels || {}).some(([key, value]) => worker.labels?.[key] !== value)) return false;
  if (selector.capabilities?.some((capability: string) => !capabilities.has("*") && !capabilities.has(capability))) return false;
  if (!source || source.type === "pretrained" || source.type === "managed" || source.globally_accessible) return true;
  return ["external", "command", "native"].includes(source.type) && source.worker_ids?.includes(worker.id);
}

function workerOptionLabel(worker: any): string {
  const environment = worker.labels?.environment ? ` · ${worker.labels.environment}` : "";
  const gpu = worker.resources?.gpus?.[0]?.name ? ` · ${worker.resources.gpus[0].name}` : "";
  return `${worker.name || worker.id}${environment}${gpu}`;
}

export function Playground() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a precise and helpful assistant.");
  const [sessionId, setSessionId] = useState<string | null | undefined>();
  const [promptOpen, setPromptOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [adapterId, setAdapterId] = useState(TRANSFORMERS_ADAPTER);
  const [workerId, setWorkerId] = useState("");
  const [commandExecutable, setCommandExecutable] = useState("");
  const [commandArgs, setCommandArgs] = useState("");
  const [settings, setSettings] = useState({ prompt_mode: "auto", temperature: 0.7, top_p: 0.9, top_k: 40, repetition_penalty: 1.05, max_tokens: 512, device: "auto", dtype: "auto" });
  const models = useQuery({ queryKey: ["models", projectId], queryFn: () => api<any>(`/projects/${projectId}/models`) });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<any[]>("/workers"), refetchInterval: 5000 });
  const sessions = useQuery({ queryKey: ["playground", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/playground/sessions`), refetchInterval: 5000 });
  const feedback = useQuery({ queryKey: ["feedback", projectId], queryFn: () => api<any[]>(`/projects/${projectId}/feedback`) });
  const generate = useMutation({ mutationFn: (data: any) => post<any>(`/projects/${projectId}/playground/generate`, data), onSuccess: (data) => { setSessionId(data.session.id); setPrompt(""); queryClient.invalidateQueries({ queryKey: ["playground", projectId] }); queryClient.invalidateQueries({ queryKey: ["tasks", projectId] }); } });
  const unload = useMutation({ mutationFn: (id: string) => post(`/workers/${id}/inference/unload`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["workers"] }) });
  const saveFeedback = useMutation({ mutationFn: ({ record, kind }: { record: any; kind: string }) => post(`/projects/${projectId}/playground/feedback`, { inference_record_id: record.id, kind, preferred_output: record.output, labels: ["playground"] }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["feedback", projectId] }) });

  const activeSession = useMemo(() => sessionId === null ? undefined : sessions.data?.find((session) => session.id === sessionId) || sessions.data?.[0], [sessions.data, sessionId]);
  const latestRecord = activeSession?.records?.[activeSession.records.length - 1];

  useEffect(() => {
    const sessionModels = sessionId === null ? undefined : activeSession?.model_version_ids;
    if (Array.isArray(sessionModels) && sessionModels.length) {
      if (modelIds.join("\0") !== sessionModels.join("\0")) setModelIds(sessionModels);
      return;
    }
    if (models.data && !models.data.models.some((model: any) => model.id === modelIds[0])) {
      const official = models.data.models.find((model: any) => model.is_official) || models.data.models[0];
      if (official) setModelIds([official.id]);
      else if (modelIds.length) setModelIds([]);
      setWorkerId("");
    }
  }, [activeSession, models.data, modelIds, sessionId]);

  const selectedModel = models.data?.models?.find((model: any) => model.id === modelIds[0]);
  const selectedSource = modelSource(selectedModel);
  const selectedSourceIssue = selectedModel?.inference_source?.type === "unsupported" ? selectedModel.inference_source.reason : null;
  const profileAdapter = selectedModel?.inference_source?.adapter_id || null;
  const profileRequest = selectedModel?.inference_source?.resource_request || {};
  const taskById = new Map<string, any>((sessions.data || []).flatMap((session: any) => (session.records || []).filter((record: any) => record.task).map((record: any) => [record.task_id, record.task])));
  const commandAdapter = adapterId.startsWith("builtin.command.");
  const profileActive = Boolean(profileAdapter && adapterId === profileAdapter);
  const profileCommand = profileActive && selectedModel?.inference_source?.type === "command";
  const profileGpu = profileActive && Number(profileRequest.gpus || 0) > 0;
  const requestedGpus = Math.max(Number(profileRequest.gpus || 0), settings.device === "cuda" ? 1 : 0);
  const compatibleWorkers = useMemo(
    () => (workers.data || []).filter((worker) => workerCanRunModel(worker, selectedModel, adapterId, requestedGpus)),
    [workers.data, selectedModel, adapterId, requestedGpus],
  );
  const hasAdapterWorker = compatibleWorkers.length > 0;
  const selectedWorkerValid = !workerId || compatibleWorkers.some((worker) => worker.id === workerId);
  const inferenceBlocked = !selectedModel || (adapterId === TRANSFORMERS_ADAPTER || profileActive) && (!hasAdapterWorker || !selectedSource || !selectedWorkerValid);

  useEffect(() => {
    setAdapterId(profileAdapter || TRANSFORMERS_ADAPTER);
  }, [selectedModel?.id, profileAdapter]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (generate.isPending || sessionBusy || !prompt.trim() || !modelIds.length || inferenceBlocked || (commandAdapter && !commandExecutable.trim())) return;
    const resourceRequest: Record<string, any> = { gpus: requestedGpus };
    if (workerId) resourceRequest.worker_selector = { worker_id: workerId };
    generate.mutate({
      session_id: activeSession?.id,
      mode: "single",
      model_version_ids: modelIds,
      prompt,
      system_prompt: systemPrompt,
      config: settings,
      adapter_id: adapterId,
      launcher_id: "builtin.local-process",
      resource_request: resourceRequest,
      command: commandAdapter ? { executable: commandExecutable, args: commandArgs.split("\n").map((value) => value.trim()).filter(Boolean), env: {}, outputs: [] } : {},
    });
  }

  const sessionBusy = Boolean(activeSession?.records?.some((record: any) => { const task = taskById.get(record.task_id); return task && !["succeeded", "failed", "cancelled", "lost"].includes(task.observed_state); }));
  const cannotRun = sessionBusy || !prompt.trim() || !modelIds.length || generate.isPending || inferenceBlocked || (commandAdapter && !commandExecutable.trim());
  const runTitle = !modelIds.length ? "项目中没有可用模型" : !selectedModel ? "所选模型已不可用，请选择模型或开始新会话" : adapterId === TRANSFORMERS_ADAPTER && selectedSourceIssue ? selectedSourceIssue : (adapterId === TRANSFORMERS_ADAPTER || profileActive) && !selectedSource ? "该模型没有权重来源" : (adapterId === TRANSFORMERS_ADAPTER || profileActive) && !hasAdapterWorker ? "没有能访问该模型的匹配 Worker" : !selectedWorkerValid ? "所选 Worker 已离线或不再满足当前推理条件" : commandAdapter && !commandExecutable.trim() ? "请先填写可执行文件" : undefined;

  return <>
    <PageHeader eyebrow="INFERENCE PLAYGROUND" title="推理 Playground" description="由 Worker 加载真实模型权重并保存实际 Prompt、Tokens、输出与性能数据。" actions={<>
      <button type="button" className={`button secondary${historyOpen ? " active" : ""}`} aria-expanded={historyOpen} aria-controls="playground-session-history" onClick={() => setHistoryOpen((open) => !open)}><History size={16} />历史会话</button>
      <button type="button" className="button secondary" onClick={() => { setSessionId(null); setPrompt(""); }}><MessageSquarePlus size={16} />新会话</button>
    </>} />
    {(workers.data || []).filter((worker: any) => worker.status === "online" && worker.resources?.inference_runtimes?.some((runtime: any) => runtime.model_id === selectedModel?.id)).map((worker: any) => <div className="output-meta" key={worker.id}><span>{worker.name} · 模型已驻留，空闲 10 分钟自动卸载，也可提前让出显卡</span><button className="button ghost small" disabled={unload.isPending || Boolean(worker.labels?.inference_reclaim_requested)} onClick={() => unload.mutate(worker.id)}>{worker.labels?.inference_reclaim_requested ? "等待释放" : "卸载空闲模型"}</button></div>)}
    <ErrorMessage error={unload.error} />
    <div className="playground-layout">
      <aside className="playground-config">
        <Panel title="运行配置"><div className="config-form">
          <label><span>模式</span><select disabled><option>单模型生成</option></select><small>A/B 与批量推理尚未开放。</small></label>
          <label><span>模型版本</span><select aria-label="模型版本" value={modelIds[0] || ""} onChange={(event) => { setModelIds([event.target.value]); setSessionId(null); setWorkerId(""); }}>{!selectedModel && <option value={modelIds[0] || ""} disabled>{modelIds[0] ? `模型 ${modelIds[0].slice(0, 8)}（已不可用）` : "请选择模型"}</option>}{models.data?.models?.map((model: any) => <option value={model.id} key={model.id}>{model.name} v{model.version}</option>)}</select><small>{modelIds.length > 0 && !selectedModel ? "所选模型已不可用，请选择模型或开始新会话" : selectedSource ? `来源：${selectedSource}` : "未绑定模型制品或模型仓库 ID"}{activeSession ? " · 切换模型会开始新会话" : ""}</small></label>
          <label><span>推理适配器</span><select value={adapterId} onChange={(event) => { setAdapterId(event.target.value); setWorkerId(""); }}>{profileAdapter && <option value={profileAdapter}>{selectedModel.inference_source.display_name || "项目原生推理"}</option>}{selectedModel?.inference_source?.type !== "native" && <option value={TRANSFORMERS_ADAPTER}>内置 Transformers 推理</option>}<option value="builtin.command.inference">用户命令推理引擎</option></select></label>
          {adapterId === TRANSFORMERS_ADAPTER && selectedSourceIssue && <Alert severity="warning" title="模型制品不能直接推理" detail={selectedSourceIssue} />}
          {adapterId === TRANSFORMERS_ADAPTER && selectedSource && !hasAdapterWorker && <Alert severity="warning" title="没有能访问该模型的 Transformers Worker" detail={'请在制品所在 Worker 运行 pip install -e ".[inference-hf]" 后重启，或先创建托管模型副本。'} />}
          {profileActive && selectedSource && !hasAdapterWorker && <Alert severity="warning" title="原生推理 Worker 未就绪" detail={`请启动声明 ${profileAdapter} 能力且能访问该模型制品的 Worker。`} />}
          <label><span>执行 Worker</span><select aria-label="执行 Worker" value={workerId} onChange={(event) => setWorkerId(event.target.value)}><option value="">自动选择兼容 Worker</option>{workerId && !selectedWorkerValid && <option value={workerId} disabled>所选 Worker 已不可用</option>}{compatibleWorkers.map((worker) => <option value={worker.id} key={worker.id}>{workerOptionLabel(worker)}</option>)}</select><small>手动选择后，本次推理只会由该 Worker 执行。</small></label>
          {workerId && !selectedWorkerValid && <Alert severity="blocker" title="所选 Worker 已不可用" detail="该 Worker 已离线，或不再满足当前模型的适配器、资源和制品访问条件；请重新选择。" />}
          {commandAdapter && <><label><span>可执行文件</span><input required value={commandExecutable} onChange={(event) => setCommandExecutable(event.target.value)} placeholder="python" /></label><label><span>命令参数（每行一个）</span><textarea value={commandArgs} onChange={(event) => setCommandArgs(event.target.value)} /></label></>}
          {!commandAdapter && <label><span>Prompt 方式</span><select value={settings.prompt_mode} onChange={(event) => setSettings({ ...settings, prompt_mode: event.target.value })}><option value="auto">自动（有模板对话，无模板续写）</option><option value="completion">原文续写</option><option value="chat">对话</option></select><small>Base 模型没有 Chat Template 时，自动模式不会注入角色标签。</small></label>}
          <label><span>System Prompt</span><textarea disabled={settings.prompt_mode === "completion"} value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />{settings.prompt_mode === "completion" && <small>原文续写只编码本次 Prompt，不注入 System Prompt 或历史轮次。</small>}</label>
          <label><span>设备</span><select disabled={profileCommand || profileGpu} value={profileCommand || profileGpu ? "cuda" : settings.device} onChange={(event) => setSettings({ ...settings, device: event.target.value })}><option value="auto">自动</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select>{(profileCommand || profileGpu) && <small>由模型推理配置固定。</small>}</label>
          <label><span>精度</span><select disabled={profileCommand} value={profileCommand ? "bfloat16" : settings.dtype} onChange={(event) => setSettings({ ...settings, dtype: event.target.value })}><option value="auto">自动</option><option value="float32">float32</option><option value="float16">float16</option><option value="bfloat16">bfloat16</option></select>{profileCommand && <small>由项目原生推理配置固定。</small>}</label>
          <Slider label="Temperature" value={settings.temperature} min={0} max={2} step={0.1} onChange={(value) => setSettings({ ...settings, temperature: value })} />
          <Slider label="Top-p" value={settings.top_p} min={0.05} max={1} step={0.05} onChange={(value) => setSettings({ ...settings, top_p: value })} />
          <Slider label="Top-k" value={settings.top_k} min={0} max={100} step={1} onChange={(value) => setSettings({ ...settings, top_k: value })} />
          <Slider label="Repetition penalty" value={settings.repetition_penalty} min={1} max={2} step={0.05} onChange={(value) => setSettings({ ...settings, repetition_penalty: value })} />
          <label><span>Max Tokens</span><input type="number" min={1} max={8192} value={settings.max_tokens} onChange={(event) => setSettings({ ...settings, max_tokens: Number(event.target.value) })} /></label>
        </div></Panel>
        <Panel title="反馈候选"><div className="feedback-count"><strong>{feedback.data?.length || 0}</strong><span>条候选等待数据审核</span></div></Panel>
      </aside>
      <section className={`chat-workspace${historyOpen ? " history-open" : ""}`}>
        {historyOpen && <aside id="playground-session-history" className="session-history" aria-label="历史会话列表">
          <header><div><strong>历史会话</strong><span>{sessions.data?.length || 0} 条</span></div></header>
          <div className="session-history-list">
            {(sessions.data || []).length ? (sessions.data || []).map((session: any) => {
              const selected = activeSession?.id === session.id && sessionId !== null;
              return <button type="button" key={session.id} className={`session-history-item${selected ? " active" : ""}`} aria-pressed={selected} onClick={() => setSessionId(session.id)}>
                <strong>{sessionPreview(session)}</strong>
                <span>{sessionModelLabel(session, models.data?.models)} · {session.records?.length || 0} 轮</span>
                <small>{formatDate(session.updated_at || session.created_at)} · {session.id.slice(0, 8)}</small>
              </button>;
            }) : <EmptyState title="还没有历史会话" description="发送一次推理后，会话会出现在这里。" />}
          </div>
        </aside>}
        <div className="chat-main">
          <div className="chat-toolbar"><div><Sparkles size={17} /><span>{activeSession ? `${activeSession.mode} · ${activeSession.id.slice(0, 8)}` : "新会话"}</span></div><button className="button ghost small" disabled={!latestRecord?.rendered_prompt} title={!latestRecord?.rendered_prompt ? "任务完成后才能查看真实渲染 Prompt" : undefined} onClick={() => setPromptOpen(true)}><Braces size={15} />查看 Prompt</button><button className="button ghost small" disabled title="当前仅支持单模型输出"><GitCompareArrows size={15} />对比</button></div>
          <div className="conversation">{activeSession?.records?.length ? activeSession.records.map((record: any) => {
            const task = taskById.get(record.task_id);
            return <div className="message-pair" key={record.id}><article className="message user-message"><div className="avatar"><User size={17} /></div><div><span>你</span><p>{record.prompt}</p></div></article><article className="message model-message"><div className="avatar"><Bot size={18} /></div><div><header><span>模型输出</span>{task && <StatusBadge status={task.observed_state} />}</header><InferenceOutput record={record} task={task} projectId={projectId} /><div className="output-meta"><span><Clock3 size={13} />{record.performance?.generation_ms || "—"} ms</span><span><Gauge size={13} />{record.performance?.tokens_per_second || "—"} tok/s</span><span>{record.input_tokens || 0} → {record.output_tokens || 0} tokens</span></div>{record.output && <footer><button onClick={() => saveFeedback.mutate({ record, kind: "eval_failure" })}><Save size={14} />保存为评测失败</button><button onClick={() => saveFeedback.mutate({ record, kind: "sft_candidate" })}><Save size={14} />保存为 SFT 候选</button><button onClick={() => saveFeedback.mutate({ record, kind: "dpo_pair" })}><Save size={14} />保存 DPO Pair</button></footer>}</div></article></div>;
          }) : <EmptyState title="开始一次真实本地推理" description="选择有权重来源的模型并输入 Prompt；在线 Worker 会加载模型并生成输出。" />}</div>
          <form className="prompt-composer" onSubmit={submit}><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="输入 Prompt，Ctrl + Enter 发送" onKeyDown={(event) => { if (event.ctrlKey && event.key === "Enter") submit(event as any); }} /><div><span><SlidersHorizontal size={14} />{settings.temperature} temp · {settings.max_tokens} max</span><button className="button primary" disabled={cannotRun} title={runTitle}><Send size={16} />{generate.isPending ? "提交中" : "运行"}</button></div><ErrorMessage error={generate.error} /></form>
        </div>
      </section>
    </div>
    {promptOpen && latestRecord?.rendered_prompt && <Modal title="实际渲染 Prompt" description="这是 Worker 最终编码并由任务保存的真实输入；原文续写不会添加角色标签。" onClose={() => setPromptOpen(false)} wide><pre className="prompt-preview">{latestRecord.rendered_prompt}</pre><div className="form-actions"><span /><button className="button primary" onClick={() => setPromptOpen(false)}>关闭</button></div></Modal>}
  </>;
}

function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  return <label className="slider-field"><span>{label}<b>{value}</b></span><input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}
