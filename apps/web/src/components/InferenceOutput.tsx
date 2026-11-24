import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiHref, post } from "../api";

const phases: Record<string, string> = {
  dispatching: "正在派发到 Worker", waiting_for_worker_capacity: "等待指定 Worker 或显卡空闲",
  preparing_model: "检查本地模型缓存", downloading_model: "下载模型文件",
  extracting_model: "解压模型文件", loading_model: "准备模型运行时",
  prefilling: "处理输入", generating: "正在生成", finalizing_outputs: "生成完成，正在保存记录",
  waiting_for_gpu_release: "等待空闲模型让出显卡", completed: "已完成",
};

export function InferenceOutput({ record, task, projectId }: { record: any; task: any; projectId: string }) {
  const client = useQueryClient();
  const [stream, setStream] = useState<any>(null);
  const [cancelError, setCancelError] = useState("");
  const terminal = ["succeeded", "failed", "cancelled", "lost"].includes(task?.observed_state);
  useEffect(() => {
    setStream(null);
    if (terminal || !task?.id || typeof EventSource === "undefined") return;
    const source = new EventSource(apiHref(`/tasks/${task.id}/events`));
    source.addEventListener("task-event", (event: MessageEvent) => {
      const message = JSON.parse(event.data);
      if (task.active_attempt_id && message.attempt_id && message.attempt_id !== task.active_attempt_id) return;
      if (["inference.snapshot", "inference.output"].includes(message.event_type)) {
        setStream((previous: any) => ({ ...previous, ...message.payload,
          performance: { ...previous?.performance, ...message.payload.performance } }));
      } else if (message.event_type === "task.assigned") {
        setStream((previous: any) => ({ ...previous, phase: "dispatching" }));
      } else if (message.event_type === "progress") {
        setStream((previous: any) => ({ ...previous, phase: message.payload.phase }));
      } else if (["task.succeeded", "task.failed", "task.cancelled", "task.lost"].includes(message.event_type)) {
        client.invalidateQueries({ queryKey: ["playground", projectId] });
        client.invalidateQueries({ queryKey: ["tasks", projectId] });
        source.close();
      }
    });
    return () => source.close();
  }, [task?.id, task?.active_attempt_id, terminal, client, projectId]);
  const savedStream = record.performance?.stream;
  const snapshot = stream || (savedStream?.attempt_id === task?.active_attempt_id ? savedStream : null);
  const text = record.output || snapshot?.output;
  const phase = snapshot?.phase || task?.current_phase;
  const performance = record.output ? record.performance : snapshot?.performance;
  const failure = ["failed", "cancelled", "lost"].includes(task?.observed_state);
  async function cancel() {
    try {
      await post(`/tasks/${task.id}/control`, { action: "cancel" });
      client.invalidateQueries({ queryKey: ["tasks", projectId] });
    } catch (error) { setCancelError(String(error)); }
  }
  return <>
    {text && <p>{text}</p>}
    {failure && <p role="alert">{task?.error?.message || `推理未完成：${task.observed_state}`}。未完成的正文不会进入后续对话。</p>}
    {!terminal && <div className="generating"><i /><span>{phases[phase] || phase || "等待 Worker"}</span></div>}
    {!terminal && task && <button className="button ghost small" onClick={cancel}>停止生成</button>}
    {cancelError && <p role="alert">{cancelError}</p>}
    {performance && <div className="output-meta">
      {performance.ttft_ms != null && <span>生成首 Token {Math.round(performance.ttft_ms)} ms</span>}
      {performance.request_ttft_ms != null && <span>请求至首 Token {Math.round(performance.request_ttft_ms)} ms</span>}
      {performance.model_cache_hit != null && <span>{performance.model_cache_hit ? "复用已加载模型" : "本次加载模型"}</span>}
      {performance.preparation_ms != null && <span>文件准备 {Math.round(performance.preparation_ms)} ms</span>}
      {performance.load_ms != null && <span>运行时准备 {Math.round(performance.load_ms)} ms</span>}
      {performance.prefix_tokens > 0 && <span>复用 {performance.prefix_tokens} 个上下文 Token</span>}
    </div>}
  </>;
}
