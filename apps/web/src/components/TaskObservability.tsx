import { useQuery } from "@tanstack/react-query";
import { ArrowDown, LoaderCircle } from "lucide-react";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError, formatDate } from "../api";
import type { Task, TaskEvent, TaskEventPage } from "../types";

export const LOG_WINDOW_SIZE = 300;
export const MAX_LOG_WINDOW_SIZE = 1800;

type EventGroup = "logs" | "resources" | "timeline" | "all";

function inEventGroup(event: TaskEvent, eventGroup: EventGroup) {
  const isLog = event.event_type === "log" || event.event_type.includes("stdout") || event.event_type.includes("stderr");
  const isResource = event.event_type === "resource.sample" || event.event_type === "resource.summary";
  if (eventGroup === "logs") return isLog;
  if (eventGroup === "resources") return isResource;
  if (eventGroup === "timeline") return !isLog && !isResource;
  return true;
}

export function useTaskEventWindow({
  taskId,
  eventGroup,
  attemptId,
  limit,
  enabled = true,
  live = false,
}: {
  taskId: string;
  eventGroup: EventGroup;
  attemptId?: string;
  limit: number;
  enabled?: boolean;
  live?: boolean;
}) {
  return useQuery<TaskEventPage>({
    queryKey: ["task-events", taskId, eventGroup, attemptId || "all-attempts", limit],
    queryFn: async () => {
      const params = new URLSearchParams({ event_group: eventGroup, limit: String(limit) });
      if (attemptId) params.set("attempt_id", attemptId);
      try {
        return await api<TaskEventPage>(`/tasks/${taskId}/event-window?${params.toString()}`);
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error;
        const legacy = await api<Task>(`/tasks/${taskId}`);
        const matching = (legacy.events || []).filter((event) => (!attemptId || event.attempt_id === attemptId) && inEventGroup(event, eventGroup));
        return { items: matching.slice(-limit), has_more: matching.length > limit };
      }
    },
    enabled: enabled && Boolean(taskId),
    refetchInterval: live ? 2000 : false,
  });
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function GenerationThroughput({ events }: { events: TaskEvent[] }) {
  const latest = events.filter(event => event.event_type === "progress" && ["generating", "autotuning"].includes(event.payload?.phase)).at(-1);
  if (!latest) return null;
  const p = latest.payload;
  if (p.phase === "autotuning") return <section className="detail-section" aria-label="生成自动调优">
    <h3>正在实测生成吞吐</h3><p>已完成 {p.completed_probes} / {p.total_probes} 组；当前在途请求 {p.submission_window}。使用相同真实输入、完整输出和质量校验，探测内容不计入正文。</p>
  </section>;
  const rate = (value: unknown) => finite(value)?.toFixed(1) ?? "未采集";
  return <section className="detail-section" aria-label="生成吞吐">
    <h3>生成吞吐</h3>
    <p>最近 {Math.round(finite(p.rate_window_seconds) || 0)} 秒 · 实际采样 {formatDate(p.sampled_at || latest.created_at)}</p>
    <dl className="detail-list">
      <div><dt>合格入库 tokens/s</dt><dd>{rate(p.recent_tokens_per_second)}</dd></div>
      <div><dt>原始生成 tokens/s（含重试和拒绝）</dt><dd>{rate(p.recent_raw_tokens_per_second)}</dd></div>
      <div><dt>本次完成请求</dt><dd>{p.raw_completed_requests ?? "未采集"}</dd></div>
      <div><dt>首遍通过率</dt><dd>{finite(p.first_pass_acceptance_rate) == null ? "未采集" : `${(Number(p.first_pass_acceptance_rate) * 100).toFixed(1)}%`}</dd></div>
      <div><dt>本次重试消耗 tokens</dt><dd>{p.retry_generated_tokens ?? "未采集"}</dd></div>
      <div><dt>本次未收录 tokens</dt><dd>{p.rejected_generated_tokens ?? "未采集"}</dd></div>
      <div><dt>本次前置跳过请求</dt><dd>{p.preflight_skipped_records ?? "未采集"}</dd></div>
      <div><dt>累计合格正文</dt><dd>{p.document_count}</dd></div>
      <div><dt>活动并发 / 在途上限</dt><dd>{p.batch_size} / {p.submission_window}</dd></div>
    </dl>
  </section>;
}

function chartPercent(value: unknown): number | null {
  const number = finite(value);
  if (number == null) return null;
  return Math.round(Math.min(100, Math.max(0, number)) * 10) / 10;
}

function ratioPercent(used: unknown, total: unknown): number | null {
  const usedNumber = finite(used);
  const totalNumber = finite(total);
  if (usedNumber == null || totalNumber == null || totalNumber <= 0) return null;
  return chartPercent((usedNumber / totalNumber) * 100);
}

const resourceLines = [
  { key: "systemCpu", name: "系统 CPU", color: "#55d7c4" },
  { key: "systemMemory", name: "系统内存", color: "#7aa2ff" },
  { key: "processCpu", name: "CPU 配额利用率", color: "#f5b95f" },
  { key: "processMemory", name: "进程内存", color: "#c084fc" },
  { key: "gpu", name: "GPU", color: "#ff7f8f" },
  { key: "gpuMemory", name: "GPU 显存", color: "#39b8f4" },
] as const;

export function ResourceUsageChart({ events }: { events: TaskEvent[] }) {
  const data = useMemo(() => events
    .filter((event) => event.event_type === "resource.sample")
    .map((event) => {
      const payload = event.payload || {};
      const gpus = Array.isArray(payload.gpus) ? payload.gpus : [];
      const gpuValues = gpus.map((gpu) => finite(gpu?.utilization_percent)).filter((value): value is number => value != null);
      const gpuUsed = gpus.reduce((sum, gpu) => sum + (finite(gpu?.memory_used_mb) || 0), 0);
      const gpuTotal = gpus.reduce((sum, gpu) => sum + (finite(gpu?.memory_total_mb) || 0), 0);
      return {
        sequence: event.sequence,
        time: new Date(event.created_at).toLocaleTimeString("zh-CN", { hour12: false }),
        systemCpu: chartPercent(payload.system?.cpu_percent),
        systemMemory: chartPercent(payload.system?.memory_percent),
        processCpu: chartPercent(payload.process?.cpu_allocation_percent),
        cpuCores: finite(payload.process?.cpu_percent) == null ? null : Number(payload.process.cpu_percent) / 100,
        allocatedCpus: finite(payload.process?.allocated_cpus),
        processMemory: ratioPercent(payload.process?.rss_bytes, payload.system?.effective_memory_total_bytes || payload.system?.memory_total_bytes),
        gpu: gpuValues.length ? chartPercent(Math.max(...gpuValues)) : null,
        gpuMemory: gpuTotal > 0 ? ratioPercent(gpuUsed, gpuTotal) : null,
      };
    }), [events]);
  const visibleLines = resourceLines.filter((line) => data.some((point) => point[line.key] != null));

  if (!data.length) return null;
  return <div className="resource-chart" aria-label="资源使用百分比折线图">
    <p>进程实际使用 {data.at(-1)?.cpuCores?.toFixed(2) ?? "—"} 核{data.at(-1)?.allocatedCpus ? ` / 分配 ${data.at(-1)?.allocatedCpus} 核` : "（历史记录未采集配额）"}；系统曲线为主机可见总体口径。</p>
    <div className="resource-chart-legend">{visibleLines.map((line) => <span key={line.key}><i style={{ background: line.color }} />{line.name}<b>{data.at(-1)?.[line.key]}%</b></span>)}</div>
    <ResponsiveContainer width="100%" height={270}>
      <LineChart data={data} margin={{ top: 10, right: 18, left: -12, bottom: 4 }}>
        <CartesianGrid stroke="#1c2c41" strokeDasharray="3 4" />
        <XAxis dataKey="time" minTickGap={42} axisLine={false} tickLine={false} tick={{ fill: "#7890ad", fontSize: 11 }} />
        <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(value) => `${value}%`} axisLine={false} tickLine={false} tick={{ fill: "#7890ad", fontSize: 11 }} />
        <Tooltip
          formatter={(value, name) => [`${value}%`, name]}
          labelFormatter={(label) => `时间 ${label}`}
          contentStyle={{ background: "#111d2d", border: "1px solid #243752", borderRadius: 10 }}
        />
        {visibleLines.map((line) => <Line key={line.key} type="monotone" dataKey={line.key} name={line.name} stroke={line.color} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  </div>;
}

function eventMessage(event: TaskEvent) {
  const message = event.payload?.message;
  if (typeof message === "string") return message;
  const line = event.payload?.line;
  return typeof line === "string" ? line : JSON.stringify(event.payload);
}

export function TerminalLogViewer({
  events,
  hasMore,
  loadingOlder,
  onLoadOlder,
}: {
  events: TaskEvent[];
  hasMore: boolean;
  loadingOlder?: boolean;
  onLoadOlder: () => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const previousOldestRef = useRef<number | undefined>(undefined);
  const preserveRef = useRef<{ height: number; top: number } | null>(null);
  const [following, setFollowing] = useState(true);
  const oldestId = events.at(0)?.id;
  const newestId = events.at(-1)?.id;

  const setFollow = (value: boolean) => {
    followRef.current = value;
    setFollowing(value);
  };

  const scrollToBottom = () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    setFollow(true);
    viewport.scrollTop = viewport.scrollHeight;
  };

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const prepended = previousOldestRef.current != null && oldestId !== previousOldestRef.current;
    if (prepended && preserveRef.current) {
      viewport.scrollTop = preserveRef.current.top + viewport.scrollHeight - preserveRef.current.height;
      preserveRef.current = null;
    } else if (followRef.current) {
      viewport.scrollTop = viewport.scrollHeight;
    }
    previousOldestRef.current = oldestId;
  }, [oldestId, newestId]);

  const loadOlder = () => {
    const viewport = viewportRef.current;
    if (viewport) preserveRef.current = { height: viewport.scrollHeight, top: viewport.scrollTop };
    setFollow(false);
    onLoadOlder();
  };

  return <div className="terminal-log-shell">
    <div className="terminal-log-toolbar">
      <span>{events.length} 条记录 · {following ? "实时跟随" : "已暂停跟随"}</span>
      <div>
        {hasMore && <button type="button" className="button ghost small" disabled={loadingOlder} onClick={loadOlder}>{loadingOlder && <LoaderCircle size={13} className="spin" />}加载更早</button>}
        {!following && <button type="button" className="button ghost small" onClick={scrollToBottom}><ArrowDown size={13} />回到底部</button>}
      </div>
    </div>
    <div
      ref={viewportRef}
      className="log-viewer terminal-log-viewer"
      aria-label="运行日志"
      onScroll={(event) => {
        const element = event.currentTarget;
        const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight <= 24;
        if (atBottom !== followRef.current) setFollow(atBottom);
      }}
    >
      {events.map((event) => <div key={event.id}><time>{formatDate(event.created_at)}</time><b>{String(event.payload?.stream || event.event_type)}</b><pre>{eventMessage(event)}</pre></div>)}
    </div>
  </div>;
}
