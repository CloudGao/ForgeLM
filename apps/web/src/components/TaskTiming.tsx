import { Clock3 } from "lucide-react";
import type { ReactNode } from "react";
import { formatDate } from "../api";
import type { Task } from "../types";

export function formatDuration(seconds?: number | null) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return "不到 1 分钟";
  const minutes = Math.max(1, Math.round(seconds / 60));
  if (minutes < 60) return `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours < 48) return remainingMinutes ? `${hours} 小时 ${remainingMinutes} 分钟` : `${hours} 小时`;
  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return remainingHours ? `${days} 天 ${remainingHours} 小时` : `${days} 天`;
}

export function TaskTiming({ task, detail = false }: { task: Task; detail?: boolean }) {
  const timing = task.timing_estimate;
  if (!timing) return null;

  let content: ReactNode;
  if (timing.status === "complete") {
    content = <span>实际执行 {formatDuration(timing.actual_duration_seconds)}</span>;
  } else if (timing.status === "waiting") {
    content = <span>任务开始并产生进度后估算耗时</span>;
  } else if (timing.status === "unavailable") {
    content = <span>任务状态切换中，暂不估算剩余时间</span>;
  } else if (timing.estimated_total_seconds == null || timing.remaining_seconds == null) {
    content = <span>{timing.status === "paused" ? "已暂停，恢复并积累进度后估算" : "正在积累有效进度，稍后估算耗时"}</span>;
  } else {
    content = <>
      <span>预计总执行 {formatDuration(timing.estimated_total_seconds)}</span>
      <span>{timing.status === "paused" ? "恢复后剩余约" : "剩余约"} {formatDuration(timing.remaining_seconds)}</span>
      {detail && timing.estimated_completion_at && <span>预计完成 {formatDate(timing.estimated_completion_at)}</span>}
    </>;
  }

  return <div className={`task-timing ${detail ? "task-timing-detail" : ""}`} title="基于当前 attempt 最近的真实进度动态估算，实际耗时可能随阶段和吞吐变化">
    <Clock3 size={detail ? 14 : 12} />
    {content}
  </div>;
}
