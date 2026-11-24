import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Archive,
  Blocks,
  Boxes,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  Database,
  ExternalLink,
  FlaskConical,
  HardDrive,
  Layers3,
  Menu,
  MessageSquareText,
  PackageCheck,
  Pause,
  Play,
  ServerCog,
  Square,
  X,
} from "lucide-react";
import { Suspense, useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, formatNumber, post } from "../api";
import type { Project, Task } from "../types";
import { ControlPlaneControl } from "./ControlPlaneControl";
import { TaskDetail } from "./TaskDetail";
import { TaskTiming } from "./TaskTiming";
import { Progress, StatusBadge } from "./ui";
import { RouteErrorBoundary } from "./RouteErrorBoundary";
import { RouteLoading } from "./RouteLoading";

const assetNavigation = [
  ["data", "项目数据集", Database],
  ["tokenizers", "Tokenizer 实验室", FlaskConical],
  ["models", "模型与 Checkpoint", Layers3],
  ["evaluations", "项目评测", Blocks],
  ["releases", "量化与发布", PackageCheck],
] as const;

const downloadPhases = new Set(["preparing", "downloading", "starting_tool", "retrying_download", "retrying_source_listing"]);
const terminalTaskStates = new Set(["succeeded", "failed", "cancelled", "lost"]);
const taskPageSize = 20;
const safeTrainingPauseProtocol = "forgelm-command-control-file-v1";

function isResumableDownload(task: Task) {
  return task.adapter_id === "builtin.data-import" && ["http", "https", "huggingface"].includes(String(task.payload?.source_type || "").toLowerCase());
}

function canPauseDownload(task: Task) {
  if (!isResumableDownload(task) || task.desired_state === "paused") return false;
  if (["queued", "assigned", "starting"].includes(task.observed_state)) return true;
  return task.observed_state === "running" && downloadPhases.has(task.current_phase || "preparing");
}

function canStopTask(task: Task) {
  return !terminalTaskStates.has(task.observed_state);
}

function supportsSafeTrainingPause(task: Task) {
  return task.kind === "train" && Boolean(task.payload?.training_run_id) && task.payload?.command?.control?.safe_pause?.protocol === safeTrainingPauseProtocol;
}

function canSafePauseTraining(task: Task) {
  return supportsSafeTrainingPause(task) && ["assigned", "starting", "running"].includes(task.observed_state) && task.desired_state !== "paused";
}

function canResumeTraining(task: Task) {
  return task.kind === "train" && ["paused", "cancelled"].includes(task.observed_state) && Boolean(task.payload?.training_run_id);
}

function trainingTaskContext(task: Task) {
  if (task.kind !== "train") return null;
  const stages = Array.isArray(task.payload?.stages) ? task.payload.stages : [];
  if (!stages.length) return null;
  const targetTokens = stages.reduce((total: number, stage: any) => total + Number(stage?.target_tokens || 0), 0);
  const cumulativeTokens = Number(stages.at(-1)?.cumulative_target_tokens || 0);
  const plannedSteps = Number(task.payload?.training_data_range?.planned_steps || 0);
  const stageNames = stages.map((stage: any) => String(stage?.name || "未命名阶段")).join(" → ");
  const parts = [stageNames];
  if (targetTokens > 0) parts.push(`本次 ${formatNumber(targetTokens)} Tokens`);
  if (cumulativeTokens > 0 && cumulativeTokens !== targetTokens) parts.push(`累计到 ${formatNumber(cumulativeTokens)} Tokens`);
  if (plannedSteps > 0) parts.push(`${formatNumber(plannedSteps)} Steps`);
  return parts.join(" · ");
}

function taskOrigin(task: Task, projectId: string) {
  if (task.payload?.pipeline_instance_id) return { href: `/projects/${projectId}/training?instance=${encodeURIComponent(String(task.payload.pipeline_instance_id))}`, label: "打开训练实例" };
  const trainingRunId = task.kind === "train" ? String(task.payload?.training_run_id || "") : "";
  if (!trainingRunId) return null;
  return {
    href: `/projects/${projectId}/training?run=${encodeURIComponent(trainingRunId)}`,
    label: "打开训练运行配置",
  };
}

function isInteractiveTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && Boolean(target.closest("button, a, input, select, textarea, summary"));
}

export function ProjectShell() {
  const { projectId = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [taskOpen, setTaskOpen] = useState(false);
  const [taskPage, setTaskPage] = useState(0);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const taskDrawerRef = useRef<HTMLElement>(null);
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api<Project>(`/projects/${projectId}`) });
  const tasks = useQuery({
    queryKey: ["tasks", projectId, "pipeline", taskPage],
    queryFn: () => api<Task[]>(`/tasks?project_id=${projectId}&scope=pipeline&limit=${taskPageSize + 1}&offset=${taskPage * taskPageSize}`),
    refetchInterval: 2000,
  });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<any[]>("/workers"), refetchInterval: 5000 });
  const taskControl = useMutation({
    mutationFn: ({ task, action }: { task: Task; action: "pause" | "resume" | "cancel" }) => post<Task>(`/tasks/${task.id}/control`, {
      action,
      reason: action === "pause"
        ? task.kind === "train"
          ? "用户从任务中心安全暂停训练"
          : "用户从任务中心暂停下载"
        : action === "resume"
          ? task.kind === "train"
            ? "用户从任务中心继续执行训练"
            : "用户从任务中心继续下载"
          : supportsSafeTrainingPause(task)
            ? "用户从任务中心立即停止训练"
            : "用户从任务中心停止任务",
    }),
    onSuccess: (updated) => queryClient.setQueryData<Task[]>(["tasks", projectId, "pipeline", taskPage], (current) => (current || []).map((task) => task.id === updated.id ? updated : task)),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["tasks", projectId] }),
  });
  const taskRows = (tasks.data || []).slice(0, taskPageSize);
  const hasNextTaskPage = (tasks.data?.length || 0) > taskPageSize;
  const activeCount = taskRows.filter((task) => !terminalTaskStates.has(task.observed_state)).length;
  function closeTaskCenter() {
    setTaskOpen(false);
    setTaskPage(0);
  }

  function changeTaskPage(page: number) {
    setTaskPage(Math.max(0, page));
    if (taskDrawerRef.current) taskDrawerRef.current.scrollTop = 0;
  }

  function openTaskOrigin(href: string) {
    setSelectedTaskId("");
    closeTaskCenter();
    navigate(href);
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><div className="brand-mark"><Boxes size={19} /></div><div><strong>ForgeLM</strong><span>Local lifecycle studio</span></div></div>
        <button className="back-link" onClick={() => navigate("/projects")}><ChevronLeft size={16} />项目中心</button>
        <div className="project-identity"><span>当前项目</span><strong>{project.data?.name || "加载中…"}</strong><small>{project.data?.target_model_name} · {project.data?.target_parameters}</small></div>
        <nav className="primary-nav" aria-label="项目导航">
          <NavLink to={`/projects/${projectId}/overview`} onClick={() => setMobileOpen(false)}><CircleGauge size={18} /><span>项目总览</span></NavLink>
          <NavLink to={`/projects/${projectId}/training`} onClick={() => setMobileOpen(false)}><ChartNoAxesCombined size={18} /><span>训练</span></NavLink>
          {assetNavigation.map(([path, label, Icon]) => <NavLink key={path} to={`/projects/${projectId}/${path}`} onClick={() => setMobileOpen(false)}><Icon size={18} /><span>{label}</span></NavLink>)}
          <NavLink to={`/projects/${projectId}/playground`} onClick={() => setMobileOpen(false)}><MessageSquareText size={18} /><span>推理 Playground</span></NavLink>
        </nav>
        <div className="sidebar-footer">
          <ControlPlaneControl />
          <button className="worker-state" onClick={() => navigate("/storage")}><HardDrive size={16} /><div><strong>存储空间管理</strong><span>盘点、预览和安全清理</span></div></button>
          <button className="worker-state" onClick={() => navigate("/execution")}><ServerCog size={16} /><div><strong>{workers.data?.filter((worker) => worker.status === "online").length || 0} 个 Worker 在线</strong><span>管理执行环境</span></div></button>
          <div className="sidebar-note"><Archive size={15} />本地制品受引用保护</div>
        </div>
      </aside>
      <div className="app-main">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="打开导航"><Menu size={20} /></button>
          <div className="breadcrumbs"><span>项目</span><b>/</b><strong>{project.data?.name || "…"}</strong></div>
          <button className="task-button" onClick={() => { setTaskPage(0); setTaskOpen(true); }}><span className={activeCount ? "pulse-dot" : "quiet-dot"} />任务中心{activeCount > 0 && <b>{activeCount}</b>}</button>
        </header>
        <main className="page"><RouteErrorBoundary><Suspense fallback={<RouteLoading />}><Outlet /></Suspense></RouteErrorBoundary></main>
      </div>
      {mobileOpen && <button className="mobile-scrim" onClick={() => setMobileOpen(false)} aria-label="关闭导航" />}
      {taskOpen && (
        <aside ref={taskDrawerRef} className="task-drawer">
          <header><div><span className="eyebrow">产线节点执行</span><h2>任务中心</h2></div><button className="icon-button" onClick={closeTaskCenter}><X size={19} /></button></header>
          <div className="task-list">
            {taskRows.map((task) => { const origin = taskOrigin(task, projectId); const context = trainingTaskContext(task); return (
              <article key={task.id} className={`task-row ${origin ? "task-row-linked" : ""}`} tabIndex={origin ? 0 : undefined} aria-label={origin ? `打开来源配置 ${task.title}` : undefined} onClick={origin ? (event) => { if (!isInteractiveTarget(event.target)) openTaskOrigin(origin.href); } : undefined} onKeyDown={origin ? (event) => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); openTaskOrigin(origin.href); } } : undefined}>
                <div className="task-row-head"><strong>{task.title}</strong><StatusBadge status={task.observed_state} /></div>
                <p className={context ? "task-origin-context" : ""}>{context || task.current_phase || task.kind}</p>
                <Progress value={task.progress} />
                <small>{Math.round(task.progress * 100)}% · {task.id.slice(0, 8)}</small>
                <TaskTiming task={task} />
                <div className="task-row-links">
                  {origin && <button type="button" className="button primary small" onClick={(event) => { event.stopPropagation(); openTaskOrigin(origin.href); }}><ExternalLink size={13} />{origin.label}</button>}
                  <button type="button" className="button ghost small task-detail-button" onClick={(event) => { event.stopPropagation(); setSelectedTaskId(task.id); }}><Activity size={13} />查看资源与日志</button>
                </div>
                {(canStopTask(task) || canResumeTraining(task)) && (
                  <div className="task-row-actions">
                    {canResumeTraining(task) && <button type="button" className="button primary small" aria-label={`继续执行 ${task.title}`} disabled={taskControl.isPending} onClick={(event) => { event.stopPropagation(); taskControl.mutate({ task, action: "resume" }); }}><Play size={13} />继续执行</button>}
                    {canSafePauseTraining(task) && <button type="button" className="button ghost small" aria-label={`安全暂停 ${task.title}`} disabled={taskControl.isPending} title="完成当前 optimizer step 并保存完整 Checkpoint 后暂停" onClick={(event) => { event.stopPropagation(); taskControl.mutate({ task, action: "pause" }); }}><Pause size={13} />安全暂停</button>}
                    {supportsSafeTrainingPause(task) && task.desired_state === "paused" && task.observed_state !== "paused" && <span>正在完成当前 Step 并保存 Checkpoint…</span>}
                    {isResumableDownload(task) && canPauseDownload(task) && <button type="button" className="button ghost small" aria-label={`暂停下载 ${task.title}`} disabled={taskControl.isPending} onClick={(event) => { event.stopPropagation(); taskControl.mutate({ task, action: "pause" }); }}><Pause size={13} />{task.observed_state === "queued" ? "暂停排队" : "暂停下载"}</button>}
                    {isResumableDownload(task) && task.observed_state === "paused" && <button type="button" className="button primary small" aria-label={`继续下载 ${task.title}`} disabled={taskControl.isPending} onClick={(event) => { event.stopPropagation(); taskControl.mutate({ task, action: "resume" }); }}><Play size={13} />继续下载</button>}
                    {isResumableDownload(task) && task.desired_state === "paused" && task.observed_state !== "paused" && <span>正在暂停当前下载进程…</span>}
                    {canStopTask(task) && <button type="button" className="button danger small" aria-label={`${supportsSafeTrainingPause(task) ? "立即停止" : "停止任务"} ${task.title}`} disabled={taskControl.isPending || task.observed_state === "stopping" || task.desired_state === "cancelled"} onClick={(event) => { event.stopPropagation(); taskControl.mutate({ task, action: "cancel" }); }}><Square size={13} />{task.observed_state === "stopping" || task.desired_state === "cancelled" ? "正在停止…" : supportsSafeTrainingPause(task) ? "立即停止" : "停止任务"}</button>}
                  </div>
                )}
              </article>
            ); })}
            {taskControl.isError && <p className="inline-error">{taskControl.error.message}</p>}
            {tasks.isPending && <p className="muted">正在加载第 {taskPage + 1} 页…</p>}
            {!tasks.isPending && !taskRows.length && <p className="muted">{taskPage ? "这一页没有更多任务。" : "当前项目还没有产线节点任务。"}</p>}
            <footer className="task-pagination" aria-label="任务分页">
              <button type="button" className="button secondary small" disabled={taskPage === 0 || tasks.isFetching} onClick={() => changeTaskPage(taskPage - 1)}><ChevronLeft size={14} />上一页</button>
              <span>第 {taskPage + 1} 页 · 每页 {taskPageSize} 条</span>
              <button type="button" className="button secondary small" disabled={!hasNextTaskPage || tasks.isFetching} onClick={() => changeTaskPage(taskPage + 1)}>下一页<ChevronRight size={14} /></button>
            </footer>
          </div>
        </aside>
      )}
      {selectedTaskId && <TaskDetail taskId={selectedTaskId} onClose={() => setSelectedTaskId("")} />}
    </div>
  );
}
