import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Blocks,
  Boxes,
  Cpu,
  Database,
  HardDrive,
  ServerCog,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";
import { api, formatDate } from "../api";
import type { Project, Worker } from "../types";
import "./Home.css";

const centers = [
  { to: "/data", label: "数据资产", description: "接入、检索、清洗并冻结可复用数据版本", icon: Database },
  { to: "/projects", label: "项目中心", description: "推进模型训练、评测、量化与发布生命周期", icon: Boxes },
  { to: "/benchmarks", label: "评测集", description: "按来源管理评测题目、评分配置与版本", icon: Blocks },
  { to: "/storage", label: "存储", description: "查看空间账本并安全清理可重建内容", icon: HardDrive },
  { to: "/execution", label: "计算", description: "管理本地、WSL 与远端执行环境", icon: Cpu },
] as const;

export function Home() {
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
  const datasets = useQuery({ queryKey: ["catalog-datasets"], queryFn: () => api<unknown[]>("/catalog/datasets") });
  const benchmarks = useQuery({ queryKey: ["catalog-evaluation-datasets"], queryFn: () => api<unknown[]>("/catalog/evaluation-datasets") });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<Worker[]>("/workers") });

  const recentProjects = [...(projects.data || [])]
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    .slice(0, 3);
  const onlineWorkers = (workers.data || []).filter((worker) => worker.status === "online").length;
  const hasLoadError = [projects, datasets, benchmarks, workers].some((query) => query.isError);

  return (
    <main className="system-page home-page">
      <section className="home-hero">
        <div>
          <p className="eyebrow">FORGELM CONTROL PLANE</p>
          <h1>从全局资产出发，<br />推进每一个模型版本。</h1>
          <p>在同一个本地控制面中组织数据、项目、评测与算力，让实验链路保持可复现、可比较、可发布。</p>
          <div className="home-hero-actions">
            <Link className="button primary" to="/projects"><Boxes size={17} />进入项目中心</Link>
            <Link className="button primary" to="/data"><Database size={17} />浏览数据资产</Link>
          </div>
        </div>
        <div className="home-signal-card">
          <Sparkles size={22} />
          <span>本地工作区</span>
          <strong>{onlineWorkers}</strong>
          <small>{workers.isPending ? "正在读取执行环境…" : `个在线 Worker · 共 ${workers.data?.length || 0} 个节点`}</small>
        </div>
      </section>

      <section className="home-metrics" aria-label="平台概览">
        <HomeMetric icon={Boxes} label="模型项目" value={projects.data?.length} loading={projects.isPending} detail="独立生命周期工作区" to="/projects" />
        <HomeMetric icon={Database} label="数据资产" value={datasets.data?.length} loading={datasets.isPending} detail="全局可发现数据集" to="/data" />
        <HomeMetric icon={Blocks} label="评测集版本" value={benchmarks.data?.length} loading={benchmarks.isPending} detail="可绑定评测集" to="/benchmarks" />
        <HomeMetric icon={ServerCog} label="在线 Worker" value={onlineWorkers} loading={workers.isPending} detail="当前可调度节点" to="/execution" />
      </section>

      {hasLoadError && <p className="home-load-note">部分实时概览暂时不可用，各中心仍可独立打开。</p>}

      <section className="home-workspace-grid">
        <div className="home-centers">
          <header><div><span>WORKSPACES</span><h2>进入工作中心</h2></div><small>按资产与职责划分</small></header>
          <div>
            {centers.map(({ to, label, description, icon: Icon }) => (
              <Link to={to} key={to}>
                <span className="home-center-icon"><Icon size={20} /></span>
                <span><strong>{label}</strong><small>{description}</small></span>
                <ArrowRight size={16} />
              </Link>
            ))}
          </div>
        </div>

        <div className="home-recent-projects">
          <header><div><span>RECENT PROJECTS</span><h2>最近项目</h2></div><Link to="/projects">查看全部<ArrowRight size={14} /></Link></header>
          <div>
            {recentProjects.map((project) => (
              <Link to={`/projects/${project.id}/overview`} key={project.id}>
                <span className="home-project-mark">{project.name.slice(0, 2).toUpperCase()}</span>
                <span><strong>{project.name}</strong><small>{project.target_model_name} · {project.target_parameters}</small></span>
                <span><b>{project.lifecycle_stage || "未开始"}</b><small>{formatDate(project.updated_at)}</small></span>
              </Link>
            ))}
            {!projects.isPending && !recentProjects.length && (
              <Link className="home-empty-projects" to="/projects"><Boxes size={20} /><span><strong>创建第一个模型项目</strong><small>从目标、工作区和验收标准开始。</small></span><ArrowRight size={16} /></Link>
            )}
            {projects.isPending && <div className="home-project-loading">正在读取最近项目…</div>}
          </div>
        </div>
      </section>
    </main>
  );
}

function HomeMetric({ icon: Icon, label, value, loading, detail, to }: { icon: typeof Boxes; label: string; value?: number; loading: boolean; detail: string; to: string }) {
  return (
    <Link to={to}>
      <Icon size={18} />
      <span>{label}</span>
      <strong>{loading ? "—" : value || 0}</strong>
      <small>{detail}</small>
      <ArrowRight size={15} />
    </Link>
  );
}
