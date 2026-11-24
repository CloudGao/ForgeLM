import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, ArrowRight, Copy, Plus, Search, Trash2 } from "lucide-react";
import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, formatDate, post } from "../api";
import { EmptyState, ErrorMessage, Field, FormActions, Modal, Progress, StatusBadge } from "../components/ui";
import type { Project } from "../types";

const stageLabels: Record<string, string> = { data_preparation: "数据准备", tokenizer: "Tokenizer", pretraining: "预训练", post_training: "后训练", evaluation: "评测", quantization: "量化", release: "发布" };

export function ProjectCenter() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [deleting, setDeleting] = useState<Project | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const deletionPreview = useQuery({ queryKey: ["project-deletion", deleting?.id], queryFn: () => api<{ fingerprint: string; can_delete: boolean; record_counts: Record<string, number>; paths: { path: string; size_bytes: number }[]; blockers: { kind: string; id?: string; path?: string }[] }>(`/projects/${deleting!.id}/deletion-preview`), enabled: Boolean(deleting), staleTime: 0 });
  const remove = useMutation({ mutationFn: () => post(`/projects/${deleting!.id}/delete`, { confirmation, fingerprint: deletionPreview.data!.fingerprint }), onSuccess: () => { setDeleting(null); setConfirmation(""); queryClient.invalidateQueries({ queryKey: ["projects"] }); }, onError: () => queryClient.invalidateQueries({ queryKey: ["project-deletion"] }) });
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
  const create = useMutation({
    mutationFn: (data: any) => post<Project>("/projects", data),
    onSuccess: (project) => { queryClient.invalidateQueries({ queryKey: ["projects"] }); setCreateOpen(false); navigate(`/projects/${project.id}/overview`); },
  });
  const archive = useMutation({ mutationFn: (id: string) => post(`/projects/${id}/archive`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }) });
  const copy = useMutation({ mutationFn: (id: string) => post<Project>(`/projects/${id}/copy`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }) });
  const filtered = (projects.data || []).filter((project) => `${project.name} ${project.target_model_name} ${project.workspace_uri}`.toLowerCase().includes(search.toLowerCase()));

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    create.mutate({
      name: form.get("name"),
      target_model_name: form.get("target_model_name"),
      workspace_uri: String(form.get("workspace_uri") || "").trim(),
      project_type: form.get("project_type"),
      target_parameters: form.get("target_parameters"),
      primary_languages: String(form.get("languages") || "").split(/[,，]/).map((x) => x.trim()).filter(Boolean),
      target_capabilities: String(form.get("capabilities") || "").split(/[,，]/).map((x) => x.trim()).filter(Boolean),
      default_hardware: { description: form.get("hardware") },
      training_budget: { description: form.get("budget") },
      acceptance_goal: { description: form.get("goal") },
    });
  }

  return (
    <main className="project-center-main">
        <header className="project-list-header">
          <div><h1>项目清单</h1><p>{projects.isPending ? "正在读取项目…" : `共 ${projects.data?.length || 0} 个项目`}</p></div>
          <button className="button primary" onClick={() => setCreateOpen(true)}><Plus size={17} />新建项目</button>
        </header>
        <div className="project-toolbar">
          <div className="search-box"><Search size={17} /><input aria-label="搜索项目" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索项目、目标模型或工作区" /></div>
          {!projects.isPending && <span className="project-result-count">显示 {filtered.length} 个</span>}
        </div>
        {projects.isPending ? (
          <div className="project-list-loading">正在读取项目…</div>
        ) : filtered.length > 0 ? (
          <section className="project-list" aria-label="项目清单">
            <header className="project-list-columns" aria-hidden="true">
              <span>项目</span><span>当前阶段</span><span>数据准备</span><span>训练状态</span><span>最后更新</span><span>操作</span>
            </header>
            {filtered.map((project) => (
              <article className="project-list-row" key={project.id}>
                <div className="project-list-identity">
                  <div className="project-icon">{project.name.slice(0, 2).toUpperCase()}</div>
                  <div><h2>{project.name}</h2><p>{project.project_type === "pretrain" ? "从零预训练" : "继续训练"} · {project.target_model_name} · {project.target_parameters}</p></div>
                </div>
                <div className="project-list-stage"><span>当前阶段</span><strong>{stageLabels[project.lifecycle_stage] || project.lifecycle_stage || "未开始"}</strong></div>
                <div className="project-list-data">
                  <div><span>{project.data_progress.dataset_count} 个数据集 · {project.data_progress.ready_versions} 个可训练版本</span><b>{project.data_progress.percent}%</b></div>
                  <Progress value={project.data_progress.percent / 100} />
                </div>
                <div className="project-list-status"><span>训练状态</span><StatusBadge status={project.recent_training_status} /></div>
                <time dateTime={project.updated_at}>{formatDate(project.updated_at)}</time>
                <div className="project-list-actions">
                  <button className="icon-button" aria-label={`复制项目 ${project.name}`} title="复制" onClick={() => copy.mutate(project.id)}><Copy size={16} /></button>
                  <button className="icon-button" aria-label={`归档项目 ${project.name}`} title="归档" onClick={() => archive.mutate(project.id)}><Archive size={16} /></button>
                  <button className="icon-button" aria-label={`删除项目 ${project.name}`} title="永久删除" onClick={() => { setConfirmation(""); remove.reset(); setDeleting(project); }}><Trash2 size={16} /></button>
                  <button className="button small" onClick={() => navigate(`/projects/${project.id}/overview`)}>打开<ArrowRight size={15} /></button>
                </div>
              </article>
            ))}
          </section>
        ) : <EmptyState title="还没有匹配的项目" description="新建一个模型项目，生命周期追踪会从数据准备开始。" action={<button className="button primary" onClick={() => setCreateOpen(true)}><Plus size={17} />新建项目</button>} />}
      {deleting && <Modal title="永久删除项目" description="删除项目记录及预览中的专属工作目录。共享原始数据保留；存在活动任务或跨项目引用时禁止执行。" onClose={() => { if (!remove.isPending) setDeleting(null); }} wide>
        {deletionPreview.isPending ? <p>正在检查引用和目录…</p> : deletionPreview.data && <>
          <p>{Object.values(deletionPreview.data.record_counts).reduce((a, b) => a + b, 0)} 条项目记录</p>
          <ul>{deletionPreview.data.paths.map((p) => <li key={p.path}>{p.path} · {(p.size_bytes / 1e9).toFixed(3)} GB</li>)}</ul>
          {!deletionPreview.data.can_delete && <><p>以下项目阻止删除：</p><ul>{deletionPreview.data.blockers.map((b, i) => <li key={i}>{b.kind} {b.id || b.path}</li>)}</ul></>}
          <Field label={`输入项目名称：${deleting.name}`}><input aria-label="确认删除的项目名称" value={confirmation} onChange={(e) => setConfirmation(e.target.value)} /></Field>
          <button className="button primary" disabled={remove.isPending || !deletionPreview.data.can_delete || confirmation !== deleting.name} onClick={() => remove.mutate()}>{remove.isPending ? "正在删除…" : "永久删除项目及列出的目录"}</button>
        </>}
        <ErrorMessage error={deletionPreview.error || remove.error} />
      </Modal>}
      {createOpen && (
        <Modal title="新建模型项目" description="先定义目标和验收标准，具体配置可在项目中继续完善。" onClose={() => setCreateOpen(false)} wide>
          <form className="form-grid two-columns" onSubmit={submit}>
            <Field label="项目名称"><input name="name" required placeholder="例如 ExampleLM-300M" /></Field>
            <Field label="目标模型名称"><input name="target_model_name" required placeholder="例如 ExampleLM-300M" /></Field>
            <div className="span-two"><Field label="项目地址" hint="控制面独占工作目录；Worker 可在自己的执行根中映射该项目"><input name="workspace_uri" placeholder="例如 F:\projects\example-lm 或 file:///workspace/example-lm" required /></Field></div>
            <Field label="项目类型"><select name="project_type"><option value="pretrain">从零预训练</option><option value="continue_training">已有模型继续训练</option></select></Field>
            <Field label="目标参数量"><input name="target_parameters" defaultValue="300M" /></Field>
            <Field label="主要语言" hint="用逗号分隔"><input name="languages" defaultValue="中文, English" /></Field>
            <Field label="主要能力" hint="用逗号分隔"><input name="capabilities" defaultValue="通用对话, 基础代码" /></Field>
            <Field label="默认硬件"><input name="hardware" placeholder="例如 4 × RTX 4090" /></Field>
            <Field label="训练预算"><input name="budget" placeholder="例如 6B Tokens / 480 GPU 小时" /></Field>
            <Field label="验收目标"><textarea name="goal" placeholder="模型发布前必须达到的指标" /></Field>
            <ErrorMessage error={create.error} />
            <div className="span-two"><FormActions submitting={create.isPending} onCancel={() => setCreateOpen(false)} submitLabel="创建并打开" /></div>
          </form>
        </Modal>
      )}
    </main>
  );
}
