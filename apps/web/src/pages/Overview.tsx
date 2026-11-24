import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowRight, Code2, Database, Gauge, GitBranch, MapPin, PackageCheck, Pencil, RefreshCw, ShieldAlert, Upload } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, formatDate, formatNumber, patch, post, postForm } from "../api";
import { Alert, ErrorMessage, Field, FormActions, MetricCard, Modal, PageHeader, Panel, Progress, StatusBadge } from "../components/ui";
import type { CodeSource } from "../types";

const purposeLabels: Record<string, string> = { training: "训练", validation: "验证", evaluation: "评测", reference: "参考" };

export function Overview() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedAction = searchParams.get("action");
  const queryClient = useQueryClient();
  const [editAddressOpen, setEditAddressOpen] = useState(false);
  const [importCodeOpen, setImportCodeOpen] = useState(false);
  const [codeSourceType, setCodeSourceType] = useState("git");
  const [reuploadSource, setReuploadSource] = useState<CodeSource | null>(null);
  const [codeFormError, setCodeFormError] = useState("");
  const dashboard = useQuery({ queryKey: ["dashboard", projectId], queryFn: () => api<any>(`/projects/${projectId}/dashboard`), refetchInterval: 4000 });
  const codeSources = useQuery({ queryKey: ["code-sources", projectId], queryFn: () => api<CodeSource[]>(`/projects/${projectId}/code-sources`), refetchInterval: (query) => (query.state.data || []).some((source) => ["queued", "assigned", "starting", "running"].includes(source.status)) ? 2000 : false });
  const updateAddress = useMutation({
    mutationFn: (workspaceUri: string) => patch<any>(`/projects/${projectId}`, { workspace_uri: workspaceUri }),
    onSuccess: (project) => {
      queryClient.setQueryData(["dashboard", projectId], (previous: any) => previous ? { ...previous, project } : previous);
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      setEditAddressOpen(false);
    },
  });
  const importCode = useMutation({
    mutationFn: (data: { upload: boolean; body: any; sourceId?: string }) => data.upload ? postForm<CodeSource>(data.sourceId ? `/code-sources/${data.sourceId}/versions/upload` : `/projects/${projectId}/code-sources/upload`, data.body) : post<CodeSource>(`/projects/${projectId}/code-sources`, data.body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["code-sources", projectId] });
      queryClient.invalidateQueries({ queryKey: ["tasks", projectId] });
      setImportCodeOpen(false);
      setReuploadSource(null);
    },
  });
  const reimportCode = useMutation({
    mutationFn: (sourceId: string) => post(`/code-sources/${sourceId}/versions`, {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["code-sources", projectId] }),
  });
  const data = dashboard.data;
  const current = data?.current || {};
  const chartData = current.training_run?.latest_metrics?.history || [];
  const dataAsset = current.data_asset;

  useEffect(() => {
    if (requestedAction !== "import-code") return;
    setReuploadSource(null);
    setCodeSourceType("git");
    setImportCodeOpen(true);
  }, [requestedAction]);

  function saveAddress(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    updateAddress.mutate(String(form.get("workspace_uri") || "").trim());
  }

  function submitCode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setCodeFormError("");
    const entrypoints = {
      training: String(form.get("training_entrypoint") || "").trim(),
      model_factory: String(form.get("model_factory_entrypoint") || "").trim(),
    };
    if (!entrypoints.training) delete (entrypoints as any).training;
    if (!entrypoints.model_factory) delete (entrypoints as any).model_factory;
    let configSchema: any = {};
    try {
      configSchema = JSON.parse(String(form.get("config_schema_json") || "{}"));
    } catch {
      setCodeFormError("模型配置 Schema 不是有效 JSON");
      return;
    }
    if (!configSchema || typeof configSchema !== "object" || Array.isArray(configSchema)) {
      setCodeFormError("模型配置 Schema 必须是 JSON 对象");
      return;
    }
    if (codeSourceType === "upload") {
      const upload = new FormData();
      upload.set("name", String(form.get("name") || ""));
      upload.set("file", form.get("file") as File);
      upload.set("entrypoints_json", JSON.stringify(entrypoints));
      upload.set("config_schema_json", JSON.stringify(configSchema));
      upload.set("dependency_spec", String(form.get("dependency_spec") || ""));
      importCode.mutate({ upload: true, body: upload, sourceId: reuploadSource?.id });
      return;
    }
    importCode.mutate({ upload: false, body: {
      name: form.get("name"),
      source_type: codeSourceType,
      source_uri: form.get("source_uri"),
      source_config: {
        ...(String(form.get("revision") || "").trim() ? { revision: String(form.get("revision")).trim() } : {}),
        ...(String(form.get("subdirectory") || "").trim() ? { subdirectory: String(form.get("subdirectory")).trim() } : {}),
      },
      entrypoints,
      config_schema: configSchema,
      dependency_spec: form.get("dependency_spec"),
      resource_request: { gpus: 0 },
    } });
  }

  return (
    <>
      <PageHeader
        eyebrow="PROJECT OVERVIEW"
        title="项目总览"
        description="汇总项目地址、代码、最近领域事实、训练趋势、关键异常和活动任务；训练实例、工作流与执行记录统一在训练页面管理。"
        actions={<button className="button secondary" onClick={() => dashboard.refetch()}><Activity size={16} />刷新状态</button>}
      />
      <Panel
        className="project-address-panel"
        title="项目地址"
        description="定位项目工作目录和项目级数据资产。"
        actions={<button className="button secondary small" onClick={() => setEditAddressOpen(true)}><Pencil size={14} />配置地址</button>}
      >
        <div className={`project-address ${data?.project?.workspace_uri ? "configured" : ""}`}>
          <MapPin size={18} />
          <div>
            <strong>{data?.project?.workspace_uri || "尚未配置"}</strong>
            <span>{data?.project?.workspace_uri ? "控制面托管制品保存在该目录的 .llm-platform 下；配置本地根的 Worker 使用自己的项目映射目录。" : "保存控制面托管制品前必须配置控制面可访问的项目目录。"}</span>
          </div>
        </div>
      </Panel>
      <Panel
        className="code-source-panel"
        title="模型代码"
        description="导入并冻结可复现的代码版本；训练计划可以绑定具体版本。"
        actions={<button className="button primary small" onClick={() => { setReuploadSource(null); setCodeSourceType("git"); setImportCodeOpen(true); }}><Upload size={14} />导入代码</button>}
      >
        {(codeSources.data || []).length ? <div className="code-source-list">{(codeSources.data || []).map((source) => {
          const version = source.current_version;
          return <article key={source.id}>
            <div className="code-source-icon">{source.source_type === "git" ? <GitBranch size={18} /> : <Code2 size={18} />}</div>
            <div className="code-source-copy"><div><strong>{source.name}</strong><StatusBadge status={version?.state || source.status} /></div><span>{source.source_uri}</span><small>{version ? `V${version.version} · ${version.git_commit?.slice(0, 10) || version.artifact?.content_hash?.slice(0, 10) || "等待制品"} · ${formatDate(version.created_at)}` : "尚无版本"}</small></div>
            {source.source_type === "upload" ? <button className="button ghost small" onClick={() => { setReuploadSource(source); setCodeSourceType("upload"); setImportCodeOpen(true); }}><Upload size={14} />上传新版本</button> : <button className="button ghost small" disabled={reimportCode.isPending || ["queued", "assigned", "starting", "running"].includes(source.status)} onClick={() => reimportCode.mutate(source.id)}><RefreshCw size={14} />重新导入</button>}
          </article>;
        })}</div> : <div className="quiet-message">尚未导入模型代码。导入后可在训练计划中锁定版本。</div>}
        <ErrorMessage error={codeSources.error || reimportCode.error} />
      </Panel>
      <section className="metrics-grid four">
        <MetricCard label="项目级最新数据" value={formatNumber(dataAsset?.token_estimate ?? current.dataset_version?.token_estimate)} detail={dataAsset ? `${dataAsset.name} · V${dataAsset.version}` : "Tokens · 尚未绑定版本"} />
        <MetricCard label="项目级最新训练 Loss" value={current.training_run?.latest_metrics?.loss ?? "—"} detail={current.training_run ? `${formatNumber(current.training_run.current_step)} steps` : "暂无训练"} tone="cyan" />
        <MetricCard label="项目级最新综合评测" value={current.evaluation?.summary?.overall ? `${(current.evaluation.summary.overall * 100).toFixed(1)}%` : "—"} detail="发布门禁套件" tone="violet" />
        <MetricCard label="项目级正式发布" value={current.release?.version || "未发布"} detail={current.release?.name || "等待模型晋升"} tone="green" />
      </section>
      <div className="content-grid overview-grid">
        <Panel title="项目级最近事实" description="这些是项目范围内的最近领域记录，仅作入口，不代表同一次训练执行。">
          <div className="snapshot-list">
            <Snapshot icon={<Database size={18} />} title="最近数据记录" value={dataAsset ? `${dataAsset.name} · V${dataAsset.version}` : current.dataset_version ? `${current.dataset_version.tier.toUpperCase()} V${current.dataset_version.version}` : "未准备"} detail={dataAsset ? `${formatNumber(dataAsset.document_count)} 条 · ${(dataAsset.purposes || []).map((purpose: string) => purposeLabels[purpose] || purpose).join(" / ")}` : `${formatNumber(current.dataset_version?.document_count)} 文档`} onClick={() => navigate("../data")} />
            <Snapshot icon={<Gauge size={18} />} title="项目 Tokenizer" value={current.tokenizer ? `${current.tokenizer.name} · V${current.tokenizer.version}` : "未选择"} detail={current.tokenizer ? `${formatNumber(current.tokenizer.vocab_size)} 词表 · ${current.tokenizer.is_default ? "项目默认" : current.tokenizer.locked ? "已锁定" : "待锁定"}` : "创建或选择可用版本"} onClick={() => navigate("../tokenizers")} />
            <Snapshot icon={<Activity size={18} />} title="最近训练运行" value={current.training_run?.current_stage || "当前无运行"} detail={current.training_run ? `${formatNumber(current.training_run.current_tokens)} tokens` : "打开训练中心创建计划"} onClick={() => navigate("../training")} />
            <Snapshot icon={<PackageCheck size={18} />} title="项目候选模型" value={current.candidate_model?.name || "暂无候选"} detail={current.candidate_model ? `v${current.candidate_model.version}` : "训练完成后产生"} onClick={() => navigate("../models")} />
          </div>
        </Panel>
        <Panel title="训练 Loss 曲线" description={current.training_run?.latest_metrics?.history_sampled ? `最近训练运行 · 全程 ${formatNumber(current.training_run.latest_metrics.history_total_points)} 条记录，均匀采样 ${chartData.length} 点；未做平滑，可能省略中间尖峰。` : "最近训练运行 · 横轴为训练步数，纵轴为训练 Loss。"}>
          {chartData.length ? <div id="training-loss" className="chart-box">
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={chartData} margin={{ top: 10, right: 8, left: -24, bottom: 0 }}>
                <XAxis dataKey="step" axisLine={false} tickLine={false} tick={{ fill: "#7890ad", fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#7890ad", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#111d2d", border: "1px solid #243752", borderRadius: 10 }} />
                <Line type="linear" dataKey="loss" name="训练 Loss" stroke="#55d7c4" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div> : <div className="quiet-message">真实训练引擎上报指标后显示趋势。</div>}
        </Panel>
      </div>
      <div className="content-grid overview-grid lower">
        <Panel title="关键异常与下一步" actions={<ShieldAlert size={18} className="panel-icon" />}>
          <div className="alert-stack">
            {(data?.alerts || []).map((item: any, index: number) => <button className="alert-link" key={index} onClick={() => navigate(`../${item.target}`)}><Alert severity={item.severity} title={item.title} /><ArrowRight size={16} /></button>)}
            {!data?.alerts?.length && <Alert severity="success" title="当前没有阻断项" detail="所有已配置生命周期对象状态正常。" />}
          </div>
        </Panel>
        <Panel title="活动任务" description="日志和资源监控位于具体任务详情。">
          <div className="compact-task-list">
            {(data?.active_tasks || []).map((task: any) => <div key={task.id}><div><strong>{task.title}</strong><StatusBadge status={task.observed_state} /></div><span>{task.current_phase || task.kind}</span><Progress value={task.progress} /></div>)}
            {!data?.active_tasks?.length && <div className="quiet-message">没有正在运行的任务。</div>}
          </div>
        </Panel>
      </div>
      {editAddressOpen && (
        <Modal title="配置项目地址" description="登记控制面独占工作目录；Worker 配置本地执行根后不需要共享这个绝对路径。已有运行或制品后，改址必须使用停服迁移流程。" onClose={() => setEditAddressOpen(false)}>
          <form onSubmit={saveAddress}>
            <Field label="项目地址" hint="必须是控制面可访问的本地目录；旧共享模式 Worker 也需要访问">
              <input name="workspace_uri" defaultValue={data?.project?.workspace_uri || ""} placeholder="F:\projects\example-lm 或 file:///workspace/example-lm" autoFocus required />
            </Field>
            <ErrorMessage error={updateAddress.error} />
            <FormActions submitting={updateAddress.isPending} onCancel={() => setEditAddressOpen(false)} />
          </form>
        </Modal>
      )}
      {importCodeOpen && (
        <Modal title={reuploadSource ? "上传代码新版本" : "导入模型代码"} description="每次导入都会生成不可变代码版本，不覆盖历史。" onClose={() => { setImportCodeOpen(false); setReuploadSource(null); }}>
          <form onSubmit={submitCode}>
            <div className="form-grid two-columns">
              <Field label="代码源名称"><input name="name" required defaultValue={reuploadSource?.name || ""} readOnly={Boolean(reuploadSource)} placeholder="例如 ExampleLM 主仓库" autoFocus /></Field>
              <Field label="来源类型"><select value={codeSourceType} disabled={Boolean(reuploadSource)} onChange={(event) => setCodeSourceType(event.target.value)}><option value="git">Git 仓库</option><option value="local">Worker 本地目录</option><option value="archive">本地/HTTP 归档</option><option value="upload">上传归档</option></select></Field>
              {codeSourceType === "upload" ? <Field label="代码归档" hint="支持 zip、tar、tar.gz"><input name="file" type="file" accept=".zip,.tar,.gz,.tgz" required /></Field> : <Field label="来源地址" hint={codeSourceType === "git" ? "Git URL 或 Worker 可访问仓库路径" : "Worker 可访问路径、file:// URI 或 HTTP(S) 归档"}><input name="source_uri" required placeholder={codeSourceType === "git" ? "https://github.com/org/model.git" : "F:\\models\\example-lm"} /></Field>}
              {codeSourceType === "git" && <Field label="Revision" hint="分支、Tag 或 Commit；留空使用仓库默认分支"><input name="revision" placeholder="main 或 commit SHA" /></Field>}
              {codeSourceType !== "upload" && <Field label="代码子目录" hint="Monorepo 可指定仓库内相对目录"><input name="subdirectory" placeholder="例如 models/example-lm" /></Field>}
              <Field label="训练入口" hint="相对代码根目录；导入时验证存在"><input name="training_entrypoint" defaultValue={reuploadSource?.current_version?.entrypoints?.training || ""} placeholder="train.py" /></Field>
              <Field label="模型工厂入口" hint="自定义模型使用；文件需导出 build_model(config, initialization)"><input name="model_factory_entrypoint" defaultValue={reuploadSource?.current_version?.entrypoints?.model_factory || ""} placeholder="model_factory.py" /></Field>
              <Field label="依赖声明" hint="只记录，不自动安装"><input name="dependency_spec" defaultValue={reuploadSource?.current_version?.dependency_spec || ""} placeholder="requirements.txt 或 uv.lock" /></Field>
              <Field label="模型配置 Schema" hint="JSON Schema properties 会自动生成模型规格表单"><textarea name="config_schema_json" defaultValue={JSON.stringify(reuploadSource?.current_version?.config_schema || {}, null, 2)} placeholder={'{"type":"object","properties":{}}'} /></Field>
            </div>
            {codeFormError && <Alert severity="error" title={codeFormError} />}
            <ErrorMessage error={importCode.error} />
            <FormActions submitting={importCode.isPending} onCancel={() => { setImportCodeOpen(false); setReuploadSource(null); }} submitLabel={reuploadSource ? "上传并创建新版本" : "导入并创建版本"} />
          </form>
        </Modal>
      )}
    </>
  );
}

function Snapshot({ icon, title, value, detail, onClick }: { icon: React.ReactNode; title: string; value: string; detail: string; onClick: () => void }) {
  return <button className="snapshot-row" onClick={onClick}><div className="snapshot-icon">{icon}</div><div><span>{title}</span><strong>{value}</strong><small>{detail}</small></div><ArrowRight size={17} /></button>;
}
