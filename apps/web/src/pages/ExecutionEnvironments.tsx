import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Cpu, Network, Play, Plus, RefreshCw, ServerCog, ShieldCheck, Terminal } from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, post } from "../api";
import { ComputeNavigation } from "../components/ComputeNavigation";
import { Alert, ErrorMessage, Field, FormActions, Modal, PageHeader, Panel, StatusBadge } from "../components/ui";

type LocalEnvironment = {
  id: string;
  kind: "windows" | "linux" | "wsl";
  name: string;
  state: string;
  injection_supported: boolean;
  start_supported: boolean;
  description: string;
  worker?: any;
  workers?: any[];
  supervision?: {
    managed: boolean;
    state: string;
    pid?: number | null;
    restart_count?: number;
    last_exit_code?: number | null;
    last_error?: string | null;
  } | null;
};

type InjectionOperation = {
  id: string;
  environment_id: string;
  profile: string;
  workspace_root: string;
  status: string;
  exit_code?: number | null;
  log_tail: string;
};

type RemoteOperation = {
  id: string;
  kind: "probe" | "provision" | "upgrade" | "repair";
  status: string;
  phase: string;
  logs: string[];
  error?: { message?: string };
};

type RemoteEnvironment = {
  id: string;
  name: string;
  ssh_host: string;
  ssh_port: number;
  ssh_user: string;
  credential_ref: string;
  credential_configured: boolean;
  host_key_fingerprint: string;
  control_plane_url: string;
  workspace_root: string;
  install_root: string;
  profile: "general" | "data";
  labels: Record<string, string>;
  status: string;
  desired_release: string;
  deployed_release?: string | null;
  worker_id?: string | null;
  last_error?: { message?: string };
  latest_operation?: RemoteOperation | null;
  transport?: "direct" | "ssh_tunnel";
  runtime?: string;
  download_policy?: "official" | "domestic";
  storage_layout?: {
    system_disk?: { mount: string; fstype: string } | null;
    data_disk?: { mount: string; fstype: string } | null;
    runtime_root: string;
    cache_root: string;
  } | null;
  tunnel?: { state: string; reconnect_count: number; last_error?: string | null } | null;
};

export function ExecutionEnvironments() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const defaultControlPlaneUrl = window.location.port === "5173"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;
  const [injectTarget, setInjectTarget] = useState<LocalEnvironment | null>(null);
  const [showRemoteForm, setShowRemoteForm] = useState(false);
  const [remoteFormError, setRemoteFormError] = useState("");
  const [credentialMode, setCredentialMode] = useState("password");
  const [remoteTransport, setRemoteTransport] = useState("ssh_tunnel");
  const environments = useQuery({ queryKey: ["execution-environments", "local"], queryFn: () => api<LocalEnvironment[]>("/execution-environments/local"), refetchInterval: 5000 });
  const operations = useQuery({ queryKey: ["execution-environments", "operations"], queryFn: () => api<InjectionOperation[]>("/execution-environments/local/operations"), refetchInterval: 2000 });
  const remoteEnvironments = useQuery({ queryKey: ["execution-environments", "remote"], queryFn: () => api<RemoteEnvironment[]>("/execution-environments/remote"), refetchInterval: 3000 });
  const localWorkers = useMemo(
    () => (environments.data || []).flatMap((environment) => environment.workers ?? (environment.worker ? [environment.worker] : [])),
    [environments.data],
  );
  const inject = useMutation({
    mutationFn: (payload: { environment_id: string; profile: string; workspace_root: string }) => post<InjectionOperation>("/execution-environments/local/inject", payload),
    onSuccess: () => {
      setInjectTarget(null);
      queryClient.invalidateQueries({ queryKey: ["execution-environments"] });
    },
  });
  const startWorker = useMutation({
    mutationFn: (environmentId: string) => post("/execution-environments/local/start", { environment_id: environmentId }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["execution-environments"] }),
  });
  const createRemote = useMutation({
    gcTime: 0,
    mutationFn: (payload: Record<string, unknown>) => post<RemoteEnvironment>("/execution-environments/remote", payload),
    onSuccess: () => {
      setShowRemoteForm(false);
      setRemoteFormError("");
      queryClient.invalidateQueries({ queryKey: ["execution-environments", "remote"] });
    },
  });
  const runRemoteOperation = useMutation({
    mutationFn: ({ environmentId, kind }: { environmentId: string; kind: RemoteOperation["kind"] }) => post<RemoteOperation>(`/execution-environments/remote/${environmentId}/operations`, { kind }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["execution-environments", "remote"] }),
  });
  const saveDownloadPolicy = useMutation({
    mutationFn: ({ id, policy }: { id: string; policy: string }) => api(`/execution-environments/remote/${id}/download-policy`, {
      method: "PUT", body: JSON.stringify({ download_policy: policy }),
    }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["execution-environments", "remote"] }),
  });

  function submitInjection(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!injectTarget) return;
    const form = new FormData(event.currentTarget);
    inject.mutate({
      environment_id: injectTarget.id,
      profile: String(form.get("profile") || "base"),
      workspace_root: String(form.get("workspace_root") || "/data/forgelm/workspaces"),
    });
  }

  function submitRemoteEnvironment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    try {
      const labels = JSON.parse(String(form.get("labels") || "{}"));
      if (!labels || Array.isArray(labels) || typeof labels !== "object") throw new Error("标签必须是 JSON 对象");
      setRemoteFormError("");
      createRemote.mutate({
        name: String(form.get("name") || ""),
        ssh_host: String(form.get("ssh_host") || ""),
        ssh_port: Number(form.get("ssh_port") || 22),
        ssh_user: String(form.get("ssh_user") || "root"),
        ssh_command: String(form.get("ssh_command") || "") || null,
        ...(credentialMode === "password" ? { ssh_password: String(form.get("ssh_password") || "") }
          : credentialMode === "path" ? { private_key_path: String(form.get("private_key_path") || "") }
          : { credential_ref: String(form.get("credential_ref") || "") }),
        transport: remoteTransport,
        runtime: String(form.get("runtime") || "base"),
        service_mode: String(form.get("service_mode") || "auto"),
        expected_host_key_fingerprint: String(form.get("expected_host_key_fingerprint") || "") || null,
        control_plane_url: String(form.get("control_plane_url") || defaultControlPlaneUrl),
        workspace_root: String(form.get("workspace_root") || ""),
        install_root: String(form.get("install_root") || ""),
        profile: form.get("runtime") === "data" ? "data" : "general",
        desired_release: String(form.get("desired_release") || "source"),
        labels,
      });
    } catch (error) {
      setRemoteFormError(error instanceof Error ? error.message : "标签 JSON 不合法");
    }
  }

  return (
    <div className="execution-page">
      <PageHeader
        eyebrow="EXECUTION ENVIRONMENTS"
        title="执行环境"
        description="本地环境可直接启动；远端通过 SSH 自动构建运行环境，并由平台维持连接。"
        actions={<button className="button secondary" onClick={() => navigate(-1)}><ArrowLeft size={16} />返回</button>}
      />
      <ComputeNavigation />
      <section className="execution-summary">
        <div><ServerCog size={18} /><span>在线 Worker</span><strong>{localWorkers.filter((worker) => worker.status === "online").length + (remoteEnvironments.data || []).filter((environment) => environment.status === "online").length}</strong></div>
        <div><Cpu size={18} /><span>本地环境</span><strong>{environments.data?.length || 0}</strong></div>
        <div><Network size={18} /><span>SSH 环境</span><strong>{remoteEnvironments.data?.length || 0}</strong></div>
      </section>

      <Panel title="本地环境" description="只显示控制面主机能够安全管理的 Windows 与 WSL 环境。">
        <div className="environment-grid">
          {(environments.data || []).map((environment) => {
            const operation = (operations.data || []).find((item) => item.environment_id === environment.id && item.status === "running")
              || (operations.data || []).filter((item) => item.environment_id === environment.id).at(-1);
            const environmentWorkers = environment.workers ?? (environment.worker ? [environment.worker] : []);
            const activeWorker = environmentWorkers.find((worker) => worker.status === "online") || environmentWorkers[0];
            const workerOnline = environmentWorkers.some((worker) => worker.status === "online");
            const nativeStarting = environment.start_supported && !workerOnline && (
              (startWorker.isPending && startWorker.variables === environment.id)
              || (environment.supervision?.managed && environment.supervision.state === "running")
            );
            return <article className="environment-card" key={environment.id}>
              <header><div className="environment-icon">{environment.kind === "wsl" ? <Terminal size={20} /> : <Cpu size={20} />}</div><StatusBadge status={operation?.status === "running" ? "injecting" : nativeStarting ? "starting" : activeWorker?.status || "not_started"} /></header>
              <div><span>{environment.kind.toUpperCase()}</span><h3>{environment.name}</h3><p>{environment.description}</p></div>
              <dl>
                <div><dt>{environment.kind === "wsl" ? "WSL 状态" : "环境状态"}</dt><dd>{environment.state}</dd></div>
                {environment.kind === "wsl" && <div><dt>生命周期监督</dt><dd>{environment.supervision?.managed ? environment.supervision.state : "未托管"}</dd></div>}
                {environment.start_supported && <div><dt>进程托管</dt><dd>{workerOnline && !environment.supervision?.managed ? "外部启动" : environment.supervision?.state || "stopped"}</dd></div>}
                <div><dt>Worker</dt><dd>{operation?.status === "running" ? "正在安装并注册…" : nativeStarting ? "正在启动并等待心跳…" : environmentWorkers.map((worker) => worker.name).join(", ") || (environment.kind === "wsl" ? "尚未注入" : "尚未启动")}</dd></div>
                <div><dt>最近心跳</dt><dd>{activeWorker?.last_heartbeat_at ? new Date(activeWorker.last_heartbeat_at).toLocaleString() : "—"}</dd></div>
                <div><dt>工作区</dt><dd>{activeWorker?.resources?.workspace?.root || operation?.workspace_root || "—"}</dd></div>
              </dl>
              {environment.supervision?.last_error && <Alert severity="error" title={environment.kind === "wsl" ? "WSL 生命周期监督异常" : "Windows Worker 启动异常"} detail={environment.supervision.last_error} />}
              {operation?.log_tail && <details><summary>最近安装日志</summary><pre>{operation.log_tail}</pre></details>}
              <footer>{environment.injection_supported
                ? <button className="button primary small" disabled={operation?.status === "running"} onClick={() => setInjectTarget(environment)}><Plus size={14} />{environment.worker ? "重新注入" : "注入 Worker"}</button>
                : environment.start_supported
                  ? <button className="button primary small" disabled={workerOnline || nativeStarting} onClick={() => startWorker.mutate(environment.id)}><Play size={14} />{workerOnline ? "Worker 已运行" : nativeStarting ? "正在启动…" : "启动 Worker"}</button>
                  : <span>该环境不支持页面启动</span>}</footer>
            </article>;
          })}
        </div>
        <ErrorMessage error={environments.error || operations.error || startWorker.error} />
      </Panel>

      <Panel title="SSH 远端环境" description="支持 AutoDL 容器与 Linux 服务器。默认通过 SSH 隧道通信，无需公开本机控制面。" className="remote-connect-panel">
        <div className="panel-toolbar">
          <Alert severity="info" title="固定主机身份与单次注册" detail="创建环境时会获取并固定 SSH Host Key。密码仅用于首次登录并安装专用公钥；平台保存受保护的私钥用于后续重连，Worker 使用节点专属凭证。" />
          <button className="button primary" onClick={() => setShowRemoteForm(true)}><Plus size={15} />添加 SSH Worker</button>
        </div>
        <div className="environment-grid remote-environment-grid">
          {(remoteEnvironments.data || []).map((environment) => {
            const operation = environment.latest_operation;
            const running = operation?.status === "queued" || operation?.status === "running";
            return <article className="environment-card" key={environment.id}>
              <header><div className="environment-icon"><ShieldCheck size={20} /></div><StatusBadge status={running ? "provisioning" : environment.status} /></header>
              <div><span>SSH · {environment.profile.toUpperCase()}</span><h3>{environment.name}</h3><p>{environment.ssh_user}@{environment.ssh_host}:{environment.ssh_port}</p></div>
              <dl>
                <div><dt>Host Key</dt><dd title={environment.host_key_fingerprint}>{environment.host_key_fingerprint}</dd></div>
                <div><dt>私钥引用</dt><dd>{environment.credential_ref} · {environment.credential_configured ? "已配置" : "未配置"}</dd></div>
                <div><dt>Worker</dt><dd>{environment.worker_id ? environment.status : running ? "等待安装/注册" : "尚未注册"}</dd></div>
                <div><dt>Release</dt><dd>{environment.deployed_release || "尚未部署"}</dd></div>
                <div><dt>工作区</dt><dd>{environment.workspace_root}</dd></div>
                {environment.storage_layout && <>
                  <div><dt>系统盘</dt><dd>{environment.storage_layout.system_disk?.mount || "未识别"}</dd></div>
                  <div><dt>数据盘</dt><dd>{environment.storage_layout.data_disk?.mount || "使用配置的工作区所在磁盘"}</dd></div>
                  <div><dt>运行依赖目录</dt><dd>{environment.storage_layout.runtime_root}</dd></div>
                  <div><dt>缓存目录</dt><dd>{environment.storage_layout.cache_root}</dd></div>
                </>}
                <div><dt>当前阶段</dt><dd>{operation?.phase || "—"}</dd></div>
                <div><dt>运行环境</dt><dd>{environment.runtime || "base"}</dd></div>
                <div><dt>连接</dt><dd>{environment.transport === "ssh_tunnel" ? `SSH 隧道 · ${environment.tunnel?.state || "stopped"}` : "HTTP 直连"}</dd></div>
                {environment.tunnel && <div><dt>重连次数</dt><dd>{environment.tunnel.reconnect_count}</dd></div>}
              </dl>
              <Field label="下载源（下次发布升级生效）"><select aria-label={`${environment.name} 下载源`} value={environment.download_policy || "official"} disabled={running || saveDownloadPolicy.isPending} onChange={(event) => saveDownloadPolicy.mutate({ id: environment.id, policy: event.target.value })}><option value="domestic">国内优先 · HF-Mirror / 清华 PyPI</option><option value="official">官方源</option></select></Field>
              {(environment.last_error?.message || operation?.error?.message) && <Alert severity="error" title="远程操作失败" detail={environment.last_error?.message || operation?.error?.message || "未知错误"} />}
              {environment.tunnel?.last_error && <Alert severity="warning" title="SSH 连接正在恢复" detail={environment.tunnel.last_error} />}
              {!!operation?.logs?.length && <details><summary>最近操作日志</summary><pre>{operation.logs.slice(-12).join("\n")}</pre></details>}
              <footer>
                <button className="button secondary small" disabled={running} onClick={() => runRemoteOperation.mutate({ environmentId: environment.id, kind: "probe" })}><RefreshCw size={14} />探测</button>
                {environment.deployed_release && <button className="button secondary small" disabled={running || !environment.credential_configured} onClick={() => runRemoteOperation.mutate({ environmentId: environment.id, kind: "repair" })}><ServerCog size={14} />修复</button>}
                <button className="button primary small" disabled={running || !environment.credential_configured} onClick={() => runRemoteOperation.mutate({ environmentId: environment.id, kind: environment.deployed_release ? "upgrade" : "provision" })}><Play size={14} />{environment.deployed_release ? "发布升级" : "初始化并部署"}</button>
              </footer>
            </article>;
          })}
          {remoteEnvironments.data?.length === 0 && <div className="empty-state"><ServerCog size={24} /><strong>还没有 SSH 远端环境</strong><span>添加主机并固定 Host Key 后，平台会立即排队初始化与部署。</span></div>}
        </div>
        <ErrorMessage error={remoteEnvironments.error || runRemoteOperation.error || saveDownloadPolicy.error} />
      </Panel>

      {injectTarget && <Modal title={`向 ${injectTarget.name} 注入 Worker`} description="ForgeLM 会在 WSL ext4 中建立独立环境、安装 Worker、注册 systemd 服务并连接当前控制面。" onClose={() => setInjectTarget(null)}>
        <form onSubmit={submitInjection} className="form-grid">
          <Field label="运行配置"><select name="profile" defaultValue="base"><option value="base">基础 Worker</option></select></Field>
          <Field label="WSL 工作区根" hint="模型数据和运行目录放在 WSL ext4，不放 /mnt/c 或 /mnt/f"><input name="workspace_root" defaultValue="/data/forgelm/workspaces" required /></Field>
          <Alert severity="warning" title="注入完成后由控制面托管" detail="控制面会监督 WSL 生命周期并维持 Worker 连接；此操作本身不会创建训练计划或训练运行。" />
          <ErrorMessage error={inject.error} />
          <FormActions submitting={inject.isPending} onCancel={() => setInjectTarget(null)} submitLabel="开始注入" />
        </form>
      </Modal>}
      {showRemoteForm && <Modal title="添加 SSH 远端 Worker" description="粘贴登录指令并选择运行环境，平台会自动安装、验证并建立连接。" onClose={() => setShowRemoteForm(false)}>
        <form onSubmit={submitRemoteEnvironment} className="form-grid">
          <Field label="环境名称"><input name="name" placeholder="autodl-gpu-01" required /></Field>
          <Field label="SSH 登录指令" hint="可直接粘贴 AutoDL 提供的 ssh -p 端口 root@主机；也可在高级设置中分开填写"><input name="ssh_command" placeholder="ssh -p 12345 root@connect.example.com" /></Field>
          <div className="form-grid two-columns">
            <Field label="登录方式"><select value={credentialMode} onChange={(event) => setCredentialMode(event.target.value)}><option value="password">首次登录密码</option><option value="path">本机私钥文件</option><option value="reference">私钥环境变量</option></select></Field>
            {credentialMode === "password" && <Field label="SSH 密码" hint="仅用于首次连接并配置专用密钥，不保存密码"><input name="ssh_password" type="password" autoComplete="new-password" required /></Field>}
            {credentialMode === "path" && <Field label="本机私钥路径" hint="控制面所在电脑上的无口令私钥文件"><input name="private_key_path" placeholder="C:\Users\用户名\.ssh\id_ed25519" required /></Field>}
            {credentialMode === "reference" && <Field label="私钥环境变量" hint="变量值为控制面电脑上的私钥路径"><input name="credential_ref" placeholder="FORGELM_SSH_KEY" required /></Field>}
            <Field label="运行环境"><select name="runtime" defaultValue="base"><option value="base">基础 Worker</option><option value="data">数据处理 / Data-Juicer</option><option value="inference">Hugging Face GPU 推理</option></select></Field>
            <Field label="连接方式"><select value={remoteTransport} onChange={(event) => setRemoteTransport(event.target.value)}><option value="ssh_tunnel">SSH 隧道（推荐）</option><option value="direct">HTTP 直连</option></select></Field>
          </div>
          {remoteTransport === "direct" && <Field label="控制面公开地址" hint="必须能从远端主机访问"><input name="control_plane_url" defaultValue={defaultControlPlaneUrl} required /></Field>}
          <details><summary>高级设置：目录、主机与服务</summary><div className="form-grid two-columns">
            <Field label="SSH 主机"><input name="ssh_host" placeholder="未粘贴指令时填写" /></Field>
            <Field label="SSH 端口"><input name="ssh_port" type="number" min="1" max="65535" defaultValue="22" /></Field>
            <Field label="SSH 用户"><input name="ssh_user" defaultValue="root" /></Field>
            <Field label="进程管理"><select name="service_mode" defaultValue="auto"><option value="auto">自动检测</option><option value="process">容器 / 独立守护进程</option><option value="systemd">systemd 服务</option></select></Field>
            <Field label="期望 Host Key 指纹" hint="可留空以首次获取并固定"><input name="expected_host_key_fingerprint" placeholder="SHA256:…" /></Field>
            <Field label="期望 Release"><input name="desired_release" defaultValue="source" required /></Field>
            <Field label="远端安装根"><input name="install_root" defaultValue="/opt/forgelm-worker" required /></Field>
            <Field label="远端工作区" hint="自动识别 AutoDL 独立数据盘，默认把工作区、Checkpoint、运行依赖和缓存放在数据盘；其他主机使用此路径。"><input name="workspace_root" defaultValue="/var/lib/forgelm/workspaces" required /></Field>
            <Field label="路由标签（JSON）"><input name="labels" defaultValue='{"location":"remote"}' required /></Field>
          </div></details>
          <Alert severity="info" title="自动构建与连接" detail="支持 root 容器；自动准备 Python 3.12 和所选依赖，GPU 环境会检查驱动与运行能力。远端需能下载安装包并允许 SSH 反向转发。控制面需保持运行和联网，长时间断线仍按任务恢复规则处理。" />
          {remoteFormError && <Alert severity="error" title="表单不合法" detail={remoteFormError} />}
          <ErrorMessage error={createRemote.error} />
          <FormActions submitting={createRemote.isPending} onCancel={() => setShowRemoteForm(false)} submitLabel="固定 Host Key 并部署" />
        </form>
      </Modal>}
    </div>
  );
}
