import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LoaderCircle, RotateCcw } from "lucide-react";
import { useState } from "react";
import { api, post } from "../api";
import { ErrorMessage, Modal } from "./ui";

export type ControlPlaneStatus = {
  managed: boolean;
  restart_supported: boolean;
  state: string;
  instance_id: string;
  pid: number;
  supervisor_pid?: number | null;
  restart_count?: number;
};

type ControlPlaneControlProps = {
  onRecovered?: () => void;
  recoveryDelay?: number;
  recoveryAttempts?: number;
};

const wait = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

export function ControlPlaneControl({
  onRecovered = () => window.location.reload(),
  recoveryDelay = 750,
  recoveryAttempts = 60,
}: ControlPlaneControlProps) {
  const queryClient = useQueryClient();
  const [confirmationOpen, setConfirmationOpen] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [restartError, setRestartError] = useState<unknown>(null);
  const status = useQuery({
    queryKey: ["control-plane"],
    queryFn: () => api<ControlPlaneStatus>("/control-plane"),
    retry: false,
    refetchInterval: restarting ? false : 10_000,
  });

  const managed = Boolean(status.data?.managed && status.data.restart_supported);
  const label = restarting
    ? "正在重启"
    : managed
      ? "控制面受管"
      : status.isLoading
        ? "检查控制面"
        : "控制面未受管";
  const unavailableReason = status.isError
    ? "无法读取本机控制面状态"
    : "当前进程未由 Supervisor 启动，请先用受管启动脚本重新启动一次";

  async function restart() {
    const previousInstanceId = status.data?.instance_id;
    if (!previousInstanceId || !managed) return;
    setRestarting(true);
    setRestartError(null);
    try {
      await post("/control-plane/restart", { reason: "用户从 ForgeLM 界面重启控制面" });
      for (let attempt = 0; attempt < recoveryAttempts; attempt += 1) {
        await wait(recoveryDelay);
        try {
          const recovered = await api<ControlPlaneStatus>("/control-plane");
          if (
            recovered.managed
            && recovered.restart_supported
            && recovered.state === "running"
            && recovered.instance_id !== previousInstanceId
          ) {
            queryClient.setQueryData(["control-plane"], recovered);
            onRecovered();
            return;
          }
        } catch {
          // The API is expected to be briefly unreachable while its child process restarts.
        }
      }
      throw new Error("控制面没有在预期时间内恢复，请检查启动终端中的 Supervisor 日志");
    } catch (error) {
      setRestartError(error);
      setRestarting(false);
    }
  }

  return (
    <div className={`control-plane-control ${managed ? "managed" : "unmanaged"}`}>
      <div className="local-chip" title={managed ? `API PID ${status.data?.pid}` : unavailableReason}>
        <span />{label}
      </div>
      <button
        type="button"
        className="icon-button control-plane-restart"
        aria-label="重启控制面"
        title={managed ? "重启控制面" : unavailableReason}
        disabled={!managed || restarting}
        onClick={() => setConfirmationOpen(true)}
      >
        {restarting ? <LoaderCircle className="spin-icon" size={16} /> : <RotateCcw size={16} />}
      </button>

      {confirmationOpen && (
        <Modal
          title="重启控制面？"
          description="API 会短暂离线，Supervisor 将拉起一个全新的受管实例。正在运行的独立 Worker 不会被终止。"
          onClose={() => { if (!restarting) setConfirmationOpen(false); }}
        >
          <div className="control-plane-dialog-body">
            <p>页面会等待新实例恢复，然后自动刷新。重启期间请不要重复提交操作。</p>
            <ErrorMessage error={restartError} />
            <div className="form-actions">
              <button type="button" className="button secondary" disabled={restarting} onClick={() => setConfirmationOpen(false)}>取消</button>
              <button type="button" className="button danger" disabled={restarting} onClick={restart}>
                {restarting ? <><LoaderCircle className="spin-icon" size={16} />正在重启…</> : <><RotateCcw size={16} />确认重启</>}
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
