#!/usr/bin/env bash
set -euo pipefail

wheel=""
environment_file=""
install_root=""
workspace_root=""
release_id=""
service_user=""
service_name="forgelm-worker"
runtime="base"
service_mode="auto"
supervisor=""
requirements_dir=""
phase="all"
download_policy="official"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel) wheel="$2"; shift 2 ;;
    --environment-file) environment_file="$2"; shift 2 ;;
    --install-root) install_root="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --release-id) release_id="$2"; shift 2 ;;
    --service-user) service_user="$2"; shift 2 ;;
    --service-name) service_name="$2"; shift 2 ;;
    --phase) phase="$2"; shift 2 ;;
    --download-policy) download_policy="$2"; shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    --service-mode) service_mode="$2"; shift 2 ;;
    --supervisor) supervisor="$2"; shift 2 ;;
    --requirements-dir) requirements_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$wheel" || -z "$environment_file" || -z "$install_root" || \
      -z "$workspace_root" || -z "$release_id" || -z "$service_user" ]]; then
  echo "missing required bootstrap argument" >&2
  exit 2
fi

if [[ "$install_root" != /* || "$workspace_root" != /* || \
      "$install_root" == "/" || "$workspace_root" == "/" || \
      "$install_root" == *"/../"* || "$install_root" == */.. || \
      "$workspace_root" == *"/../"* || "$workspace_root" == */.. || \
      ! "$install_root" =~ ^(/[A-Za-z0-9._-]+)+$ || \
      ! "$workspace_root" =~ ^(/[A-Za-z0-9._-]+)+$ || \
      "${install_root#/}" != */* || "${workspace_root#/}" != */* || \
      "$install_root" == "$workspace_root" || \
      "$install_root" == "$workspace_root/"* || \
      "$workspace_root" == "$install_root/"* ]]; then
  echo "install and workspace roots must be disjoint safe absolute paths with at least two components" >&2
  exit 2
fi
if [[ ! "$release_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "release id contains unsupported characters" >&2
  exit 2
fi
if [[ "$service_name" != "forgelm-worker" ]]; then
  echo "unsupported service name" >&2
  exit 2
fi
if ! id "$service_user" >/dev/null 2>&1; then
  echo "service user does not exist: $service_user" >&2
  exit 2
fi
if [[ ! -f "$wheel" || ! -f "$environment_file" ]]; then
  echo "wheel or environment file is missing" >&2
  exit 2
fi

case "$phase" in all|prepare|activate) ;; *) exit 2 ;; esac
case "$download_policy" in official|domestic) ;; *) exit 2 ;; esac
case "$runtime" in base|data|inference) ;; *) echo "invalid runtime" >&2; exit 2 ;; esac
case "$service_mode" in auto|systemd|process) ;; *) echo "invalid service mode" >&2; exit 2 ;; esac
test -f "$supervisor"
test "$(uname -s)" = Linux
# Reject aliases to the filesystem root or overlapping directories before writing.
install_root="$(realpath -m "$install_root")"
workspace_root="$(realpath -m "$workspace_root")"
if [[ "$install_root" == / || "$workspace_root" == / || "$install_root" == "$workspace_root" || "$install_root" == "$workspace_root/"* || "$workspace_root" == "$install_root/"* ]]; then
  echo "resolved install/workspace paths are unsafe" >&2; exit 2
fi
if [[ ! "$install_root" =~ ^(/[A-Za-z0-9._-]+)+$ || ! "$workspace_root" =~ ^(/[A-Za-z0-9._-]+)+$ ]]; then
  echo "resolved paths contain unsupported characters" >&2; exit 2
fi

# Do not relocate a venv: console-script shebangs embed its absolute path.
service_group="$(id -gn "$service_user")"
bulk_root="$workspace_root/.runtime"
release_dir="$bulk_root/releases/$release_id"
state_root="$install_root/state"
mkdir -p "$workspace_root" "$bulk_root/releases" "$workspace_root/.tmp" "$workspace_root/.cache"
export TMPDIR="$workspace_root/.tmp"
export XDG_CACHE_HOME="$workspace_root/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
# Recover only incomplete installer-owned releases; validated rollback releases remain.
# The lock must precede cleanup so concurrent upgrades cannot remove one another.
exec 9>"$workspace_root/.runtime/install.lock"
flock -n 9 || { echo "another installation is running" >&2; exit 1; }
current_release="$(readlink -f "$install_root/current" || true)"
for failed in "$install_root"/releases/*; do
  if [[ ! -L "$install_root/releases" && -d "$failed" && ! -L "$failed" && ! -f "$failed/ready" && "$(realpath "$failed")" == "$install_root/releases/"* && "$(realpath "$failed")" != "$current_release" ]]; then
    echo "[storage] removing incomplete installation: $failed"
    rm -rf -- "$failed"
  fi
done
mkdir -p "$state_root"
chmod 700 "$state_root"
pip_install() {
  local interpreter="$1"
  shift
  if [[ "$download_policy" == domestic ]]; then
    echo "[download] PyPI: TUNA mirror first"
    if "$interpreter" -m pip install --disable-pip-version-check --timeout 30 --retries 2 \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "$@"; then
      return 0
    fi
    echo "[download] mirror unavailable or missing a pinned package; retrying official PyPI"
  fi
  "$interpreter" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple "$@"
}
ensure_uv() {
  if [[ ! -x "$bulk_root/tools/bin/uv" ]]; then
    pip_install python3 --target "$bulk_root/tools" 'uv==0.8.22'
  fi
}
if [[ "$phase" != activate ]]; then
echo "[environment] detecting Python and GPU"
df -Pk "$install_root" "$workspace_root"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; fi

python_bin=""
for candidate in python3.12 /root/miniconda3/bin/python /opt/conda/bin/python python3; do
  if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert sys.version_info[:2] == (3,12)' 2>/dev/null; then
    python_bin="$(command -v "$candidate")"; break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "[python] installing managed Python 3.12"
  command -v python3 >/dev/null || { echo "Python 3 with pip is required to bootstrap Python 3.12" >&2; exit 1; }
  ensure_uv
  export UV_PYTHON_INSTALL_DIR="$bulk_root/python"
  "$bulk_root/tools/bin/uv" python install 3.12
  python_bin="$("$bulk_root/tools/bin/uv" python find --managed-python 3.12)"
fi
if [[ ! -f "$release_dir/ready" ]]; then
  mkdir -p "$release_dir"
  if ! "$python_bin" -m venv "$release_dir/venv"; then
    echo "[python] system venv/ensurepip unavailable; preparing a seeded uv environment"
    ensure_uv
    "$bulk_root/tools/bin/uv" venv --allow-existing --seed --python "$python_bin" "$release_dir/venv"
  fi
  py="$release_dir/venv/bin/python"
  pip_install "$py" --upgrade pip
  echo "[dependencies] installing Worker runtime: $runtime"
  pip_install "$py" "$wheel"
  case "$runtime" in
    data) pip_install "$py" "${wheel}[data-juicer]" ;;
    inference) pip_install "$py" "${wheel}[inference-hf]" ;;
  esac
  echo "[validation] checking installed runtime"
  "$py" -m pip check
  "$py" -c 'import llm_platform_worker.main'
  if [[ "$runtime" == "data" ]]; then
    "$py" -c 'from llm_platform_worker.adapters.data_juicer import data_juicer_available; assert data_juicer_available(), "Data-Juicer runtime unavailable"'
  fi
  if [[ "$runtime" == "inference" ]]; then
    "$py" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check GPU driver compatibility"; print("CUDA", torch.version.cuda, torch.cuda.get_device_name())'
  fi

  touch "$release_dir/ready"
fi

fi
if [[ "$phase" == prepare ]]; then exit 0; fi
test -f "$release_dir/ready" || { echo "release has not passed validation" >&2; exit 1; }
echo "[activation] activating validated release"
install -m 600 "$environment_file" "$install_root/worker.env"
install -m 700 "$supervisor" "$install_root/supervisor.py"
ln -sfn "$release_dir" "$install_root/.current"
mv -Tf "$install_root/.current" "$install_root/current"
if [[ "$(id -u)" == 0 ]]; then
  chown -R "$service_user:$service_group" "$install_root"
  chown "$service_user:$service_group" "$workspace_root"
  chown -R "$service_user:$service_group" "$bulk_root" "$workspace_root/.cache" "$workspace_root/.tmp"
fi
if [[ "$service_mode" == auto ]]; then
  if [[ -d /run/systemd/system && "$(id -u)" == 0 ]]; then service_mode=systemd; else service_mode=process; fi
fi
if [[ "$service_mode" == systemd ]]; then
  test -d /run/systemd/system || { echo "systemd is not running; use process mode" >&2; exit 1; }
  cat >"/etc/systemd/system/$service_name.service" <<EOF
[Unit]
Description=ForgeLM Remote Worker
After=network-online.target
[Service]
Type=simple
User=$service_user
Group=$service_group
WorkingDirectory=$workspace_root
ExecStart=$install_root/current/venv/bin/python $install_root/supervisor.py $install_root
Restart=always
RestartSec=3
TimeoutStopSec=40
[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable "$service_name.service"
  systemctl restart "$service_name.service"
  cat >"$install_root/start-worker" <<EOF
#!/usr/bin/env bash
if [[ \$(id -u) == 0 ]]; then systemctl start $service_name.service; else sudo -n systemctl start $service_name.service; fi
EOF
else
  # A detached supervisor holds flock; repeated starts cannot create duplicate Agents.
  cat >"$install_root/start-worker" <<EOF
#!/usr/bin/env bash
nohup "$install_root/current/venv/bin/python" "$install_root/supervisor.py" "$install_root" </dev/null >>"$state_root/supervisor.log" 2>&1 9>&- &
EOF
  touch "$state_root/reload"
fi
chmod 700 "$install_root/start-worker"
if [[ "$(id -u)" == 0 ]]; then chown -R "$service_user:$service_group" "$install_root"; fi
if [[ "$service_mode" == process ]]; then
  if [[ "$(id -un)" == "$service_user" ]]; then
    bash "$install_root/start-worker"
  else
    runuser -u "$service_user" -- bash "$install_root/start-worker"
  fi
fi
echo "[started] Worker launched ($service_mode); waiting for Agent enrollment/heartbeat"
