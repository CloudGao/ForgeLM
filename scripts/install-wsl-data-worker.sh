#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_user="${FORGELM_DATA_WORKER_USER:-${FORGELM_WORKER_USER:-$(id -un)}}"
worker_group="$(id -gn "$worker_user")"
worker_home="$(getent passwd "$worker_user" | cut -d: -f6)"
worker_root="${FORGELM_DATA_WORKER_ROOT:-/data/forgelm/data-worker}"
workspace_root="${LLM_PLATFORM_WORKER_WORKSPACE_ROOT:-$worker_root/workspaces}"
venv_root="${FORGELM_DATA_WORKER_VENV:-$worker_root/venv}"
env_file="$worker_root/worker.env"
unit_name="forgelm-data-worker.service"
unit_template="$repo_root/scripts/forgelm-data-worker.service.template"
api_url="${LLM_PLATFORM_API_URL:-http://127.0.0.1:8000}"
agent_token="${LLM_PLATFORM_AGENT_TOKEN:-local-development-token}"
worker_name="${LLM_PLATFORM_WORKER_NAME:-data-wsl-$(hostname)}"
worker_cpu_limit="${LLM_PLATFORM_WORKER_CPU_LIMIT:-4}"
distro_name="${WSL_DISTRO_NAME:-unknown-wsl}"
worker_labels="${LLM_PLATFORM_WORKER_LABELS:-}"
if [[ -z "$worker_labels" ]]; then
  worker_labels="{\"location\":\"local\",\"os\":\"linux\",\"environment\":\"wsl\",\"workspace_mode\":\"worker-local\",\"distro\":\"$distro_name\",\"worker_role\":\"data\"}"
fi
start_service=false

usage() {
  echo "Usage: bash scripts/install-wsl-data-worker.sh [--start]"
  echo ""
  echo "Installs an isolated, CPU-limited, GPU-disabled Data Worker in WSL ext4."
  echo "Environment overrides: FORGELM_DATA_WORKER_ROOT, FORGELM_DATA_WORKER_VENV,"
  echo "LLM_PLATFORM_API_URL, LLM_PLATFORM_AGENT_TOKEN, LLM_PLATFORM_WORKER_NAME,"
  echo "LLM_PLATFORM_WORKER_WORKSPACE_ROOT, LLM_PLATFORM_WORKER_LABELS, and"
  echo "LLM_PLATFORM_WORKER_CPU_LIMIT."
  echo "Existing HTTP(S)/ALL/NO_PROXY variables are copied into the private env file."
}

for argument in "$@"; do
  case "$argument" in
    --start) start_service=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d /run/systemd/system ]]; then
  echo "systemd is not active. Enable it in /etc/wsl.conf first." >&2
  exit 1
fi
if [[ ! -f "$unit_template" ]]; then
  echo "Missing service template: $unit_template" >&2
  exit 1
fi
if [[ ! "$worker_cpu_limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "LLM_PLATFORM_WORKER_CPU_LIMIT must be a positive integer." >&2
  exit 1
fi
for value in "$worker_root" "$workspace_root" "$venv_root" "$api_url" "$agent_token" "$worker_name" "$worker_labels"; do
  if [[ "$value" == *"'"* || "$value" == *$'\n'* ]]; then
    echo "Worker configuration values must not contain single quotes or newlines." >&2
    exit 1
  fi
done
proxy_names=(http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY)
for proxy_name in "${proxy_names[@]}"; do
  proxy_value="${!proxy_name-}"
  if [[ "$proxy_value" == *"'"* || "$proxy_value" == *$'\n'* ]]; then
    echo "$proxy_name must not contain single quotes or newlines." >&2
    exit 1
  fi
done

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

run_as_worker() {
  if [[ "$(id -un)" == "$worker_user" ]]; then
    "$@"
  else
    runuser -u "$worker_user" -- env HOME="$worker_home" "$@"
  fi
}

run_as_root install -d -m 0755 -o "$worker_user" -g "$worker_group" "$worker_root"
run_as_root install -d -m 0755 -o "$worker_user" -g "$worker_group" "$workspace_root"

if [[ ! -x "$venv_root/bin/python" ]]; then
  if ! run_as_worker python3 -m venv "$venv_root"; then
    echo "Python venv support is missing. Run: sudo apt-get install python3-venv" >&2
    exit 1
  fi
fi

run_as_worker "$venv_root/bin/python" -m pip install --disable-pip-version-check \
  -e "$repo_root[data-juicer]"
run_as_worker "$venv_root/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3'
run_as_worker "$venv_root/bin/python" -m pip install --disable-pip-version-check \
  'transformers>=4.45,<6'

env_tmp="$(mktemp)"
unit_tmp="$(mktemp)"
cleanup() {
  rm -f "$env_tmp" "$unit_tmp"
}
trap cleanup EXIT

{
  printf "LLM_PLATFORM_API_URL='%s'\n" "$api_url"
  printf "LLM_PLATFORM_AGENT_TOKEN='%s'\n" "$agent_token"
  printf "LLM_PLATFORM_WORKER_NAME='%s'\n" "$worker_name"
  printf "LLM_PLATFORM_WORKER_WORKSPACE_ROOT='%s'\n" "$workspace_root"
  printf "LLM_PLATFORM_WORKER_LABELS='%s'\n" "$worker_labels"
  printf "LLM_PLATFORM_WORKER_PROFILE='data'\n"
  printf "LLM_PLATFORM_WORKER_DISABLE_GPU='1'\n"
  printf "LLM_PLATFORM_WORKER_CPU_LIMIT='%s'\n" "$worker_cpu_limit"
  for proxy_name in "${proxy_names[@]}"; do
    proxy_value="${!proxy_name-}"
    if [[ -n "$proxy_value" ]]; then
      printf "%s='%s'\n" "$proxy_name" "$proxy_value"
    fi
  done
} > "$env_tmp"
run_as_root install -m 0600 -o "$worker_user" -g "$worker_group" "$env_tmp" "$env_file"

sed \
  -e "s|@@WORKER_USER@@|$worker_user|g" \
  -e "s|@@WORKER_HOME@@|$worker_home|g" \
  -e "s|@@REPO_ROOT@@|$repo_root|g" \
  -e "s|@@ENV_FILE@@|$env_file|g" \
  -e "s|@@VENV_PYTHON@@|$venv_root/bin/python|g" \
  "$unit_template" > "$unit_tmp"
run_as_root install -m 0644 "$unit_tmp" "/etc/systemd/system/$unit_name"
run_as_root systemctl daemon-reload
run_as_root systemctl enable "$unit_name"
if [[ "$start_service" == true ]]; then
  run_as_root systemctl restart "$unit_name"
fi

echo "ForgeLM WSL Data Worker is installed and enabled."
echo "Workspace: $workspace_root"
echo "Profile:   data (GPU disabled, CPU limit $worker_cpu_limit)"
if [[ "$start_service" == true ]]; then
  echo "The Data Worker service is running."
else
  echo "Start once: sudo systemctl start $unit_name"
fi
echo "Status:    systemctl status $unit_name"
echo "Logs:      journalctl -u $unit_name -f"
echo "Stop:      sudo systemctl stop $unit_name"
