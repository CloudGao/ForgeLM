#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv-linux/bin/python ]]; then
  echo "Run: bash scripts/setup-linux.sh" >&2
  exit 1
fi

: "${LLM_PLATFORM_API_URL:=http://127.0.0.1:8000}"
: "${LLM_PLATFORM_AGENT_TOKEN:=local-development-token}"
: "${LLM_PLATFORM_WORKER_WORKSPACE_ROOT:?Set LLM_PLATFORM_WORKER_WORKSPACE_ROOT to a Worker-local directory}"
export LLM_PLATFORM_API_URL
export LLM_PLATFORM_AGENT_TOKEN
export LLM_PLATFORM_WORKER_WORKSPACE_ROOT
export LLM_PLATFORM_WORKER_RELEASE_ROOT="${LLM_PLATFORM_WORKER_RELEASE_ROOT:-$LLM_PLATFORM_WORKER_WORKSPACE_ROOT/../releases}"

exec .venv-linux/bin/python -m llm_platform_worker.main "$@"
