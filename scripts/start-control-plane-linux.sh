#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv-linux/bin/python ]]; then
  echo "Run: bash scripts/setup-linux.sh" >&2
  exit 1
fi

.venv-linux/bin/alembic upgrade head
exec .venv-linux/bin/python -m llm_platform.supervisor
