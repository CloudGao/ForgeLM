#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$#" -gt 0 ]]; then
  echo "Model runtimes are installed through external plugin suites." >&2
  exit 2
fi

python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install --disable-pip-version-check -c constraints.txt -e '.[dev]'



echo "ForgeLM Linux environment is ready."
