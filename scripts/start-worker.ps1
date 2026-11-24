$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Run scripts/setup.ps1 first."
}

if (-not $env:LLM_PLATFORM_STORAGE_ROOT) {
    $env:LLM_PLATFORM_STORAGE_ROOT = & ".venv\Scripts\python.exe" -c "from llm_platform.config import get_settings; print(get_settings().storage_root)"
}
if (-not $env:LLM_PLATFORM_WORKER_RELEASE_ROOT) {
    $env:LLM_PLATFORM_WORKER_RELEASE_ROOT = if ($env:LLM_PLATFORM_STORAGE_ROOT) { Join-Path $env:LLM_PLATFORM_STORAGE_ROOT "platform\releases" } else { Join-Path $Root ".llm-platform\releases" }
}

& ".venv\Scripts\python.exe" -m llm_platform_worker.main @args
