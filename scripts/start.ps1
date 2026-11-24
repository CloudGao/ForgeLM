$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Run scripts/setup.ps1 first."
}

if (-not (Test-Path "apps\web\dist\index.html")) {
    pnpm run build:web
}

$ApiHost = if ($env:LLM_PLATFORM_API_HOST) { $env:LLM_PLATFORM_API_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:LLM_PLATFORM_API_PORT) { $env:LLM_PLATFORM_API_PORT } else { "8000" }
if (-not $env:LLM_PLATFORM_STORAGE_ROOT) {
    $env:LLM_PLATFORM_STORAGE_ROOT = & ".venv\Scripts\python.exe" -c "from llm_platform.config import get_settings; print(get_settings().storage_root)"
}
if (-not $env:LLM_PLATFORM_WORKER_RELEASE_ROOT) {
    $env:LLM_PLATFORM_WORKER_RELEASE_ROOT = if ($env:LLM_PLATFORM_STORAGE_ROOT) { Join-Path $env:LLM_PLATFORM_STORAGE_ROOT "platform\releases" } else { Join-Path $Root ".llm-platform\releases" }
}

$Worker = Start-Process `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "llm_platform_worker.main" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

try {
    Write-Host "ForgeLM API is listening on ${ApiHost}:${ApiPort}" -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m llm_platform.supervisor
}
finally {
    if ($Worker -and -not $Worker.HasExited) {
        Stop-Process -Id $Worker.Id
    }
}
