$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Run scripts/setup.ps1 first."
}

$ApiHost = if ($env:LLM_PLATFORM_API_HOST) { $env:LLM_PLATFORM_API_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:LLM_PLATFORM_API_PORT) { $env:LLM_PLATFORM_API_PORT } else { "8000" }
$WebHost = if ($env:LLM_PLATFORM_WEB_HOST) { $env:LLM_PLATFORM_WEB_HOST } else { "127.0.0.1" }
if (-not $env:LLM_PLATFORM_WORKER_RELEASE_ROOT) {
    $env:LLM_PLATFORM_WORKER_RELEASE_ROOT = Join-Path $Root ".llm-platform\releases"
}

$Api = Start-Process `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "llm_platform.main" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru
$Worker = Start-Process `
    -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "llm_platform_worker.main" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru

try {
    Write-Host "ForgeLM web development server is starting on ${WebHost}:5173." -ForegroundColor Cyan
    pnpm run dev:web -- --host $WebHost
}
finally {
    foreach ($Process in @($Api, $Worker)) {
        if ($Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id
        }
    }
}
