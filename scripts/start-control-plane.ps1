$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Run scripts/setup.ps1 first."
}

if (-not (Test-Path "apps\web\dist\index.html")) {
    pnpm run build:web
}

& ".venv\Scripts\python.exe" -m llm_platform.supervisor
