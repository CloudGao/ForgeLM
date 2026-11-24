param(
    [switch]$Inference,
    [switch]$SkipDataTools
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed." }
}

$PythonExtras = if ($Inference) { ".[dev,inference-hf]" } else { ".[dev]" }
& ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -c constraints.txt -e $PythonExtras
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
pnpm install --frozen-lockfile
if ($LASTEXITCODE -ne 0) { throw "Web dependency installation failed." }
pnpm run build:web
if ($LASTEXITCODE -ne 0) { throw "Web build failed." }
& ".venv\Scripts\alembic.exe" upgrade head
if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }

if (-not $SkipDataTools) {
    $ToolsDir = Join-Path $Root ".llm-platform\tools"
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

    $GitBash = "C:\Program Files\Git\bin\bash.exe"
    if (-not (Test-Path -LiteralPath $GitBash)) {
        throw "Git Bash is required by the HF-Mirror hfd adapter: $GitBash"
    }

    $HfdPath = Join-Path $ToolsDir "hfd.sh"
    $HfdSha256 = "524894EF4F8E82DA6180156A96BD9678D56D77E877B7F69A9D253DA97D28866F"
    if (-not (Test-Path -LiteralPath $HfdPath)) {
        $HfdTemp = New-TemporaryFile
        Invoke-WebRequest -UseBasicParsing -Uri "https://hf-mirror.com/hfd/hfd.sh" -OutFile $HfdTemp
        if ((Get-FileHash -LiteralPath $HfdTemp -Algorithm SHA256).Hash -ne $HfdSha256) {
            Remove-Item -LiteralPath $HfdTemp -Force
            throw "hfd.sh checksum verification failed."
        }
        Move-Item -LiteralPath $HfdTemp -Destination $HfdPath
    }
    elseif ((Get-FileHash -LiteralPath $HfdPath -Algorithm SHA256).Hash -ne $HfdSha256) {
        throw "Existing hfd.sh does not match the pinned checksum: $HfdPath"
    }

    $AriaFolder = Join-Path $ToolsDir "aria2-1.37.0-win-64bit-build1"
    $AriaExe = Join-Path $AriaFolder "aria2c.exe"
    if (-not (Test-Path -LiteralPath $AriaExe)) {
        $AriaTemp = Join-Path ([System.IO.Path]::GetTempPath()) ("forgelm-aria2-" + [guid]::NewGuid().ToString("N") + ".zip")
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip" -OutFile $AriaTemp
        $AriaSha256 = "67D015301EEF0B612191212D564C5BB0A14B5B9C4796B76454276A4D28D9B288"
        if ((Get-FileHash -LiteralPath $AriaTemp -Algorithm SHA256).Hash -ne $AriaSha256) {
            Remove-Item -LiteralPath $AriaTemp -Force
            throw "aria2 checksum verification failed."
        }
        Expand-Archive -LiteralPath $AriaTemp -DestinationPath $ToolsDir
        Remove-Item -LiteralPath $AriaTemp -Force
    }
    if (-not (Test-Path -LiteralPath $AriaExe)) {
        throw "aria2 installation did not produce the expected executable: $AriaExe"
    }
}

Write-Host "ForgeLM setup complete." -ForegroundColor Green
if (-not $Inference) {
    Write-Host "Transformers inference is optional. Run .\scripts\setup.ps1 -Inference to install it." -ForegroundColor Yellow
}
