param(
    [string]$Distro = "MicroLM-Ubuntu",
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"

$environments = Invoke-RestMethod "$($ApiUrl.TrimEnd('/'))/api/execution-environments/local"
$environment = $environments | Where-Object { $_.id -eq "wsl:$Distro" }
if (-not $environment) {
    throw "The ForgeLM control plane cannot discover WSL distro: $Distro"
}
if (-not $environment.supervision.managed) {
    throw "The WSL Worker is not managed by ForgeLM. Inject it from the Execution Environments page."
}
if ($environment.supervision.state -ne "running") {
    throw "The ForgeLM WSL lifecycle supervisor is $($environment.supervision.state): $($environment.supervision.last_error)"
}

Write-Host "ForgeLM is supervising $Distro; no interactive keepalive terminal is required." -ForegroundColor Green
