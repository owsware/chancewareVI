# Chanceware diagnostics and metrics launcher

param(
    [Parameter(Mandatory)]
    [string]$GamePath,
    [string]$RuntimeBinary = "",
    [string]$LogLevel = "info,victory_gpu::host=info,victory_gpu::pipeline=info,victory_gpu::draw=info",
    [string]$PresentTarget = "",
    [switch]$D3d12Debug,
    [switch]$Dred
)

$GamePath = (Get-Item -LiteralPath $GamePath -ErrorAction Stop).FullName
$DiagBinary = Join-Path $PSScriptRoot "target\diag\chanceware.exe"
$ReleaseBinary = Join-Path $PSScriptRoot "target\release\chanceware.exe"

$BinaryPath = if ($RuntimeBinary) {
    (Get-Item -LiteralPath $RuntimeBinary -ErrorAction Stop).FullName
} elseif (Test-Path $DiagBinary) {
    $DiagBinary
} else {
    $ReleaseBinary
}

if (-not (Test-Path $BinaryPath)) {
    Write-Host "[INFO] Diagnostics binary not found. Building with --profile diag..." -ForegroundColor Cyan
    cargo +1.98.1 build --locked --manifest-path (Join-Path $PSScriptRoot "Cargo.toml") --profile diag -p victory --features diagnostics
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $BinaryPath = $DiagBinary
}

if ($PresentTarget) {
    $env:VICTORY_PRESENT_TARGET = $PresentTarget
} else {
    Remove-Item Env:VICTORY_PRESENT_TARGET -ErrorAction SilentlyContinue
}
Remove-Item Env:VICTORY_SKIP_PIPELINES -ErrorAction SilentlyContinue
if ($D3d12Debug) { $env:VICTORY_D3D12_DEBUG = "1" } else { Remove-Item Env:VICTORY_D3D12_DEBUG -ErrorAction SilentlyContinue }
if ($Dred) { $env:VICTORY_D3D12_DRED = "1" } else { Remove-Item Env:VICTORY_D3D12_DRED -ErrorAction SilentlyContinue }
$env:VICTORY_LOG = $LogLevel
$env:RUST_BACKTRACE = "1"

$LogDir = Join-Path $env:LOCALAPPDATA "victory\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_$Timestamp.log"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Chanceware live diagnostics" -ForegroundColor Cyan
Write-Host "  Runtime Binary: $BinaryPath" -ForegroundColor Yellow
Write-Host "  Log Level: $LogLevel" -ForegroundColor Yellow
Write-Host "  Log File: $LogFile" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

& $BinaryPath run $GamePath --execute --console series-x 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $LogFile
exit $LASTEXITCODE
