Param(
    [switch]$DryRun,
    [switch]$Live
)

$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot\.."

$venvActivate = Join-Path (Get-Location) 'venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    & $venvActivate
}

$env:QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID = '1'
$env:QUANT_PAPER_TRADING_MODE = if ($Live) { 'live' } else { 'simulation' }
$env:QUANT_REALTIME_DATA_SOURCE = 'mock'

if ($DryRun) {
    $env:QUANT_PAPER_MAX_CYCLES = '3'
    $env:QUANT_PAPER_SLEEP_SECONDS = '2'
    $env:QUANT_PAPER_FORCE_RUN_OUTSIDE_HOURS = '1'
} else {
    $env:QUANT_PAPER_MAX_CYCLES = '0'
    $env:QUANT_PAPER_SLEEP_SECONDS = '0'
    $env:QUANT_PAPER_FORCE_RUN_OUTSIDE_HOURS = '0'
}

python scripts/paper_trade_preflight.py
python src/paper_trading.py
