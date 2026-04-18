$ErrorActionPreference = 'Stop'

# Monthly retrain pipeline
# 1) Refresh cached market data
# 2) Retrain latest Stage-3 config
# 3) Run rolling recent validation window

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:QUANT_REFRESH_DATA = '1'
$env:QUANT_HISTORY_YEARS = '7'
$env:QUANT_TRAIN_START = '2023-01-01'
$env:QUANT_TARGET_MODE = 'absolute'
$env:QUANT_TARGET_ABS_THRESHOLD = '0.001'
$env:QUANT_TARGET_HORIZON_DAYS = '1'
$env:QUANT_HOLDING_DAYS = '1'
$env:QUANT_OBJECTIVE_MODE = 'ranking'
$env:QUANT_SIMPLE_HIDDEN_SIZE = '64'
$env:QUANT_TRAIN_EPOCHS = '80'
$env:QUANT_TRAIN_PATIENCE = '15'
$env:QUANT_CONFIDENCE_PENALTY = '0.005'
$env:QUANT_SIGNAL_INVERSION_MODE = 'none'
$env:QUANT_PURGED_WF_ENABLED = '1'
$env:QUANT_PURGE_DAYS = '5'
$env:QUANT_XS_RANK_ALL_FEATURES = '1'
$env:QUANT_HYBRID_ENSEMBLE_ENABLED = '1'
$env:QUANT_HYBRID_BLEND_WEIGHT = '0.35'
$env:QUANT_IC_BREAKER_ENABLED = '1'
$env:QUANT_KELLY_LITE_ENABLED = '1'

Write-Host 'Running monthly retrain + validation backtest...'
$validationEnd = (Get-Date).AddDays(-1).ToString('yyyy-MM-dd')
$validationStart = (Get-Date).AddDays(-3).ToString('yyyy-MM-dd')
Write-Host "Validation window: $validationStart to $validationEnd"
venv/Scripts/python.exe train.py --start $validationStart --end $validationEnd
