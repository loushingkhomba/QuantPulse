param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$StartDate = "2026-04-08",
    [string]$EndDate = "2026-06-13",
    [string]$PythonExe,
    [string]$CampaignName = "forward_phaseA_2026",
    [int]$MaxDataLagDays = 3,
    [switch]$AllowStaleData
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}

$DailyScript = Join-Path $ProjectRoot "paper_live_60d.py"
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $DailyScript)) {
    throw "Daily script not found: $DailyScript"
}

$Today = (Get-Date).Date
$Start = [datetime]::ParseExact($StartDate, "yyyy-MM-dd", $null).Date
$End = [datetime]::ParseExact($EndDate, "yyyy-MM-dd", $null).Date

if ($Today -lt $Start -or $Today -gt $End) {
    Write-Host "Outside configured campaign window ($StartDate to $EndDate). Skipping run."
    exit 0
}

if ($Today.DayOfWeek -in @([System.DayOfWeek]::Saturday, [System.DayOfWeek]::Sunday)) {
    Write-Host "Weekend detected. Skipping run."
    exit 0
}

$args = @(
    $DailyScript,
    "--campaign-name", $CampaignName,
    "--campaign-start-date", $StartDate,
    "--max-data-lag-days", "$MaxDataLagDays"
)

if ($AllowStaleData) {
    $args += "--allow-stale-data"
}

& $PythonExe @args
exit $LASTEXITCODE
