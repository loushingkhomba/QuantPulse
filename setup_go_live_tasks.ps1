param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$PythonExe,
    [string]$DailyTime = "08:45",
    [string]$WeeklyTime = "17:15"
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}

$DailyScript = Join-Path $ProjectRoot "paper_live_60d.py"
$WeeklyScript = Join-Path $ProjectRoot "go_live_weekly_report.py"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $DailyScript)) {
    throw "Daily script not found: $DailyScript"
}
if (-not (Test-Path $WeeklyScript)) {
    throw "Weekly script not found: $WeeklyScript"
}

$DailyAction = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$DailyScript`"" -WorkingDirectory $ProjectRoot
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($DailyTime, "HH:mm", $null))
$DailySettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$DailyTask = New-ScheduledTask -Action $DailyAction -Trigger $DailyTrigger -Settings $DailySettings -Description "Run the daily paper-live signal update"

$WeeklyAction = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$WeeklyScript`"" -WorkingDirectory $ProjectRoot
$WeeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At ([datetime]::ParseExact($WeeklyTime, "HH:mm", $null))
$WeeklySettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$WeeklyTask = New-ScheduledTask -Action $WeeklyAction -Trigger $WeeklyTrigger -Settings $WeeklySettings -Description "Run the weekly go-live gate report"

foreach ($TaskName in @("QuantPulse_DailyPaper", "QuantPulse_WeeklyGateReport")) {
    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

Register-ScheduledTask -TaskName "QuantPulse_DailyPaper" -InputObject $DailyTask | Out-Null
Register-ScheduledTask -TaskName "QuantPulse_WeeklyGateReport" -InputObject $WeeklyTask | Out-Null

Write-Host "Scheduled tasks registered successfully."
Write-Host "Daily task: QuantPulse_DailyPaper at $DailyTime"
Write-Host "Weekly task: QuantPulse_WeeklyGateReport at $WeeklyTime"
