param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$PythonExe,
    [string]$DailyTime = "20:00",
    [string]$PhaseStartTime = "08:45",
    [string]$ReviewTime = "20:15",
    [string]$CampaignName = "forward_phaseA_2026"
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}

$DailyWrapper = Join-Path $ProjectRoot "run_go_live_daily.ps1"
$MilestoneScript = Join-Path $ProjectRoot "go_live_milestone.py"

foreach ($Path in @($PythonExe, $DailyWrapper, $MilestoneScript)) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

function New-PlanTaskAction {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments
    )

    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath) + $Arguments
    $quotedArgs = foreach ($arg in $argList) {
        if ($null -eq $arg) {
            continue
        }
        $escaped = $arg -replace '"', '""'
        if ($escaped -match '\s') {
            '"' + $escaped + '"'
        }
        else {
            $escaped
        }
    }

    return New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($quotedArgs -join ' ') -WorkingDirectory $ProjectRoot
}

function New-OneTimeTask {
    param(
        [string]$TaskName,
        [datetime]$When,
        [Microsoft.Management.Infrastructure.CimInstance]$Action,
        [string]$Description
    )

    $Trigger = New-ScheduledTaskTrigger -Once -At $When
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description

    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask -TaskName $TaskName -InputObject $Task | Out-Null
}

function New-DailyTask {
    param(
        [string]$TaskName,
        [datetime]$At,
        [Microsoft.Management.Infrastructure.CimInstance]$Action,
        [string]$Description
    )

    $Trigger = New-ScheduledTaskTrigger -Daily -At $At
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description

    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask -TaskName $TaskName -InputObject $Task | Out-Null
}

$DailyAt = [datetime]::ParseExact($DailyTime, "HH:mm", $null)
$PhaseStartAt = [datetime]::ParseExact($PhaseStartTime, "HH:mm", $null)
$ReviewAt = [datetime]::ParseExact($ReviewTime, "HH:mm", $null)

$DailyAction = New-PlanTaskAction -ScriptPath $DailyWrapper -Arguments @()
New-DailyTask -TaskName "QuantPulse_GoLive_DailyPaper" -At $DailyAt -Action $DailyAction -Description "Run the daily paper-live campaign wrapper"

$Milestones = @(
    @{ Name = "QuantPulse_PhaseA_Start_2026-04-08"; When = [datetime]::ParseExact("2026-04-08 $PhaseStartTime", "yyyy-MM-dd HH:mm", $null); Kind = "phase-start"; Phase = "PhaseA"; Label = "START Phase A, Week 1"; Description = "Mark Phase A start" },
    @{ Name = "QuantPulse_Week1_GateReview_2026-04-11"; When = [datetime]::ParseExact("2026-04-11 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 1 gate review"; Description = "Run Week 1 gate review" },
    @{ Name = "QuantPulse_Week2_GateReview_2026-04-17"; When = [datetime]::ParseExact("2026-04-17 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 2 gate review"; Description = "Run Week 2 gate review" },
    @{ Name = "QuantPulse_Week3_GateReview_2026-04-25"; When = [datetime]::ParseExact("2026-04-25 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 3 gate review"; Description = "Run Week 3 gate review" },
    @{ Name = "QuantPulse_Week4_GateReview_2026-05-02"; When = [datetime]::ParseExact("2026-05-02 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 4 gate review + Phase A exit decision"; Description = "Run Week 4 gate review and Phase A exit decision" },
    @{ Name = "QuantPulse_PhaseB_Start_2026-05-05"; When = [datetime]::ParseExact("2026-05-05 $PhaseStartTime", "yyyy-MM-dd HH:mm", $null); Kind = "phase-start"; Phase = "PhaseB"; Label = "START Phase B"; Description = "Mark Phase B start" },
    @{ Name = "QuantPulse_Week5_GateReview_2026-05-09"; When = [datetime]::ParseExact("2026-05-09 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 5 gate review"; Description = "Run Week 5 gate review" },
    @{ Name = "QuantPulse_Week6_GateReview_2026-05-16"; When = [datetime]::ParseExact("2026-05-16 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "review"; Phase = ""; Label = "Week 6 gate review + Phase B exit decision"; Description = "Run Week 6 gate review and Phase B exit decision" },
    @{ Name = "QuantPulse_PhaseC_Start_2026-05-19"; When = [datetime]::ParseExact("2026-05-19 $PhaseStartTime", "yyyy-MM-dd HH:mm", $null); Kind = "phase-start"; Phase = "PhaseC"; Label = "START Phase C"; Description = "Mark Phase C start" },
    @{ Name = "QuantPulse_Full_GoLive_Decision_2026-06-13"; When = [datetime]::ParseExact("2026-06-13 $ReviewTime", "yyyy-MM-dd HH:mm", $null); Kind = "final-decision"; Phase = ""; Label = "Full go-live decision"; Description = "Run the full go-live decision" }
)

foreach ($Milestone in $Milestones) {
    $Args = @("--kind", $Milestone.Kind, "--label", $Milestone.Label, "--campaign-name", $CampaignName)
    if ($Milestone.Phase) {
        $Args += @("--phase", $Milestone.Phase)
    }
    $Action = New-PlanTaskAction -ScriptPath $MilestoneScript -Arguments $Args
    New-OneTimeTask -TaskName $Milestone.Name -When $Milestone.When -Action $Action -Description $Milestone.Description
}

Write-Host "Go-live schedule registered successfully."
Write-Host "Daily paper task: QuantPulse_GoLive_DailyPaper at $DailyTime"
Write-Host "Phase start time: $PhaseStartTime"
Write-Host "Review time: $ReviewTime"
Write-Host "Campaign name: $CampaignName"
Write-Host "Milestone dates:"
foreach ($Milestone in $Milestones) {
    Write-Host "  $($Milestone.Name) -> $($Milestone.When)"
}
