param(
  [string]$TargetRoot = (Get-Location).Path
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$codeTrain = Join-Path $here "code/train.py"
if (Test-Path $codeTrain) {
  Copy-Item $codeTrain (Join-Path $TargetRoot "train.py") -Force
}

$srcModel = Join-Path $here "src/src/model.py"
if (Test-Path $srcModel) {
  $destDir = Join-Path $TargetRoot "src"
  if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
  Copy-Item $srcModel (Join-Path $destDir "model.py") -Force
}

Write-Output "RESTORE_COMPLETE from $here"
