param(
  [string]$TargetRoot = (Get-Location).Path
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($folder in @('code', 'src', 'logs', 'models')) {
  $source = Join-Path $here $folder
  if (Test-Path $source) {
    Get-ChildItem $source -Recurse -File | ForEach-Object {
      $relative = $_.FullName.Substring($source.Length).TrimStart('\')
      $destination = Join-Path (Join-Path $TargetRoot $folder) $relative
      $destinationDir = Split-Path -Parent $destination
      if (-not (Test-Path $destinationDir)) {
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
      }
      Copy-Item $_.FullName $destination -Force
    }
  }
}

Write-Output "RESTORE_COMPLETE from $here"
