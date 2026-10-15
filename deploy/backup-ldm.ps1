<#
.SYNOPSIS
  Back up the LDM data and models (safe while the app is running). Restore steps: handover\DEPLOYMENT_AND_ROLLBACK.md.

.DESCRIPTION
  Copies into <Destination>\ldm_backup_<yyyyMMdd_HHmmss>\:
    db\litter_monitor.db       consistent copy made with SQLite's online-backup API (alerts, detections, jobs, settings, cameras)
    storage\alerts, storage\captures, storage\ingest (ledger + results)   evidence snapshots and ingest records
    models\litter_yolo\        every registered model version (+ SHA256SUMS of each best.pt)
    SHA256SUMS.txt             checksums of every copied file
  Not included by default: storage\videos (large annotated videos; add -IncludeVideos) and .env (secrets: add
  -IncludeEnv only for a destination that is as protected as this PC).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\backup-ldm.ps1 -Destination D:\LDM_backups
#>
param(
  [string]$Destination = (Join-Path $env:USERPROFILE 'LDM_backups'),
  [switch]$IncludeVideos,
  [switch]$IncludeEnv
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $Destination "ldm_backup_$stamp"
New-Item -ItemType Directory -Force -Path (Join-Path $out 'db') | Out-Null

$py = if (Test-Path (Join-Path $Root '.venv\Scripts\python.exe')) { Join-Path $Root '.venv\Scripts\python.exe' } else { (& py -c "import sys; print(sys.executable)").Trim() }
$db = if ($env:LM_DB_PATH) { $env:LM_DB_PATH } else { Join-Path $Root 'storage\litter_monitor.db' }
if (Test-Path $db) {
  & $py -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()" $db (Join-Path $out 'db\litter_monitor.db')
  if ($LASTEXITCODE -ne 0) { throw "database backup failed" }
  Write-Host "  [ OK ] database (online backup)"
} else { Write-Host "  [WARN] no database at $db" -ForegroundColor Yellow }

$dirs = @('storage\alerts', 'storage\captures', 'storage\ingest', 'models\litter_yolo')
if ($IncludeVideos) { $dirs += 'storage\videos' }
foreach ($d in $dirs) {
  $src = Join-Path $Root $d
  if (Test-Path $src) {
    $dst = Join-Path $out $d
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Copy-Item -Recurse -Force $src $dst
    Write-Host "  [ OK ] $d"
  }
}
if ($IncludeEnv -and (Test-Path (Join-Path $Root '.env'))) {
  Copy-Item (Join-Path $Root '.env') (Join-Path $out '.env')
  Write-Host "  [WARN] .env copied: it contains the admin token - keep this backup protected" -ForegroundColor Yellow
}
$sums = Get-ChildItem -Recurse -File $out | ForEach-Object {
  "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.FullName.Substring($out.Length + 1)
}
$sums | Set-Content -Encoding utf8 (Join-Path $out 'SHA256SUMS.txt')
$size = (Get-ChildItem -Recurse -File $out | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("backup: {0}  ({1:N1} MB, {2} files)" -f $out, $size, $sums.Count) -ForegroundColor Green
