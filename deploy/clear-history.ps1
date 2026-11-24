<#
.SYNOPSIS
  Clear the dashboard history (alerts, jobs, results, events, detections). Always backs up first; files are MOVED to
  storage\_quarantine\history_<timestamp>\, never deleted.

.DESCRIPTION
  1. Refuses while any job / analysis is queued or running.
  2. Asks you to type CLEAR HISTORY.
  3. Stops the app (deploy\stop-ldm.ps1).
  4. python -m backend.services.history_reset run: snapshot, backup-ldm.ps1 -IncludeVideos + checksum verification,
     JSON export of the history tables, one-transaction delete + VACUUM, folder move + MANIFEST.json, offline checks.
  5. Starts the app again (deploy\start-ldm.ps1) and verifies: history counts 0, model/config/ledger checksums
     unchanged, /api/health reports the same model, every dashboard history endpoint empty.
  If any step fails it stops there and does NOT restart the app, so you can inspect first.
  Not touched: models\, dataset\, config\, .env, deploy\logs, storage\eval, storage\runs, storage\monitoring, settings,
  the export ledger. Undo: handover\DEPLOYMENT_AND_ROLLBACK.md (restore the DB from the backup, move folders back).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\clear-history.ps1 -BackendPort 8001
#>
param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3010,
  [string]$BackupDest = (Join-Path $env:USERPROFILE 'LDM_backups')
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = if (Test-Path (Join-Path $Root '.venv\Scripts\python.exe')) { Join-Path $Root '.venv\Scripts\python.exe' } else { 'py' }
$tool = @('-m', 'backend.services.history_reset', '--backup-dest', $BackupDest)

Write-Host "== what would be cleared"
& $py @tool status
if ($LASTEXITCODE -eq 3) { throw "a job or analysis is still queued/running - wait for it to finish, then retry" }
if ($LASTEXITCODE -ne 0) { throw "status check failed" }

$phrase = Read-Host "Type CLEAR HISTORY to back up and clear the history (anything else cancels)"
if ($phrase -cne 'CLEAR HISTORY') { Write-Host "cancelled - nothing changed"; exit 1 }

Write-Host "== stopping the app"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'stop-ldm.ps1')

Write-Host "== reset"
& $py @tool --health-url "http://127.0.0.1:$BackendPort" run --confirm --phrase $phrase
if ($LASTEXITCODE -ne 0) { throw "reset STOPPED (exit $LASTEXITCODE) - the app was not restarted; read the message above" }
$q = Get-ChildItem (Join-Path $Root 'storage\_quarantine') -Directory -Filter 'history_*' | Sort-Object Name | Select-Object -Last 1

Write-Host "== starting the app"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-ldm.ps1') -BackendPort $BackendPort -FrontendPort $FrontendPort
if ($LASTEXITCODE -ne 0) { throw "start-ldm.ps1 failed (exit $LASTEXITCODE)" }

Write-Host "== verifying"
& $py @tool --health-url "http://127.0.0.1:$BackendPort" verify --quarantine $q.FullName
if ($LASTEXITCODE -ne 0) { throw "verification FAILED - see the report above (quarantine: $($q.FullName))" }
Write-Host "history cleared and verified. Quarantine: $($q.FullName)" -ForegroundColor Green
