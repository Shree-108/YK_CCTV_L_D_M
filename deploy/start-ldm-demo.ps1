<#
.SYNOPSIS
  Start a SEPARATE demo instance of LDM: its own database, storage, dataset folder and model copy; no notifications.

.DESCRIPTION
  Uses only settings the application already supports (backend/core/config.py):
    LM_STORAGE_DIR, LM_DB_PATH, LM_AUTH_DB_PATH, LM_DATASET_DIR, LM_MODELS_DIR   -> all under -DemoRoot
    PYTHON_DOTENV_DISABLED=1   the demo backend does NOT read .env, so SMTP / webhook / Telegram / CCTV credentials
                               and the production admin token are never loaded (python-dotenv >= 1.1 required, checked)
    ADMIN_API_TOKEN            a separate random demo token, stored in <DemoRoot>\demo_admin_token.txt
    LM_START_CAMERAS=false     no CCTV cameras
  Ports 8100 (API) / 3110 (UI) by default, so it can run next to the normal instance (8000 / 3010).
  PID files and logs: deploy\run\demo\ and deploy\logs\demo\ (never the normal instance's files).
  UI build: deploy\run\demo\next-ui (git-ignored); tsconfig.json and next-env.d.ts are restored byte-for-byte after it.
  A fingerprint of the production storage/, dataset/, models/ and .env is taken at start
  (<DemoRoot>\production_before.json); deploy\stop-ldm-demo.ps1 compares it again.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\start-ldm-demo.ps1
  powershell -ExecutionPolicy Bypass -File deploy\start-ldm-demo.ps1 -DemoRoot D:\LDM_demo_data
#>
[CmdletBinding()]
param(
  [string]$DemoRoot = "",
  [int]$BackendPort = 8100,
  [int]$FrontendPort = 3110,
  [string]$Python = "",
  [string]$SourceModelsDir = "",   # where v3 is copied FROM (default: <project>\models); only read
  [switch]$Rebuild,
  [int]$BackendTimeoutSec = 240,
  [int]$FrontendTimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot 'logs\demo'
$RunDir = Join-Path $PSScriptRoot 'run\demo'
New-Item -ItemType Directory -Force -Path $LogDir, $RunDir | Out-Null
Set-Location $Root

function Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Info($m) { Write-Host "  $m" }
function Fail($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }
function Step($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Get-Listener([int]$port) {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue }
}
function Get-Json([string]$url) { try { Invoke-RestMethod -Uri $url -TimeoutSec 5 -UseBasicParsing } catch { $null } }
function Start-Hidden([string]$exe, [string[]]$arguments, [string]$out, [string]$err, [switch]$Wait) {
  $cmdline = '"' + $exe + '" ' + (($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
  $full = '"' + $cmdline + ' > "' + $out + '" 2> "' + $err + '""'
  Start-Process -FilePath $env:ComSpec -ArgumentList @('/d', '/s', '/c', $full) -WorkingDirectory $Root -WindowStyle Hidden -PassThru -Wait:$Wait
}
function Save-Pid([string]$name, $proc, [int]$port) {
  @{ pid = $proc.Id; port = $port; started = (Get-Date).ToString('s'); demo_root = $DemoRoot } | ConvertTo-Json |
    Set-Content -Encoding utf8 (Join-Path $RunDir "$name.json")
}
function Full([string]$p) { [IO.Path]::GetFullPath($p).TrimEnd('\') }

Write-Host "Yatrikart Litter Detection - SEPARATE DEMO instance" -ForegroundColor Cyan

# ------------------------------------------------------------------ 1. paths and guards
Step "1/6 Demo folders and safety checks"
if (-not $DemoRoot) { $DemoRoot = Join-Path (Split-Path -Parent $Root) 'LDM_demo_data' }
$DemoRoot = Full $DemoRoot
$RootFull = Full $Root
if ($DemoRoot -eq $RootFull -or $DemoRoot.StartsWith($RootFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
  Fail "-DemoRoot must be OUTSIDE the project folder ($RootFull), so demo data can never mix with production storage or git."
}
if (($BackendPort -eq 8000) -or ($FrontendPort -eq 3010) -or ($BackendPort -eq $FrontendPort)) { Fail "Use ports different from the normal instance (8000 / 3010) and from each other." }
foreach ($p in $BackendPort, $FrontendPort) {
  $l = Get-Listener $p
  if ($l) { Fail "Port $p is already in use by $($l.ProcessName) (PID $($l.Id)). Stop it (deploy\stop-ldm-demo.ps1) or choose another port." }
}
$Storage = Join-Path $DemoRoot 'storage'; $Dataset = Join-Path $DemoRoot 'dataset'; $Models = Join-Path $DemoRoot 'models'
New-Item -ItemType Directory -Force -Path $Storage, $Dataset, (Join-Path $Models 'litter_yolo') | Out-Null
$Db = Join-Path $Storage 'litter_monitor_demo.db'
$ProdDb = Full (Join-Path $Root 'storage\litter_monitor.db')
if ((Full $Db) -eq $ProdDb) { Fail "demo database path equals the production database" }
Ok "demo root: $DemoRoot"
Info "database: $Db"

if (-not $Python) {
  $venv = Join-Path $Root '.venv\Scripts\python.exe'
  if (Test-Path $venv) { $Python = $venv }
  elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = (& py -c "import sys; print(sys.executable)").Trim() }
  else { $Python = (Get-Command python).Source }
}
$dotenvOk = (& $Python -c "import dotenv.main as m; print(hasattr(m, '_load_dotenv_disabled'))").Trim()
if ($dotenvOk -ne 'True') { Fail "python-dotenv is too old to honour PYTHON_DOTENV_DISABLED (need >= 1.1): the demo could load .env secrets. Upgrade python-dotenv." }
Ok "python-dotenv supports PYTHON_DOTENV_DISABLED (.env will not be read by the demo API)"

# ------------------------------------------------------------------ 2. model copy (production weights are only read)
Step "2/6 Model copy for the demo"
if (-not $SourceModelsDir) { $SourceModelsDir = Join-Path $Root 'models' }
$srcYolo = Join-Path $SourceModelsDir 'litter_yolo'
$versions = @(Get-ChildItem $srcYolo -Directory -ErrorAction SilentlyContinue |
  Where-Object { (Test-Path (Join-Path $_.FullName 'best.pt')) -and (Test-Path (Join-Path $_.FullName 'model_meta.json')) })
if ($versions.Count -eq 0) { Fail "No model weights in $srcYolo" }
foreach ($v in $versions) {
  $dst = Join-Path $Models "litter_yolo\$($v.Name)"
  $srcHash = (Get-FileHash (Join-Path $v.FullName 'best.pt') -Algorithm SHA256).Hash
  if (-not (Test-Path (Join-Path $dst 'best.pt'))) {
    New-Item -ItemType Directory -Force $dst | Out-Null
    Get-ChildItem $v.FullName -File | Where-Object { $_.Name -in 'best.pt', 'model_meta.json', 'class_names.json', 'data.yaml', 'args.yaml' } |
      Copy-Item -Destination $dst
  }
  $dstHash = (Get-FileHash (Join-Path $dst 'best.pt') -Algorithm SHA256).Hash
  if ($dstHash -ne $srcHash) { Fail "demo copy of $($v.Name)\best.pt differs from the source ($dstHash vs $srcHash)" }
  Ok "$($v.Name): demo copy verified (sha256 $($srcHash.Substring(0,12).ToLower())...)"
}

# ------------------------------------------------------------------ 3. production fingerprint
Step "3/6 Fingerprint of production data (before)"
$before = Join-Path $DemoRoot 'production_before.json'
& $Python (Join-Path $PSScriptRoot 'check_demo_isolation.py') snapshot $before
if ($LASTEXITCODE -ne 0) { Fail "could not fingerprint production data" }

# ------------------------------------------------------------------ 4. backend
Step "4/6 Demo API on port $BackendPort"
$tokFile = Join-Path $DemoRoot 'demo_admin_token.txt'
if (-not (Test-Path $tokFile)) {
  $bytes = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  ('demo-' + [Convert]::ToBase64String($bytes).Replace('+', 'A').Replace('/', 'B').TrimEnd('=')) | Set-Content -Encoding ascii -NoNewline $tokFile
}
# settings already supported by backend/core/config.py - process environment of THIS script only
$env:PYTHON_DOTENV_DISABLED = '1'
$env:LM_STORAGE_DIR = $Storage
$env:LM_DB_PATH = $Db
$env:LM_AUTH_DB_PATH = Join-Path $Storage 'auth_demo.db'
$env:LM_DATASET_DIR = $Dataset
$env:LM_MODELS_DIR = $Models
$env:LM_ROI_SEGMENTER_DIR = Join-Path $Root 'models\roi_segmenter\upernet-convnext-tiny'   # read-only use
$env:LM_START_CAMERAS = 'false'
$env:ADMIN_API_TOKEN = (Get-Content $tokFile -Raw).Trim()
$env:CORS_ORIGINS = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
$env:PYTHONUNBUFFERED = '1'
# a value set in the parent shell would still reach the API: remove every notification / camera credential
foreach ($k in 'SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD', 'ALERT_EMAIL_TO', 'ALERT_EMAIL_FROM', 'ALERT_WEBHOOK_URL',
              'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'CCTV_HOST', 'CCTV_USERNAME', 'CCTV_PASSWORD') {
  Remove-Item "env:$k" -ErrorAction SilentlyContinue
}
$bLog = Join-Path $LogDir 'backend.log'; $bErr = Join-Path $LogDir 'backend.err.log'
$bp = Start-Hidden $Python @('-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', "$BackendPort") $bLog $bErr
Save-Pid 'backend' $bp $BackendPort
$deadline = (Get-Date).AddSeconds($BackendTimeoutSec); $health = $null
while ((Get-Date) -lt $deadline) {
  $health = Get-Json "http://127.0.0.1:$BackendPort/api/health"; if ($health) { break }
  if ($bp.HasExited) { Get-Content $bErr -Tail 30; Fail "demo API stopped during start-up" }
  Start-Sleep -Seconds 2
}
if (-not $health) { Fail "demo API did not answer within $BackendTimeoutSec s (see $bErr)" }
if (-not $health.model) { Fail "demo API has no model: $($health.model_error)" }
$notif = @($health.notifications | Where-Object { $_.configured })
if ($notif.Count -gt 0) { Fail "a notification provider is configured in the demo API ($($notif.provider -join ', ')): stopping is required" }
if ($health.cameras.configured -ne 0) { Fail "demo API has cameras configured" }
if (-not (Test-Path $Db)) { Fail "demo database was not created at $Db" }
Ok "demo API healthy: model $($health.model.version) (from $Models), notifications: none configured, cameras: 0"

# ------------------------------------------------------------------ 5. UI
Step "5/6 Demo web UI on port $FrontendPort"
$node = (Get-Command node).Source
$nextBin = Join-Path $Root 'node_modules\next\dist\bin\next'
if (-not (Test-Path $nextBin)) { Fail "node_modules missing (npm install)" }
$ApiUrl = "http://localhost:$BackendPort"
$env:LDM_API_ORIGIN = "http://127.0.0.1:$BackendPort"   # server-side /ldm-api rewrite (next.config.mjs)
$env:NEXT_DIST_DIR = 'deploy/run/demo/next-ui'
$env:NEXT_TELEMETRY_DISABLED = '1'
$dist = Join-Path $Root 'deploy\run\demo\next-ui'
$stamp = Join-Path $dist 'ldm-demo-build.json'
$need = $Rebuild -or -not (Test-Path (Join-Path $dist 'BUILD_ID')) -or -not (Test-Path $stamp)
if (-not $need) {
  $built = (Get-Item (Join-Path $dist 'BUILD_ID')).LastWriteTime
  if ((Get-Content $stamp -Raw | ConvertFrom-Json).api -ne $ApiUrl) { $need = $true }
  if (Get-ChildItem -Recurse -File -Path app, frontend, components, lib, public -ErrorAction SilentlyContinue |
      Where-Object { $_.LastWriteTime -gt $built } | Select-Object -First 1) { $need = $true }
}
if ($need) {
  Info "building the demo UI into deploy\run\demo\next-ui (a few minutes)..."
  $keep = @{}
  foreach ($f in 'tsconfig.json', 'next-env.d.ts') { if (Test-Path $f) { $keep[$f] = [IO.File]::ReadAllBytes((Join-Path $Root $f)) } }
  try { $b = Start-Hidden $node @($nextBin, 'build') (Join-Path $LogDir 'ui-build.log') (Join-Path $LogDir 'ui-build.err.log') -Wait }
  finally { foreach ($f in $keep.Keys) { [IO.File]::WriteAllBytes((Join-Path $Root $f), $keep[$f]) } }
  if ($b.ExitCode -ne 0) { Get-Content (Join-Path $LogDir 'ui-build.log') -Tail 30; Fail "UI build failed" }
  @{ api = $ApiUrl; built = (Get-Date).ToString('s') } | ConvertTo-Json | Set-Content -Encoding utf8 $stamp
  Ok "demo UI built (tsconfig.json / next-env.d.ts restored)"
} else { Ok "demo UI build up to date" }
$fp = Start-Hidden $node @($nextBin, 'start', '-p', "$FrontendPort", '-H', '127.0.0.1') (Join-Path $LogDir 'ui.log') (Join-Path $LogDir 'ui.err.log')
Save-Pid 'frontend' $fp $FrontendPort
$deadline = (Get-Date).AddSeconds($FrontendTimeoutSec); $up = $false
while ((Get-Date) -lt $deadline) {
  try { $r = Invoke-WebRequest "http://127.0.0.1:$FrontendPort/ldm" -UseBasicParsing -TimeoutSec 20; if ($r.StatusCode -eq 200) { $up = $true; break } } catch { }
  if ($fp.HasExited) { Fail "demo UI stopped during start-up (see $LogDir\ui.err.log)" }
  Start-Sleep -Seconds 2
}
if (-not $up) { Fail "demo UI did not answer on port $FrontendPort" }
Ok "demo UI answering"

# ------------------------------------------------------------------ 6. summary
Step "6/6 Ready - DEMO instance"
Write-Host ""
Write-Host "  Open:          http://localhost:$FrontendPort/ldm" -ForegroundColor Green
Write-Host "  Demo data:     $DemoRoot   (database, uploads, alerts, snapshots, job outputs)"
Write-Host "  Admin token:   $tokFile   (demo-only; paste in Settings when an admin action is needed)"
Write-Host "  Not isolated:  saving a CCTV camera password on the Live CCTV page writes the project .env - do not use it in the demo."
Write-Host "  Stop + check:  powershell -ExecutionPolicy Bypass -File deploy\stop-ldm-demo.ps1"
Write-Host ""
exit 0
