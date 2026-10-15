<#
.SYNOPSIS
  Start the Yatrikart Litter Detection (LDM) demo: backend API, health check, then the web UI.

.DESCRIPTION
  1. Checks Python, Python packages, Node.js, node_modules, the model files and .env.
  2. Starts the FastAPI backend (or re-uses an LDM backend that is already running on the port).
  3. Waits until GET /api/health answers and reports which model is loaded.
  4. Builds the web UI if needed (into .next-demo, never touching a running `next dev`) and starts it.
  5. Verifies the browser will be allowed to call the backend (CORS) from the UI address.
  Safe to run twice: running LDM processes are detected and re-used, nothing is started twice.
  Stop with deploy\stop-ldm.ps1 (only processes started by this script are stopped).

  Local only (127.0.0.1). -Lan (reachable from other PCs) is DISABLED until login protection is enabled
  (API security plan Step 2-4): the API itself also rejects every non-local request until then.
  CCTV cameras are NOT started (LM_START_CAMERAS=false) unless -EnableCameras is given.
  The model used is whatever is active in the app (production: v3_store_cctv); this script never changes it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1
  powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1 -FrontendPort 3010 -BackendPort 8000
  powershell -ExecutionPolicy Bypass -File deploy\start-ldm.ps1 -Dev        # next dev instead of a production build
#>
[CmdletBinding()]
param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 3010,
  [switch]$Dev,                 # run `next dev` (hot reload) instead of `next build` + `next start`
  [switch]$Rebuild,             # force a fresh production build of the UI
  [switch]$Lan,                 # bind to 0.0.0.0 and use this PC's LAN address (other PCs on the network can open it)
  [switch]$EnableCameras,       # keep LM_START_CAMERAS from .env (default: cameras are not started for the demo)
  [string]$Python = "",         # explicit python.exe (default: .venv, then the py launcher, then python)
  [int]$BackendTimeoutSec = 240,
  [int]$FrontendTimeoutSec = 600
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot 'logs'
$RunDir = Join-Path $PSScriptRoot 'run'
New-Item -ItemType Directory -Force -Path $LogDir, $RunDir | Out-Null
Set-Location $Root

function Info($m) { Write-Host "  $m" }
function Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; exit 1 }
function Step($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }

function Get-Listener([int]$port) {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue }
}
function Get-Json([string]$url, [int]$timeout = 5, [hashtable]$headers = @{}) {
  try { return Invoke-RestMethod -Uri $url -TimeoutSec $timeout -Headers $headers -UseBasicParsing } catch { return $null }
}
function Test-LdmBackend([int]$port) {
  $h = Get-Json "http://127.0.0.1:$port/api/health"
  return ($h -ne $null -and $h.PSObject.Properties.Name -contains 'uptime_sec' -and $h.PSObject.Properties.Name -contains 'model')
}
function Test-LdmFrontend([int]$port) {
  try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -TimeoutSec 20 -UseBasicParsing; return ($r.StatusCode -eq 200 -and $r.Content -match 'Yatrikart') }
  catch { return $false }
}
function Save-Pid([string]$name, $proc, [int]$port, [string]$what) {
  @{ pid = $proc.Id; port = $port; what = $what; started = (Get-Date).ToString('s') } | ConvertTo-Json |
    Set-Content -Encoding utf8 (Join-Path $RunDir "$name.json")
}
function Rotate-Log([string]$path) { if (Test-Path $path) { Move-Item -Force $path "$path.prev" } }
# Start a hidden background process whose output goes to log files. It goes through `cmd /c ... > log` started with
# ShellExecute, so the child does NOT inherit this console's handles (otherwise a caller that captures this script's
# output, e.g. `start-ldm.ps1 > out.txt` or a scheduled task, would wait forever for the servers to exit).
function Start-Hidden([string]$exe, [string[]]$arguments, [string]$out, [string]$err, [switch]$Wait) {
  $cmdline = '"' + $exe + '" ' + (($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
  $full = '"' + $cmdline + ' > "' + $out + '" 2> "' + $err + '""'
  $p = Start-Process -FilePath $env:ComSpec -ArgumentList @('/d', '/s', '/c', $full) -WorkingDirectory $Root -WindowStyle Hidden -PassThru -Wait:$Wait
  return $p
}
function Show-LogTail([string]$path, [int]$n = 30) {
  if (Test-Path $path) { Write-Host "  --- last lines of $path ---"; Get-Content $path -Tail $n | ForEach-Object { Write-Host "  | $_" } }
}

Write-Host "Yatrikart Litter Detection - demo startup" -ForegroundColor Cyan
Write-Host "project: $Root"

# The API has no user login yet: never offer it to the network (API security plan, Step 1).
if ($Lan) {
  Fail ("-Lan is disabled: the application has no login protection yet, so CCTV video, snapshots and data would be " +
        "readable by anyone on the network. The API also rejects non-local requests until then. Run without -Lan and " +
        "open the demo on this PC (http://localhost:$FrontendPort).")
}

# ------------------------------------------------------------------ 1. checks
Step "1/5 Checking dependencies and configuration"
if (-not $Python) {
  $venv = Join-Path $Root '.venv\Scripts\python.exe'
  if (Test-Path $venv) { $Python = $venv }
  elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = (& py -c "import sys; print(sys.executable)").Trim() }
  elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = (& python -c "import sys; print(sys.executable)").Trim() }
}
if (-not $Python -or -not (Test-Path $Python)) { Fail "Python not found. Install Python 3.10+ (or pass -Python C:\path\python.exe)." }
$pyver = (& $Python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])").Trim()
if ([version]$pyver -lt [version]'3.10') { Fail "Python $pyver is too old (3.10+ needed): $Python" }
Ok "Python $pyver ($Python)"

$missing = (& $Python -c @"
import importlib.util as u
mods = {'fastapi':'fastapi','uvicorn':'uvicorn[standard]','multipart':'python-multipart','ultralytics':'ultralytics','torch':'torch',
        'cv2':'opencv-python','numpy':'numpy','yaml':'pyyaml','dotenv':'python-dotenv','psutil':'psutil','pydantic':'pydantic'}
print(' '.join(p for m, p in mods.items() if u.find_spec(m) is None))
"@).Trim()
if ($missing) { Fail "Missing Python packages: $missing`n         Install with:  `"$Python`" -m pip install -r requirements.txt" }
Ok "Python packages (fastapi, uvicorn, ultralytics, torch, opencv, ...)"

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { Fail "Node.js not found. Install Node.js 20.9 or newer." }
$nodever = (& node -v).Trim().TrimStart('v')
if ([version]$nodever -lt [version]'20.9.0') { Fail "Node.js $nodever is too old for Next.js 16 (20.9+ needed)." }
Ok "Node.js $nodever"
$nextBin = Join-Path $Root 'node_modules\next\dist\bin\next'
if (-not (Test-Path $nextBin)) { Fail "node_modules missing. Run once:  npm install   (in $Root)" }
Ok "node_modules present"

# same rule as backend/core/config.py: LM_MODELS_DIR (process env, then .env), default models\, relative to the project
$ModelsDir = [Environment]::GetEnvironmentVariable('LM_MODELS_DIR')
if (-not $ModelsDir -and (Test-Path (Join-Path $Root '.env'))) {
  $m = Select-String -Path (Join-Path $Root '.env') -Pattern '^\s*LM_MODELS_DIR\s*=\s*(\S+)' | Select-Object -First 1
  if ($m) { $ModelsDir = $m.Matches[0].Groups[1].Value.Trim() }
}
if (-not $ModelsDir) { $ModelsDir = 'models' }
if (-not [IO.Path]::IsPathRooted($ModelsDir)) { $ModelsDir = Join-Path $Root $ModelsDir }
$YoloDir = Join-Path $ModelsDir 'litter_yolo'
$models = @(Get-ChildItem $YoloDir -Directory -ErrorAction SilentlyContinue |
  Where-Object { (Test-Path (Join-Path $_.FullName 'best.pt')) -and (Test-Path (Join-Path $_.FullName 'model_meta.json')) })
if ($models.Count -eq 0) { Fail "No model weights found in $YoloDir\<version>\best.pt (with model_meta.json). Restore v3_store_cctv or fix LM_MODELS_DIR (see deploy\TROUBLESHOOTING.md)." }
Ok ("Model weights found: " + (($models | ForEach-Object Name) -join ', ') + "  (the app loads the version marked active)")

$envFile = Join-Path $Root '.env'
if (-not (Test-Path $envFile)) {
  Warn ".env not found: the API runs with defaults and admin functions are disabled. Create it from .env.example."
} else {
  $tok = Select-String -Path $envFile -Pattern '^\s*ADMIN_API_TOKEN\s*=\s*(\S+)' -ErrorAction SilentlyContinue
  if ($tok) { Ok ".env present (admin token set - value not shown)" } else { Warn ".env has no ADMIN_API_TOKEN: admin functions (cameras, settings, model activation) are disabled." }
}

if ($BackendPort -eq $FrontendPort) { Fail "-BackendPort and -FrontendPort must be different (both are $BackendPort)." }

# addresses
$BindHost = '127.0.0.1'
$PublicHost = 'localhost'
if ($Lan) {
  $ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
         Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
         Select-Object -First 1).IPAddress
  if (-not $ip) { Fail "-Lan: could not determine this PC's LAN IPv4 address." }
  $BindHost = '0.0.0.0'; $PublicHost = $ip
  Warn "LAN mode: the demo will be reachable from other PCs on this network at http://${ip}:$FrontendPort (Windows may ask to allow it through the firewall). Do not use on public networks."
}
$ApiUrl = "http://${PublicHost}:$BackendPort"
$UiUrl = "http://${PublicHost}:$FrontendPort"
$origins = @("http://localhost:$FrontendPort", "http://127.0.0.1:$FrontendPort")
if ($Lan) { $origins += "http://${PublicHost}:$FrontendPort" }

# ------------------------------------------------------------------ 2. backend
Step "2/5 Backend API on port $BackendPort"
$backendLog = Join-Path $LogDir 'backend.log'
$backendErr = Join-Path $LogDir 'backend.err.log'
$listener = Get-Listener $BackendPort
if ($listener) {
  if (Test-LdmBackend $BackendPort) {
    Ok "An LDM backend is already running on port $BackendPort (PID $($listener.Id)) - re-using it, not starting a second one."
  } else {
    Fail "Port $BackendPort is used by another program: $($listener.ProcessName) (PID $($listener.Id)). Stop it or use -BackendPort <free port>."
  }
} else {
  $existing = [Environment]::GetEnvironmentVariable('CORS_ORIGINS')
  $fromDotenv = $null
  if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern '^\s*CORS_ORIGINS\s*=\s*(.+)$' | Select-Object -First 1
    if ($m) { $fromDotenv = $m.Matches[0].Groups[1].Value.Trim() }
  }
  $all = @()
  foreach ($s in @($existing, $fromDotenv)) { if ($s) { $all += ($s -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) } }
  $all += $origins
  $env:CORS_ORIGINS = (($all | Select-Object -Unique) -join ',')
  if (-not $EnableCameras) { $env:LM_START_CAMERAS = 'false' }
  $env:PYTHONUNBUFFERED = '1'
  Rotate-Log $backendLog; Rotate-Log $backendErr
  $bp = Start-Hidden $Python @('-m', 'uvicorn', 'backend.main:app', '--host', $BindHost, '--port', "$BackendPort") $backendLog $backendErr
  Save-Pid 'backend' $bp $BackendPort 'uvicorn backend.main:app'
  Info "started backend (PID $($bp.Id)); allowed UI origins: $($env:CORS_ORIGINS)"
  if (-not $EnableCameras) { Info "CCTV cameras will not be started (LM_START_CAMERAS=false for this run)." }
}

Step "3/5 Waiting for GET /api/health"
$deadline = (Get-Date).AddSeconds($BackendTimeoutSec)
$health = $null
while ((Get-Date) -lt $deadline) {
  $health = Get-Json "http://127.0.0.1:$BackendPort/api/health"
  if ($health) { break }
  if ($bp -and $bp.HasExited) { Show-LogTail $backendErr; Fail "The backend stopped during start-up (exit code $($bp.ExitCode)). See $backendErr" }
  Start-Sleep -Seconds 2
}
if (-not $health) { Show-LogTail $backendErr; Fail "No answer from http://127.0.0.1:$BackendPort/api/health within $BackendTimeoutSec s. See $backendErr" }
if ($health.model) {
  Ok "Backend healthy: model $($health.model.version) on $($health.model.device), $($health.model.classes.Count) classes"
} else {
  Warn "Backend is up but NO MODEL is loaded: $($health.model_error)  -> image/video detection will fail. See deploy\TROUBLESHOOTING.md"
}

# CORS: the exact cause of the old 'Backend offline' banner
$cors = $null
try {
  $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/health" -Headers @{ Origin = $origins[0] } -TimeoutSec 5 -UseBasicParsing
  $cors = $r.Headers['Access-Control-Allow-Origin']
} catch { }
if ($cors -ne $origins[0]) {
  Fail ("The running backend does not allow the UI address $($origins[0]) (CORS), so the UI would show 'Backend offline'.`n" +
        "         It was started outside this script with a different CORS_ORIGINS. Stop that backend and run this script again,`n" +
        "         or add $($origins[0]) to CORS_ORIGINS in .env and restart it.")
}
Ok "Backend allows requests from $($origins[0])"

# ------------------------------------------------------------------ 3. frontend
Step "4/5 Web UI on port $FrontendPort"
$feLog = Join-Path $LogDir 'frontend.log'
$feErr = Join-Path $LogDir 'frontend.err.log'
$env:NEXT_PUBLIC_API_URL = $ApiUrl
$env:NEXT_DIST_DIR = '.next-demo'
$env:NEXT_TELEMETRY_DISABLED = '1'
$fl = Get-Listener $FrontendPort
if ($fl) {
  if (Test-LdmFrontend $FrontendPort) {
    Ok "The LDM web UI is already running on port $FrontendPort (PID $($fl.Id)) - re-using it."
  } else {
    Fail "Port $FrontendPort is used by another program: $($fl.ProcessName) (PID $($fl.Id)). Stop it or use -FrontendPort <free port>."
  }
} else {
  if (-not $Dev) {
    $distDir = Join-Path $Root '.next-demo'
    $stampFile = Join-Path $distDir 'ldm-build.json'
    $need = $Rebuild -or -not (Test-Path (Join-Path $distDir 'BUILD_ID')) -or -not (Test-Path $stampFile)
    if (-not $need) {
      $stamp = Get-Content $stampFile -Raw | ConvertFrom-Json
      if ($stamp.api -ne $ApiUrl) { $need = $true; Info "API address changed ($($stamp.api) -> $ApiUrl): rebuilding" }
      $built = (Get-Item (Join-Path $distDir 'BUILD_ID')).LastWriteTime
      $newer = Get-ChildItem -Recurse -File -Path app, frontend, components, lib, public -ErrorAction SilentlyContinue |
               Where-Object { $_.LastWriteTime -gt $built } | Select-Object -First 1
      foreach ($f in @('next.config.mjs', 'package.json', 'postcss.config.mjs')) {
        if ((Test-Path $f) -and (Get-Item $f).LastWriteTime -gt $built) { $newer = Get-Item $f }
      }
      if ($newer) { $need = $true; Info "source changed since the last build ($($newer.Name)): rebuilding" }
    }
    if ($need) {
      Info "building the web UI (first time takes a few minutes; log: $feLog.build)..."
      $buildLog = "$feLog.build"
      Rotate-Log $buildLog
      # next build rewrites tsconfig.json / next-env.d.ts for the demo dist folder: keep the developer copies as they were
      $keep = @{}
      foreach ($f in @('tsconfig.json', 'next-env.d.ts')) { if (Test-Path $f) { $keep[$f] = [IO.File]::ReadAllBytes((Join-Path $Root $f)) } }
      try {
        $b = Start-Hidden $node.Source @($nextBin, 'build') $buildLog "$buildLog.err" -Wait
      } finally {
        foreach ($f in $keep.Keys) { [IO.File]::WriteAllBytes((Join-Path $Root $f), $keep[$f]) }
      }
      if ($b.ExitCode -ne 0) { Show-LogTail $buildLog 40; Show-LogTail "$buildLog.err" 20; Fail "UI build failed (exit $($b.ExitCode)). See $buildLog" }
      @{ api = $ApiUrl; built = (Get-Date).ToString('s') } | ConvertTo-Json | Set-Content -Encoding utf8 $stampFile
      Ok "web UI built"
    } else { Ok "web UI build is up to date" }
    $feArgs = @($nextBin, 'start', '-p', "$FrontendPort", '-H', $BindHost)
  } else {
    $feArgs = @($nextBin, 'dev', '-p', "$FrontendPort", '-H', $BindHost)
  }
  Rotate-Log $feLog; Rotate-Log $feErr
  $fp = Start-Hidden $node.Source $feArgs $feLog $feErr
  Save-Pid 'frontend' $fp $FrontendPort 'next'
  Info "started web UI (PID $($fp.Id), $(if ($Dev) { 'next dev' } else { 'next start' }))"
  $deadline = (Get-Date).AddSeconds($FrontendTimeoutSec)
  $up = $false
  while ((Get-Date) -lt $deadline) {
    if (Test-LdmFrontend $FrontendPort) { $up = $true; break }
    if ($fp.HasExited) { Show-LogTail $feErr; Show-LogTail $feLog; Fail "The web UI stopped during start-up (exit $($fp.ExitCode)). See $feLog" }
    Start-Sleep -Seconds 2
  }
  if (-not $up) { Show-LogTail $feLog; Fail "The web UI did not answer on port $FrontendPort within $FrontendTimeoutSec s. See $feLog" }
  Ok "web UI answering on port $FrontendPort"
}

# ------------------------------------------------------------------ 4. summary
Step "5/5 Ready"
Write-Host ""
Write-Host "  Open:        $UiUrl" -ForegroundColor Green
Write-Host "  API:         $ApiUrl/api/health   (API docs are off; LM_ENABLE_API_DOCS=true enables $ApiUrl/docs locally)"
Write-Host "  Logs:        $LogDir"
Write-Host "  Stop:        powershell -ExecutionPolicy Bypass -File deploy\stop-ldm.ps1"
Write-Host ""
exit 0
