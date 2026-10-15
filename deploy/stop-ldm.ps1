<#
.SYNOPSIS
  Stop the LDM demo processes that deploy\start-ldm.ps1 started (web UI first, then the backend API).

.DESCRIPTION
  Only processes recorded in deploy\run\*.json are stopped, and only if their command line still matches
  (a PID re-used by Windows for another program is never touched). A `npm run dev` or backend you started
  yourself in a terminal is left alone: stop it with Ctrl+C in that terminal.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File deploy\stop-ldm.ps1
#>
$ErrorActionPreference = 'Stop'
$RunDir = Join-Path $PSScriptRoot 'run'
$stopped = 0
foreach ($name in @('frontend', 'backend')) {
  $f = Join-Path $RunDir "$name.json"
  if (-not (Test-Path $f)) { Write-Host "  $name : not started by start-ldm.ps1 (nothing to stop)"; continue }
  $rec = Get-Content $f -Raw | ConvertFrom-Json
  $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($rec.pid)" -ErrorAction SilentlyContinue
  $pattern = if ($name -eq 'backend') { 'uvicorn' } else { 'next' }
  if (-not $p) {
    Write-Host "  $name : PID $($rec.pid) is no longer running"
  } elseif ($p.CommandLine -notmatch $pattern) {
    Write-Host "  $name : PID $($rec.pid) now belongs to another program ($($p.Name)) - not touched" -ForegroundColor Yellow
  } else {
    # /T stops the whole process tree (Next.js and uvicorn start worker processes)
    & taskkill.exe /PID $rec.pid /T /F | Out-Null
    Write-Host "  $name : stopped (PID $($rec.pid), port $($rec.port))" -ForegroundColor Green
    $stopped++
  }
  Remove-Item $f -Force
}
foreach ($port in @(8000, 3010)) {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) {
    $pr = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    Write-Host "  note: port $port is still in use by $($pr.ProcessName) (PID $($c.OwningProcess)) - not started by this script, left running"
  }
}
Write-Host "done ($stopped process tree(s) stopped)"
