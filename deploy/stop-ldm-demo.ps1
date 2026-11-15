<#
.SYNOPSIS
  Stop the demo instance started by deploy\start-ldm-demo.ps1 and check that production data is unchanged.
  Only the process trees recorded in deploy\run\demo\*.json are stopped; the normal instance is never touched.
#>
[CmdletBinding()]
param([string]$Python = "")
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $PSScriptRoot 'run\demo'
$demoRoot = $null
foreach ($name in 'frontend', 'backend') {
  $f = Join-Path $RunDir "$name.json"
  if (-not (Test-Path $f)) { continue }
  $rec = Get-Content $f -Raw | ConvertFrom-Json
  if ($rec.demo_root) { $demoRoot = $rec.demo_root }
  $l = Get-NetTCPConnection -LocalPort $rec.port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  & taskkill.exe /PID $rec.pid /T /F 2>$null | Out-Null
  if ($l) { Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue }
  Remove-Item $f
  Write-Host "  stopped demo $name (port $($rec.port))"
}
if ($demoRoot -and (Test-Path (Join-Path $demoRoot 'production_before.json'))) {
  if (-not $Python) { $Python = (Get-Command python).Source }
  $after = Join-Path $demoRoot 'production_after.json'
  & $Python (Join-Path $PSScriptRoot 'check_demo_isolation.py') snapshot $after | Out-Null
  & $Python (Join-Path $PSScriptRoot 'check_demo_isolation.py') compare (Join-Path $demoRoot 'production_before.json') $after
  exit $LASTEXITCODE
}
Write-Host "  (no demo fingerprint found - isolation not checked)"
