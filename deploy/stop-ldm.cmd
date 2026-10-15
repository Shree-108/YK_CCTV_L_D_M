@echo off
REM Double-click to stop the demo processes started by start-ldm.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-ldm.ps1" %*
pause
