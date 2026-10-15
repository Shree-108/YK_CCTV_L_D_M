@echo off
REM Double-click to start the Yatrikart Litter Detection demo. Arguments are passed on (e.g. -FrontendPort 3010).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-ldm.ps1" %*
pause
