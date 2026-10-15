@echo off
rem One-time setup for the invoice extractor: creates .venv-invoice and installs the OCR engine.
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

if not exist ".venv-invoice\Scripts\python.exe" (
    echo Creating virtual environment .venv-invoice ...
    %PY% -m venv .venv-invoice || goto :fail
)
set "VPY=.venv-invoice\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r invoice_extractor\requirements.txt || goto :fail
if not exist "invoices\input" mkdir "invoices\input"
if not exist "invoices\output" mkdir "invoices\output"
"%VPY%" -m invoice_extractor.samples || goto :fail
echo.
echo Setup complete.
echo   1. Copy invoice images (JPG/JPEG/PNG) into: %CD%\invoices\input
echo   2. Double-click run_invoice_extraction.bat
echo   Sample invoices for a test run are in invoices\samples
if not "%NOPAUSE%"=="1" pause
exit /b 0

:fail
echo.
echo Setup FAILED - see the messages above. Check that Python 3.10+ is installed and on PATH.
if not "%NOPAUSE%"=="1" pause
exit /b 1
