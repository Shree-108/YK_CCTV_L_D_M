@echo off
rem Extract product line items from every image in invoices\input into invoices\output\products.csv.
rem Extra arguments are passed through, e.g.:
rem   run_invoice_extraction.bat --input "D:\Invoices\March" --output "D:\Invoices\March\out"
rem   run_invoice_extraction.bat --reprocess
setlocal
cd /d "%~dp0"
set "VPY=.venv-invoice\Scripts\python.exe"
if not exist "%VPY%" (
    where py >nul 2>nul && (set "VPY=py") || (set "VPY=python")
)
"%VPY%" -m invoice_extractor %*
set "RC=%ERRORLEVEL%"
if not "%NOPAUSE%"=="1" pause
exit /b %RC%
