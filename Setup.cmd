@echo off
setlocal
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0Setup-Chanceware.ps1" %*
if errorlevel 1 (
    echo.
    echo Setup stopped. See the message above.
    pause
    exit /b 1
)
