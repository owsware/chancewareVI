@echo off
setlocal
if "%~1"=="" (
    echo Usage: %~nx0 "D:\path\to\title.exe"
    exit /b 2
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_with_metrics.ps1" -GamePath "%~1"
set "result=%errorlevel%"
pause
exit /b %result%
