@echo off
setlocal
title LiveFace-Studio
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Check if virtual environment exists
if not exist "%~dp0venv\Scripts\python.exe" (
    echo =================================================================
    echo  [LiveFace-Studio] Environment not initialized!
    echo  Please run 'setup.bat' first to configure your environment.
    echo =================================================================
    echo.
    pause
    exit /b 1
)

echo =================================================================
echo                    Launching LiveFace-Studio
echo =================================================================
echo.

call "%~dp0venv\Scripts\python.exe" run.py --execution-provider cuda %*
if errorlevel 1 (
    echo.
    echo [LiveFace-Studio] Process finished or encountered an issue.
    pause
)
