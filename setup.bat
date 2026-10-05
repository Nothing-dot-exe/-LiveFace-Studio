@echo off
setlocal enabledelayedexpansion
title LiveFace-Studio — Setup
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo =================================================================
echo           LiveFace-Studio — Automated Environment Setup
echo =================================================================
echo.

:: 1. Check for Python
echo [1/5] Checking Python installation...
set "PY_CMD="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0venv\Scripts\python.exe"
    echo [OK] Existing virtual environment found.
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            set "PY_CMD=py"
        )
    )
)

if "%PY_CMD%"=="" (
    echo.
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo NOTE: Ensure you check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Create virtual environment if it doesn't exist
if not exist "%~dp0venv\Scripts\python.exe" (
    echo.
    echo [2/5] Creating Python virtual environment in .\venv...
    %PY_CMD% -m venv "%~dp0venv"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [2/5] Virtual environment verified.
)

set "VENV_PY=%~dp0venv\Scripts\python.exe"

:: 3. Upgrade pip tooling
echo.
echo [3/5] Updating pip and core build tools...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet

:: 4. Install dependencies from requirements.txt
echo.
echo [4/5] Installing project dependencies (this may take a few minutes)...
"%VENV_PY%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have had issues during pip install. Continuing to verification...
)

:: 5. Run automated model download & environment verification
echo.
echo [5/5] Downloading required AI models and verifying accelerator...
"%VENV_PY%" setup_env.py
if %errorlevel% neq 0 (
    echo.
    echo [WARN] Model setup or verification encountered an issue.
    echo You may re-run setup.bat or launch run.bat directly.
    echo.
    pause
    exit /b 1
)

echo.
echo =================================================================
echo   [SUCCESS] Setup finished! You are ready to use LiveFace-Studio.
echo   To launch the app at any time, simply double-click 'run.bat'.
echo =================================================================
echo.
pause
