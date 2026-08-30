@echo off
REM  AI-Generated Image Detector -- setup & run helper (Windows)
REM  Double-click this file, or run it from a terminal.
setlocal
cd /d "%~dp0"

REM  Prefer the project venv if it exists, else any python on PATH.
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

where %PY% >nul 2>nul
if errorlevel 1 (
  echo.
  echo   Python was not found. Install Python 3.10-3.12 from python.org
  echo   ^(tick "Add python.exe to PATH" in the installer^) and run this again.
  echo.
  pause
  exit /b 1
)

"%PY%" scripts\menu.py
pause
