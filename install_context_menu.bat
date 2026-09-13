@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run.bat first to set up the app, then run this installer again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" windows_context_menu.py %*
pause
