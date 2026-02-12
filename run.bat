@echo off
setlocal

cd /d %~dp0

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Virtual env not found at %PY%
  echo Bootstrapping .venv and installing deps...

  rem Prefer the Python launcher if available
  where py >nul 2>nul
  if %errorlevel%==0 (
    py -3 -m venv .venv
  ) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
      python -m venv .venv
    ) else (
      echo Error: Python not found. Install Python 3 and try again.
      echo.
      pause
      exit /b 1
    )
  )

  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

"%PY%" main.py
