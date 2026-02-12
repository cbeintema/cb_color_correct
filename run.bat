@echo off
setlocal

cd /d %~dp0

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Missing venv python at %PY%
  echo Create it and install deps:
  echo   py -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

"%PY%" main.py
