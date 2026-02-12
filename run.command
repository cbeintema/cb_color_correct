#!/bin/bash
set -euo pipefail

# Set encoding to prevent pip compile errors
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

cd "$(dirname "$0")"

PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Virtual env not found at $PY"
  echo "Bootstrapping .venv and installing deps..."

  if command -v python3 >/dev/null 2>&1; then
    PY_BOOTSTRAP="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_BOOTSTRAP="python"
  else
    echo "Error: python3 (or python) not found on PATH."
    echo
    read -r -p "Press Enter to exit..." _
    exit 1
  fi

  "$PY_BOOTSTRAP" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

"$PY" main.py
