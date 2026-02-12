#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python at $PY"
  echo "Create it and install deps:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  echo
  read -r -p "Press Enter to exit..." _
  exit 1
fi

"$PY" main.py
