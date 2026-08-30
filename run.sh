#!/usr/bin/env bash
# AI-Generated Image Detector -- setup & run helper (Linux / macOS)
#   bash run.sh
set -e
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python || true)"

if [ -z "$PY" ]; then
  echo
  echo "  Python was not found. Install Python 3.10-3.12 and run this again."
  echo
  exit 1
fi

exec "$PY" scripts/menu.py
