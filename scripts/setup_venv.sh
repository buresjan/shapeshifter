#!/usr/bin/env bash
set -euo pipefail

# Use whatever python3 is available
PY=$(command -v python3)
if [ -z "$PY" ]; then
    echo "python3 not found" >&2
    exit 1
fi

$PY -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
