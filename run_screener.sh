#!/usr/bin/env bash
# Mac / Linux launcher. In Terminal:   bash run_screener.sh
# First run: creates a private Python environment, installs packages,
# downloads ~3 GB of SEC data, then screens ~2,000 companies (60-90 min).
#
# Options:  bash run_screener.sh --test           quick 5-min install check
#           bash run_screener.sh --refresh-data   re-pull the SEC data first
set -e
cd "$(dirname "$0")"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python was not found. Install Python 3.11+ from https://www.python.org/downloads/ and re-run."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating Python environment (one time)..."
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python scripts/run_screener.py "$@"
