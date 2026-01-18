#!/bin/bash
# Zero-1-to-3 Novel View Synthesis App
# Simple CLI launcher

cd "$(dirname "$0")"

PYTHON_BIN=""
for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3 is required but not found in PATH."
    exit 1
fi

PY_VERSION="$($PYTHON_BIN --version 2>&1)"
if echo "$PY_VERSION" | grep -q "3.13"; then
    echo "WARNING: Python 3.13 is not fully supported by PyTorch on macOS."
    echo "If you encounter crashes, install Python 3.11 and re-run this script."
fi

if [ "$(uname)" = "Darwin" ] && [ -z "$ZERO123_DEVICE" ]; then
    export ZERO123_DEVICE=cpu
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    "$PYTHON_BIN" -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Starting Zero-1-to-3 Novel View Synthesis..."
python app.py
