#!/bin/bash
# Zero-1-to-3 Novel View Synthesis App
# Simple CLI launcher

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Starting Zero-1-to-3 Novel View Synthesis..."
python app.py
