#!/bin/bash
# Quick script to run spatial audio test with proper environment

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
if [ -f "$PROJECT_DIR/env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/env/bin/activate"
else
    echo "Error: Virtual environment not found at $PROJECT_DIR/env"
    echo "Please create it with: python3 -m venv env"
    exit 1
fi

# Check if PyOpenAL is installed
python3 -c "import openal" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyOpenAL not found. Installing..."
    pip install PyOpenAL
fi

# Run the test
echo "Running spatial audio test..."
python3 "$SCRIPT_DIR/index.py"
