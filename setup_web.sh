#!/bin/bash

# Setup script for web interface
# This installs Flask if needed

echo "Setting up web interface..."

if [ ! -d "env" ]; then
    echo "Error: Virtual environment not found."
    exit 1
fi

source env/bin/activate

echo "Installing Flask and psutil..."
pip install flask psutil

echo ""
echo "Setup complete!"
echo ""
echo "To start the web interface, run:"
echo "  ./web_launch.py"
echo ""
echo "Then open your browser to: http://localhost:8080"
echo ""

