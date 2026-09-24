#!/bin/bash

# Project Daredevil - Main Launch Script
# This script runs the full integration test with predefined settings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Defaults (can be overridden by flags)
CAMERA=1
CLASSES="person bottle"
VOLUME=0.3
CONFIDENCE=0.3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --classes)
            shift
            CLASSES="$@"
            break
            ;;
        --volume)
            VOLUME="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./main.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --camera N          Camera index (default: 1)"
            echo "  --classes A B C     Target classes (default: person bottle)"
            echo "  --volume X          Master volume 0.0-1.0 (default: 0.3)"
            echo "  --confidence Y      Confidence threshold 0.0-1.0 (default: 0.3)"
            echo ""
            echo "Examples:"
            echo "  ./main.sh"
            echo "  ./main.sh --camera 0 --classes person bottle cup --volume 0.3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the project directory
if [ ! -d "env" ]; then
    echo "Error: Virtual environment not found. Please run this script from the Project-Daredevil directory."
    exit 1
fi

# Display banner
echo "============================================"
echo "  Project Daredevil - Starting System"
echo "============================================"
echo ""
print_info "Camera: $CAMERA"
print_info "Classes: $CLASSES"
print_info "Volume: $VOLUME"
print_info "Confidence: $CONFIDENCE"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Activate virtual environment and run
source env/bin/activate && python3 main.py --camera "$CAMERA" --classes $CLASSES --volume "$VOLUME" --confidence "$CONFIDENCE"

