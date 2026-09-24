#!/bin/bash

# Project Daredevil - Test Runner Script
# This script activates the virtual environment and runs integration tests

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project directory
if [ ! -d "env" ]; then
    print_error "Virtual environment not found. Please run this script from the Project-Daredevil directory."
    exit 1
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source env/bin/activate

# Function to install dependencies
install_dependencies() {
    print_info "Installing/upgrading dependencies..."
    pip install --upgrade pip
    pip install torch torchvision transformers opencv-python numpy ultralytics PyOpenAL
    print_info "Dependencies installed successfully!"
}

# Function to check if dependencies are installed
check_dependencies() {
    print_info "Checking dependencies..."
    python3 -c "import torch; import cv2; import transformers; import numpy; import ultralytics; import openal" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some dependencies are missing. Would you like to install them? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            install_dependencies
        else
            print_error "Cannot proceed without dependencies."
            exit 1
        fi
    else
        print_info "All dependencies are installed!"
    fi
}

# Parse command line arguments
case "${1:-menu}" in
    "full")
        print_info "Running full integration test..."
        check_dependencies
        python3 main.py
        ;;
    "depth")
        print_info "Running detection depth integration..."
        check_dependencies
        shift  # Remove "depth" from arguments
        python3 detection_depth_integration.py "$@"
        ;;
    "deps")
        print_info "Installing dependencies..."
        install_dependencies
        ;;
    "menu"|*)
        echo ""
        echo "============================================"
        echo "Project Daredevil - Test Runner"
        echo "============================================"
        echo ""
        echo "Usage:"
        echo "  ./run_tests.sh full                        - Run full integration test"
        echo "  ./run_tests.sh depth [options]            - Run detection depth integration"
        echo "  ./run_tests.sh deps                       - Install/upgrade dependencies"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh full"
        echo "  ./run_tests.sh depth --camera 0 --verbose --classes person bottle --confidence 0.3"
        echo ""
        echo "Available options for depth:"
        echo "  --camera <index>          Camera index (default: 0)"
        echo "  --width <width>           Frame width (default: 1280)"
        echo "  --height <height>         Frame height (default: 720)"
        echo "  --model <path>            YOLO model path (default: yolov8n.pt)"
        echo "  --classes <class> ...     Target classes (default: person bottle car)"
        echo "  --confidence <value>     Confidence threshold (default: 0.5)"
        echo "  --depth-model <id>        Depth model ID (default: Intel/dpt-hybrid-midas)"
        echo "  --save-dir <dir>          Directory to save snapshots"
        echo "  --verbose                Enable verbose output"
        echo "  --no-persistence         Disable temporal persistence"
        echo "  --persistence-duration <sec>  Persistence duration (default: 2.0)"
        echo "  --max-missing-frames <n>  Max frames to keep object (default: 5)"
        echo ""
        ;;
esac

print_info "Done!"

