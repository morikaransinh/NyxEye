# Demo Commands - Quick Reference

## TL;DR run ```./setup_web.sh && ./web_launch.py``` in terminal.

## Easiest Way to Launch

### Option A: One-Command Launch (Simplest!)
```bash
./main.sh
```

### Option B: Configurable Launch
```bash
# Use default settings
./main.sh

# Customize settings
./main.sh --camera 0 --classes person bottle cup --volume 0.3 --confidence 0.4

# Show help
./main.sh --help
```

### Option C: Web Interface (Most User-Friendly)
```bash
# Set up the local web interface, first time only
./setup_web.sh

# Start the web control panel
./web_launch.py

# Then open your browser to: http://localhost:8080
```

## Quick Start (Recommended) - Alternative Methods

### Option 1: Individual Scripts (Simplest)

```bash
# Run full integration (Detection + Depth + Spatial Audio)
source env/bin/activate && python3 main.py

# Run detection depth integration with specific args
source env/bin/activate && python3 detection_depth_integration.py --camera 0 --verbose --classes person bottle --confidence 0.3
```

### Option 2: Unified Script

Use the `run_tests.sh` script for easy testing:

```bash
# Show menu and usage
./run_tests.sh

# Run full integration test (Detection + Depth + Spatial Audio)
./run_tests.sh full

# Run detection depth integration
./run_tests.sh depth --camera 0 --verbose --classes person bottle --confidence 0.3

# Install/upgrade dependencies
./run_tests.sh deps
```

## Manual Commands (Alternative)

### Full Integration Test (Detection + Depth + Spatial Audio)

```bash
# Default
source env/bin/activate && python3 main.py

# Use camera 1
source env/bin/activate && python3 main.py --camera 1

# Use camera 1 with custom classes and volume
source env/bin/activate && python3 main.py --camera 1 --classes person bottle --volume 0.3 --confidence 0.3
```

### Detection + Depth Only (No Audio)

```bash
# Laptop camera
source env/bin/activate && python3 detection_depth_integration.py --camera 0 --verbose --classes person bottle --confidence 0.3

# Alternative command
source env/bin/activate && python3 depth/detection_depth_stream.py --classes person bottle cup
```

**Note:** External camera (Continuity Camera on iPhone) only works if:
- Connected to the same WiFi, OR
- Plugged into the Mac (more reliable)
- iPhone has sufficient battery level

### Spatial Audio Test Only

```bash
source env/bin/activate && python3 spatial-audio/spatial_audio_simple.py
```

