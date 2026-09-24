# NyxEye

> **Attribution:** This project is based on [Project Daredevil](https://github.com/MIT-Assistive-Technology/Project-Daredevil) by MIT Assistive Technology, used under the MIT License. See [LICENSE](LICENSE) for the original copyright notice.

**Spatial Audio Blind Assistance**

A computer vision and spatial audio system that translates depth perception into sound — like digital echolocation. Project Daredevil explores how consumer devices (iPhone, AirPods) can provide affordable, real-time spatial audio feedback to blind and low-vision users.

## ➡️ The app: [`ios/`](ios/README.md)

The finished deliverable is a **native iOS app** (iPhone Pro LiDAR + AirPods spatial audio). It reimplements the pipeline with real metric depth, correct depth→audio polarity, temporal smoothing, hearing protection, and a VoiceOver-first UI. See [ios/README.md](ios/README.md) for build/deploy instructions.

Everything below documents the original Python prototype, kept as reference.

## [Design Review Slides](https://docs.google.com/presentation/d/1qIadIwMdu863MYsWHZKzjytCuk-a4r40REGFNMkRoNw/edit?usp=sharing)

## Overview

Our goal is to create a proof-of-concept system that translates depth perception into sound—like digital echolocation—to enhance spatial awareness in everyday environments.

### System Modules

1. **Camera Module** - Handles video streaming from various sources including phone cameras
2. **Detection Module** - Object detection and tracking (YOLO-based)
3. **Depth Module** - Monocular depth estimation using Hugging Face DPT models
4. **Spatial Audio Module** - Converts object positions and depths to spatial audio

### Key Features

- **Phone Camera Support**: Stream from iPhone/Android via various methods
- **Real-time Object Detection**: Live tracking of objects (focused on water bottles)
- **Monocular Depth Estimation**: Depth estimation without depth sensors
- **Spatial Audio**: Stereo panning based on object position and depth
- **Modular Design**: Each component can be used independently
- **Apple Silicon Optimized**: Uses MPS acceleration for depth processing

## Project Motivations

- Make spatial awareness assistance affordable and portable
- Provide subtle, continuous cues (ambient whooshes, localized pitch shifts) instead of overwhelming object-to-sound mappings
- Enable detection of key social and safety cues:
  - An approaching handshake
  - Objects moving into one's path
  - "The last 10 feet" problem
  - Ambient depth shifts in hallways or open spaces

## Success Criteria

Our prototype should demonstrate:

- Real-time object detection, depth estimation and directional audio
- Clear, intuitive audio depth cues for our codesigners
- Smooth integration of all system components

Future goals include:

- Voice command integration
- iOS native app (LiDAR + ARKit + AirPods)
- Continuous ambient spatial audio
- Lightweight deployment on AI/AR glasses (Meta Ray-Ban, open source SDK solutions)

## Current Limitations

- **Safety**: Audio feedback must not interfere with natural hearing
- **Hardware**: Standard webcams have limited field of view
- **Learning Curve**: Users need time to interpret depth-based audio cues

## Installation

### Prerequisites

- Python 3.8+
- macOS (for Apple Silicon optimization)
- Camera access permissions

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MIT-Assistive-Technology/Project-Daredevil.git
cd Project-Daredevil
```

2. Create and activate virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision transformers opencv-python numpy ultralytics PyOpenAL pygame
```

4. **For spatial audio (macOS only)**, install OpenAL:
```bash
brew install openal-soft
```

## Quick Start Commands

### ⚡ Quickest Way to Launch

```bash
# One command - uses your default settings (camera 1, person bottle)
./main.sh

# Or use the configurable version
./main.sh --camera 1 --classes person bottle --volume 0.3 --confidence 0.3

# Or use the web interface (most user-friendly)
./setup_web.sh  # First time only
./web_launch.py  # Then open http://localhost:8080
```

### Run Everything (Recommended)
```bash
# Full system with detection, depth, and spatial audio
source env/bin/activate && python3 main.py

# Just depth + detection (no audio)
source env/bin/activate && python3 depth/detection_depth_stream.py

# Live depth streaming only
source env/bin/activate && python3 depth/depth_stream.py
```

### Test Everything Works
```bash
# List available cameras
python3 camera/index.py

# Test depth processing
python3 depth/test_depth_integration.py
```

### Individual Components
```bash
# Test depth processing only
python3 depth/test_depth_integration.py

# Live depth streaming
python3 depth/depth_stream.py
```

## Repository Structure

```bash
Project-Daredevil/
├── camera/           # Camera streaming module
├── detection/        # Object detection & tracking
├── depth/           # Depth estimation module
├── spatial-audio/   # Spatial audio processing
├── main.py          # Central Python entrypoint
├── main.sh          # Single launch script (configurable)
├── web_launch.py    # Web control panel
├── env/             # Virtual environment
└── README.md        # This file
```

## Performance

- **Depth Processing**: ~12.8 FPS on Apple Silicon
- **Camera Streaming**: 30 FPS
- **Memory Usage**: ~1GB for depth model
- **Latency**: ~78ms per frame for depth processing

## Development

Project progress is tracked in our [GitHub Project Board](https://github.com/orgs/MIT-Assistive-Technology/projects/1/views/1).

### Testing

Run integration tests:
```bash
python3 depth/test_depth_integration.py
```

---

## Technical Documentation

### Module Documentation

Monocular depth estimation using Hugging Face DPT models.

**Key Features**:
- Offline-capable depth estimation
- Bounding box ROI processing
- Multiple normalization methods
- Integration-ready interface

**Usage**:
```python
from depth import DepthProcessor, create_depth_processor

processor = create_depth_processor()
result = processor.get_depth_for_spatial_audio(frame, bbox)
depth_value = result['normalized_depth']  # 0.0 to 1.0
```

#### Detection Module (`detection/`)

Object detection and tracking (YOLO-based).

**Status**: Completed

**Focus**: Water bottle detection for demo purposes

#### Spatial Audio Module (`spatial-audio/`)

Real 3D spatial audio using OpenAL, compatible with Apple AirPods.

**Status**: ✅ Completed

**Features**:
- True 3D spatial audio using OpenAL
- AirPods spatial audio compatible
- Calm white noise for object localization
- Real-time 3D positioning (30-60 Hz)
- Depth-based distance rendering
- Multi-object support (up to 10 simultaneous sources)

**Usage**:
```bash
# Test spatial audio standalone
python3 spatial-audio/index.py

# Run full integration with detection + depth + audio
python3 spatial-audio/integration.py
```

See [spatial-audio/README.md](spatial-audio/README.md) for full documentation.

### Coordinate System

The system uses a universal coordinate system:

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (0 to frame_width)
- **Y-axis**: Top to bottom (0 to frame_height)
- **Bounding Box Format**: [x1, y1, x2, y2]
- **Depth Values**: Normalized to 0.0-1.0 range

## Phone Camera Setup

### Method 1: DroidCam (Android)

1. **Install DroidCam**:
   - Download DroidCam from Google Play Store
   - Install DroidCam Server on Mac from [https://droidcam.com](https://droidcam.com)

2. **Setup**:
   - Connect via USB or WiFi
   - DroidCam will appear as camera device

### Method 2: IP Webcam (Android)

1. **Install IP Webcam**:
   - Download IP Webcam from Google Play Store

2. **Setup**:
   - Start IP Webcam on phone
   - Note the IP address (e.g., 192.168.1.100:8080)
   - Use IP camera streaming in the camera module

### Method 3: Continuity Camera (iPhone)

1. **Enable Continuity Camera**:
   - iPhone must be signed into same Apple ID as Mac
   - iPhone must be nearby and unlocked
   - For a more reliable connection, can directly connect to Mac via wire AND with permission granted

2. **Usage**:
   - Continuity Camera appears as camera index 1
   - Automatically detected by the camera module

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
**Solution**: 
1. Check camera permissions in System Preferences
2. Try different camera indices (0, 1, 2)
3. Restart camera applications

**Problem**: Phone camera not detected
**Solution**:
1. Ensure DroidCam is running on phone
2. Check USB/WiFi connection
3. Try IP camera method

### Depth Processing Issues

**Problem**: Model loading fails
**Solution**:
1. Check internet connection for first download
2. Verify sufficient disk space
3. Check PyTorch installation

**Problem**: Low performance
**Solution**:
1. Use smaller frame sizes
2. Process only necessary bounding boxes
3. Use median method for depth calculation

### Permission Issues

**Problem**: Camera access denied
**Solution**:
1. Go to System Preferences > Security & Privacy > Camera
2. Enable camera access for Terminal/Python
3. Restart the application

---

## Credits

- Hugging Face - DPT depth estimation models
- Ultralytics - YOLO object detection
- OpenCV - Computer vision framework
- Apple - Continuity Camera API

**MIT Assistive Technology Club**  
Fall 2025 Project