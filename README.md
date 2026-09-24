# NyxEye

**Spatial Audio Blind Assistance**

A computer vision and spatial audio system that translates depth perception into sound — like digital echolocation. It uses a webcam to detect objects, estimate how close they are, and convert that into stereo spatial audio cues (played through headphones) so a user can "hear" their surroundings.

## How it works

1. **Detection + Depth** (`depth/detection_depth_stream.py`) — captures webcam frames, runs YOLO object detection, and estimates each object's relative depth.
2. **Spatial Audio** (`spatial-audio/spatial_audio_simple.py`) — converts an object's screen position and depth into stereo panning + volume/pitch cues.
3. **Integration** (`spatial-audio/integration.py`) — glues detection/depth output into the spatial audio engine, running the full real-time loop.
4. **Entry point** (`main.py`) — parses CLI args and starts the system.

## Installation

### Prerequisites
- Python 3.8+
- A webcam

### Setup

```bash
python -m venv env
source env/bin/activate   # on Windows: env\Scripts\activate
pip install ultralytics opencv-python pygame numpy
```

> YOLO model weights (`yolov8n.pt`) are downloaded automatically by `ultralytics` on first run.

## Usage

```bash
python main.py --camera 0 --classes person bottle cup --volume 0.15
```

| Flag | Description | Default |
|---|---|---|
| `--camera` | Camera index | `0` |
| `--classes` | Target object classes to detect | `person bottle cup` |
| `--volume` | Master audio volume | `0.15` |
| `--confidence` | Detection confidence threshold | `0.5` |
| `--verbose` | Enable verbose output | off |

Press `Ctrl+C` to stop.

## License

MIT — see [LICENSE](LICENSE). Original work by MIT Assistive Technology; modifications and trimming by this repository's maintainer.
