# Testing Guide

This document explains how to launch and test the entire depth processing program.

## Overview

The depth processing pipeline has been split into three modular components:

1. **`depth_integration.py`** - Main integration module that loads video, extracts frames, and coordinates the pipeline
2. **`depth_processing.py`** - Processes depth maps to extract columns, depths, and azimuth angles
3. **`spatial_audio.py`** - Spatial audio output module (enabled) - generates stereo audio from detected open spaces

## Prerequisites

### Required Dependencies

```bash
pip install numpy opencv-python torch torchvision scipy
```

### Optional Dependencies

- For MiDaS depth estimation: PyTorch with torch.hub support or HuggingFace transformers
- For spatial audio: `sounddevice` (for playback) and `soundfile` (for saving WAV files)

## Running the Main Pipeline

### Basic Usage

```bash
cd environment
python depth_integration.py --video /path/to/video.mov --run-midas --process-depth
```

### Command Line Arguments

- `--video` (required): Path to input video file
- `--run-midas`: Run MiDaS depth estimation on extracted frames
- `--process-depth`: Process depth maps to extract open spaces
- `--frames-dir`: Directory for extracted frames (default: `../data/frames`)
- `--depth-dir`: Directory for depth PNG files (default: `../data/depth`)
- `--depth-npy-dir`: Directory for depth NPY files (default: `../data/depth_npy`)
- `--audio-dir`: Directory for audio output (default: `../data/audio`)
- `--sample-every-s`: Extract one frame every N seconds (default: 0.2)
- `--max-frames`: Maximum number of frames to process (default: 50)
- `--model`: MiDaS model type (default: "DPT_Large") - only used with torch.hub
- `--use-huggingface`: Use HuggingFace transformers instead of torch.hub (recommended, more reliable)
- `--huggingface-model`: HuggingFace model ID (default: "Intel/dpt-hybrid-midas")

### Example Workflow

1. **Extract frames and run depth estimation:**
   ```bash
   python depth_integration.py --video newSample.MOV --run-midas --max-frames 10
   ```

2. **Process existing depth maps:**
   ```bash
   python depth_integration.py --video newSample.MOV --process-depth
   ```

3. **Full pipeline (with spatial audio):**
   ```bash
   python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --process-depth --max-frames 20
   ```

## Running Unit Tests

### Test All Modules

Run all unit tests:

```bash
cd environment
python -m pytest test_depth_processing.py test_spatial_audio.py test_depth_integration.py -v
```

Or use unittest:

```bash
python test_depth_processing.py
python test_spatial_audio.py
python test_depth_integration.py
```

### Individual Test Suites

**Test depth processing module:**
```bash
python test_depth_processing.py
```

This tests:
- `detect_open_spaces`: Column depth extraction from bottom 1/3
- `find_open_space_columns`: Peak detection in depth signal
- `columns_to_azimuths`: Angle calculation from column indices
- `process_depth_map`: Complete pipeline integration

**Test spatial audio module:**
```bash
python test_spatial_audio.py
```

This tests:
- Enabled state and audio generation
- Azimuth to pan conversion
- Depth to frequency/gain mapping
- Stereo panning
- Filtered noise generation
- Reverb addition
- Function signatures and edge cases

**Test integration module:**
```bash
python test_depth_integration.py
```

This tests:
- Frame extraction from video
- Depth map processing coordination
- Complete pipeline integration
- Spatial audio integration with depth processing

### Running Specific Test Cases

To run a specific test class:

```bash
python test_depth_processing.py TestDetectOpenSpaces
```

To run a specific test method:

```bash
python test_depth_processing.py TestDetectOpenSpaces.test_basic_extraction
```

## Testing Individual Components

### Testing Depth Processing Directly

You can import and use the depth processing functions directly:

```python
from depth_processing import process_depth_map, load_depth_from_file
import numpy as np

# Create a test depth map
depth_map = np.random.rand(300, 400).astype(np.float32)

# Process it
result = process_depth_map(depth_map)

print(f"Open space columns: {result['open_space_columns']}")
print(f"Azimuths (degrees): {result['azimuths_deg']}")
```

### Testing with Existing Depth Maps

If you have existing depth maps in `../data/depth_npy/`:

```python
from depth_processing import load_depth_from_file, process_depth_map

depth_map = load_depth_from_file("../data/depth_npy/depth_000000.npy")
result = process_depth_map(depth_map)

print(f"Found {len(result['open_space_columns'])} open spaces")
for col, az in zip(result['open_space_columns'], result['azimuths_deg']):
    print(f"  Column {col}: {az:.1f} degrees")
```

## Expected Output

### Pipeline Output

When running the full pipeline, you should see:

```
============================================================
DAREDEVIL - Modular Depth Processing Pipeline
============================================================
Configuration:
  Video: /path/to/video.mov
  Run MiDaS: True
  Process depth: True
  Spatial audio enabled: True
============================================================

[1/4] Extracting frames...
Extracted 10 frames at 30.0 FPS

[2/4] Running depth estimation...
  Using HuggingFace transformers (more reliable)...
[1/10] wrote ../data/depth/depth_000000.png
...
Depth estimation complete

[3/4] Processing depth maps to extract open spaces...
  Saved visualization: ../data/test_results/frame_000000_open_spaces.png
Processed 10 depth maps
  Frame 0: 3 open spaces at [-15.2°, 0.0°, +18.5°]
  Frame 1: 2 open spaces at [-22.1°, +10.3°]
  ...

[4/4] Generating spatial audio...
Generated 10 spatial audio files in ../data/audio

============================================================
Pipeline complete!
============================================================
```

### Test Output

When running tests, you should see:

```
test_basic_extraction (__main__.TestDetectOpenSpaces) ... ok
test_bottom_third_extraction (__main__.TestDetectOpenSpaces) ... ok
test_column_averaging (__main__.TestDetectOpenSpaces) ... ok
...
----------------------------------------------------------------------
Ran 25 tests in 2.345s

OK
```

## Troubleshooting

### Common Issues

1. **MiDaS model download fails:**
   - Check internet connection
   - Try a different model: `--model DPT_Hybrid`

2. **No depth maps found:**
   - Ensure `--run-midas` was used first
   - Check that depth files exist in `../data/depth_npy/`

3. **Video cannot be opened:**
   - Verify video path is correct
   - Check video file format (MOV, MP4, AVI supported)

4. **Import errors:**
   - Ensure you're running from the `environment/` directory
   - Check that all dependencies are installed

### Debug Mode

For more verbose output, you can modify the print statements in the code or add debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Test Coverage

The unit tests cover:

- **Depth Processing:**
  - Column depth extraction (bottom 1/3 averaging)
  - Peak detection with prominence thresholding
  - Azimuth angle calculation
  - Edge cases (empty maps, uniform depth, etc.)

- **Spatial Audio:**
  - Enabled state and audio generation
  - Azimuth to pan conversion
  - Depth to frequency/gain mapping
  - Stereo panning algorithms
  - Filtered noise generation
  - Reverb addition
  - Function signatures and edge cases

- **Integration:**
  - Frame extraction from video
  - Depth map processing coordination
  - Complete pipeline flow

## Spatial Audio Output

Spatial audio is now enabled and generates stereo audio files for each frame with detected open spaces:

- **Audio files**: Saved to `../data/audio/` as `audio_frame_XXXXXX.wav`
- **Format**: Stereo WAV files (48kHz sample rate)
- **Content**: Filtered white noise positioned in stereo based on azimuth angles
- **Frequency**: Varies by depth (far = lower frequency, close = higher frequency)
- **Reverb**: Added based on depth (far = more reverb)

To test spatial audio generation:
```python
from spatial_audio import generate_spatial_audio
import numpy as np

open_space_columns = [30, 50, 70]
column_depths = np.ones(100) * 0.5
azimuths_deg = [-10.0, 0.0, 10.0]

audio = generate_spatial_audio(
    open_space_columns,
    column_depths,
    azimuths_deg,
    duration_s=2.0
)
```

## Additional Resources

- Algorithm notes are in the docstrings of each module
- Depth processing algorithm: See `depth_processing.py` comments
- Spatial audio algorithm: See `spatial_audio.py` comments

