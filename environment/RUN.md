# Running the Depth Processing Pipeline

This guide explains how to run the complete depth processing pipeline on a video file to extract open spaces and generate spatial audio data.

## Quick Start

```bash
cd environment
python depth_integration.py --video newSample.MOV --run-midas --process-depth
```

This will:
1. Extract frames from your video
2. Estimate depth maps using MiDaS
3. Process depth maps to find open spaces (hallways, doorways, etc.)
4. Calculate azimuth angles for each detected open space
5. Generate spatial audio files for each frame with detected open spaces

## Prerequisites

### Install Dependencies

```bash
pip install numpy opencv-python torch torchvision scipy transformers
```

**Optional (for spatial audio):**
```bash
pip install sounddevice soundfile
```

**Note:** The first time you run with `--run-midas`, the model weights will be downloaded (this may take a few minutes and requires internet connection). Using `--use-huggingface` is recommended for more reliable model loading.

### Video File

You need a video file to process. Supported formats include:
- `.MOV` (iPhone videos)
- `.MP4`
- `.AVI`
- Other formats supported by OpenCV

## Step-by-Step Guide

### Step 1: Extract Frames and Run Depth Estimation

This is the first step that processes your video:

```bash
python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --max-frames 20
```

**What this does:**
- Extracts frames from your video (one every 0.2 seconds by default)
- Runs depth estimation on each frame (using HuggingFace transformers, recommended)
- Saves depth maps as PNG and NPY files in `../data/depth/` and `../data/depth_npy/`

**Options:**
- `--max-frames 20`: Process only the first 20 frames (faster for testing)
- `--sample-every-s 0.5`: Extract frames less frequently (every 0.5 seconds instead of 0.2)
- `--use-huggingface`: Use HuggingFace transformers (recommended, more reliable than torch.hub)
- `--huggingface-model MODEL_ID`: Specify HuggingFace model (default: "Intel/dpt-hybrid-midas")
- `--model DPT_Hybrid`: Use a faster/smaller model with torch.hub (only if not using --use-huggingface)

### Step 2: Process Depth Maps to Find Open Spaces

After depth estimation, process the depth maps to find open spaces:

```bash
python depth_integration.py --video newSample.MOV --process-depth
```

**What this does:**
- Loads depth maps from `../data/depth_npy/`
- Analyzes the bottom third of each depth map
- Finds columns with high depth (indicating open spaces like hallways)
- Calculates azimuth angles (direction from camera center) for each open space

**Output:**
You'll see output like:
```
[3/4] Processing depth maps to extract open spaces...
  Saved visualization: ../data/test_results/frame_000000_open_spaces.png
Processed 20 depth maps
  Frame 0: 3 open spaces at [-15.2°, 0.0°, +18.5°]
  Frame 1: 2 open spaces at [-22.1°, +10.3°]
  ...
```

**Visualization:**
- A visualization of the first frame with green lines/overlay showing detected open space columns is saved to `../data/test_results/frame_000000_open_spaces.png`

### Step 3: Run Complete Pipeline

To do everything in one command (including spatial audio generation):

```bash
python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --process-depth
```

This will:
1. Extract frames
2. Estimate depth maps
3. Process depth maps to find open spaces
4. Generate spatial audio files for each frame

## Complete Example

Here's a complete example processing a video:

```bash
# Navigate to the environment directory
cd environment

# Run the complete pipeline
python depth_integration.py \
    --video newSample.MOV \
    --run-midas \
    --use-huggingface \
    --process-depth \
    --max-frames 30 \
    --sample-every-s 0.2
```

**Expected output:**
```
============================================================
DAREDEVIL - Modular Depth Processing Pipeline
============================================================
Configuration:
  Video: newSample.MOV
  Run MiDaS: True
  Process depth: True
  Spatial audio enabled: True
============================================================

[1/4] Extracting frames...
Extracted 30 frames at 30.0 FPS

[2/4] Running depth estimation...
  Using HuggingFace transformers (more reliable)...
[1/30] wrote ../data/depth/depth_000000.png
[11/30] wrote ../data/depth/depth_000010.png
...
Depth estimation complete

[3/4] Processing depth maps to extract open spaces...
  Saved visualization: ../data/test_results/frame_000000_open_spaces.png
Processed 30 depth maps
  Frame 0: 3 open spaces at [-15.2°, 0.0°, +18.5°]
  Frame 1: 2 open spaces at [-22.1°, +10.3°]
  Frame 2: 4 open spaces at [-30.0°, -10.5°, +5.2°, +25.0°]
  ...

[4/4] Generating spatial audio...
Generated 30 spatial audio files in ../data/audio

============================================================
Pipeline complete!
============================================================
```

## Command Line Options

### Required Arguments

- `--video PATH`: Path to your input video file (required)

### Processing Options

- `--run-midas`: Run depth estimation on extracted frames
- `--use-huggingface`: Use HuggingFace transformers instead of torch.hub (recommended, more reliable)
- `--huggingface-model MODEL_ID`: HuggingFace model ID (default: "Intel/dpt-hybrid-midas")
- `--process-depth`: Process depth maps to extract open spaces, calculate azimuths, and generate spatial audio
- `--model MODEL_TYPE`: MiDaS model to use with torch.hub (default: `DPT_Large`, only if not using --use-huggingface)

### Frame Extraction Options

- `--sample-every-s SECONDS`: Extract one frame every N seconds (default: 0.2)
- `--max-frames N`: Maximum number of frames to process (default: 50)

### Output Directories

- `--frames-dir PATH`: Directory for extracted frames (default: `../data/frames`)
- `--depth-dir PATH`: Directory for depth PNG files (default: `../data/depth`)
- `--depth-npy-dir PATH`: Directory for depth NPY files (default: `../data/depth_npy`)
- `--audio-dir PATH`: Directory for audio output (default: `../data/audio`)

## Understanding the Output

### Frame Extraction

Frames are saved as PNG images in `../data/frames/`:
- `frame_000000.png`, `frame_000001.png`, etc.

### Depth Maps

Depth maps are saved in two formats:
- **PNG files** (`../data/depth/`): 16-bit depth images for visualization
- **NPY files** (`../data/depth_npy/`): Normalized float32 arrays for processing

### Open Space Detection Results

The pipeline outputs:
- **Column indices**: Which columns in the image contain open spaces
- **Azimuth angles**: Direction from camera center in degrees
  - Negative = left of center
  - Zero = straight ahead
  - Positive = right of center
- **Visualization**: First frame with green lines/overlay showing detected open spaces (saved to `../data/test_results/`)

Example: `[-15.2°, 0.0°, +18.5°]` means:
- One open space 15.2° to the left
- One open space straight ahead (0°)
- One open space 18.5° to the right

### Spatial Audio Output

For each frame with detected open spaces, a stereo audio file is generated:
- **Location**: `../data/audio/audio_frame_XXXXXX.wav`
- **Format**: Stereo WAV files (48kHz sample rate)
- **Content**: Filtered white noise positioned in stereo based on azimuth angles
- **Frequency**: Varies by depth (far objects = lower frequency, close objects = higher frequency)
- **Reverb**: Added based on depth (far objects = more reverb)
- **Panning**: Each open space is positioned in stereo based on its azimuth angle

## Tips and Best Practices

### For Faster Processing

1. **Use fewer frames for testing:**
   ```bash
   --max-frames 10
   ```

2. **Extract frames less frequently:**
   ```bash
   --sample-every-s 0.5  # Every 0.5 seconds instead of 0.2
   ```

3. **Use HuggingFace (recommended):**
   ```bash
   --use-huggingface  # More reliable than torch.hub
   ```

### For Better Results

1. **Use more frames:**
   ```bash
   --max-frames 100  # Process more frames for better coverage
   ```

2. **Extract frames more frequently:**
   ```bash
   --sample-every-s 0.1  # Every 0.1 seconds for smoother results
   ```

3. **Use HuggingFace with default model:**
   ```bash
   --use-huggingface  # Recommended for reliability
   ```

### Processing Large Videos

For long videos, process in batches:

```bash
# First batch: frames 0-50
python depth_integration.py --video video.MOV --run-midas --max-frames 50

# Second batch: frames 50-100 (requires modifying the code or using different directories)
# Or process the entire video at once (may take a while)
python depth_integration.py --video video.MOV --run-midas --max-frames 200
```

## Troubleshooting

### "Could not open video" Error

- Verify the video file path is correct
- Check that the video file format is supported by OpenCV
- Try using an absolute path: `--video /full/path/to/video.MOV`

### Model Download Fails

**Error:** `RuntimeError: It looks like there is no internet connection...` or SSL certificate errors

**Solutions:**

1. **Use HuggingFace (recommended, more reliable):**
   ```bash
   # Use HuggingFace transformers instead of torch.hub
   python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --max-frames 1
   ```
   This is more reliable and handles SSL/certificate issues better.

2. **First-time download requires internet:**
   ```bash
   # Make sure you have internet connection, then run:
   python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --max-frames 1
   ```
   The model will be downloaded and cached for future offline use.

3. **Check your internet connection:**
   - Verify you can access the internet: `curl https://huggingface.co`
   - For HuggingFace models, check: `curl https://huggingface.co/Intel/dpt-hybrid-midas`

4. **Verify cache location:**
   ```bash
   # HuggingFace cache location
   ls -la ~/.cache/huggingface/
   ```
   If the cache exists but still fails, try clearing and re-downloading:
   ```bash
   rm -rf ~/.cache/huggingface/
   # Then run again with internet
   ```

5. **If using torch.hub (not recommended):**
   - Cache location: `~/.cache/torch/hub/`
   - After first successful download, you can use offline
   - However, `--use-huggingface` is recommended for better reliability

### Out of Memory Errors

- Reduce `--max-frames` to process fewer frames at once
- Use `--use-huggingface` with a smaller model if available
- Process the video in smaller batches

### No Open Spaces Detected

- This is normal for some frames (e.g., looking at a wall)
- Try adjusting the detection parameters in `depth_processing.py`:
  - `MIN_PROM`: Minimum prominence for peak detection
  - `K_CANDIDATES`: Maximum number of open spaces per frame

## Next Steps

After running the pipeline, you can:

1. **View the depth maps**: Check `../data/depth/` for PNG visualizations
2. **View open space visualization**: Check `../data/test_results/frame_000000_open_spaces.png` for the first frame with detected open spaces highlighted
3. **Listen to spatial audio**: Check `../data/audio/` for WAV files - each file represents one frame's detected open spaces as spatial audio
4. **Analyze the results**: The open space data is printed to console
5. **Use the data**: The processed depth maps, open space data, and spatial audio can be used for:
   - Navigation assistance
   - Obstacle detection
   - Spatial audio feedback
   - Other computer vision applications

## Example Workflows

### Quick Test Run

```bash
# Quick test with just 5 frames
python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --process-depth --max-frames 5
```

### Full Processing

```bash
# Process entire video (or first 100 frames)
python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --process-depth --max-frames 100
```

### Depth Estimation Only

```bash
# Just estimate depth, don't process for open spaces or generate audio
python depth_integration.py --video newSample.MOV --run-midas --use-huggingface --max-frames 50
```

### Process Existing Depth Maps

```bash
# If you already have depth maps, just process them
python depth_integration.py --video newSample.MOV --process-depth
```

## Output Files Location

All output files are saved in the `../data/` directory:

```
data/
├── frames/          # Extracted video frames (PNG)
├── depth/           # Depth maps as PNG images
├── depth_npy/       # Depth maps as NumPy arrays (for processing)
├── audio/           # Spatial audio files (WAV format, stereo)
└── test_results/    # Visualization of detected open spaces (first frame)
```

## Performance Notes

- **First run**: Slower due to model download and initialization
- **Subsequent runs**: Faster as model is cached
- **Processing time**: Approximately 1-3 seconds per frame (depending on hardware)
- **GPU acceleration**: Automatically used if CUDA is available

## Getting Help

If you encounter issues:

1. Check that all dependencies are installed
2. Verify your video file is valid and accessible
3. Check the console output for error messages
4. See `TESTING.md` for unit test examples
5. Review the algorithm notes in the code comments

