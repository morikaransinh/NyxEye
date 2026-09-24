# Spatial Audio Module

This module provides **real 3D spatial audio** using OpenAL, fully compatible with Apple AirPods spatial audio on macOS. It converts detected object positions and depths into immersive 3D sound that you can hear in actual 3D space.

## Features

- **True 3D Spatial Audio**: Uses OpenAL for real spatial positioning (not fake stereo panning)
- **AirPods Compatible**: Works with Apple's spatial audio feature
- **Calm White Noise**: Gentle, ambient sound for object localization
- **Real-time Updates**: Smooth audio positioning at 30-60 Hz
- **Depth-based Distance**: Objects sound closer or farther based on depth estimation
- **Multi-object Support**: Handles multiple simultaneous audio sources (up to 10)

## Installation

### Prerequisites

1. **macOS** (required for AirPods spatial audio)

2. **Install OpenAL-Soft**:
```bash
brew install openal-soft
```

3. **Install PyOpenAL**:
```bash
pip install PyOpenAL
```

### Full Setup

**IMPORTANT**: Always activate the virtual environment first!

```bash
# Activate virtual environment (required!)
cd ../Project-Daredevil
source env/bin/activate

# Install OpenAL
brew install openal-soft

# Install Python dependencies (in venv)
pip install PyOpenAL numpy
```

**See [SETUP.md](SETUP.md) for detailed setup instructions and troubleshooting.**

## Usage

### Quick Test

Test the spatial audio engine standalone:

```bash
cd spatial-audio
python3 index.py
```

You should hear calm white noise positioned in 3D space at different locations.

### Integrated with Detection + Depth

Run the full system with detection, depth, and spatial audio:

```bash
python3 spatial-audio/integration.py
```

**Options:**
```bash
# Specify target objects
python3 spatial-audio/integration.py --classes person bottle cup

# Adjust volume (0.0 to 1.0)
python3 spatial-audio/integration.py --volume 0.3

# Enable verbose output
python3 spatial-audio/integration.py --verbose

# Use specific camera
python3 spatial-audio/integration.py --camera 1
```

### Programmatic Usage

```python
from spatial_audio.index import SpatialAudioEngine

# Initialize engine
engine = SpatialAudioEngine(max_sources=10, master_volume=0.2)

# Start the engine
engine.start()

# Update object positions
# x: 0.0 (left) to 1.0 (right)
# y: 0.0 (top) to 1.0 (bottom)
# depth: 0.0 (near) to 1.0 (far)
engine.update_object('person_1', x=0.5, y=0.3, depth=0.7)

# Update another object
engine.update_object('bottle_2', x=0.8, y=0.2, depth=0.9)

# Stop when done
engine.stop()
```

## How It Works

### 3D Positioning

The system converts 2D screen positions to 3D spatial coordinates:

1. **X Position** (left-right):
   - 0.0 = extreme left
   - 0.5 = center
   - 1.0 = extreme right

2. **Y Position** (top-bottom):
   - 0.0 = top of screen
   - 0.5 = middle
   - 1.0 = bottom

3. **Depth** (near-far):
   - 0.0 = close to viewer
   - 0.5 = medium distance
   - 1.0 = far away

### Audio Properties

- **Sound**: Calm white noise (low-pass filtered for gentleness)
- **Volume**: Automatically adjusted based on detection confidence
- **Positioning**: Updates at 30-60 Hz for smooth movement
- **Attenuation**: Distance-based volume falloff

## AirPods Setup

To experience true 3D spatial audio with AirPods:

1. **Connect AirPods** to your Mac
2. **Enable Spatial Audio** in Bluetooth settings
3. **Run the application**
4. **Move objects** in front of the camera to hear them move in 3D space

### Testing Spatial Audio

1. Run the test:
   ```bash
   python3 spatial-audio/index.py
   ```

2. You should hear white noise at different positions:
   - Left side of the "screen"
   - Right side
   - Center, moving back and forth

3. Objects should sound:
   - **Closer** when depth is near (low depth value)
   - **Farther** when depth is far (high depth value)
   - **Positioned** based on their X/Y location on screen

## Integration with Detection

The `integration.py` module automatically connects object detection and depth estimation to spatial audio:

```python
from spatial_audio.integration import IntegratedSpatialAudioSystem

system = IntegratedSpatialAudioSystem(
    target_classes=['person', 'bottle', 'cup'],
    master_volume=0.2
)
system.run()
```

## Troubleshooting

### "PyOpenAL not available"

Install PyOpenAL:
```bash
pip install PyOpenAL
```

### "Failed to open audio device"

1. Check that no other application is using the audio device
2. Try restarting the audio system:
   ```bash
   sudo killall coreaudiod
   ```
3. Make sure AirPods are connected to your Mac

### No 3D spatial audio effect

1. Check that **Spatial Audio is enabled** in macOS Bluetooth settings
2. Make sure you're using **AirPods** (not all headphones support spatial audio)
3. Try adjusting the volume: `--volume 0.3`

### Audio lag or stuttering

1. Reduce the number of simultaneous sources: `--max-sources 5`
2. Lower the audio tick rate in the code
3. Check system performance with Activity Monitor

## API Reference

### SpatialAudioEngine

#### Constructor
```python
SpatialAudioEngine(
    max_sources=10,          # Max simultaneous audio sources
    master_volume=0.3,       # Master volume (0.0-1.0)
    audio_tick_rate=60.0,    # Update rate in Hz
)
```

#### Methods

**update_object**(object_id, x, y, depth, volume=None, active=True)
- Update object's position in 3D space
- x, y: Screen position (0.0-1.0)
- depth: Depth value (0.0=near, 1.0=far)
- volume: Optional volume override

**remove_object**(object_id)
- Remove an object from spatial audio

**start**()
- Start the audio engine

**stop**()
- Stop the audio engine

**get_stats**()
- Get engine statistics

## Examples

### Single Object Tracking

```python
engine = SpatialAudioEngine()
engine.start()

# Update object position in real-time
for i in range(100):
    x = 0.3 + (i / 100.0) * 0.4  # Move from left to right
    engine.update_object('moving_object', x=x, y=0.5, depth=0.5)
    time.sleep(0.1)

engine.stop()
```

### Multiple Objects

```python
engine = SpatialAudioEngine(max_sources=5)
engine.start()

# Create multiple objects at different positions
engine.update_object('person', x=0.3, y=0.2, depth=0.4)
engine.update_object('bottle', x=0.7, y=0.3, depth=0.8)
engine.update_object('cup', x=0.5, y=0.5, depth=0.6)

# Let them play
time.sleep(10)

engine.stop()
```

## License

MIT - Part of Project Daredevil
