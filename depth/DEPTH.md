# Depth Module Documentation

This module provides advanced monocular depth estimation capabilities for the Project Daredevil spatial audio system. It processes camera frames and bounding boxes to generate comprehensive depth information optimized for spatial audio applications.

## Overview

The depth module consists of several components:

- **DepthProcessor**: Core depth estimation and processing
- **EnhancedDepthProcessor**: Advanced depth processing with temporal tracking, quality assessment, and spatial audio optimization
- **DepthStream**: Live depth streaming with visualization
- **Integration Tests**: Comprehensive testing suite
- **Example Integration**: Demo with camera and mock detection
- **Enhanced Example**: Advanced demo with temporal tracking and quality metrics

## Installation

### Prerequisites

- Python 3.8+
- macOS (for Apple Silicon optimization)
- Camera access permissions

### Dependencies

```bash
# Activate virtual environment
source env/bin/activate

# Install required packages
pip install torch torchvision transformers opencv-python numpy ultralytics
```

## Quick Start

### Basic Usage

```bash
# Live depth streaming
python3 depth/depth_stream.py

# Test depth processing
python3 depth/test_depth_integration.py
```

### Enhanced Depth Processing

```bash
# Test enhanced depth processing
python3 depth/test_enhanced_depth.py

# Run integrated detection + depth with enhanced processing
python3 detection_depth_integration.py
```

### Individual Components

```bash
# Test depth processing
python3 depth/test_depth_integration.py

# Live depth streaming
python3 depth/depth_stream.py
```

## Enhanced Depth Processing

The `EnhancedDepthProcessor` provides advanced depth processing capabilities optimized for spatial audio applications:

### Key Features

- **Multi-Layer Reference System**: User space, background, scene-specific, and dynamic references
- **Temporal Depth Tracking**: Motion detection, velocity calculation, and trajectory prediction
- **Quality Assessment**: Signal-to-noise ratio, edge sharpness, spatial coherence analysis
- **Spatial Audio Optimization**: Azimuth/elevation angles, audio priority, proximity warnings
- **Context-Aware Processing**: Scene classification, object relationships, spatial density analysis

### Enhanced Metrics

#### Depth Quality Metrics
```python
@dataclass
class DepthQualityMetrics:
    signal_to_noise_ratio: float      # Depth map quality
    edge_sharpness: float             # Object boundary clarity
    spatial_coherence: float          # Consistency across object surface
    temporal_consistency: float       # Stability over time
    illumination_quality: float       # Lighting conditions
    texture_richness: float           # Surface detail availability
    overall_confidence: float         # Overall depth reliability
```

#### Spatial Audio Metrics
```python
@dataclass
class SpatialAudioMetrics:
    azimuth_angle: float              # Left-right position (-180° to +180°)
    elevation_angle: float            # Up-down position (-90° to +90°)
    distance_category: str           # 'very_close', 'close', 'medium', 'far'
    audio_priority: int               # Priority for audio rendering (1-10)
    spatial_uncertainty: float       # Uncertainty in spatial position
    proximity_warning: bool           # Should trigger proximity alert?
    movement_direction: str          # Movement classification
    movement_speed: str               # 'slow', 'medium', 'fast'
    audio_intensity: float            # Volume/intensity (0.0-1.0)
    audio_frequency: float           # Pitch/frequency modulation
```

### Reference Point Selection

The enhanced processor uses a sophisticated multi-layer reference system:

1. **User Reference**: Camera position and personal space boundaries
2. **Background Reference**: Corner and edge sampling for scene context
3. **Scene-Specific References**: Indoor/outdoor context-aware references
4. **Dynamic References**: Moving objects with consistent depth

### Temporal Tracking

- **Depth History**: Tracks depth changes over time for each object
- **Motion Detection**: Identifies approaching, receding, lateral, or stationary movement
- **Velocity Calculation**: Calculates depth change rate (m/s)
- **Trajectory Prediction**: Predicts future positions for motion compensation

### Usage Example

```python
from depth import EnhancedDepthProcessor, create_enhanced_depth_processor

# Create enhanced processor
processor = create_enhanced_depth_processor()

# Get comprehensive depth information
result = processor.get_enhanced_depth_for_spatial_audio(
    frame, bbox, object_id="person_1", object_class="person"
)

# Access enhanced metrics
raw_depth = result['raw_depth']
normalized_depth = result['normalized_depth']
depth_layer = result['depth_layer']  # 'very_close', 'close', 'medium', 'far'
quality_metrics = result['quality_metrics']
spatial_audio = result['spatial_audio']

# Use for spatial audio
audio_priority = spatial_audio.audio_priority
proximity_warning = spatial_audio.proximity_warning
azimuth_angle = spatial_audio.azimuth_angle
elevation_angle = spatial_audio.elevation_angle
```

## API Reference

### DepthProcessor

Core depth estimation and processing class.

#### Methods

- `estimate_depth(frame)`: Estimate depth map from camera frame
- `get_depth_from_bbox(depth_map, bbox, method)`: Extract depth value from bounding box
- `normalize_depth(depth_value, method)`: Normalize depth value to 0.0-1.0 range
- `get_depth_for_spatial_audio(frame, bbox, method, normalization)`: Complete processing pipeline
- `get_depth_stats()`: Get current depth statistics
- `reset_depth_stats()`: Reset depth statistics

#### Parameters

- `method`: Depth calculation method ('mean', 'median', 'min', 'max')
- `normalization`: Normalization method ('relative', 'statistical', 'reference')

### EnhancedDepthProcessor

Advanced depth processing with temporal tracking, quality assessment, and spatial audio optimization.

#### Methods

- `get_enhanced_depth_for_spatial_audio(frame, bbox, object_id, object_class)`: Get comprehensive depth information
- `get_spatial_context(frame, detections)`: Analyze spatial context
- `reset_tracking()`: Reset all tracking data

#### Enhanced Features

- **ReferencePointManager**: Multi-layer reference point system
- **TemporalDepthTracker**: Motion detection and trajectory prediction
- **DepthQualityAssessor**: Comprehensive quality metrics
- **SpatialContextAnalyzer**: Scene analysis and object relationships

### DepthStream

Live depth streaming with real-time visualization.

#### Controls

- `q`: Quit application
- `s`: Save snapshot (if save directory specified)
- `r`: Reset depth statistics
- `b`: Toggle bounding box display

## Testing

### Integration Tests

The test suite verifies all core functionality:

```bash
python3 depth/test_depth_integration.py
```

**Test Coverage:**
- Depth processing with bounding boxes
- Multiple normalization methods
- Different depth calculation methods
- Integration interface
- Performance benchmarks

### Performance Tests

Expected performance metrics:
- Processing time: ~74ms per frame
- FPS: ~13.5 on Apple Silicon
- Memory usage: ~1GB for model

## Camera Configuration

### Built-in Laptop Camera

Configure in `depth/depth_stream.py`:
```python
camera = CameraStream(camera_index=0)
```

### External Camera (iPhone/USB)

Configure in `depth/depth_stream.py`:
```python
camera = CameraStream(camera_index=1)
```

**Requirements for iPhone:**
- iPhone signed into same Apple ID as Mac
- iPhone nearby and unlocked
- macOS Ventura or later

## Coordinate System

The module uses a universal coordinate system:

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (0 to frame_width)
- **Y-axis**: Top to bottom (0 to frame_height)
- **Bounding Box Format**: [x1, y1, x2, y2]

## Integration Examples

### With Object Detection

```python
from depth.depth_processor import create_depth_processor

# Initialize processor
processor = create_depth_processor()

# Process detection results
for detection in detections:
    bbox = detection['bbox']
    result = processor.get_depth_for_spatial_audio(frame, bbox)
    
    # Add depth to detection
    detection['depth'] = result['normalized_depth']
```

### With Spatial Audio

```python
# Get depth for spatial audio
result = processor.get_depth_for_spatial_audio(frame, bbox)

# Use normalized depth (0.0-1.0 range)
spatial_audio.set_depth(result['normalized_depth'])
```

## Configuration Options

### Depth Calculation Methods

- **mean**: Average depth in ROI
- **median**: Median depth in ROI (recommended)
- **min**: Minimum depth in ROI
- **max**: Maximum depth in ROI

### Normalization Methods

- **relative**: Normalize based on current depth statistics
- **statistical**: Z-score based normalization
- **reference**: Normalize relative to reference depth

### Camera Settings

```python
from camera.index import CameraStream

camera = CameraStream(
    camera_index=1,
    width=1280,
    height=720,
    fps=30
)
```

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
**Solution**: 
1. Check camera permissions in System Preferences
2. Try different camera indices
3. Restart camera applications

**Problem**: External camera not detected
**Solution**:
1. Ensure iPhone is signed into same Apple ID
2. Check USB/WiFi connection
3. Verify iPhone is nearby and unlocked

### Depth Processing Issues

**Problem**: Model loading fails
**Solution**:
1. Check internet connection for first download
2. Verify sufficient disk space (~1GB)
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


