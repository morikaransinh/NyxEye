#!/usr/bin/env python3
"""
Spatial Audio Module for Project Daredevil

This module provides real 3D spatial audio using OpenAL, compatible with
Apple AirPods spatial audio on macOS. It renders calm white noise audio
sources positioned in 3D space based on detected object positions and depths.

Main components:
- SpatialAudioEngine: Core 3D spatial audio engine using OpenAL
- IntegratedSpatialAudioSystem: Full integration with detection + depth

Usage:
    from spatial_audio import SpatialAudioEngine

    engine = SpatialAudioEngine()
    engine.start()

    # Update object positions
    engine.update_object('person_1', x=0.5, y=0.3, depth=0.7)
"""

from .index import SpatialAudioEngine, PYOAL_AVAILABLE

__all__ = ["SpatialAudioEngine", "PYOAL_AVAILABLE"]

__version__ = "1.0.0"
