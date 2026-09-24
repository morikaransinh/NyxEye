#!/usr/bin/env python3
"""
Spatial Audio Module for Project Daredevil

This module provides real 3D spatial audio using OpenAL, compatible with
Apple AirPods spatial audio on macOS. It renders calm white noise audio
sources positioned in 3D space based on detected object positions and depths.

Usage:
    from spatial_audio import SpatialAudioEngine

    engine = SpatialAudioEngine()
    engine.start()

    # Update object positions
    engine.update_object('person_1', x=0.5, y=0.3, depth=0.7)
    engine.update_object('bottle_2', x=0.8, y=0.2, depth=0.9)
"""

import numpy as np
import threading
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import openal

    PYOAL_AVAILABLE = True
    print("✓ PyOpenAL loaded successfully")
except ImportError as e:
    PYOAL_AVAILABLE = False
    print(f"⚠ PyOpenAL not available: {e}")
    warnings.warn("PyOpenAL not installed. Install with: pip install PyOpenAL")
except Exception as e:
    PYOAL_AVAILABLE = False
    print(f"⚠ OpenAL library not found: {e}")
    print("Trying to set up OpenAL environment...")

    # Try to set up OpenAL library path
    import platform

    if platform.system() == "Darwin":  # macOS
        import os

        possible_paths = [
            "/opt/homebrew/lib/libopenal.dylib",
            "/opt/homebrew/Cellar/openal-soft/*/lib/libopenal.dylib",
            "/usr/local/lib/libopenal.dylib",
            "/Library/Frameworks/OpenAL.framework/OpenAL",
        ]

        # Try to set environment variable
        for path in possible_paths:
            if "*" in path:
                import glob

                matches = glob.glob(path)
                if matches:
                    path = matches[0]

            if os.path.exists(path):
                os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(path)
                print(f"Found OpenAL at: {path}")
                try:
                    import openal as al
                    from openal import oalOpen

                    PYOAL_AVAILABLE = True
                    print("✓ OpenAL loaded after setting path")
                    break
                except:
                    continue

    if not PYOAL_AVAILABLE:
        warnings.warn(
            "OpenAL library couldn't be found. Install with: brew install openal-soft"
        )


class SpatialAudioEngine:
    """
    3D Spatial Audio Engine using OpenAL for real spatial positioning.

    This engine creates 3D sound sources that can be positioned anywhere in 3D space,
    compatible with AirPods spatial audio on macOS.
    """

    def __init__(
        self,
        max_sources: int = 10,
        master_volume: float = 0.3,
        audio_tick_rate: float = 60.0,
        listener_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        listener_forward: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        listener_up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ):
        """
        Initialize the spatial audio engine.

        Args:
            max_sources: Maximum number of simultaneous audio sources
            master_volume: Master volume level (0.0 to 1.0)
            audio_tick_rate: Update rate for audio positioning (Hz)
            listener_position: Listener position in 3D space
            listener_forward: Forward direction vector for listener
            listener_up: Up direction vector for listener
        """
        if not PYOAL_AVAILABLE:
            raise RuntimeError(
                "PyOpenAL is not available. Install with: pip install PyOpenAL\n"
                "Note: On macOS, you may need: brew install openal-soft"
            )

        self.max_sources = max_sources
        self.master_volume = master_volume
        self.audio_tick_rate = audio_tick_rate

        # Listener configuration
        self.listener_position = np.array(listener_position, dtype=np.float32)
        self.listener_forward = np.array(listener_forward, dtype=np.float32)
        self.listener_up = np.array(listener_up, dtype=np.float32)

        # Object tracking
        self.objects: Dict[str, dict] = {}
        self.lock = threading.Lock()

        # OpenAL initialization
        self.device = None
        self.context = None
        self.sources = {}
        self.active_sources = set()

        # Audio state
        self.is_running = False
        self.audio_thread = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        print("Spatial Audio Engine initialized")
        print(f"Max sources: {max_sources}")
        print(f"Master volume: {master_volume}")
        print(f"Audio tick rate: {audio_tick_rate} Hz")

    def _initialize_openal(self):
        """Initialize OpenAL device and context."""
        try:
            # Open the default audio device using low-level API
            self.device = openal.alc.alcOpenDevice(None)

            if not self.device:
                raise RuntimeError("Failed to open audio device")

            print(f"Audio device opened successfully")

            # Create context
            self.context = openal.alc.alcCreateContext(self.device, None)
            if not self.context:
                openal.alc.alcCloseDevice(self.device)
                raise RuntimeError("Failed to create audio context")

            # Make context current
            if not openal.alc.alcMakeContextCurrent(self.context):
                openal.alc.alcDestroyContext(self.context)
                openal.alc.alcCloseDevice(self.device)
                raise RuntimeError("Failed to make context current")

            # Configure listener
            self._update_listener()

            print("OpenAL initialized successfully")

        except Exception as e:
            print(f"Error initializing OpenAL: {e}")
            print(
                "Make sure audio device is available and not in use by another application"
            )
            if hasattr(self, "device") and self.device:
                try:
                    openal.alc.alcCloseDevice(self.device)
                except:
                    pass
            raise

    def _update_listener(self):
        """Update listener position and orientation."""
        import ctypes

        # Set listener position
        openal.al.alListener3f(
            openal.AL_POSITION,
            float(self.listener_position[0]),
            float(self.listener_position[1]),
            float(self.listener_position[2]),
        )

        # Set listener orientation (forward vector, then up vector)
        orientation_array = np.concatenate(
            [self.listener_forward, self.listener_up]
        ).astype(np.float32)
        orientation = (ctypes.c_float * len(orientation_array))(*orientation_array)
        openal.al.alListenerfv(openal.AL_ORIENTATION, orientation)

        # Set master volume
        openal.al.alListenerf(openal.AL_GAIN, self.master_volume)

    def _generate_white_noise(
        self, duration: float = 1.0, sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Generate calm white noise for spatial audio.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Array of audio samples
        """
        num_samples = int(duration * sample_rate)

        # Generate calm white noise with gentle filtering
        # Use lower amplitude for calm effect
        noise = np.random.normal(0, 0.3, num_samples).astype(np.float32)

        # Apply gentle low-pass filter for calmness
        # Simple moving average to smooth the noise
        window_size = 10
        if num_samples > window_size:
            kernel = np.ones(window_size) / window_size
            noise = np.convolve(noise, kernel, mode="same")

        return noise

    def _create_audio_source(self, object_id: str) -> Optional[int]:
        """Create a new OpenAL source for an object."""
        # Generate white noise buffer
        noise_data = self._generate_white_noise(duration=0.5)  # 0.5 second buffer

        # Create buffer
        buffer_id = openal.al.alGenBuffers(1)
        openal.al.alBufferData(
            buffer_id,
            openal.AL_FORMAT_MONO16,
            (noise_data * 32767).astype(np.int16).tobytes(),
            44100,  # sample rate
        )

        # Create source
        source_id = openal.al.alGenSources(1)

        # Attach buffer to source and loop
        openal.al.alSourcei(source_id, openal.AL_BUFFER, buffer_id)
        openal.al.alSourcei(source_id, openal.AL_LOOPING, openal.AL_TRUE)

        # Configure source properties
        openal.al.alSourcef(source_id, openal.AL_GAIN, 0.2)  # Initial volume
        openal.al.alSourcef(source_id, openal.AL_PITCH, 1.0)

        # Set position to origin initially
        openal.al.alSource3f(source_id, openal.AL_POSITION, 0.0, 0.0, 0.0)

        # Enable 3D positioning
        openal.al.alSourcei(source_id, openal.AL_SOURCE_RELATIVE, openal.AL_FALSE)

        # Set distance model for realistic 3D positioning
        openal.al.alDistanceModel(openal.AL_INVERSE_DISTANCE_CLAMPED)
        openal.al.alSourcef(source_id, openal.AL_REFERENCE_DISTANCE, 1.0)
        openal.al.alSourcef(source_id, openal.AL_MAX_DISTANCE, 10.0)
        openal.al.alSourcef(source_id, openal.AL_ROLLOFF_FACTOR, 1.0)

        return source_id

    def _screen_to_3d_position(
        self,
        x: float,
        y: float,
        depth: float,
        screen_width: float = 1.0,
        screen_height: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Convert 2D screen position and depth to 3D spatial position.

        Args:
            x: X position on screen (0.0 to 1.0, left to right)
            y: Y position on screen (0.0 to 1.0, top to bottom)
            depth: Depth value (0.0 = near, 1.0 = far)
            screen_width: Screen width in meters (virtual)
            screen_height: Screen height in meters (virtual)

        Returns:
            3D position as (x, y, z) in meters
        """
        # Convert screen coordinates to 3D space
        # Center the screen at (0, 0, -2) meters
        # X: left (-) to right (+)
        # Y: top (+) to bottom (-)
        # Z: depth from viewer (-Z is forward)

        # Normalize screen coordinates to [-1, 1] range
        x_pos = (x - 0.5) * 2.0  # -1 (left) to +1 (right)
        y_pos = (0.5 - y) * 2.0  # +1 (top) to -1 (bottom)

        # Map to real-world scale (meter scale)
        x_meters = x_pos * screen_width * 0.5
        y_meters = y_pos * screen_height * 0.5

        # Map depth to Z position (closer = larger negative Z)
        # Depth 0.0 (near) -> z = -1.0m
        # Depth 1.0 (far) -> z = -5.0m
        z_meters = -(1.0 + depth * 4.0)

        return (x_meters, y_meters, z_meters)

    def update_object(
        self,
        object_id: str,
        x: float,
        y: float,
        depth: float,
        volume: Optional[float] = None,
        active: bool = True,
    ):
        """
        Update an object's position in 3D space.

        Args:
            object_id: Unique identifier for the object
            x: X position on screen (0.0 to 1.0, left to right)
            y: Y position on screen (0.0 to 1.0, top to bottom)
            depth: Depth value (0.0 = near/front, 1.0 = far/back)
            volume: Optional volume level (0.0 to 1.0). If None, uses master_volume.
            active: Whether the object should produce sound
        """
        with self.lock:
            # Store object state
            self.objects[object_id] = {
                "x": x,
                "y": y,
                "depth": depth,
                "volume": volume if volume is not None else self.master_volume,
                "active": active,
                "last_update": time.time(),
            }

    def remove_object(self, object_id: str):
        """Remove an object from spatial audio."""
        with self.lock:
            if object_id in self.objects:
                # Stop and delete source if it exists
                if object_id in self.sources:
                    try:
                        openal.al.alSourceStop(self.sources[object_id])
                        openal.al.alDeleteSources([self.sources[object_id]])
                    except:
                        pass
                    del self.sources[object_id]
                    self.active_sources.discard(object_id)

                del self.objects[object_id]

    def remove_all_objects(self):
        """Remove all objects from spatial audio."""
        with self.lock:
            for object_id in list(self.objects.keys()):
                self.remove_object(object_id)

    def _update_audio_loop(self):
        """Main audio update loop running in separate thread."""
        while self.is_running:
            try:
                with self.lock:
                    current_time = time.time()

                    # Update each object's audio source
                    for object_id, obj_data in self.objects.items():
                        # Create source if it doesn't exist
                        if object_id not in self.sources:
                            if len(self.sources) < self.max_sources:
                                source_id = self._create_audio_source(object_id)
                                if source_id:
                                    self.sources[object_id] = source_id
                                    if obj_data["active"]:
                                        openal.al.alSourcePlay(source_id)
                                        self.active_sources.add(object_id)
                            continue

                        source_id = self.sources[object_id]

                        # Update 3D position
                        x, y, depth = obj_data["x"], obj_data["y"], obj_data["depth"]
                        pos_3d = self._screen_to_3d_position(x, y, depth)
                        openal.al.alSource3f(source_id, openal.AL_POSITION, *pos_3d)

                        # Update volume
                        openal.al.alSourcef(
                            source_id, openal.AL_GAIN, obj_data["volume"]
                        )

                        # Update playback state
                        state = openal.al.alGetSourcei(
                            source_id, openal.AL_SOURCE_STATE
                        )

                        if obj_data["active"]:
                            if state != openal.AL_PLAYING:
                                openal.al.alSourcePlay(source_id)
                                self.active_sources.add(object_id)
                        else:
                            if state == openal.AL_PLAYING:
                                openal.al.alSourcePause(source_id)
                                self.active_sources.discard(object_id)

                        # Check if object should be removed (not updated for 5 seconds)
                        if current_time - obj_data["last_update"] > 5.0:
                            self.remove_object(object_id)

                # Sleep to control update rate
                time.sleep(1.0 / self.audio_tick_rate)

            except Exception as e:
                print(f"Error in audio update loop: {e}")
                time.sleep(0.1)

    def start(self):
        """Start the spatial audio engine."""
        if self.is_running:
            print("Spatial audio engine already running")
            return

        # Initialize OpenAL
        self._initialize_openal()

        # Start audio update thread
        self.is_running = True
        self.audio_thread = threading.Thread(
            target=self._update_audio_loop, daemon=True
        )
        self.audio_thread.start()

        print("Spatial audio engine started")

    def stop(self):
        """Stop the spatial audio engine."""
        if not self.is_running:
            return

        # Stop audio loop
        self.is_running = False

        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)

        # Clean up all sources
        with self.lock:
            self.remove_all_objects()

            # Delete remaining sources
            for source_id in self.sources.values():
                try:
                    openal.al.alSourceStop(source_id)
                    openal.al.alDeleteSources([source_id])
                except:
                    pass

        # Clean up OpenAL context and device
        if self.context:
            openal.alc.alcMakeContextCurrent(None)
            openal.alc.alcDestroyContext(self.context)
            self.context = None

        if self.device:
            openal.alc.alcCloseDevice(self.device)
            self.device = None

        print("Spatial audio engine stopped")

    def get_stats(self) -> dict:
        """Get engine statistics."""
        with self.lock:
            return {
                "total_objects": len(self.objects),
                "active_sources": len(self.active_sources),
                "max_sources": self.max_sources,
                "master_volume": self.master_volume,
            }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage and test
if __name__ == "__main__":
    import sys

    if not PYOAL_AVAILABLE:
        print("PyOpenAL not available. Please install:")
        print("  pip install PyOpenAL")
        print("  brew install openal-soft  # On macOS")
        sys.exit(1)

    print("Testing Spatial Audio Engine...")
    print("You should hear calm white noise positioned in 3D space")
    print("Press Ctrl+C to exit\n")

    try:
        engine = SpatialAudioEngine(max_sources=10, master_volume=0.2)
        engine.start()

        # Test objects at different positions
        print("Adding test objects...")
        time.sleep(0.5)

        # Object on the left, front
        engine.update_object("test_1", x=0.2, y=0.3, depth=0.3)
        time.sleep(1)

        # Object on the right, back
        engine.update_object("test_2", x=0.8, y=0.2, depth=0.8)
        time.sleep(1)

        # Object in center, medium depth
        engine.update_object("test_3", x=0.5, y=0.5, depth=0.5)
        time.sleep(1)

        # Simulate movement
        print("Simulating object movement...")
        for i in range(10):
            x = 0.3 + (i / 10.0) * 0.4
            engine.update_object("test_1", x=x, y=0.3, depth=0.3)
            time.sleep(0.2)

        print("\nTest complete. Check that you can hear 3D positioned audio.")

        # Keep running
        while True:
            time.sleep(1)
            stats = engine.get_stats()
            print(
                f"Active objects: {stats['total_objects']}, Active sources: {stats['active_sources']}"
            )

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if "engine" in locals():
            engine.stop()
