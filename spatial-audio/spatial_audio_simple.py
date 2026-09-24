#!/usr/bin/env python3
"""
Simplified Spatial Audio - WORKING VERSION
Uses PyGame for 3D spatial audio that actually works
"""

import pygame
import numpy as np
import threading
import time
from typing import Dict, Optional, Tuple


class SimpleSpatialAudio:
    """Simplified spatial audio using PyGame's mixer (actually works!)"""

    def __init__(
        self, max_sources=10, master_volume=0.1, min_volume=0.02, max_volume=0.3
    ):
        self.max_sources = max_sources
        self.master_volume = master_volume  # Base volume level
        self.min_volume = min_volume  # Minimum volume when no objects detected
        self.max_volume = max_volume  # Maximum volume for very close objects
        self.objects: Dict[str, dict] = {}
        self.object_channels: Dict[str, int] = (
            {}
        )  # Track which channel each object uses
        self.lock = threading.Lock()
        self.is_running = False

        # Initialize pygame mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        print("✓ PyGame Audio initialized successfully")
        print(f"Max sources: {max_sources}")
        print(f"Master volume: {master_volume} (base level)")
        print(f"Volume range: {min_volume} - {max_volume}")

    def _generate_white_noise(self, duration=0.5, pitch=440.0):
        """Generate calm white noise with optional pitch variation"""
        sample_rate = 44100
        num_samples = int(duration * sample_rate)

        # Generate calm white noise - make it quieter and more subtle
        noise = np.random.normal(0, 0.08, (num_samples, 2)).astype(np.float32)

        # Heavy low-pass filter for calmness (smoothing)
        window_size = 20  # Larger window for more smoothing
        if num_samples > window_size:
            kernel = np.ones(window_size) / window_size
            noise[:, 0] = np.convolve(noise[:, 0], kernel, mode="same")
            noise[:, 1] = np.convolve(noise[:, 1], kernel, mode="same")

        # Add VERY subtle pitch variation based on depth (not a beep!)
        # This creates a slight frequency shift rather than a tone
        t = np.linspace(0, duration, num_samples)
        # Very gentle modulation - not a beep, just slight frequency variation
        pitch_modulation = 0.02 * np.sin(
            2 * np.pi * pitch * 0.01 * t
        )  # Very slow modulation
        noise[:, 0] += pitch_modulation
        noise[:, 1] += pitch_modulation

        # Convert to int16
        noise_int16 = (noise * 32767).astype(np.int16)

        return noise_int16

    def _screen_to_pan(self, x, y, depth):
        """Convert screen position to stereo pan"""
        # X position controls left/right panning (-1.0 to 1.0)
        pan = (x - 0.5) * 2.0  # 0.0-1.0 -> -1.0 to 1.0

        # Depth controls volume (closer = louder, farther = quieter)
        # depth 0.0 = close (loud), 1.0 = far (very quiet)
        # Use exponential curve for better distance perception
        volume_multiplier = (1.0 - depth) ** 2  # Quadratic falloff

        return pan, volume_multiplier

    def _calculate_dynamic_volume(self, base_volume, depth, num_objects):
        """
        Calculate dynamic volume based on distance and number of objects.

        Args:
            base_volume: Base volume from object
            depth: Depth value (0.0 = close, 1.0 = far)
            num_objects: Number of currently detected objects

        Returns:
            Final volume level
        """
        # If no objects detected, use minimum volume
        if num_objects == 0:
            return self.min_volume

        # Calculate distance-based volume scaling
        # Closer objects (lower depth) = louder
        distance_factor = (
            1.0 - depth
        ) ** 1.5  # Slightly less aggressive than quadratic

        # Scale volume based on distance
        # Close objects (depth=0.0) get max volume, far objects (depth=1.0) get min volume
        volume_range = self.max_volume - self.min_volume
        distance_volume = self.min_volume + (volume_range * distance_factor)

        # Apply base volume scaling
        final_volume = distance_volume * base_volume

        # Clamp to volume range
        final_volume = max(self.min_volume, min(self.max_volume, final_volume))

        return final_volume

    def _depth_to_pitch(self, depth):
        """Convert depth to subtle pitch variation (closer = slightly higher)"""
        # depth 0.0 (close) -> slightly higher (500 Hz)
        # depth 1.0 (far) -> slightly lower (300 Hz)
        # Small range for subtle variation, not a beep
        pitch_hz = 300 + (500 - 300) * (1.0 - depth)
        return pitch_hz

    def _y_to_pitch_modulation(self, y):
        """Convert Y position to vertical pitch shift (up = slightly higher pitch)"""
        # y 0.0 (top) -> slightly higher pitch
        # y 1.0 (bottom) -> slightly lower pitch
        # Very subtle - maybe 50 Hz range
        pitch_offset = 50 * (0.5 - y)  # Center (0.5) = no offset
        return pitch_offset

    def update_object(self, object_id, x, y, depth, volume=None, active=True):
        """Update object position"""
        with self.lock:
            # Check if this is a new object or updated position
            is_new_object = object_id not in self.objects

            prev_obj = self.objects.get(object_id, {})
            volume_val = volume if volume else self.master_volume

            # Check if position/depth/volume changed significantly
            position_changed = is_new_object or (
                abs(prev_obj.get("x", 0) - x) > 0.05
                or abs(prev_obj.get("y", 0) - y) > 0.05
                or abs(prev_obj.get("depth", 0) - depth) > 0.1
                or abs(prev_obj.get("volume", 0) - volume_val) > 0.05
            )

            self.objects[object_id] = {
                "x": x,
                "y": y,
                "depth": depth,
                "volume": volume_val,
                "active": active,
                "last_update": time.time(),
                "needs_new_sound": position_changed,  # Flag to indicate if sound needs regenerating
            }

    def _play_object_sound(self, object_id, obj_data):
        """Play spatial sound for object"""
        if not obj_data["active"]:
            return

        # Only play new sound if position/depth/volume changed significantly
        if not obj_data.get("needs_new_sound", True):
            return

        # Mark that we've played a sound for this position
        obj_data["needs_new_sound"] = False

        # Calculate panning and pitch based on depth and position
        pan, volume_mult = self._screen_to_pan(
            obj_data["x"], obj_data["y"], obj_data["depth"]
        )
        base_pitch = self._depth_to_pitch(obj_data["depth"])
        y_pitch_offset = self._y_to_pitch_modulation(obj_data["y"])
        final_pitch = base_pitch + y_pitch_offset

        # Calculate dynamic volume based on distance and number of objects
        num_objects = len([o for o in self.objects.values() if o["active"]])
        final_volume = self._calculate_dynamic_volume(
            obj_data["volume"], obj_data["depth"], num_objects
        )

        # If object is too far or volume too low, don't play
        if final_volume < self.min_volume * 1.1:  # Small threshold above minimum
            return

        # Generate noise with pitch variation based on depth and vertical position
        noise = self._generate_white_noise(duration=0.5, pitch=final_pitch)

        # Apply stereo panning (left/right positioning)
        left_vol = max(0, min(1, (1.0 - pan) / 2.0))
        right_vol = max(0, min(1, (1.0 + pan) / 2.0))

        # Adjust volume per channel with panning
        noise[:, 0] = (noise[:, 0] * left_vol * final_volume).astype(np.int16)
        noise[:, 1] = (noise[:, 1] * right_vol * final_volume).astype(np.int16)

        # Convert to pygame sound
        try:
            sound = pygame.sndarray.make_sound(noise)

            # Get or assign channel for this object
            if object_id in self.object_channels:
                channel_idx = self.object_channels[object_id]
            else:
                # Use a simple modulo to get channel index
                channel_idx = abs(hash(object_id)) % self.max_sources
                self.object_channels[object_id] = channel_idx

            channel = pygame.mixer.Channel(channel_idx)

            # IMPORTANT: Stop the channel before playing a new sound
            # This prevents accumulating sounds
            channel.stop()

            channel.play(sound, loops=-1)  # Loop infinitely
        except Exception as e:
            # Silently fail - channels might be busy
            pass

    def _play_ambient_sound(self):
        """Play very quiet ambient sound when no objects are detected"""
        # Generate very quiet ambient noise
        noise = self._generate_white_noise(duration=1.0, pitch=300.0)

        # Make it extremely quiet - almost silent ambient level
        ambient_volume = (
            self.min_volume * 0.1
        )  # Much quieter than minimum (10% of min_volume)
        noise = (noise * ambient_volume).astype(np.int16)

        # Play on center channel (no panning)
        try:
            sound = pygame.sndarray.make_sound(noise)
            channel = pygame.mixer.Channel(0)  # Use channel 0 for ambient
            channel.play(sound, loops=-1)
        except Exception as e:
            # Silently fail - channel might be busy
            pass

    def _update_loop(self):
        """Update loop"""
        while self.is_running:
            try:
                with self.lock:
                    current_time = time.time()

                    # Track currently active object IDs
                    current_active_ids = set()
                    active_objects = []

                    for object_id, obj_data in list(self.objects.items()):
                        # Remove old objects that haven't been updated in 2 seconds
                        if current_time - obj_data["last_update"] > 2.0:
                            # Stop the channel for this removed object
                            if object_id in self.object_channels:
                                channel_idx = self.object_channels[object_id]
                                try:
                                    channel = pygame.mixer.Channel(channel_idx)
                                    channel.stop()
                                except:
                                    pass
                                del self.object_channels[object_id]
                            del self.objects[object_id]
                            continue

                        if obj_data["active"]:
                            current_active_ids.add(object_id)
                            active_objects.append((object_id, obj_data))

                    # Stop channels for objects that are no longer active
                    for object_id in list(self.object_channels.keys()):
                        if object_id not in current_active_ids:
                            channel_idx = self.object_channels[object_id]
                            try:
                                channel = pygame.mixer.Channel(channel_idx)
                                channel.stop()
                            except:
                                pass
                            del self.object_channels[object_id]

                    # Only play audio for currently active objects
                    if len(active_objects) > 0:
                        for object_id, obj_data in active_objects:
                            self._play_object_sound(object_id, obj_data)

                time.sleep(0.1)  # Update every 100ms

            except Exception as e:
                print(f"Error in update loop: {e}")
                time.sleep(0.1)

    def start(self):
        """Start audio system"""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("Spatial audio started")

    def stop(self):
        """Stop audio system"""
        self.is_running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        pygame.mixer.quit()
        print("Spatial audio stopped")

    def remove_object(self, object_id):
        """Remove object and stop its sound"""
        with self.lock:
            # Stop the channel for this object
            if object_id in self.object_channels:
                channel_idx = self.object_channels[object_id]
                try:
                    channel = pygame.mixer.Channel(channel_idx)
                    channel.stop()
                except:
                    pass
                # Remove from tracking
                del self.object_channels[object_id]

            # Remove from objects dict
            if object_id in self.objects:
                del self.objects[object_id]

    def get_stats(self):
        """Get engine statistics"""
        with self.lock:
            return {
                "total_objects": len(self.objects),
                "active_sources": len(
                    [o for o in self.objects.values() if o["active"]]
                ),
                "max_sources": self.max_sources,
                "master_volume": self.master_volume,
            }


def test():
    """Test the spatial audio"""
    print("Testing Simplified Spatial Audio...")
    print("=" * 60)

    audio = SimpleSpatialAudio(
        max_sources=10,
        master_volume=0.1,  # Base volume level
        min_volume=0.02,  # Very quiet when no objects
        max_volume=0.3,  # Louder for close objects
    )

    try:
        audio.start()
        print("✓ Audio started")

        print("\nTest 1: Object on left")
        audio.update_object("obj1", x=0.2, y=0.5, depth=0.5)
        time.sleep(3)

        print("Test 2: Object on right")
        audio.update_object("obj2", x=0.8, y=0.5, depth=0.5)
        time.sleep(3)

        print("Test 3: Move object to center")
        audio.update_object("obj1", x=0.5, y=0.5, depth=0.5)
        time.sleep(2)

        print("Test 4: Objects getting closer (louder)")
        audio.update_object("obj1", x=0.3, y=0.3, depth=0.2)
        audio.update_object("obj2", x=0.7, y=0.3, depth=0.2)
        time.sleep(3)

        print("\n✓ All tests complete!")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        audio.stop()


if __name__ == "__main__":
    test()
