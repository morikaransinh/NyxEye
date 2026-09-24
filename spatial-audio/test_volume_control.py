#!/usr/bin/env python3
"""
Test script to demonstrate the new volume control features:
- Quiet when no objects detected
- Gradually louder as objects get closer
"""

import sys
import os
import time

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from spatial_audio_simple import SimpleSpatialAudio


def test_volume_control():
    """Test the volume control features"""
    print("Testing Volume Control Features")
    print("=" * 50)
    print("This test demonstrates:")
    print("1. Very quiet ambient sound when no objects detected")
    print("2. Gradually louder sound as objects get closer")
    print("3. Distance-based volume scaling")
    print("=" * 50)

    # Initialize with quiet settings
    audio = SimpleSpatialAudio(
        max_sources=5,
        master_volume=0.1,  # Base volume
        min_volume=0.01,  # Very quiet ambient
        max_volume=0.25,  # Louder for close objects
    )

    try:
        audio.start()
        print("✓ Audio system started")
        print("\nPhase 1: No objects (should be very quiet ambient)")
        time.sleep(3)

        print("\nPhase 2: Far object (depth=0.8, should be quiet)")
        audio.update_object("far_obj", x=0.5, y=0.5, depth=0.8, volume=0.1)
        time.sleep(3)

        print("\nPhase 3: Medium distance object (depth=0.5, should be moderate)")
        audio.update_object("medium_obj", x=0.3, y=0.3, depth=0.5, volume=0.1)
        time.sleep(3)

        print("\nPhase 4: Close object (depth=0.2, should be louder)")
        audio.update_object("close_obj", x=0.7, y=0.7, depth=0.2, volume=0.1)
        time.sleep(3)

        print("\nPhase 5: Very close object (depth=0.1, should be loudest)")
        audio.update_object("very_close_obj", x=0.5, y=0.5, depth=0.1, volume=0.1)
        time.sleep(3)

        print("\nPhase 6: Multiple objects at different distances")
        audio.update_object("obj1", x=0.2, y=0.2, depth=0.3, volume=0.1)
        audio.update_object("obj2", x=0.8, y=0.8, depth=0.7, volume=0.1)
        audio.update_object("obj3", x=0.5, y=0.5, depth=0.1, volume=0.1)
        time.sleep(4)

        print("\nPhase 7: Remove all objects (back to quiet ambient)")
        audio.remove_object("far_obj")
        audio.remove_object("medium_obj")
        audio.remove_object("close_obj")
        audio.remove_object("very_close_obj")
        audio.remove_object("obj1")
        audio.remove_object("obj2")
        audio.remove_object("obj3")
        time.sleep(3)

        print("\n✓ Volume control test complete!")
        print("You should have heard:")
        print("- Very quiet ambient sound when no objects")
        print("- Gradually increasing volume as objects got closer")
        print("- Multiple objects creating layered audio")
        print("- Return to quiet ambient when objects removed")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        audio.stop()
        print("Audio system stopped")


if __name__ == "__main__":
    test_volume_control()
