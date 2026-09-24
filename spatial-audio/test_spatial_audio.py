#!/usr/bin/env python3
"""Quick test script for spatial audio."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index import SpatialAudioEngine
import time

print("Testing Spatial Audio Engine...")
print("=" * 60)

try:
    engine = SpatialAudioEngine(max_sources=10, master_volume=0.2)
    engine.start()
    print("✓ Engine started successfully")

    print("\nTest 1: Adding object on left side")
    engine.update_object("test_1", x=0.2, y=0.5, depth=0.5)
    time.sleep(2)

    print("Test 2: Adding object on right side")
    engine.update_object("test_2", x=0.8, y=0.5, depth=0.5)
    time.sleep(2)

    print("Test 3: Moving first object to center")
    engine.update_object("test_1", x=0.5, y=0.5, depth=0.5)
    time.sleep(2)

    print("Test 4: Moving objects closer (smaller depth)")
    engine.update_object("test_1", x=0.3, y=0.3, depth=0.3)
    engine.update_object("test_2", x=0.7, y=0.3, depth=0.3)
    time.sleep(2)

    print("Test 5: Removing one object")
    engine.remove_object("test_1")
    time.sleep(1)

    print("\nAll tests completed!")
    print("You should have heard calm white noise positioned in 3D space")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

finally:
    try:
        engine.stop()
        print("Engine stopped")
    except:
        pass

print("=" * 60)
print("Test complete!")
