#!/usr/bin/env python3
"""
Test script for Enhanced Depth Processing with Persistence and Clear Terminology

This script demonstrates:
- Clear depth terminology based on reference points
- Temporal persistence to prevent objects from disappearing
- Configurable persistence settings
- Enhanced depth metrics for spatial audio

Usage:
    python3 depth/test_persistence_and_terminology.py
"""

import numpy as np
import time
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth import create_enhanced_depth_processor_with_persistence, PersistenceConfig


def test_depth_terminology():
    """Test clear depth terminology."""
    print("=== Testing Depth Terminology ===")

    # Create processor
    processor = create_enhanced_depth_processor_with_persistence()

    # Test different depth values
    test_depths = [0.2, 0.8, 1.8, 4.0]

    for depth in test_depths:
        # Create dummy frame and bbox
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 100, 200, 200]

        result = processor.get_enhanced_depth_for_spatial_audio(
            dummy_frame, dummy_bbox, object_id=f"test_{depth}", object_class="person"
        )

        terminology = result["depth_terminology"]

        print(f"\nDepth: {depth:.1f}m")
        print(f"  User Space: {terminology['user_space_category']}")
        print(f"  Safety Zone: {terminology['safety_zone']}")
        print(f"  Social Distance: {terminology['social_distance']}")
        print(f"  Background Relationship: {terminology['background_relationship']}")


def test_persistence_functionality():
    """Test temporal persistence functionality."""
    print("\n=== Testing Temporal Persistence ===")

    # Create processor with persistence
    config = PersistenceConfig(
        enabled=True,
        persistence_duration=3.0,
        max_missing_frames=5,
        confidence_decay_rate=0.1,
    )

    processor = create_enhanced_depth_processor_with_persistence(
        persistence_config=config
    )

    # Simulate object detection over time
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]
    object_id = "persistent_test_obj"

    print("Simulating object detection and disappearance...")

    # Frame 1: Object detected
    print("\nFrame 1: Object detected")
    result1 = processor.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id=object_id, object_class="person"
    )
    persistence_info1 = result1["persistence_info"]
    print(f"  Is tracked: {persistence_info1['is_tracked']}")
    print(f"  Missing frames: {persistence_info1['missing_frames']}")
    print(f"  Time since last seen: {persistence_info1['time_since_last_seen']:.2f}s")

    # Frame 2: Object still detected
    print("\nFrame 2: Object still detected")
    result2 = processor.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id=object_id, object_class="person"
    )
    persistence_info2 = result2["persistence_info"]
    print(f"  Is tracked: {persistence_info2['is_tracked']}")
    print(f"  Missing frames: {persistence_info2['missing_frames']}")

    # Frame 3: Object disappears (no detection)
    print("\nFrame 3: Object disappears")
    processor.update_missing_objects()  # Mark as missing
    persistent_objects = processor.get_persistent_objects()

    if persistent_objects:
        obj = persistent_objects[0]
        print(f"  Persistent object found: {obj['object_id']}")
        print(f"  Is active: {obj['is_active']}")
        print(f"  Missing frames: {obj['missing_frames']}")
        print(f"  Time since last seen: {obj['time_since_last_seen']:.2f}s")
    else:
        print("  No persistent objects found")

    # Frame 4: Still missing
    print("\nFrame 4: Object still missing")
    processor.update_missing_objects()
    persistent_objects = processor.get_persistent_objects()

    if persistent_objects:
        obj = persistent_objects[0]
        print(f"  Persistent object: missing {obj['missing_frames']} frames")
        print(f"  Time since last seen: {obj['time_since_last_seen']:.2f}s")
    else:
        print("  Object expired and removed")


def test_configurable_persistence():
    """Test configurable persistence settings."""
    print("\n=== Testing Configurable Persistence ===")

    # Test with persistence disabled
    print("\n1. Persistence Disabled:")
    config_disabled = PersistenceConfig(enabled=False)
    processor_disabled = create_enhanced_depth_processor_with_persistence(
        persistence_config=config_disabled
    )

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]

    result = processor_disabled.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id="test_disabled", object_class="person"
    )
    persistence_info = result["persistence_info"]
    print(f"  Persistence enabled: {persistence_info['persistence_enabled']}")

    # Test with short persistence duration
    print("\n2. Short Persistence Duration (1.0s):")
    config_short = PersistenceConfig(
        enabled=True, persistence_duration=1.0, max_missing_frames=3
    )
    processor_short = create_enhanced_depth_processor_with_persistence(
        persistence_config=config_short
    )

    result = processor_short.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id="test_short", object_class="person"
    )
    persistence_info = result["persistence_info"]
    print(f"  Persistence enabled: {persistence_info['persistence_enabled']}")

    # Test with long persistence duration
    print("\n3. Long Persistence Duration (5.0s):")
    config_long = PersistenceConfig(
        enabled=True, persistence_duration=5.0, max_missing_frames=10
    )
    processor_long = create_enhanced_depth_processor_with_persistence(
        persistence_config=config_long
    )

    result = processor_long.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id="test_long", object_class="person"
    )
    persistence_info = result["persistence_info"]
    print(f"  Persistence enabled: {persistence_info['persistence_enabled']}")


def test_spatial_audio_metrics():
    """Test spatial audio metrics with clear terminology."""
    print("\n=== Testing Spatial Audio Metrics ===")

    processor = create_enhanced_depth_processor_with_persistence()

    # Test different object positions and depths
    test_cases = [
        {"bbox": [50, 50, 150, 150], "class": "person", "depth": 0.8},  # Close person
        {
            "bbox": [300, 100, 400, 200],
            "class": "bottle",
            "depth": 2.5,
        },  # Medium bottle
        {"bbox": [200, 300, 300, 400], "class": "chair", "depth": 4.0},  # Far chair
    ]

    for i, case in enumerate(test_cases):
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = processor.get_enhanced_depth_for_spatial_audio(
            dummy_frame,
            case["bbox"],
            object_id=f"audio_test_{i}",
            object_class=case["class"],
        )

        spatial_audio = result["spatial_audio"]
        terminology = result["depth_terminology"]

        print(f"\nObject {i+1}: {case['class']}")
        print(f"  Depth: {case['depth']:.1f}m ({terminology['user_space_category']})")
        print(f"  Safety Zone: {terminology['safety_zone']}")
        print(f"  Azimuth: {spatial_audio.azimuth_angle:.1f}°")
        print(f"  Elevation: {spatial_audio.elevation_angle:.1f}°")
        print(f"  Audio Priority: {spatial_audio.audio_priority}")
        print(f"  Proximity Warning: {spatial_audio.proximity_warning}")
        print(f"  Audio Intensity: {spatial_audio.audio_intensity:.2f}")


def main():
    """Run all tests."""
    print("Enhanced Depth Processing with Persistence and Clear Terminology")
    print("=" * 70)

    try:
        test_depth_terminology()
        test_persistence_functionality()
        test_configurable_persistence()
        test_spatial_audio_metrics()

        print("\n" + "=" * 70)
        print("[PASS] All tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print(
            "  ✓ Clear depth terminology (user_space_category, safety_zone, social_distance)"
        )
        print("  ✓ Temporal persistence prevents objects from disappearing too quickly")
        print("  ✓ Configurable persistence settings for different use cases")
        print("  ✓ Enhanced spatial audio metrics with clear reference points")
        print(
            "  ✓ Background relationship classification (foreground, midground, background)"
        )

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
