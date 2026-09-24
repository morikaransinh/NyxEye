#!/usr/bin/env python3
"""
Test script for depth module integration with bounding boxes.

This script tests the depth module's ability to:
1. Process frames with bounding boxes
2. Return normalized depth values
3. Integrate with other modules
"""

import cv2
import numpy as np
import time
from depth_processor import DepthProcessor, create_depth_processor


def test_depth_processing():
    """Test basic depth processing functionality."""
    print("=== Testing Depth Processing ===")

    # Create processor
    processor = create_depth_processor()
    print("✓ Depth processor created")

    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("✓ Test frame created")

    # Test bounding boxes (simulating object detection results)
    test_bboxes = [
        [100, 100, 200, 200],  # Object 1
        [300, 150, 400, 250],  # Object 2
        [150, 300, 250, 400],  # Object 3
    ]

    print("✓ Test bounding boxes created")

    # Process each bounding box
    results = []
    for i, bbox in enumerate(test_bboxes):
        print(f"Processing bbox {i+1}: {bbox}")

        start_time = time.time()
        result = processor.get_depth_for_spatial_audio(test_frame, bbox)
        end_time = time.time()

        results.append(result)
        print(f"  Raw depth: {result['raw_depth']:.3f}")
        print(f"  Normalized depth: {result['normalized_depth']:.3f}")
        print(f"  Processing time: {end_time - start_time:.3f}s")
        print()

    # Test depth statistics
    stats = processor.get_depth_stats()
    print("Depth statistics:")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")

    return results


def test_normalization_methods():
    """Test different normalization methods."""
    print("\n=== Testing Normalization Methods ===")

    processor = create_depth_processor()
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = [100, 100, 200, 200]

    # Get raw depth
    depth_map = processor.estimate_depth(test_frame)
    raw_depth = processor.get_depth_from_bbox(depth_map, test_bbox)

    print(f"Raw depth: {raw_depth:.3f}")

    # Test different normalization methods
    methods = ["relative", "statistical", "reference"]

    for method in methods:
        normalized = processor.normalize_depth(raw_depth, method)
        print(f"{method.capitalize()} normalization: {normalized:.3f}")


def test_depth_calculation_methods():
    """Test different depth calculation methods."""
    print("\n=== Testing Depth Calculation Methods ===")

    processor = create_depth_processor()
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = [100, 100, 200, 200]

    depth_map = processor.estimate_depth(test_frame)

    methods = ["mean", "median", "min", "max"]

    for method in methods:
        depth_value = processor.get_depth_from_bbox(depth_map, test_bbox, method)
        print(f"{method.capitalize()} method: {depth_value:.3f}")


def test_integration_interface():
    """Test the integration interface for other modules."""
    print("\n=== Testing Integration Interface ===")

    # Simulate object detection results
    detection_results = [
        {"bbox": [100, 100, 200, 200], "class": "water bottle", "confidence": 0.95},
        {"bbox": [300, 150, 400, 250], "class": "water bottle", "confidence": 0.87},
        {"bbox": [150, 300, 250, 400], "class": "water bottle", "confidence": 0.92},
    ]

    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Process with depth module
    processor = create_depth_processor()

    print("Processing detection results with depth:")
    for i, detection in enumerate(detection_results):
        bbox = detection["bbox"]
        result = processor.get_depth_for_spatial_audio(test_frame, bbox)

        print(f"Object {i+1}:")
        print(f"  Bbox: {bbox}")
        print(f"  Class: {detection['class']}")
        print(f"  Confidence: {detection['confidence']:.2f}")
        print(f"  Raw depth: {result['raw_depth']:.3f}")
        print(f"  Normalized depth: {result['normalized_depth']:.3f}")
        print(f"  ROI stats: {result['roi_stats']}")
        print()


def test_performance():
    """Test performance with multiple frames."""
    print("\n=== Testing Performance ===")

    processor = create_depth_processor()
    test_bbox = [100, 100, 200, 200]

    num_frames = 10
    total_time = 0

    print(f"Processing {num_frames} frames...")

    for i in range(num_frames):
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        result = processor.get_depth_for_spatial_audio(test_frame, test_bbox)
        end_time = time.time()

        frame_time = end_time - start_time
        total_time += frame_time

        print(f"Frame {i+1}: {frame_time:.3f}s")

    avg_time = total_time / num_frames
    fps = 1.0 / avg_time

    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Estimated FPS: {fps:.1f}")


def main():
    """Run all tests."""
    print("Depth Module Integration Tests")
    print("=" * 40)

    try:
        # Run tests
        test_depth_processing()
        test_normalization_methods()
        test_depth_calculation_methods()
        test_integration_interface()
        test_performance()

        print("\n" + "=" * 40)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
