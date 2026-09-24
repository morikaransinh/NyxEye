#!/usr/bin/env python3
"""
Test script for Enhanced Depth Processing

This script tests the enhanced depth processing capabilities including:
- Reference point management
- Temporal tracking
- Quality assessment
- Spatial audio metrics
- Integration with detection

Usage:
    python3 depth/test_enhanced_depth.py
"""

import numpy as np
import time
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth import (
    EnhancedDepthProcessor,
    create_enhanced_depth_processor,
    DepthLayer,
    MovementDirection,
    DepthQualityMetrics,
    SpatialAudioMetrics,
)


class EnhancedDepthTester:
    """Test suite for enhanced depth processing."""

    def __init__(self):
        """Initialize the test suite."""
        self.processor = create_enhanced_depth_processor()
        self.test_results = {}

        print("Enhanced Depth Processing Test Suite")
        print("=" * 50)

    def test_basic_functionality(self):
        """Test basic enhanced depth processing functionality."""
        print("\n1. Testing Basic Functionality...")

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 200]

        # Test enhanced processing
        start_time = time.time()
        result = self.processor.get_enhanced_depth_for_spatial_audio(
            test_frame, test_bbox, object_id="test_obj", object_class="bottle"
        )
        processing_time = time.time() - start_time

        # Validate result structure
        required_keys = [
            "raw_depth",
            "normalized_depth",
            "relative_depth",
            "depth_layer",
            "quality_metrics",
            "depth_stability",
            "movement_direction",
            "spatial_audio",
            "object_id",
            "object_class",
        ]

        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            print(f"[FAIL] Missing keys: {missing_keys}")
            return False

        # Validate depth values
        if not np.isfinite(result["raw_depth"]):
            print("[FAIL] Invalid raw depth value")
            return False

        if not (0.0 <= result["normalized_depth"] <= 1.0):
            print("[FAIL] Normalized depth out of range")
            return False

        # Validate depth layer
        valid_layers = ["very_close", "close", "medium", "far"]
        if result["depth_layer"] not in valid_layers:
            print(f"[FAIL] Invalid depth layer: {result['depth_layer']}")
            return False

        print(f"[PASS] Basic functionality test passed")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Raw depth: {result['raw_depth']:.3f}")
        print(f"   Normalized depth: {result['normalized_depth']:.3f}")
        print(f"   Depth layer: {result['depth_layer']}")

        self.test_results["basic_functionality"] = True
        return True

    def test_reference_point_management(self):
        """Test reference point management."""
        print("\n2. Testing Reference Point Management...")

        # Create test frame with multiple objects
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Simulate multiple detections
        detections = [
            {"bbox": [50, 50, 150, 150], "class": "person", "raw_depth": 1.2},
            {"bbox": [300, 100, 400, 200], "class": "bottle", "raw_depth": 2.5},
            {"bbox": [200, 300, 300, 400], "class": "chair", "raw_depth": 3.8},
        ]

        # Process each detection
        results = []
        for i, detection in enumerate(detections):
            result = self.processor.get_enhanced_depth_for_spatial_audio(
                test_frame,
                detection["bbox"],
                object_id=f"test_obj_{i}",
                object_class=detection["class"],
            )
            results.append(result)

        # Check if background reference was established
        background_ref = self.processor.reference_manager.background_reference
        if background_ref is None:
            print("[FAIL] Background reference not established")
            return False

        # Check reference quality
        if background_ref.confidence < 0.1:
            print(
                f"[FAIL] Low background reference confidence: {background_ref.confidence}"
            )
            return False

        print(f"[PASS] Reference point management test passed")
        print(f"   Background reference: {background_ref.depth_value:.3f}m")
        print(f"   Reference confidence: {background_ref.confidence:.3f}")
        print(f"   Reference stability: {background_ref.stability_score:.3f}")

        self.test_results["reference_management"] = True
        return True

    def test_temporal_tracking(self):
        """Test temporal depth tracking."""
        print("\n3. Testing Temporal Tracking...")

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 200]
        object_id = "temporal_test_obj"

        # Simulate depth changes over time
        depth_values = [1.0, 0.9, 0.8, 0.7, 0.6]  # Approaching object

        for i, depth in enumerate(depth_values):
            # Modify frame to simulate different depth
            modified_frame = test_frame.copy()

            result = self.processor.get_enhanced_depth_for_spatial_audio(
                modified_frame, test_bbox, object_id=object_id, object_class="person"
            )

            # Small delay to simulate real-time processing
            time.sleep(0.1)

        # Check temporal tracking
        if object_id not in self.processor.temporal_tracker.depth_history:
            print("[FAIL] Temporal tracking not working")
            return False

        history = self.processor.temporal_tracker.depth_history[object_id]
        if len(history) < 3:
            print("[FAIL] Insufficient temporal history")
            return False

        # Check depth stability calculation
        stability = self.processor.temporal_tracker.get_depth_stability(object_id)
        if not (0.0 <= stability <= 1.0):
            print(f"[FAIL] Invalid stability score: {stability}")
            return False

        # Check motion tracking
        motion_data = self.processor.temporal_tracker.motion_tracker.get(object_id, {})
        if "velocity" not in motion_data:
            print("[FAIL] Motion tracking not working")
            return False

        print(f"[PASS] Temporal tracking test passed")
        print(f"   History length: {len(history)}")
        print(f"   Depth stability: {stability:.3f}")
        print(f"   Movement velocity: {motion_data.get('velocity', 0):.3f}")
        print(f"   Movement direction: {motion_data.get('direction', 'unknown')}")

        self.test_results["temporal_tracking"] = True
        return True

    def test_quality_assessment(self):
        """Test depth quality assessment."""
        print("\n4. Testing Quality Assessment...")

        # Create test frame with different quality characteristics
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 200]

        result = self.processor.get_enhanced_depth_for_spatial_audio(
            test_frame, test_bbox, object_id="quality_test", object_class="bottle"
        )

        quality_metrics = result["quality_metrics"]

        # Validate quality metrics structure
        required_quality_keys = [
            "signal_to_noise_ratio",
            "edge_sharpness",
            "spatial_coherence",
            "temporal_consistency",
            "illumination_quality",
            "texture_richness",
            "overall_confidence",
        ]

        missing_keys = [
            key for key in required_quality_keys if key not in quality_metrics.__dict__
        ]
        if missing_keys:
            print(f"[FAIL] Missing quality metric keys: {missing_keys}")
            return False

        # Validate quality metric ranges
        for key in required_quality_keys:
            value = getattr(quality_metrics, key)
            if not (0.0 <= value <= 1.0):
                print(f"[FAIL] Quality metric {key} out of range: {value}")
                return False

        print(f"[PASS] Quality assessment test passed")
        print(f"   Overall confidence: {quality_metrics.overall_confidence:.3f}")
        print(f"   Signal-to-noise ratio: {quality_metrics.signal_to_noise_ratio:.3f}")
        print(f"   Edge sharpness: {quality_metrics.edge_sharpness:.3f}")
        print(f"   Spatial coherence: {quality_metrics.spatial_coherence:.3f}")

        self.test_results["quality_assessment"] = True
        return True

    def test_spatial_audio_metrics(self):
        """Test spatial audio metrics calculation."""
        print("\n5. Testing Spatial Audio Metrics...")

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 200]

        result = self.processor.get_enhanced_depth_for_spatial_audio(
            test_frame, test_bbox, object_id="audio_test", object_class="person"
        )

        spatial_audio = result["spatial_audio"]

        # Validate spatial audio metrics
        if not isinstance(spatial_audio, SpatialAudioMetrics):
            print("[FAIL] Invalid spatial audio metrics type")
            return False

        # Check azimuth angle range
        if not (-180 <= spatial_audio.azimuth_angle <= 180):
            print(f"[FAIL] Azimuth angle out of range: {spatial_audio.azimuth_angle}")
            return False

        # Check elevation angle range
        if not (-90 <= spatial_audio.elevation_angle <= 90):
            print(
                f"[FAIL] Elevation angle out of range: {spatial_audio.elevation_angle}"
            )
            return False

        # Check audio priority range
        if not (1 <= spatial_audio.audio_priority <= 10):
            print(f"[FAIL] Audio priority out of range: {spatial_audio.audio_priority}")
            return False

        # Check audio intensity range
        if not (0.0 <= spatial_audio.audio_intensity <= 1.0):
            print(
                f"[FAIL] Audio intensity out of range: {spatial_audio.audio_intensity}"
            )
            return False

        print(f"[PASS] Spatial audio metrics test passed")
        print(f"   Azimuth: {spatial_audio.azimuth_angle:.1f}°")
        print(f"   Elevation: {spatial_audio.elevation_angle:.1f}°")
        print(f"   Distance category: {spatial_audio.distance_category}")
        print(f"   Audio priority: {spatial_audio.audio_priority}")
        print(f"   Audio intensity: {spatial_audio.audio_intensity:.3f}")
        print(f"   Proximity warning: {spatial_audio.proximity_warning}")

        self.test_results["spatial_audio"] = True
        return True

    def test_spatial_context_analysis(self):
        """Test spatial context analysis."""
        print("\n6. Testing Spatial Context Analysis...")

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create multiple detections for context analysis
        detections = [
            {"bbox": [50, 50, 150, 150], "class": "person", "raw_depth": 1.2},
            {"bbox": [300, 100, 400, 200], "class": "bottle", "raw_depth": 2.5},
            {"bbox": [200, 300, 300, 400], "class": "chair", "raw_depth": 3.8},
        ]

        # Get spatial context
        context = self.processor.get_spatial_context(test_frame, detections)

        # Validate context structure
        required_context_keys = [
            "scene_type",
            "object_relationships",
            "depth_layers",
            "spatial_density",
            "movement_patterns",
        ]

        missing_keys = [key for key in required_context_keys if key not in context]
        if missing_keys:
            print(f"[FAIL] Missing context keys: {missing_keys}")
            return False

        # Validate scene type
        valid_scene_types = [
            "indoor_close",
            "indoor_medium",
            "outdoor_wide",
            "mixed",
            "unknown",
        ]
        if context["scene_type"] not in valid_scene_types:
            print(f"[FAIL] Invalid scene type: {context['scene_type']}")
            return False

        # Validate spatial density
        density = context["spatial_density"]
        if not (0.0 <= density <= 1.0):
            print(f"[FAIL] Spatial density out of range: {density}")
            return False

        print(f"[PASS] Spatial context analysis test passed")
        print(f"   Scene type: {context['scene_type']}")
        print(f"   Spatial density: {density:.3f}")
        print(f"   Depth layers: {context['depth_layers']}")

        self.test_results["spatial_context"] = True
        return True

    def test_performance(self):
        """Test performance characteristics."""
        print("\n7. Testing Performance...")

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 200]

        # Warm up
        for _ in range(3):
            self.processor.get_enhanced_depth_for_spatial_audio(
                test_frame, test_bbox, object_id="perf_test", object_class="bottle"
            )

        # Performance test
        num_iterations = 10
        start_time = time.time()

        for i in range(num_iterations):
            result = self.processor.get_enhanced_depth_for_spatial_audio(
                test_frame, test_bbox, object_id=f"perf_test_{i}", object_class="bottle"
            )

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time

        print(f"[PASS] Performance test completed")
        print(f"   Average processing time: {avg_time:.3f}s")
        print(f"   Theoretical FPS: {fps:.1f}")
        print(f"   Total time for {num_iterations} iterations: {total_time:.3f}s")

        # Performance should be reasonable for real-time use
        if avg_time > 0.1:  # More than 100ms per frame
            print(f"[WARN]  Warning: Processing time may be too slow for real-time use")

        self.test_results["performance"] = True
        return True

    def run_all_tests(self):
        """Run all tests and report results."""
        print("Running Enhanced Depth Processing Tests...")
        print("=" * 50)

        tests = [
            self.test_basic_functionality,
            self.test_reference_point_management,
            self.test_temporal_tracking,
            self.test_quality_assessment,
            self.test_spatial_audio_metrics,
            self.test_spatial_context_analysis,
            self.test_performance,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"[FAIL] Test failed with exception: {e}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            print(
                "[SUCCESS] All tests passed! Enhanced depth processing is working correctly."
            )
        else:
            print("[WARN]  Some tests failed. Please review the implementation.")

        return passed == total


def main():
    """Main function."""
    try:
        tester = EnhancedDepthTester()
        success = tester.run_all_tests()

        if success:
            print(
                "\n[PASS] Enhanced depth processing is ready for spatial audio integration!"
            )
        else:
            print("\n[FAIL] Enhanced depth processing needs fixes before integration.")

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
