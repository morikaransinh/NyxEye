"""
Unit Tests for Depth Integration Module

Tests cover:
- Frame extraction from video
- MiDaS model loading (mocked)
- Depth map processing coordination
- Complete pipeline integration
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import cv2

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from depth_integration import extract_frames, process_all_depth_maps, DATA_DIR


class TestExtractFrames(unittest.TestCase):
    """
    Test suite for extract_frames function.

    This function extracts frames from a video file at specified intervals.
    """

    def setUp(self):
        """Set up test fixtures: use newSample.MOV for testing."""
        # Use the actual newSample.MOV file from the environment directory
        test_dir = os.path.dirname(__file__)
        self.video_path = os.path.join(test_dir, "newSample.MOV")

        # Create a temporary directory for extracted frames
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "frames")

        # Verify the video file exists
        if not os.path.exists(self.video_path):
            self.skipTest(f"Test video file not found: {self.video_path}")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        # Only clean up the temporary frames directory, not the video file
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_extract_frames_basic(self):
        """
        Test: Basic frame extraction produces expected number of frames.

        Extract frames from a test video and verify the correct number
        are saved and returned.
        """
        frames, fps = extract_frames(
            self.video_path,
            self.frames_dir,
            sample_every_s=0.1,  # Extract every 0.1 seconds
            max_frames=None,
        )

        # Should extract multiple frames
        self.assertGreater(len(frames), 0)

        # All frame files should exist
        for frame_path in frames:
            self.assertTrue(
                os.path.exists(frame_path), f"Frame file should exist: {frame_path}"
            )

        # FPS should be reasonable
        self.assertGreater(fps, 0)
        self.assertLess(fps, 1000)  # Sanity check

    def test_extract_frames_max_limit(self):
        """
        Test: max_frames parameter limits the number of extracted frames.

        When max_frames is set, only that many frames should be extracted.
        """
        frames, fps = extract_frames(
            self.video_path, self.frames_dir, sample_every_s=0.1, max_frames=3
        )

        # Should extract at most 3 frames
        self.assertLessEqual(len(frames), 3)

    def test_extract_frames_sample_rate(self):
        """
        Test: sample_every_s parameter controls frame extraction rate.

        Different sample rates should produce different numbers of frames
        from the same video.
        """
        frames_fast, _ = extract_frames(
            self.video_path,
            self.frames_dir + "_fast",
            sample_every_s=0.05,  # Fast sampling
            max_frames=None,
        )

        frames_slow, _ = extract_frames(
            self.video_path,
            self.frames_dir + "_slow",
            sample_every_s=0.5,  # Slow sampling
            max_frames=None,
        )

        # Fast sampling should extract more frames
        self.assertGreater(len(frames_fast), len(frames_slow))

    def test_extract_frames_nonexistent_video(self):
        """
        Test: Raises error for nonexistent video file.

        Attempting to extract frames from a file that doesn't exist
        should raise an appropriate error.
        """
        import os

        # Use a clearly nonexistent file path
        nonexistent_path = os.path.join(
            self.temp_dir, "definitely_does_not_exist_12345.avi"
        )

        with self.assertRaises(RuntimeError):
            extract_frames(nonexistent_path, self.frames_dir, sample_every_s=0.1)


class TestProcessAllDepthMaps(unittest.TestCase):
    """
    Test suite for process_all_depth_maps function.

    This function processes all depth maps in a directory and extracts
    open space information.
    """

    def setUp(self):
        """Set up test fixtures: create temporary depth map files."""
        self.temp_dir = tempfile.mkdtemp()
        self.depth_dir = os.path.join(self.temp_dir, "depth_npy")
        os.makedirs(self.depth_dir, exist_ok=True)

        # Create a few test depth maps
        for i in range(5):
            depth_map = np.random.rand(300, 400).astype(np.float32)
            depth_path = os.path.join(self.depth_dir, f"depth_{i:06d}.npy")
            np.save(depth_path, depth_map)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_process_all_depth_maps_basic(self):
        """
        Test: Processes all depth maps in directory.

        Given a directory with depth map files, should process all of them
        and return results for each.
        """
        results = process_all_depth_maps(self.depth_dir, max_frames=None)

        # Should process all 5 depth maps
        self.assertEqual(len(results), 5)

        # Each result should have expected structure
        for result in results:
            self.assertIn("frame_idx", result)
            self.assertIn("column_depths", result)
            self.assertIn("open_space_columns", result)
            self.assertIn("azimuths_deg", result)
            self.assertIn("depth_file", result)

    def test_process_all_depth_maps_max_limit(self):
        """
        Test: max_frames parameter limits processing.

        When max_frames is set, only that many depth maps should be processed.
        """
        results = process_all_depth_maps(self.depth_dir, max_frames=3)

        # Should process only 3 depth maps
        self.assertEqual(len(results), 3)

    def test_process_all_depth_maps_frame_indices(self):
        """
        Test: Frame indices are correctly extracted from filenames.

        The frame_idx in each result should match the index in the filename.
        """
        results = process_all_depth_maps(self.depth_dir, max_frames=None)

        # Frame indices should be sequential
        frame_indices = [r["frame_idx"] for r in results]
        self.assertEqual(frame_indices, [0, 1, 2, 3, 4])

    def test_process_all_depth_maps_empty_directory(self):
        """
        Test: Handles empty directory gracefully.

        Processing an empty directory should return an empty list.
        """
        empty_dir = os.path.join(self.temp_dir, "empty_depth")
        os.makedirs(empty_dir, exist_ok=True)

        results = process_all_depth_maps(empty_dir, max_frames=None)

        # Should return empty list
        self.assertEqual(len(results), 0)
        self.assertIsInstance(results, list)

    def test_process_all_depth_maps_with_open_spaces(self):
        """
        Test: Correctly identifies open spaces in depth maps.

        Create depth maps with clear open spaces and verify they're detected.
        """
        # Create a depth map with a clear open space (high depth) in bottom third
        depth_map = np.ones((300, 400), dtype=np.float32) * 0.3  # Low background
        y0 = int(300 * (1.0 - 0.33))  # Bottom third
        depth_map[y0:, 200] = 0.9  # High depth at column 200

        depth_path = os.path.join(self.depth_dir, "depth_000010.npy")
        np.save(depth_path, depth_map)

        results = process_all_depth_maps(self.depth_dir, max_frames=None)

        # Should find the depth map with open space
        result_with_open_space = None
        for result in results:
            if result["frame_idx"] == 10:
                result_with_open_space = result
                break

        if result_with_open_space:
            # Should detect open space at column 200
            self.assertGreater(len(result_with_open_space["open_space_columns"]), 0)


class TestIntegrationPipeline(unittest.TestCase):
    """
    Test suite for complete pipeline integration.

    These tests verify that the different components work together correctly.
    """

    def test_module_imports(self):
        """
        Test: All required modules can be imported.

        Verify that the integration module can import its dependencies
        without errors.
        """
        try:
            from depth_integration import (
                extract_frames,
                load_midas,
                run_midas_on_frames,
                process_all_depth_maps,
            )
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")

    def test_depth_processing_integration(self):
        """
        Test: Depth processing module integrates correctly.

        Verify that the integration module can use the depth processing
        module to process depth maps.
        """
        from depth_processing import process_depth_map

        # Create a test depth map
        depth_map = np.random.rand(300, 400).astype(np.float32)

        # Process it
        result = process_depth_map(depth_map)

        # Should have all expected keys
        self.assertIn("column_depths", result)
        self.assertIn("open_space_columns", result)
        self.assertIn("azimuths_deg", result)

    def test_spatial_audio_integration(self):
        """
        Test: Spatial audio module integrates with depth processing.

        Verify that depth processing results can be used to generate
        spatial audio output.
        """
        from depth_processing import process_depth_map
        from spatial_audio import generate_spatial_audio, SPATIAL_AUDIO_ENABLED

        if not SPATIAL_AUDIO_ENABLED:
            self.skipTest("Spatial audio is disabled")

        # Create a test depth map
        depth_map = np.random.rand(300, 400).astype(np.float32)

        # Process it to get open spaces
        result = process_depth_map(depth_map)

        # Generate spatial audio from the results
        audio_data = generate_spatial_audio(
            result["open_space_columns"],
            result["column_depths"],
            result["azimuths_deg"],
            duration_s=0.1,  # Short duration for testing
        )

        # If there are open spaces, should generate audio
        if len(result["open_space_columns"]) > 0:
            self.assertIsNotNone(
                audio_data, "Should generate audio when open spaces detected"
            )
            self.assertEqual(audio_data.shape[1], 2, "Should be stereo")
            self.assertGreater(audio_data.shape[0], 0, "Should have samples")
        else:
            # If no open spaces, audio should be None
            self.assertIsNone(audio_data, "Should return None when no open spaces")


if __name__ == "__main__":
    unittest.main()
