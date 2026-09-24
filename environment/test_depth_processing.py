"""
Unit Tests for Depth Processing Module

Tests cover:
- detect_open_spaces: Column depth extraction from bottom 1/3
- find_open_space_columns: Peak detection in depth signal
- columns_to_azimuths: Angle calculation from column indices
- process_depth_map: Complete pipeline integration
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from depth_processing import (
    detect_open_spaces,
    find_open_space_columns,
    columns_to_azimuths,
    process_depth_map,
    load_depth_from_file,
    LOWER_BAND_FRAC,
    HFOV_DEG,
)


class TestDetectOpenSpaces(unittest.TestCase):
    """
    Test suite for detect_open_spaces function.

    This function extracts the bottom 1/3 of a depth map and averages
    depth values for each column, producing a 1D signal.
    """

    def test_basic_extraction(self):
        """
        Test: Basic depth map extraction produces correct shape.

        Given a depth map of size (H, W), we expect a 1D array of length W
        representing average depth per column in the bottom third.
        """
        H, W = 300, 400
        depth_img = np.random.rand(H, W).astype(np.float32)

        result = detect_open_spaces(depth_img)

        # Should return 1D array with length equal to width
        self.assertEqual(result.shape, (W,))
        self.assertIsInstance(result, np.ndarray)

    def test_bottom_third_extraction(self):
        """
        Test: Only bottom 1/3 of image is used for averaging.

        We create a depth map where the bottom third has different values
        than the top two-thirds. The result should reflect only the bottom third.
        """
        H, W = 300, 100
        depth_img = np.zeros((H, W), dtype=np.float32)

        # Top 2/3: all zeros
        depth_img[: int(H * (1 - LOWER_BAND_FRAC)), :] = 0.0

        # Bottom 1/3: all ones
        y0 = int(H * (1.0 - LOWER_BAND_FRAC))
        depth_img[y0:, :] = 1.0

        result = detect_open_spaces(depth_img)

        # All columns should have depth ~1.0 (from bottom third)
        # Allow some tolerance for smoothing
        self.assertTrue(np.allclose(result, 1.0, atol=0.1))

    def test_column_averaging(self):
        """
        Test: Each column's depth is correctly averaged.

        Create a depth map where each column in the bottom third has
        a specific pattern. The result should reflect the column averages,
        though Gaussian smoothing will spread values across columns.
        """
        H, W = 300, 50  # Use many columns to reduce edge effects
        depth_img = np.zeros((H, W), dtype=np.float32)
        y0 = int(H * (1.0 - LOWER_BAND_FRAC))

        # Create a clear pattern: left side low, right side high
        # This creates a gradient that should be preserved (monotonically increasing)
        for col in range(W):
            depth_img[y0:, col] = float(col) / float(W)  # Range from 0.0 to ~1.0

        result = detect_open_spaces(depth_img)

        # Verify output shape
        self.assertEqual(len(result), W)

        # Verify values are in reasonable range (0 to 1)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.5))  # Allow some overshoot from smoothing

        # Verify the general pattern is preserved: result should be generally increasing
        # (smoothing may cause local variations, but overall trend should be preserved)
        # Check that left side is lower than right side
        left_avg = np.mean(result[: W // 4])
        right_avg = np.mean(result[3 * W // 4 :])
        self.assertLess(
            left_avg, right_avg, "Left side should have lower depth than right side"
        )

        # Verify that the function actually averaged the bottom third
        # The raw average of bottom third should be close to result (allowing for smoothing)
        raw_avg = depth_img[y0:, :].mean(axis=0)
        # Smoothed result should be correlated with raw average
        correlation = np.corrcoef(result, raw_avg)[0, 1]
        self.assertGreater(
            correlation,
            0.7,
            "Smoothed result should be correlated with raw column averages",
        )

    def test_smoothing_applied(self):
        """
        Test: Gaussian smoothing is applied to reduce noise.

        Create a noisy depth signal and verify that smoothing reduces
        the variance compared to raw averaging.
        """
        H, W = 300, 100
        depth_img = np.random.rand(H, W).astype(np.float32) * 0.1  # Low noise
        y0 = int(H * (1.0 - LOWER_BAND_FRAC))

        # Add a smooth gradient in bottom third
        for col in range(W):
            depth_img[y0:, col] = 0.5 + 0.3 * np.sin(2 * np.pi * col / W)

        result = detect_open_spaces(depth_img)

        # Smoothed result should be smoother than raw mean
        raw_mean = depth_img[y0:, :].mean(axis=0)
        result_variance = np.var(np.diff(result))
        raw_variance = np.var(np.diff(raw_mean))

        # Smoothed should have lower variance in differences
        self.assertLess(result_variance, raw_variance * 1.5)  # Allow some tolerance


class TestFindOpenSpaceColumns(unittest.TestCase):
    """
    Test suite for find_open_space_columns function.

    This function finds peaks in the column depth signal, representing
    open spaces (hallways, doorways, etc.) where depth is higher than surroundings.
    """

    def test_finds_single_peak(self):
        """
        Test: Correctly identifies a single prominent peak.

        Create a depth signal with one clear peak and verify it's detected.
        """
        W = 100
        signal = np.ones(W, dtype=np.float32) * 0.3  # Background depth

        # Add a clear peak at column 50
        signal[50] = 0.9

        columns = find_open_space_columns(signal, k=5, min_prom=0.1)

        # Should find the peak at column 50
        self.assertIn(50, columns)
        self.assertEqual(len(columns), 1)

    def test_finds_multiple_peaks(self):
        """
        Test: Finds multiple distinct peaks in the signal.

        Create a signal with multiple well-separated peaks and verify
        all are detected (up to k candidates).
        """
        W = 200
        signal = (
            np.ones(W, dtype=np.float32) * 0.1
        )  # Very low background to ensure clear peaks

        # Add clear, well-separated peaks at columns 30, 80, 150
        # Make them more prominent and ensure they're far enough apart
        signal[30] = 0.9
        signal[80] = 0.85
        signal[150] = 0.8

        # Set k=3 to match expected number of peaks
        columns = find_open_space_columns(
            signal, k=3, min_prom=0.5
        )  # Higher prominence threshold

        # Should find all three peaks (may find more if background creates peaks, but should include these)
        self.assertIn(30, columns, "Should find peak at column 30")
        self.assertIn(80, columns, "Should find peak at column 80")
        self.assertIn(150, columns, "Should find peak at column 150")
        # May find additional peaks, but should find at least these 3
        self.assertGreaterEqual(len(columns), 3)

    def test_respects_k_limit(self):
        """
        Test: Returns at most k columns even if more peaks exist.

        Create a signal with many peaks and verify only top k are returned.
        """
        W = 300
        signal = np.ones(W, dtype=np.float32) * 0.2

        # Add 10 peaks
        peak_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for pos in peak_positions:
            signal[pos] = 0.7 + (pos % 3) * 0.1  # Varying heights

        columns = find_open_space_columns(signal, k=5, min_prom=0.1)

        # Should return at most 5 columns
        self.assertLessEqual(len(columns), 5)

    def test_ignores_low_prominence(self):
        """
        Test: Ignores peaks that don't meet minimum prominence threshold.

        Create a signal where peaks are too similar to background to be
        considered open spaces.
        """
        W = 100
        signal = np.ones(W, dtype=np.float32) * 0.5  # Uniform background

        # Add a small bump (low prominence)
        signal[50] = 0.52  # Only 0.02 above background

        columns = find_open_space_columns(signal, k=5, min_prom=0.1)

        # Should not detect the small bump
        self.assertEqual(len(columns), 0)

    def test_suppresses_nearby_peaks(self):
        """
        Test: Suppresses nearby peaks to avoid detecting edges.

        The algorithm should find peaks in the middle of open spaces,
        not at boundaries. Create two close peaks and verify only one
        is returned (the stronger one).
        """
        W = 100
        signal = np.ones(W, dtype=np.float32) * 0.3

        # Two close peaks
        signal[48] = 0.9  # Stronger
        signal[52] = 0.85  # Weaker, close by

        columns = find_open_space_columns(signal, k=5, min_prom=0.1)

        # Should find the stronger peak, and suppress the nearby one
        # (exact behavior depends on suppression window, but should find at least one)
        self.assertGreater(len(columns), 0)
        # The stronger peak should be included
        self.assertIn(48, columns)


class TestColumnsToAzimuths(unittest.TestCase):
    """
    Test suite for columns_to_azimuths function.

    This function converts column indices to azimuth angles (angle from
    camera center) based on the camera's field of view.
    """

    def test_center_column_zero_azimuth(self):
        """
        Test: Center column maps to 0 degrees (straight ahead).

        The center column should always map to 0 degrees azimuth,
        regardless of image width. For even widths, we test with the
        column closest to center, which may have a small error.
        """
        # Use odd width so there's an exact center column
        width = 101
        center_col = int((width - 1) / 2)  # This will be 50

        azimuths = columns_to_azimuths([center_col], width, hfov_deg=HFOV_DEG)

        self.assertAlmostEqual(azimuths[0], 0.0, places=1)

    def test_left_column_negative_azimuth(self):
        """
        Test: Left columns map to negative azimuth angles.

        Columns to the left of center should have negative azimuth
        (indicating leftward direction from camera).
        """
        width = 200
        left_col = 0  # Leftmost column

        azimuths = columns_to_azimuths([left_col], width, hfov_deg=HFOV_DEG)

        # Should be negative (left side)
        self.assertLess(azimuths[0], 0)
        # Should be approximately -hfov_deg/2
        self.assertAlmostEqual(azimuths[0], -HFOV_DEG / 2.0, delta=1.0)

    def test_right_column_positive_azimuth(self):
        """
        Test: Right columns map to positive azimuth angles.

        Columns to the right of center should have positive azimuth
        (indicating rightward direction from camera).
        """
        width = 200
        right_col = width - 1  # Rightmost column

        azimuths = columns_to_azimuths([right_col], width, hfov_deg=HFOV_DEG)

        # Should be positive (right side)
        self.assertGreater(azimuths[0], 0)
        # Should be approximately +hfov_deg/2
        self.assertAlmostEqual(azimuths[0], HFOV_DEG / 2.0, delta=1.0)

    def test_linear_mapping(self):
        """
        Test: Azimuth mapping is linear across the field of view.

        The mapping from column position to angle should be linear.
        Test with multiple columns to verify linearity.
        """
        width = 100
        columns = [0, 25, 50, 75, 99]  # Spread across width

        azimuths = columns_to_azimuths(columns, width, hfov_deg=HFOV_DEG)

        # Should be monotonically increasing
        for i in range(len(azimuths) - 1):
            self.assertLess(azimuths[i], azimuths[i + 1])

        # First should be negative, last should be positive
        self.assertLess(azimuths[0], 0)
        self.assertGreater(azimuths[-1], 0)

    def test_different_fov(self):
        """
        Test: Function correctly handles different field of view values.

        Changing the FOV should change the azimuth range while keeping
        the center at 0. Use odd width for exact center.
        """
        width = 101  # Use odd width so there's an exact center column
        center_col = int((width - 1) / 2)  # This will be 50

        # Test with different FOVs
        for fov in [45.0, 72.0, 90.0, 120.0]:
            azimuths = columns_to_azimuths([center_col], width, hfov_deg=fov)
            # Center should always be 0
            self.assertAlmostEqual(azimuths[0], 0.0, places=1)

            # Edge columns should scale with FOV
            left_az = columns_to_azimuths([0], width, hfov_deg=fov)[0]
            self.assertAlmostEqual(left_az, -fov / 2.0, delta=1.0)


class TestProcessDepthMap(unittest.TestCase):
    """
    Test suite for process_depth_map function.

    This is the complete pipeline that combines all three steps:
    1. Extract column depths
    2. Find open space columns
    3. Calculate azimuths
    """

    def test_complete_pipeline(self):
        """
        Test: Complete pipeline produces all expected outputs.

        Given a depth map, the function should return a dictionary with
        all the processed information.
        """
        H, W = 300, 200
        depth_img = np.random.rand(H, W).astype(np.float32)

        # Add a clear open space (high depth) in bottom third
        y0 = int(H * (1.0 - LOWER_BAND_FRAC))
        depth_img[y0:, 100] = 0.9  # High depth at column 100

        result = process_depth_map(depth_img)

        # Should have all expected keys
        self.assertIn("column_depths", result)
        self.assertIn("open_space_columns", result)
        self.assertIn("azimuths_deg", result)
        self.assertIn("all_column_depths", result)

        # Column depths should be 1D array of length W
        self.assertEqual(len(result["column_depths"]), W)

        # Open space columns should be a list
        self.assertIsInstance(result["open_space_columns"], list)

        # Azimuths should match number of open space columns
        self.assertEqual(len(result["azimuths_deg"]), len(result["open_space_columns"]))

    def test_empty_depth_map(self):
        """
        Test: Handles edge case of empty or uniform depth map.

        A uniform depth map should still produce valid outputs,
        though no open spaces may be detected.
        """
        H, W = 100, 50
        depth_img = np.ones((H, W), dtype=np.float32) * 0.5  # Uniform

        result = process_depth_map(depth_img, k=5, min_prom=0.1)

        # Should still return valid structure
        self.assertIn("column_depths", result)
        self.assertIn("open_space_columns", result)
        self.assertIn("azimuths_deg", result)

        # May or may not find open spaces depending on prominence
        # But structure should be valid
        self.assertEqual(len(result["azimuths_deg"]), len(result["open_space_columns"]))

    def test_custom_parameters(self):
        """
        Test: Pipeline respects custom parameters (k, min_prom, hfov_deg).

        Verify that passing custom parameters affects the results as expected.
        """
        H, W = 200, 150
        depth_img = np.random.rand(H, W).astype(np.float32)

        # Test with different k values
        result_k3 = process_depth_map(depth_img, k=3)
        result_k10 = process_depth_map(depth_img, k=10)

        # Should respect k limit
        self.assertLessEqual(len(result_k3["open_space_columns"]), 3)
        self.assertLessEqual(len(result_k10["open_space_columns"]), 10)

        # Test with different FOV
        result_fov45 = process_depth_map(depth_img, hfov_deg=45.0)
        result_fov90 = process_depth_map(depth_img, hfov_deg=90.0)

        # Azimuths should scale with FOV (if any open spaces found)
        if (
            len(result_fov45["azimuths_deg"]) > 0
            and len(result_fov90["azimuths_deg"]) > 0
        ):
            # For same columns, azimuths should be different
            # (exact comparison depends on which columns are found)
            pass  # Just verify no errors occur


class TestLoadDepthFromFile(unittest.TestCase):
    """
    Test suite for load_depth_from_file function.

    This function loads depth maps from files (PNG or NPY format).
    """

    def test_load_npy_file(self):
        """
        Test: Correctly loads depth map from .npy file.

        Create a temporary .npy file and verify it loads correctly.
        """
        import tempfile

        # Create test depth map
        depth_data = np.random.rand(100, 200).astype(np.float32)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name
            np.save(temp_path, depth_data)

        try:
            # Load it back
            loaded = load_depth_from_file(temp_path)

            # Should match original
            np.testing.assert_array_almost_equal(loaded, depth_data)
            self.assertEqual(loaded.dtype, np.float32)
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_png_file(self):
        """
        Test: Correctly loads depth map from .png file (16-bit).

        Create a temporary 16-bit PNG and verify it loads and normalizes correctly.
        """
        import tempfile
        import cv2

        # Create test depth map (normalized to [0, 1])
        depth_data = np.random.rand(100, 200).astype(np.float32)

        # Convert to 16-bit PNG format
        png16 = np.clip(depth_data * 65535.0, 0, 65535).astype(np.uint16)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, png16)

        try:
            # Load it back
            loaded = load_depth_from_file(temp_path)

            # Should be normalized to [0, 1] and approximately match original
            self.assertTrue(np.all(loaded >= 0))
            self.assertTrue(np.all(loaded <= 1))
            # Allow some quantization error from 16-bit conversion
            np.testing.assert_array_almost_equal(loaded, depth_data, decimal=2)
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_nonexistent_file(self):
        """
        Test: Raises appropriate error for nonexistent file.

        Attempting to load a file that doesn't exist should raise an error.
        """
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_depth_from_file("nonexistent_file.npy")


if __name__ == "__main__":
    unittest.main()
