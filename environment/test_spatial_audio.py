"""
Unit Tests for Spatial Audio Module
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from spatial_audio import (
    SPATIAL_AUDIO_ENABLED,
    play_positioned_sound,
    map_columns_to_fov,
    generate_spatial_audio,
    generate_continuous_spatial_audio,
    azimuth_to_pan,
    depth_to_frequency,
    depth_to_gain,
    pan_stereo,
    generate_filtered_noise,
    add_reverb,
    export_spatial_audio,
    CAMERA_FOV_DEG,
    SAMPLE_RATE
)


class TestSpatialAudioEnabled(unittest.TestCase):
    """
    Test suite for spatial audio module when enabled.
    
    These tests verify that the module correctly implements spatial audio
    functionality.
    """
    
    def test_module_enabled(self):
        """
        Test: Module is correctly marked as enabled.
        
        The SPATIAL_AUDIO_ENABLED flag should be True to indicate
        that spatial audio functionality is implemented.
        """
        self.assertTrue(SPATIAL_AUDIO_ENABLED, 
                       "Spatial audio should be enabled")
    
    def test_map_columns_to_fov(self):
        """
        Test: map_columns_to_fov correctly maps columns to FOV angles.
        
        Columns should be mapped to angles within the camera's FOV.
        Center column should map to 0°, leftmost to negative, rightmost to positive.
        """
        n_columns = 101  # Use odd number for exact center
        center_col = n_columns // 2  # This is 50, the exact center
        
        # Center column should map to 0°
        result = map_columns_to_fov([center_col], n_columns=n_columns, fov_deg=72.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.0, places=1, 
                              msg="Center column should map to 0°")
        
        # Left column should map to negative angle
        result = map_columns_to_fov([0], n_columns=n_columns, fov_deg=72.0)
        self.assertLess(result[0], 0, "Left column should map to negative angle")
        
        # Right column should map to positive angle
        result = map_columns_to_fov([n_columns-1], n_columns=n_columns, fov_deg=72.0)
        self.assertGreater(result[0], 0, "Right column should map to positive angle")
    
    def test_generate_spatial_audio_basic(self):
        """
        Test: generate_spatial_audio creates valid audio output.
        
        When given valid open space columns and depths, should generate
        a stereo audio array.
        """
        open_space_columns = [30, 50, 70]
        column_depths = np.ones(100) * 0.5  # Uniform depth
        azimuths_deg = [-10.0, 0.0, 10.0]
        
        result = generate_spatial_audio(
            open_space_columns,
            column_depths,
            azimuths_deg,
            duration_s=0.1  # Short duration for testing
        )
        
        self.assertIsNotNone(result, "Should generate audio when enabled")
        self.assertEqual(result.shape[1], 2, "Should be stereo (2 channels)")
        self.assertGreater(result.shape[0], 0, "Should have samples")
    
    def test_generate_spatial_audio_empty(self):
        """
        Test: generate_spatial_audio handles empty input gracefully.
        
        When no open spaces are detected, should return None.
        """
        result = generate_spatial_audio(
            [],
            np.random.rand(100),
            [],
            duration_s=0.1
        )
        
        self.assertIsNone(result, "Should return None for empty input")


class TestSpatialAudioInterface(unittest.TestCase):
    """
    Test suite for spatial audio function interfaces.
    
    These tests verify that the function signatures are correct and
    handle edge cases gracefully, even when disabled.
    """
    
    def test_play_positioned_sound_signature(self):
        """
        Test: play_positioned_sound accepts correct parameters.
        
        Verify that the function can be called with expected parameters
        without raising type errors.
        """
        audio_data = np.random.rand(1000).astype(np.float32)
        
        # Should not raise errors with valid parameters
        try:
            result = play_positioned_sound(
                audio_data,
                column=50,
                n_columns=100,
                sample_rate=SAMPLE_RATE
            )
            self.assertIsInstance(result, bool)
        except TypeError as e:
            self.fail(f"Function signature incorrect: {e}")
    
    def test_map_columns_to_fov_signature(self):
        """
        Test: map_columns_to_fov accepts correct parameters.
        
        Verify that the function can be called with expected parameters
        without raising type errors.
        """
        columns = [10, 50, 90]
        
        # Should not raise errors with valid parameters
        try:
            result = map_columns_to_fov(
                columns,
                n_columns=100,
                fov_deg=CAMERA_FOV_DEG
            )
            self.assertIsInstance(result, list)
        except TypeError as e:
            self.fail(f"Function signature incorrect: {e}")
    
    def test_generate_spatial_audio_signature(self):
        """
        Test: generate_spatial_audio accepts correct parameters.
        
        Verify that the function can be called with expected parameters
        without raising type errors.
        """
        open_space_columns = [10, 50, 90]
        column_depths = np.random.rand(100).astype(np.float32)
        azimuths_deg = [-20.0, 0.0, 20.0]
        
        # Should not raise errors with valid parameters
        try:
            result = generate_spatial_audio(
                open_space_columns,
                column_depths,
                azimuths_deg,
                duration_s=0.1
            )
            # Result should be None (empty) or numpy array (when enabled)
            self.assertTrue(result is None or isinstance(result, np.ndarray))
        except TypeError as e:
            self.fail(f"Function signature incorrect: {e}")
    
    def test_edge_cases(self):
        """
        Test: Functions handle edge cases gracefully.
        
        Test with edge cases like empty lists, zero columns, etc.
        """
        # Empty columns
        result1 = map_columns_to_fov([], n_columns=100, fov_deg=CAMERA_FOV_DEG)
        self.assertIsInstance(result1, list)
        
        # Single column
        result2 = map_columns_to_fov([50], n_columns=100, fov_deg=CAMERA_FOV_DEG)
        self.assertIsInstance(result2, list)
        
        # Empty audio data
        audio_data = np.array([])
        result3 = play_positioned_sound(audio_data, column=0, n_columns=100)
        self.assertIsInstance(result3, bool)


class TestSpatialAudioConstants(unittest.TestCase):
    """
    Test suite for spatial audio constants.
    
    Verify that configuration constants are set correctly.
    """
    
    def test_camera_fov_constant(self):
        """
        Test: CAMERA_FOV_DEG is set to iPhone Wide lens value.
        
        The camera FOV should be 72 degrees for iPhone Wide lens as specified.
        """
        self.assertEqual(CAMERA_FOV_DEG, 72.0,
                        "Camera FOV should be 72° for iPhone Wide lens")
    
    def test_sample_rate_constant(self):
        """
        Test: SAMPLE_RATE is set to standard audio rate.
        
        The sample rate should be 48000 Hz (matches original implementation).
        """
        self.assertEqual(SAMPLE_RATE, 48000,
                        "Sample rate should be 48000 Hz")


class TestSpatialAudioFunctions(unittest.TestCase):
    """
    Test suite for individual spatial audio helper functions.
    
    Tests the core audio processing functions: panning, frequency mapping, etc.
    """
    
    def test_azimuth_to_pan(self):
        """
        Test: azimuth_to_pan correctly converts angles to pan values.
        
        Center (0°) should map to 0.5, left to <0.5, right to >0.5.
        """
        # Center should be 0.5
        pan_center = azimuth_to_pan(0.0)
        self.assertAlmostEqual(pan_center, 0.5, places=2,
                              msg="Center azimuth (0°) should map to pan 0.5")
        
        # Left should be < 0.5
        pan_left = azimuth_to_pan(-36.0)  # Left edge of 72° FOV
        self.assertLess(pan_left, 0.5, "Left azimuth should map to pan < 0.5")
        
        # Right should be > 0.5
        pan_right = azimuth_to_pan(36.0)  # Right edge of 72° FOV
        self.assertGreater(pan_right, 0.5, "Right azimuth should map to pan > 0.5")
    
    def test_depth_to_frequency(self):
        """
        Test: depth_to_frequency maps depth to frequency correctly.
        
        Far depth (1.0) should map to lower frequency than close depth (0.0).
        """
        freq_close = depth_to_frequency(0.0)  # Close
        freq_far = depth_to_frequency(1.0)     # Far
        
        self.assertGreater(freq_close, freq_far,
                          "Close objects should have higher frequency than far objects")
        self.assertGreater(freq_close, 0, "Frequency should be positive")
        self.assertGreater(freq_far, 0, "Frequency should be positive")
    
    def test_depth_to_gain(self):
        """
        Test: depth_to_gain maps depth to gain correctly.
        
        Gain should be in reasonable range [0, 1] for all depth values.
        """
        gain_close = depth_to_gain(0.0)
        gain_far = depth_to_gain(1.0)
        
        self.assertGreater(gain_close, 0, "Gain should be positive")
        self.assertLess(gain_close, 2.0, "Gain should be reasonable")
        self.assertGreater(gain_far, 0, "Gain should be positive")
        self.assertLess(gain_far, 2.0, "Gain should be reasonable")
    
    def test_pan_stereo(self):
        """
        Test: pan_stereo correctly pans mono to stereo.
        
        Should produce stereo output with correct shape and panning.
        """
        mono = np.random.rand(1000).astype(np.float32)
        
        # Center pan
        stereo_center = pan_stereo(mono, 0.5)
        self.assertEqual(stereo_center.shape, (1000, 2), "Should be stereo")
        # Center should have roughly equal left/right
        left_center = np.abs(stereo_center[:, 0]).mean()
        right_center = np.abs(stereo_center[:, 1]).mean()
        self.assertAlmostEqual(left_center, right_center, places=1,
                              msg="Center pan should have equal left/right")
        
        # Left pan
        stereo_left = pan_stereo(mono, 0.0)
        left_left = np.abs(stereo_left[:, 0]).mean()
        right_left = np.abs(stereo_left[:, 1]).mean()
        self.assertGreater(left_left, right_left,
                          "Left pan should have more left channel")
        
        # Right pan
        stereo_right = pan_stereo(mono, 1.0)
        left_right = np.abs(stereo_right[:, 0]).mean()
        right_right = np.abs(stereo_right[:, 1]).mean()
        self.assertGreater(right_right, left_right,
                          "Right pan should have more right channel")
    
    def test_generate_filtered_noise(self):
        """
        Test: generate_filtered_noise creates valid filtered noise.
        
        Should produce audio signal with correct length and properties.
        """
        noise = generate_filtered_noise(duration_s=0.1, sample_rate=48000,
                                      center_freq=1000.0, bandwidth=500.0)
        
        self.assertEqual(len(noise), int(0.1 * 48000), "Should have correct length")
        self.assertIsInstance(noise, np.ndarray, "Should be numpy array")
        self.assertEqual(noise.dtype, np.float32, "Should be float32")
    
    def test_add_reverb(self):
        """
        Test: add_reverb adds reverb to signal.
        
        Should produce signal with same length, potentially different amplitude.
        """
        signal = np.random.rand(1000).astype(np.float32)
        reverb_signal = add_reverb(signal, depth_norm=0.5, sample_rate=48000)
        
        self.assertEqual(len(reverb_signal), len(signal), "Should have same length")
        self.assertIsInstance(reverb_signal, np.ndarray, "Should be numpy array")
    
    def test_generate_continuous_spatial_audio(self):
        """
        Test: generate_continuous_spatial_audio creates continuous audio from multiple frames.
        
        Should generate a single audio file matching the video duration with
        each frame's audio placed at the correct time position.
        """
        # Create mock frame results
        frame_results = [
            {
                'frame_idx': 0,
                'open_space_columns': [30, 50],
                'column_depths': np.ones(100) * 0.5,
                'azimuths_deg': [-10.0, 0.0]
            },
            {
                'frame_idx': 1,
                'open_space_columns': [70],
                'column_depths': np.ones(100) * 0.6,
                'azimuths_deg': [10.0]
            },
            {
                'frame_idx': 2,
                'open_space_columns': [],  # No open spaces
                'column_depths': np.ones(100) * 0.3,
                'azimuths_deg': []
            }
        ]
        
        video_duration_s = 1.0  # 1 second video
        sample_every_s = 0.2  # Sample every 0.2 seconds
        
        result = generate_continuous_spatial_audio(
            frame_results,
            video_duration_s,
            sample_every_s,
            sample_rate=48000
        )
        
        # Should generate audio matching video duration
        expected_samples = int(video_duration_s * 48000)
        self.assertIsNotNone(result, "Should generate audio")
        self.assertEqual(result.shape[0], expected_samples, 
                        f"Should have {expected_samples} samples for {video_duration_s}s video")
        self.assertEqual(result.shape[1], 2, "Should be stereo")
        
        # First frame's audio should be at the beginning (frame 0 at time 0)
        # Second frame's audio should be at 0.2s (frame 1)
        # Third frame has no audio, so should be silence
        
        # Check that audio is not all zeros (at least some frames have audio)
        self.assertGreater(np.max(np.abs(result)), 0, 
                          "Should have some non-zero audio")


if __name__ == '__main__':
    unittest.main()

