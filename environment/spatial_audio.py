"""
Spatial Audio Output Module

This module handles spatial audio output based on depth and azimuth information.
Maps detected open space columns to spatial audio using stereo panning.

The module maps n columns from depth to the camera's field of view (72° for iPhone Wide lens)
and generates positioned audio using sounddevice.

Algorithm:
- Each detected open space column has an azimuth angle (from depth_processing)
- Azimuth is converted to stereo pan (left/right positioning)
- White noise is filtered by frequency based on depth (far = lower freq, close = higher freq)
- Reverb is added based on depth (far = more reverb)
- All sounds are mixed into a stereo output
"""

import numpy as np
import math
from typing import List, Optional, Tuple

# Configuration
CAMERA_FOV_DEG = 72.0  # iPhone Wide lens horizontal field of view
SAMPLE_RATE = 48000  # Standard audio sample rate (matches original implementation)

# Module is now enabled
SPATIAL_AUDIO_ENABLED = True


def generate_filtered_noise(
    duration_s: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
    center_freq: float = 1000.0,
    bandwidth: float = 500.0,
) -> np.ndarray:
    """
    Generate band-limited brown noise (smooth ocean-like sound).

    Brown noise has a 6 dB/octave slope and is generated using cumulative sum
    of uniform random numbers, then band-limited with high-pass at 20 Hz and
    low-pass at 40 Hz to prevent system overload.

    Args:
        duration_s: Duration in seconds
        sample_rate: Audio sample rate
        center_freq: (Unused, kept for compatibility)
        bandwidth: (Unused, kept for compatibility)

    Returns:
        Band-limited brown noise as 1D numpy array
    """
    n = int(duration_s * sample_rate)
    
    # Generate brown noise using cumulative sum method
    # Step 1: Create zero-mean uniformly distributed random numbers [-1, 1]
    uniform_random = 2.0 * np.random.rand(n).astype(np.float32) - 1.0
    
    # Step 2: Cumulative sum with scaling factor
    # Scale factor 'c' controls the amplitude - adjust to prevent overflow
    c = 0.01  # Scaling factor to keep amplitude reasonable
    brown_noise = np.zeros(n, dtype=np.float32)
    brown_noise[0] = c * uniform_random[0]
    for i in range(1, n):
        brown_noise[i] = brown_noise[i - 1] + c * uniform_random[i]
    
    # Remove DC offset
    brown_noise = brown_noise - np.mean(brown_noise)
    
    # Step 3: Apply band-limiting filters
    # High-pass filter at 20 Hz (first-order)
    hp_cutoff = 20.0
    alpha_hp = 1.0 / (1.0 + 2.0 * np.pi * hp_cutoff / sample_rate)
    hp_filtered = np.zeros_like(brown_noise)
    hp_filtered[0] = brown_noise[0]
    for i in range(1, n):
        hp_filtered[i] = alpha_hp * (
            hp_filtered[i - 1] + brown_noise[i] - brown_noise[i - 1]
        )
    
    # Low-pass filter at 40 Hz (first-order)
    lp_cutoff = 40.0
    alpha_lp = 2.0 * np.pi * lp_cutoff / sample_rate
    alpha_lp = alpha_lp / (1.0 + alpha_lp)
    lp_filtered = np.zeros_like(hp_filtered)
    lp_filtered[0] = hp_filtered[0]
    for i in range(1, n):
        lp_filtered[i] = alpha_lp * hp_filtered[i] + (1.0 - alpha_lp) * lp_filtered[i - 1]
    
    filtered = lp_filtered
    
    # Apply window to prevent clicks at boundaries
    fade_samples = min(int(0.01 * sample_rate), n // 10)  # 10ms fade or 10% of length
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        filtered[:fade_samples] *= fade_in
        filtered[-fade_samples:] *= fade_out
    
    # Normalize to reasonable amplitude
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak * 0.3  # Scale to reasonable level
    
    return filtered


def add_reverb(
    signal: np.ndarray, depth_norm: float, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Add simple reverb based on depth (farther = more reverb).

    Args:
        signal: Input audio signal
        depth_norm: Normalized depth value [0, 1] where 1 = far
        sample_rate: Audio sample rate

    Returns:
        Signal with reverb added
    """
    # More reverb for farther objects
    delay_ms = int(30 + depth_norm * 150)  # 30-180ms delay
    delay_samples = int(delay_ms * sample_rate / 1000)

    if delay_samples >= len(signal):
        return signal

    reverb = np.zeros_like(signal)
    reverb[delay_samples:] = signal[:-delay_samples] * (0.15 + depth_norm * 0.35)

    return signal + reverb


def azimuth_to_pan(az_deg: float) -> float:
    """
    Convert azimuth angle to stereo pan value.

    Args:
        az_deg: Azimuth angle in degrees (-36 to +36 for 72° FOV)

    Returns:
        Pan value [0, 1] where 0 = left, 0.5 = center, 1 = right
    """
    # Map from [-36, +36] to [0, 1] for 72° FOV
    # Center (0°) maps to 0.5
    pan = (az_deg + CAMERA_FOV_DEG / 2.0) / CAMERA_FOV_DEG
    return np.clip(pan, 0.0, 1.0)


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """
    Pan mono signal to stereo.

    Args:
        mono: Mono audio signal (1D array)
        pan: Pan value [0, 1] where 0 = left, 0.5 = center, 1 = right

    Returns:
        Stereo audio signal (samples x 2)
    """
    left_gain = math.cos(pan * math.pi / 2)
    right_gain = math.cos((1 - pan) * math.pi / 2)
    L = (mono * left_gain).astype(np.float32)
    R = (mono * right_gain).astype(np.float32)
    return np.stack([L, R], axis=-1)


def depth_to_frequency(depth_norm: float) -> float:
    """
    Map depth to frequency band center.

    Closer objects = higher frequency (more treble)
    Farther objects = lower frequency (more bass)

    Args:
        depth_norm: Normalized depth [0, 1] where 1 = far

    Returns:
        Center frequency in Hz
    """
    return 3500 - (depth_norm * 2800)  # 3500Hz (close) to 700Hz (far)


def depth_to_gain(depth_norm: float) -> float:
    """
    Map depth to volume gain.

    Args:
        depth_norm: Normalized depth [0, 1] where 1 = far

    Returns:
        Gain multiplier
    """
    return 0.3 + 0.7 * (depth_norm**0.7)


def play_positioned_sound(
    audio_data: np.ndarray, column: int, n_columns: int, sample_rate: int = SAMPLE_RATE
) -> bool:
    """
    Play a positioned sound based on column index.

    Maps column position to stereo panning:
    - Leftmost column (0) -> full left
    - Center column (n_columns/2) -> center
    - Rightmost column (n_columns-1) -> full right

    Args:
        audio_data: Mono audio data (1D numpy array)
        column: Column index (0 to n_columns-1)
        n_columns: Total number of columns
        sample_rate: Audio sample rate

    Returns:
        True if successful, False otherwise
    """
    if not SPATIAL_AUDIO_ENABLED:
        return False

    try:
        import sounddevice as sd

        # Convert column to pan value
        pan = column / (n_columns - 1) if n_columns > 1 else 0.5

        # Create stereo from mono
        left_gain = np.sqrt(1 - pan)
        right_gain = np.sqrt(pan)

        stereo = np.column_stack([audio_data * left_gain, audio_data * right_gain])

        sd.play(stereo, sample_rate)
        sd.wait()  # Wait for playback to finish
        return True
    except ImportError:
        print(
            "Warning: sounddevice not available. Install with: pip install sounddevice"
        )
        return False
    except Exception as e:
        print(f"Error playing sound: {e}")
        return False


def map_columns_to_fov(
    columns: List[int], n_columns: int, fov_deg: float = CAMERA_FOV_DEG
) -> List[float]:
    """
    Map column indices to angles within the camera's field of view.

    This is a convenience function that uses the same logic as columns_to_azimuths
    from depth_processing. The depth detection covers n_columns, and we map
    columns to angles within the FOV range.

    Args:
        columns: List of column indices
        n_columns: Total number of columns in depth map
        fov_deg: Camera field of view in degrees

    Returns:
        List of angles in degrees relative to camera center
        (range: [-fov_deg/2, +fov_deg/2])
    """
    if not SPATIAL_AUDIO_ENABLED:
        return []

    # Use the same calculation as depth_processing.columns_to_azimuths
    center = (n_columns - 1) / 2.0
    norm = (np.array(columns) - center) / center  # Normalize to [-1, 1]
    az = norm * (fov_deg / 2.0)  # Map to [-fov_deg/2, +fov_deg/2]
    return az.tolist()


def generate_spatial_audio(
    open_space_columns: List[int],
    column_depths: np.ndarray,
    azimuths_deg: List[float],
    duration_s: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Generate spatial audio output for detected open spaces.

    For each detected open space:
    - Maps azimuth angle to stereo panning
    - Generates filtered white noise with frequency based on depth
    - Adds reverb based on depth (far = more reverb)
    - Mixes all sounds into stereo output

    Args:
        open_space_columns: List of column indices representing open spaces
        column_depths: Array of depth values per column (1D array)
        azimuths_deg: List of azimuth angles in degrees for each column
        duration_s: Duration of audio in seconds
        sample_rate: Audio sample rate

    Returns:
        Stereo audio array (samples x 2) or None if disabled
    """
    if not SPATIAL_AUDIO_ENABLED:
        return None

    if len(open_space_columns) == 0:
        return None

    n_samples = int(duration_s * sample_rate)
    mix = np.zeros((n_samples, 2), dtype=np.float32)

    # Create azimuth-depth pairs
    # Get depth value for each column
    azimuth_depth_pairs = []
    for i, col in enumerate(open_space_columns):
        if 0 <= col < len(column_depths):
            depth_norm = float(column_depths[col])
            az = azimuths_deg[i] if i < len(azimuths_deg) else 0.0
            azimuth_depth_pairs.append((az, depth_norm))

    # If no valid pairs, return None
    if len(azimuth_depth_pairs) == 0:
        return None

    # Generate audio for each open space
    for az, depth_norm in azimuth_depth_pairs:
        # Convert azimuth to pan
        pan = azimuth_to_pan(az)

        # Generate filtered noise with frequency based on depth
        center_freq = depth_to_frequency(depth_norm)
        bandwidth = 150 + (depth_norm * 150)  # 150-300Hz bandwidth

        filtered_noise = generate_filtered_noise(
            duration_s, sample_rate, center_freq, bandwidth
        )

        # Add reverb based on depth
        filtered_noise = add_reverb(filtered_noise, depth_norm, sample_rate)

        # Apply gain based on depth
        gain = depth_to_gain(depth_norm) * 1.5  # Boost for clarity
        filtered_noise = filtered_noise * gain

        # Pan to stereo
        stereo_noise = pan_stereo(filtered_noise, pan)

        # Add to mix (truncate if needed)
        mix_len = min(len(stereo_noise), len(mix))
        mix[:mix_len, :] += stereo_noise[:mix_len, :]

    # Normalize to prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix / (peak / 0.95)

    return mix


def smooth_open_spaces_temporally(
    frame_results: List[dict], smoothing_factor: float = 0.7
) -> List[dict]:
    """
    Smooth open space columns across frames using exponential smoothing.

    This prevents open spaces from jumping around between frames by
    gradually transitioning between detected positions. Uses a simple
    approach: smooth the azimuth angles directly.

    Args:
        frame_results: List of frame result dictionaries
        smoothing_factor: Exponential smoothing factor (0-1), higher = more smoothing
                         0.7 means 70% of previous value, 30% of new value

    Returns:
        List of smoothed frame results
    """
    if len(frame_results) == 0:
        return frame_results

    smoothed_results = []
    prev_azimuths = None

    for i, result in enumerate(frame_results):
        current_azimuths = result.get("azimuths_deg", [])

        if i == 0:
            # First frame: no smoothing needed
            smoothed_results.append(result.copy())
            prev_azimuths = current_azimuths.copy() if current_azimuths else []
            continue

        # Apply exponential smoothing to azimuths
        if len(current_azimuths) > 0:
            if prev_azimuths is not None and len(prev_azimuths) > 0:
                # Match current azimuths to previous ones by finding closest pairs
                smoothed_azimuths = []
                used_prev = set()

                for az in current_azimuths:
                    # Find closest previous azimuth
                    best_match = None
                    best_dist = float("inf")
                    best_idx = -1

                    for j, prev_az in enumerate(prev_azimuths):
                        if j in used_prev:
                            continue
                        dist = abs(az - prev_az)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = prev_az
                            best_idx = j

                    if (
                        best_match is not None and best_dist < 45.0
                    ):  # Match if within 45 degrees
                        # Smooth between previous and current
                        used_prev.add(best_idx)
                        smoothed_az = (
                            smoothing_factor * best_match + (1 - smoothing_factor) * az
                        )
                        smoothed_azimuths.append(smoothed_az)
                    else:
                        # New open space, fade in from center
                        smoothed_az = (
                            smoothing_factor * 0.0 + (1 - smoothing_factor) * az
                        )
                        smoothed_azimuths.append(smoothed_az)

                # Add any unmatched previous azimuths (fade out)
                for j, prev_az in enumerate(prev_azimuths):
                    if j not in used_prev:
                        faded_az = prev_az * smoothing_factor
                        if abs(faded_az) > 2.0:  # Only keep if still significant
                            smoothed_azimuths.append(faded_az)

                # Update result
                smoothed_result = result.copy()
                smoothed_result["azimuths_deg"] = smoothed_azimuths
                smoothed_results.append(smoothed_result)
                prev_azimuths = smoothed_azimuths.copy()
            else:
                # No previous azimuths, use current
                smoothed_results.append(result.copy())
                prev_azimuths = current_azimuths.copy()
        else:
            # No open spaces in current frame, fade out previous
            if prev_azimuths is not None and len(prev_azimuths) > 0:
                faded_azimuths = [az * smoothing_factor for az in prev_azimuths]
                # Only keep if still significant
                if any(abs(az) > 2.0 for az in faded_azimuths):
                    smoothed_result = result.copy()
                    smoothed_result["azimuths_deg"] = [
                        az for az in faded_azimuths if abs(az) > 2.0
                    ]
                    # Update columns to match (approximate)
                    if len(result.get("open_space_columns", [])) == 0:
                        # Use previous columns if available
                        prev_result = (
                            smoothed_results[-1] if smoothed_results else result
                        )
                        smoothed_result["open_space_columns"] = prev_result.get(
                            "open_space_columns", []
                        )[: len(smoothed_result["azimuths_deg"])]
                    smoothed_results.append(smoothed_result)
                    prev_azimuths = [az for az in faded_azimuths if abs(az) > 2.0]
                else:
                    smoothed_results.append(result.copy())
                    prev_azimuths = []
            else:
                smoothed_results.append(result.copy())
                prev_azimuths = []

    return smoothed_results


def generate_continuous_spatial_audio(
    frame_results: List[dict],
    video_duration_s: float,
    sample_every_s: float,
    sample_rate: int = SAMPLE_RATE,
    crossfade_ms: float = 50.0,
    temporal_smoothing: bool = True,
) -> Optional[np.ndarray]:
    """
    Generate a single continuous audio file from all frame results.

    Creates one audio file that matches the video length, with each frame's
    audio placed at the correct time position. Includes crossfading between
    frames to prevent choppy transitions.

    Args:
        frame_results: List of dictionaries, each containing:
            - 'frame_idx': Frame index
            - 'open_space_columns': List of column indices
            - 'column_depths': Array of depth values per column
            - 'azimuths_deg': List of azimuth angles in degrees
        video_duration_s: Total video duration in seconds
        sample_every_s: Time interval between sampled frames in seconds
        sample_rate: Audio sample rate
        crossfade_ms: Crossfade duration in milliseconds between frames
        temporal_smoothing: Whether to apply temporal smoothing to open spaces

    Returns:
        Stereo audio array (samples x 2) matching video duration, or None if disabled
    """
    if not SPATIAL_AUDIO_ENABLED:
        return None

    # Apply temporal smoothing if requested
    if temporal_smoothing:
        frame_results = smooth_open_spaces_temporally(
            frame_results, smoothing_factor=0.7
        )

    # Calculate total number of samples for the full video
    total_samples = int(video_duration_s * sample_rate)
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    # Sort results by frame_idx to ensure correct ordering
    sorted_results = sorted(frame_results, key=lambda x: x.get("frame_idx", 0))

    # Create a map of frame_idx to results for quick lookup
    frame_map = {r.get("frame_idx", 0): r for r in sorted_results}

    # Generate ONE continuous brown noise stream for the entire video
    # This ensures smooth, continuous audio without discontinuities
    # Brown noise: cumulative sum of uniform random numbers
    uniform_random = 2.0 * np.random.rand(total_samples).astype(np.float32) - 1.0
    c = 0.01  # Scaling factor
    continuous_noise = np.zeros(total_samples, dtype=np.float32)
    continuous_noise[0] = c * uniform_random[0]
    for i in range(1, total_samples):
        continuous_noise[i] = continuous_noise[i - 1] + c * uniform_random[i]
    
    # Remove DC offset
    continuous_noise = continuous_noise - np.mean(continuous_noise)
    
    # Apply band-limiting filters (high-pass at 20 Hz, low-pass at 40 Hz)
    # High-pass filter at 20 Hz
    hp_cutoff = 20.0
    alpha_hp = 1.0 / (1.0 + 2.0 * np.pi * hp_cutoff / sample_rate)
    hp_filtered = np.zeros_like(continuous_noise)
    hp_filtered[0] = continuous_noise[0]
    for i in range(1, total_samples):
        hp_filtered[i] = alpha_hp * (
            hp_filtered[i - 1] + continuous_noise[i] - continuous_noise[i - 1]
        )
    
    # Low-pass filter at 40 Hz
    lp_cutoff = 40.0
    alpha_lp = 2.0 * np.pi * lp_cutoff / sample_rate
    alpha_lp = alpha_lp / (1.0 + alpha_lp)
    lp_filtered = np.zeros_like(hp_filtered)
    lp_filtered[0] = hp_filtered[0]
    for i in range(1, total_samples):
        lp_filtered[i] = alpha_lp * hp_filtered[i] + (1.0 - alpha_lp) * lp_filtered[i - 1]
    
    continuous_noise = lp_filtered
    
    # Normalize to reasonable amplitude
    peak = np.max(np.abs(continuous_noise))
    if peak > 0:
        continuous_noise = continuous_noise / peak * 0.3

    # Apply window to prevent clicks at boundaries
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        continuous_noise[:fade_samples] *= fade_in
        continuous_noise[-fade_samples:] *= fade_out

    # Process in overlapping windows with smooth interpolation
    # This maintains continuity while allowing frequency/pan to change over time
    window_size = int(0.1 * sample_rate)  # 100ms windows
    hop_size = int(0.02 * sample_rate)  # 20ms hop (80% overlap for smoothness)

    for window_start in range(0, total_samples, hop_size):
        window_end = min(window_start + window_size, total_samples)
        window_noise = continuous_noise[window_start:window_end]

        # Get frame for center of window
        window_center_time = (window_start + window_size // 2) / sample_rate
        frame_idx = int(window_center_time / sample_every_s)

        # Get frame result (interpolate if needed)
        if frame_idx in frame_map:
            result = frame_map[frame_idx]
        else:
            if len(frame_map) == 0:
                continue
            closest_idx = min(frame_map.keys(), key=lambda x: abs(x - frame_idx))
            if abs(closest_idx - frame_idx) <= 3:
                result = frame_map[closest_idx]
            else:
                continue

        open_space_columns = result.get("open_space_columns", [])
        column_depths = result.get("column_depths", np.array([]))
        azimuths_deg = result.get("azimuths_deg", [])

        if len(open_space_columns) == 0:
            continue

        # Use first open space (or could average multiple)
        col = open_space_columns[0]
        if 0 <= col < len(column_depths):
            depth_norm = float(column_depths[col])
        else:
            depth_norm = 0.5

        az = azimuths_deg[0] if len(azimuths_deg) > 0 else 0.0

        # Calculate parameters
        # Note: Brown noise is already band-limited (20-40 Hz), so we just apply
        # gain and panning based on depth/azimuth
        pan = azimuth_to_pan(az)
        gain = depth_to_gain(depth_norm) * 1.5

        # Brown noise is already filtered, just apply gain
        filtered_window = window_noise * gain

        # Apply panning
        left_gain = math.cos(pan * math.pi / 2)
        right_gain = math.cos((1 - pan) * math.pi / 2)

        # Apply windowing function for smooth overlap
        window_func = np.ones(len(filtered_window))
        if window_start > 0 or window_end < total_samples:
            # Hann window for smooth overlap
            window_func = np.hanning(len(filtered_window))

        # Add to mix with windowing
        actual_len = min(len(filtered_window), total_samples - window_start)
        mix[window_start : window_start + actual_len, 0] += (
            filtered_window[:actual_len] * left_gain * window_func[:actual_len]
        )
        mix[window_start : window_start + actual_len, 1] += (
            filtered_window[:actual_len] * right_gain * window_func[:actual_len]
        )

    # Apply final fade-out to the entire mix to prevent click at end
    fade_out_samples = int(0.01 * sample_rate)  # 10ms fade-out
    if fade_out_samples > 0 and len(mix) > fade_out_samples:
        fade_out = np.linspace(1, 0, fade_out_samples).reshape(-1, 1)
        mix[-fade_out_samples:, :] *= fade_out

    # Remove DC offset from final mix (can cause clicks)
    for channel in range(mix.shape[1]):
        mix[:, channel] = mix[:, channel] - np.mean(mix[:, channel])

    # Apply gentle high-pass filter to remove any remaining low-frequency artifacts
    # This helps prevent clicks from sudden amplitude changes
    if len(mix) > 1:
        for channel in range(mix.shape[1]):
            # Simple first-order high-pass at ~20Hz
            alpha = 1.0 / (1.0 + 2.0 * np.pi * 20.0 / sample_rate)
            hp_filtered = np.zeros_like(mix[:, channel])
            hp_filtered[0] = mix[0, channel]
            for i in range(1, len(mix)):
                hp_filtered[i] = alpha * (
                    hp_filtered[i - 1] + mix[i, channel] - mix[i - 1, channel]
                )
            mix[:, channel] = hp_filtered

    # Normalize to prevent clipping (use soft limiting)
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        # Soft compression instead of hard clipping
        threshold = 0.95
        ratio = 0.1  # Compression ratio
        mix = np.where(
            np.abs(mix) > threshold,
            np.sign(mix) * (threshold + (np.abs(mix) - threshold) * ratio),
            mix,
        )
        # Then normalize
        new_peak = np.max(np.abs(mix))
        if new_peak > 0:
            mix = mix / new_peak * 0.95

    return mix


def export_spatial_audio(
    audio_data: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE
) -> bool:
    """
    Export spatial audio to WAV file.

    Args:
        audio_data: Stereo audio array (samples x 2)
        output_path: Path to save WAV file
        sample_rate: Audio sample rate

    Returns:
        True if successful, False otherwise
    """
    if not SPATIAL_AUDIO_ENABLED:
        return False

    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
        return True
    except ImportError:
        print("Warning: soundfile not available. Install with: pip install soundfile")
        return False
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False
