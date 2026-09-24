"""
Depth Processing Module

This module processes depth maps to extract:
- Column depth values (averaged from bottom 1/3 of image)
- Open space columns (peaks in depth signal)
- Azimuth angles (angle from camera center for each column)

Algorithm:
1. Detect_open_spaces: Split depth map into bottom 1/3, average depth per column
2. Find_open_space_columns: Find peaks in the averaged depth array
3. Azimuth: Calculate angle from camera center based on field of view
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d, median_filter

# Configuration constants
LOWER_BAND_FRAC = 0.33  # Use bottom 1/3 of the image
SMOOTH_KERNEL = 5  # Gaussian smoothing kernel size (pixels)
MIN_PROM = 0.01  # Minimum prominence for peak detection (0..1 depth scale)
K_CANDIDATES = 5  # Maximum number of open-space directions per frame
HFOV_DEG = 72.0  # Horizontal field of view in degrees (iPhone Wide lens)

# Improved hallway detection parameters
MIN_HALLWAY_WIDTH = 5  # Minimum width in columns for a hallway region


def detect_open_spaces(depth_img: np.ndarray) -> np.ndarray:
    """
    Split the depth map into the bottom 1/3 and average depth from each column.

    This collapses the depth map to a 1D array where each element represents
    the average depth for that column in the bottom third of the image.

    Uses multiple passes of filtering to smooth the signal and better identify
    consistent regions (hallways).

    Args:
        depth_img: 2D depth map (H x W) with normalized depth values [0, 1]

    Returns:
        1D array of average depth per column (length = W), smoothed
    """
    H, W = depth_img.shape
    y0 = int(H * (1.0 - LOWER_BAND_FRAC))  # Start of bottom 1/3
    band = depth_img[y0:H, :]  # Extract bottom third
    sig = band.mean(axis=0)  # Average depth per column

    # Apply multiple passes of low-pass filtering for better smoothing
    # This helps identify consistent regions (hallways) vs noise
    sig_s = sig.copy()

    # First pass: stronger smoothing to remove high-frequency noise
    sig_s = gaussian_filter1d(sig_s, sigma=SMOOTH_KERNEL * 2)

    # Second pass: lighter smoothing to preserve structure
    sig_s = gaussian_filter1d(sig_s, sigma=SMOOTH_KERNEL)

    # Optional: median filter to remove outliers while preserving edges
    sig_s = median_filter(sig_s, size=5)

    return sig_s


def find_open_space_columns(
    column_depth_signal: np.ndarray,
    k: int = K_CANDIDATES,
    min_prom: float = MIN_PROM,
    window_size: int = 41,
) -> List[int]:
    """
    Find open space columns by detecting regions of consistently high depth.

    For hallways, we look for regions where:
    - Depth is consistently high (not just a single peak)
    - The region is wider than a minimum threshold (hallways span multiple columns)
    - Depth is significantly higher than surrounding areas (walls)

    This improved algorithm finds "plateaus" of high depth rather than just peaks,
    which better represents hallways and open corridors.

    Args:
        column_depth_signal: 1D array of average depth per column
        k: Maximum number of open-space columns to return
        min_prom: Minimum prominence (difference from surrounding) to be considered
        window_size: Size of window for prominence calculation (should be odd)

    Returns:
        List of column indices representing open spaces, sorted by quality (best first)
    """
    s = column_depth_signal.copy()
    W = len(s)

    # Calculate statistics
    median_depth = np.median(s)

    # For hallways, we want regions with CONSISTENT depth (low variance)
    # Hallways extend forward uniformly, so depth should be relatively constant across columns
    # Calculate local variance (consistency) using a sliding window
    window_half = window_size // 2
    local_variance = np.zeros(W)
    local_mean = np.zeros(W)

    for i in range(W):
        left = max(0, i - window_half)
        right = min(W, i + window_half + 1)
        window_data = s[left:right]
        local_mean[i] = np.mean(window_data)
        local_variance[i] = np.var(window_data)

    # Hallways have LOW variance (consistent depth)
    # Find regions where variance is low (consistent = hallway)

    # Create a combined score: low variance (consistent) is the key indicator
    scores = np.zeros(W)
    for i in range(W):
        # High score for low variance (consistent = hallway)
        # Use inverse variance with stronger weighting - very low variance gets very high score
        # Normalize variance to [0, 1] range first
        max_var = np.max(local_variance)
        if max_var > 0:
            normalized_var = local_variance[i] / max_var
        else:
            normalized_var = 0
        # Invert: low variance -> high score (exponential to strongly favor very low variance)
        consistency_score = np.exp(
            -normalized_var * 10
        )  # Exponential decay favors very low variance

        # Depth score: prefer values around median (typical hallway depth)
        # But this is secondary to consistency
        depth_diff = abs(s[i] - median_depth)
        depth_score = np.exp(-depth_diff * 5)  # Exponential decay from median

        # Combined score: STRONGLY prioritize consistency (low variance = 95% weight)
        # Hallways are defined by consistency, not absolute depth
        scores[i] = consistency_score * 0.95 + depth_score * 0.05

    # Find regions where score is above threshold (consistent + reasonable depth)
    score_threshold = np.percentile(scores, 65)  # Top 35% of scores
    above_threshold = scores >= score_threshold

    # Find connected regions (hallways)
    regions = []
    in_region = False
    region_start = 0

    for i in range(W):
        if above_threshold[i] and not in_region:
            # Start of a new region
            region_start = i
            in_region = True
        elif not above_threshold[i] and in_region:
            # End of a region
            region_end = i
            region_width = region_end - region_start

            # Only consider regions that are wide enough (hallways span multiple columns)
            if region_width >= MIN_HALLWAY_WIDTH:
                # Calculate region properties
                region_cols = np.arange(region_start, region_end)
                region_depths = s[region_cols]
                avg_depth = np.mean(region_depths)
                max_depth_in_region = np.max(region_depths)
                center_col = region_start + region_width // 2

                # Calculate prominence: how much deeper is this region than surroundings?
                # Look at areas just outside the region
                left_boundary = max(0, region_start - window_size // 2)
                right_boundary = min(W, region_end + window_size // 2)

                # Get depths outside the region
                outside_left = s[left_boundary:region_start] if region_start > 0 else []
                outside_right = s[region_end:right_boundary] if region_end < W else []
                outside_depths = (
                    np.concatenate([outside_left, outside_right])
                    if len(outside_left) > 0 or len(outside_right) > 0
                    else np.array([0])
                )

                # Calculate region consistency (variance within the region)
                region_variance = np.var(region_depths)

                # For hallways, consistency (low variance) is more important than prominence
                # Calculate a consistency score (inverse of variance)
                max_possible_var = np.var(s)  # Overall variance of signal
                if max_possible_var > 0:
                    consistency_score = 1.0 - (region_variance / max_possible_var)
                else:
                    consistency_score = 1.0

                # Calculate prominence if we have outside data
                if len(outside_depths) > 0:
                    surrounding_avg = np.mean(outside_depths)
                    prominence = avg_depth - surrounding_avg
                else:
                    prominence = 0.0

                # Score: prioritize consistency (low variance) over depth/prominence
                # Very consistent regions (low variance) are hallways
                score = (
                    consistency_score * 0.7
                    + (region_width / W) * 0.2
                    + abs(prominence) * 0.1
                )

                # Accept regions with either:
                # 1. High consistency (low variance) - this is the key indicator
                # 2. OR sufficient prominence (traditional check)
                if consistency_score > 0.8 or prominence >= min_prom:
                    regions.append(
                        {
                            "center": center_col,
                            "start": region_start,
                            "end": region_end,
                            "width": region_width,
                            "avg_depth": avg_depth,
                            "max_depth": max_depth_in_region,
                            "prominence": prominence,
                            "consistency": consistency_score,
                            "variance": region_variance,
                            "score": score,
                        }
                    )

            in_region = False

    # Handle case where region extends to end of signal
    if in_region:
        region_end = W
        region_width = region_end - region_start
        if region_width >= MIN_HALLWAY_WIDTH:
            region_cols = np.arange(region_start, region_end)
            region_depths = s[region_cols]
            avg_depth = np.mean(region_depths)
            center_col = region_start + region_width // 2

            # Calculate consistency for region at end
            region_variance = np.var(region_depths)
            max_possible_var = np.var(s)
            if max_possible_var > 0:
                consistency_score = 1.0 - (region_variance / max_possible_var)
            else:
                consistency_score = 1.0

            left_boundary = max(0, region_start - window_size // 2)
            outside_left = (
                s[left_boundary:region_start] if region_start > 0 else np.array([0])
            )
            if len(outside_left) > 0:
                surrounding_avg = np.mean(outside_left)
                prominence = avg_depth - surrounding_avg
            else:
                prominence = 0.0

            score = (
                consistency_score * 0.7
                + (region_width / W) * 0.2
                + abs(prominence) * 0.1
            )

            if consistency_score > 0.8 or prominence >= min_prom:
                regions.append(
                    {
                        "center": center_col,
                        "start": region_start,
                        "end": region_end,
                        "width": region_width,
                        "avg_depth": avg_depth,
                        "max_depth": np.max(region_depths),
                        "prominence": prominence,
                        "consistency": consistency_score,
                        "variance": region_variance,
                        "score": score,
                    }
                )

    # If no wide regions found, fall back to peak detection for narrow openings
    if len(regions) == 0:
        # Use original peak detection as fallback
        picks = []
        s_copy = s.copy()
        half_window = window_size // 2

        for _ in range(k * 2):
            idx = int(np.argmax(s_copy))
            left = max(0, idx - half_window)
            right = min(len(s_copy), idx + half_window + 1)
            median_surrounding = np.median(s_copy[left:right])
            prom = s_copy[idx] - median_surrounding

            if prom < min_prom:
                break

            picks.append((idx, s_copy[idx], prom))
            suppress_left = max(0, idx - 15)
            suppress_right = min(len(s_copy), idx + 16)
            s_copy[suppress_left:suppress_right] = -np.inf

        # Return peak centers
        picks = sorted(picks, key=lambda t: t[1], reverse=True)[:k]
        return [p[0] for p in picks]

    # Sort regions by score (best first)
    regions.sort(key=lambda r: r["score"], reverse=True)

    # For each region, return columns that span the hallway width
    # For wide hallways, sample columns at regular intervals to show the full extent
    cols = []
    for region in regions[:k]:
        start = region["start"]
        end = region["end"]
        width = region["width"]

        # For wide regions (hallways), sample columns to show the full extent
        # Sample every N columns, with at least 3 samples and at most 20
        if width > 20:
            # Sample at regular intervals
            step = max(1, width // 15)  # Sample ~15 points across the hallway
            region_cols = list(range(start, end, step))
            # Always include the last column
            if region_cols[-1] != end - 1:
                region_cols.append(end - 1)
        else:
            # For narrower regions, include all columns
            region_cols = list(range(start, end))

        cols.extend(region_cols)

    # Remove duplicates and sort
    cols = sorted(list(set(cols)))

    # Limit total number of columns if we have too many
    if len(cols) > k * 20:  # Allow up to 20 columns per region
        # Keep the best columns (prioritize from best regions)
        cols = cols[: k * 20]

    return cols


def columns_to_azimuths(
    columns: List[int], width: int, hfov_deg: float = HFOV_DEG
) -> List[float]:
    """
    Convert column indices to azimuth angles (angle from camera center).

    The camera's field of view is an arc. If the horizontal FOV is hfov_deg,
    then the leftmost column is at -hfov_deg/2 and rightmost is at +hfov_deg/2.

    Using arclength formula: s = r * theta
    - The distance from center (in normalized coordinates) maps linearly to angle
    - Center column (width/2) maps to 0 degrees (straight ahead)
    - Left edge maps to -hfov_deg/2
    - Right edge maps to +hfov_deg/2

    Args:
        columns: List of column indices
        width: Width of the image (number of columns)
        hfov_deg: Horizontal field of view in degrees

    Returns:
        List of azimuth angles in degrees (negative = left, positive = right, 0 = center)
    """
    center = (width - 1) / 2.0
    norm = (np.array(columns) - center) / center  # Normalize to [-1, 1]
    az = norm * (hfov_deg / 2.0)  # Map to [-hfov_deg/2, +hfov_deg/2]
    return az.tolist()


def process_depth_map(
    depth_img: np.ndarray,
    k: int = K_CANDIDATES,
    min_prom: float = MIN_PROM,
    hfov_deg: float = HFOV_DEG,
) -> dict:
    """
    Complete depth processing pipeline: extract column depths, find open spaces, calculate azimuths.

    Args:
        depth_img: 2D depth map (H x W) with normalized depth values [0, 1]
        k: Maximum number of open-space columns to return
        min_prom: Minimum prominence for peak detection
        hfov_deg: Horizontal field of view in degrees

    Returns:
        Dictionary with:
            - 'column_depths': 1D array of average depth per column
            - 'open_space_columns': List of column indices representing open spaces
            - 'azimuths_deg': List of azimuth angles in degrees for open space columns
            - 'all_column_depths': Full array of depth values for all columns
    """
    # Step 1: Extract column depth signal from bottom 1/3
    column_depths = detect_open_spaces(depth_img)

    # Step 2: Find open space columns (peaks)
    open_space_columns = find_open_space_columns(column_depths, k=k, min_prom=min_prom)

    # Step 3: Calculate azimuth angles for open space columns
    azimuths_deg = columns_to_azimuths(open_space_columns, depth_img.shape[1], hfov_deg)

    return {
        "column_depths": column_depths,
        "open_space_columns": open_space_columns,
        "azimuths_deg": azimuths_deg,
        "all_column_depths": column_depths,  # Alias for convenience
    }


def load_depth_from_file(depth_path: str) -> np.ndarray:
    """
    Load a depth map from a file (PNG or NPY).

    Args:
        depth_path: Path to depth file (.png or .npy)

    Returns:
        Normalized depth map as float32 array [0, 1]
    """
    if depth_path.endswith(".npy"):
        d = np.load(depth_path).astype(np.float32)
    else:
        # Assume PNG (16-bit)
        d16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d16 is None:
            raise ValueError(f"Could not load depth image: {depth_path}")
        d16 = d16.astype(np.float32)
        d = d16 / 65535.0  # Normalize from 16-bit to [0, 1]

    return d
