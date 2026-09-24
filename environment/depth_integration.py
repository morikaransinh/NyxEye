"""
Depth Integration Module

This module coordinates the entire pipeline:
1. Loads video and extracts frames
2. Feeds frames to depth estimator (MiDaS)
3. Processes depth maps to extract columns, depths, and azimuths
4. Coordinates audio output (currently disabled)

This is the main entry point for the depth processing pipeline.
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch

# Import our modular components
from depth_processing import process_depth_map, load_depth_from_file, HFOV_DEG
from spatial_audio import (
    SPATIAL_AUDIO_ENABLED,
    generate_spatial_audio,
    generate_continuous_spatial_audio,
    export_spatial_audio,
)

# Try to import HuggingFace-based depth processor (more reliable than torch.hub)
try:
    depth_module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "depth")
    )
    if depth_module_path not in sys.path:
        sys.path.insert(0, depth_module_path)
    from depth.depth_processor import DepthProcessor, create_depth_processor

    HUGGINGFACE_DEPTH_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from depth_processor import DepthProcessor, create_depth_processor

        HUGGINGFACE_DEPTH_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_DEPTH_AVAILABLE = False
        DepthProcessor = None
        create_depth_processor = None

# Paths (project-local defaults)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
DEPTH_NPY_DIR = os.path.join(DATA_DIR, "depth_npy")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TEST_RESULTS_DIR = os.path.join(DATA_DIR, "test_results")

# Ensure directories exist
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(DEPTH_NPY_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to input video file

    Returns:
        Video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
        return duration
    else:
        raise RuntimeError(f"Could not determine video duration from {video_path}")


def extract_frames(
    video_path: str, out_dir: str, sample_every_s: float = 0.2, max_frames: int = None
):
    """
    Extract frames from video at specified intervals.

    Args:
        video_path: Path to input video file
        out_dir: Directory to save extracted frames
        sample_every_s: Extract one frame every N seconds
        max_frames: Maximum number of frames to extract (None = no limit)

    Returns:
        Tuple of (list of frame file paths, video FPS)
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback guess

    step = max(int(round(fps * sample_every_s)), 1)

    frames = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            fn = os.path.join(out_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(fn, frame)
            frames.append(fn)
            saved += 1
            if max_frames and saved >= max_frames:
                break

        idx += 1

    cap.release()
    return frames, fps


def load_midas_huggingface(model_id: str = "Intel/dpt-hybrid-midas"):
    """
    Load MiDaS depth estimation model using HuggingFace transformers.
    This is more reliable than torch.hub and avoids SSL/git issues.

    Args:
        model_id: HuggingFace model ID (e.g., "Intel/dpt-hybrid-midas")

    Returns:
        DepthProcessor instance
    """
    if not HUGGINGFACE_DEPTH_AVAILABLE:
        raise RuntimeError(
            "HuggingFace transformers not available. Install with: pip install transformers"
        )

    print(
        f"Loading depth model '{model_id}' via HuggingFace (may download if not cached)..."
    )
    print("  Note: Model will be cached for offline use after first download.")
    try:
        processor = create_depth_processor(model_id=model_id)
        print("Depth processor loaded successfully.")
        return processor
    except Exception as e:
        print(f"Error loading depth processor: {repr(e)}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection for first-time download")
        print("  2. Verify transformers is installed: pip install transformers")
        print("  3. Check available disk space in ~/.cache/huggingface/")
        raise


def load_midas(model_type: str = "DPT_Large"):
    """
    Load MiDaS depth estimation model from torch.hub.

    Args:
        model_type: Type of MiDaS model (e.g., "DPT_Large", "DPT_Hybrid")

    Returns:
        Tuple of (model, transform)
    """
    # Check cache first
    cache_dir = os.path.expanduser("~/.cache/torch/hub")
    midas_cache = os.path.join(cache_dir, "intel-isl_MiDaS_master")

    print(f"Loading MiDaS model '{model_type}' via torch.hub...")

    # Check if cache exists
    if os.path.exists(midas_cache):
        print(f"  Found MiDaS cache at: {midas_cache}")
        print("  Attempting to load from cache (offline mode)...")
    else:
        print(f"  Cache not found at: {midas_cache}")
        print("  Will attempt to download (requires internet connection)...")

    # Workaround for SSL certificate issues on macOS
    # torch.hub uses git internally, so we need to configure git's SSL settings
    import ssl
    import urllib.request

    # os is already imported at module level

    # Configure git to handle SSL (torch.hub uses git internally)
    # Temporarily disable SSL verification for git (workaround for certificate issues)
    original_git_ssl = os.environ.get("GIT_SSL_NO_VERIFY", "")
    os.environ["GIT_SSL_NO_VERIFY"] = "1"

    # Also set for Python's urllib
    original_ssl_verify = os.environ.get("PYTHONHTTPSVERIFY", "")
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    # Create an unverified SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Patch urllib.request to use unverified SSL
    original_urlopen = urllib.request.urlopen

    def patched_urlopen(*args, **kwargs):
        if "context" not in kwargs:
            kwargs["context"] = ssl_context
        return original_urlopen(*args, **kwargs)

    try:
        # Patch urllib for this operation
        urllib.request.urlopen = patched_urlopen

        # Try to load with force_reload=False first (uses cache if available)
        # If cache exists, this should work offline
        midas = torch.hub.load(
            "intel-isl/MiDaS", model_type, force_reload=False, trust_repo=True
        )
        midas.to(device)
        midas.eval()
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", force_reload=False, trust_repo=True
        )
        if "DPT" in model_type:
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform
        print("MiDaS loaded successfully.")
        return midas, transform
    except RuntimeError as e:
        error_msg = str(e)
        if (
            "internet connection" in error_msg.lower()
            or "could not be found in the cache" in error_msg
        ):
            print("\n" + "=" * 60)
            print("ERROR: MiDaS model not found and no internet connection")
            print("=" * 60)

            # Detailed cache diagnostics
            print("\nCache Diagnostics:")
            print(f"  Cache directory: {cache_dir}")
            print(f"  Cache exists: {os.path.exists(cache_dir)}")
            print(f"  MiDaS cache: {midas_cache}")
            print(f"  MiDaS cache exists: {os.path.exists(midas_cache)}")

            if os.path.exists(midas_cache):
                # List contents of cache directory
                try:
                    cache_contents = os.listdir(midas_cache)
                    print(f"  Cache contents ({len(cache_contents)} items):")
                    for item in cache_contents[:10]:  # Show first 10
                        item_path = os.path.join(midas_cache, item)
                        item_type = "dir" if os.path.isdir(item_path) else "file"
                        print(f"    - {item} ({item_type})")
                    if len(cache_contents) > 10:
                        print(f"    ... and {len(cache_contents) - 10} more items")
                except Exception as list_err:
                    print(f"  Could not list cache contents: {list_err}")

            print("\nSolutions:")
            print("1. Connect to the internet and run again (model will be cached)")
            print("2. If you have internet, try:")
            print("   - Check your network connection")
            print("   - Verify GitHub is accessible")
            print(
                "   - Try: python -c \"import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large')\""
            )
            print("\n3. For offline use, download the model first:")
            print("   - Run this script with internet connection once")
            print("   - Model will be cached in: ~/.cache/torch/hub/")
            print("   - After caching, you can use offline")
            print("\n4. If cache exists but still fails:")
            print(
                "   - Try clearing cache: rm -rf ~/.cache/torch/hub/intel-isl_MiDaS_master"
            )
            print("   - Then re-download with internet connection")
            print("\n5. SSL Certificate Issue Detected:")
            print("   - This is likely an SSL certificate verification problem")
            print("   - The code now attempts to work around this")
            print("=" * 60 + "\n")
        raise
    except Exception as e:
        print("Error loading MiDaS:", repr(e))
        print("\nTroubleshooting:")
        print("1. Ensure you have internet connection (first time only)")
        print("2. Check that PyTorch is properly installed")
        print(
            "3. Verify torch.hub is working: python -c \"import torch; print(torch.hub.list('intel-isl/MiDaS'))\""
        )
        raise
    finally:
        # Always restore original urlopen and environment
        urllib.request.urlopen = original_urlopen
        if original_ssl_verify:
            os.environ["PYTHONHTTPSVERIFY"] = original_ssl_verify
        elif "PYTHONHTTPSVERIFY" in os.environ:
            del os.environ["PYTHONHTTPSVERIFY"]
        if original_git_ssl:
            os.environ["GIT_SSL_NO_VERIFY"] = original_git_ssl
        elif "GIT_SSL_NO_VERIFY" in os.environ:
            del os.environ["GIT_SSL_NO_VERIFY"]


def run_midas_on_frames(
    frames: list,
    out_png_dir: str,
    out_npy_dir: str = None,
    model=None,
    transform=None,
    processor=None,
):
    """
    Run MiDaS depth estimation on extracted frames.

    Args:
        frames: List of frame file paths
        out_png_dir: Directory to save depth PNG files
        out_npy_dir: Directory to save depth NPY files (optional)
        model: MiDaS model (torch.hub style, optional)
        transform: MiDaS transform (torch.hub style, optional)
        processor: DepthProcessor (HuggingFace style, optional)
    """
    # Use HuggingFace processor if available, otherwise fall back to torch.hub model
    if processor is not None:
        # Use HuggingFace-based depth processor
        os.makedirs(out_png_dir, exist_ok=True)
        if out_npy_dir:
            os.makedirs(out_npy_dir, exist_ok=True)

        for i, fp in enumerate(frames):
            img_bgr = cv2.imread(fp)
            if img_bgr is None:
                print("Warning: failed to read frame:", fp)
                continue

            # Estimate depth using HuggingFace processor
            depth_map = processor.estimate_depth(img_bgr)

            # HuggingFace returns depth in meters (higher = farther)
            # Normalize to [0, 1] where 1 = far (hallway extends forward)
            # This matches the torch.hub normalization where high values = far
            d = depth_map - depth_map.min()
            if d.max() > 0:
                d = d / d.max()
            # Keep as-is: high values = far = hallway (correct for our algorithm)

            # Save PNG as 16-bit
            png16 = np.clip(d * 65535.0, 0, 65535).astype(np.uint16)
            out_png = os.path.join(out_png_dir, f"depth_{i:06d}.png")
            cv2.imwrite(out_png, png16)

            if out_npy_dir:
                np.save(
                    os.path.join(out_npy_dir, f"depth_{i:06d}.npy"),
                    d.astype(np.float32),
                )

            if i % 10 == 0:
                print(f"[{i+1}/{len(frames)}] wrote {out_png}")

        print("Depth estimation pass complete.")
        return

    # Fall back to torch.hub model
    if model is None or transform is None:
        raise RuntimeError(
            "MiDaS model and transform must be provided (or use HuggingFace processor)."
        )

    os.makedirs(out_png_dir, exist_ok=True)
    if out_npy_dir:
        os.makedirs(out_npy_dir, exist_ok=True)

    for i, fp in enumerate(frames):
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            print("Warning: failed to read frame:", fp)
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            pred = model(input_batch)
            pred = (
                torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # Normalize depth to [0, 1] (relative per-frame)
        d = pred - pred.min()
        if d.max() > 0:
            d = d / d.max()

        # Save PNG as 16-bit
        png16 = np.clip(d * 65535.0, 0, 65535).astype(np.uint16)
        out_png = os.path.join(out_png_dir, f"depth_{i:06d}.png")
        cv2.imwrite(out_png, png16)

        if out_npy_dir:
            np.save(
                os.path.join(out_npy_dir, f"depth_{i:06d}.npy"), d.astype(np.float32)
            )

        if i % 10 == 0:
            print(f"[{i+1}/{len(frames)}] wrote {out_png}")

    print("MiDaS pass complete.")


def visualize_open_spaces_on_frame(
    frame_path: str, open_space_columns: list, output_path: str
):
    """
    Draw green vertical lines on a frame to visualize detected open space columns.

    Args:
        frame_path: Path to the original frame image
        open_space_columns: List of column indices where open spaces were detected
        output_path: Path to save the visualized frame
    """
    # Load the original frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: Could not load frame {frame_path}")
        return False

    # Create a copy to draw on
    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    # Draw green vertical lines for each open space column
    # Sort columns to draw them in order
    sorted_cols = sorted([col for col in open_space_columns if 0 <= col < width])

    for col in sorted_cols:
        # Draw a green vertical line (thickness 2 for visibility)
        cv2.line(vis_frame, (col, 0), (col, height), (0, 255, 0), 2)

    # Also draw a semi-transparent overlay to show hallway regions more clearly
    if len(sorted_cols) > 0:
        # Find contiguous regions
        regions = []
        if len(sorted_cols) > 0:
            region_start = sorted_cols[0]
            for i in range(1, len(sorted_cols)):
                if (
                    sorted_cols[i] - sorted_cols[i - 1] > 3
                ):  # Gap of more than 3 columns
                    # End of current region
                    regions.append((region_start, sorted_cols[i - 1]))
                    region_start = sorted_cols[i]
            regions.append((region_start, sorted_cols[-1]))

        # Draw semi-transparent green overlay for each region
        overlay = vis_frame.copy()
        for start_col, end_col in regions:
            cv2.rectangle(overlay, (start_col, 0), (end_col, height), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, vis_frame, 0.85, 0, vis_frame)

    # Save the visualized frame
    cv2.imwrite(output_path, vis_frame)
    return True


def process_all_depth_maps(
    depth_npy_dir: str, max_frames: int = None, frames_dir: str = None
):
    """
    Process all depth maps in the directory and extract open spaces.

    Args:
        depth_npy_dir: Directory containing depth .npy files
        max_frames: Maximum number of frames to process (None = all)
        frames_dir: Directory containing original frames (for visualization)

    Returns:
        List of dictionaries, each containing:
            - 'frame_idx': Frame index
            - 'column_depths': 1D array of average depth per column
            - 'open_space_columns': List of column indices
            - 'azimuths_deg': List of azimuth angles in degrees
    """
    depth_files = sorted(glob.glob(os.path.join(depth_npy_dir, "depth_*.npy")))

    if max_frames:
        depth_files = depth_files[:max_frames]

    results = []

    for depth_file in depth_files:
        # Extract frame index from filename
        basename = os.path.basename(depth_file)
        frame_idx = int(basename.replace("depth_", "").replace(".npy", ""))

        # Load and process depth map
        depth_img = load_depth_from_file(depth_file)
        result = process_depth_map(depth_img)

        result["frame_idx"] = frame_idx
        result["depth_file"] = depth_file
        results.append(result)

        # Visualize all frames (for testing purposes)
        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            if os.path.exists(frame_path):
                output_path = os.path.join(
                    TEST_RESULTS_DIR, f"frame_{frame_idx:06d}_open_spaces.png"
                )
                if visualize_open_spaces_on_frame(
                    frame_path, result["open_space_columns"], output_path
                ):
                    if (
                        frame_idx == 0 or (frame_idx + 1) % 10 == 0
                    ):  # Print every 10th frame
                        print(f"  Saved visualization: {output_path}")

    return results


def main():
    """Main entry point for the depth integration pipeline."""
    parser = argparse.ArgumentParser(description="Depth → Spatial audio (modular)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--frames-dir", type=str, default=FRAMES_DIR)
    parser.add_argument("--depth-dir", type=str, default=DEPTH_DIR)
    parser.add_argument("--depth-npy-dir", type=str, default=DEPTH_NPY_DIR)
    parser.add_argument("--audio-dir", type=str, default=AUDIO_DIR)
    parser.add_argument("--sample-every-s", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument(
        "--run-midas", action="store_true", help="Run MiDaS on extracted frames"
    )
    parser.add_argument(
        "--model", type=str, default="DPT_Large", help="MiDaS model type (torch.hub)"
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use HuggingFace transformers instead of torch.hub (more reliable)",
    )
    parser.add_argument(
        "--huggingface-model",
        type=str,
        default="Intel/dpt-hybrid-midas",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--process-depth",
        action="store_true",
        help="Process depth maps to extract open spaces",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DAREDEVIL - Modular Depth Processing Pipeline")
    print("=" * 60)
    print("Configuration:")
    print(f"  Video: {args.video}")
    print(f"  Run MiDaS: {args.run_midas}")
    print(f"  Process depth: {args.process_depth}")
    print(f"  Spatial audio enabled: {SPATIAL_AUDIO_ENABLED}")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[1/3] Extracting frames...")
    # For continuous audio, extract all frames to cover full video
    # max_frames only limits extraction if not processing depth for audio
    extract_max_frames = (
        None if (args.process_depth and SPATIAL_AUDIO_ENABLED) else args.max_frames
    )
    frames, fps = extract_frames(
        args.video,
        args.frames_dir,
        sample_every_s=args.sample_every_s,
        max_frames=extract_max_frames,
    )
    print(f"Extracted {len(frames)} frames at {fps:.1f} FPS")

    # Step 2: Run depth estimation if requested
    if args.run_midas:
        print("\n[2/3] Running depth estimation...")
        try:
            if args.use_huggingface or HUGGINGFACE_DEPTH_AVAILABLE:
                # Use HuggingFace (more reliable, avoids SSL issues)
                print("  Using HuggingFace transformers (more reliable)...")
                processor = load_midas_huggingface(args.huggingface_model)
                run_midas_on_frames(
                    frames,
                    args.depth_dir,
                    out_npy_dir=args.depth_npy_dir,
                    processor=processor,
                )
                print("Depth estimation complete")
            else:
                # Fall back to torch.hub
                print("  Using torch.hub (may have SSL issues)...")
                midas, midas_transform = load_midas(args.model)
                run_midas_on_frames(
                    frames,
                    args.depth_dir,
                    out_npy_dir=args.depth_npy_dir,
                    model=midas,
                    transform=midas_transform,
                )
                print("MiDaS depth estimation complete")
        except Exception as e:
            print(f"Depth estimation failed: {repr(e)}")
            print("  Skipping depth estimation due to error.")
            if not args.use_huggingface:
                print(
                    "  Tip: Try --use-huggingface flag for more reliable model loading"
                )
    else:
        print("\n[2/3] Skipping depth estimation (use --run-midas to enable)")

    # Step 3: Process depth maps
    if args.process_depth:
        print("\n[3/3] Processing depth maps to extract open spaces...")
        # For continuous audio, process ALL available depth files to cover full video
        # max_frames only applies to frame extraction, not depth processing
        results = process_all_depth_maps(
            args.depth_npy_dir,
            max_frames=None,  # Process all available depth files
            frames_dir=args.frames_dir,
        )
        print(f"Processed {len(results)} depth maps")

        # Print summary
        for result in results[:5]:  # Show first 5
            frame_idx = result["frame_idx"]
            cols = result["open_space_columns"]
            az = result["azimuths_deg"]
            azimuth_str = ", ".join([f"{a:+.1f}°" for a in az]) if az else "none"
            print(f"  Frame {frame_idx}: {len(cols)} open spaces at [{azimuth_str}]")

        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more frames")
    else:
        print("\n[3/3] Skipping depth processing (use --process-depth to enable)")

    # Step 4: Generate spatial audio
    if SPATIAL_AUDIO_ENABLED and args.process_depth:
        print("\n[4/4] Generating continuous spatial audio...")
        os.makedirs(args.audio_dir, exist_ok=True)

        # Get video duration
        try:
            video_duration_s = get_video_duration(args.video)
            print(f"  Video duration: {video_duration_s:.2f} seconds")
        except Exception as e:
            print(f"  Warning: Could not get video duration: {e}")
            # Fallback: estimate from number of frames and sample rate
            video_duration_s = len(results) * args.sample_every_s
            print(
                f"  Estimated duration: {video_duration_s:.2f} seconds (from {len(results)} frames)"
            )

        # Generate continuous audio from all frame results
        continuous_audio = generate_continuous_spatial_audio(
            results, video_duration_s, args.sample_every_s
        )

        if continuous_audio is not None:
            # Save single continuous audio file
            audio_path = os.path.join(args.audio_dir, "spatial_audio.wav")
            if export_spatial_audio(continuous_audio, audio_path):
                print(f"  Generated continuous spatial audio: {audio_path}")
                print(f"  Audio duration: {len(continuous_audio) / 48000:.2f} seconds")
            else:
                print("  Failed to save audio file")
        else:
            print("  No audio generated (no open spaces detected)")
    elif SPATIAL_AUDIO_ENABLED:
        print(
            "\n[4/4] Spatial audio enabled but depth processing not run (use --process-depth)"
        )
    else:
        print("\n[4/4] Spatial audio disabled")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
