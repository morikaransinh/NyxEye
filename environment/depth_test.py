"""
Local-friendly version of the Colab notebook exported to test.py.

Changes made:
- Removed Colab-only shell (!) commands and apt-get installs.
- Replaced /content paths with project-local ./data paths.
- Depth estimation only - audio generation removed
"""

import os
import sys
import math
import time
import glob
import json
import random
import argparse

import numpy as np
import cv2
import torch

# cd /Users/annachan/Project-Daredevil/environment
#python3 depth_test.py --video sample.mov --run-midas

# Run --> python3 depth_test.py --video newSample.mov --run-midas --visualize 5

# Optional dependencies
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

# Paths (project-local defaults)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
DEPTH_NPY_DIR = os.path.join(DATA_DIR, "depth_npy")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(DEPTH_NPY_DIR, exist_ok=True)

# 2) Utilities: video frame extraction @ 0.2s (5 FPS)
def extract_frames(video_path, out_dir, sample_every_s=0.2, max_frames=None):
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

print("Frame extraction utility ready.")

# 3) Load MiDaS from torch.hub (Intel-ISL)
def load_midas(model_type="DPT_Large"):
    print(f"Loading MiDaS model '{model_type}' via torch.hub (this may download weights)...")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_type:
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform
        print("MiDaS loaded successfully.")
        return midas, transform
    except Exception as e:
        print("Error loading MiDaS:", repr(e))
        raise

# Try to lazily load the model only when user requests running MiDaS.
midas = None
midas_transform = None

# 6) Run MiDaS on frames
def run_midas_on_frames(frames, out_png_dir, out_npy_dir=None, model=None, transform=None):
    if model is None or transform is None:
        raise RuntimeError("MiDaS model and transform must be provided.")
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
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()


        # Normalize depth to [0, 1] (relative per-frame)
        d = pred - pred.min()
        if d.max() > 0:
            d = d / d.max()
        # d = 1.0 - d # added this line 11/21

        # Save PNG as 16-bit
        png16 = np.clip(d * 65535.0, 0, 65535).astype(np.uint16)
        out_png = os.path.join(out_png_dir, f"depth_{i:06d}.png")
        cv2.imwrite(out_png, png16)

        if out_npy_dir:
            np.save(os.path.join(out_npy_dir, f"depth_{i:06d}.npy"), d.astype(np.float32))

        if i % 10 == 0:
            print(f"[{i+1}/{len(frames)}] wrote {out_png}")
    print("MiDaS pass complete.")

# 7) Open-space detection
#
# DEFINITION: An "open space" is a contiguous region where depth is HIGH
# (far from camera = nothing blocking the path). We want to find the CENTER
# of these open regions, not the edges.
#
# APPROACH:
# 1. Threshold the depth signal to find "far" regions (above some percentile)
# 2. Find contiguous runs of "far" pixels
# 3. Return the CENTER of each run (that's where the open space is)

if gaussian_filter1d is None:
    # simple numpy smoothing fallback
    def gaussian_filter1d(x, sigma):
        from math import exp
        # very small Gaussian kernel
        radius = max(1, int(3 * sigma))
        xs = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (xs / float(sigma)) ** 2)
        k = k / k.sum()
        return np.convolve(x, k, mode="same")

# 11/20 -add DEPTH_THRESHOLD_PERCENTILE, MIN_RUN_WIDTH, EDGE_MARGIN, remove MIN_PROM
# LOWER_BAND_FRAC = 0.33       # lower 1/3 of the image
SMOOTH_KERNEL   = 5          # for gaussian smoothing (pixels on the 1D column signal)
DEPTH_THRESHOLD_PERCENTILE = 70 # Columns above this threshold % are OPEN
MIN_RUN_WIDTH = 20           # minimum width (pixels) to count as an open space
HFOV_DEG = 90.0              # approximate camera horizontal field of view (HOV)
K_CANDIDATES    = 5          # max number of open-space directions per frame (increased for more detail)
EDGE_MARGIN = 0.10       # ignore detections in outer 10% of image (edge)

# 11/21 edit to try getting middle portion
def column_depth_signal(depth_img):
    # collapse middle portion of depth image to 1D signal (avg depth per column)
    H, W = depth_img.shape
    # y0 = int(H * (1.0 - LOWER_BAND_FRAC))
    y_start = int(H * 0.30) # start at 30% from top
    y_end = int(H* 0.70) # end at 70% from top
    band = depth_img[y_start: y_end, :]
    sig = band.mean(axis=0)  # average depth per column
    # Smooth the signal to reduce noise
    sig_s = gaussian_filter1d(sig, sigma=SMOOTH_KERNEL)
    return sig_s

# 11/20 changed this funciton to fit new approach - see docstring
def find_open_space_columns(sig, k=K_CANDIDATES):
    """
    Find CENTERS of open space region (contingous high depth areas)
    Returns list of column indices reprsenting centers of open spaces.
    """
    W = len(sig)

    # Define edge margins to ignore
    left_margin = int(W * EDGE_MARGIN)
    right_margin = int(W * (1 - EDGE_MARGIN))

    # Threshold: columns with depth above this are "open"
    threshold = np.percentile(sig, DEPTH_THRESHOLD_PERCENTILE)

    # 11/21 also changed this sign to adjust inverting depth
    # Binary mask: True where depth > threshold
    open_mask = sig < threshold

    # Find contiguous runs of True values
    runs = []
    in_run = False
    run_start = 0

    for i in range(W):
        if open_mask[i] and not in_run:
            # Start of a new run
            in_run = True
            run_start = i
        elif not open_mask[i] and in_run:
            # End of a run
            in_run = False
            run_end = i
            runs.append((run_start, run_end))

    # Don't forget the last run if it extends to the edge
    if in_run:
        runs.append((run_start, W))

    # Filter runs: must be wide enough and not in edge margins
    valid_runs = []
    for start, end in runs:
        width = end - start
        center = (start + end) // 2

        # Check minimum width
        if width < MIN_RUN_WIDTH:
            continue

        # Check not in edge margins
        if center < left_margin or center > right_margin:
            continue

        # Score by average depth in the run (higher = more open)
        avg_depth = np.mean(sig[start:end])
        valid_runs.append((center, avg_depth, width))

    # 11/21 edit to reverse property
    # Sort by depth (most open first), then take top k
    valid_runs.sort(key=lambda x: x[1], reverse=False)

    # Return just the center columns
    cols = [r[0] for r in valid_runs[:k]]

    return cols

def columns_to_azimuths(cols, width, hfov_deg=HFOV_DEG):
    center = (width - 1) / 2.0
    norm = (np.array(cols) - center) / center  # -1..1
    az = norm * (hfov_deg / 2.0)              # degrees
    return az.tolist()

def detect_open_spaces_for_depths(depth_glob):
    results = []
    paths = sorted(glob.glob(depth_glob))
    for p in paths:
        d16 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d16 is None:
            continue
        d16 = d16.astype(np.float32)
        d = d16 / 65535.0
        sig = column_depth_signal(d)
        cols = find_open_space_columns(sig)
        az = columns_to_azimuths(cols, d.shape[1])
        results.append({"path": p, "cols": cols, "azimuths_deg": az})
    return results

# 11/20 added this
# creates image that Alison mentioned to see what its doing with depth
def visualize_open_space_detection(depth_path, output_path=None):
    """
    Create a visualization showing:
    - The depth image
    - The 1D depth signal (column averages)
    - Detected open space regions highlighted
    - Center points marked
    """
    import matplotlib.pyplot as plt

    # Load depth image
    d16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d16 is None:
        print(f"Could not load: {depth_path}")
        return
    d = d16.astype(np.float32) / 65535.0
    H, W = d.shape

    # Get the signal and detections
    sig = column_depth_signal(d)
    cols = find_open_space_columns(sig)

    # DEBUG: Print actual values
    print(f"Depth min: {d.min()}, max: {d.max()}")
    print(f"Signal min: {sig.min()}, max: {sig.max()}")
    print(f"Center column value: {sig[W//2]}")
    print(f"Edge column value: {sig[50]}")

    # Calculate threshold for visualization
    threshold = np.percentile(sig, DEPTH_THRESHOLD_PERCENTILE)
    open_mask = sig < threshold

    # Find runs for visualization
    runs = []
    in_run = False
    run_start = 0
    for i in range(W):
        if open_mask[i] and not in_run:
            in_run = True
            run_start = i
        elif not open_mask[i] and in_run:
            in_run = False
            runs.append((run_start, i))
    if in_run:
        runs.append((run_start, W))

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1) Depth image with detected centers marked
    ax1 = axes[0]
    depth_colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    ax1.imshow(depth_rgb)
    ax1.set_title("Depth Map with Detected Open Space Centers", fontsize=12)

    # Draw vertical lines at detected centers
    for col in cols:
        ax1.axvline(x=col, color='lime', linewidth=2, linestyle='--', alpha=0.8)

    # Show the analysis band (lower 1/3)
    # y0 = int(H * (1.0 - LOWER_BAND_FRAC))
    y_start = int(H * 0.15)
    y_end = int(H * 0.45)
    ax1.axhline(y=y_start, color='cyan', linewidth=1, linestyle=':', alpha=0.7)
    ax1.axhline(y=y_end, color='cyan', linewidth=1, linestyle=':', alpha=0.7)
    ax1.text(10, y_start - 10, "Analysis band (upper-middle)", color='cyan', fontsize=9)

    # Show edge margins
    left_margin = int(W * EDGE_MARGIN)
    right_margin = int(W * (1 - EDGE_MARGIN))
    ax1.axvline(x=left_margin, color='red', linewidth=1, linestyle=':', alpha=0.5)
    ax1.axvline(x=right_margin, color='red', linewidth=1, linestyle=':', alpha=0.5)
    ax1.set_ylabel("Pixels")

    # 2) 1D depth signal with threshold and regions
    ax2 = axes[1]
    x_axis = np.arange(W)
    ax2.plot(x_axis, sig, 'b-', linewidth=1.5, label='Depth signal (smoothed)')
    ax2.axhline(y=threshold, color='orange', linewidth=2, linestyle='--',
                label=f'Threshold ({DEPTH_THRESHOLD_PERCENTILE}th percentile)')

    # Shade open regions
    for start, end in runs:
        ax2.axvspan(start, end, alpha=0.3, color='green', label='Open region' if start == runs[0][0] else '')

    # Mark detected centers
    for col in cols:
        ax2.axvline(x=col, color='lime', linewidth=2, linestyle='--', alpha=0.8)
        ax2.plot(col, sig[col], 'go', markersize=10, markeredgecolor='black', markeredgewidth=2)

    # Show edge margins
    ax2.axvspan(0, left_margin, alpha=0.2, color='red')
    ax2.axvspan(right_margin, W, alpha=0.2, color='red')

    ax2.set_xlabel("Column (pixels)")
    ax2.set_ylabel("Average Depth (0=far, 1=close)")
    ax2.set_title("1D Depth Signal with Open Space Detection", fontsize=12)
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, W)
    ax2.grid(True, alpha=0.3)

    # 3) Binary mask showing open vs closed
    ax3 = axes[2]
    mask_img = np.zeros((50, W, 3), dtype=np.uint8)
    for i in range(W):
        if open_mask[i]:
            mask_img[:, i] = [0, 200, 0]  # Green for open
        else:
            mask_img[:, i] = [200, 0, 0]  # Red for blocked

    # Highlight detected centers
    for col in cols:
        mask_img[:, max(0, col-3):min(W, col+4)] = [255, 255, 0]  # Yellow for centers

    ax3.imshow(mask_img, aspect='auto')
    ax3.set_title("Open Space Mask (Green=Open, Red=Blocked, Yellow=Detected Centers)", fontsize=12)
    ax3.set_xlabel("Column (pixels)")
    ax3.set_yticks([])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_path}")
    else:
        plt.show()

    plt.close()

    return cols

# 11/20 added the --visualize argument, added "visualize if requested" script
# basically creates folder for output images "Project-Daredevil/data/visualizations/"
# Calls visualize_open_space detection()
# Saves PNG with depth map, graph, color mask; prints where images saved
def main():
    parser = argparse.ArgumentParser(description="Depth estimation (local)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--frames-dir", type=str, default=FRAMES_DIR)
    parser.add_argument("--depth-dir", type=str, default=DEPTH_DIR)
    parser.add_argument("--depth-npy-dir", type=str, default=DEPTH_NPY_DIR)
    parser.add_argument("--sample-every-s", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--run-midas", action="store_true", help="Run MiDaS on extracted frames")
    parser.add_argument("--model", type=str, default="DPT_Large", help="MiDaS model type")
    parser.add_argument("--visualize", type=int, default=0, help="Number of frames to visualize (saves PNGs)")
    args = parser.parse_args()

    print("="*60)
    print("🔍 DEPTH ESTIMATION TEST")
    print("="*60)
    print("Configuration:")
    print(f"  Video: {args.video}")
    print(f"  Run MiDaS: {args.run_midas}")
    print("="*60)

    # 1) Extract frames
    print("\n[1/3] 🎬 Extracting frames...")
    frames, fps = extract_frames(args.video, args.frames_dir, sample_every_s=args.sample_every_s, max_frames=args.max_frames)
    print(f"✓ Extracted {len(frames)} frames at {fps:.1f} FPS")

    # 2) Run MiDaS if requested
    if args.run_midas:
        print("\n[2/3] 🔍 Running MiDaS depth estimation...")
        global midas, midas_transform
        try:
            midas, midas_transform = load_midas(args.model)
            run_midas_on_frames(frames, args.depth_dir, out_npy_dir=args.depth_npy_dir, model=midas, transform=midas_transform)
            print("✓ MiDaS depth estimation complete")
        except Exception as e:
            print(f"✗ MiDaS run failed: {repr(e)}")
            print("  Skipping MiDaS due to error.")
    else:
        print("\n[2/3] ⏭️  Skipping MiDaS (use --run-midas to enable)")

    # 3) Detect open-space candidates
    print("\n[3/3] 🗺️  Detecting open-space candidates...")
    depth_glob = os.path.join(args.depth_dir, "depth_*.png")
    open_space = detect_open_spaces_for_depths(depth_glob)
    print(f"✓ Detected open-space candidates for {len(open_space)} frames")

    # Print results
    for i, result in enumerate(open_space):
        az = result["azimuths_deg"]
        azimuth_str = ", ".join([f"{a:+.1f}°" for a in az]) if az else "none detected"
        print(f"  Frame {i}: [{azimuth_str}]")

    # 4) Visualize if requested
    if args.visualize > 0:
        print(f"\n[4/4] 📊 Generating visualizations for {args.visualize} frames...")
        VIS_DIR = os.path.join(DATA_DIR, "visualizations")
        os.makedirs(VIS_DIR, exist_ok=True)

        depth_paths = sorted(glob.glob(os.path.join(args.depth_dir, "depth_*.png")))
        for i in range(min(args.visualize, len(depth_paths))):
            out_path = os.path.join(VIS_DIR, f"viz_{i:06d}.png")
            visualize_open_space_detection(depth_paths[i], out_path)

        print(f"✓ Visualizations saved to: {VIS_DIR}")

    print("\n" + "="*60)
    print("✓ Pipeline complete!")
    print("="*60)

if __name__ == "__main__":
    main()
