"""
Local-friendly version of the Colab notebook exported to test.py.
"""

import os
import sys
import math
import time
import glob
import argparse

import numpy as np
import cv2
import torch
import soundfile as sf

# Optional dependencies
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

# Try import pygame for live playback (more stable than simpleaudio)
try:
    import pygame

    pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=512)
    AUDIO_PLAYBACK_AVAILABLE = True
    print("pygame loaded successfully")
except Exception as e:
    print(f"Error: pygame import failed: {e}")
    print("  Install with: pip3 install pygame")
    AUDIO_PLAYBACK_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

# Paths (project-local defaults)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
DEPTH_NPY_DIR = os.path.join(DATA_DIR, "depth_npy")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(DEPTH_NPY_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


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
    print(
        f"Loading MiDaS model '{model_type}' via torch.hub (this may download weights)..."
    )
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


# 6) Run MiDaS on frames
def run_midas_on_frames(
    frames, out_png_dir, out_npy_dir=None, model=None, transform=None
):
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


# 7) Open-space detection
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


LOWER_BAND_FRAC = 0.33  # lower 1/3 of the image
SMOOTH_KERNEL = 5  # for gaussian smoothing (pixels on the 1D column signal)
MIN_PROM = 0.01  # minimum prominence for a peak (0..1 depth scale) - LOWERED for better detection
K_CANDIDATES = (
    5  # max number of open-space directions per frame (increased for more detail)
)
HFOV_DEG = 90.0  # approximate camera horizontal FOV


def column_depth_signal(depth_img):
    H, W = depth_img.shape
    y0 = int(H * (1.0 - LOWER_BAND_FRAC))
    band = depth_img[y0:H, :]
    sig = band.mean(axis=0)  # average depth per column
    # Smooth the signal to reduce noise
    sig_s = gaussian_filter1d(sig, sigma=SMOOTH_KERNEL)
    return sig_s


def find_open_space_columns(sig, k=K_CANDIDATES, min_prom=MIN_PROM):
    s = sig.copy()
    picks = []
    for _ in range(k * 3):
        idx = int(np.argmax(s))
        prom = s[idx] - np.median(s[max(0, idx - 20) : idx + 21])
        if prom < min_prom:
            break
        picks.append((idx, s[idx], prom))
        left = max(0, idx - 25)
        right = min(len(s), idx + 26)
        s[left:right] = -np.inf
    picks = sorted(picks, key=lambda t: t[1], reverse=True)[:k]
    cols = [p[0] for p in picks]
    return cols


def columns_to_azimuths(cols, width, hfov_deg=HFOV_DEG):
    center = (width - 1) / 2.0
    norm = (np.array(cols) - center) / center  # -1..1
    az = norm * (hfov_deg / 2.0)  # degrees
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


# 8) Enhanced Spatial Audio - Continuous Frequency-Varying Noise
def generate_filtered_noise(
    duration_s=2.0, sample_rate=48000, center_freq=1000, bandwidth=500
):
    """Generate bandpass-filtered white noise centered at a specific frequency"""
    n = int(duration_s * sample_rate)
    noise = np.random.normal(0, 0.15, size=(n,)).astype(np.float32)

    # FFT filtering to create bandpass
    X = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)

    # Bandpass filter: only keep frequencies around center_freq ± bandwidth
    low_cutoff = max(0, center_freq - bandwidth)
    high_cutoff = center_freq + bandwidth

    # Create frequency mask
    mask = np.zeros_like(freqs)
    mask[(freqs >= low_cutoff) & (freqs <= high_cutoff)] = 1.0

    # Apply Gaussian rolloff for smoother filtering
    for i, f in enumerate(freqs):
        if f < low_cutoff or f > high_cutoff:
            dist = min(abs(f - low_cutoff), abs(f - high_cutoff))
            mask[i] = np.exp(-((dist / (bandwidth * 0.3)) ** 2))

    X = X * mask
    filtered = np.fft.irfft(X).astype(np.float32)

    return filtered


def add_reverb(signal, depth_norm, sample_rate=48000):
    """Add simple reverb based on depth (farther = more reverb)"""
    # More reverb for farther objects
    delay_ms = int(30 + depth_norm * 150)  # 30-180ms delay
    delay_samples = int(delay_ms * sample_rate / 1000)

    if delay_samples >= len(signal):
        return signal

    reverb = np.zeros_like(signal)
    reverb[delay_samples:] = signal[:-delay_samples] * (0.15 + depth_norm * 0.35)

    return signal + reverb


def azimuth_to_pan(az_deg):
    """Convert azimuth to stereo pan (0=left, 0.5=center, 1=right)"""
    return (az_deg + 90.0) / 180.0


def pan_stereo(mono, pan):
    """Pan mono signal to stereo"""
    left_gain = math.cos(pan * math.pi / 2)
    right_gain = math.cos((1 - pan) * math.pi / 2)
    L = (mono * left_gain).astype(np.float32)
    R = (mono * right_gain).astype(np.float32)
    return np.stack([L, R], axis=-1)


def depth_to_frequency(depth_norm):
    """Map depth to frequency band center (closer = higher freq, farther = lower freq)"""
    # Much wider frequency range for more dramatic variation
    # Closer objects = higher frequency (more treble)
    # Farther objects = lower frequency (more bass)
    return 3500 - (depth_norm * 2800)  # 3500Hz (close) to 700Hz (far)


def depth_to_gain(depth_norm):
    """Map depth to volume (farther = slightly louder for detection)"""
    return 0.3 + 0.7 * (depth_norm**0.7)


def export_frame_audio(
    frame_idx,
    azimuths_deg,
    duration_s=2.0,
    sr=48000,
    out_dir=AUDIO_DIR,
    depth_npy_dir=DEPTH_NPY_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(depth_npy_dir, f"depth_{frame_idx:06d}.npy")
    if not os.path.exists(npy_path):
        print("Missing depth npy:", npy_path)
        return None
    d = np.load(npy_path)
    H, W = d.shape
    y0 = int(H * (1.0 - LOWER_BAND_FRAC))
    band = d[y0:H, :]
    col_mean = band.mean(axis=0)

    # Start with silence
    mix = np.zeros((int(duration_s * sr), 2), dtype=np.float32)

    # If no azimuths detected, use default positions
    if len(azimuths_deg) == 0:
        print(
            f"  Warning: No azimuths detected for frame {frame_idx}, using sweep across field"
        )
        azimuths_deg = [-45.0, -22.5, 0.0, 22.5, 45.0]  # Wide sweep

    # Sort azimuths and get their depths for better frequency separation
    azimuth_depth_pairs = []
    for az in azimuths_deg:
        pan = azimuth_to_pan(az)
        col = int((pan) * (W - 1))
        col = np.clip(col, 0, W - 1)
        dist_norm = float(col_mean[col])
        azimuth_depth_pairs.append((az, dist_norm))

    # Generate continuous filtered noise for each direction
    for az, dist_norm in azimuth_depth_pairs:
        pan = azimuth_to_pan(az)

        # Generate filtered noise with frequency based on depth
        center_freq = depth_to_frequency(dist_norm)
        # Narrower bandwidth for more distinct tones
        bandwidth = 150 + (dist_norm * 150)  # 150-300Hz bandwidth

        filtered_noise = generate_filtered_noise(duration_s, sr, center_freq, bandwidth)

        # Add reverb based on depth
        filtered_noise = add_reverb(filtered_noise, dist_norm, sr)

        # Apply gain based on depth - boost for clarity
        gain = depth_to_gain(dist_norm) * 1.5  # Increased gain
        filtered_noise = filtered_noise * gain

        # Pan to stereo
        stereo_noise = pan_stereo(filtered_noise, pan)

        # Add to mix
        mix[: len(stereo_noise), :] += stereo_noise

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix / (peak / 0.95)

    out_path = os.path.join(out_dir, f"audio_frame_{frame_idx:06d}.wav")
    sf.write(out_path, mix, sr)
    return out_path


def play_audio_file(audio_path):
    """Play audio file using pygame (more stable than simpleaudio)"""
    if not AUDIO_PLAYBACK_AVAILABLE:
        print("  Warning: pygame not available, skipping playback")
        return False

    try:
        # Load and play the sound
        sound = pygame.mixer.Sound(audio_path)
        channel = sound.play()

        # Wait for playback to finish
        while channel.get_busy():
            pygame.time.wait(100)

        return True
    except Exception as e:
        print(f"  Error: Playback error: {e}")
        return False


def play_audio_files_continuous(audio_paths, crossfade_ms=500):
    """Play multiple audio files continuously with crossfade"""
    if not AUDIO_PLAYBACK_AVAILABLE:
        print("  Warning: pygame not available, skipping playback")
        return False

    if len(audio_paths) == 0:
        return False

    try:
        # Use multiple channels for overlapping playback
        pygame.mixer.set_num_channels(len(audio_paths) + 2)

        channels = []
        sounds = []

        # Pre-load all sounds
        for path in audio_paths:
            sounds.append(pygame.mixer.Sound(path))

        # Play first sound
        channel = sounds[0].play()
        channels.append(channel)

        # Calculate when to start each subsequent sound
        for i in range(1, len(sounds)):
            # Get duration of previous sound
            prev_duration_ms = int(sounds[i - 1].get_length() * 1000)

            # Wait until it's time to start the next sound (with overlap)
            wait_time_ms = prev_duration_ms - crossfade_ms

            if wait_time_ms > 0:
                pygame.time.wait(wait_time_ms)

            # Start next sound (it will overlap/crossfade)
            channel = sounds[i].play()
            channels.append(channel)

        # Wait for all sounds to finish
        while any(ch.get_busy() for ch in channels):
            pygame.time.wait(100)

        return True
    except Exception as e:
        print(f"  Error: Continuous playback error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Depth → Spatial audio (local)")
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
        "--model", type=str, default="DPT_Large", help="MiDaS model type"
    )
    parser.add_argument(
        "--export", type=int, default=3, help="How many frames to export audio for"
    )
    parser.add_argument(
        "--live", action="store_true", help="Play audio live (requires pygame)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Play frames continuously without gaps (crossfade)",
    )
    parser.add_argument(
        "--crossfade",
        type=int,
        default=500,
        help="Crossfade duration in ms (default: 500)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DAREDEVIL - Frequency-Varying Spatial Audio")
    print("=" * 60)
    print("Configuration:")
    print(f"  Video: {args.video}")
    print(f"  Run MiDaS: {args.run_midas}")
    print(f"  Export frames: {args.export}")
    print(f"  Live playback: {args.live}")
    print(f"  Continuous mode: {args.continuous}")
    if args.continuous:
        print(f"  Crossfade: {args.crossfade}ms")
    print(f"  Audio playback available: {AUDIO_PLAYBACK_AVAILABLE}")
    print("=" * 60)

    # 1) Extract frames
    print("\n[1/4] Extracting frames...")
    frames, fps = extract_frames(
        args.video,
        args.frames_dir,
        sample_every_s=args.sample_every_s,
        max_frames=args.max_frames,
    )
    print(f"Extracted {len(frames)} frames at {fps:.1f} FPS")

    # 2) Run MiDaS if requested
    if args.run_midas:
        print("\n[2/4] Running MiDaS depth estimation...")
        try:
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
            print(f"Error: MiDaS run failed: {repr(e)}")
            print("  Skipping MiDaS due to error.")
    else:
        print("\n[2/4] Skipping MiDaS (use --run-midas to enable)")

    # 3) Detect open-space candidates
    print("\n[3/4] Detecting open-space candidates...")
    depth_glob = os.path.join(args.depth_dir, "depth_*.png")
    open_space = detect_open_spaces_for_depths(depth_glob)
    print(f"Detected open-space candidates for {len(open_space)} frames")

    # 4) Export and play audio
    print(
        f"\n[4/4] Generating frequency-varying audio for {min(args.export, len(open_space))} frames..."
    )
    out_paths = []
    for i in range(min(args.export, len(open_space))):
        az = open_space[i]["azimuths_deg"]
        p = export_frame_audio(
            i,
            az,
            duration_s=2.0,
            sr=48000,
            out_dir=args.audio_dir,
            depth_npy_dir=args.depth_npy_dir,
        )
        if p:
            out_paths.append(p)
            azimuth_str = (
                ", ".join([f"{a:+.1f}°" for a in az]) if az else "default sweep"
            )
            print(f"  Frame {i}: [{azimuth_str}]")

    print(f"\nGenerated {len(out_paths)} audio files")

    # Live playback
    if args.live:
        if not AUDIO_PLAYBACK_AVAILABLE:
            print("\nWarning: Audio playback not available. Install with:")
            print("    pip3 install pygame")
            print("\n  WAV files saved for manual playback:")
            for p in out_paths:
                print(f"    {p}")
        else:
            print("\n" + "=" * 60)
            print("LIVE SPATIAL AUDIO PLAYBACK")
            print("=" * 60)
            print("Put on your headphones and experience the spatial audio!")
            print("Listen for white noise that shifts in pitch based on depth...")
            print("Higher pitch = closer objects, Lower pitch = farther objects")

            if args.continuous:
                print(f"\nContinuous playback mode (crossfade: {args.crossfade}ms)")
                print()

                # Show all frames that will play
                for i, audio_path in enumerate(out_paths):
                    azimuth_str = (
                        ", ".join([f"{a:+.1f}°" for a in open_space[i]["azimuths_deg"]])
                        if open_space[i]["azimuths_deg"]
                        else "default sweep"
                    )
                    print(
                        f"   Frame {i+1}: {os.path.basename(audio_path)} [{azimuth_str}]"
                    )

                print("\nPlaying all frames continuously...")
                success = play_audio_files_continuous(
                    out_paths, crossfade_ms=args.crossfade
                )

                if success:
                    print("Done")
                else:
                    print("Error: Failed")
            else:
                print("\nFrame-by-frame playback mode")
                print()

                for i, audio_path in enumerate(out_paths):
                    azimuth_str = (
                        ", ".join([f"{a:+.1f}°" for a in open_space[i]["azimuths_deg"]])
                        if open_space[i]["azimuths_deg"]
                        else "default sweep"
                    )
                    print(f"\nFrame {i+1}/{len(out_paths)}")
                    print(f"   File: {os.path.basename(audio_path)}")
                    print(f"   Directions: [{azimuth_str}]")
                    print(f"   Playing... ", end="", flush=True)

                    success = play_audio_file(audio_path)
                    if success:
                        print("Done")
                    else:
                        print("Error: Failed")

                    # Small pause between frames
                    if i < len(out_paths) - 1:
                        time.sleep(0.3)

            print("\n" + "=" * 60)
            print("Live playback complete!")
            print("=" * 60)
    else:
        print("\nTip: Use --live flag to enable live audio playback")
        print("   Use --live --continuous for seamless continuous playback")
        print("   WAV files saved:")
        for p in out_paths:
            print(f"    {p}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
