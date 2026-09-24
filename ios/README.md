# Daredevil iOS

Native iOS app: LiDAR depth → open-space / obstacle detection → HRTF spatial audio for blind and low-vision users. This replaces the Python prototype (kept in the repo as reference); the goals from the codesign sessions are unchanged.

## Requirements

- iPhone Pro model with LiDAR (tested target: iPhone 17 Pro)
- AirPods (Pro recommended, transparency mode on)
- Xcode 16+ on macOS

## Run it on your iPhone

1. Open `ios/Daredevil.xcodeproj` in Xcode.
2. Select the **Daredevil** scheme, and your iPhone as the destination (plug it in, or use Wi-Fi pairing).
3. In the project settings → **Daredevil** target → *Signing & Capabilities*, pick your Apple ID team. A free personal team works.
4. Press **Run**. On first launch on-device, approve the developer certificate under *Settings → General → VPN & Device Management*.
5. On the phone: grant camera access, put in your AirPods (transparency mode), tap **Start scanning** — or double-tap anywhere with two fingers (VoiceOver Magic Tap).

If the project file is missing or stale, regenerate it: `brew install xcodegen && cd ios && xcodegen generate`.

## What you'll hear

| Sound | Meaning |
|---|---|
| Continuous ocean-like whoosh, placed in space | A walkable opening (≥3 m deep, ≥8° wide) in that direction; lower pitch = deeper |
| Pulsing noise | Obstacle within 2 m in your walking corridor; faster + louder + higher pitch = closer |
| Silence while scanning | No confirmed openings, nothing close ahead |
| Speech | Status changes only ("Open space to the left", "Obstacle 1 meter ahead") |

## Architecture

- **`DaredevilKit/`** — pure Swift algorithms, no ARKit/audio dependencies, fully unit-tested:
  - `DepthGrid` — metric depth in the upright camera frame (meters; larger = farther; invalid ≠ open)
  - `OpenSpaceDetector` — absolute metric threshold per column band; walls and dead ends produce *no* cue
  - `ObstacleDetector` — high-confidence-only robust nearest in the ±15° corridor
  - `OpenSpaceTracker` / `ObstacleSmoother` — confirm-before-emit, fade-out-in-place, release hysteresis
  - `CueMapping` — distance → 700–3500 Hz band-pass center, gains, pulse rate, 3D position
  - `HearingGuard` — all hearing-safety policy (see below)
- **`Daredevil/`** — app layer: `ARDepthSource` (LiDAR → DepthGrid, FOV from intrinsics), `SpatialAudioEngine` (brown-noise slots → HRTF environment node → peak limiter), `SpeechAnnouncer`, SwiftUI views.

## Hearing protection

Every gain passes through `HearingGuard`; nothing sets a raw node gain directly:

1. Per-source ceiling (0.8) and a total mix budget (1.2) — more cues never means more loudness.
2. Master volume hard-capped at 0.6 linear; the UI slider maps inside that cap.
3. Gains rise at ≤2.0/s (no sudden onsets); drops are instantaneous.
4. Watchdog: if depth frames stall >0.6 s, audio fades to silence — silence is the fail-safe state.
5. An Apple PeakLimiter sits before the output as a final brickwall.

## Tests

```bash
cd ios
xcodebuild test -project Daredevil.xcodeproj -scheme DaredevilKit \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

82 tests cover the detectors, trackers, cue mapping, spoken phrases, and every HearingGuard property. Several are regression guards against specific prototype bugs (wall-as-hallway, inverted proximity loudness, cue azimuth sliding to center on fade, sub-audible carrier band).
