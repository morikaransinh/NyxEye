import Foundation

/// A tracked opening with temporal smoothing state.
public struct TrackedSpace: Equatable {
    public let id: Int
    /// Smoothed center azimuth in degrees.
    public let azimuthDegrees: Float
    public let angularWidthDegrees: Float
    public let depthMeters: Float
    /// Rendering gain 0...1: ramps in when confirmed, fades out *in place*
    /// when the opening disappears.
    public let gain: Float
}

public struct OpenSpaceTrackerConfig {
    /// Detections match an existing track when within this azimuth gate.
    public var matchGateDegrees: Float = 15.0
    /// Consecutive frames a new opening must appear before it is emitted
    /// (debounce against single-frame noise).
    public var confirmFrames: Int = 3
    /// Frames a missing track keeps its last gain before starting to fade.
    public var holdFrames: Int = 3
    /// Frames over which a disappearing track fades from its gain to 0.
    public var fadeFrames: Int = 8
    /// Exponential smoothing factor for azimuth/width/depth (0 = frozen,
    /// 1 = no smoothing).
    public var smoothingAlpha: Float = 0.4
    /// Frames over which a confirmed track ramps from 0 to full gain.
    public var rampInFrames: Int = 4

    public init() {}
}

/// Temporal tracker for open spaces.
///
/// Two properties the earlier prototype got wrong, guaranteed here:
/// - New openings never "fade in from center": a track's azimuth starts at the
///   detection's azimuth and is only ever smoothed toward subsequent
///   *detections*.
/// - Disappearing openings never move: azimuth is frozen at its last smoothed
///   value while the gain fades to zero. (The prototype decayed azimuth toward
///   0°, steering users toward a direction that no longer existed.)
public final class OpenSpaceTracker {
    private struct Track {
        let id: Int
        var azimuth: Float
        var width: Float
        var depth: Float
        var seenFrames: Int
        var missedFrames: Int
        var gainAtLoss: Float
        var gain: Float
        var confirmed: Bool
    }

    private let config: OpenSpaceTrackerConfig
    private var tracks: [Track] = []
    private var nextID = 1

    public init(config: OpenSpaceTrackerConfig = OpenSpaceTrackerConfig()) {
        self.config = config
    }

    public func reset() {
        tracks.removeAll()
    }

    /// Advance one frame with the latest detections. Returns confirmed tracks
    /// with nonzero gain, deepest first.
    public func update(with detections: [OpenSpace]) -> [TrackedSpace] {
        var unmatched = detections

        // Greedy nearest-azimuth matching, existing tracks first.
        for i in tracks.indices {
            var bestIndex: Int? = nil
            var bestDelta = config.matchGateDegrees
            for (j, detection) in unmatched.enumerated() {
                let delta = abs(detection.azimuthDegrees - tracks[i].azimuth)
                if delta <= bestDelta {
                    bestDelta = delta
                    bestIndex = j
                }
            }
            if let j = bestIndex {
                let d = unmatched.remove(at: j)
                let a = config.smoothingAlpha
                tracks[i].azimuth += a * (d.azimuthDegrees - tracks[i].azimuth)
                tracks[i].width += a * (d.angularWidthDegrees - tracks[i].width)
                tracks[i].depth += a * (d.depthMeters - tracks[i].depth)
                tracks[i].seenFrames += 1
                tracks[i].missedFrames = 0
                if !tracks[i].confirmed && tracks[i].seenFrames >= config.confirmFrames {
                    tracks[i].confirmed = true
                }
                if tracks[i].confirmed {
                    let step = 1.0 / Float(max(1, config.rampInFrames))
                    tracks[i].gain = min(1, tracks[i].gain + step)
                    tracks[i].gainAtLoss = tracks[i].gain
                }
            } else {
                tracks[i].missedFrames += 1
                if tracks[i].missedFrames > config.holdFrames {
                    // Fade in place: gain decays, azimuth untouched.
                    let fadeProgress = Float(tracks[i].missedFrames - config.holdFrames) / Float(max(1, config.fadeFrames))
                    tracks[i].gain = max(0, tracks[i].gainAtLoss * (1 - fadeProgress))
                }
            }
        }

        // Remove dead tracks: fully faded, or unconfirmed and missing.
        tracks.removeAll { track in
            (track.missedFrames > 0 && !track.confirmed) ||
            (track.missedFrames > config.holdFrames && track.gain <= 0)
        }

        // New tracks for unmatched detections (start unconfirmed, gain 0).
        for detection in unmatched {
            tracks.append(Track(
                id: nextID,
                azimuth: detection.azimuthDegrees,
                width: detection.angularWidthDegrees,
                depth: detection.depthMeters,
                seenFrames: 1,
                missedFrames: 0,
                gainAtLoss: 0,
                gain: 0,
                confirmed: config.confirmFrames <= 1
            ))
            nextID += 1
        }

        return tracks
            .filter { $0.confirmed && $0.gain > 0 }
            .sorted { $0.depth > $1.depth }
            .map { TrackedSpace(id: $0.id,
                                azimuthDegrees: $0.azimuth,
                                angularWidthDegrees: $0.width,
                                depthMeters: $0.depth,
                                gain: $0.gain) }
    }
}

public struct ObstacleSmootherConfig {
    /// Consecutive frames an obstacle must be present before warning.
    public var confirmFrames: Int = 2
    /// Once warning, keep warning until the obstacle is farther than
    /// warnDistance * this factor (release hysteresis against flutter).
    public var releaseFactor: Float = 1.15
    public var warnDistanceMeters: Float = 2.0
    /// Exponential smoothing for distance and azimuth.
    public var smoothingAlpha: Float = 0.5

    public init() {}
}

/// Debounce + hysteresis for obstacle warnings so the cue neither flickers on
/// single-frame noise nor stutters at the warning boundary.
public final class ObstacleSmoother {
    private let config: ObstacleSmootherConfig
    private var presentFrames = 0
    private var active = false
    private var distance: Float = 0
    private var azimuth: Float = 0

    public init(config: ObstacleSmootherConfig = ObstacleSmootherConfig()) {
        self.config = config
    }

    public func reset() {
        presentFrames = 0
        active = false
    }

    public func update(with detection: Obstacle?) -> Obstacle? {
        guard let detection else {
            presentFrames = 0
            active = false
            return nil
        }

        if presentFrames == 0 {
            distance = detection.distanceMeters
            azimuth = detection.azimuthDegrees
        } else {
            let a = config.smoothingAlpha
            distance += a * (detection.distanceMeters - distance)
            azimuth += a * (detection.azimuthDegrees - azimuth)
        }
        presentFrames += 1

        if !active {
            if presentFrames >= config.confirmFrames && distance < config.warnDistanceMeters {
                active = true
            }
        } else if distance > config.warnDistanceMeters * config.releaseFactor {
            active = false
        }

        return active ? Obstacle(distanceMeters: distance, azimuthDegrees: azimuth) : nil
    }
}
