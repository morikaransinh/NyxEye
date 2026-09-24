import Foundation

/// Pure functions mapping detections to audio rendering parameters.
///
/// Conventions carried over from the codesign sessions (and now actually
/// wired to the output, unlike the prototype where the 700–3500 Hz mapping
/// was computed and discarded):
/// - Distance -> band-pass center frequency: near = high (3500 Hz),
///   far = low (700 Hz).
/// - Open spaces: farther = slightly louder (a gentle beckon toward depth).
/// - Obstacles: closer = louder and faster pulses (urgency), the opposite
///   polarity — the prototype had this inverted.
public enum CueMapping {
    /// Depth range the frequency/gain maps span, matching useful LiDAR range.
    public static let maxDepthMeters: Float = 5.0

    public static let nearFrequencyHz: Float = 3500
    public static let farFrequencyHz: Float = 700

    /// Band-pass center frequency for a cue at `depthMeters`.
    /// Clamped: anything at or beyond maxDepthMeters renders at 700 Hz,
    /// anything at 0 renders at 3500 Hz.
    public static func frequency(forDepthMeters depth: Float) -> Float {
        let t = normalizedDepth(depth)
        return nearFrequencyHz + (farFrequencyHz - nearFrequencyHz) * t
    }

    /// Band-pass width grows with distance (150 Hz near, 300 Hz far) so far
    /// cues sound diffuse and near cues sound focused.
    public static func bandwidth(forDepthMeters depth: Float) -> Float {
        150 + 150 * normalizedDepth(depth)
    }

    /// Open-space gain in 0...1: compressive curve, farther = louder.
    /// 0.3 floor keeps confirmed openings audible even when shallow.
    public static func openSpaceGain(forDepthMeters depth: Float) -> Float {
        0.3 + 0.7 * pow(normalizedDepth(depth), 0.7)
    }

    /// Obstacle gain in 0...1: full urgency at arm's length, tapering with
    /// distance. Strictly decreasing in distance — closer is ALWAYS louder.
    public static func obstacleGain(forDistanceMeters distance: Float) -> Float {
        let near: Float = 0.5   // full gain at or inside this distance
        let far: Float = 3.0    // matches ObstacleConfig.reportDistanceMeters
        let t = (min(max(distance, near), far) - near) / (far - near)
        return 1.0 - 0.8 * t   // 1.0 near -> 0.2 at report edge
    }

    /// Obstacle pulse rate in Hz: 8 pulses/s at arm's length down to 1.5/s
    /// at the far report edge. Strictly decreasing in distance.
    public static func obstaclePulseRateHz(forDistanceMeters distance: Float) -> Float {
        let near: Float = 0.5
        let far: Float = 3.0
        let t = (min(max(distance, near), far) - near) / (far - near)
        return 8.0 - 6.5 * t
    }

    /// Unit-vector position for a spatial audio source at the given azimuth,
    /// in the listener's frame: +x right, +y up, -z ahead.
    public static func sourcePosition(azimuthDegrees: Float) -> (x: Float, y: Float, z: Float) {
        let radians = azimuthDegrees * .pi / 180
        return (x: sin(radians), y: 0, z: -cos(radians))
    }

    static func normalizedDepth(_ depth: Float) -> Float {
        min(max(depth / maxDepthMeters, 0), 1)
    }
}
