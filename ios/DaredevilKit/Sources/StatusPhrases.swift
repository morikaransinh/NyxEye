import Foundation

/// Builds the spoken status phrases from detection state. Speech announces
/// *what changed*; the spatial noise cues carry continuous direction and
/// distance.
public enum StatusPhrases {
    /// nil means "nothing worth saying" (scanning, no cues).
    public static func describe(openSpaces: [TrackedSpace], obstacle: Obstacle?) -> String? {
        if let obstacle {
            let direction = directionWord(azimuthDegrees: obstacle.azimuthDegrees)
            let meters = (obstacle.distanceMeters * 2).rounded() / 2
            return "Obstacle \(formatMeters(meters)) ahead\(direction.map { ", \($0)" } ?? "")"
        }
        if openSpaces.isEmpty {
            return nil
        }
        let directions = openSpaces.map { directionWord(azimuthDegrees: $0.azimuthDegrees) ?? "ahead" }
        if directions.count == 1 {
            return "Open space \(directions[0])"
        }
        return "Open spaces \(directions.joined(separator: " and "))"
    }

    /// Coarse direction bucket; nil means straight ahead (within 8°).
    /// Coarse on purpose: azimuth jitter must not churn out new phrases.
    public static func directionWord(azimuthDegrees az: Float) -> String? {
        switch az {
        case ..<(-24): return "far left"
        case ..<(-8): return "to the left"
        case ...8: return nil
        case ...24: return "to the right"
        default: return "far right"
        }
    }

    public static func formatMeters(_ m: Float) -> String {
        if m == m.rounded() {
            return "\(Int(m)) meter\(m == 1 ? "" : "s")"
        }
        return String(format: "%.1f meters", m)
    }
}