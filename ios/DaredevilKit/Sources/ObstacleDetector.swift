import Foundation

/// The nearest obstacle in the user's walking corridor.
public struct Obstacle: Equatable {
    /// Distance to the obstacle in meters.
    public let distanceMeters: Float
    /// Azimuth of the obstacle in degrees (negative = left).
    public let azimuthDegrees: Float

    public init(distanceMeters: Float, azimuthDegrees: Float) {
        self.distanceMeters = distanceMeters
        self.azimuthDegrees = azimuthDegrees
    }
}

public struct ObstacleConfig {
    /// Only columns within +/- this azimuth are considered — the corridor the
    /// user is about to walk through.
    public var corridorHalfAngleDegrees: Float = 15.0
    /// Vertical band analyzed (head to knee height, generous).
    public var bandTopFraction: Float = 0.15
    public var bandBottomFraction: Float = 0.85
    /// Report the nearest obstacle when closer than this. Deliberately larger
    /// than the warning threshold (ObstacleSmootherConfig.warnDistanceMeters)
    /// so the smoother's release hysteresis can see obstacles just beyond the
    /// warning boundary instead of losing them entirely.
    public var reportDistanceMeters: Float = 3.0
    /// Only high-confidence samples count. Obstacle warnings are
    /// safety-critical; noise must not fire them.
    public var minConfidence: UInt8 = 2
    /// Robust near-distance statistic: this percentile of a column's valid
    /// samples, so single-pixel noise can't fake a nearby obstacle.
    public var nearPercentile: Float = 0.1
    /// Minimum valid samples per column for it to participate.
    public var minSamplesPerColumn: Int = 8

    public init() {}
}

/// Finds the nearest obstacle inside the central walking corridor of a
/// metric depth grid. Distances are meters; smaller = closer = more urgent.
public enum ObstacleDetector {

    public static func detect(in grid: DepthGrid, config: ObstacleConfig = ObstacleConfig()) -> Obstacle? {
        let band = grid.rowBand(topFraction: config.bandTopFraction, bottomFraction: config.bandBottomFraction)

        // Per-column robust near distance within the corridor.
        var columnNear: [(column: Int, near: Float)] = []
        var samples: [Float] = []
        samples.reserveCapacity(band.count)

        for column in 0..<grid.width {
            let azimuth = grid.azimuthDegrees(forColumn: column)
            guard abs(azimuth) <= config.corridorHalfAngleDegrees else { continue }

            samples.removeAll(keepingCapacity: true)
            for row in band {
                let d = grid.depth(row: row, column: column)
                if DepthGrid.isValid(d) && grid.confidence(row: row, column: column) >= config.minConfidence {
                    samples.append(d)
                }
            }
            guard samples.count >= config.minSamplesPerColumn else { continue }
            columnNear.append((column, percentile(&samples, config.nearPercentile)))
        }

        guard !columnNear.isEmpty else { return nil }

        // Median-of-3 smoothing across neighboring columns so one noisy column
        // can't fire a warning on its own.
        var nearest: (column: Int, near: Float)? = nil
        for i in columnNear.indices {
            let lo = max(columnNear.startIndex, i - 1)
            let hi = min(columnNear.endIndex - 1, i + 1)
            var window = columnNear[lo...hi].map { $0.near }
            let smoothed = OpenSpaceDetector.medianOf(&window)
            if nearest == nil || smoothed < nearest!.near {
                nearest = (columnNear[i].column, smoothed)
            }
        }

        guard let found = nearest, found.near < config.reportDistanceMeters else { return nil }
        return Obstacle(
            distanceMeters: found.near,
            azimuthDegrees: grid.azimuthDegrees(forColumn: found.column)
        )
    }

    /// Percentile of a mutable buffer (sorts in place). p in [0, 1].
    static func percentile(_ values: inout [Float], _ p: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        values.sort()
        let clamped = max(0, min(1, p))
        let index = Int(clamped * Float(values.count - 1))
        return values[index]
    }
}
