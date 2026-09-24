import Foundation

/// A 2D grid of metric depth samples in the camera's upright frame.
///
/// Invariants (established by the capture layer, relied on everywhere else):
/// - `depths` are meters along the camera ray. Larger = farther. There is no
///   inverse-depth anywhere in this codebase; LiDAR gives metric depth directly.
/// - Row-major storage. Column 0 is the leftmost column of the user's view,
///   row 0 is the top.
/// - A sample is *invalid* if it is NaN, infinite, or <= 0. Invalid samples are
///   never treated as "open space" — a missing LiDAR return can be a glass door.
public struct DepthGrid {
    public let width: Int
    public let height: Int
    /// Row-major depth samples in meters. Count == width * height.
    public let depths: [Float]
    /// Per-sample confidence 0 (low), 1 (medium), 2 (high).
    /// Empty array means all samples are high confidence.
    public let confidences: [UInt8]
    /// Horizontal field of view covered by the grid, in degrees.
    public let horizontalFOVDegrees: Float

    public init(width: Int, height: Int, depths: [Float], confidences: [UInt8] = [], horizontalFOVDegrees: Float) {
        precondition(width > 0 && height > 0, "DepthGrid dimensions must be positive")
        precondition(depths.count == width * height, "depths count must equal width * height")
        precondition(confidences.isEmpty || confidences.count == depths.count,
                     "confidences must be empty or match depths count")
        precondition(horizontalFOVDegrees > 0 && horizontalFOVDegrees < 180,
                     "horizontal FOV must be in (0, 180) degrees")
        self.width = width
        self.height = height
        self.depths = depths
        self.confidences = confidences
        self.horizontalFOVDegrees = horizontalFOVDegrees
    }

    @inlinable
    public func depth(row: Int, column: Int) -> Float {
        depths[row * width + column]
    }

    @inlinable
    public func confidence(row: Int, column: Int) -> UInt8 {
        confidences.isEmpty ? 2 : confidences[row * width + column]
    }

    @inlinable
    public static func isValid(_ depth: Float) -> Bool {
        depth.isFinite && depth > 0
    }

    /// Azimuth of a column center in degrees. Negative = left, 0 = straight
    /// ahead, positive = right. Columns are mapped by pixel center, so column 0
    /// maps to just inside -FOV/2 and the middle of the grid maps to ~0°.
    public func azimuthDegrees(forColumn column: Int) -> Float {
        let normalized = (Float(column) + 0.5) / Float(width) - 0.5
        return normalized * horizontalFOVDegrees
    }

    /// Angular width of a single column in degrees.
    public var degreesPerColumn: Float {
        horizontalFOVDegrees / Float(width)
    }

    /// Row range for a vertical band given as fractions of height (top inclusive,
    /// bottom exclusive). Clamped to valid rows; always contains at least one row.
    public func rowBand(topFraction: Float, bottomFraction: Float) -> Range<Int> {
        let top = max(0, min(height - 1, Int(Float(height) * topFraction)))
        let bottom = max(top + 1, min(height, Int(Float(height) * bottomFraction)))
        return top..<bottom
    }
}
