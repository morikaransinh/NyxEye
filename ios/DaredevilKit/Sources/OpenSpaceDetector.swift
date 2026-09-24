import Foundation

/// A detected walkable opening (hallway, doorway, gap between obstacles).
public struct OpenSpace: Equatable {
    /// Center azimuth in degrees (negative = left).
    public let azimuthDegrees: Float
    /// Angular width in degrees.
    public let angularWidthDegrees: Float
    /// Median metric depth of the opening in meters.
    public let depthMeters: Float

    public init(azimuthDegrees: Float, angularWidthDegrees: Float, depthMeters: Float) {
        self.azimuthDegrees = azimuthDegrees
        self.angularWidthDegrees = angularWidthDegrees
        self.depthMeters = depthMeters
    }
}

public struct OpenSpaceConfig {
    /// Vertical band of the grid to analyze, as fractions of height.
    /// Roughly chest-to-waist height when the phone is held upright — the
    /// region a person would walk through.
    public var bandTopFraction: Float = 0.35
    public var bandBottomFraction: Float = 0.75
    /// A column counts as open only if its median depth is at least this far.
    /// Absolute and metric: facing a wall produces no open space at all.
    public var openDistanceMeters: Float = 3.0
    /// Minimum fraction of valid samples a column needs before it can be
    /// classified at all. Columns below this are unknown, and unknown is NOT
    /// open (missing LiDAR returns can be glass or absorbing surfaces).
    public var minValidFraction: Float = 0.3
    /// Minimum angular width for a run of open columns to count as an opening.
    /// Narrower gaps are not walkable.
    public var minAngularWidthDegrees: Float = 8.0
    /// Maximum number of openings to report, deepest first.
    public var maxSpaces: Int = 2

    public init() {}
}

/// Finds walkable openings in a metric depth grid.
///
/// Unlike the earlier Python prototype (which scored per-frame *relative*
/// depth by column variance and therefore could classify a flat wall as a
/// hallway), this detector requires columns to be *absolutely* far away in
/// meters. No openings are reported when facing a wall or dead end.
public enum OpenSpaceDetector {

    /// Per-column classification used internally and by tests.
    public enum ColumnClass: Equatable {
        case open(medianDepth: Float)
        case closed(medianDepth: Float)
        case unknown
    }

    /// Classify every column of the grid's analysis band.
    public static func classifyColumns(in grid: DepthGrid, config: OpenSpaceConfig = OpenSpaceConfig()) -> [ColumnClass] {
        let band = grid.rowBand(topFraction: config.bandTopFraction, bottomFraction: config.bandBottomFraction)
        var result: [ColumnClass] = []
        result.reserveCapacity(grid.width)
        var samples: [Float] = []
        samples.reserveCapacity(band.count)

        for column in 0..<grid.width {
            samples.removeAll(keepingCapacity: true)
            for row in band {
                let d = grid.depth(row: row, column: column)
                if DepthGrid.isValid(d) {
                    samples.append(d)
                }
            }
            let validFraction = Float(samples.count) / Float(band.count)
            if validFraction < config.minValidFraction {
                result.append(.unknown)
                continue
            }
            let median = medianOf(&samples)
            if median >= config.openDistanceMeters {
                result.append(.open(medianDepth: median))
            } else {
                result.append(.closed(medianDepth: median))
            }
        }
        return result
    }

    /// Detect walkable openings: contiguous runs of open columns at least
    /// `minAngularWidthDegrees` wide, reported deepest-first.
    public static func detect(in grid: DepthGrid, config: OpenSpaceConfig = OpenSpaceConfig()) -> [OpenSpace] {
        let classes = classifyColumns(in: grid, config: config)
        let degreesPerColumn = grid.degreesPerColumn
        var spaces: [OpenSpace] = []

        var runStart: Int? = nil
        var runDepths: [Float] = []

        func closeRun(endExclusive: Int) {
            guard let start = runStart else { return }
            runStart = nil
            let widthDegrees = Float(endExclusive - start) * degreesPerColumn
            guard widthDegrees >= config.minAngularWidthDegrees else {
                runDepths.removeAll(keepingCapacity: true)
                return
            }
            let centerColumn = Float(start + endExclusive - 1) / 2.0
            let azimuth = (centerColumn + 0.5) / Float(grid.width) - 0.5
            spaces.append(OpenSpace(
                azimuthDegrees: azimuth * grid.horizontalFOVDegrees,
                angularWidthDegrees: widthDegrees,
                depthMeters: medianOf(&runDepths)
            ))
            runDepths.removeAll(keepingCapacity: true)
        }

        for (column, cls) in classes.enumerated() {
            if case .open(let depth) = cls {
                if runStart == nil { runStart = column }
                runDepths.append(depth)
            } else {
                closeRun(endExclusive: column)
            }
        }
        closeRun(endExclusive: classes.count)

        spaces.sort { $0.depthMeters > $1.depthMeters }
        if spaces.count > config.maxSpaces {
            spaces.removeSubrange(config.maxSpaces...)
        }
        return spaces
    }

    /// Median of a mutable buffer (partially sorts in place). Empty input -> 0.
    static func medianOf(_ values: inout [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let mid = values.count / 2
        values.sort()
        if values.count % 2 == 1 {
            return values[mid]
        }
        return (values[mid - 1] + values[mid]) / 2
    }
}
