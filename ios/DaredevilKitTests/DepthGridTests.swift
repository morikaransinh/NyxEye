import XCTest
@testable import DaredevilKit

/// Helpers to build synthetic depth grids for tests.
enum TestGrids {
    /// Grid filled with a constant depth.
    static func uniform(width: Int = 60, height: Int = 40, depth: Float, fov: Float = 72) -> DepthGrid {
        DepthGrid(width: width, height: height,
                  depths: [Float](repeating: depth, count: width * height),
                  horizontalFOVDegrees: fov)
    }

    /// Grid where columns in `openColumns` have `farDepth` and all others
    /// `nearDepth` — a synthetic corridor scene.
    static func corridor(width: Int = 60, height: Int = 40,
                         openColumns: Range<Int>,
                         farDepth: Float = 5.0, nearDepth: Float = 1.0,
                         fov: Float = 72) -> DepthGrid {
        var depths = [Float](repeating: nearDepth, count: width * height)
        for row in 0..<height {
            for col in openColumns {
                depths[row * width + col] = farDepth
            }
        }
        return DepthGrid(width: width, height: height, depths: depths, horizontalFOVDegrees: fov)
    }
}

final class DepthGridTests: XCTestCase {
    func testAzimuthOfCenterColumnIsZero() {
        let grid = TestGrids.uniform(width: 61, depth: 1)
        // Column 30 of 61 is the exact center.
        XCTAssertEqual(grid.azimuthDegrees(forColumn: 30), 0, accuracy: 0.01)
    }

    func testAzimuthSignConvention() {
        let grid = TestGrids.uniform(width: 60, depth: 1)
        XCTAssertLessThan(grid.azimuthDegrees(forColumn: 0), 0, "left of view must be negative azimuth")
        XCTAssertGreaterThan(grid.azimuthDegrees(forColumn: 59), 0, "right of view must be positive azimuth")
    }

    func testAzimuthEdgesApproachHalfFOV() {
        let grid = TestGrids.uniform(width: 200, depth: 1, fov: 72)
        XCTAssertEqual(grid.azimuthDegrees(forColumn: 0), -36, accuracy: 0.5)
        XCTAssertEqual(grid.azimuthDegrees(forColumn: 199), 36, accuracy: 0.5)
    }

    func testAzimuthIsLinearInColumn() {
        let grid = TestGrids.uniform(width: 100, depth: 1)
        let step1 = grid.azimuthDegrees(forColumn: 11) - grid.azimuthDegrees(forColumn: 10)
        let step2 = grid.azimuthDegrees(forColumn: 71) - grid.azimuthDegrees(forColumn: 70)
        XCTAssertEqual(step1, step2, accuracy: 0.0001)
    }

    func testAzimuthScalesWithFOV() {
        let narrow = TestGrids.uniform(width: 100, depth: 1, fov: 60)
        let wide = TestGrids.uniform(width: 100, depth: 1, fov: 120)
        XCTAssertEqual(wide.azimuthDegrees(forColumn: 0), 2 * narrow.azimuthDegrees(forColumn: 0), accuracy: 0.001)
    }

    func testValiditySemantics() {
        XCTAssertTrue(DepthGrid.isValid(1.5))
        XCTAssertFalse(DepthGrid.isValid(0), "zero depth is invalid")
        XCTAssertFalse(DepthGrid.isValid(-1), "negative depth is invalid")
        XCTAssertFalse(DepthGrid.isValid(.nan), "NaN is invalid")
        XCTAssertFalse(DepthGrid.isValid(.infinity), "infinity is invalid")
    }

    func testConfidenceDefaultsToHighWhenAbsent() {
        let grid = TestGrids.uniform(depth: 1)
        XCTAssertEqual(grid.confidence(row: 0, column: 0), 2)
    }

    func testRowBandClampsAndIsNonEmpty() {
        let grid = TestGrids.uniform(width: 10, height: 30, depth: 1)
        let band = grid.rowBand(topFraction: 0.35, bottomFraction: 0.75)
        XCTAssertEqual(band, 10..<22)
        // Degenerate fractions still give at least one row.
        XCTAssertFalse(grid.rowBand(topFraction: 0.99, bottomFraction: 0.99).isEmpty)
        XCTAssertFalse(grid.rowBand(topFraction: 0, bottomFraction: 0).isEmpty)
    }
}
