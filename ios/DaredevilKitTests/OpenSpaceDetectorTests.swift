import XCTest
@testable import DaredevilKit

final class OpenSpaceDetectorTests: XCTestCase {

    // MARK: - The failures that motivated the rewrite

    /// The Python prototype scored columns by depth *variance*, so a flat wall
    /// (uniform depth, zero variance) scored as a perfect "hallway". A wall
    /// must produce NO open space.
    func testFlatWallProducesNoOpenSpace() {
        let wall = TestGrids.uniform(depth: 1.2)
        XCTAssertTrue(OpenSpaceDetector.detect(in: wall).isEmpty)
    }

    /// A dead end just beyond arm's reach is still closed.
    func testDeadEndBelowThresholdProducesNoOpenSpace() {
        let deadEnd = TestGrids.uniform(depth: 2.9)  // threshold is 3.0
        XCTAssertTrue(OpenSpaceDetector.detect(in: deadEnd).isEmpty)
    }

    /// Missing LiDAR returns (glass doors, absorbing surfaces) must never be
    /// read as openings.
    func testInvalidDepthIsNotOpen() {
        var depths = [Float](repeating: Float.nan, count: 60 * 40)
        // A few valid near samples so the scene isn't entirely unknown.
        for i in 0..<60 { depths[i] = 1.0 }
        let glass = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertTrue(OpenSpaceDetector.detect(in: glass).isEmpty,
                      "columns dominated by invalid samples must be unknown, not open")
    }

    // MARK: - True positives

    func testUniformDeepSpaceIsOneWideOpening() {
        let field = TestGrids.uniform(depth: 5.0)
        let spaces = OpenSpaceDetector.detect(in: field)
        XCTAssertEqual(spaces.count, 1)
        XCTAssertEqual(spaces[0].azimuthDegrees, 0, accuracy: 1.0)
        XCTAssertEqual(spaces[0].angularWidthDegrees, 72, accuracy: 1.5)
        XCTAssertEqual(spaces[0].depthMeters, 5.0, accuracy: 0.01)
    }

    func testCenteredCorridorIsDetectedAtCenter() {
        let grid = TestGrids.corridor(openColumns: 24..<36)
        let spaces = OpenSpaceDetector.detect(in: grid)
        XCTAssertEqual(spaces.count, 1)
        XCTAssertEqual(spaces[0].azimuthDegrees, 0, accuracy: 1.5)
        XCTAssertEqual(spaces[0].depthMeters, 5.0, accuracy: 0.01)
    }

    func testDoorOnLeftHasNegativeAzimuth() {
        let grid = TestGrids.corridor(openColumns: 5..<17)
        let spaces = OpenSpaceDetector.detect(in: grid)
        XCTAssertEqual(spaces.count, 1)
        XCTAssertLessThan(spaces[0].azimuthDegrees, -15)
    }

    func testDoorOnRightHasPositiveAzimuth() {
        let grid = TestGrids.corridor(openColumns: 43..<55)
        let spaces = OpenSpaceDetector.detect(in: grid)
        XCTAssertEqual(spaces.count, 1)
        XCTAssertGreaterThan(spaces[0].azimuthDegrees, 15)
    }

    func testTwoOpeningsBothReportedDeepestFirst() {
        var depths = [Float](repeating: 1.0, count: 60 * 40)
        for row in 0..<40 {
            for col in 5..<17 { depths[row * 60 + col] = 4.0 }   // left, shallower
            for col in 43..<55 { depths[row * 60 + col] = 6.0 }  // right, deeper
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        let spaces = OpenSpaceDetector.detect(in: grid)
        XCTAssertEqual(spaces.count, 2)
        XCTAssertGreaterThan(spaces[0].depthMeters, spaces[1].depthMeters, "deepest first")
        XCTAssertGreaterThan(spaces[0].azimuthDegrees, 0, "deeper opening is on the right")
        XCTAssertLessThan(spaces[1].azimuthDegrees, 0)
    }

    // MARK: - Constraints

    func testNarrowGapIsRejected() {
        // 3 columns of 60 across 72° = 3.6°, well under the 8° minimum.
        let grid = TestGrids.corridor(openColumns: 30..<33)
        XCTAssertTrue(OpenSpaceDetector.detect(in: grid).isEmpty,
                      "gaps narrower than a walkable width must not be reported")
    }

    func testMaxSpacesRespected() {
        // Four separate wide openings; only maxSpaces (2) reported.
        var depths = [Float](repeating: 1.0, count: 120 * 40)
        for row in 0..<40 {
            for start in [5, 35, 65, 95] {
                for col in start..<(start + 20) { depths[row * 120 + col] = 5.0 }
            }
        }
        let grid = DepthGrid(width: 120, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertEqual(OpenSpaceDetector.detect(in: grid).count, 2)
    }

    func testOpeningOnlyOutsideAnalysisBandIsIgnored() {
        // Deep region only in the top rows (e.g., over-the-head ceiling gap);
        // the walkable band is 35%-75% of height.
        var depths = [Float](repeating: 1.0, count: 60 * 40)
        for row in 0..<8 {
            for col in 0..<60 { depths[row * 60 + col] = 6.0 }
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertTrue(OpenSpaceDetector.detect(in: grid).isEmpty)
    }

    /// Median is robust: a minority of noisy far samples in a near column
    /// must not open it, and vice versa.
    func testMedianRobustToMinorityOutliers() {
        var depths = [Float](repeating: 1.0, count: 60 * 40)
        // The analysis band is rows 14..<30 (16 rows); make 5 of them (~30%)
        // spuriously far in every column.
        for row in 14..<19 {
            for col in 0..<60 { depths[row * 60 + col] = 6.0 }
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertTrue(OpenSpaceDetector.detect(in: grid).isEmpty,
                      "30% far outliers must not defeat the median")
    }

    func testMedianOfOddAndEvenCounts() {
        var odd: [Float] = [3, 1, 2]
        XCTAssertEqual(OpenSpaceDetector.medianOf(&odd), 2)
        var even: [Float] = [4, 1, 3, 2]
        XCTAssertEqual(OpenSpaceDetector.medianOf(&even), 2.5)
        var empty: [Float] = []
        XCTAssertEqual(OpenSpaceDetector.medianOf(&empty), 0)
    }
}
