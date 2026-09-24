import XCTest
@testable import DaredevilKit

final class ObstacleDetectorTests: XCTestCase {

    /// Scene with an obstacle plane at `distance` covering the central
    /// corridor, background at 4 m.
    private func obstacleScene(distance: Float, width: Int = 60, height: Int = 40,
                               confidence: UInt8 = 2) -> DepthGrid {
        var depths = [Float](repeating: 4.0, count: width * height)
        var confidences = [UInt8](repeating: 2, count: width * height)
        for row in 0..<height {
            for col in 20..<40 {  // central third ≈ within ±12° of center
                depths[row * width + col] = distance
                confidences[row * width + col] = confidence
            }
        }
        return DepthGrid(width: width, height: height, depths: depths,
                         confidences: confidences, horizontalFOVDegrees: 72)
    }

    func testNearObstacleIsDetectedWithCorrectDistance() {
        let obstacle = ObstacleDetector.detect(in: obstacleScene(distance: 1.0))
        XCTAssertNotNil(obstacle)
        XCTAssertEqual(obstacle!.distanceMeters, 1.0, accuracy: 0.05)
        XCTAssertEqual(obstacle!.azimuthDegrees, 0, accuracy: 13)
    }

    func testFarSceneProducesNoObstacle() {
        XCTAssertNil(ObstacleDetector.detect(in: obstacleScene(distance: 3.5)),
                     "nothing inside report range -> no obstacle")
    }

    func testClearCorridorProducesNoObstacle() {
        XCTAssertNil(ObstacleDetector.detect(in: TestGrids.uniform(depth: 4.0)))
    }

    /// Safety property: obstacle warnings require HIGH confidence depth.
    /// A low-confidence blob at 1 m must not fire.
    func testLowConfidenceSamplesAreIgnored() {
        XCTAssertNil(ObstacleDetector.detect(in: obstacleScene(distance: 1.0, confidence: 0)))
        XCTAssertNil(ObstacleDetector.detect(in: obstacleScene(distance: 1.0, confidence: 1)))
    }

    /// A single noisy pixel at 0.3 m must not fake an obstacle: the percentile
    /// statistic and cross-column smoothing exist exactly for this.
    func testSinglePixelNoiseDoesNotFireWarning() {
        var depths = [Float](repeating: 4.0, count: 60 * 40)
        depths[20 * 60 + 30] = 0.3
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertNil(ObstacleDetector.detect(in: grid))
    }

    /// Obstacles outside the walking corridor (beyond ±15°) are not warned
    /// about — brushing past a wall on your side is normal.
    func testObstacleOutsideCorridorIsIgnored() {
        var depths = [Float](repeating: 4.0, count: 60 * 40)
        for row in 0..<40 {
            for col in 0..<8 { depths[row * 60 + col] = 0.8 }  // far left edge
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        XCTAssertNil(ObstacleDetector.detect(in: grid))
    }

    func testObstacleLeftOfCenterHasNegativeAzimuth() {
        var depths = [Float](repeating: 4.0, count: 60 * 40)
        for row in 0..<40 {
            for col in 22..<27 { depths[row * 60 + col] = 1.0 }  // slightly left
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        let obstacle = ObstacleDetector.detect(in: grid)
        XCTAssertNotNil(obstacle)
        XCTAssertLessThan(obstacle!.azimuthDegrees, 0)
    }

    /// The detector reports the NEAREST obstacle when several are present.
    func testNearestOfTwoObstaclesWins() {
        var depths = [Float](repeating: 4.0, count: 60 * 40)
        for row in 0..<40 {
            for col in 22..<27 { depths[row * 60 + col] = 2.2 }
            for col in 33..<38 { depths[row * 60 + col] = 0.9 }
        }
        let grid = DepthGrid(width: 60, height: 40, depths: depths, horizontalFOVDegrees: 72)
        let obstacle = ObstacleDetector.detect(in: grid)
        XCTAssertNotNil(obstacle)
        XCTAssertEqual(obstacle!.distanceMeters, 0.9, accuracy: 0.05)
        XCTAssertGreaterThan(obstacle!.azimuthDegrees, 0, "the nearer, right-side obstacle wins")
    }

    func testPercentileHelper() {
        var values: [Float] = [5, 1, 4, 2, 3]
        XCTAssertEqual(ObstacleDetector.percentile(&values, 0), 1)
        XCTAssertEqual(ObstacleDetector.percentile(&values, 1), 5)
        XCTAssertEqual(ObstacleDetector.percentile(&values, 0.5), 3)
        var empty: [Float] = []
        XCTAssertEqual(ObstacleDetector.percentile(&empty, 0.5), 0)
    }
}
