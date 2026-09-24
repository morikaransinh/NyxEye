import XCTest
@testable import DaredevilKit

final class StatusPhrasesTests: XCTestCase {
    private func tracked(azimuth: Float, depth: Float = 4) -> TrackedSpace {
        TrackedSpace(id: 1, azimuthDegrees: azimuth, angularWidthDegrees: 15, depthMeters: depth, gain: 1)
    }

    func testObstacleTakesPriorityOverOpenSpaces() {
        let phrase = StatusPhrases.describe(
            openSpaces: [tracked(azimuth: 0)],
            obstacle: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0)
        )
        XCTAssertEqual(phrase, "Obstacle 1 meter ahead")
    }

    func testObstacleDistanceRoundsToHalfMeters() {
        let phrase = StatusPhrases.describe(
            openSpaces: [],
            obstacle: Obstacle(distanceMeters: 1.68, azimuthDegrees: 0)
        )
        XCTAssertEqual(phrase, "Obstacle 1.5 meters ahead")
    }

    func testObstacleIncludesDirectionWhenOffCenter() {
        let phrase = StatusPhrases.describe(
            openSpaces: [],
            obstacle: Obstacle(distanceMeters: 2.0, azimuthDegrees: -12)
        )
        XCTAssertEqual(phrase, "Obstacle 2 meters ahead, to the left")
    }

    func testSingleOpenSpaceAhead() {
        XCTAssertEqual(StatusPhrases.describe(openSpaces: [tracked(azimuth: 3)], obstacle: nil),
                       "Open space ahead")
    }

    func testTwoOpenSpacesJoined() {
        let phrase = StatusPhrases.describe(
            openSpaces: [tracked(azimuth: -30), tracked(azimuth: 15)],
            obstacle: nil
        )
        XCTAssertEqual(phrase, "Open spaces far left and to the right")
    }

    func testNothingToSayReturnsNil() {
        XCTAssertNil(StatusPhrases.describe(openSpaces: [], obstacle: nil))
    }

    func testDirectionBucketsAreCoarse() {
        XCTAssertNil(StatusPhrases.directionWord(azimuthDegrees: 0))
        XCTAssertNil(StatusPhrases.directionWord(azimuthDegrees: 7.9))
        XCTAssertEqual(StatusPhrases.directionWord(azimuthDegrees: -10), "to the left")
        XCTAssertEqual(StatusPhrases.directionWord(azimuthDegrees: 10), "to the right")
        XCTAssertEqual(StatusPhrases.directionWord(azimuthDegrees: -30), "far left")
        XCTAssertEqual(StatusPhrases.directionWord(azimuthDegrees: 30), "far right")
    }

    func testMeterFormatting() {
        XCTAssertEqual(StatusPhrases.formatMeters(1), "1 meter")
        XCTAssertEqual(StatusPhrases.formatMeters(2), "2 meters")
        XCTAssertEqual(StatusPhrases.formatMeters(1.5), "1.5 meters")
    }
}
