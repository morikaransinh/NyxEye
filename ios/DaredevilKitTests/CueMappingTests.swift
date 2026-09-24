import XCTest
@testable import DaredevilKit

final class CueMappingTests: XCTestCase {

    // MARK: - Frequency mapping (the cue that was dead in the prototype)

    func testFrequencyRangeEndpoints() {
        XCTAssertEqual(CueMapping.frequency(forDepthMeters: 0), 3500, accuracy: 0.1)
        XCTAssertEqual(CueMapping.frequency(forDepthMeters: 5), 700, accuracy: 0.1)
    }

    func testFrequencyIsClampedOutsideRange() {
        XCTAssertEqual(CueMapping.frequency(forDepthMeters: -1), 3500, accuracy: 0.1)
        XCTAssertEqual(CueMapping.frequency(forDepthMeters: 100), 700, accuracy: 0.1)
        XCTAssertEqual(CueMapping.frequency(forDepthMeters: .infinity), 700, accuracy: 0.1)
    }

    func testFrequencyDecreasesWithDistance() {
        var last = CueMapping.frequency(forDepthMeters: 0)
        for d in stride(from: Float(0.5), through: 5, by: 0.5) {
            let f = CueMapping.frequency(forDepthMeters: d)
            XCTAssertLessThan(f, last, "closer = higher pitch, strictly")
            last = f
        }
    }

    func testFrequencyStaysAudibleOnEarbuds() {
        // Regression guard: the prototype shipped 20-40 Hz noise that earbuds
        // cannot reproduce. Every reachable frequency must stay well inside
        // the audible band.
        for d in stride(from: Float(0), through: 10, by: 0.25) {
            let f = CueMapping.frequency(forDepthMeters: d)
            XCTAssertGreaterThanOrEqual(f, 200, "frequency must be reproducible on earbuds")
            XCTAssertLessThanOrEqual(f, 8000, "frequency must stay comfortable")
        }
    }

    func testBandwidthGrowsWithDistanceWithinRange() {
        XCTAssertEqual(CueMapping.bandwidth(forDepthMeters: 0), 150, accuracy: 0.1)
        XCTAssertEqual(CueMapping.bandwidth(forDepthMeters: 5), 300, accuracy: 0.1)
        XCTAssertLessThan(CueMapping.bandwidth(forDepthMeters: 1), CueMapping.bandwidth(forDepthMeters: 4))
    }

    // MARK: - Obstacle polarity (inverted in the prototype)

    /// THE bug: the prototype rendered closer objects QUIETER. Obstacle gain
    /// must be strictly non-increasing with distance, and strictly greater
    /// close-in than far away.
    func testObstacleGainCloserIsLouder() {
        var last: Float = .infinity
        for d in stride(from: Float(0.3), through: 3.5, by: 0.1) {
            let g = CueMapping.obstacleGain(forDistanceMeters: d)
            XCTAssertLessThanOrEqual(g, last, "gain must never rise with distance")
            last = g
        }
        XCTAssertGreaterThan(CueMapping.obstacleGain(forDistanceMeters: 0.5),
                             2 * CueMapping.obstacleGain(forDistanceMeters: 3.0),
                             "arm's length must be clearly louder than the report edge")
    }

    func testObstacleGainBounded() {
        for d in stride(from: Float(-1), through: 10, by: 0.25) {
            let g = CueMapping.obstacleGain(forDistanceMeters: d)
            XCTAssertGreaterThanOrEqual(g, 0)
            XCTAssertLessThanOrEqual(g, 1)
        }
    }

    func testPulseRateFasterWhenCloser() {
        var last: Float = .infinity
        for d in stride(from: Float(0.3), through: 3.5, by: 0.1) {
            let r = CueMapping.obstaclePulseRateHz(forDistanceMeters: d)
            XCTAssertLessThanOrEqual(r, last, "pulse rate must never rise with distance")
            last = r
        }
        XCTAssertEqual(CueMapping.obstaclePulseRateHz(forDistanceMeters: 0.5), 8, accuracy: 0.01)
        XCTAssertEqual(CueMapping.obstaclePulseRateHz(forDistanceMeters: 3.0), 1.5, accuracy: 0.01)
    }

    // MARK: - Open-space gain

    func testOpenSpaceGainFartherIsLouderAndBounded() {
        // Deliberate opposite polarity: depth beckons.
        XCTAssertLessThan(CueMapping.openSpaceGain(forDepthMeters: 3),
                          CueMapping.openSpaceGain(forDepthMeters: 5))
        for d in stride(from: Float(0), through: 10, by: 0.5) {
            let g = CueMapping.openSpaceGain(forDepthMeters: d)
            XCTAssertGreaterThanOrEqual(g, 0.3, "confirmed openings stay audible")
            XCTAssertLessThanOrEqual(g, 1.0)
        }
    }

    // MARK: - Spatial position

    func testSourcePositionStraightAheadIsMinusZ() {
        let p = CueMapping.sourcePosition(azimuthDegrees: 0)
        XCTAssertEqual(p.x, 0, accuracy: 0.001)
        XCTAssertEqual(p.z, -1, accuracy: 0.001)
    }

    func testSourcePositionLeftRightConvention() {
        let left = CueMapping.sourcePosition(azimuthDegrees: -30)
        XCTAssertLessThan(left.x, 0, "negative azimuth = left = negative x")
        let right = CueMapping.sourcePosition(azimuthDegrees: 30)
        XCTAssertGreaterThan(right.x, 0)
    }

    func testSourcePositionIsUnitVector() {
        for az in stride(from: Float(-90), through: 90, by: 15) {
            let p = CueMapping.sourcePosition(azimuthDegrees: az)
            let length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            XCTAssertEqual(length, 1, accuracy: 0.001)
        }
    }
}
