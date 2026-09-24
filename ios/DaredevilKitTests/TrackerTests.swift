import XCTest
@testable import DaredevilKit

final class OpenSpaceTrackerTests: XCTestCase {
    private func space(azimuth: Float, width: Float = 15, depth: Float = 4) -> OpenSpace {
        OpenSpace(azimuthDegrees: azimuth, angularWidthDegrees: width, depthMeters: depth)
    }

    func testNewSpaceRequiresConfirmationFrames() {
        let tracker = OpenSpaceTracker()  // confirmFrames = 3
        XCTAssertTrue(tracker.update(with: [space(azimuth: 20)]).isEmpty, "frame 1: unconfirmed")
        XCTAssertTrue(tracker.update(with: [space(azimuth: 20)]).isEmpty, "frame 2: unconfirmed")
        let confirmed = tracker.update(with: [space(azimuth: 20)])
        XCTAssertEqual(confirmed.count, 1, "frame 3: confirmed")
    }

    func testSingleFrameFlickerNeverEmitted() {
        let tracker = OpenSpaceTracker()
        _ = tracker.update(with: [space(azimuth: 20)])
        let after = tracker.update(with: [])
        XCTAssertTrue(after.isEmpty, "one-frame detection must never produce a cue")
    }

    /// Regression guard for the prototype bug where a NEW space "faded in
    /// from center": its azimuth was blended from 0°, reporting an opening at
    /// +30° as +9°. A confirmed track must report the detected azimuth.
    func testNewSpaceAppearsAtItsRealAzimuthNotBlendedFromCenter() {
        let tracker = OpenSpaceTracker()
        for _ in 0..<5 {
            _ = tracker.update(with: [space(azimuth: 30)])
        }
        let tracked = tracker.update(with: [space(azimuth: 30)])
        XCTAssertEqual(tracked[0].azimuthDegrees, 30, accuracy: 0.5)
    }

    /// Regression guard for the worse prototype bug: a DISAPPEARED space's
    /// azimuth decayed toward 0°, steering the user straight ahead toward an
    /// opening that no longer existed. Fading tracks must stay put.
    func testDisappearedSpaceFadesInPlaceWithoutMoving() {
        let tracker = OpenSpaceTracker()
        for _ in 0..<6 {
            _ = tracker.update(with: [space(azimuth: 30)])
        }
        // Space disappears; hold then fade.
        var sawFading = false
        for _ in 0..<20 {
            let tracked = tracker.update(with: [])
            for t in tracked {
                sawFading = true
                XCTAssertEqual(t.azimuthDegrees, 30, accuracy: 0.5,
                               "fading track must not move (was sliding to 0° in the prototype)")
            }
        }
        XCTAssertTrue(sawFading, "track should fade over several frames, not vanish instantly")
        XCTAssertTrue(tracker.update(with: []).isEmpty, "eventually fully faded")
    }

    func testGainRampsInMonotonically() {
        let tracker = OpenSpaceTracker()
        var lastGain: Float = 0
        for frame in 0..<10 {
            let tracked = tracker.update(with: [space(azimuth: 10)])
            if let t = tracked.first {
                XCTAssertGreaterThanOrEqual(t.gain, lastGain, "gain must not jump down while visible (frame \(frame))")
                lastGain = t.gain
            }
        }
        XCTAssertEqual(lastGain, 1.0, accuracy: 0.001, "gain reaches full after ramp")
    }

    func testGainFadesMonotonicallyAfterLoss() {
        let tracker = OpenSpaceTracker()
        for _ in 0..<8 { _ = tracker.update(with: [space(azimuth: 10)]) }
        var lastGain: Float = 1.1
        while true {
            let tracked = tracker.update(with: [])
            guard let t = tracked.first else { break }
            XCTAssertLessThanOrEqual(t.gain, lastGain, "fade must be monotonic")
            lastGain = t.gain
        }
    }

    func testAzimuthSmoothingFollowsDetections() {
        let tracker = OpenSpaceTracker()
        for _ in 0..<5 { _ = tracker.update(with: [space(azimuth: 10)]) }
        // Detection moves to 20°; smoothed azimuth approaches it over frames.
        var azimuth: Float = 10
        for _ in 0..<15 {
            if let t = tracker.update(with: [space(azimuth: 20)]).first {
                azimuth = t.azimuthDegrees
            }
        }
        XCTAssertEqual(azimuth, 20, accuracy: 1.0)
    }

    func testTwoSpacesKeepDistinctIdentities() {
        let tracker = OpenSpaceTracker()
        var tracked: [TrackedSpace] = []
        for _ in 0..<6 {
            tracked = tracker.update(with: [space(azimuth: -25, depth: 3.5), space(azimuth: 25, depth: 5)])
        }
        XCTAssertEqual(tracked.count, 2)
        XCTAssertEqual(tracked[0].depthMeters, 5, accuracy: 0.1, "deepest first")
        let ids = Set(tracked.map { $0.id })
        XCTAssertEqual(ids.count, 2)
        // Next frame preserves IDs.
        let again = tracker.update(with: [space(azimuth: -25, depth: 3.5), space(azimuth: 25, depth: 5)])
        XCTAssertEqual(Set(again.map { $0.id }), ids, "matched tracks keep their identity")
    }

    func testResetClearsAllTracks() {
        let tracker = OpenSpaceTracker()
        for _ in 0..<6 { _ = tracker.update(with: [space(azimuth: 10)]) }
        tracker.reset()
        XCTAssertTrue(tracker.update(with: []).isEmpty)
    }
}

final class ObstacleSmootherTests: XCTestCase {
    func testWarningRequiresConsecutiveFrames() {
        let smoother = ObstacleSmoother()  // confirmFrames = 2
        XCTAssertNil(smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0)),
                     "single frame must not warn")
        XCTAssertNotNil(smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0)))
    }

    func testNoWarningBeyondWarnDistance() {
        let smoother = ObstacleSmoother()  // warn at 2.0 m
        for _ in 0..<5 {
            XCTAssertNil(smoother.update(with: Obstacle(distanceMeters: 2.5, azimuthDegrees: 0)))
        }
    }

    /// Hysteresis: once warning, an obstacle hovering just past the threshold
    /// must not flutter the warning off. It releases only beyond 2.0 * 1.15.
    func testReleaseHysteresis() {
        let smoother = ObstacleSmoother()
        _ = smoother.update(with: Obstacle(distanceMeters: 1.8, azimuthDegrees: 0))
        XCTAssertNotNil(smoother.update(with: Obstacle(distanceMeters: 1.8, azimuthDegrees: 0)))
        // Drifts to 2.1 m — inside the release band, stays active.
        for _ in 0..<10 {
            XCTAssertNotNil(smoother.update(with: Obstacle(distanceMeters: 2.1, azimuthDegrees: 0)),
                            "2.1 m is within release hysteresis; warning must persist")
        }
        // Far beyond release: deactivates (after smoothing catches up).
        var released = false
        for _ in 0..<20 {
            if smoother.update(with: Obstacle(distanceMeters: 2.9, azimuthDegrees: 0)) == nil {
                released = true
                break
            }
        }
        XCTAssertTrue(released)
    }

    func testNilDetectionClearsWarning() {
        let smoother = ObstacleSmoother()
        _ = smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0))
        _ = smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0))
        XCTAssertNil(smoother.update(with: nil), "obstacle gone -> warning gone")
    }

    func testDistanceIsSmoothedNotJumpy() {
        let smoother = ObstacleSmoother()
        _ = smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0))
        _ = smoother.update(with: Obstacle(distanceMeters: 1.0, azimuthDegrees: 0))
        // A one-frame spike to 1.8 m moves the estimate only partway.
        let smoothed = smoother.update(with: Obstacle(distanceMeters: 1.8, azimuthDegrees: 0))
        XCTAssertNotNil(smoothed)
        XCTAssertLessThan(smoothed!.distanceMeters, 1.6)
        XCTAssertGreaterThan(smoothed!.distanceMeters, 1.0)
    }
}
