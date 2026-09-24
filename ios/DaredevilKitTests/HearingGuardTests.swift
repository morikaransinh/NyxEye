import XCTest
@testable import DaredevilKit

/// Safety-critical tests. The prototype had no output limiting at all: up to
/// ten unclamped looping sources and unvalidated user volume. Every property
/// here is a hard requirement.
final class HearingGuardTests: XCTestCase {

    // MARK: - Master volume

    func testMasterVolumeIsClampedToCeiling() {
        XCTAssertEqual(HearingGuard.clampMasterVolume(5.0), HearingGuard.masterCeiling)
        XCTAssertEqual(HearingGuard.clampMasterVolume(1.0), HearingGuard.masterCeiling)
        XCTAssertEqual(HearingGuard.clampMasterVolume(-1), 0)
        XCTAssertEqual(HearingGuard.clampMasterVolume(0.3), 0.3)
    }

    func testMasterCeilingIsWellBelowFullScale() {
        XCTAssertLessThanOrEqual(HearingGuard.masterCeiling, 0.7,
                                 "master ceiling must leave substantial headroom")
    }

    // MARK: - Per-source ceiling and mix budget

    func testPerSourceCeilingApplied() {
        let guard_ = HearingGuard()
        let out = guard_.processedGains(requested: [10, 0, 0], dt: 100)
        XCTAssertLessThanOrEqual(out[0], HearingGuard.perSourceCeiling)
    }

    func testNegativeGainsClampedToZero() {
        let guard_ = HearingGuard()
        let out = guard_.processedGains(requested: [-5, 0.2, 0], dt: 100)
        XCTAssertEqual(out[0], 0)
    }

    /// More cues must never mean more total loudness: the mix budget scales
    /// everything down proportionally.
    func testMixBudgetCapsTotalGain() {
        let guard_ = HearingGuard()
        let out = guard_.processedGains(requested: [0.8, 0.8, 0.8], dt: 100)
        XCTAssertLessThanOrEqual(out.reduce(0, +), HearingGuard.maxTotalGain + 0.001)
        // Proportions preserved.
        XCTAssertEqual(out[0], out[1], accuracy: 0.001)
        XCTAssertEqual(out[1], out[2], accuracy: 0.001)
    }

    func testMixBudgetHoldsForAnyRequest() {
        let guard_ = HearingGuard()
        for _ in 0..<100 {
            let request = (0..<3).map { _ in Float.random(in: 0...100) }
            let out = guard_.processedGains(requested: request, dt: 100)
            XCTAssertLessThanOrEqual(out.reduce(0, +), HearingGuard.maxTotalGain + 0.001)
            for g in out {
                XCTAssertGreaterThanOrEqual(g, 0)
                XCTAssertLessThanOrEqual(g, HearingGuard.perSourceCeiling + 0.001)
            }
        }
    }

    // MARK: - Slew limiting

    /// Gains rise gradually — no sudden loud onset. From silence, one 20 ms
    /// tick can raise gain by at most maxRisePerSecond * 0.02.
    func testRiseIsRateLimited() {
        let guard_ = HearingGuard()
        let out = guard_.processedGains(requested: [0.8, 0, 0], dt: 0.02)
        XCTAssertLessThanOrEqual(out[0], HearingGuard.maxRisePerSecond * 0.02 + 0.001,
                                 "first tick from silence must be quiet")
    }

    func testRiseConvergesToTarget() {
        let guard_ = HearingGuard()
        var out: [Float] = []
        for _ in 0..<100 {
            out = guard_.processedGains(requested: [0.5, 0, 0], dt: 0.02)
        }
        XCTAssertEqual(out[0], 0.5, accuracy: 0.001)
    }

    /// Drops are INSTANT — quieting must never be delayed.
    func testDropIsInstantaneous() {
        let guard_ = HearingGuard()
        for _ in 0..<100 { _ = guard_.processedGains(requested: [0.5, 0, 0], dt: 0.02) }
        let out = guard_.processedGains(requested: [0.05, 0, 0], dt: 0.001)
        XCTAssertEqual(out[0], 0.05, accuracy: 0.001, "gain reduction must not be slew-limited")
    }

    /// A brand-new source layout ramps in from zero rather than snapping on.
    func testSourceCountChangeRampsFromSilence() {
        let guard_ = HearingGuard()
        for _ in 0..<100 { _ = guard_.processedGains(requested: [0.5], dt: 0.02) }
        let out = guard_.processedGains(requested: [0.5, 0.5, 0.5], dt: 0.02)
        for g in out {
            XCTAssertLessThanOrEqual(g, HearingGuard.maxRisePerSecond * 0.02 + 0.001)
        }
    }

    func testResetSlewForcesRampIn() {
        let guard_ = HearingGuard()
        for _ in 0..<100 { _ = guard_.processedGains(requested: [0.5, 0, 0], dt: 0.02) }
        guard_.resetSlew()
        let out = guard_.processedGains(requested: [0.5, 0, 0], dt: 0.02)
        XCTAssertLessThanOrEqual(out[0], HearingGuard.maxRisePerSecond * 0.02 + 0.001)
    }

    // MARK: - Watchdog: silence is the fail-safe state

    func testWatchdogIsSilentBeforeAnyFrame() {
        let guard_ = HearingGuard()
        XCTAssertEqual(guard_.watchdogMultiplier(at: 100), 0,
                       "no depth data ever -> no sound")
    }

    func testWatchdogFullWhileFramesFresh() {
        let guard_ = HearingGuard()
        guard_.noteFrame(at: 100)
        XCTAssertEqual(guard_.watchdogMultiplier(at: 100.1), 1)
        XCTAssertEqual(guard_.watchdogMultiplier(at: 100.59), 1)
    }

    func testWatchdogFadesOutAfterStall() {
        let guard_ = HearingGuard()  // timeout 0.6 s, fade 0.3 s
        guard_.noteFrame(at: 100)
        let midFade = guard_.watchdogMultiplier(at: 100.75)  // 0.15 into fade
        XCTAssertEqual(midFade, 0.5, accuracy: 0.01)
        XCTAssertEqual(guard_.watchdogMultiplier(at: 101), 0, "fully silent after fade")
        XCTAssertEqual(guard_.watchdogMultiplier(at: 200), 0, "stays silent")
    }

    func testWatchdogRecoversWhenFramesResume() {
        let guard_ = HearingGuard()
        guard_.noteFrame(at: 100)
        XCTAssertEqual(guard_.watchdogMultiplier(at: 105), 0)
        guard_.noteFrame(at: 105.1)
        XCTAssertEqual(guard_.watchdogMultiplier(at: 105.2), 1)
    }

    func testResetSlewClearsFrameHistory() {
        let guard_ = HearingGuard()
        guard_.noteFrame(at: 100)
        guard_.resetSlew()
        XCTAssertEqual(guard_.watchdogMultiplier(at: 100.1), 0,
                       "after reset the watchdog must be silent until a new frame arrives")
    }

    func testNaNGainIsSilenced() {
        let guard_ = HearingGuard()
        let out = guard_.processedGains(requested: [.nan, 0.2, .nan], dt: 100)
        XCTAssertEqual(out[0], 0, "NaN must never reach the audio graph")
        XCTAssertEqual(out[2], 0)
        XCTAssertFalse(out.contains { $0.isNaN })
    }

    // MARK: - End-to-end worst case

    /// Absolute output bound: even with adversarial inputs (huge gains, huge
    /// user volume, many ticks), rendered gain x master volume stays under
    /// maxTotalGain x masterCeiling.
    func testWorstCaseOutputBound() {
        let guard_ = HearingGuard()
        guard_.noteFrame(at: 0)
        var maxObservedTotal: Float = 0
        for tick in 0..<500 {
            let t = Double(tick) * 0.02
            guard_.noteFrame(at: t)
            let gains = guard_.processedGains(requested: [Float.infinity, 1000, 999], dt: 0.02)
            let watchdog = guard_.watchdogMultiplier(at: t)
            let master = HearingGuard.clampMasterVolume(.infinity)
            maxObservedTotal = max(maxObservedTotal, gains.reduce(0, +) * watchdog * master)
        }
        XCTAssertLessThanOrEqual(maxObservedTotal,
                                 HearingGuard.maxTotalGain * HearingGuard.masterCeiling + 0.001)
    }
}
