import Foundation

/// Central hearing-protection policy. Every gain that reaches the audio
/// engine passes through this — there is no code path that can set a raw
/// gain on a node directly.
///
/// Layers of protection (all enforced here, plus a peak limiter in the
/// audio graph as a final brickwall):
/// 1. Per-source ceiling: no single cue exceeds `perSourceCeiling`.
/// 2. Mix budget: if the sum of requested gains exceeds `maxTotalGain`,
///    all sources are scaled down proportionally. The mix can never grow
///    louder just because more cues appeared.
/// 3. Master ceiling: user volume is clamped to `masterCeiling` — the UI
///    physically cannot request more.
/// 4. Slew limiting: gains may only *rise* at `maxRisePerSecond`; drops are
///    instantaneous. No sudden loud onsets, immediate quieting.
/// 5. Watchdog: if depth frames stop arriving (camera stall, crash, app
///    backgrounded), output fades to silence within a fraction of a second.
///    Silence is the fail-safe state.
public final class HearingGuard {
    /// Hard upper bound on user master volume (linear gain).
    public static let masterCeiling: Float = 0.6
    /// Hard upper bound on any single source's gain.
    public static let perSourceCeiling: Float = 0.8
    /// Hard upper bound on the sum of all source gains.
    public static let maxTotalGain: Float = 1.2
    /// Maximum linear-gain rise per second.
    public static let maxRisePerSecond: Float = 2.0

    /// Depth-frame watchdog: newest frame older than this means the sensing
    /// pipeline is stalled and output must fade out.
    public let watchdogTimeout: TimeInterval
    /// Time over which the watchdog fades output to zero after timing out.
    public let watchdogFadeDuration: TimeInterval

    private var lastFrameTime: TimeInterval?
    private var currentGains: [Float] = []

    public init(watchdogTimeout: TimeInterval = 0.6, watchdogFadeDuration: TimeInterval = 0.3) {
        self.watchdogTimeout = watchdogTimeout
        self.watchdogFadeDuration = watchdogFadeDuration
    }

    /// Clamp a user-requested master volume to the safe range.
    public static func clampMasterVolume(_ requested: Float) -> Float {
        min(max(requested, 0), masterCeiling)
    }

    /// Record that a depth frame arrived at time `t` (seconds, monotonic).
    public func noteFrame(at t: TimeInterval) {
        lastFrameTime = t
    }

    /// Watchdog output multiplier at time `t`: 1 while frames are fresh,
    /// fading linearly to 0 once they are stale. No frames ever -> 0.
    public func watchdogMultiplier(at t: TimeInterval) -> Float {
        guard let last = lastFrameTime else { return 0 }
        let age = t - last
        if age <= watchdogTimeout { return 1 }
        let fade = 1 - (age - watchdogTimeout) / watchdogFadeDuration
        return Float(min(max(fade, 0), 1))
    }

    /// Apply all protection layers to the requested per-source gains.
    /// `dt` is seconds since the previous call (for slew limiting).
    /// Returns the gains that may actually be applied to the audio graph.
    public func processedGains(requested: [Float], dt: TimeInterval) -> [Float] {
        // Layer 1: per-source ceiling (and floor at 0). NaN is silenced.
        var gains = requested.map { min(max($0.isNaN ? 0 : $0, 0), Self.perSourceCeiling) }

        // Layer 2: mix budget.
        let total = gains.reduce(0, +)
        if total > Self.maxTotalGain {
            let scale = Self.maxTotalGain / total
            for i in gains.indices { gains[i] *= scale }
        }

        // Layer 4: slew — rises are rate-limited, drops are instant.
        if currentGains.count != gains.count {
            // Source layout changed; treat unknown previous gains as 0 so new
            // sources ramp in rather than snapping on.
            currentGains = Array(repeating: 0, count: gains.count)
        }
        let maxRise = Self.maxRisePerSecond * Float(max(dt, 0))
        for i in gains.indices {
            let previous = currentGains[i]
            if gains[i] > previous {
                gains[i] = min(gains[i], previous + maxRise)
            }
        }
        currentGains = gains
        return gains
    }

    /// Reset slew state (e.g., after the engine stops), so the next start
    /// ramps in from silence.
    public func resetSlew() {
        currentGains.removeAll()
        lastFrameTime = nil
    }
}
