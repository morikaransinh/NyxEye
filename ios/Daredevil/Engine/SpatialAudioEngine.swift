import AVFoundation
import DaredevilKit

/// A cue the engine should currently render.
struct SpatialCue: Equatable {
    enum Kind: Equatable {
        /// Continuous whoosh beckoning toward an opening.
        case openSpace
        /// Pulsed warning; rate rises as distance falls.
        case obstacle(pulseRateHz: Float)
    }
    let kind: Kind
    let azimuthDegrees: Float
    let depthMeters: Float
    /// Requested gain 0...1 before HearingGuard processing.
    let gain: Float
}

/// Renders spatial cues with AVAudioEngine.
///
/// Graph: 3 fixed slots (2 open-space + 1 obstacle), each an
/// AVAudioPlayerNode fed 100 ms segments of band-passed brown noise
/// (NoiseRenderer keeps filter state so segments join seamlessly), connected
/// mono into an AVAudioEnvironmentNode with HRTF rendering, then through an
/// Apple PeakLimiter as the final brickwall before the output.
///
/// Fixed slots deliberately bound the worst case: there is no code path that
/// can add a fourth source. Every gain passes through HearingGuard; the
/// watchdog fades everything out if depth frames stop arriving.
final class SpatialAudioEngine {
    static let slotCount = 3
    static let openSpaceSlots = 0..<2
    static let obstacleSlot = 2

    private let engine = AVAudioEngine()
    private let environment = AVAudioEnvironmentNode()
    private let limiter: AVAudioUnitEffect
    private let players: [AVAudioPlayerNode]
    private let renderers: [NoiseRenderer]

    private let hearingGuard: HearingGuard
    private let controlQueue = DispatchQueue(label: "daredevil.audio.control", qos: .userInitiated)
    private var controlTimer: DispatchSourceTimer?

    /// Segment length for noise scheduling.
    private let segmentDuration: Double = 0.1

    // State shared with the control loop (accessed on controlQueue only).
    private var cues: [SpatialCue?] = Array(repeating: nil, count: slotCount)
    private var lastControlTick: TimeInterval = 0
    private var pulsePhase: Double = 0
    private(set) var isRunning = false

    /// Master volume, already clamped by HearingGuard when set.
    private var masterVolume: Float = 0.4

    init(hearingGuard: HearingGuard) {
        self.hearingGuard = hearingGuard

        var description = AudioComponentDescription()
        description.componentType = kAudioUnitType_Effect
        description.componentSubType = kAudioUnitSubType_PeakLimiter
        description.componentManufacturer = kAudioUnitManufacturer_Apple
        limiter = AVAudioUnitEffect(audioComponentDescription: description)

        players = (0..<Self.slotCount).map { _ in AVAudioPlayerNode() }
        renderers = (0..<Self.slotCount).map { _ in NoiseRenderer() }

        engine.attach(environment)
        engine.attach(limiter)
        players.forEach { engine.attach($0) }

        let monoFormat = renderers[0].format
        let stereoFormat = AVAudioFormat(standardFormatWithSampleRate: monoFormat.sampleRate, channels: 2)

        for player in players {
            engine.connect(player, to: environment, format: monoFormat)
            player.renderingAlgorithm = .HRTFHQ
            player.volume = 0
        }
        engine.connect(environment, to: limiter, format: stereoFormat)
        engine.connect(limiter, to: engine.mainMixerNode, format: stereoFormat)

        environment.listenerPosition = AVAudio3DPoint(x: 0, y: 0, z: 0)
        environment.listenerAngularOrientation = AVAudio3DAngularOrientation(yaw: 0, pitch: 0, roll: 0)
    }

    // MARK: - Lifecycle

    func start() throws {
        guard !isRunning else { return }

        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try audioSession.setActive(true)

        hearingGuard.resetSlew()
        engine.mainMixerNode.outputVolume = masterVolume
        try engine.start()
        players.forEach { $0.play() }

        isRunning = true
        // Prime each slot with two segments; every completed segment
        // schedules its replacement, keeping playback self-clocked against
        // the audio hardware with no timer drift.
        controlQueue.sync {
            for slot in 0..<Self.slotCount {
                scheduleSegment(slot: slot)
                scheduleSegment(slot: slot)
            }
        }
        startControlLoop()
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        controlTimer?.cancel()
        controlTimer = nil

        // Silence-first teardown: volumes to zero before the graph stops.
        players.forEach { $0.volume = 0 }
        players.forEach { $0.stop() }
        engine.stop()
        hearingGuard.resetSlew()
        controlQueue.sync { cues = Array(repeating: nil, count: Self.slotCount) }
    }

    func setMasterVolume(_ requested: Float) {
        let clamped = HearingGuard.clampMasterVolume(requested)
        masterVolume = clamped
        if isRunning {
            engine.mainMixerNode.outputVolume = clamped
        }
    }

    /// Rotate the listener to follow AirPods head yaw (degrees). Cues stay
    /// anchored to the phone's frame; the head turns within it.
    func setListenerYaw(degrees: Float) {
        environment.listenerAngularOrientation = AVAudio3DAngularOrientation(yaw: degrees, pitch: 0, roll: 0)
    }

    /// Update the cues to render. Order: up to 2 open spaces then 1 obstacle.
    func setCues(openSpaces: [SpatialCue], obstacle: SpatialCue?) {
        controlQueue.async { [weak self] in
            guard let self else { return }
            for (offset, slot) in Self.openSpaceSlots.enumerated() {
                self.cues[slot] = offset < openSpaces.count ? openSpaces[offset] : nil
            }
            self.cues[Self.obstacleSlot] = obstacle
        }
    }

    // MARK: - Noise scheduling

    /// Render and schedule one segment for a slot; on completion, schedule
    /// the next. Two segments stay in flight per slot, clocked by the audio
    /// hardware itself. Must be called on controlQueue.
    private func scheduleSegment(slot: Int) {
        guard isRunning else { return }
        let frameCount = AVAudioFrameCount(renderers[0].sampleRate * segmentDuration)
        let depth = cues[slot]?.depthMeters ?? CueMapping.maxDepthMeters
        let buffer = renderers[slot].render(
            frameCount: frameCount,
            centerHz: CueMapping.frequency(forDepthMeters: depth),
            bandwidthHz: CueMapping.bandwidth(forDepthMeters: depth)
        )
        players[slot].scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.controlQueue.async {
                self.scheduleSegment(slot: slot)
            }
        }
    }

    // MARK: - Control loop (gain, position, pulse, watchdog)

    private func startControlLoop() {
        lastControlTick = ProcessInfo.processInfo.systemUptime
        let timer = DispatchSource.makeTimerSource(queue: controlQueue)
        timer.schedule(deadline: .now(), repeating: 0.02)
        timer.setEventHandler { [weak self] in
            self?.controlTick()
        }
        timer.resume()
        controlTimer = timer
    }

    private func controlTick() {
        guard isRunning else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let dt = now - lastControlTick
        lastControlTick = now

        pulsePhase += dt

        // Requested gains per slot (pulse envelope applied to obstacle).
        var requested = [Float](repeating: 0, count: Self.slotCount)
        for slot in 0..<Self.slotCount {
            guard let cue = cues[slot] else { continue }
            switch cue.kind {
            case .openSpace:
                requested[slot] = cue.gain
            case .obstacle(let rate):
                // 50% duty square-ish pulse with 10 ms raised-cosine edges.
                let period = 1.0 / Double(max(rate, 0.5))
                let phase = pulsePhase.truncatingRemainder(dividingBy: period) / period
                requested[slot] = cue.gain * Float(pulseEnvelope(phase: phase, period: period))
            }
        }

        // Hearing protection: ceilings, mix budget, slew, watchdog.
        let watchdog = hearingGuard.watchdogMultiplier(at: now)
        let processed = hearingGuard.processedGains(requested: requested, dt: dt)

        for slot in 0..<Self.slotCount {
            players[slot].volume = processed[slot] * watchdog
            if let cue = cues[slot] {
                let p = CueMapping.sourcePosition(azimuthDegrees: cue.azimuthDegrees)
                players[slot].position = AVAudio3DPoint(x: p.x, y: p.y, z: p.z)
            }
        }
    }

    /// Smooth pulse envelope: on for half the period, with short cosine ramps.
    private func pulseEnvelope(phase: Double, period: Double) -> Double {
        let edge = min(0.01 / period, 0.1)  // 10 ms edges as fraction of period
        switch phase {
        case ..<edge:
            return 0.5 - 0.5 * cos(.pi * phase / edge)
        case ..<(0.5 - edge):
            return 1
        case ..<0.5:
            return 0.5 + 0.5 * cos(.pi * (phase - (0.5 - edge)) / edge)
        default:
            return 0
        }
    }
}
