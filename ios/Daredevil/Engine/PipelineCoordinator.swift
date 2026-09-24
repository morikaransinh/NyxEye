import Foundation
import Combine
import DaredevilKit

/// Glue between sensing and rendering: depth grids in, spatial cues +
/// published UI state out. Owns the full pipeline lifecycle.
@MainActor
final class PipelineCoordinator: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var statusText = "Not scanning"
    @Published private(set) var openSpaces: [TrackedSpace] = []
    @Published private(set) var obstacle: Obstacle?
    @Published private(set) var isDepthSupported = ARDepthSource.isSupported

    @Published var openSpacesEnabled = true { didSet { announcer.resetTransitions() } }
    @Published var obstaclesEnabled = true { didSet { announcer.resetTransitions() } }
    @Published var speechEnabled = true { didSet { announcer.isEnabled = speechEnabled } }
    @Published var headTrackingEnabled = false {
        didSet { headTrackingEnabled ? startHeadTrackingIfRunning() : stopHeadTracking() }
    }
    /// User volume 0...1 as shown in UI; mapped onto the safe range
    /// [0, HearingGuard.masterCeiling] — full slider is still capped.
    @Published var userVolume: Float = 0.6 {
        didSet { audioEngine.setMasterVolume(userVolume * HearingGuard.masterCeiling) }
    }

    private let depthSource = ARDepthSource()
    private let hearingGuard = HearingGuard()
    private lazy var audioEngine = SpatialAudioEngine(hearingGuard: hearingGuard)
    private let headTracker = HeadTracker()
    private let announcer = SpeechAnnouncer()

    private let openSpaceTracker = OpenSpaceTracker()
    private let obstacleSmoother = ObstacleSmoother()
    private var openSpaceConfig = OpenSpaceConfig()
    private var obstacleConfig = ObstacleConfig()

    init() {
        depthSource.onDepthGrid = { [weak self] grid, timestamp in
            self?.processFrame(grid: grid, timestamp: timestamp)
        }
    }

    // MARK: - Lifecycle

    func start() {
        guard !isRunning, isDepthSupported else { return }
        do {
            try audioEngine.start()
        } catch {
            statusText = "Audio failed to start"
            return
        }
        openSpaceTracker.reset()
        obstacleSmoother.reset()
        announcer.resetTransitions()
        depthSource.start()
        if headTrackingEnabled { startHeadTrackingIfRunning() }
        audioEngine.setMasterVolume(userVolume * HearingGuard.masterCeiling)
        isRunning = true
        statusText = "Scanning"
        announcer.announce("Scanning started", urgent: true)
    }

    func stop() {
        guard isRunning else { return }
        depthSource.stop()
        audioEngine.stop()
        stopHeadTracking()
        isRunning = false
        openSpaces = []
        obstacle = nil
        statusText = "Not scanning"
        announcer.announce("Scanning stopped", urgent: true)
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    // MARK: - Frame processing (called on the depth queue)

    nonisolated private func processFrame(grid: DepthGrid, timestamp: TimeInterval) {
        // Detection runs off-main; only publishing hops to the main actor.
        let detectedSpaces = OpenSpaceDetector.detect(in: grid)
        let detectedObstacle = ObstacleDetector.detect(in: grid)

        Task { @MainActor in
            self.hearingGuard.noteFrame(at: ProcessInfo.processInfo.systemUptime)

            let tracked = self.openSpacesEnabled ? self.openSpaceTracker.update(with: detectedSpaces) : []
            let smoothed = self.obstaclesEnabled ? self.obstacleSmoother.update(with: detectedObstacle) : nil

            self.openSpaces = tracked
            self.obstacle = smoothed
            self.pushCues(tracked: tracked, obstacle: smoothed)
            self.updateStatus(tracked: tracked, obstacle: smoothed)
        }
    }

    private func pushCues(tracked: [TrackedSpace], obstacle: Obstacle?) {
        let openCues = tracked.prefix(2).map { space in
            SpatialCue(
                kind: .openSpace,
                azimuthDegrees: space.azimuthDegrees,
                depthMeters: space.depthMeters,
                gain: CueMapping.openSpaceGain(forDepthMeters: space.depthMeters) * space.gain
            )
        }

        var obstacleCue: SpatialCue? = nil
        if let obstacle {
            obstacleCue = SpatialCue(
                kind: .obstacle(pulseRateHz: CueMapping.obstaclePulseRateHz(forDistanceMeters: obstacle.distanceMeters)),
                azimuthDegrees: obstacle.azimuthDegrees,
                depthMeters: obstacle.distanceMeters,
                gain: CueMapping.obstacleGain(forDistanceMeters: obstacle.distanceMeters)
            )
        }

        // When an obstacle is active, duck open-space cues so the warning
        // stands out without getting louder overall.
        let duck: Float = obstacleCue != nil ? 0.5 : 1.0
        let duckedOpenCues = openCues.map {
            SpatialCue(kind: $0.kind, azimuthDegrees: $0.azimuthDegrees,
                       depthMeters: $0.depthMeters, gain: $0.gain * duck)
        }

        audioEngine.setCues(openSpaces: duckedOpenCues, obstacle: obstacleCue)
    }

    private func updateStatus(tracked: [TrackedSpace], obstacle: Obstacle?) {
        if let phrase = StatusPhrases.describe(openSpaces: tracked, obstacle: obstacle) {
            statusText = phrase
            announcer.announce(phrase, urgent: obstacle != nil)
        } else {
            statusText = "Scanning — no openings detected"
        }
    }

    // MARK: - Head tracking

    private func startHeadTrackingIfRunning() {
        guard isRunning, headTracker.isAvailable else { return }
        headTracker.onYawDegrees = { [weak self] yaw in
            self?.audioEngine.setListenerYaw(degrees: yaw)
        }
        headTracker.start()
    }

    private func stopHeadTracking() {
        headTracker.stop()
        audioEngine.setListenerYaw(degrees: 0)
    }
}
