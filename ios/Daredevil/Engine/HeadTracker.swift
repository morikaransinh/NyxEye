import CoreMotion

/// Tracks AirPods head yaw via CMHeadphoneMotionManager so spatial cues stay
/// world-anchored while the head turns. Yaw is reported relative to the
/// orientation when tracking started (assumed to match the phone's heading).
final class HeadTracker {
    private let manager = CMHeadphoneMotionManager()
    private var referenceYaw: Double?

    /// Called on the main queue with head yaw in degrees relative to start.
    var onYawDegrees: ((Float) -> Void)?

    var isAvailable: Bool {
        manager.isDeviceMotionAvailable
    }

    func start() {
        guard manager.isDeviceMotionAvailable else { return }
        referenceYaw = nil
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let yaw = motion.attitude.yaw  // radians
            if self.referenceYaw == nil {
                self.referenceYaw = yaw
            }
            let delta = yaw - (self.referenceYaw ?? yaw)
            self.onYawDegrees?(Float(delta * 180 / .pi))
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        referenceYaw = nil
    }

    /// Re-zero: current head orientation becomes "straight ahead".
    func recenter() {
        referenceYaw = nil
    }
}
