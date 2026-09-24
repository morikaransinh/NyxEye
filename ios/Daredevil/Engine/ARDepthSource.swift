import ARKit
import DaredevilKit

/// Captures LiDAR depth from ARKit and delivers upright, metric `DepthGrid`s.
///
/// ARKit's depth map is Float32 meters in the sensor's landscape orientation;
/// this class rotates it 90° clockwise into the portrait frame the app runs
/// in (EXIF orientation 6, same as portrait photos) and computes the true
/// horizontal FOV from the camera intrinsics — no hardcoded FOV constants.
final class ARDepthSource: NSObject, ARSessionDelegate {
    /// Called on `queue` with each downsampled depth grid and its timestamp.
    var onDepthGrid: ((DepthGrid, TimeInterval) -> Void)?

    /// Minimum interval between delivered grids (seconds). Detection at 10 Hz
    /// is plenty; the AR session itself runs at 60.
    var deliveryInterval: TimeInterval = 0.1

    private let session = ARSession()
    private let queue = DispatchQueue(label: "daredevil.depth", qos: .userInitiated)
    private var lastDeliveryTime: TimeInterval = 0

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    func start() {
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics = .smoothedSceneDepth
        } else {
            configuration.frameSemantics = .sceneDepth
        }
        configuration.planeDetection = []
        session.run(configuration, options: [.resetTracking])
        lastDeliveryTime = 0
    }

    func stop() {
        session.pause()
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard frame.timestamp - lastDeliveryTime >= deliveryInterval else { return }
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        lastDeliveryTime = frame.timestamp

        guard let grid = Self.makeGrid(from: sceneDepth, camera: frame.camera) else { return }
        onDepthGrid?(grid, frame.timestamp)
    }

    // MARK: - Conversion

    static func makeGrid(from sceneDepth: ARDepthData, camera: ARCamera) -> DepthGrid? {
        let depthBuffer = sceneDepth.depthMap
        guard CVPixelBufferGetPixelFormatType(depthBuffer) == kCVPixelFormatType_DepthFloat32 else { return nil }

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        guard let depthBase = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }

        // Landscape sensor dimensions.
        let srcWidth = CVPixelBufferGetWidth(depthBuffer)    // e.g. 256
        let srcHeight = CVPixelBufferGetHeight(depthBuffer)  // e.g. 192
        let depthStride = CVPixelBufferGetBytesPerRow(depthBuffer) / MemoryLayout<Float>.size
        let depthPtr = depthBase.assumingMemoryBound(to: Float.self)

        // Optional confidence map (UInt8, same dimensions).
        var confidencePtr: UnsafeMutablePointer<UInt8>? = nil
        var confidenceStride = 0
        let confidenceBuffer = sceneDepth.confidenceMap
        if let cb = confidenceBuffer {
            CVPixelBufferLockBaseAddress(cb, .readOnly)
            confidenceStride = CVPixelBufferGetBytesPerRow(cb)
            confidencePtr = CVPixelBufferGetBaseAddress(cb)?.assumingMemoryBound(to: UInt8.self)
        }
        defer {
            if let cb = confidenceBuffer { CVPixelBufferUnlockBaseAddress(cb, .readOnly) }
        }

        // Rotate 90° CW: portrait grid is srcHeight wide x srcWidth tall.
        // dst[r][c] = src[srcHeight - 1 - c][r]
        let dstWidth = srcHeight
        let dstHeight = srcWidth
        var depths = [Float](repeating: 0, count: dstWidth * dstHeight)
        var confidences = [UInt8](repeating: 2, count: confidencePtr != nil ? dstWidth * dstHeight : 0)

        for r in 0..<dstHeight {
            for c in 0..<dstWidth {
                let srcRow = srcHeight - 1 - c
                let srcCol = r
                depths[r * dstWidth + c] = depthPtr[srcRow * depthStride + srcCol]
                if let cp = confidencePtr {
                    confidences[r * dstWidth + c] = cp[srcRow * confidenceStride + srcCol]
                }
            }
        }

        // Portrait horizontal FOV = landscape vertical FOV, from intrinsics:
        // fov = 2 * atan(imageHeight / (2 * fy)).
        let fy = camera.intrinsics[1][1]
        let imageHeight = Float(camera.imageResolution.height)
        guard fy > 0 else { return nil }
        let hfovDegrees = 2 * atan(imageHeight / (2 * fy)) * 180 / .pi

        return DepthGrid(width: dstWidth,
                         height: dstHeight,
                         depths: depths,
                         confidences: confidences,
                         horizontalFOVDegrees: hfovDegrees)
    }
}
