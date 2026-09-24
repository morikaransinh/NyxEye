import AVFoundation

/// Renders band-passed brown noise into PCM buffers with persistent filter
/// state, so consecutive segments join seamlessly and the band-pass center
/// can glide with depth between segments.
///
/// Brown noise carrier (leaky integrator over white noise) was chosen with
/// the codesigners: non-fatiguing and ocean-like. Unlike the prototype —
/// which band-limited it to 20–40 Hz, below what earbuds can reproduce —
/// the band-pass here is centered from 700 to 3500 Hz by CueMapping, so the
/// frequency-as-distance cue is actually audible.
final class NoiseRenderer {
    let sampleRate: Double
    let format: AVAudioFormat

    // Brown noise state (leaky integrator).
    private var brown: Float = 0
    private var rng = SystemRandomNumberGenerator()

    // RBJ band-pass biquad state.
    private var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0

    // Gentle post-filter gain smoothing to avoid segment-boundary steps.
    private var currentCenter: Float = 1000
    private var currentBandwidth: Float = 200

    init(sampleRate: Double = 48_000) {
        self.sampleRate = sampleRate
        self.format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    }

    /// Render `frameCount` samples of band-passed brown noise. The filter
    /// center/bandwidth glide linearly across the segment from their previous
    /// values to the requested ones.
    func render(frameCount: AVAudioFrameCount, centerHz: Float, bandwidthHz: Float) -> AVAudioPCMBuffer {
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount
        let out = buffer.floatChannelData![0]

        let n = Int(frameCount)
        let startCenter = currentCenter
        let startBandwidth = currentBandwidth

        for i in 0..<n {
            let t = Float(i) / Float(n)
            let center = startCenter + (centerHz - startCenter) * t
            let bandwidth = startBandwidth + (bandwidthHz - startBandwidth) * t

            // Brown noise: leaky integration of white noise.
            let white = Float.random(in: -1...1, using: &rng)
            brown = (brown + 0.02 * white) / 1.02
            let sample = brown * 20  // integrator output is tiny; bring to ~unit range

            // RBJ constant-skirt band-pass, recomputed per sample for glide.
            // Cheap enough at 48 kHz mono x 3 sources.
            let omega = 2 * Float.pi * center / Float(sampleRate)
            let q = max(center / max(bandwidth, 1), 0.5)
            let alpha = sin(omega) / (2 * q)
            let b0 = alpha
            let b2 = -alpha
            let a0 = 1 + alpha
            let a1 = -2 * cos(omega)
            let a2 = 1 - alpha

            let y = (b0 * sample + b2 * x2 - a1 * y1 - a2 * y2) / a0
            x2 = x1
            x1 = sample
            y2 = y1
            y1 = y

            out[i] = max(-1, min(1, y))
        }

        currentCenter = centerHz
        currentBandwidth = bandwidthHz
        return buffer
    }
}
