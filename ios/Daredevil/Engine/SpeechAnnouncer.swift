import AVFoundation
import DaredevilKit

/// Speaks status *transitions* only, throttled, so speech never piles up over
/// the continuous spatial cues. Speech carries semantics; the noise cues
/// carry direction and distance (pattern borrowed from CurbToCar).
final class SpeechAnnouncer {
    private let synthesizer = AVSpeechSynthesizer()
    private var lastPhrase: String?
    private var lastSpokenAt: TimeInterval = 0
    /// Minimum seconds between repeated announcements of a *different* phrase.
    private let cooldown: TimeInterval = 3.0

    var isEnabled = true

    /// Speak `phrase` if it differs from the last spoken phrase and the
    /// cooldown has elapsed. Urgent phrases bypass the cooldown (not the
    /// transition gate) and interrupt at a word boundary.
    func announce(_ phrase: String, urgent: Bool = false) {
        guard isEnabled else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard phrase != lastPhrase else { return }
        guard urgent || now - lastSpokenAt >= cooldown else { return }

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }
        let utterance = AVSpeechUtterance(string: phrase)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        synthesizer.speak(utterance)
        lastPhrase = phrase
        lastSpokenAt = now
    }

    /// Forget the last phrase so the same status can be announced again
    /// (used when scanning restarts).
    func resetTransitions() {
        lastPhrase = nil
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
