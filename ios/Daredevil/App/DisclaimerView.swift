import SwiftUI

/// Blocking first-launch safety disclaimer, modeled on CurbToCar's honest
/// single-screen approach.
struct DisclaimerView: View {
    let onAccept: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "waveform.and.mic")
                .font(.system(size: 56))
                .accessibilityHidden(true)

            Text("Before you start")
                .font(.largeTitle.bold())

            Text("""
            Daredevil is an experimental research prototype. It translates camera depth into spatial audio, but it cannot guarantee accuracy and it is not a mobility aid. Always use your cane, guide dog, or other primary mobility tools.

            Keep your AirPods in transparency mode so audio cues never block the sounds of the world around you.
            """)
            .font(.body)
            .multilineTextAlignment(.leading)

            Spacer()

            Button(action: onAccept) {
                Text("I understand — continue")
                    .font(.title3.bold())
                    .frame(maxWidth: .infinity, minHeight: 56)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityHint("Confirms you understand Daredevil is a prototype and not a replacement for your mobility aid")
        }
        .padding(24)
    }
}
