import SwiftUI

struct HelpView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    helpSection(
                        icon: "1.circle.fill",
                        title: "Getting started",
                        items: [
                            "Put in your AirPods and enable transparency mode.",
                            "Hold the phone upright at chest height, camera facing forward.",
                            "Tap Start scanning, or double-tap anywhere with two fingers.",
                            "Turn slowly and listen for where the sounds sit in space."
                        ]
                    )
                    helpSection(
                        icon: "waveform",
                        title: "What the sounds mean",
                        items: [
                            "Continuous ocean-like whoosh: a walkable opening in that direction. Lower pitch means it is deeper; slightly louder means farther.",
                            "Pulsing sound: an obstacle within 2 meters in front of you. Faster and louder pulses mean it is closer. Higher pitch means closer too.",
                            "Silence while scanning: no confirmed openings, and nothing close ahead.",
                            "Speech announces changes only, so it stays out of the way."
                        ]
                    )
                    helpSection(
                        icon: "shield.fill",
                        title: "Safety",
                        items: [
                            "Daredevil is a research prototype, not a mobility aid. Keep using your cane or guide dog.",
                            "Volume is capped well below maximum, rises gradually, and the app mutes itself if the camera stalls.",
                            "Glass and mirrors can fool the sensor. Missing depth is never treated as open space, but stay cautious.",
                            "Use transparency mode so you always hear the real world."
                        ]
                    )
                }
                .padding(24)
            }
            .navigationTitle("Help")
            .toolbar {
                Button("Done") { dismiss() }
                    .accessibilityHint("Closes help")
            }
        }
    }

    private func helpSection(icon: String, title: String, items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.title3.bold())
                .accessibilityAddTraits(.isHeader)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .top, spacing: 8) {
                    Text("•").accessibilityHidden(true)
                    Text(item)
                }
            }
        }
    }
}
