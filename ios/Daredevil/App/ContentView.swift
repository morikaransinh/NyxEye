import SwiftUI
import DaredevilKit

struct ContentView: View {
    @EnvironmentObject private var coordinator: PipelineCoordinator
    @State private var showHelp = false

    var body: some View {
        NavigationStack {
            Form {
                statusSection
                startStopSection
                cuesSection
                volumeSection
                settingsSection
            }
            .navigationTitle("Daredevil")
            .toolbar {
                Button {
                    showHelp = true
                } label: {
                    Image(systemName: "questionmark.circle")
                }
                .accessibilityLabel("Help")
                .accessibilityHint("Explains what each sound means and how to use the app")
            }
            .sheet(isPresented: $showHelp) {
                HelpView()
            }
        }
        // VoiceOver Magic Tap (two-finger double-tap) toggles scanning —
        // the single most important action is always one gesture away.
        .accessibilityAction(.magicTap) {
            coordinator.toggle()
        }
    }

    private var statusSection: some View {
        Section {
            HStack(spacing: 12) {
                Circle()
                    .fill(coordinator.isRunning ? Color.green : Color.secondary)
                    .frame(width: 14, height: 14)
                    .accessibilityHidden(true)
                Text(coordinator.statusText)
                    .font(.title3)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Status: \(coordinator.statusText)")
            .accessibilityAddTraits(.updatesFrequently)

            if !coordinator.isDepthSupported {
                Label("This device has no LiDAR sensor. Daredevil needs an iPhone Pro model.",
                      systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            }
        }
    }

    private var startStopSection: some View {
        Section {
            Button {
                coordinator.toggle()
            } label: {
                Text(coordinator.isRunning ? "Stop scanning" : "Start scanning")
                    .font(.title2.bold())
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, minHeight: 64)
            }
            .listRowInsets(EdgeInsets())
            .background(coordinator.isRunning ? Color.red : Color.green)
            .disabled(!coordinator.isDepthSupported)
            .accessibilityHint(coordinator.isRunning
                ? "Stops the camera and silences all audio cues"
                : "Starts sensing depth and playing spatial audio cues. You can also double-tap with two fingers anywhere to start or stop.")
        } footer: {
            Text("Hold the phone upright at chest height, camera facing forward. Two-finger double-tap anywhere starts or stops scanning.")
        }
    }

    private var cuesSection: some View {
        Section {
            Toggle("Open spaces", isOn: $coordinator.openSpacesEnabled)
                .accessibilityHint("Continuous ocean-like sound placed in the direction of walkable openings. Deeper openings sound lower in pitch.")
            Toggle("Obstacle warnings", isOn: $coordinator.obstaclesEnabled)
                .accessibilityHint("Pulsing sound when something is within 2 meters ahead. Pulses get faster and louder as you get closer.")
            Toggle("Spoken status", isOn: $coordinator.speechEnabled)
                .accessibilityHint("Speaks changes, like open space to the left, or obstacle 1 meter ahead")
            Toggle("AirPods head tracking", isOn: $coordinator.headTrackingEnabled)
                .accessibilityHint("Keeps sounds anchored to the world when you turn your head. Requires AirPods.")
        } header: {
            Text("Audio cues")
        } footer: {
            Text("Whoosh = walkable opening in that direction. Fast pulses = obstacle close ahead. Higher pitch = closer.")
        }
    }

    private var volumeSection: some View {
        Section {
            HStack {
                Image(systemName: "speaker.fill")
                    .accessibilityHidden(true)
                Slider(value: $coordinator.userVolume, in: 0...1)
                    .accessibilityLabel("Cue volume")
                    .accessibilityValue("\(Int(coordinator.userVolume * 100)) percent")
                Image(systemName: "speaker.wave.3.fill")
                    .accessibilityHidden(true)
            }
        } header: {
            Text("Volume")
        } footer: {
            Text("Daredevil caps its loudness well below maximum to protect your hearing, and mutes itself if sensing ever stalls.")
        }
    }

    private var settingsSection: some View {
        Section {
            Button("Replay safety information") {
                showHelp = true
            }
            .frame(minHeight: 44)
            .accessibilityHint("Opens the help sheet with safety notes and sound meanings")
        }
    }
}
