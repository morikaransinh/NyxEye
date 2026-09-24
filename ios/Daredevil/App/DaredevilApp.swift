import SwiftUI

@main
struct DaredevilApp: App {
    @AppStorage("hasAcceptedDisclaimer") private var hasAcceptedDisclaimer = false
    @StateObject private var coordinator = PipelineCoordinator()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(coordinator)
                .fullScreenCover(isPresented: .init(
                    get: { !hasAcceptedDisclaimer },
                    set: { hasAcceptedDisclaimer = !$0 }
                )) {
                    DisclaimerView {
                        hasAcceptedDisclaimer = true
                    }
                    .interactiveDismissDisabled(true)
                }
        }
    }
}
