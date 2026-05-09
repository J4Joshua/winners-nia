import MWDATCore
import SwiftUI

#if DEBUG
import MWDATMockDevice
#endif

@main
struct SimpleRayBanApp: App {
    init() {
        do {
            try Wearables.configure()
        } catch {
            assertionFailure("Wearables SDK configuration failed: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onOpenURL { url in
                    Task { _ = try? await Wearables.shared.handleUrl(url) }
                }
        }
    }
}
