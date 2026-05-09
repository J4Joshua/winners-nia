import SwiftUI

struct ContentView: View {
    @StateObject private var vm = GlassesViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                statusCard
                preview
                controls
                if let photo = vm.capturedPhoto {
                    Divider()
                    Text("Last photo").font(.caption).foregroundStyle(.secondary)
                    Image(uiImage: photo)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 160)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                Spacer()
            }
            .padding()
            .navigationTitle("Ray-Ban Meta")
            .onAppear { vm.start() }
        }
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Registration: \(vm.registrationState)")
            Text("Devices paired: \(vm.deviceCount)")
            Text("Camera permission: \(vm.cameraPermission)")
            Text("Stream: \(vm.streamState)")
            if let error = vm.lastError {
                Text("Error: \(error)").foregroundStyle(.red)
            }
        }
        .font(.subheadline.monospaced())
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16).fill(.black)
            if let frame = vm.currentFrame {
                Image(uiImage: frame).resizable().scaledToFit()
            } else {
                Text("No video").foregroundStyle(.white.opacity(0.6))
            }
        }
        .aspectRatio(4.0/3.0, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Button("Register") { vm.register() }
                    .buttonStyle(.borderedProminent)
                Button("Unregister") { vm.unregister() }
                    .buttonStyle(.bordered)
            }
            HStack(spacing: 12) {
                Button("Request camera") { vm.requestCameraPermission() }
                    .buttonStyle(.bordered)
                Button("Refresh") { vm.refreshCameraPermission() }
                    .buttonStyle(.bordered)
            }
            HStack(spacing: 12) {
                Button("Start stream") { vm.startStream() }
                    .buttonStyle(.borderedProminent)
                Button("Stop") { vm.stopStream() }
                    .buttonStyle(.bordered)
                Button("Capture") { vm.capturePhoto() }
                    .buttonStyle(.bordered)
            }
            #if DEBUG
            Button("Enable mock device") { vm.enableMockDevice() }
                .font(.caption)
            #endif
        }
    }
}

#Preview {
    ContentView()
}
