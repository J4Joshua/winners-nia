import Foundation
import MWDATCamera
import MWDATCore
import SwiftUI
import UIKit

#if DEBUG
import MWDATMockDevice
#endif

@MainActor
final class GlassesViewModel: ObservableObject {
    @Published var registrationState: String = "unknown"
    @Published var streamState: String = "stopped"
    @Published var currentFrame: UIImage?
    @Published var capturedPhoto: UIImage?
    @Published var deviceCount: Int = 0
    @Published var cameraPermission: String = "unknown"
    @Published var lastError: String?

    private let wearables: WearablesInterface = Wearables.shared
    private let deviceSelector: AutoDeviceSelector
    private var deviceSession: DeviceSession?
    private var streamSession: StreamSession?

    private var stateToken: AnyListenerToken?
    private var frameToken: AnyListenerToken?
    private var photoToken: AnyListenerToken?
    private var errorToken: AnyListenerToken?

    init() {
        self.deviceSelector = AutoDeviceSelector(wearables: Wearables.shared)
    }

    func start() {
        Task {
            for await state in wearables.registrationStateStream() {
                self.registrationState = "\(state)"
            }
        }
        Task {
            for await devices in wearables.devicesStream() {
                self.deviceCount = devices.count
            }
        }
    }

    func register() {
        Task {
            do { try await wearables.startRegistration() } catch {
                lastError = "register: \(error.localizedDescription)"
            }
        }
    }

    func unregister() {
        Task { try? await wearables.startUnregistration() }
    }

    func refreshCameraPermission() {
        Task {
            do {
                let status = try await wearables.checkPermissionStatus(.camera)
                self.cameraPermission = "\(status)"
            } catch {
                self.cameraPermission = "error: \(error.localizedDescription)"
            }
        }
    }

    func requestCameraPermission() {
        Task {
            do {
                let status = try await wearables.requestPermission(.camera)
                self.cameraPermission = "\(status)"
            } catch {
                self.cameraPermission = "error: \(error.localizedDescription)"
            }
        }
    }

    func startStream() {
        guard streamSession == nil else { return }
        Task { await beginStream() }
    }

    private func beginStream() async {
        do {
            streamState = "checking permission"
            let permission = try await wearables.checkPermissionStatus(.camera)
            let granted: PermissionStatus
            if permission != .granted {
                streamState = "requesting permission"
                granted = try await wearables.requestPermission(.camera)
            } else {
                granted = permission
            }
            guard granted == .granted else {
                lastError = "Camera permission not granted"
                streamState = "stopped"
                return
            }

            streamState = "creating session"
            let session = try wearables.createSession(deviceSelector: deviceSelector)
            deviceSession = session

            // Listen for session errors in parallel
            let errorTask = Task { [weak self] in
                for await err in session.errorStream() {
                    await MainActor.run { self?.lastError = "session: \(err)" }
                }
            }

            streamState = "starting session"
            try session.start()

            if session.state != .started {
                streamState = "waiting for session"
                for await state in session.stateStream() {
                    streamState = "session: \(state)"
                    if state == .started { break }
                    if state == .stopped {
                        deviceSession = nil
                        streamState = "stopped"
                        errorTask.cancel()
                        return
                    }
                }
            }
            errorTask.cancel()

            streamState = "adding stream"
            let config = StreamSessionConfig(
                videoCodec: VideoCodec.raw,
                resolution: StreamingResolution.medium,
                frameRate: 24
            )
            guard let stream = try session.addStream(config: config) else {
                lastError = "stream: addStream returned nil"
                streamState = "stopped"
                return
            }
            streamSession = stream
            attachListeners(to: stream)
            streamState = "starting stream"
            await stream.start()
        } catch {
            lastError = "stream: \(error.localizedDescription)"
            streamState = "stopped"
        }
    }

    private func attachListeners(to stream: StreamSession) {
        stateToken = stream.statePublisher.listen { [weak self] (state: StreamSessionState) in
            Task { @MainActor in self?.streamState = "\(state)" }
        }
        frameToken = stream.videoFramePublisher.listen { [weak self] (frame: VideoFrame) in
            guard let image = frame.makeUIImage() else { return }
            Task { @MainActor in self?.currentFrame = image }
        }
        photoToken = stream.photoDataPublisher.listen { [weak self] (photo: PhotoData) in
            guard let image = UIImage(data: photo.data) else { return }
            Task { @MainActor in self?.capturedPhoto = image }
        }
        errorToken = stream.errorPublisher.listen { [weak self] (error: StreamSessionError) in
            Task { @MainActor in self?.lastError = "\(error)" }
        }
    }

    func stopStream() {
        let stream = streamSession
        let session = deviceSession
        streamSession = nil
        deviceSession = nil
        stateToken = nil
        frameToken = nil
        photoToken = nil
        errorToken = nil
        currentFrame = nil
        Task {
            await stream?.stop()
            session?.stop()
        }
    }

    func capturePhoto() {
        _ = streamSession?.capturePhoto(format: .jpeg)
    }

    #if DEBUG
    func enableMockDevice() {
        Task {
            let kit = MockDeviceKit.shared
            kit.enable()
            let device = kit.pairRaybanMeta()
            if let url = Bundle.main.url(forResource: "plant", withExtension: "mp4") {
                device.services.camera.setCameraFeed(fileURL: url)
            }
            if let url = Bundle.main.url(forResource: "plant", withExtension: "png") {
                device.services.camera.setCapturedImage(fileURL: url)
            }
            device.powerOn()
            device.don()
        }
    }
    #endif
}
