# SimpleRayBan — Architecture

A thin iOS app that streams frames from Ray-Ban Meta glasses to a Python ML backend.

## End-to-end data flow

```
┌──────────────────┐    Bluetooth    ┌──────────────────┐
│ Ray-Ban Meta     │ ◄─────────────► │ Meta AI app      │
│ glasses          │ (Classic + LE)  │ (companion)      │
└──────────────────┘                 └────────┬─────────┘
                                              │ MWDAT
                                              │ external-accessory
                                              ▼
                                     ┌──────────────────┐
                                     │ SimpleRayBan     │  ← your iOS app
                                     │ (Swift, DAT SDK) │     (thin layer)
                                     └────────┬─────────┘
                                              │ WebSocket
                                              │ JPEG frames + control
                                              ▼
                                     ┌──────────────────┐
                                     │ Python server    │  ← ML lives here
                                     │ FastAPI + model  │
                                     └────────┬─────────┘
                                              │ result (JSON)
                                              ▼ ─ ─ ─ back to iOS for display
```

The iOS app is intentionally thin: it owns Bluetooth/SDK plumbing and a WebSocket; everything else (model, business logic, storage) lives in Python.

## Inside the iOS app

```
                            SimpleRayBan (Swift)
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────┐   │
  │  │ ContentView │◄──│ GlassesViewModel│◄──│ Wearables.shared │   │
  │  │ (SwiftUI)   │   │ @MainActor      │   │ DeviceSession    │   │
  │  │             │──►│ @Published vars │──►│ StreamSession    │   │
  │  └─────────────┘   └────────┬────────┘   │ (DAT SDK)        │   │
  │       buttons,              │            └──────────────────┘   │
  │       preview,              │                     │             │
  │       status                │                     │ frames      │
  │                             ▼                     ▼             │
  │                      ┌─────────────────────────────────┐        │
  │                      │ FrameForwarder                  │        │
  │                      │  ┌────────────────────────────┐ │        │
  │                      │  │ URLSessionWebSocketTask    │ │        │
  │                      │  │  - encodeFrame() → JPEG    │ │        │
  │                      │  │  - send(.data(jpeg))       │ │        │
  │                      │  └────────────────────────────┘ │        │
  │                      └────────────────┬────────────────┘        │
  └───────────────────────────────────────┼─────────────────────────┘
                                          │
                                          ▼
                                    ws://server/stream
```

## Module map

| Layer | Responsibility | File |
|---|---|---|
| Entry | SDK init, URL callback | `SimpleRayBanApp.swift` |
| UI | Buttons, status, preview, photo display | `ContentView.swift` |
| Coordination | Stream lifecycle, permissions, state, frame fan-out | `GlassesViewModel.swift` |
| Transport | WebSocket connection, JPEG encoding, reconnect | `FrameForwarder.swift` *(to add)* |

The view model is the only place that holds SDK objects. The view binds to its `@Published` properties; the forwarder receives frames via a callback the view model owns. Neither UI nor SDK code knows about the network.

## Frame pipeline

1. Glasses → SDK delivers a `VideoFrame` to `videoFramePublisher.listen { ... }` (off main thread)
2. View model:
   - keeps the latest frame for preview (`@MainActor` hop)
   - hands it to `FrameForwarder` for transmission
3. Forwarder:
   - re-encodes frame as JPEG (drop quality to taste; 0.6–0.8 is a good range)
   - sends over WebSocket as binary

### Backpressure

At 24fps × medium resolution × JPEG you're around 5-15 Mbps. **Do not** enqueue every frame — if the network is slow, the queue grows, latency explodes, and you OOM.

Pattern: hold a single "latest pending frame" slot. When a new frame arrives:

- If the WebSocket has an in-flight send, replace the slot (drop the previous frame).
- When the previous send completes, take whatever is in the slot and send it.

This trades smoothness for low latency, which is what you want for live ML.

## Network protocol (suggested)

WebSocket at `ws://<host>:8000/stream`.

- Client → server: binary JPEG frames; optional text control messages (`{"type":"start","session":"..."}`)
- Server → client: text JSON results (`{"frame_id":42,"label":"...","conf":0.91}`)

For one-off photos (`capturePhoto`), POST `multipart/form-data` to `http://<host>:8000/photo` instead — simpler and avoids interleaving with the stream.

## Where the ML runs

| Option | Latency | Iteration speed | Setup |
|---|---|---|---|
| **Python server** (recommended) | network RTT + inference | fast — `uvicorn --reload` | FastAPI + PyTorch |
| **On-device Core ML** | inference only | slow — convert+rebuild app | `coremltools` PyTorch → CoreML |

Start with the server; move hot paths to Core ML only if/when latency demands it.

## Build & run

| Goal | How |
|---|---|
| Develop UI without glasses | iOS Simulator + **Enable mock device** button |
| Test with real glasses | Plug iPhone into Mac → free Apple ID signing → `⌘R` |
| Iterate on Python | `uvicorn server:app --reload` — Swift reconnects automatically |

## Open items

- [ ] Add `FrameForwarder.swift` with WebSocket + latest-frame-only backpressure
- [ ] Add result handler in view model (`@Published var lastResult: String?`)
- [ ] Stand up FastAPI server skeleton (`/stream` WebSocket + dummy classifier)
- [ ] Decide on auth (token in WebSocket query string is fine for dev)
- [ ] Reconnect/backoff strategy on Wi-Fi drop
