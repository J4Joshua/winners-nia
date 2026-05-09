# Mobile app

Spotify SDK playback + PKCE auth. Emit **events** (skip depth/timing, listen ratio, replay, volume Δ, likes, add-to-playlist, session stats) and **context** (hour, DOW, coarse location one-hot, weather: temp/condition/humidity/is-daytime). POST to your API → `spotify:track:` order. Weather-heavy features optional until ~2–3 weeks of logs/user.

**Privacy:** log buckets not raw GPS where possible; match fields to [`../ml/README.md`](../ml/README.md).

| Concern | Approach |
|---------|----------|
| Auth / playback | PKCE; native Spotify SDKs |
| Events | Batched HTTPS |
| Queue | Server-ranked URIs → Spotify queue |

Add `ios/`, `android/`, or `src/` when stack is chosen.
