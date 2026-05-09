# Attune

Spotify-backed mobile app: per-user playlists (known + discovery) using embeddings, context (time, place, weather), and playback signals (skip, replay, volume). Architecture and numbers: [`ml/README.md`](ml/README.md).

| Path | Purpose |
|------|---------|
| [`app/`](app/) | Auth, playback, events, API client |
| [`datasets/`](datasets/) | ETL, schemas; HF base corpus for Song Tower — [`datasets/README.md`](datasets/README.md) |
| [`ml/`](ml/) | Models, training, deploy — [`ml/README.md`](ml/README.md) |

**Legal** — [Spotify policy](https://developer.spotify.com/policy) / [terms](https://www.spotify.com/legal/end-user-agreement/); HF data has its own license ([dataset card](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)). Align retention and training with both.

**Status** — Docs + empty dirs only.
