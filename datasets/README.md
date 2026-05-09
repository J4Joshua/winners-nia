# Datasets

**Base (pre-train Song Tower):** [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) — ~114k rows, BSD per card (recheck HF). Has `track_id`, names, `popularity` / `duration_ms` / `explicit`, Spotify-style audio columns (`danceability` … `tempo`, `time_signature`), `track_genre`. Pin revision for reproducibility. 26-d Song input = chosen subset + normalizer (document in `schemas/` when fixed).

**Live:** per-user API pulls (policy + rate limits); app event logs (skips, progress, volume, coarse location, weather) — schema in `schemas/` when code exists. HF table ≠ production telemetry.

**Splits:** HF — random or stratified by genre; per-user — **time-based** so future sessions don’t leak.

**Layout:** `schemas/`, `scripts/` (HF + ETL), `samples/` (synthetic CI only).

**Legal:** HF license vs Spotify terms for API data — audit what you store and why.
