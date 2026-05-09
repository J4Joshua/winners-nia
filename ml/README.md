# Machine learning

Four blocks: **Song Tower** (shared, frozen) embeds tracks from tabular audio features; **User Tower** (per user, online-updated) maps history + behavior into the same 128-d space; **Context Encoder** gates a context shift on that vector; **GRU** outputs a session query for “next” in embedding space. Retrieval: **FAISS** (~500 candidates) → dot-product rank → known/new interleave (~20 tracks).

## Models

| Block | Spec | Train / run |
|-------|------|-------------|
| **Song** | MLP 26→256→128, **InfoNCE**, ~114k rows ([HF dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)) | GPU pre-train once; **no grad** in prod / online step |
| **User** | MLP 17→256→128 | Cold fit on Spotify history (~minutes CPU); online: 1 Adam step (e.g. lr 1e-5, clip 0.3) on **User weights only**, song frozen; replay ~200, batch new + ~8 random past; aim sub-2ms CPU |
| **Context** | User 128-d + ~16-d context (hour, DOW, skip stats, location 1-hot, volume Δ, weather: temp/condition/humidity/is-daytime); **learned gate** on shift strength | Shared weights; ramp weather after ~2–3 weeks/user |
| **GRU** | Step input ~141-d (128 song emb + listen ratio, skip, replay, vol Δ, hour, session skip rate, consecutive skips, location 3, position); **256-d** hidden → **128-d** query | Offline after **~1–2 weeks** real sessions |

**Shared:** Song + Context + GRU weights, global FAISS. **Per user:** User Tower (~212 KB fp32), replay buffer, session state for GRU. **Scale:** optional **MAML**-style few-step init instead of full cold User train.

## Labels (event → weight)

Replay 1.0 (strong User signal, e.g. 3×); add-to-playlist ~0.95; complete >~80% ~0.85; vol up +~0.15; listen >~50% ~0.60; early skip ~5–30s ~0.15; skip <~5s ~0.0; three consecutive skips → train stronger context shift. Tune thresholds but keep train/serve identical.

## Inference

User → Context → FAISS (known / unknown pools) → GRU query → dot rank → interleave (e.g. 2 known / 3 new, max 3 new streak); tilt mix from skip/replay rate and location heuristics (e.g. gym vs home night).

## Ops (rough)

Song pre-train: short **GPU** job (large InfoNCE batches). User cold + weekly User+GRU: modest CPU/GPU. **Prod start:** small **VPS** (REST + online UT + ~150–200 ms rank on CPU); add serverless GPU for inference if needed.

## Layout

`training/`, `evaluation/`, `export/` (checkpoints, FAISS — gitignore binaries), optional `notebooks/`. Pin deps when code exists; no secrets in repo.
