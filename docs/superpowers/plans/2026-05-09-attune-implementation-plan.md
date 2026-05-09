# Attune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hackathon **Ship It**: **Expo** (iOS + Android + **Web**) with **Spotify login on the client** (PKCE), **Convex** for data + token **B**, **RunPod Serverless** for rank/online-step via official `runsync` API. Train models on RunPod per [`ml/README.md`](../../../ml/README.md). Details: [`docs/runpod-serverless.md`](../../runpod-serverless.md), [`docs/decisions/0001-stack.md`](../../decisions/0001-stack.md). **Product / onboarding / self-improvement:** [`docs/product-onboarding-learning.md`](../../product-onboarding-learning.md).

**Architecture:** Client collects **weather, time, location, playback** → Convex stores events + refresh token; **Convex actions** call `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync` with JSON `{ input: { ... } }` and `Authorization: Bearer RUNPOD_API_KEY` (server-only). Worker handler returns `{ uris, ... }` in `output`.

**Tech stack (locked):** Expo (native + web) · Convex · RunPod Serverless + RunPod train jobs · `packages/attune_ml` · JSON Schema.

---

## AI compute plan — what runs on a GPU, which GPU, how long

Use this as the **budget** for RunPod console settings and hackathon timeboxing. **Re-measure** on first real run and paste timings into `ml/evaluation/benchmarks.md` (create when you have numbers).

### Where compute runs (three lanes)

| Lane | RunPod product | Purpose |
|------|----------------|---------|
| **A. GPU Pod (interactive / batch)** | **Pod** with on-demand GPU | Long jobs: Song InfoNCE pre-train, embedding export forward pass, Context/GRU offline trains, optional User cold GPU train |
| **B. CPU Pod or laptop** | Pod **CPU** only / dev machine | HF → Parquet, tests, **FAISS-CPU** index build (`IndexFlatIP`, ~114k×128), User cold train if you accept ~8–15 min CPU |
| **C. Serverless endpoint** | **Serverless** queue + **handler** | **Inference only**: `rank`, `online_step` (short `runsync`); GPU **tier** chosen per row below |

Official references: [Endpoint configurations](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) (GPU $/s, FlashBoot, timeouts), [Send requests](https://docs.runpod.io/serverless/endpoints/send-requests) (`/runsync`, policies).

### GPU / CPU choice by job (Attune-specific)

| Job | Preferred hardware | Why | Wall-clock estimate (order of magnitude) | Cost sanity (check [runpod.io/gpu-pricing](https://www.runpod.io/gpu-pricing); numbers move) |
|-----|---------------------|-----|---------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Song Tower — InfoNCE** | **NVIDIA RTX 4090 24 GB** (Pod) | Large in-batch negatives (target batch **256–512**); 24 GB fits MLP + big batch | **~15–40 min** for ~20–35 epochs over **~114k** tracks (your earlier target ~20 min @ 30 ep @ 4090-class); add **+5–15 min** first-time CUDA/caches | **~$0.2–$0.6** at ~$0.35–$0.75/hr effective |
| **Song embeddings — forward** | Same 4090 Pod (reuse right after train) | Batch forward entire catalog into 128-d | **~5–20 min** for 114k rows at high batch; CPU fallback **~30–90 min** | marginal if chained same session |
| **FAISS index build** | **CPU** (16 GB RAM plenty) | `IndexFlatIP` on 114k float32 vectors ≈ **~60 MB** vectors + ids; GPU unnecessary | **~1–10 min** | CPU Pod pennies or laptop |
| **User Tower — cold** | **CPU 8 vCPU** Pod *or* **4090** if you want <5 min | Tiny MLP 17→256→128; Spotify history featurization is I/O bound | **~5–15 min CPU**; **~2–8 min GPU** if parallelized | keep on CPU for $ unless demo crunch |
| **Context Encoder — offline** | **L4 / RTX 4090** Pod | Depends on synthetic vs real paired data size; small MLP + gating | **~20 min–2 h** (wide range until data fixed) | start **synthetic ≤30 min** hackathon path |
| **GRU — offline** | **4090** or **L4** Pod | Session batches; teacher forcing | **~15–45 min** for first useful checkpoint on **days** of sessions; **synthetic** sessions **~10–30 min** | one GPU session before bake into Serverless image |
| **Serverless `rank` + `online_step`** | Endpoint GPU tier: **L4 / A5000 / 3090** (24 GB) *or* **4090 PRO** | Load FAISS + torch + small models each worker; [RunPod flex pricing table](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) lists **~$0.00019–$0.00031/s** flex for 24 GB-class | **Per request:** inference **~0.2–2 s GPU** after warm; **cold start + model load** often **~10–60 s** first hit (mitigate below) | pay **per second active** + brief idle per endpoint settings |
| **Convex** | N/A | Never hosts torch/FAISS | ms–low s | Convex billing separate |

### Serverless settings (inference lane C) — match RunPod UI to this checklist

- **GPU priority:** e.g. **L4** first, **4090** second (availability vs cost per [GPU priority](https://docs.runpod.io/serverless/endpoints/endpoint-configurations#gpu-priority-and-worker-distribution)).
- **Execution timeout:** default **600 s**; rank should finish in **<120 s** GPU — set Convex `runsync?wait=120000` and override `policy.executionTimeout` if you hit 600s ceiling on cold starts.
- **FlashBoot:** **on** (default per docs) to cut revive time after idle.
- **Active workers:** **0** dev / **1** during judge window if you can afford **eliminating cold starts** (billed while idle — turn off after).
- **Max workers:** **2–5** cap for demo cost control.
- **Idle timeout:** raise slightly (e.g. **30–60 s**) during demo so second judge request avoids full reload (still costs idle GPU seconds).

### InfoNCE / representation (AI literature — why batch & GPU)

InfoNCE quality scales with **effective negatives** (in-batch negatives common). Large batches need **GPU memory** → 24 GB 4090/L4 class justified vs 16 GB A4000 if batch 512 OOMs. Recent work explores large-batch alternatives (e.g. small-batch methods in contrastive learning literature); for hackathon **default in-batch InfoNCE + 256–512 batch** is the straight path.

### Deliverables each training job must write (for reproducibility)

| Artifact | Produced by | Consumed by |
|----------|-------------|---------------|
| `song_tower_v1.pt` + `train_config.yaml` | Song Pod train | Serverless Docker **COPY** |
| `embeddings_song_v1.parquet` (or `.npy`) | Export step | FAISS builder + optional audit |
| `faiss_song_v1.index` + `faiss_song_v1_ids.npy` | CPU FAISS build | Serverless image |
| `user_tower_init.pt` (optional) | Cold User job | Convex `userWeights` or blob store |
| `context_encoder_v1.pt` | Context Pod train | Serverless image |
| `gru_v1.pt` | GRU Pod train | Serverless image (feature-flag) |

---

## File structure (greenfield — create as you go)

| Path | Responsibility |
|------|------------------|
| `pyproject.toml` | Python package: train/export/FAISS (RunPod + local) |
| `packages/attune_ml/` | Song/User/Context/GRU, `features_song_v1`, train CLIs |
| `convex/` | `schema.ts`, mutations, **actions** (`fetch` RunPod `/runsync` + `/health`), Spotify code exchange |
| `workers/runpod/` | **Serverless** Docker image: `handler.py` (`import runpod`), Dockerfile, baked FAISS + checkpoints — see [`docs/runpod-serverless.md`](../../runpod-serverless.md) |
| `app/` | **Expo** (iOS, Android, **Web**): client Spotify auth, native Remote + **Web Playback SDK** on web, Convex hooks |
| `datasets/schemas/` | JSON Schema (optional codegen to TS validators) |
| `datasets/scripts/` | HF → Parquet |
| `ml/export/` | Checkpoints, FAISS (gitignored); COPY into `workers/runpod/` image build |
| `ml/evaluation/` | `benchmarks.md`, train logs, retrieval@K numbers |

Files that change together: `song_v1.py` + tests + Convex event payload types (`convex/events.ts` validators mirroring schema).

---

### Task 0: Decision log + repo bootstrap

**Files:**

- Create: `pyproject.toml`
- Create: `docs/decisions/0001-stack.md`
- Create: `.gitignore` (add `ml/export/`, `.venv/`, `__pycache__/`, `data/`)

- [ ] **Step 1: Write `pyproject.toml`** with `[project] name = "attune-workspace"`, requires-python `>=3.12`, deps: `torch`, `numpy`, `datasets`, `pandas`, `pytest`, `ruff`, `httpx`, `fastapi`, `uvicorn`, `jsonschema`, `faiss-cpu` (or `faiss-gpu` on train host only).

- [ ] **Step 2: Confirm** ADR + [`docs/runpod-serverless.md`](../../runpod-serverless.md): Ship It, Expo **three platforms**, Serverless **`runsync`**, Spotify **client login + Convex B**.

- [ ] **Step 3: Init git if needed** and commit.

```bash
git add pyproject.toml .gitignore docs/decisions/0001-stack.md
git commit -m "chore: bootstrap pyproject and stack decisions"
```

---

### Task 1: JSON Schemas v1 (contracts)

**Files:**

- Create: `datasets/schemas/song_features_v1.json`
- Create: `datasets/schemas/playback_event_v1.json`
- Create: `datasets/schemas/playlist_request_v1.json`
- Create: `datasets/schemas/playlist_response_v1.json`
- Create: `packages/attune_ml/tests/fixtures/sample_playback_event.json`
- Create: `packages/attune_ml/tests/test_schemas.py`

- [ ] **Step 1: Write failing test** `test_playback_event_validates_against_schema`:

```python
import json
from pathlib import Path
import jsonschema

def test_playback_event_validates_against_schema():
    schema = json.loads(
        Path("datasets/schemas/playback_event_v1.json").read_text()
    )
    sample = json.loads(
        Path("packages/attune_ml/tests/fixtures/sample_playback_event.json").read_text()
    )
    jsonschema.validate(instance=sample, schema=schema)
```

- [ ] **Step 2: Run test — expect FAIL** (schema or fixture missing).

```bash
cd /Users/leo/personal/winners-nia && python -m pytest packages/attune_ml/tests/test_schemas.py::test_playback_event_validates_against_schema -v
```

Expected: file not found or validation error.

- [ ] **Step 3: Add minimal JSON Schemas** — `playback_event_v1.json` with `"type": "object"`, `"required": ["event_id", "user_id", "spotify_track_uri", "listen_ratio", "skip", "client_schema_version"]`, properties for `skip_bucket` enum, `hour_local`, `dow`, `location_bucket`, optional `weather`.

- [ ] **Step 4: Add fixture** `sample_playback_event.json` satisfying schema (use UUID v4 string for `event_id`).

- [ ] **Step 5: Run test — expect PASS**

```bash
python -m pytest packages/attune_ml/tests/test_schemas.py -v
```

- [ ] **Step 6: Commit**

```bash
git add datasets/schemas packages/attune_ml/tests
git commit -m "feat: add v1 JSON schemas and validation test"
```

---

### Task 2: Song feature vector `features_song_v1` (26-d, float32)

**Files:**

- Create: `packages/attune_ml/attune_ml/features/song_v1.py`
- Create: `packages/attune_ml/attune_ml/features/__init__.py`
- Create: `packages/attune_ml/tests/test_song_v1.py`
- Create: `packages/attune_ml/tests/fixtures/hf_track_row.json` (one object with HF column names and numeric values)

**Definition (lock this — Song Tower input):** 26 dims = **12** key one-hot for `key` in `0..11` (if missing or `-1`, all zeros) + **14** scalars in order: `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `norm_loudness = clamp((loudness + 60) / 60, 0, 1)`, `tempo / 240`, `mode`, `explicit` (0/1), `popularity / 100`, `min(duration_ms / 330_000, 1)`, `time_signature / 7` (clip to `[0,1]`). So **12 + 14 = 26**.

- [ ] **Step 1: Add fixture** `packages/attune_ml/tests/fixtures/hf_track_row.json` with at least: `"key": 1`, `"danceability": 0.5`, `"energy": 0.5`, `"speechiness": 0.1`, `"acousticness": 0.2`, `"instrumentalness": 0.0`, `"liveness": 0.1`, `"valence": 0.6`, `"loudness": -10.0`, `"tempo": 120.0`, `"mode": 1`, `"explicit": false`, `"popularity": 50`, `"duration_ms": 200000`, `"time_signature": 4`.

- [ ] **Step 2: Write failing test** `packages/attune_ml/tests/test_song_v1.py`:

```python
import json
from pathlib import Path
import numpy as np
from attune_ml.features.song_v1 import features_song_v1

def test_features_song_v1_fixture_key_and_scalars():
    row = json.loads(Path("packages/attune_ml/tests/fixtures/hf_track_row.json").read_text())
    v = features_song_v1(row)
    assert v.shape == (26,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()
    assert v[1] == 1.0 and v[0] == 0.0
    assert v[12] == 0.5 and v[13] == 0.5
    loud_norm = (-10.0 + 60.0) / 60.0
    assert v[19] == pytest.approx(loud_norm)
    assert v[20] == pytest.approx(120.0 / 240.0)
    assert v[24] == pytest.approx(min(1.0, 200_000 / 330_000))
    assert v[25] == pytest.approx(4.0 / 7.0)
```

Add `import pytest` at top of test file.

- [ ] **Step 3: Run pytest — expect FAIL** (module missing).

```bash
cd /Users/leo/personal/winners-nia && python -m pytest packages/attune_ml/tests/test_song_v1.py -v
```

- [ ] **Step 4: Implement `features_song_v1`** in `song_v1.py`:

```python
import numpy as np

def features_song_v1(row: dict) -> np.ndarray:
    key = int(row.get("key", -1))
    oh = np.zeros(12, dtype=np.float32)
    if 0 <= key <= 11:
        oh[key] = 1.0
    loud = float(row.get("loudness", -60))
    norm_loud = max(0.0, min(1.0, (loud + 60.0) / 60.0))
    dur = float(row.get("duration_ms", 0))
    dur_n = min(1.0, dur / 330_000.0)
    ts = float(row.get("time_signature", 4))
    ts_n = min(1.0, max(0.0, ts / 7.0))
    tempo = float(row.get("tempo", 0)) / 240.0
    rest = np.array(
        [
            float(row["danceability"]),
            float(row["energy"]),
            float(row["speechiness"]),
            float(row["acousticness"]),
            float(row["instrumentalness"]),
            float(row["liveness"]),
            float(row["valence"]),
            norm_loud,
            tempo,
            float(row.get("mode", 0)),
            1.0 if row.get("explicit") in (True, 1, "true") else 0.0,
            float(row.get("popularity", 0)) / 100.0,
            dur_n,
            ts_n,
        ],
        dtype=np.float32,
    )
    return np.concatenate([oh, rest], axis=0)
```

- [ ] **Step 5: Run pytest — expect PASS**

```bash
python -m pytest packages/attune_ml/tests/test_song_v1.py -v
```

- [ ] **Step 6: Commit**

```bash
git add packages/attune_ml/attune_ml/features packages/attune_ml/tests
git commit -m "feat: add features_song_v1 (26-d)"
```

---

### Task 3: Song Tower module + unit test forward pass

**Files:**

- Create: `packages/attune_ml/attune_ml/models/song_tower.py`
- Create: `packages/attune_ml/attune_ml/models/__init__.py`
- Modify: `packages/attune_ml/tests/test_song_v1.py` or new `test_song_tower.py`

- [ ] **Step 1: Failing test** — random input `(4, 26)` → output `(4, 128)` L2-norm ≈ 1 row-wise.

- [ ] **Step 2: Implement `SongTower(torch.nn.Module)`** with Linear 26→256, ReLU, Linear 256→128, `F.normalize(out, dim=-1)`.

- [ ] **Step 3: pytest PASS + commit** `feat: add SongTower MLP`

---

### Task 4: HF materialize script + lock row

**Files:**

- Create: `datasets/scripts/materialize_hf.py`
- Create: `datasets/LOCK` (text: dataset revision string from `datasets.load_dataset` info)

- [ ] **Step 1: Script** downloads `maharshipandya/spotify-tracks-dataset`, writes `data/hf_train.parquet`, prints row count.

- [ ] **Step 2: Run** `python datasets/scripts/materialize_hf.py` (needs network); commit `datasets/LOCK` only if small; **do not** commit parquet if >10MB — add `data/` to gitignore.

- [ ] **Step 3: Commit** `feat: add HF materialize script`

---

### Task 5: Song Tower training CLI (InfoNCE) — full train on **RunPod GPU Pod**

**Hardware:** Start a **Pod** with **RTX 4090 24 GB** (fallback **L4 24 GB** if 4090 unavailable). Install CUDA-matched `torch` wheel in `requirements-train.txt`. Do **not** use Serverless queue for multi-hour train (use Pod SSH / web terminal or `runpodctl`).

**Hyperparameters (v1 defaults — log in `train_config.yaml` output):**

| Knob | Value | Note |
|------|-------|------|
| Optimizer | AdamW | lr `3e-4`–`1e-3`, weight decay `1e-2` |
| Batch size | **256** (L4) / **512** (4090) | Reduce if OOM; InfoNCE benefits from large negatives in-batch |
| Epochs | **25–35** | Early-stop on val proxy (held-out track IDs) |
| Temperature τ | **0.07–0.12** | Standard contrastive range |
| AMP | `torch.cuda.amp` enabled | Speed + memory |
| Determinism | `seed` + cudnn deterministic flag | Repro |

**Files:**

- Create: `packages/attune_ml/attune_ml/train/song_infonce.py`
- Create: `packages/attune_ml/tests/test_song_infonce_shapes.py`
- Create: `ml/evaluation/log_train_run.py` (optional) — append epoch, loss, wall s to JSONL

- [ ] **Step 1: Test** in-batch InfoNCE: batch 8, loss finite, gradients flow to SongTower params.

- [ ] **Step 2: Implement** `train_step` + epoch loop + **checkpoint every N epochs** + **best checkpoint** by val metric.

- [ ] **Step 3: Pod smoke** — `python -m attune_ml.train.song_infonce --epochs 1 --batch 512 --limit-rows 8192` on GPU (**<3 min**).

- [ ] **Step 4: Full train** on full `data/hf_train.parquet` — expect **~15–40 min** wall; record **GPU name, batch, epochs, final loss** in `ml/evaluation/benchmarks.md`.

- [ ] **Step 5: Commit** `feat: song tower InfoNCE training loop`

---

### Task 6: Export embeddings + build FAISS (**CPU lane B**)

**Hardware:** **CPU Pod 4–8 vCPU** or laptop. GPU **optional** (4090 speeds batched forward ~3–10× but not required for 114k rows).

**Files:**

- Create: `packages/attune_ml/attune_ml/index/build_faiss.py`
- Create: `packages/attune_ml/attune_ml/index/export_embeddings.py` (if not merged into train script)

- [ ] **Step 1: Export** all `track_id` + 128-d emb to `ml/export/embeddings_song_v1.parquet` (gitignored). Use **torch.no_grad()**, batch size **2048–8192** on GPU if available.

- [ ] **Step 2: Build** `ml/export/faiss_song_v1.index` (**IndexFlatIP**, L2-normalize vectors before add per `ml/README` dot-product = cosine). `faiss_song_v1_ids.npy` parallel array of `track_id` strings.

- [ ] **Step 3: Smoke test** — random query returns 10 ids; **disk size** note in `benchmarks.md` (index ~60–80 MB class for 114k×128 float32).

- [ ] **Step 4: Commit** `feat: FAISS index builder`

---

### Task 7: Convex project + schema

**Files:**

- Create: `convex/schema.ts` — tables: `users` (spotifyId, encryptedRefresh), `playbackEvents`, `userWeights`, `onboardingJobs`; env-driven **no** RunPod key in schema — use `RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` in Convex **dashboard env** only
- Create: `convex/tsconfig.json`, `package.json` scripts `convex dev`
- Create: `app/` Expo `app.json` / `package.json` with `convex` client

- [ ] **Step 1:** `npx convex dev` smoke; empty schema deploys.

- [ ] **Step 2:** Define `playbackEvents` with fields matching `playback_event_v1` (validate in mutation with `v` from `convex-helpers` or hand-rolled checks).

- [ ] **Step 3:** Commit `feat: convex schema v1`

---

### Task 8: Convex `events.append` mutation (batch + idempotency)

**Files:**

- Create: `convex/events.ts` — `appendBatch` mutation: args `{ idempotencyKey, events[] }`; use unique index on `(userId, idempotencyKey)` or dedicated `eventBatches` doc; reject duplicate with same stored response pattern.

- [ ] **Step 1:** Convex dashboard or `convex test` — duplicate idempotency key does not double-insert events.

- [ ] **Step 2:** Commit `feat: convex event ingest`

---

### Task 8b: Convex `spotify.link` + cold User weights

**Hardware split:** **Spotify token exchange** is Convex CPU only. **Cold User Tower** training: default **CPU Pod 8 vCPU** (**~8–15 min** wall per user per `ml/README`); optional **4090 Pod** (**~2–8 min**) if you batch multiple users or enlarge MLP for demo.

**Files:**

- `convex/spotify.ts` — mutation accepts **`code`** + `redirectUri` from Expo; server exchanges with Spotify; stores **refresh** (encrypted); creates `users` row.
- `packages/attune_ml/attune_ml/train/user_cold.py` — CLI: `--history-json` from Convex export → `user_tower_init.pt` + **17-d** `user_features_v1` spec documented in code.
- Cold path trigger: Convex **action** starts RunPod **Pod** via API *or* runs `onboard_train` **only if** you implement that op in Serverless (prefer **Pod** for long train to avoid Serverless `executionTimeout`).

- [ ] **Step 1:** End-to-end: Expo code → Convex → user row + **weights blob** or **storage URL** in `userWeights`.

- [ ] **Step 2:** Log **train seconds + CPU/GPU SKU** to `benchmarks.md`.

- [ ] **Step 3:** Commit `feat: spotify link and cold user`

---

### Task 9: RunPod Serverless worker + image deploy (**inference lane C**)

**GPU template (console):** Pick **24 GB class** — primary **L4** or **RTX 4090 PRO** per [GPU configuration table](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) (flex vs active $/s). **VRAM budget:** FAISS index (~80 MB) + **4× FP32** small nets (**<500 MB** params total) + PyTorch overhead → **4 GB** sufficient; 24 GB gives headroom for **batch rank** experiments.

**Docker build:** Multi-stage: (1) `requirements-serverless.txt` pinned `torch` CUDA wheel matching endpoint CUDA selector; (2) `COPY ml/export/faiss_song_v1.index` + ids + `song_tower_v1.pt` + `context_encoder_v1.pt` (stub ok) + `gru_v1.pt` optional; (3) `handler.py` loads models **at import** or first request with **global singleton** to amortize cold start.

**Files:**

- Create: `workers/runpod/handler.py` — `handler(event)` with `operation`: `rank` | `online_step`; read `event["input"]`; return dict (becomes **`output`** in `runsync` response).
- Create: `workers/runpod/Dockerfile` — `pip install runpod torch ...`; COPY artifacts; `CMD` starts `runpod.serverless.start`.
- Create: `workers/runpod/README.md` — document **GPU type**, **image tag**, **endpoint id** env wiring.
- Register endpoint in RunPod console; copy **ENDPOINT_ID**.

- [ ] **Step 1:** Local `runpod` dev test (see [local testing](https://docs.runpod.io/serverless/development/local-testing)) returns mock `uris` **<500 ms** after warm.

- [ ] **Step 2:** Push image to registry; deploy Serverless template; set **FlashBoot on**, **executionTimeout** ≥ **300000** ms (5 min) if cold loads are slow; `curl` **`GET .../health`** and **`POST .../runsync`** with smoke `input` per [`docs/runpod-serverless.md`](../../runpod-serverless.md).

- [ ] **Step 3:** Measure **cold vs warm** latency (phone stopwatch + logs): record `delayTime`, `executionTime` from RunPod JSON in `benchmarks.md`.

- [ ] **Step 4:** Convex env `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY` set.

- [ ] **Step 5:** Commit `feat: runpod serverless worker`

---

### Task 10: Convex `playlist.generate` action → `runsync`

**Files:**

- Create: `convex/playlist.ts` — `generate` **action**: `fetch(\`https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync?wait=120000\`, { ... body: JSON.stringify({ input: { operation: "rank", ... } }) })`; parse JSON **`output`** field from RunPod response; map errors (401, 429 backoff).

- [ ] **Step 1:** Convex dashboard test with mock or real endpoint.

- [ ] **Step 2:** Commit `feat: playlist action calls runsync`

---

### Task 11: Online User step via Serverless

**Hardware:** Same Serverless GPU worker as `rank`; forward is **tiny** vs FAISS — expect **~50–300 ms GPU** after warm for one Adam step + replay batch (measure).

**Files:**

- Extend `workers/runpod/handler.py` — `operation: "online_step"` branch: load `user_weights` from `input`, run **one** Adam step (lr `1e-5`, grad clip `0.3`), return `weightsBase64` + `version`.
- `convex/learning.ts` — after event mutation, **action** calls `runsync` with `policy.executionTimeout` **≥ 120000** ms for cold-safe margin.

- [ ] **Step 1:** Unit test handler branch in CI (pytest) on CPU torch (no GPU CI required).

- [ ] **Step 2:** Commit `feat: online step via runsync`

---

### Task 12: Context Encoder (train **Pod lane A**; bake into Serverless image)

**Hardware:** **RTX 4090** or **L4** Pod. Dataset v0 can be **synthetic** (random context + song pairs) for hackathon **<30 min** train; replace with Convex-exported real pairs post-demo.

**Files:**

- `packages/attune_ml/attune_ml/models/context_encoder.py`
- `packages/attune_ml/attune_ml/train/train_context.py` — loss: contrastive or BCE on gated residual magnitude regularizer (match `ml/README` gate).

- [ ] **Step 1:** Train **≥5 epochs** synthetic; wall **~15–45 min** on 4090; export `context_encoder_v1.pt`.

- [ ] **Step 2:** `rank` path loads Context; gated residual matches [`ml/README.md`](../../../ml/README.md).

- [ ] **Step 3:** Rebuild Serverless image including new `context.pt`; redeploy endpoint.

- [ ] **Step 4:** Commit `feat: context in serverless worker`

---

### Task 13: GRU (export from Convex → **Pod train** → new worker image)

**Hardware:** **4090** Pod recommended for hidden size **256** and session batches; **L4** ok with smaller batch.

**Files:**

- `packages/attune_ml/attune_ml/models/session_gru.py`
- `packages/attune_ml/attune_ml/train/train_gru.py` — export `sessions.jsonl` schema: one line per step with **141-d** feature vector definition frozen in code comments.

- [ ] **Step 1:** Synthetic sessions smoke: **~10–25 min** first train; real sessions later **~20–60 min**.

- [ ] **Step 2:** Export `gru_v1.pt`; feature-flag `ATTUNE_GRU_ENABLED` in handler.

- [ ] **Step 3:** Bump worker image tag; redeploy Serverless.

- [ ] **Step 4:** Commit `feat: gru in worker`

---

### Task 14: Expo — Spotify (client) + three platforms + Convex

**Files:**

- `app/` — **iOS/Android:** `expo-auth-session` PKCE + native Spotify module / config plugin where required. **Web:** same PKCE redirect hosts + **Spotify Web Playback SDK** for playback. After auth: send **`code`** to Convex mutation `spotify.link` (name TBD) to complete **B** exchange.
- Telemetry: weather, time, location, playback → `appendBatch`.

- [ ] **Step 1:** All three platforms sign in and hit Convex (web may be dev-only for hackathon — still document).

- [ ] **Step 2:** `app/QA.md` checklist per platform.

- [ ] **Step 3:** Commit `feat: expo spotify web+native + convex`

---

### Task 15: Ship It — Convex prod + EAS (iOS/Android) + Web + RunPod

**Files:**

- `docs/deploy.md` — `npx convex deploy`; EAS profiles; web hosting; RunPod endpoint id; **e2e** `curl` RunPod `/health` + Convex test.

- [ ] **Step 1:** Public URLs for judges (app install + web optional).

- [ ] **Step 2:** Commit `chore: deploy docs`

---

## Ship It scope note

**Task 16 (Always-on / Nia)** is **out of scope** unless track changes — no crons required for judging.

---

## Self-review (writing-plans checklist)

1. **Spec coverage:** GPU matrix + Serverless settings + expanded train tasks + benchmarks file — covered.
2. **Placeholder scan:** Spotify token **B** + client login path explicit.
3. **Type consistency:** RunPod response `output` matches handler return shape; Convex parses same.

---

## Execution handoff

**Plan complete:** `docs/superpowers/plans/2026-05-09-attune-implementation-plan.md`

**1. Subagent-Driven (recommended)** — Fresh subagent per Task (`Task 1`, `Task 2`, …), review between tasks.

**2. Inline execution** — Use `superpowers:executing-plans` with batch checkpoints after Tasks **6** (FAISS), **9** (Serverless live), **10** (playlist `runsync`), **15** (deploy).

**Which approach?** (Pick when you start execution.)
