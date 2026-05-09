# Machine learning (PyTorch)

**Attune** uses a shared **Song Tower** (26-d tabular features → 128-d L2 embedding) trained with **InfoNCE** on the [Spotify tracks HuggingFace dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset), plus a per-user **User Tower** (17-d → 128-d) for taste. This package is the training and smoke-test CLI; all heavy lifting is **PyTorch** (CUDA when available, AMP on GPU).

## Install

From the repo root:

```bash
chmod +x ml/scripts/setup_training_env.sh   # once
./ml/scripts/setup_training_env.sh          # auto: extras-only if torch already importable
./ml/scripts/setup_training_env.sh --full   # force pip torch + full ml/requirements.txt
./ml/scripts/setup_training_env.sh --extras-only   # RunPod / any env that already ships PyTorch + CUDA
```

Or manually:

```bash
pip install -r ml/requirements.txt
pip install spotipy   # optional: Spotify OAuth for ml spotify
```

### PyTorch version

Training code needs **PyTorch 2.2+** (`torch.amp`, optional `torch.compile`). Recommended pairing for GPU pods: **PyTorch 2.4.x + CUDA 12.4**.

**RunPod PyTorch templates** (pick an image, then `./ml/scripts/setup_training_env.sh --extras-only` so pip does not replace the container’s CUDA-matched build):

| Template (example tag)                                     | PyTorch   | CUDA   | Notes                          |
| ---------------------------------------------------------- | --------- | ------ | ------------------------------ |
| `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | **2.4.0** | 12.4   | **Default recommendation**     |
| `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` | 2.2.0     | 12.1   | Fine                           |
| `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` | 2.1.0     | 11.8   | Older stack                    |
| PyTorch 2.8.x templates                                    | varies    | varies | Confirm exact tag in RunPod UI |

Files: `ml/requirements.txt` (includes torch for laptop/bare venv), `ml/requirements-train-extras.txt` (no torch — for RunPod).

### Tokens (where to get them)

| Token                         | Used for                                                                             | Where                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Spotify access token**      | `python -m ml spotify`                                                               | Spotify removed the old **Web API Console** “Get Token” pages. Use either: **(A)** [Developer Dashboard](https://developer.spotify.com/dashboard) → create an app → **Settings** → add a **Redirect URI** that matches `--redirect-uri` (default `http://127.0.0.1:8888/callback` or `http://localhost:8888/callback`) → `pip install spotipy` → `python -m ml spotify --client-id … --client-secret …` (opens browser, OAuth with scopes `user-top-read` + `user-read-recently-played`). **(B)** Paste a **Bearer** token from any OAuth flow that obtained those scopes: `python -m ml spotify --token …` (tokens expire, ~1 hour). See [Authorization code flow](https://developer.spotify.com/documentation/web-api/tutorials/code-flow). |
| **`HF_TOKEN`** (Hugging Face) | Uploading models (`ml train --push-to-hub`), gated datasets, Hub downloads if needed | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **write** if you push models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

These are **different** products: Spotify proves _your listening account_; Hugging Face proves _your HF account_.

### Spotify `audio-features` returns 403

Some developer apps no longer get `/v1/audio-features` (Spotify returns **403**). The CLI **does not crash**: it fills audio-derived fields with **neutral defaults** and still saves **`top_track_ids`** / **`recent_track_ids`** so **`train-user`** (embedding centroids) keeps working. Check `audio_features_ok` in the generated JSON `debug` section.

### Weather and “context”

The **full Attune design** uses a separate **Context Encoder** (time, location buckets, **weather**, skips, etc.) that _shifts_ the user vector before search. That model is **not implemented** in this repo yet—only **Song Tower** + **User Tower** (17-d profile → 128-d). So you **cannot** pass weather into `ml run` today and get a context-conditioned embedding from this package.

What you _can_ do now: train a **User Tower** that matches **your** Spotify taste (cold start), then run retrieval and inspect printed track IDs.

### GPU pod — Song Tower train

```bash
cd /path/to/winners-nia
./ml/scripts/setup_training_env.sh --extras-only
export HF_TOKEN=hf_...   # optional
python -m ml train --gpu-preset h100 --output-dir ml/export
```

## Unified CLI (recommended)

All commands are `python -m ml <subcommand> ...` from the repository root.

| Subcommand   | Purpose                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `train`      | Train Song Tower on the HF dataset, export TorchScript + embeddings, optionally push to Hugging Face Hub |
| `run`        | Load artifacts (local or Hub), embed user profile, nearest-neighbor search over the catalog              |
| `faiss`      | Build `IndexFlatIP` from exported embeddings                                                             |
| `hub`        | Upload `ml/export/` artifacts to a Hugging Face **model** repo                                           |
| `train-user` | Cold-start **User Tower** from `ml spotify` JSON + Song embeddings (Hub or local)                        |
| `spotify`    | Pull your top/recent tracks via Web API → write `my_user.json` (17-d features + track IDs)               |

```bash
python -m ml                    # usage
python -m ml train --help
python -m ml run --help
```

### User test (default model on Hugging Face)

The public Song Tower + embeddings live at [**MrlolDev/attune-v0**](https://huggingface.co/MrlolDev/attune-v0). You do **not** need local `ml/export/*` files if you use the default Hub id.

**Fastest** — built-in fake user profile (no Spotify):

```bash
cd /path/to/winners-nia
./ml/scripts/setup_training_env.sh --extras-only   # or pip install -r ml/requirements.txt
python -m ml run --demo-user chill --top-k 20
```

That downloads `song_tower_v1.pt`, `song_embeddings_v1.npy`, and `song_ids_v1.npy` from the Hub, builds a 128-d query from the **chill** demo 17-d vector (random **User Tower** weights — good for pipeline check, not “real” taste until you train User Tower), and prints the top similar track IDs.

**With your Spotify taste** (Dashboard app + Spotipy OAuth, or a Bearer token with `user-top-read` + `user-read-recently-played` — see **Tokens** above):

```bash
python -m ml spotify --token "YOUR_BEARER_TOKEN" --out ml/cli/my_user.json
python -m ml train-user --user-json ml/cli/my_user.json --output ml/export/my_user_tower.pt
python -m ml run --user-json ml/cli/my_user.json --user-tower ml/export/my_user_tower.pt --top-k 20
```

`train-user` aligns your User Tower embedding to the **average Song embedding** of your **top tracks** that appear in the HF catalog (`top_track_ids` in the JSON). If many IDs are missing from the catalog, training may fail—increase overlap by using the same track universe as the Song Tower training set.

Optional: `--hub-repo MrlolDev/attune-v0` (same as the default) or another repo. For local files instead of Hub, use `--song-tower`, `--embeddings`, and `--ids` together.

**Training upload default:** `python -m ml train --push-to-hub` pushes to `MrlolDev/attune-v0` unless you pass `--hub-repo org/other-name`.

### End-to-end example

```bash
# 1) Train and push (default hub: MrlolDev/attune-v0 — override with --hub-repo if needed)
python -m ml train --gpu-preset h100 --output-dir ml/export --push-to-hub

# 2) FAISS index (CPU, optional — speeds up search once you have local embeddings)
python -m ml faiss --embeddings ml/export/song_embeddings_v1.npy --ids ml/export/song_ids_v1.npy

# 3) Your Spotify taste → JSON
python -m ml spotify --token YOUR_TOKEN --out ml/cli/my_user.json

# 4) Retrieval from local export + optional FAISS
python -m ml run \
  --song-tower ml/export/song_tower_v1.pt \
  --embeddings ml/export/song_embeddings_v1.npy \
  --ids ml/export/song_ids_v1.npy \
  --faiss-index ml/export/faiss_song_v1.index \
  --user-json ml/cli/my_user.json \
  --top-k 20
```

Hub-only run (uses [**MrlolDev/attune-v0**](https://huggingface.co/MrlolDev/attune-v0) by default):

```bash
python -m ml run --demo-user chill
# same as:
python -m ml run --hub-repo MrlolDev/attune-v0 --demo-user chill
```

### Legacy module entrypoints (still supported)

```bash
python -m ml.training.train_song_tower --gpu-preset 4090 ...
python -m ml.cli.test_user --song-tower ml/export/song_tower_v1.pt ...   # same as `ml run`
```

## GPU recommendation

InfoNCE quality scales with **in-batch negatives** (batch size). Prefer GPUs with lots of VRAM.

| Priority | GPU                     | Preset              | Batch | Notes                                                                |
| -------- | ----------------------- | ------------------- | ----- | -------------------------------------------------------------------- |
| **Best** | **NVIDIA H100** 80GB    | `--gpu-preset h100` | 2048  | Fast wall-clock + largest batches; enables `torch.compile` in preset |
| Strong   | **NVIDIA A100** 40/80GB | `--gpu-preset a100` | 1024  | Great cost/perf on many clouds                                       |
| Good     | **RTX 4090** 24GB       | `--gpu-preset 4090` | 512   | Default consumer; fine for hackathon timelines                       |
| Budget   | **NVIDIA L4** 24GB      | `--gpu-preset l4`   | 256   | Common on serverless / inference hosts                               |

Rough Song Tower train time (30 epochs, ~114k tracks): **~5–10 min** (H100), **~8–15 min** (A100), **~20–40 min** (4090), **~30–60 min** (L4). Use `--gpu-preset <name>` so batch size, DataLoader workers, and compile flags stay aligned with your card.

Authentication for Hugging Face upload: set `HF_TOKEN` or pass `--hub-token` on `train` / `hub`.

## Layout

```
ml/
├── __main__.py              # python -m ml
├── cli/
│   ├── main.py              # train | run | faiss | hub | spotify
│   ├── run_args.py
│   ├── spotify_profile.py
│   └── test_user.py         # thin wrapper → run
├── training/
│   ├── engine.py            # PyTorch train loop, checkpoints, export
│   ├── presets.py           # GPU presets
│   ├── args.py
│   ├── dataset.py
│   └── train_song_tower.py
├── inference/
│   ├── run.py
│   ├── loaders.py
│   ├── search.py
│   ├── user_query.py
│   └── demo_users.py
├── models/
│   ├── song_tower.py
│   └── user_tower.py
├── scripts/
│   └── setup_training_env.sh  # deps (RunPod: --extras-only)
└── export/
    ├── build_faiss.py
    └── push_to_hub.py
```

## Architecture overview (from product spec)

| Block             | Role                                                              |
| ----------------- | ----------------------------------------------------------------- |
| **Song Tower**    | Shared frozen encoder; trained once on HF catalog                 |
| **User Tower**    | Per-user 17-d taste → 128-d (cold train + online updates in prod) |
| **Context / GRU** | Planned: context shift + session query (not in this CLI yet)      |

Retrieval in production: user/context → FAISS (~500 candidates) → rank → interleave known/new tracks.

## Demo user profiles (`ml run --demo-user`)

| Name    | Vibe                      |
| ------- | ------------------------- |
| `pop`   | Upbeat / afternoon-shaped |
| `chill` | Acoustic / evening-shaped |
| `hype`  | High energy / fast tempo  |

For **your** account, use `ml spotify` then `ml run --user-json ...`.
