"""
Push Song Tower artifacts to HuggingFace Hub.

Uploads:
  • song_tower_v1.pt        (TorchScript — used by RunPod worker)
  • song_tower_best.pt      (full checkpoint — resume training)
  • song_embeddings_v1.npy  (128-d embeddings for all ~114k tracks)
  • song_ids_v1.npy         (parallel track ID array)
  • faiss_song_v1.index     (if present — built by build_faiss.py)
  • config.json             (model metadata)
  • README.md               (model card)

Usage:
    python -m ml.export.push_to_hub \
        --repo-id MrlolDev/attune-v0 \
        --output-dir ml/export \
        [--token $HF_TOKEN]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def push_song_tower(
    repo_id: str,
    output_dir: str | Path,
    token: str | None = None,
    private: bool = False,
) -> None:
    from huggingface_hub import HfApi, create_repo  # type: ignore

    output_dir = Path(output_dir)
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)
    print(f"Repo: https://huggingface.co/{repo_id}")

    # Collect files to upload
    artifacts: dict[str, Path] = {
        "song_tower_v1.pt": output_dir / "song_tower_v1.pt",
        "song_tower_best.pt": output_dir / "song_tower_best.pt",
        "song_embeddings_v1.npy": output_dir / "song_embeddings_v1.npy",
        "song_ids_v1.npy": output_dir / "song_ids_v1.npy",
    }

    faiss_path = output_dir / "faiss_song_v1.index"
    if faiss_path.exists():
        artifacts["faiss_song_v1.index"] = faiss_path

    # Write config.json
    config = {
        "model": "SongTower",
        "version": "v1",
        "input_dim": 26,
        "hidden_dim": 256,
        "embedding_dim": 128,
        "activation": "relu",
        "dropout": 0.1,
        "l2_normalize": True,
        "loss": "InfoNCE",
        "dataset": "maharshipandya/spotify-tracks-dataset",
        "feature_spec": "features_song_v1",
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    artifacts["config.json"] = config_path

    # Write model card
    readme = _model_card(repo_id, config)
    readme_path = output_dir / "README_hub.md"
    readme_path.write_text(readme)
    artifacts["README.md"] = readme_path

    # Upload
    for hub_name, local_path in artifacts.items():
        if not local_path.exists():
            print(f"  skip (not found): {local_path}")
            continue
        size_mb = local_path.stat().st_size / 1e6
        print(f"  uploading {hub_name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hub_name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    print(f"\nDone → https://huggingface.co/{repo_id}")


def _model_card(repo_id: str, config: dict) -> str:
    return f"""---
license: mit
tags:
  - pytorch
  - music-recommendation
  - contrastive-learning
  - embedding
  - attune
---

# Attune Song Tower v1

Contrastive MLP that embeds Spotify tracks into a shared 128-d L2-normalized space for two-tower music retrieval.

## Architecture

| Layer | Size |
|-------|------|
| Input | 26-d (12 key one-hot + 14 audio scalars) |
| Hidden | 256-d + ReLU + Dropout(0.1) |
| Output | 128-d + L2-norm |

Loss: **InfoNCE** (symmetric) with in-batch negatives (batch 512, τ=0.07).  
Dataset: [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (~114k tracks).

## Files

| File | Description |
|------|-------------|
| `song_tower_v1.pt` | TorchScript export — use for inference/RunPod |
| `song_tower_best.pt` | Full training checkpoint — resume training |
| `song_embeddings_v1.npy` | Pre-computed 128-d embeddings for all tracks |
| `song_ids_v1.npy` | Parallel Spotify track ID array |
| `faiss_song_v1.index` | FAISS IndexFlatIP — ANN search index |
| `config.json` | Model metadata |

## Quick start

```python
import torch
import numpy as np

model = torch.jit.load("song_tower_v1.pt")
model.eval()

# 26-d feature vector (see config.json for spec)
x = torch.zeros(1, 26)
with torch.no_grad():
    emb = model(x)  # (1, 128)
```

## Feature vector spec (26-d)

```
[0–11]  key one-hot (C=0 … B=11; -1/missing → all zeros)
[12]    danceability
[13]    energy
[14]    speechiness
[15]    acousticness
[16]    instrumentalness
[17]    liveness
[18]    valence
[19]    norm_loudness = clamp((loudness + 60) / 60, 0, 1)
[20]    tempo_norm = clamp(tempo / 240, 0, 1)
[21]    mode (0 or 1)
[22]    explicit (0 or 1)
[23]    popularity_norm = popularity / 100
[24]    duration_norm = min(duration_ms / 330000, 1)
[25]    time_signature_norm = clamp(time_signature / 7, 0, 1)
```
"""


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Push Song Tower to HuggingFace Hub")
    p.add_argument("--repo-id", required=True, help="HF repo id, e.g. myorg/song-tower-v1")
    p.add_argument("--output-dir", default="ml/export")
    p.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")
    push_song_tower(args.repo_id, args.output_dir, token=token, private=args.private)


if __name__ == "__main__":
    main()
