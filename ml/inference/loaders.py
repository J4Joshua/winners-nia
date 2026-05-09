"""Load Song Tower (TorchScript or PyTorch checkpoint) and optional User Tower."""

from __future__ import annotations

import json  # noqa: F401 (used via _json alias inside functions)
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ml.models.song_tower import SongTower
from ml.models.user_tower import UserTower


def load_song_tower(path: str | Path, device: torch.device) -> nn.Module:
    path = Path(path)
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.eval()
        return m
    except Exception:
        pass
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model = SongTower()
    model.load_state_dict(state)
    return model.to(device).eval()


def load_user_tower(path: str | Path, device: torch.device) -> UserTower:
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model = UserTower()
    model.load_state_dict(state)
    return model.to(device).eval()


def load_artifacts_from_hub(
    repo_id: str,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, str]]:
    import json as _json
    from huggingface_hub import hf_hub_download  # type: ignore

    ts_path = hf_hub_download(repo_id, "song_tower_v1.pt")
    emb_path = hf_hub_download(repo_id, "song_embeddings_v1.npy")
    ids_path = hf_hub_download(repo_id, "song_ids_v1.npy")
    model = load_song_tower(ts_path, device)
    embeddings = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path, allow_pickle=True)

    track_names: dict[str, str] = {}
    try:
        names_path = hf_hub_download(repo_id, "song_names_v1.json")
        track_names = _json.loads(Path(names_path).read_text(encoding="utf-8"))
    except Exception:
        pass  # older Hub repos without names file — names shown blank

    return model, embeddings, ids, track_names
