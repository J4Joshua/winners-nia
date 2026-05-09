"""
HuggingFace dataset loader + 26-d song feature extractor.

Dataset: maharshipandya/spotify-tracks-dataset (~114k rows)
Feature spec: 12-d key one-hot + 14 audio scalars = 26-d total
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_song_features(row: dict) -> np.ndarray | None:
    """
    Convert one dataset row → 26-d float32 numpy array.
    Returns None if the row is missing critical fields.
    """
    try:
        vec = np.zeros(26, dtype=np.float32)

        # Indices 0–11: key one-hot (12 semitones)
        key = row.get("key", -1)
        if key is not None and 0 <= int(key) <= 11:
            vec[int(key)] = 1.0

        # Indices 12–25: 14 audio scalars
        def _get(field: str, default: float = 0.0) -> float:
            v = row.get(field)
            return float(v) if v is not None else default

        vec[12] = _get("danceability")
        vec[13] = _get("energy")
        vec[14] = _get("speechiness")
        vec[15] = _get("acousticness")
        vec[16] = _get("instrumentalness")
        vec[17] = _get("liveness")
        vec[18] = _get("valence")

        loudness = _get("loudness", -60.0)
        vec[19] = float(np.clip((loudness + 60.0) / 60.0, 0.0, 1.0))

        tempo = _get("tempo", 0.0)
        vec[20] = float(np.clip(tempo / 240.0, 0.0, 1.0))

        vec[21] = float(_get("mode"))
        vec[22] = 1.0 if row.get("explicit") else 0.0
        vec[23] = float(np.clip(_get("popularity") / 100.0, 0.0, 1.0))

        duration_ms = _get("duration_ms", 0.0)
        vec[24] = float(np.clip(duration_ms / 330_000.0, 0.0, 1.0))

        time_sig = _get("time_signature", 4.0)
        vec[25] = float(np.clip(time_sig / 7.0, 0.0, 1.0))

        return vec

    except Exception:
        return None


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SpotifyTracksDataset(Dataset):
    """
    Wraps the pre-processed feature matrix for Song Tower training.

    Each __getitem__ returns the same feature vector twice so the training
    loop can pass both through the model (with dropout active) to get two
    different augmented views — (anchor, positive) pair for InfoNCE.
    """

    def __init__(
        self,
        features: np.ndarray,
        track_ids: list[str],
    ) -> None:
        assert features.shape[1] == 26, f"Expected 26-d, got {features.shape[1]}"
        self.features = torch.from_numpy(features)
        self.track_ids = track_ids

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        # Both views come from the same raw vector;
        # dropout inside SongTower creates the augmentation difference.
        return x, x


# ── Dataset builder ───────────────────────────────────────────────────────────

def load_spotify_dataset(
    val_fraction: float = 0.10,
    seed: int = 42,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[SpotifyTracksDataset, SpotifyTracksDataset, list[str]]:
    """
    Download + preprocess the HF dataset; split 90/10 by track ID.

    Returns:
        train_ds, val_ds, all_track_ids
    """
    from datasets import load_dataset  # type: ignore

    print("Loading maharshipandya/spotify-tracks-dataset from HuggingFace...")
    hf = load_dataset(
        "maharshipandya/spotify-tracks-dataset",
        split="train",
        cache_dir=cache_dir,
        revision=revision,
    )

    features_list: list[np.ndarray] = []
    track_ids: list[str] = []

    print(f"Extracting features from {len(hf):,} rows...")
    for row in hf:
        feat = extract_song_features(row)
        if feat is None:
            continue
        tid = row.get("track_id") or row.get("id") or str(len(track_ids))
        features_list.append(feat)
        track_ids.append(str(tid))

    features = np.stack(features_list, axis=0)  # (N, 26)
    print(f"  → kept {len(features):,} tracks after filtering")

    # Val split by unique track ID (prevents leakage across duplicate rows)
    unique_ids = list(dict.fromkeys(track_ids))  # preserve order, deduplicate
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_val_ids = max(1, int(len(unique_ids) * val_fraction))
    val_id_set = set(unique_ids[:n_val_ids])

    train_idx = [i for i, tid in enumerate(track_ids) if tid not in val_id_set]
    val_idx = [i for i, tid in enumerate(track_ids) if tid in val_id_set]

    train_ds = SpotifyTracksDataset(features[train_idx], [track_ids[i] for i in train_idx])
    val_ds = SpotifyTracksDataset(features[val_idx], [track_ids[i] for i in val_idx])

    print(f"  → train: {len(train_ds):,}  val: {len(val_ds):,}")
    return train_ds, val_ds, track_ids
