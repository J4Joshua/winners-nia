"""
HuggingFace dataset loader + 45-d song feature extractor.

Dataset: maharshipandya/spotify-tracks-dataset (~114k rows)
Feature spec:
  Indices  0-25: original 26-d audio features (12-d key one-hot + 14 scalars)
  Indices 26-44: 19-d macro-genre one-hot

Genre-aware pairing:
  SpotifyTracksDataset pairs each anchor with a random *different* track in the
  same macro-genre, giving InfoNCE a genuine supervisory signal rather than just
  dropout noise. Same-genre tracks should cluster — that's what good recs need.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Macro-genre taxonomy ──────────────────────────────────────────────────────
# 114 Spotify genres collapsed into 19 meaningful groups.
# Index 18 is the catch-all "other" for any unmapped genre.

NUM_MACRO_GENRES = 19

MACRO_GENRE_NAMES = [
    "pop",           # 0
    "rock",          # 1
    "alternative",   # 2
    "electronic",    # 3
    "dance_edm",     # 4
    "hip_hop",       # 5
    "rnb_soul",      # 6
    "jazz",          # 7
    "ambient_calm",  # 8  (classical, piano, ambient, sleep, study, chill, sad)
    "metal",         # 9
    "punk",          # 10
    "country_folk",  # 11
    "latin",         # 12
    "world",         # 13
    "reggae",        # 14
    "children",      # 15
    "jpop_kpop",     # 16
    "gospel",        # 17
    "other",         # 18
]

GENRE_TO_MACRO: dict[str, int] = {
    # Pop (0)
    "pop": 0, "power-pop": 0, "pop-film": 0, "indie-pop": 0, "synth-pop": 0,
    "cantopop": 0, "mandopop": 0, "romance": 0, "happy": 0, "party": 0,
    # Rock (1)
    "rock": 1, "hard-rock": 1, "alt-rock": 1, "rock-n-roll": 1, "rockabilly": 1,
    "psych-rock": 1, "british": 1, "guitar": 1, "j-rock": 1,
    # Alternative / Indie (2)
    "alternative": 2, "indie": 2, "emo": 2, "grunge": 2, "goth": 2,
    # Electronic (3)
    "electronic": 3, "electro": 3, "idm": 3, "breakbeat": 3, "industrial": 3,
    # Dance / EDM / House / Techno (4)
    "dance": 4, "edm": 4, "club": 4, "disco": 4, "house": 4,
    "deep-house": 4, "progressive-house": 4, "chicago-house": 4,
    "detroit-techno": 4, "techno": 4, "trance": 4, "minimal-techno": 4,
    "hardstyle": 4, "garage": 4, "drum-and-bass": 4, "dubstep": 4,
    "dub": 4, "trip-hop": 4, "dancehall": 4,
    # Hip-hop (5)
    "hip-hop": 5,
    # R&B / Soul / Funk (6)
    "r-n-b": 6, "soul": 6, "funk": 6, "groove": 6, "blues": 6,
    # Jazz (7)
    "jazz": 7,
    # Ambient / Calm (8)
    "classical": 8, "opera": 8, "piano": 8, "new-age": 8, "ambient": 8,
    "sleep": 8, "study": 8, "acoustic": 8, "songwriter": 8,
    "singer-songwriter": 8, "chill": 8, "sad": 8,
    # Metal (9)
    "metal": 9, "heavy-metal": 9, "death-metal": 9, "black-metal": 9,
    "metalcore": 9, "grindcore": 9, "hardcore": 9,
    # Punk (10)
    "punk": 10, "punk-rock": 10,
    # Country / Folk (11)
    "country": 11, "folk": 11, "bluegrass": 11, "honky-tonk": 11,
    # Latin (12)
    "latin": 12, "latino": 12, "reggaeton": 12, "salsa": 12, "samba": 12,
    "forro": 12, "pagode": 12, "sertanejo": 12, "mpb": 12, "tango": 12,
    # World / International (13)
    "world-music": 13, "indian": 13, "spanish": 13, "french": 13,
    "german": 13, "swedish": 13, "turkish": 13, "iranian": 13,
    "malay": 13, "brazil": 13, "afrobeat": 13,
    # Reggae / Ska (14)
    "reggae": 14, "ska": 14,
    # Children / Comedy / Disney (15)
    "children": 15, "kids": 15, "disney": 15, "comedy": 15, "show-tunes": 15,
    # J-Pop / K-Pop / Anime (16)
    "j-dance": 16, "j-idol": 16, "j-pop": 16, "k-pop": 16, "anime": 16,
    # Gospel (17)
    "gospel": 17,
    # Other (18) — default fallback, not listed here
}

SONG_FEATURE_DIM = 26 + NUM_MACRO_GENRES  # 45


# ── Feature extraction ────────────────────────────────────────────────────────

def genre_to_macro(genre: str | None) -> int:
    """Map a raw dataset genre string to a macro-genre index (0-18)."""
    if not genre:
        return NUM_MACRO_GENRES - 1  # "other"
    return GENRE_TO_MACRO.get(str(genre).strip().lower(), NUM_MACRO_GENRES - 1)


def macro_genre_onehot(macro_idx: int) -> np.ndarray:
    vec = np.zeros(NUM_MACRO_GENRES, dtype=np.float32)
    vec[macro_idx] = 1.0
    return vec


def extract_song_features(row: dict) -> tuple[np.ndarray, int] | None:
    """
    Convert one dataset row → (45-d float32 array, macro_genre_idx).
    Returns None if the row is missing critical fields.

    Layout:
      [0:12]  key one-hot (12 semitones)
      [12:26] 14 audio scalars
      [26:45] 19-d macro-genre one-hot
    """
    try:
        audio = np.zeros(26, dtype=np.float32)

        # Indices 0–11: key one-hot
        key = row.get("key", -1)
        if key is not None and 0 <= int(key) <= 11:
            audio[int(key)] = 1.0

        def _get(field: str, default: float = 0.0) -> float:
            v = row.get(field)
            return float(v) if v is not None else default

        audio[12] = _get("danceability")
        audio[13] = _get("energy")
        audio[14] = _get("speechiness")
        audio[15] = _get("acousticness")
        audio[16] = _get("instrumentalness")
        audio[17] = _get("liveness")
        audio[18] = _get("valence")

        loudness = _get("loudness", -60.0)
        audio[19] = float(np.clip((loudness + 60.0) / 60.0, 0.0, 1.0))

        tempo = _get("tempo", 0.0)
        audio[20] = float(np.clip(tempo / 240.0, 0.0, 1.0))

        audio[21] = float(_get("mode"))
        audio[22] = 1.0 if row.get("explicit") else 0.0
        audio[23] = float(np.clip(_get("popularity") / 100.0, 0.0, 1.0))

        duration_ms = _get("duration_ms", 0.0)
        audio[24] = float(np.clip(duration_ms / 330_000.0, 0.0, 1.0))

        time_sig = _get("time_signature", 4.0)
        audio[25] = float(np.clip(time_sig / 7.0, 0.0, 1.0))

        macro_idx = genre_to_macro(row.get("track_genre"))
        genre_vec = macro_genre_onehot(macro_idx)

        return np.concatenate([audio, genre_vec]), macro_idx

    except Exception:
        return None


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SpotifyTracksDataset(Dataset):
    """
    Dataset for Song Tower training with Supervised Contrastive Loss (SupCon).

    Each __getitem__(i) returns (features[i], genre_label[i]).
    The training loop uses the genre labels to identify all same-genre samples
    in the batch as positives — far richer signal than one-to-one pairing.
    """

    def __init__(
        self,
        features: np.ndarray,
        track_ids: list[str],
        genre_labels: np.ndarray,  # int32, shape (N,), macro-genre index per track
    ) -> None:
        assert features.shape[1] == SONG_FEATURE_DIM, (
            f"Expected {SONG_FEATURE_DIM}-d features, got {features.shape[1]}"
        )
        self.features = torch.from_numpy(features)
        self.track_ids = track_ids
        self.genre_labels = genre_labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], int(self.genre_labels[idx])


# ── Dataset builder ───────────────────────────────────────────────────────────

def load_spotify_dataset(
    val_fraction: float = 0.10,
    seed: int = 42,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[SpotifyTracksDataset, SpotifyTracksDataset, list[str], dict[str, str]]:
    """
    Download + preprocess the HF dataset; split 90/10 by track ID.

    Returns:
        train_ds, val_ds, all_track_ids, track_names
        where track_names is {track_id: "Track Name — Artist"}
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
    genre_labels_list: list[int] = []
    track_ids: list[str] = []
    track_names: dict[str, str] = {}

    print(f"Extracting features from {len(hf):,} rows...")
    for row in hf:
        result = extract_song_features(row)
        if result is None:
            continue
        feat, macro_idx = result
        tid = str(row.get("track_id") or row.get("id") or len(track_ids))
        features_list.append(feat)
        genre_labels_list.append(macro_idx)
        track_ids.append(tid)

        if tid not in track_names:
            name = str(row.get("track_name") or row.get("name") or "?")
            artists = row.get("artists") or ""
            if isinstance(artists, list):
                artist = artists[0] if artists else "?"
            else:
                artist = str(artists).strip("[]'\" ")
            track_names[tid] = f"{name} — {artist}"

    features = np.stack(features_list, axis=0)          # (N, 45)
    genre_labels = np.array(genre_labels_list, dtype=np.int32)  # (N,)
    print(f"  → kept {len(features):,} tracks  ({len(track_names):,} unique IDs)")

    # Genre distribution summary
    unique, counts = np.unique(genre_labels, return_counts=True)
    genre_summary = "  → macro-genre counts: " + "  ".join(
        f"{MACRO_GENRE_NAMES[int(g)]}={c}" for g, c in zip(unique, counts)
    )
    print(genre_summary)

    # Val split by unique track ID (prevents leakage across duplicate rows)
    unique_ids = list(dict.fromkeys(track_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_val_ids = max(1, int(len(unique_ids) * val_fraction))
    val_id_set = set(unique_ids[:n_val_ids])

    train_idx = [i for i, tid in enumerate(track_ids) if tid not in val_id_set]
    val_idx = [i for i, tid in enumerate(track_ids) if tid in val_id_set]

    train_ds = SpotifyTracksDataset(
        features[train_idx],
        [track_ids[i] for i in train_idx],
        genre_labels[train_idx],
    )
    val_ds = SpotifyTracksDataset(
        features[val_idx],
        [track_ids[i] for i in val_idx],
        genre_labels[val_idx],
    )

    print(f"  → train: {len(train_ds):,}  val: {len(val_ds):,}")
    return train_ds, val_ds, track_ids, track_names
