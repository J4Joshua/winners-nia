"""
Build a FAISS IndexFlatIP from pre-computed song embeddings.

Run this after train_song_tower.py exports song_embeddings_v1.npy.

Usage:
    python -m ml.export.build_faiss \
        --embeddings ml/export/song_embeddings_v1.npy \
        --ids ml/export/song_ids_v1.npy \
        --output ml/export/faiss_song_v1.index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_index(
    embeddings_path: str | Path,
    ids_path: str | Path,
    output_path: str | Path,
) -> None:
    try:
        import faiss  # type: ignore
    except ImportError:
        raise SystemExit("faiss-cpu not installed. Run: pip install faiss-cpu")

    embeddings_path = Path(embeddings_path)
    ids_path = Path(ids_path)
    output_path = Path(output_path)

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)  # (N, 128)
    ids = np.load(ids_path, allow_pickle=True)

    assert embeddings.ndim == 2, "Expected (N, 128)"
    n, d = embeddings.shape
    print(f"  → {n:,} embeddings, dim={d}")

    # L2-normalize before building an inner-product index
    # (embeddings are already normalized from the model, but be explicit)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"  → index.ntotal = {index.ntotal:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"Saved → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity check
    q = embeddings[:1]
    scores, indices = index.search(q, 5)
    top_ids = ids[indices[0]]
    print(f"\nSanity check (track[0] nearest neighbors): {top_ids[:5]}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS index from song embeddings")
    p.add_argument("--embeddings", default="ml/export/song_embeddings_v1.npy")
    p.add_argument("--ids", default="ml/export/song_ids_v1.npy")
    p.add_argument("--output", default="ml/export/faiss_song_v1.index")
    args = p.parse_args()
    build_index(args.embeddings, args.ids, args.output)


if __name__ == "__main__":
    main()
