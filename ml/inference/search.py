"""Nearest-neighbor search over precomputed song embeddings (FAISS or numpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def nearest_neighbors(
    query: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    top_k: int = 20,
    faiss_index_path: str | None = None,
    names: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    if faiss_index_path and Path(faiss_index_path).exists():
        import faiss  # type: ignore

        index = faiss.read_index(faiss_index_path)
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q, top_k)
        return [(str(ids[i]), float(scores[0][j])) for j, i in enumerate(indices[0])]

    q = query / (np.linalg.norm(query) + 1e-8)
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    scores = emb @ q

    # Fetch a wide window so dedup doesn't shrink results below top_k
    fetch = min(top_k * 8, len(scores))
    top_idx = np.argpartition(scores, -fetch)[-fetch:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    # Deduplicate by track ID first, then by normalised "title — artist" string.
    # The dataset has duplicate rows for the same song under different IDs; a
    # case-folded, whitespace-stripped key catches those too.
    seen_ids:   set[str] = set()
    seen_names: set[str] = set()
    out: list[tuple[str, float]] = []

    for i in top_idx:
        tid = str(ids[i])
        if tid in seen_ids:
            continue
        seen_ids.add(tid)

        if names:
            raw = names.get(tid, "")
            key = raw.strip().lower()
            if key and key in seen_names:
                continue
            if key:
                seen_names.add(key)

        out.append((tid, float(scores[i])))
        if len(out) == top_k:
            break

    return out
