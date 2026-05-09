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
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(str(ids[i]), float(scores[i])) for i in top_idx]
