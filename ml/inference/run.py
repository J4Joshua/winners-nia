"""Top-level retrieval run (Song/User towers + neighbor search)."""

from __future__ import annotations

import time
from argparse import Namespace

import numpy as np

from ml.inference.defaults import DEFAULT_HUB_MODEL_REPO
from ml.inference.device import resolve_device
from ml.inference.loaders import load_artifacts_from_hub, load_song_tower
from ml.inference.search import nearest_neighbors
from ml.inference.user_query import build_user_query_embedding


def run_inference(args: Namespace) -> None:
    device = resolve_device(getattr(args, "device", None))
    print(f"Device: {device}")

    tower_path = getattr(args, "song_tower", None)
    if tower_path:
        emb_path = getattr(args, "embeddings", None)
        ids_path = getattr(args, "ids", None)
        if not emb_path or not ids_path:
            raise SystemExit("With --song-tower, also pass --embeddings and --ids")
        song_model = load_song_tower(tower_path, device)
        print(f"Loaded Song Tower: {tower_path}")
        embeddings = np.load(emb_path).astype(np.float32)
        ids = np.load(ids_path, allow_pickle=True)
        print(f"Embeddings: {emb_path}")
    else:
        repo = getattr(args, "hub_repo", None) or DEFAULT_HUB_MODEL_REPO
        song_model, embeddings, ids = load_artifacts_from_hub(repo, device)
        print(f"Loaded from Hugging Face Hub: https://huggingface.co/{repo}")

    print(f"Shape: {embeddings.shape}  tracks: {len(ids):,}\n")

    t0 = time.perf_counter()
    query, profile_label = build_user_query_embedding(args, device)
    query_ms = (time.perf_counter() - t0) * 1000
    print(f"Profile: {profile_label}")
    print(f"Query vector built in {query_ms:.1f} ms\n")

    t0 = time.perf_counter()
    top_k = int(getattr(args, "top_k", 20))
    results = nearest_neighbors(
        query,
        embeddings,
        ids,
        top_k=top_k,
        faiss_index_path=getattr(args, "faiss_index", None),
    )
    search_ms = (time.perf_counter() - t0) * 1000

    print(f"Top {top_k} nearest tracks  (search: {search_ms:.1f} ms)")
    print("─" * 56)
    print(f"{'Rank':>4}  {'Score':>6}  Track ID")
    print("─" * 56)
    for rank, (tid, score) in enumerate(results, 1):
        print(f"{rank:>4}  {score:>6.4f}  {tid}")

    qt = getattr(args, "query_track", None)
    if qt:
        id_list = list(ids)
        if qt in id_list:
            idx = id_list.index(qt)
            emb = embeddings[idx]
            q_n = query / (np.linalg.norm(query) + 1e-8)
            e_n = emb / (np.linalg.norm(emb) + 1e-8)
            score = float(np.dot(q_n, e_n))
            all_scores = embeddings @ q_n
            rank = int((all_scores > score).sum()) + 1
            print(f"\nTrack {qt}: score={score:.4f}, rank={rank:,}/{len(ids):,}")
        else:
            print(f"\nTrack {qt!r} not in index")

    print(f"\nTotal: {query_ms + search_ms:.1f} ms")
