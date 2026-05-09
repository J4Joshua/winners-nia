"""
Cold-start User Tower: align the user embedding to their positive tracks in Song
Tower space using a ranking loss — not just a centroid regression.

Ranking loss: for each mini-batch of P positive embeddings and N random negatives,
push the user embedding closer to every positive and away from every negative.
This prevents the "average-everything" collapse of centroid regression and gives
the tower a real understanding of preference vs. non-preference.

    python -m ml train-user \\
        --user-json ml/cli/my_user.json \\
        --output ml/export/my_user_tower.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.inference.defaults import DEFAULT_HUB_MODEL_REPO
from ml.models.user_tower import USER_FEATURE_DIM, UserTower


def load_catalog_from_hub(repo_id: str) -> tuple[np.ndarray, np.ndarray]:
    from huggingface_hub import hf_hub_download  # type: ignore

    emb_path = hf_hub_download(repo_id, "song_embeddings_v1.npy")
    ids_path = hf_hub_download(repo_id, "song_ids_v1.npy")
    embeddings = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path, allow_pickle=True)
    return embeddings, ids


def _recency_weight(added_at: str, now: "datetime | None" = None) -> float:
    """Convert an ISO-8601 added_at string to a recency weight [0.2, 1.0]."""
    from datetime import datetime, timezone, timedelta
    if not added_at:
        return 0.5
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        ts = datetime.fromisoformat(added_at.replace("Z", "+00:00"))
        age_days = max(0, (now - ts).days)
    except (ValueError, TypeError):
        return 0.5
    if age_days <= 30:   return 1.0
    if age_days <= 90:   return 0.8
    if age_days <= 365:  return 0.5
    return 0.2


def resolve_positive_ids(profile: dict, prefer: str) -> list[str]:
    """Return a deduplicated list of positive track IDs from the profile."""
    if prefer == "top":
        return [str(x) for x in profile.get("top_track_ids") or []]
    if prefer == "recent":
        return [str(x) for x in profile.get("recent_track_ids") or []]
    if prefer == "liked":
        return [str(x) for x in profile.get("liked_track_ids") or []]

    # "all": top + recent + liked + playlist tracks, deduplicated
    sources = (
        list(profile.get("top_track_ids") or [])
        + list(profile.get("recent_track_ids") or [])
        + list(profile.get("liked_track_ids") or [])
    )
    for pl in profile.get("playlists") or []:
        for item in pl.get("tracks") or []:
            tid = item.get("track_id") if isinstance(item, dict) else str(item)
            if tid:
                sources.append(str(tid))

    out: list[str] = []
    seen: set[str] = set()
    for x in sources:
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_positive_weights(profile: dict, positive_ids: list[str]) -> np.ndarray:
    """
    Return per-positive recency weights for ranking loss weighting.

    top_track_ids:    1.0  (Spotify's own relevance-ranked signal)
    recent_track_ids: 0.9  (very recent activity)
    liked_track_ids:  0.7  (library save, less time-sensitive)
    playlist tracks:  0.2–1.0 based on added_at date
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    top_set    = set(str(x) for x in profile.get("top_track_ids") or [])
    recent_set = set(str(x) for x in profile.get("recent_track_ids") or [])
    liked_set  = set(str(x) for x in profile.get("liked_track_ids") or [])

    # Build playlist track → best recency weight mapping
    pl_weights: dict[str, float] = {}
    for pl in profile.get("playlists") or []:
        for item in pl.get("tracks") or []:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("track_id", ""))
            w = _recency_weight(item.get("added_at", ""), now)
            if tid and w > pl_weights.get(tid, 0):
                pl_weights[tid] = w

    weights = []
    for tid in positive_ids:
        if tid in top_set:
            weights.append(1.0)
        elif tid in recent_set:
            weights.append(0.9)
        elif tid in liked_set:
            weights.append(0.7)
        else:
            weights.append(pl_weights.get(tid, 0.5))

    arr = np.array(weights, dtype=np.float32)
    return arr / arr.sum()  # normalise to sum=1


def _ranking_loss(
    user_emb: torch.Tensor,    # (1, 128) L2-normalized
    pos_embs: torch.Tensor,    # (P, 128) L2-normalized
    neg_embs: torch.Tensor,    # (N, 128) L2-normalized
    pos_weights: torch.Tensor, # (P,) summing to 1 — recency-based importance weights
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Weighted multi-positive InfoNCE.

    Loss = -sum_p w_p * log [exp(u·p/tau) / (sum_p' exp + sum_n exp)]

    Weights encode recency: recently-added playlist tracks, top tracks, and
    recent plays matter more than tracks added years ago to the library.
    """
    u = user_emb  # (1, 128)
    pos_sims = (u @ pos_embs.T).squeeze(0) / temperature   # (P,)
    neg_sims = (u @ neg_embs.T).squeeze(0) / temperature   # (N,)
    all_sims = torch.cat([pos_sims, neg_sims])              # (P+N,)
    log_denom = torch.logsumexp(all_sims, dim=0)
    # Weighted average of per-positive log-probs
    loss = -(pos_weights * (pos_sims - log_denom)).sum()
    return loss


def train_user_cold(args: argparse.Namespace) -> None:
    profile_path = Path(args.user_json)
    profile = json.loads(profile_path.read_text())
    feats = profile.get("features")
    if not feats or len(feats) != USER_FEATURE_DIM:
        raise SystemExit(f"user-json must contain features list of length {USER_FEATURE_DIM}")

    positive_ids = resolve_positive_ids(profile, args.positives)
    if len(positive_ids) < args.min_positives:
        raise SystemExit(
            f"Need at least {args.min_positives} track IDs in profile (top_track_ids / recent). "
            "Re-run: python -m ml spotify --token ... --out ml/cli/my_user.json"
        )

    if getattr(args, "embeddings", None) and getattr(args, "ids", None):
        embeddings = np.load(args.embeddings).astype(np.float32)
        catalog_ids = np.load(args.ids, allow_pickle=True)
    else:
        repo = getattr(args, "hub_repo", None) or DEFAULT_HUB_MODEL_REPO
        print(f"Loading catalog embeddings from Hugging Face: {repo}")
        embeddings, catalog_ids = load_catalog_from_hub(repo)

    id_to_row: dict[str, int] = {
        str(rid): i
        for i, rid in enumerate(
            catalog_ids.tolist() if hasattr(catalog_ids, "tolist") else catalog_ids
        )
    }

    rows: list[int] = []
    matched_ids: list[str] = []
    missing: list[str] = []
    for tid in positive_ids:
        row = id_to_row.get(str(tid))
        if row is not None:
            rows.append(row)
            matched_ids.append(str(tid))
        else:
            missing.append(tid)

    if missing:
        print(f"Warning: {len(missing)} Spotify IDs not in catalog (skipped). Examples: {missing[:5]}")

    if len(rows) < args.min_positives:
        raise SystemExit(
            f"Only {len(rows)} tracks matched the catalog (need {args.min_positives}). "
            "Dataset may use different IDs — try retraining Song Tower with your market's data."
        )

    pos_embs_np = embeddings[rows]                              # (P, 128)
    pos_weights = resolve_positive_weights(profile, matched_ids)  # (P,) summing to 1

    # Negative pool: all catalog rows NOT in the positive set
    pos_set = set(rows)
    all_rows = np.arange(len(embeddings))
    neg_pool = all_rows[~np.isin(all_rows, list(pos_set))]  # (N_total,)

    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu")
    x           = torch.tensor([feats],       dtype=torch.float32, device=device)  # (1, 17)
    pos_embs_t  = torch.tensor(pos_embs_np,  dtype=torch.float32, device=device)  # (P, 128)
    pos_weights_t = torch.tensor(pos_weights, dtype=torch.float32, device=device)  # (P,)

    torch.manual_seed(args.seed)
    model = UserTower(dropout=0.0).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine LR decay (no warmup needed; tiny network converges fast)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    n_neg_per_step = min(512, len(neg_pool))   # sample this many negatives each step
    rng = np.random.default_rng(args.seed)

    best_loss = float("inf")
    best_state: dict | None = None

    model.train()
    print(f"Training User Tower: {len(rows)} positives, {len(neg_pool):,} neg pool, {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        user_emb = model(x)  # (1, 128)

        # Fresh random negatives every step — avoids memorisation
        neg_idx = rng.choice(neg_pool, size=n_neg_per_step, replace=False)
        neg_embs_t = torch.tensor(
            embeddings[neg_idx], dtype=torch.float32, device=device
        )

        loss = _ranking_loss(user_emb, pos_embs_t, neg_embs_t, pos_weights_t, temperature=0.07)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % max(1, args.epochs // 8) == 0 or epoch == args.epochs:
            with torch.no_grad():
                # Report mean cosine similarity to all positives
                sim = F.cosine_similarity(model(x), pos_embs_t, dim=-1).mean()
            lr_now = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:5d}  loss={loss.item():.5f}  mean_cos={sim.item():.4f}  lr={lr_now:.1e}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict()}, out_path)
    print(f"\nSaved User Tower → {out_path}  (best loss {best_loss:.5f})")
    print("\nRun retrieval:")
    print(f"  python -m ml run --user-json {profile_path} --user-tower {out_path}")


def parse_train_user_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cold-start User Tower from Spotify profile + Song embeddings")
    p.add_argument("--user-json",   required=True, help="Output of `ml spotify` (needs top_track_ids)")
    p.add_argument("--output",      default="ml/export/my_user_tower.pt")
    p.add_argument("--hub-repo",    default=None,  help="HF model repo with song_embeddings_v1.npy + song_ids_v1.npy")
    p.add_argument("--embeddings",  default=None,  help="Local song_embeddings_v1.npy (optional)")
    p.add_argument("--ids",         default=None,  help="Local song_ids_v1.npy (optional)")
    p.add_argument("--positives",   choices=("top", "recent", "liked", "all"), default="all")
    p.add_argument("--epochs",      type=int,   default=2000)
    p.add_argument("--lr",          type=float, default=5e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--min-positives", type=int, default=5)
    p.add_argument("--cpu",         action="store_true", help="Force CPU")
    return p.parse_args(argv)


def main() -> None:
    train_user_cold(parse_train_user_args())


if __name__ == "__main__":
    main()
