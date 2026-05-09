"""
Cold-start User Tower: align 17-d profile embedding to the centroid of your
positive tracks (top tracks from Spotify) in Song Tower space.

Requires user JSON from `ml spotify` with top_track_ids, plus catalog embeddings.

    python -m ml train-user \\
        --user-json ml/cli/my_user.json \\
        --output ml/export/my_user_tower.pt

Uses embeddings from Hugging Face by default (MrlolDev/attune-v0) or local paths.
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


def resolve_positive_ids(profile: dict, prefer: str) -> list[str]:
    if prefer == "top":
        return [str(x) for x in profile.get("top_track_ids") or []]
    if prefer == "recent":
        return [str(x) for x in profile.get("recent_track_ids") or []]
    combined = list(profile.get("top_track_ids") or []) + list(profile.get("recent_track_ids") or [])
    out: list[str] = []
    seen: set[str] = set()
    for x in combined:
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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

    if args.embeddings and args.ids:
        embeddings = np.load(args.embeddings).astype(np.float32)
        catalog_ids = np.load(args.ids, allow_pickle=True)
    else:
        repo = args.hub_repo or DEFAULT_HUB_MODEL_REPO
        print(f"Loading catalog embeddings from Hugging Face: {repo}")
        embeddings, catalog_ids = load_catalog_from_hub(repo)

    id_to_row: dict[str, int] = {}
    for i, rid in enumerate(catalog_ids.tolist() if hasattr(catalog_ids, "tolist") else catalog_ids):
        id_to_row[str(rid)] = i

    rows: list[int] = []
    missing: list[str] = []
    for tid in positive_ids:
        row = id_to_row.get(str(tid))
        if row is not None:
            rows.append(row)
        else:
            missing.append(tid)

    if missing:
        print(f"Warning: {len(missing)} Spotify IDs not in HF catalog (skipped). Examples: {missing[:5]}")

    if len(rows) < args.min_positives:
        raise SystemExit(
            f"Only {len(rows)} tracks matched the catalog (need {args.min_positives}). "
            "Dataset may use different IDs than Spotify — try training Song Tower on data that includes your markets."
        )

    sel = embeddings[rows]
    target = sel.mean(axis=0)
    target = target / (np.linalg.norm(target) + 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.tensor([feats], dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(args.seed)
    model = UserTower(dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = 1.0 - F.cosine_similarity(pred, target_t, dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            with torch.no_grad():
                sim = F.cosine_similarity(model(x), target_t, dim=-1).item()
            print(f"epoch {epoch:4d}  loss={loss.item():.5f}  cosine(pred,target)={sim:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict()}, out_path)
    print(f"\nSaved User Tower → {out_path}")
    print("\nRun retrieval:")
    print(f"  python -m ml run --user-json {profile_path} --user-tower {out_path}")


def parse_train_user_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cold-start User Tower from Spotify profile + Song embeddings")
    p.add_argument("--user-json", required=True, help="Output of `ml spotify` (needs top_track_ids)")
    p.add_argument("--output", default="ml/export/my_user_tower.pt")
    p.add_argument("--hub-repo", default=None, help="HF model repo with song_embeddings_v1.npy + song_ids_v1.npy")
    p.add_argument("--embeddings", default=None, help="Local song_embeddings_v1.npy (optional)")
    p.add_argument("--ids", default=None, help="Local song_ids_v1.npy (optional)")
    p.add_argument("--positives", choices=("top", "recent", "both"), default="top", help="Which Spotify IDs supervise")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-positives", type=int, default=5)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args(argv)


def main() -> None:
    train_user_cold(parse_train_user_args())


if __name__ == "__main__":
    main()
