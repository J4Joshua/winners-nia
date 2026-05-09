"""Build user query vector (128-d) from demo profile, JSON, or random."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.inference.demo_users import DEMO_USERS
from ml.inference.loaders import load_user_tower
from ml.models.user_tower import UserTower


def build_user_query_embedding(
    args: Namespace,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    """Returns L2-normalized query vector (128,) and a label for logging."""

    if getattr(args, "user_json", None):
        data = json.loads(Path(args.user_json).read_text())
        features = data.get("features", data)
        label = str(data.get("name", args.user_json))
    elif getattr(args, "demo_user", None):
        profile = DEMO_USERS[args.demo_user]
        features = profile["features"]
        label = str(profile["name"])
    else:
        rng = np.random.default_rng()
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        return v, "random unit vector"

    feat_list = list(features)
    if len(feat_list) != 17:
        raise ValueError(f"Expected 17-d user features, got {len(feat_list)}")

    x = torch.tensor([feat_list], dtype=torch.float32, device=device)

    user_model: nn.Module
    if getattr(args, "user_tower", None):
        user_model = load_user_tower(args.user_tower, device)
    else:
        print("  (no --user-tower: random UserTower weights — for taste alignment, train User Tower)")
        user_model = UserTower().to(device)
        user_model.eval()

    with torch.no_grad():
        q = user_model(x).cpu().numpy()[0]

    return q.astype(np.float32), label
