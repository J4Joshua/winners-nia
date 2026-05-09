"""
Online User Tower update — one Adam step per like/dislike signal.

The User Tower weights are the only thing updated; the 17-d input features
stay fixed.  A small LR (1e-4) prevents catastrophic forgetting over a
single session while still visibly shifting recommendations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def online_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    user_features: torch.Tensor,   # (1, 17)
    track_embedding: torch.Tensor, # (1, 128) — from catalog, already L2-normed
    like: bool,
    steps: int = 1,
    clip_norm: float = 0.5,
) -> float:
    """
    Push the user embedding toward (like=True) or away from (like=False)
    the target track embedding.  Returns cosine similarity after update.
    """
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(user_features)                              # (1, 128)
        sim = F.cosine_similarity(pred, track_embedding, dim=-1) # (1,)
        loss = (1.0 - sim).mean() if like else sim.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_after = model(user_features)
        sim_after = F.cosine_similarity(pred_after, track_embedding, dim=-1)
    return float(sim_after.mean())
