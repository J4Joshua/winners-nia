"""Song Tower: residual MLP 45 → 512 ⊕ res → 256 → 128, L2-normalized.

Input layout (45-d):
  [0:12]  key one-hot (12 semitones)
  [12:26] 14 audio scalars (danceability, energy, speechiness, acousticness,
           instrumentalness, liveness, valence, loudness_norm, tempo_norm,
           mode, explicit, popularity_norm, duration_norm, time_sig_norm)
  [26:45] 19-d macro-genre one-hot

Training:
  SupCon   — all same-genre in batch are positives (two augmented views)
  Uniformity — spreads embeddings across the hypersphere (Wang & Isola 2020)
  Prototype  — pulls embeddings toward their per-genre centroid (alignment)
  τ-anneal   — temperature 0.15→0.05 over training (coarse then fine structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.dataset import SONG_FEATURE_DIM  # 45

EMBEDDING_DIM = 128

# Slice boundaries for feature augmentation
_AUDIO_START = 12   # first audio scalar index
_AUDIO_END   = 26   # first genre one-hot index


class _ResidualBlock(nn.Module):
    """
    Pre-activation residual block: x → BN → GELU → Linear → BN → GELU → Linear → + x

    Residual connections allow deeper effective training — gradients flow directly
    to early layers, preventing the vanishing gradient problem that flat MLPs suffer
    beyond 3 layers.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SongTower(nn.Module):
    """
    Maps a 45-d song feature vector to a 128-d L2-normalized embedding.

    Architecture:
      45 → Linear(512) → ResidualBlock(512) → ResidualBlock(512)
         → Linear(256) → GELU → Linear(128) → L2-norm

    Two residual blocks give effective depth-5 behaviour with the gradient flow
    of a depth-2 network — key for stable training on mixed sparse+dense inputs.
    """

    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(SONG_FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        self.res1 = _ResidualBlock(512, dropout)
        self.res2 = _ResidualBlock(512, dropout)
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, EMBEDDING_DIM),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.res1(h)
        h = self.res2(h)
        return F.normalize(self.proj_out(h), dim=-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time embedding (eval mode, no dropout / batch noise)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_features(x: torch.Tensor, noise_std: float = 0.04) -> torch.Tensor:
    """
    Add Gaussian noise to audio scalar features only (indices 12-25), clip to [0,1].
    Key one-hot and genre one-hot are left untouched — they are categorical.
    """
    out = x.clone()
    noise = torch.randn(x.size(0), _AUDIO_END - _AUDIO_START, device=x.device) * noise_std
    out[:, _AUDIO_START:_AUDIO_END] = (x[:, _AUDIO_START:_AUDIO_END] + noise).clamp(0.0, 1.0)
    return out


# ── Losses ────────────────────────────────────────────────────────────────────

def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al. 2020).

    z      : (N, D) L2-normalized embeddings (two views of B samples → N = 2B)
    labels : (N,)   integer genre labels, repeated twice [l1…lB, l1…lB]

    For each anchor z_i, positives = all z_j with labels[j] == labels[i], j ≠ i.
    Negatives = everything else in the batch.

    Loss = mean over anchors of:
        -1/|P_i| * sum_{p in P_i} log [exp(z_i·z_p/τ) / sum_{a≠i} exp(z_i·z_a/τ)]

    With a batch of 512 and 19 genres, each anchor sees ~50 positives on average —
    orders of magnitude richer supervision than single-pair InfoNCE.
    """
    n = z.size(0)
    device = z.device

    sim = z @ z.T / temperature  # (N, N)

    # Numerical stability: subtract row-max before softmax
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask   = torch.eye(n, dtype=torch.bool, device=device)
    same_label  = labels.unsqueeze(0) == labels.unsqueeze(1)     # (N, N)
    pos_mask    = same_label & ~self_mask                        # positives (same genre, not self)

    # Log-softmax denominator over all non-self
    exp_sim = torch.exp(sim)
    exp_no_self = exp_sim.masked_fill(self_mask, 0.0)
    log_denom = torch.log(exp_no_self.sum(dim=1, keepdim=True) + 1e-9)  # (N, 1)

    log_prob = sim - log_denom  # (N, N)

    n_pos = pos_mask.float().sum(dim=1)   # (N,) — how many positives each anchor has
    valid = n_pos > 0

    if valid.sum() == 0:
        # Extremely rare: batch landed only one sample per genre.
        # Fall back to standard InfoNCE on the diagonal pairs.
        half = n // 2
        idx = torch.arange(half, device=device)
        labels_diag = torch.cat([idx, idx])
        return F.cross_entropy(sim * temperature, labels_diag)

    per_anchor = -(log_prob * pos_mask.float()).sum(dim=1)  # (N,)
    per_anchor = per_anchor[valid] / n_pos[valid]
    return per_anchor.mean()


def prototype_alignment_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Prototype alignment loss: pull each embedding toward its genre centroid.

    For each genre, compute the mean embedding (prototype) from the current batch.
    Then minimize cosine distance from each sample to its prototype.

    This complements SupCon (which repels wrong-genre pairs) by explicitly
    anchoring each genre cluster around a stable centroid, reducing intra-genre
    variance and making the embedding space more organised.

    Combined loss: L = SupCon + 0.5·Uniformity + 0.3·Prototype
    """
    device = z.device
    unique_labels = labels.unique()
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() < 2:
            continue
        prototype = F.normalize(z[mask].mean(0, keepdim=True), dim=-1)  # (1, D)
        # cosine distance = 1 - cosine_similarity (z is already L2-normalised)
        total_loss = total_loss + (1.0 - (z[mask] * prototype).sum(dim=-1)).mean()
        count += 1

    return total_loss / max(1, count)


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss (Wang & Isola 2020).

    Encourages embeddings to spread uniformly over the hypersphere, preventing
    mode collapse where all embeddings cluster in a small region.

    loss = log E[exp(-t * ||z_i - z_j||²)]   (lower = more uniform)

    Combined with SupCon: total_loss = supcon + λ * uniformity
    Typical λ = 0.5-1.0. Larger λ = more spread, potentially less genre clustering.
    """
    sq_dists = torch.cdist(z, z, p=2).pow(2)   # (N, N)
    return torch.log(torch.exp(-t * sq_dists).mean() + 1e-9)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Standard InfoNCE (kept for evaluation / fallback use)."""
    logits = anchor @ positive.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
