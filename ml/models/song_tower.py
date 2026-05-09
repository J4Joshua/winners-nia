"""Song Tower: MLP 45 → 512 → 256 → 128, L2-normalized embedding.

Input layout (45-d):
  [0:12]  key one-hot (12 semitones)
  [12:26] 14 audio scalars (danceability, energy, speechiness, acousticness,
           instrumentalness, liveness, valence, loudness_norm, tempo_norm,
           mode, explicit, popularity_norm, duration_norm, time_sig_norm)
  [26:45] 19-d macro-genre one-hot

Training uses Supervised Contrastive Loss (SupCon) — all same-genre tracks in
a batch are treated as positives. Two augmented views are generated per sample
(Gaussian noise on audio scalars + dropout) to further enrich the signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.dataset import SONG_FEATURE_DIM  # 45

EMBEDDING_DIM = 128

# Slice boundaries for feature augmentation
_AUDIO_START = 12   # first audio scalar index
_AUDIO_END   = 26   # first genre one-hot index


class SongTower(nn.Module):
    """
    Maps a 45-d song feature vector to a 128-d L2-normalized embedding.

    Architecture: 45 → 512 (BN+ReLU+Drop) → 256 (BN+ReLU+Drop) → 128 → L2-norm

    BatchNorm stabilises training with mixed feature types (sparse one-hot +
    dense audio scalars). Dropout + Gaussian input noise create two distinct
    augmented views from the same raw features, which SupCon exploits.
    """

    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SONG_FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, EMBEDDING_DIM),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

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


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Standard InfoNCE (kept for evaluation / fallback use)."""
    logits = anchor @ positive.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
