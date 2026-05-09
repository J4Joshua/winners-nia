"""Song Tower: MLP 26 → 256 → 128, L2-normalized embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


SONG_FEATURE_DIM = 26
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


class SongTower(nn.Module):
    """
    Maps a 26-d song feature vector to a 128-d L2-normalized embedding.

    Architecture: Linear(26, 256) → ReLU → Dropout → Linear(256, 128) → L2-norm

    The dropout is active only during training (augmentation for InfoNCE pairs).
    Running the same input twice during train gives two different views — that's
    how we generate (anchor, positive) pairs without needing metadata labels.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SONG_FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 26) → (B, 128) L2-normalized
        return F.normalize(self.net(x), dim=-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time embedding (no dropout)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent loss with in-batch negatives.

    Both anchor and positive are already L2-normalized (B, 128).
    Logits shape: (B, B). Labels = diagonal (each row's positive is its own index).
    """
    logits = anchor @ positive.T / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    # Symmetric: treat both directions
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2
