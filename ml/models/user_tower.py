"""User Tower: MLP 17 → 256 → 128, L2-normalized embedding (per-user, online-updated)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


USER_FEATURE_DIM = 17
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


class UserTower(nn.Module):
    """
    Maps a 17-d user feature vector to a 128-d L2-normalized embedding.

    Architecture matches Song Tower so embeddings live in the same space.
    Trained cold on user's Spotify history; fine-tuned online (1 Adam step per event).
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(USER_FEATURE_DIM, HIDDEN_DIM),
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
        # x: (B, 17) → (B, 128) L2-normalized
        return F.normalize(self.net(x), dim=-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
