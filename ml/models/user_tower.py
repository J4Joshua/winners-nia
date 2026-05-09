"""User Tower: MLP 17 → 256 → 256 → 128, L2-normalized embedding (per-user, online-updated).

Uses LayerNorm + GELU instead of BatchNorm + ReLU:
  - LayerNorm works correctly at batch_size=1 (single-user inference & online updates)
  - GELU gives smoother gradient flow for fine-grained preference optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


USER_FEATURE_DIM = 17
EMBEDDING_DIM = 128


class UserTower(nn.Module):
    """
    Maps a 17-d user feature vector to a 128-d L2-normalized embedding.

    Architecture matches Song Tower embedding dim so both towers share a space.
    Trained cold on user's Spotify history with ranking loss; fine-tuned online
    (1 Adam step per like/dislike event).

    LayerNorm instead of BatchNorm: correct semantics at batch_size=1 (used
    during both online updates and single-user inference in the TUI session).
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(USER_FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, EMBEDDING_DIM),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
