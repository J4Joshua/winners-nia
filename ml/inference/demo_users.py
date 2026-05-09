"""Built-in demo user profiles (17-d features_user_v1)."""

from __future__ import annotations

DEMO_USERS: dict[str, dict[str, object]] = {
    "pop": {
        "name": "Pop listener",
        "features": [
            0.72, 0.70, 0.60, 0.15, 0.02, 0.52, 0.70, 0.30, 0.60, 0.25,
            0.15, 0.75, 0.40, 8.0, 0.50, 0.87, 0.60,
        ],
    },
    "chill": {
        "name": "Chill / acoustic listener",
        "features": [
            0.45, 0.35, 0.50, 0.65, 0.20, 0.35, 0.50, 0.45, 0.40, 0.15,
            0.25, 0.85, 0.30, 6.0, -0.71, -0.71, 0.80,
        ],
    },
    "hype": {
        "name": "High energy / EDM listener",
        "features": [
            0.88, 0.92, 0.65, 0.05, 0.35, 0.83, 0.85, 0.25, 0.75, 0.35,
            0.10, 0.65, 0.55, 12.0, 0.98, 0.20, 0.50,
        ],
    },
}
