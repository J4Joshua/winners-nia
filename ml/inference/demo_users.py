"""Built-in demo / vibe profiles (17-d features_user_v1).

Feature order
─────────────
 0  mean_danceability       1  mean_energy           2  mean_valence
 3  mean_acousticness       4  mean_instrumentalness 5  mean_tempo_norm
 6  mean_loudness_norm      7  genre_diversity       8  listening_recency
 9  skip_rate_30d          10  replay_rate_30d      11  avg_listen_ratio
12  discovery_ratio        13  session_length_avg   14  peak_hour_sin
15  peak_hour_cos          16  account_age_norm
"""

from __future__ import annotations

DEMO_USERS: dict[str, dict[str, object]] = {
    # ── genre-based ──────────────────────────────────────────────────────────
    "pop": {
        "name": "Pop listener",
        "features": [
            0.72, 0.70, 0.60, 0.15, 0.02, 0.52,   # dance energy valence acous instr tempo
            0.70, 0.30, 0.60, 0.25, 0.15, 0.75,   # loud genre recency skip replay ratio
            0.40, 8.0,  0.50, 0.87, 0.60,          # discovery session peak_sin peak_cos age
        ],
    },
    "hype": {
        "name": "High energy / EDM",
        "features": [
            0.88, 0.92, 0.65, 0.05, 0.35, 0.83,
            0.85, 0.25, 0.75, 0.35, 0.10, 0.65,
            0.55, 12.0, 0.98, 0.20, 0.50,
        ],
    },
    "rap": {
        "name": "Hip-hop / rap",
        "features": [
            0.82, 0.75, 0.55, 0.10, 0.03, 0.58,
            0.75, 0.20, 0.65, 0.30, 0.20, 0.72,
            0.35, 9.0,  0.71, 0.71, 0.55,
        ],
    },
    "rnb": {
        "name": "R&B / soul",
        "features": [
            0.70, 0.60, 0.65, 0.25, 0.05, 0.48,
            0.60, 0.20, 0.60, 0.20, 0.25, 0.80,
            0.30, 7.0,  0.71, -0.71, 0.65,
        ],
    },
    "rock": {
        "name": "Rock listener",
        "features": [
            0.55, 0.80, 0.50, 0.15, 0.10, 0.65,
            0.80, 0.30, 0.55, 0.35, 0.15, 0.70,
            0.40, 8.0,  0.00, 1.00, 0.70,
        ],
    },
    "metal": {
        "name": "Metal / heavy",
        "features": [
            0.40, 0.95, 0.30, 0.10, 0.15, 0.78,
            0.90, 0.25, 0.50, 0.40, 0.20, 0.75,
            0.35, 9.0, -0.50, 0.87, 0.65,
        ],
    },
    "jazz": {
        "name": "Jazz / blues",
        "features": [
            0.55, 0.45, 0.55, 0.40, 0.30, 0.42,
            0.45, 0.55, 0.50, 0.15, 0.30, 0.85,
            0.45, 7.0,  0.00, -1.0, 0.80,
        ],
    },
    "classical": {
        "name": "Classical / orchestral",
        "features": [
            0.25, 0.35, 0.45, 0.70, 0.75, 0.38,
            0.30, 0.35, 0.55, 0.10, 0.35, 0.90,
            0.30, 6.0,  0.00, -1.0, 0.85,
        ],
    },

    # ── mood / vibe ───────────────────────────────────────────────────────────
    "chill": {
        "name": "Chill / laid-back",
        "features": [
            0.45, 0.35, 0.50, 0.65, 0.20, 0.35,
            0.45, 0.45, 0.40, 0.15, 0.25, 0.85,
            0.30, 6.0, -0.71, -0.71, 0.80,
        ],
    },
    "sad": {
        "name": "Sad / melancholic",
        "features": [
            0.35, 0.30, 0.15, 0.55, 0.15, 0.32,
            0.35, 0.30, 0.50, 0.20, 0.30, 0.80,
            0.25, 5.0, -1.00,  0.00, 0.70,
        ],
    },
    "happy": {
        "name": "Happy / upbeat",
        "features": [
            0.75, 0.72, 0.88, 0.20, 0.04, 0.55,
            0.68, 0.35, 0.65, 0.20, 0.20, 0.78,
            0.45, 8.0,  0.87,  0.50, 0.60,
        ],
    },
    "focus": {
        "name": "Focus / deep work",
        "features": [
            0.40, 0.45, 0.40, 0.50, 0.65, 0.40,
            0.38, 0.40, 0.55, 0.10, 0.40, 0.90,
            0.50, 4.0,  0.00,  1.00, 0.75,
        ],
    },
    "workout": {
        "name": "Workout / gym",
        "features": [
            0.80, 0.90, 0.60, 0.08, 0.08, 0.78,
            0.85, 0.20, 0.70, 0.30, 0.15, 0.70,
            0.40, 10.0, 0.87, -0.50, 0.55,
        ],
    },
    "late-night": {
        "name": "Late night drive",
        "features": [
            0.50, 0.40, 0.35, 0.40, 0.25, 0.42,
            0.42, 0.35, 0.45, 0.20, 0.25, 0.80,
            0.35, 7.0, -0.87, -0.50, 0.70,
        ],
    },
    "rainy": {
        "name": "Rainy day",
        "features": [
            0.38, 0.32, 0.25, 0.70, 0.20, 0.30,
            0.30, 0.40, 0.50, 0.15, 0.35, 0.85,
            0.30, 5.0, -0.71, -0.71, 0.75,
        ],
    },
    "party": {
        "name": "Party / dance floor",
        "features": [
            0.90, 0.85, 0.70, 0.05, 0.05, 0.75,
            0.82, 0.25, 0.80, 0.25, 0.10, 0.65,
            0.50, 14.0, 0.50, 0.87, 0.50,
        ],
    },
    "morning": {
        "name": "Morning / sunrise",
        "features": [
            0.60, 0.55, 0.70, 0.35, 0.10, 0.48,
            0.52, 0.40, 0.60, 0.15, 0.20, 0.80,
            0.40, 5.0,  1.00,  0.00, 0.65,
        ],
    },
    "sleep": {
        "name": "Sleep / ambient",
        "features": [
            0.20, 0.15, 0.35, 0.80, 0.80, 0.22,
            0.15, 0.30, 0.50, 0.05, 0.50, 0.95,
            0.25, 3.0, -1.00,  0.00, 0.80,
        ],
    },
}
