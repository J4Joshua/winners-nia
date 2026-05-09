"""
Fetch Spotify listening data and write a 17-d user feature JSON.

Prefer:

    python -m ml spotify --token ... --out ml/cli/my_user.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def compute_user_features(
    top_tracks: list[dict],
    audio_features: list[dict],
    recent_tracks: list[dict],
    display_name: str = "",
) -> dict:
    valid_af = [af for af in audio_features if af and af.get("danceability") is not None]

    def mean_af(field: str) -> float:
        vals = [af[field] for af in valid_af if field in af]
        return sum(vals) / len(vals) if vals else 0.5

    mean_danceability = mean_af("danceability")
    mean_energy = mean_af("energy")
    mean_valence = mean_af("valence")
    mean_acousticness = mean_af("acousticness")
    mean_instrumentalness = mean_af("instrumentalness")
    mean_tempo_norm = min(mean_af("tempo") / 240.0, 1.0)
    loudness_vals = [af["loudness"] for af in valid_af if "loudness" in af]
    mean_loudness_norm = (
        sum(min(max((l + 60.0) / 60.0, 0.0), 1.0) for l in loudness_vals) / len(loudness_vals)
        if loudness_vals
        else 0.5
    )

    genre_diversity = 0.4

    hour_counts: Counter[int] = Counter()
    unique_tracks_7d: set[str] = set()
    all_recent_ids: list[str] = []

    now = datetime.now(timezone.utc)
    cutoff_7d = now - timedelta(days=7)

    for item in recent_tracks:
        track = item.get("track", {})
        tid = track.get("id", "")
        played_at_str = item.get("played_at", "")

        try:
            played_at = datetime.fromisoformat(played_at_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue

        hour_counts[played_at.hour] += 1
        all_recent_ids.append(tid)
        if played_at >= cutoff_7d:
            unique_tracks_7d.add(tid)

    total_recent = len(all_recent_ids)
    listening_recency = len(unique_tracks_7d) / max(total_recent, 1)
    discovery_ratio = len(set(all_recent_ids)) / max(total_recent, 1)

    peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 15
    peak_hour_sin = math.sin(2 * math.pi * peak_hour / 24)
    peak_hour_cos = math.cos(2 * math.pi * peak_hour / 24)

    session_lengths: list[int] = []
    if recent_tracks:
        session_len = 1
        timestamps: list[datetime] = []
        for item in recent_tracks:
            try:
                ts = datetime.fromisoformat(item["played_at"].replace("Z", "+00:00"))
                timestamps.append(ts)
            except (KeyError, TypeError, ValueError):
                timestamps.append(now)

        for i in range(1, len(timestamps)):
            gap = abs((timestamps[i - 1] - timestamps[i]).total_seconds())
            if gap < 1800:
                session_len += 1
            else:
                session_lengths.append(session_len)
                session_len = 1
        session_lengths.append(session_len)

    session_length_avg = sum(session_lengths) / len(session_lengths) if session_lengths else 5.0

    skip_rate_30d = 0.5
    replay_rate_30d = 0.1
    avg_listen_ratio = 0.75
    account_age_norm = 0.5

    features = [
        mean_danceability,
        mean_energy,
        mean_valence,
        mean_acousticness,
        mean_instrumentalness,
        mean_tempo_norm,
        mean_loudness_norm,
        genre_diversity,
        listening_recency,
        skip_rate_30d,
        replay_rate_30d,
        avg_listen_ratio,
        discovery_ratio,
        session_length_avg,
        peak_hour_sin,
        peak_hour_cos,
        account_age_norm,
    ]

    top_track_ids = [str(t.get("id")) for t in top_tracks if t.get("id")]
    recent_track_ids: list[str] = []
    seen_recent: set[str] = set()
    for item in recent_tracks:
        tr = item.get("track") or {}
        tid = tr.get("id") if isinstance(tr, dict) else None
        if tid and str(tid) not in seen_recent:
            seen_recent.add(str(tid))
            recent_track_ids.append(str(tid))

    return {
        "name": display_name or "Spotify User",
        "features": features,
        "top_track_ids": top_track_ids,
        "recent_track_ids": recent_track_ids,
        "debug": {
            "mean_danceability": round(mean_danceability, 3),
            "mean_energy": round(mean_energy, 3),
            "mean_valence": round(mean_valence, 3),
            "mean_acousticness": round(mean_acousticness, 3),
            "mean_instrumentalness": round(mean_instrumentalness, 3),
            "mean_tempo_bpm": round(mean_tempo_norm * 240, 1),
            "peak_hour": peak_hour,
            "session_length_avg": round(session_length_avg, 1),
            "discovery_ratio": round(discovery_ratio, 3),
            "top_tracks_used": len(valid_af),
            "recent_tracks_used": total_recent,
        },
    }


def fetch_with_token(token: str) -> dict:
    import json as json_lib
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"}

    def get(url: str) -> dict | list:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as r:
            return json_lib.loads(r.read())

    print("Fetching /me ...")
    me = get("https://api.spotify.com/v1/me")
    display_name = me.get("display_name", "")
    print(f"  → Logged in as: {display_name}")

    print("Fetching top tracks (medium_term, 50) ...")
    top_resp = get("https://api.spotify.com/v1/me/top/tracks?time_range=medium_term&limit=50")
    top_tracks = top_resp.get("items", [])
    print(f"  → {len(top_tracks)} tracks")

    track_ids = [t["id"] for t in top_tracks if t.get("id")]
    print(f"Fetching audio features for {len(track_ids)} tracks ...")
    ids_str = ",".join(track_ids[:50])
    af_resp = get(f"https://api.spotify.com/v1/audio-features?ids={ids_str}")
    audio_features = af_resp.get("audio_features", [])
    print(f"  → {len([a for a in audio_features if a])} valid")

    print("Fetching recently played (50) ...")
    recent_resp = get("https://api.spotify.com/v1/me/player/recently-played?limit=50")
    recent_tracks = recent_resp.get("items", [])
    print(f"  → {len(recent_tracks)} plays")

    return {
        "display_name": display_name,
        "top_tracks": top_tracks,
        "audio_features": audio_features,
        "recent_tracks": recent_tracks,
    }


def fetch_with_oauth(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    try:
        import spotipy  # type: ignore
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore
    except ImportError:
        raise SystemExit(
            "spotipy not installed. Run: pip install spotipy\n"
            "Or use --token from https://developer.spotify.com/console/"
        )

    scope = "user-top-read user-read-recently-played"
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
        )
    )

    print("Fetching /me ...")
    me = sp.me()
    display_name = me.get("display_name", "")
    print(f"  → Logged in as: {display_name}")

    print("Fetching top tracks (medium_term, 50) ...")
    top_resp = sp.current_user_top_tracks(limit=50, time_range="medium_term")
    top_tracks = top_resp.get("items", [])
    print(f"  → {len(top_tracks)} tracks")

    track_ids = [t["id"] for t in top_tracks if t.get("id")]
    print(f"Fetching audio features for {len(track_ids)} tracks ...")
    audio_features = sp.audio_features(track_ids[:50])
    print(f"  → {len([a for a in audio_features if a])} valid")

    print("Fetching recently played (50) ...")
    recent_resp = sp.current_user_recently_played(limit=50)
    recent_tracks = recent_resp.get("items", [])
    print(f"  → {len(recent_tracks)} plays")

    return {
        "display_name": display_name,
        "top_tracks": top_tracks,
        "audio_features": audio_features,
        "recent_tracks": recent_tracks,
    }


def parse_spotify_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ml spotify",
        description="Build 17-d user features from Spotify Web API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Token flow (fastest):
  https://developer.spotify.com/console/get-recently-played/
  Get Token → enable user-top-read + user-read-recently-played → copy

OAuth flow:
  pip install spotipy
  python -m ml spotify --client-id ... --client-secret ...
""",
    )
    auth = p.add_mutually_exclusive_group(required=True)
    auth.add_argument("--token", type=str)
    auth.add_argument("--client-id", type=str)
    p.add_argument("--client-secret", type=str, default=None)
    p.add_argument("--redirect-uri", type=str, default="http://localhost:8888/callback")
    p.add_argument("--out", type=str, default="ml/cli/my_user.json")
    return p.parse_args(argv)


def run_spotify_profile(args: argparse.Namespace) -> None:
    if args.token:
        data = fetch_with_token(args.token)
    else:
        secret = args.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not secret:
            raise SystemExit("--client-secret or SPOTIFY_CLIENT_SECRET required with --client-id")
        data = fetch_with_oauth(args.client_id, secret, args.redirect_uri)

    profile = compute_user_features(
        top_tracks=data["top_tracks"],
        audio_features=data["audio_features"],
        recent_tracks=data["recent_tracks"],
        display_name=data.get("display_name", ""),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2))

    print(f"\nSaved → {out_path}")
    print("\nFeature summary:")
    for k, v in profile["debug"].items():
        print(f"  {k:<30} {v}")

    print("\nThen cold-train User Tower + run:")
    print(f"  python -m ml train-user --user-json {args.out} --output ml/export/my_user_tower.pt")
    print(f"  python -m ml run --user-json {args.out} --user-tower ml/export/my_user_tower.pt")


def main() -> None:
    run_spotify_profile(parse_spotify_args())


if __name__ == "__main__":
    main()
