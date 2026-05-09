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


from ml.cli.spotify_audio import fetch_audio_features_http, fetch_audio_features_spotify, spotify_access_token


def hydrate_top_tracks_http(access_token: str, top_tracks: list[dict]) -> list[dict]:
    """Replace items with Full Track objects from GET /v1/tracks (top-tracks often omit popularity)."""
    import json as json_lib
    import urllib.request

    ids = [str(t["id"]) for t in top_tracks if t.get("id")]
    if not ids:
        return top_tracks
    by_id: dict[str, dict] = {}
    try:
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            ids_str = ",".join(chunk)
            url = f"https://api.spotify.com/v1/tracks?ids={ids_str}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
            with urllib.request.urlopen(req) as r:
                body = json_lib.loads(r.read().decode())
            for tr in body.get("tracks") or []:
                if isinstance(tr, dict) and tr.get("id"):
                    by_id[str(tr["id"])] = tr
    except Exception:
        return top_tracks
    out: list[dict] = []
    for t in top_tracks:
        tid = t.get("id")
        if tid and str(tid) in by_id:
            out.append(by_id[str(tid)])
        else:
            out.append(t)
    return out


def primary_artist_ids_from_top_tracks(top_tracks: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in top_tracks:
        arts = t.get("artists") or []
        if not arts or not isinstance(arts[0], dict):
            continue
        aid = arts[0].get("id")
        if aid and str(aid) not in seen:
            seen.add(str(aid))
            out.append(str(aid))
    return out


def fetch_artists_http(access_token: str, artist_ids: list[str]) -> list[dict]:
    import json as json_lib
    import urllib.error
    import urllib.request

    if not artist_ids:
        return []
    all_a: list[dict] = []
    for i in range(0, len(artist_ids), 50):
        chunk = artist_ids[i : i + 50]
        ids_str = ",".join(chunk)
        url = f"https://api.spotify.com/v1/artists?ids={ids_str}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
        try:
            with urllib.request.urlopen(req) as r:
                body = json_lib.loads(r.read().decode())
        except urllib.error.HTTPError:
            break
        items = body.get("artists") if isinstance(body, dict) else None
        if not items:
            continue
        all_a.extend(a for a in items if isinstance(a, dict))
    return all_a


def structural_diversity_top_tracks(top_tracks: list[dict]) -> float:
    """How spread out primary artists are in top tracks (no extra API calls)."""
    aids = primary_artist_ids_from_top_tracks(top_tracks)
    if not aids:
        return 0.4
    return min(len(aids) / 20.0, 1.0)


def genre_diversity_from_top_tracks(access_token: str, top_tracks: list[dict]) -> float:
    structural = structural_diversity_top_tracks(top_tracks)
    aids = primary_artist_ids_from_top_tracks(top_tracks)
    if not aids:
        return structural
    try:
        artists = fetch_artists_http(access_token, aids)
    except Exception:
        return structural
    genres_flat: list[str] = []
    for a in artists:
        for g in a.get("genres") or []:
            if isinstance(g, str) and g:
                genres_flat.append(g)
    if not genres_flat:
        return structural
    from_tags = min(len(set(genres_flat)) / 15.0, 1.0)
    return max(from_tags, structural)


def track_metadata_debug(top_tracks: list[dict]) -> dict[str, float]:
    if not top_tracks:
        return {"mean_popularity_norm": 0.5, "mean_duration_norm": 0.5, "explicit_fraction": 0.5}
    pops: list[float] = []
    durs: list[float] = []
    ex: list[float] = []
    for t in top_tracks:
        pops.append(float(t.get("popularity") or 0) / 100.0)
        dm = int(t.get("duration_ms") or 0)
        durs.append(min(dm / 330000.0, 1.0))
        ex.append(1.0 if t.get("explicit") else 0.0)
    n = len(top_tracks)
    return {
        "mean_popularity_norm": round(sum(pops) / n, 3),
        "mean_duration_norm": round(sum(durs) / n, 3),
        "explicit_fraction": round(sum(ex) / n, 3),
    }


def compute_user_features(
    top_tracks: list[dict],
    audio_features: list[dict | None],
    recent_tracks: list[dict],
    display_name: str = "",
    *,
    genre_diversity_score: float | None = None,
    liked_track_ids: list[str] | None = None,
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
        sum(min(max((loud + 60.0) / 60.0, 0.0), 1.0) for loud in loudness_vals) / len(loudness_vals)
        if loudness_vals
        else 0.5
    )

    genre_diversity = genre_diversity_score if genre_diversity_score is not None else 0.4

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
        "liked_track_ids": liked_track_ids or [],
        "debug": {
            "mean_danceability": round(mean_danceability, 3),
            "mean_energy": round(mean_energy, 3),
            "mean_valence": round(mean_valence, 3),
            "mean_acousticness": round(mean_acousticness, 3),
            "mean_instrumentalness": round(mean_instrumentalness, 3),
            "mean_tempo_bpm": round(mean_tempo_norm * 240, 1),
            "genre_diversity": round(genre_diversity, 3),
            "peak_hour": peak_hour,
            "session_length_avg": round(session_length_avg, 1),
            "discovery_ratio": round(discovery_ratio, 3),
            "top_track_count": len(top_tracks),
            "audio_features_count": len(valid_af),
            "recent_tracks_used": total_recent,
            "audio_features_ok": len(valid_af) > 0,
            "unique_primary_artists": len(primary_artist_ids_from_top_tracks(top_tracks)),
            **track_metadata_debug(top_tracks),
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

    print("Hydrating top tracks (/v1/tracks, popularity & full fields) ...")
    top_tracks = hydrate_top_tracks_http(token, top_tracks)
    print(f"  → {len(top_tracks)} tracks")

    track_ids = [t["id"] for t in top_tracks if t.get("id")]
    batch_ids = track_ids[:100]
    print(f"Fetching audio features for {len(batch_ids)} tracks ...")
    audio_features = fetch_audio_features_http(token, batch_ids)
    print(f"  → {len([a for a in audio_features if a])} valid (audio analysis)")

    print("Fetching artist genres (top tracks) ...")
    genre_div = genre_diversity_from_top_tracks(token, top_tracks)
    print(f"  → genre_diversity ≈ {genre_div:.2f}")

    print("Fetching recently played (50) ...")
    recent_resp = get("https://api.spotify.com/v1/me/player/recently-played?limit=50")
    recent_tracks = recent_resp.get("items", [])
    print(f"  → {len(recent_tracks)} plays")

    return {
        "display_name": display_name,
        "top_tracks": top_tracks,
        "audio_features": audio_features,
        "recent_tracks": recent_tracks,
        "genre_diversity_score": genre_div,
    }


def fetch_liked_tracks_http(access_token: str, limit: int = 2000) -> list[str]:
    """
    Paginate GET /v1/me/tracks → return list of track IDs.
    Requires user-library-read scope. Returns up to `limit` IDs.
    """
    import urllib.error
    import urllib.request
    import json as _json

    ids: list[str] = []
    url: str | None = f"https://api.spotify.com/v1/me/tracks?limit=50"
    while url and len(ids) < limit:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                body = _json.loads(r.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            print(f"  Warning: liked tracks fetch failed: {exc}")
            break
        for item in body.get("items") or []:
            track = item.get("track") if isinstance(item, dict) else None
            if isinstance(track, dict) and track.get("id"):
                ids.append(str(track["id"]))
        url = body.get("next")  # None when no more pages
    return ids[:limit]


def fetch_with_oauth(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    try:
        import spotipy  # type: ignore
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore
    except ImportError:
        raise SystemExit(
            "spotipy not installed. Run: pip install spotipy\n"
            "Or obtain a user access token via OAuth (see https://developer.spotify.com/documentation/web-api/tutorials/code-flow)"
        )

    scope = "user-top-read user-read-recently-played user-library-read"
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

    print("Hydrating top tracks (/v1/tracks, popularity & full fields) ...")
    top_tracks = hydrate_top_tracks_http(spotify_access_token(sp), top_tracks)
    print(f"  → {len(top_tracks)} tracks")

    track_ids = [t["id"] for t in top_tracks if t.get("id")]
    print(f"Fetching audio features for {len(track_ids)} tracks ...")
    audio_features = fetch_audio_features_spotify(sp, track_ids)
    print(f"  → {len([a for a in audio_features if a])} valid (audio analysis)")

    print("Fetching artist genres (top tracks) ...")
    genre_div = genre_diversity_from_top_tracks(spotify_access_token(sp), top_tracks)
    print(f"  → genre_diversity ≈ {genre_div:.2f}")

    print("Fetching recently played (50) ...")
    recent_resp = sp.current_user_recently_played(limit=50)
    recent_tracks = recent_resp.get("items", [])
    print(f"  → {len(recent_tracks)} plays")

    print("Fetching liked songs (up to 2000) ...")
    liked_track_ids = fetch_liked_tracks_http(spotify_access_token(sp), limit=2000)
    print(f"  → {len(liked_track_ids)} liked tracks")

    return {
        "display_name": display_name,
        "top_tracks": top_tracks,
        "audio_features": audio_features,
        "recent_tracks": recent_tracks,
        "genre_diversity_score": genre_div,
        "liked_track_ids": liked_track_ids,
    }


def parse_spotify_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ml spotify",
        description="Build 17-d user features from Spotify Web API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Token flow (recommended — Console “Get Token” pages are retired):

  1) https://developer.spotify.com/dashboard → Create app → Settings
  2) Redirect URIs: add the same value you pass as --redirect-uri
     (e.g. http://127.0.0.1:8888/callback — Spotify prefers loopback IPs over "localhost" for new apps)
  3) pip install spotipy
  4) python -m ml spotify --client-id YOUR_ID --client-secret YOUR_SECRET

Optional: paste a short-lived Bearer token from any OAuth tool:
  python -m ml spotify --token YOUR_ACCESS_TOKEN

Scopes required: user-top-read, user-read-recently-played

Docs: https://developer.spotify.com/documentation/web-api/tutorials/code-flow
""",
    )
    auth = p.add_mutually_exclusive_group(required=True)
    auth.add_argument("--token", type=str)
    auth.add_argument("--client-id", type=str)
    p.add_argument("--client-secret", type=str, default=None)
    p.add_argument("--redirect-uri", type=str, default="http://127.0.0.1:8888/callback")
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
        genre_diversity_score=data.get("genre_diversity_score"),
        liked_track_ids=data.get("liked_track_ids"),
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
