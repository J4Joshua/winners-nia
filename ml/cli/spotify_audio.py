"""
Spotify often returns 403 on GET /v1/audio-features for many developer apps.
We fall back to neutral placeholders so profile export still works.

Audio-features requests use urllib (not spotipy's HTTP client) so 403 does not emit
spotipy's noisy "HTTP Error for GET..." line.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

AUDIO_FEATURES_403_HINT = """
Spotify returned 403 on /v1/audio-features. Many apps lose access to audio analysis.
Continuing without audio features: audio-derived scalars in the 17-d vector use neutral defaults (0.5).
Your top_track_ids / recent_track_ids are still real — train-user (centroid of embeddings) still works.
"""


def fetch_audio_features_http(access_token: str, track_ids: list[str]) -> list[dict | None]:
    """GET /v1/audio-features with Bearer token; quiet on 403; returns parallel list."""
    if not track_ids:
        return []
    batch = track_ids[:100]
    ids_str = ",".join(batch)
    url = f"https://api.spotify.com/v1/audio-features?ids={ids_str}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    try:
        with urllib.request.urlopen(req) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(AUDIO_FEATURES_403_HINT.strip())
            return [None] * len(batch)
        raise
    audio_features = body.get("audio_features", []) if isinstance(body, dict) else []
    out: list[dict | None] = []
    for i in range(len(batch)):
        af = audio_features[i] if i < len(audio_features) else None
        out.append(af if isinstance(af, dict) else None)
    return out


def spotify_access_token(sp: object) -> str:
    """Bearer string from an authenticated spotipy ``Spotify`` instance."""
    return _access_token_from_spotipy(sp)


def _access_token_from_spotipy(sp: object) -> str:
    auth = getattr(sp, "auth_manager", None)
    if auth is None:
        raise TypeError("spotipy Spotify client requires auth_manager")
    getter = getattr(auth, "get_access_token", None)
    if callable(getter):
        try:
            tok = getter(as_dict=False)
        except TypeError:
            tok = getter()
        if isinstance(tok, str) and tok:
            return tok
        if isinstance(tok, dict) and tok.get("access_token"):
            return str(tok["access_token"])
    cached = getattr(auth, "get_cached_token", None)
    if callable(cached):
        info = cached()
        if isinstance(info, dict) and info.get("access_token"):
            return str(info["access_token"])
    raise RuntimeError("Could not read Spotify access token from spotipy auth_manager")


def fetch_audio_features_spotify(sp: object, track_ids: list[str]) -> list[dict | None]:
    """OAuth spotipy client → Bearer token → urllib audio-features (no spotipy 403 noise)."""
    token = spotify_access_token(sp)
    return fetch_audio_features_http(token, track_ids)
