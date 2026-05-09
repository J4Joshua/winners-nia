"""
Resolve Spotify track IDs → display strings ("Title — Artist").

Uses GET /v1/tracks (public endpoint, only needs a Bearer token).
Token can come from:
  --spotify-token <token>   raw Bearer string (copy from browser dev-tools)
  SPOTIFY_TOKEN env var

No spotipy required — plain urllib.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def resolve_track_names(
    track_ids: list[str],
    token: str | None = None,
) -> dict[str, str]:
    """
    Returns {track_id: "Title — Artist"} for as many IDs as Spotify resolves.
    Silently skips failures so the caller always gets a (possibly partial) dict.
    """
    bearer = token or os.environ.get("SPOTIFY_TOKEN", "")
    if not bearer:
        return {}

    result: dict[str, str] = {}
    for i in range(0, len(track_ids), 50):
        chunk = track_ids[i : i + 50]
        url = f"https://api.spotify.com/v1/tracks?ids={','.join(chunk)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = json.loads(r.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
        for tr in body.get("tracks") or []:
            if not isinstance(tr, dict) or not tr.get("id"):
                continue
            tid = str(tr["id"])
            name = tr.get("name", "?")
            artists = tr.get("artists") or []
            artist = artists[0]["name"] if artists and isinstance(artists[0], dict) else "?"
            result[tid] = f"{name} — {artist}"
    return result
