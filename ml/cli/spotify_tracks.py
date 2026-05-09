"""
Resolve Spotify track IDs → display strings ("Title — Artist").

Token options (in priority order):
  1. --client-id / --client-secret  → auto client-credentials token (recommended)
  2. --spotify-token <token>         → raw Bearer string
  3. SPOTIFY_TOKEN env var           → raw Bearer string

No spotipy required — plain urllib.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def _client_credentials_token(client_id: str, client_secret: str) -> str | None:
    """POST /api/token with client credentials; returns access token string or None."""
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=data,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode())
        return str(body["access_token"])
    except Exception as exc:
        print(f"  [spotify] client-credentials token failed: {exc}")
        return None


def resolve_track_names(
    track_ids: list[str],
    token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> dict[str, str]:
    """
    Returns {track_id: "Title — Artist"}.
    Prints a warning if the API call fails so the user knows what happened.
    """
    # Prefer auto-token from client credentials
    bearer = token or os.environ.get("SPOTIFY_TOKEN", "")

    _cid = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
    _cs = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if _cid and _cs:
        auto = _client_credentials_token(_cid, _cs)
        if auto:
            bearer = auto

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
        except urllib.error.HTTPError as exc:
            print(f"  [spotify] /v1/tracks returned {exc.code} — token may be expired")
            break
        except (urllib.error.URLError, OSError) as exc:
            print(f"  [spotify] network error: {exc}")
            break
        for tr in body.get("tracks") or []:
            if not isinstance(tr, dict) or not tr.get("id"):
                continue
            tid = str(tr["id"])
            name = tr.get("name", "?")
            artists = tr.get("artists") or []
            artist = artists[0]["name"] if artists and isinstance(artists[0], dict) else "?"
            result[tid] = f"{name} — {artist}"

    if track_ids and not result:
        print("  [spotify] no track names resolved — check token / credentials")

    return result
