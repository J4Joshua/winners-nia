"""
Attune interactive TUI — full flow from Spotify login to recommendations.

  python -m ml session                                         # full onboarding
  python -m ml session --user-json my.json --user-tower t.pt   # skip to recs

Onboarding flow (when no --user-json / --user-tower given)
──────────────────────────────────────────────────────────
  1. Enter Spotify client-id + client-secret
  2. Browser opens for OAuth → Spotify profile + liked songs fetched
  3. User Tower trains (cold-start on liked/top/recent tracks)
  4. Session screen opens

Session keyboard
─────────────────
  ↑ / ↓       navigate songs
  l           ♥ like   (EMA pull toward this song, updates session aggregate)
  d           ✕ dislike (EMA push away, updates session aggregate)
  v           pick vibe / mood / genre
  t           pick time of day
  w           pick weather  (☀️ sunny · 🌧️ rainy · ❄️ cold …)
  p           pick place    (🏠 home · 🏋️ gym · ☕ café …)
  x           clear all context
  r           refresh (soft dislike on queue, new recommendations)
  s           save tower
  q           quit (auto-saves)

Learning algorithm
──────────────────
  Inspired by Spotify CoSeRNN: query = long_term_emb + session_delta + context
  Long-term: EMA of user tower output, updated with each like/dislike (no backward pass)
  Session delta: EMA aggregate of liked / disliked song embeddings this session
  Context: song tower projections of vibe/weather/location, + learned per-context offsets
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from ml.models.song_tower import EMBEDDING_DIM
from ml.inference.defaults import DEFAULT_HUB_MODEL_REPO
from ml.inference.demo_users import DEMO_USERS
from ml.inference.device import resolve_device
from ml.inference.loaders import load_artifacts_from_hub, load_user_tower
from ml.inference.search import nearest_neighbors

# ── constants ─────────────────────────────────────────────────────────────────

TIME_HOURS: dict[str, int] = {
    "🌅  Morning": 8, "☀️  Afternoon": 14, "🌆  Evening": 19, "🌙  Night": 23,
}

# Weather presets → existing vibe names (used to get a vibe embedding for blending)
WEATHER_CONTEXTS: dict[str, str] = {
    "☀️  Sunny":        "happy",
    "🌤️  Partly cloudy": "morning",
    "☁️  Overcast":     "chill",
    "🌧️  Rainy":        "rainy",
    "⛈️  Stormy":       "sad",
    "❄️  Cold / snow":  "sleep",
    "🌙  Clear night":  "late-night",
}

# Location presets → vibe names
LOCATION_CONTEXTS: dict[str, str] = {
    "🏠  Home":          "chill",
    "🏋️  Gym":           "workout",
    "☕  Café":           "focus",
    "🚗  Commute":       "pop",
    "🌳  Outdoors":      "morning",
    "🏢  Office":        "focus",
    "🎉  Party":         "party",
    "😴  Winding down":  "sleep",
    "✈️  Travelling":    "hype",
    "📚  Studying":      "focus",
}

VIBES: list[str] = list(DEMO_USERS.keys())
IDX_PEAK_SIN, IDX_PEAK_COS = 14, 15
AUDIO_INDICES = list(range(6))

DEFAULT_JSON = Path("ml/cli/my_user.json")
DEFAULT_TOWER = Path("ml/export/my_user_tower.pt")


# ── pure-Python session state ─────────────────────────────────────────────────

def _vibe_to_song_features(vibe_name: str) -> np.ndarray:
    """
    Convert a 17-d vibe profile → 45-d Song Tower input.

    Layout:
      [0:12]   key one-hot  (left at zeros — vibe has no key preference)
      [12:26]  audio scalars mapped from vibe profile
      [26:45]  19-d macro-genre one-hot based on vibe→genre mapping
    """
    from ml.training.dataset import NUM_MACRO_GENRES, SONG_FEATURE_DIM

    # Vibe name → macro-genre index for the genre one-hot
    _VIBE_MACRO: dict[str, int] = {
        "pop": 0, "happy": 0, "morning": 0,
        "rock": 1, "workout": 1,
        "hype": 4, "party": 4, "late-night": 4,
        "rap": 5,
        "rnb": 6,
        "jazz": 7,
        "classical": 8, "chill": 8, "rainy": 8, "sad": 8,
        "focus": 8, "sleep": 8,
        "metal": 9,
    }

    vv = list(DEMO_USERS[vibe_name]["features"])
    vec = np.zeros(SONG_FEATURE_DIM, dtype=np.float32)

    # Audio scalars (indices 12-25)
    vec[12] = vv[0]               # danceability
    vec[13] = vv[1]               # energy
    vec[15] = vv[3]               # acousticness
    vec[16] = vv[4]               # instrumentalness
    vec[18] = vv[2]               # valence
    vec[19] = vv[6] if len(vv) > 6 else 0.5  # loudness_norm
    vec[20] = vv[5]               # tempo_norm
    vec[23] = 0.5                 # popularity (neutral)
    vec[24] = 0.5                 # duration (neutral)
    vec[25] = 4.0 / 7.0          # time_signature (4/4)

    # Genre one-hot (indices 26-44)
    macro = _VIBE_MACRO.get(vibe_name, NUM_MACRO_GENRES - 1)  # default "other"
    vec[26 + macro] = 1.0

    return vec


class SessionState:
    def __init__(
        self,
        profile: dict,
        song_tower: nn.Module,
        user_tower: nn.Module,
        embeddings: np.ndarray,
        ids: np.ndarray,
        names: dict[str, str],
        device: torch.device,
        save_path: Path,
        top_k: int = 20,
        spotify_token: str | None = None,
    ) -> None:
        self.user_name: str = profile.get("name", "User")
        self.base_features: list[float] = list(profile["features"])
        self.song_tower = song_tower
        self.user_tower = user_tower
        self.embeddings = embeddings
        self.ids = ids
        self.names = names
        self.device = device
        self.save_path = save_path
        self.top_k = top_k
        self.spotify_token: str | None = spotify_token
        self.now_playing: str | None = None   # track id currently playing
        self.id_to_emb: dict[str, np.ndarray] = {
            str(ids[i]): embeddings[i] for i in range(len(ids))
        }
        self.context_vibe:     str | None = None
        self.context_time:     str | None = None
        self.context_weather:  str | None = None
        self.context_location: str | None = None
        self.feedback_log: list[str] = []

        self._vibe_emb_cache: dict[str, np.ndarray] = {}

        # ── Pre-computed context similarity matrix ─────────────────────────────
        # Maps every possible context key → its Song Tower vibe embedding.
        # Used to propagate learning across similar contexts:
        #   when you like a "gym" song, "commute" (similar energy vibe) also
        #   gets a partial update proportional to cosine(gym_emb, commute_emb).
        # This lets the model generalise: "gym+rainy" learnings partially transfer
        # to "gym+sunny" and "commute+rainy", combining knowledge across contexts.
        self._all_ctx_embs: dict[str, np.ndarray] = self._precompute_context_embs()

        # ── CoSeRNN-style online learning (no backward pass) ──────────────────
        # Live embedding: starts from User Tower output, updated in embedding
        # space via EMA on each like/dislike. No gradient descent = no
        # catastrophic forgetting, and 100x faster than Adam steps.
        self._live_emb: np.ndarray = self._compute_base_emb()

        # Session aggregates: EMA of liked / disliked song embeddings.
        # Blended into the query for continuous in-session adaptation.
        # Inspired by Spotify CoSeRNN: query = long_term + session_delta + context.
        self._liked_agg:    np.ndarray | None = None
        self._disliked_agg: np.ndarray | None = None

        # Session stats (for TUI display)
        self.n_likes    = 0
        self.n_dislikes = 0
        self.n_refreshes = 0

        # Learned per-context preference offsets
        self._context_offsets: dict[str, np.ndarray] = {}

    def set_vibe(self,     v: str | None) -> None: self.context_vibe = v
    def set_time(self,     t: str | None) -> None: self.context_time = t
    def set_weather(self,  w: str | None) -> None: self.context_weather = w
    def set_location(self, l: str | None) -> None: self.context_location = l

    def clear_context(self) -> None:
        self.context_vibe = self.context_time = None
        self.context_weather = self.context_location = None

    def context_label(self) -> str:
        parts = []
        if self.context_time:     parts.append(self.context_time.split()[1])   # strip emoji
        if self.context_weather:  parts.append(self.context_weather.split()[1])
        if self.context_location: parts.append(self.context_location.split()[1])
        if self.context_vibe:     parts.append(self.context_vibe)
        return "  ·  ".join(parts) if parts else "none"

    def _ctx_features(self) -> list[float]:
        """17-d user feature vector with time context applied."""
        feats = list(self.base_features)
        if self.context_time:
            h = TIME_HOURS.get(self.context_time, 14)
            feats[IDX_PEAK_SIN] = math.sin(2 * math.pi * h / 24)
            feats[IDX_PEAK_COS] = math.cos(2 * math.pi * h / 24)
        return feats

    def _compute_base_emb(self) -> np.ndarray:
        """Run User Tower forward pass to get the initial embedding."""
        x = torch.tensor([self.base_features], dtype=torch.float32, device=self.device)
        self.user_tower.eval()
        with torch.no_grad():
            return self.user_tower(x).cpu().numpy()[0].astype(np.float32)

    def _user_embedding(self) -> np.ndarray:
        """Return the live embedding (EMA-updated, no forward pass needed)."""
        return self._live_emb

    def _song_tower_vibe_embedding(self, vibe_name: str) -> np.ndarray:
        """Project a vibe name through the Song Tower → 128-d embedding (cached)."""
        if vibe_name not in self._vibe_emb_cache:
            vec = _vibe_to_song_features(vibe_name)
            x = torch.tensor([vec], dtype=torch.float32, device=self.device)
            self.song_tower.eval()
            with torch.no_grad():
                emb = self.song_tower(x).cpu().numpy()[0].astype(np.float32)
            self._vibe_emb_cache[vibe_name] = emb
        return self._vibe_emb_cache[vibe_name]

    def _precompute_context_embs(self) -> dict[str, np.ndarray]:
        """
        Compute Song Tower embeddings for every possible context.

        Maps context key (e.g. 'p:🏋️  Gym') → 128-d embedding.
        Used to build the cross-context similarity kernel so that learning
        in one context (gym) transfers proportionally to related contexts (commute).
        """
        ctx_embs: dict[str, np.ndarray] = {}
        for weather_key, vibe_name in WEATHER_CONTEXTS.items():
            if vibe_name in DEMO_USERS:
                emb = self._song_tower_vibe_embedding(vibe_name)
                if emb is not None:
                    ctx_embs[f"w:{weather_key}"] = emb
        for loc_key, vibe_name in LOCATION_CONTEXTS.items():
            if vibe_name in DEMO_USERS:
                emb = self._song_tower_vibe_embedding(vibe_name)
                if emb is not None:
                    ctx_embs[f"p:{loc_key}"] = emb
        for vibe_name in DEMO_USERS:
            emb = self._song_tower_vibe_embedding(vibe_name)
            if emb is not None:
                ctx_embs[f"v:{vibe_name}"] = emb
        return ctx_embs

    def _active_context_keys(self) -> list[str]:
        """Stable string keys for each active context slot."""
        keys = []
        if self.context_weather:  keys.append(f"w:{self.context_weather}")
        if self.context_location: keys.append(f"p:{self.context_location}")
        if self.context_vibe:     keys.append(f"v:{self.context_vibe}")
        if self.context_time:     keys.append(f"t:{self.context_time}")
        return keys

    def _update_context_offsets(self, song_emb: np.ndarray, like: bool, strength: float = 1.0) -> None:
        """
        Update context offsets with cross-context kernel propagation.

        Direct update: active context keys get full strength EMA update.
        Cross-context: ALL known context embeddings get a partial update weighted
        by cosine similarity to the active context. Similarity decay rate:
          - sim > 0.7 → 40% propagation (very similar contexts share knowledge)
          - sim > 0.4 → 15% propagation (somewhat similar)
          - sim ≤ 0.4 → 0%  (unrelated contexts are unaffected)

        Example: liked a gym song → gym (100%), commute (60%), office (30%) update.
        Next time at commute, some gym preferences are already baked in.
        """
        lr   = (0.20 if like else 0.12) * strength
        zeros = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        active_keys = self._active_context_keys()

        for active_key in active_keys:
            active_emb = self._all_ctx_embs.get(active_key)

            # Direct update for the active context
            cur = self._context_offsets.get(active_key, zeros.copy())
            self._context_offsets[active_key] = (
                cur + lr * (song_emb - cur) if like else cur - lr * song_emb
            )

            # Cross-context propagation to all other known contexts
            if active_emb is None:
                continue
            for other_key, other_emb in self._all_ctx_embs.items():
                if other_key == active_key:
                    continue
                sim = float(np.dot(active_emb, other_emb))  # both L2-normalized
                if sim <= 0.4:
                    continue
                cross_lr = lr * (0.40 if sim > 0.7 else 0.15)
                cur_other = self._context_offsets.get(other_key, zeros.copy())
                self._context_offsets[other_key] = (
                    cur_other + cross_lr * (song_emb - cur_other)
                    if like else cur_other - cross_lr * song_emb
                )

    def _query(self) -> np.ndarray:
        """
        CoSeRNN-inspired session query.

        query = normalize(
            live_emb                  # long-term preference (EMA-updated per feedback)
            + 0.35 * liked_agg        # session pull: toward songs we liked this session
            - 0.15 * disliked_agg     # session push: away from songs we disliked
            + context_offsets         # per-context learned shifts
            + vibe/weather/location   # explicit context blends from Song Tower
        )

        Unlike Spotify CoSeRNN (which uses RNN over sessions), we adapt in real-time
        within a single session using EMA updates — no backward pass required.
        """
        emb = self._live_emb.copy()

        # Session aggregates (continuous in-session adaptation, Spotify CoSeRNN §3.3)
        if self._liked_agg is not None:
            emb = emb + 0.35 * self._liked_agg
        if self._disliked_agg is not None:
            emb = emb - 0.15 * self._disliked_agg

        # Collect (embedding, weight) for each active context
        overlays: list[tuple[np.ndarray, float]] = []
        if self.context_vibe and self.context_vibe in DEMO_USERS:
            overlays.append((self._song_tower_vibe_embedding(self.context_vibe), 0.50))
        if self.context_weather:
            vibe_name = WEATHER_CONTEXTS.get(self.context_weather)
            if vibe_name and vibe_name in DEMO_USERS:
                overlays.append((self._song_tower_vibe_embedding(vibe_name), 0.25))
        if self.context_location:
            vibe_name = LOCATION_CONTEXTS.get(self.context_location)
            if vibe_name and vibe_name in DEMO_USERS:
                overlays.append((self._song_tower_vibe_embedding(vibe_name), 0.25))

        for ctx_emb, w in overlays:
            emb = emb + w * ctx_emb

        # Add learned per-context offsets
        for key in self._active_context_keys():
            if key in self._context_offsets:
                emb = emb + self._context_offsets[key]

        norm = np.linalg.norm(emb)
        return (emb / (norm + 1e-8)).astype(np.float32)

    def recommendations(self) -> list[tuple[str, float, str, str]]:
        results = nearest_neighbors(self._query(), self.embeddings, self.ids, top_k=self.top_k, names=self.names)
        out = []
        for tid, score in results:
            d = self.names.get(tid, tid).split(" — ", 1)
            out.append((tid, score, d[0], d[1] if len(d) > 1 else ""))
        return out

    def _ema_update_live(self, song_emb: np.ndarray, like: bool, alpha: float) -> None:
        """
        EMA update of the live embedding — no backward pass, no catastrophic forgetting.

        like    → pull live_emb toward song_emb  (alpha = 0.08)
        dislike → push live_emb away from song_emb (alpha = 0.05)

        This is equivalent to one step of projected gradient descent on the
        cosine similarity objective, but in closed form.
        """
        direction = song_emb if like else -song_emb
        updated   = self._live_emb + alpha * direction
        norm      = np.linalg.norm(updated)
        self._live_emb = (updated / norm) if norm > 1e-8 else self._live_emb

    def _ema_update_aggregate(self, song_emb: np.ndarray, like: bool) -> None:
        """Update session aggregates with EMA decay = 0.7 (recent matters more)."""
        if like:
            if self._liked_agg is None:
                self._liked_agg = song_emb.copy()
            else:
                self._liked_agg = 0.7 * self._liked_agg + 0.3 * song_emb
                n = np.linalg.norm(self._liked_agg)
                if n > 1e-8:
                    self._liked_agg /= n
        else:
            if self._disliked_agg is None:
                self._disliked_agg = song_emb.copy()
            else:
                self._disliked_agg = 0.7 * self._disliked_agg + 0.3 * song_emb
                n = np.linalg.norm(self._disliked_agg)
                if n > 1e-8:
                    self._disliked_agg /= n

    def _update(self, tid: str, like: bool) -> float | None:
        if tid not in self.id_to_emb:
            return None
        song_emb = self.id_to_emb[tid].astype(np.float32)

        # Compute similarity before update (for display)
        sim_before = float(np.dot(self._live_emb, song_emb))

        # EMA update — no backward pass, no catastrophic forgetting
        alpha = 0.08 if like else 0.05
        self._ema_update_live(song_emb, like, alpha)
        self._ema_update_aggregate(song_emb, like)

        # Compute similarity after update (shows learning)
        sim_after = float(np.dot(self._live_emb, song_emb))

        if like:
            self.n_likes += 1
        else:
            self.n_dislikes += 1

        # Per-context offset update
        if self._active_context_keys():
            self._update_context_offsets(song_emb, like=like, strength=1.0)

        name  = self.names.get(tid, tid).split(" — ")[0]
        ctx   = " [" + "+".join(k.split(":")[0] for k in self._active_context_keys()) + "]" \
                if self._active_context_keys() else ""
        delta = sim_after - sim_before
        sign  = "+" if delta >= 0 else ""
        self.feedback_log.append(
            f"{'♥' if like else '✕'}{ctx}  {name[:32]}   {sim_before:.3f}→{sim_after:.3f} ({sign}{delta:.3f})"
        )
        return sim_after

    def soft_dislike_queue(self, tids: list[str]) -> None:
        """
        Refresh pressed: treat all shown songs as weak negatives.
        Uses a smaller alpha (0.02) so it's a gentle nudge, not a full dislike.
        """
        embs = [self.id_to_emb[tid].astype(np.float32) for tid in tids if tid in self.id_to_emb]
        if not embs:
            return
        avg = np.mean(embs, axis=0).astype(np.float32)
        n   = np.linalg.norm(avg)
        if n > 1e-8:
            avg /= n

        self._ema_update_live(avg, like=False, alpha=0.02)
        self._ema_update_aggregate(avg, like=False)
        self.n_refreshes += 1

        if self._active_context_keys():
            for emb in embs:
                self._update_context_offsets(emb, like=False, strength=0.10)

        self.feedback_log.append(f"↺  queue skipped ({len(embs)} songs) — soft nudge")

    def play_track(self, tid: str) -> str:
        """
        Start playing a track on the user's active Spotify device.

        Returns a status message for the feedback log.
        Requires a token with 'user-modify-playback-state' scope and Spotify Premium.
        """
        if not self.spotify_token:
            return "⚠️  No Spotify token — login via 'ml session' onboarding to enable playback"

        import urllib.request, urllib.error as _ue, json as _json
        uri    = f"spotify:track:{tid}"
        name   = self.names.get(tid, tid)
        body   = _json.dumps({"uris": [uri]}).encode()
        req    = urllib.request.Request(
            "https://api.spotify.com/v1/me/player/play",
            data=body,
            method="PUT",
            headers={
                "Authorization": f"Bearer {self.spotify_token}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 204:
                    self.now_playing = tid
                    return f"▶  {name}"
                return f"⚠️  Spotify returned {resp.status}"
        except _ue.HTTPError as e:
            if e.code == 404:
                return "⚠️  No active Spotify device — open Spotify first"
            if e.code == 403:
                return "⚠️  Playback needs Spotify Premium + user-modify-playback-state scope"
            if e.code == 401:
                return "⚠️  Token expired — paste a fresh token"
            return f"⚠️  Spotify error {e.code}"
        except Exception as exc:
            return f"⚠️  Playback error: {exc}"

    def like(self, tid: str) -> float | None: return self._update(tid, True)
    def dislike(self, tid: str) -> float | None: return self._update(tid, False)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.user_tower.cpu().state_dict()}, self.save_path)
        self.user_tower.to(self.device)
        self.feedback_log.append(f"💾  Saved → {self.save_path}")


# ── TUI (built lazily so import works without textual installed) ──────────────

def _build_app(
    hub_repo: str,
    device: torch.device,
    user_json_path: Path | None,
    tower_path: Path | None,
    top_k: int,
    spotify_token: str | None = None,
):
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.screen import Screen, ModalScreen
    from textual.widgets import (
        Button, DataTable, Footer, Header, Input,
        Label, ListItem, ListView, Static,
    )
    # ── shared catalog (loaded once) ─────────────────────────────────────────

    catalog: dict = {}  # filled during loading

    # ── reusable picker modal ────────────────────────────────────────────────

    class PickerModal(ModalScreen):
        DEFAULT_CSS = """
        PickerModal { align: center middle; }
        PickerModal > Label { margin: 1 2; text-style: bold; color: $accent; }
        PickerModal > ListView {
            width: 48; height: auto; max-height: 22;
            border: round $accent; background: $surface;
        }
        """
        BINDINGS = [("escape", "dismiss_none", "Cancel")]

        def __init__(self, title: str, options: list[str]) -> None:
            super().__init__()
            self._title, self._options = title, options

        def compose(self) -> ComposeResult:
            yield Label(self._title)
            # Textual IDs must be alphanumeric/underscore/hyphen — use index-based IDs
            # and resolve back to the original string via _options on selection.
            yield ListView(*[
                ListItem(Label(f"  {o}"), id=f"opt-{i}")
                for i, o in enumerate(self._options)
            ])

        def on_list_view_selected(self, e: ListView.Selected) -> None:
            if e.item.id and e.item.id.startswith("opt-"):
                idx = int(e.item.id[4:])
                self.dismiss(self._options[idx])
            else:
                self.dismiss(None)

        def action_dismiss_none(self) -> None:
            self.dismiss(None)

    # ── onboard screen ───────────────────────────────────────────────────────

    class OnboardScreen(Screen):
        CSS = """
        OnboardScreen { align: center middle; layout: vertical; }
        #card {
            width: 72; height: auto;
            border: round $accent; padding: 2 3;
            background: $surface;
        }
        #title  { text-style: bold; color: $accent; margin-bottom: 1; }
        #note   { color: $text-muted; margin: 1 0; }
        #hint   { color: $text-muted; text-style: italic; margin-top: 1; }
        #error  { color: $error; margin-top: 1; }
        #tok    { margin: 0 0 1 0; }
        #submit { margin-top: 1; width: 100%; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            with Static(id="card"):
                yield Label("🎵  Attune", id="title")
                yield Label("Paste your Spotify Bearer token:", id="note")
                yield Input(placeholder="BQB3…", id="tok")
                yield Button("Connect →", id="submit", variant="primary")
                yield Static(
                    "[dim]Get a token: open.spotify.com → DevTools (F12) → Network tab\n"
                    "→ any request to api.spotify.com → Authorization header → copy after 'Bearer '[/dim]",
                    id="hint",
                )
                yield Static("", id="error")
            yield Footer()

        def _submit(self) -> None:
            token = self.query_one("#tok", Input).value.strip()
            if not token:
                self.query_one("#error", Static).update("[red]Token cannot be empty[/red]")
                return
            self.app.push_screen(ProgressScreen(token))

        def on_button_pressed(self, e: Button.Pressed) -> None:
            if e.button.id == "submit":
                self._submit()

        def on_input_submitted(self, e: Input.Submitted) -> None:
            self._submit()

    # ── progress screen ──────────────────────────────────────────────────────

    class ProgressScreen(Screen):
        CSS = """
        ProgressScreen { align: center middle; layout: vertical; }
        #log {
            width: 72; height: 22;
            border: round $primary; padding: 1 2;
            background: $surface;
        }
        """

        def __init__(self, token: str) -> None:
            super().__init__()
            self._token = token
            self._lines: list[str] = []

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("", id="log")
            yield Footer()

        def on_mount(self) -> None:
            self._log("⏳  Starting…")
            self._run_onboarding()

        def _log(self, msg: str) -> None:
            self._lines.append(msg)
            self.query_one("#log", Static).update("\n".join(self._lines[-20:]))

        @work(thread=True)
        def _run_onboarding(self) -> None:
            def log(msg: str) -> None:
                self.app.call_from_thread(self._log, msg)

            try:
                # 1. Validate token + get username
                log("🔐  Verifying token…")
                import urllib.request, urllib.error as _ue
                req = urllib.request.Request(
                    "https://api.spotify.com/v1/me",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        me = json.loads(r.read())
                except _ue.HTTPError as exc:
                    self.app.call_from_thread(self._log, f"[red]Token rejected ({exc.code}). Paste a fresh token.[/red]")
                    return
                display_name = me.get("display_name", "User")
                log(f"  ✓  Logged in as [bold]{display_name}[/bold]")

                # 2. Load catalog
                log("📦  Loading song catalog from Hub…")
                model, embeddings, ids, names = load_artifacts_from_hub(hub_repo, device)
                catalog["model"] = model
                catalog["embeddings"] = embeddings
                catalog["ids"] = ids
                catalog["names"] = names
                log(f"  ✓  {len(ids):,} tracks  ({len(names):,} named)")

                # 3. Fetch Spotify data via token
                from ml.cli.spotify_profile import (
                    compute_user_features, fetch_with_token,
                )
                log("🎵  Fetching your Spotify data…")
                data = fetch_with_token(self._token)
                log(f"  ✓  {len(data['top_tracks'])} top  ·  {len(data['recent_tracks'])} recent  ·  {len(data.get('liked_track_ids', []))} liked")

                # 4. Compute profile + save
                profile = compute_user_features(
                    top_tracks=data["top_tracks"],
                    audio_features=data["audio_features"],
                    recent_tracks=data["recent_tracks"],
                    display_name=display_name,
                    genre_diversity_score=data.get("genre_diversity_score"),
                    liked_track_ids=data.get("liked_track_ids"),
                    playlists=data.get("playlists"),
                )
                DEFAULT_JSON.parent.mkdir(parents=True, exist_ok=True)
                DEFAULT_JSON.write_text(json.dumps(profile, indent=2))
                log(f"  ✓  Profile saved → {DEFAULT_JSON}")

                # 5. Train User Tower
                log("🧠  Training User Tower (cold-start)…")
                import argparse
                from ml.training.train_user_cold import train_user_cold
                train_user_cold(argparse.Namespace(
                    user_json=str(DEFAULT_JSON),
                    output=str(DEFAULT_TOWER),
                    hub_repo=None, embeddings=None, ids=None,
                    positives="all", epochs=2000, lr=5e-3,
                    seed=42, min_positives=3,
                    cpu=device.type == "cpu",
                ))
                log(f"  ✓  User Tower saved → {DEFAULT_TOWER}")

                # 6. Enter session
                log("✨  All done — opening recommendations…")
                user_tower = load_user_tower(DEFAULT_TOWER, device)
                state = SessionState(
                    profile=profile,
                    song_tower=catalog["model"],
                    user_tower=user_tower,
                    embeddings=embeddings,
                    ids=ids,
                    names=names,
                    device=device,
                    save_path=DEFAULT_TOWER,
                    top_k=top_k,
                    spotify_token=self._token,
                )
                self.app.call_from_thread(
                    lambda: self.app.switch_screen(SessionScreen(state))
                )

            except Exception as exc:
                import traceback
                self.app.call_from_thread(self._log, f"[red]Error: {exc}[/red]\n{traceback.format_exc()[-300:]}")

    # ── session screen ───────────────────────────────────────────────────────

    class SessionScreen(Screen):
        CSS = """
        SessionScreen { layout: vertical; }
        #context-bar {
            height: 2; background: $surface; padding: 0 2;
            border: round $accent; content-align: left middle;
        }
        #stats-bar {
            height: 1; background: $surface-darken-1; padding: 0 2;
            content-align: left middle; color: $text-muted;
        }
        DataTable { height: 1fr; border: round $primary; }
        DataTable > .datatable--cursor { background: $accent 30%; }
        DataTable > .datatable--header { text-style: bold; color: $accent; }
        #feedback-log {
            height: 8; padding: 0 2;
            border: round $warning; background: $surface-darken-1; color: $text-muted;
        }
        """
        BINDINGS: ClassVar[list[Binding]] = [
            Binding("enter",   "play",        "▶ Play",    show=True),
            Binding("l",       "like",         "♥ Like"),
            Binding("d",       "dislike",      "✕ Dislike"),
            Binding("v",       "pick_vibe",    "Vibe"),
            Binding("t",       "pick_time",    "Time"),
            Binding("w",       "pick_weather", "Weather"),
            Binding("p",       "pick_place",   "Place"),
            Binding("x",       "clear_ctx",    "Clear ctx"),
            Binding("r",       "refresh_recs", "Refresh"),
            Binding("s",       "save",         "Save"),
            Binding("q",       "quit_session", "Quit"),
        ]

        def __init__(self, state: SessionState) -> None:
            super().__init__()
            self.state = state
            self._row_ids: list[str] = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("", id="context-bar")
            yield Static("", id="stats-bar")
            t = DataTable(id="songs", cursor_type="row", zebra_stripes=True)
            t.add_columns("  #", "Match", "█ █ █ █ █", "Track", "Artist")
            yield t
            yield Static("", id="feedback-log")
            yield Footer()

        def on_mount(self) -> None:
            self._refresh_bar()
            self._refresh_songs()

        @staticmethod
        def _score_bar(score: float, width: int = 10) -> str:
            """Render a score between 0-1 as a filled block bar."""
            filled = max(0, min(width, round(score * width)))
            return "█" * filled + "░" * (width - filled)

        def _refresh_bar(self) -> None:
            ctx = self.state.context_label()
            ctx_text = f"  [yellow]{ctx}[/yellow]" if ctx else "  [dim]no context[/dim]"
            self.query_one("#context-bar", Static).update(
                f"[bold]{self.state.user_name}[/bold]{ctx_text}"
                f"   [dim]v=vibe  t=time  w=weather  p=place  x=clear[/dim]"
            )
            st = self.state
            likes_part    = f"[green]♥ {st.n_likes}[/green]" if st.n_likes    else "♥ 0"
            dislikes_part = f"[red]✕ {st.n_dislikes}[/red]" if st.n_dislikes  else "✕ 0"
            refresh_part  = f"[cyan]↺ {st.n_refreshes}[/cyan]" if st.n_refreshes else ""
            agg_hint = ""
            if st._liked_agg is not None:
                sim = float(np.dot(st._live_emb, st._liked_agg))
                agg_hint = f"   pull={sim:+.2f}"
            parts = [likes_part, dislikes_part]
            if st.n_refreshes:
                parts.append(refresh_part)
            self.query_one("#stats-bar", Static).update(
                "  " + "  ·  ".join(parts) + f"{agg_hint}   [dim]session[/dim]"
            )

        def _refresh_songs(self) -> None:
            recs = self.state.recommendations()
            table = self.query_one("#songs", DataTable)
            table.clear()
            self._row_ids = []
            playing = self.state.now_playing
            for i, (tid, score, track, artist) in enumerate(recs, 1):
                pct      = f"{score * 100:4.1f}%"
                bar      = self._score_bar(score)
                play_ind = "▶" if tid == playing else " "
                table.add_row(f"{i:>3}", pct, bar, f"{play_ind} {track}", artist)
                self._row_ids.append(tid)

        def _refresh_log(self) -> None:
            self.query_one("#feedback-log", Static).update(
                "\n".join(self.state.feedback_log[-7:])
            )

        def _cur_tid(self) -> str | None:
            t = self.query_one("#songs", DataTable)
            return self._row_ids[t.cursor_row] if 0 <= t.cursor_row < len(self._row_ids) else None

        def action_play(self) -> None:
            if tid := self._cur_tid():
                msg = self.state.play_track(tid)
                self.state.feedback_log.append(msg)
                self._refresh_log()
                self._refresh_songs()  # update ▶ indicator

        def action_like(self) -> None:
            if tid := self._cur_tid():
                self.state.like(tid)
                self._refresh_bar()
                self._refresh_log()
                self._refresh_songs()

        def action_dislike(self) -> None:
            if tid := self._cur_tid():
                self.state.dislike(tid)
                self._refresh_bar()
                self._refresh_log()
                self._refresh_songs()

        def _ctx_done(self) -> None:
            self._refresh_bar()
            self._refresh_songs()

        def action_pick_vibe(self) -> None:
            def _done(v: str | None) -> None:
                if v is not None:
                    self.state.set_vibe(v)
                    self._ctx_done()
            self.app.push_screen(PickerModal("Mood / genre vibe:", VIBES), _done)

        def action_pick_time(self) -> None:
            def _done(v: str | None) -> None:
                if v is not None:
                    self.state.set_time(v)
                    self._ctx_done()
            self.app.push_screen(PickerModal("Time of day:", list(TIME_HOURS)), _done)

        def action_pick_weather(self) -> None:
            def _done(v: str | None) -> None:
                if v is not None:
                    self.state.set_weather(v)
                    self._ctx_done()
            self.app.push_screen(PickerModal("Current weather:", list(WEATHER_CONTEXTS)), _done)

        def action_pick_place(self) -> None:
            def _done(v: str | None) -> None:
                if v is not None:
                    self.state.set_location(v)
                    self._ctx_done()
            self.app.push_screen(PickerModal("Where are you?", list(LOCATION_CONTEXTS)), _done)

        def action_clear_ctx(self) -> None:
            self.state.clear_context()
            self._ctx_done()

        def action_refresh_recs(self) -> None:
            self.state.soft_dislike_queue(list(self._row_ids))
            self._refresh_bar()
            self._refresh_log()
            self._refresh_songs()

        def action_save(self) -> None:
            self.state.save()
            self._refresh_log()

        def action_quit_session(self) -> None:
            self.state.save()
            self.app.exit()

    # ── main app ─────────────────────────────────────────────────────────────

    class AttuneApp(App):
        TITLE = "Attune"
        CSS = """
        Screen Header { background: $primary; }
        """

        def on_mount(self) -> None:
            # If both profile + tower already exist, skip onboarding
            if (
                user_json_path and user_json_path.exists()
                and tower_path and tower_path.exists()
            ):
                self._load_existing()
            else:
                self.push_screen(OnboardScreen())

        @work(thread=True)
        def _load_existing(self) -> None:
            def log(msg: str) -> None:
                pass  # silent — goes directly to session

            song_model, embeddings, ids, names = load_artifacts_from_hub(hub_repo, device)
            profile = json.loads(user_json_path.read_text())
            user_tower = load_user_tower(tower_path, device)
            state = SessionState(
                profile=profile,
                song_tower=song_model,
                user_tower=user_tower,
                embeddings=embeddings,
                ids=ids,
                names=names,
                device=device,
                save_path=tower_path,
                top_k=top_k,
                spotify_token=spotify_token,
            )
            self.call_from_thread(lambda: self.push_screen(SessionScreen(state)))

    return AttuneApp()


# ── entry point ───────────────────────────────────────────────────────────────

def run_session(args) -> None:
    try:
        import textual  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "textual is required for `ml session`.\n"
            "  pip install 'textual>=0.80.0'"
        )

    device = resolve_device(getattr(args, "device", None))
    hub_repo: str = getattr(args, "hub_repo", None) or DEFAULT_HUB_MODEL_REPO
    top_k: int = getattr(args, "top_k", 20)
    user_json_path: Path | None = Path(args.user_json) if getattr(args, "user_json", None) else None
    tower_path: Path | None = Path(args.user_tower) if getattr(args, "user_tower", None) else None

    if user_json_path is None and DEFAULT_JSON.exists():
        user_json_path = DEFAULT_JSON
    if tower_path is None and DEFAULT_TOWER.exists():
        tower_path = DEFAULT_TOWER

    spotify_token: str | None = (
        getattr(args, "spotify_token", None)
        or __import__("os").environ.get("SPOTIFY_TOKEN")
    )

    app = _build_app(
        hub_repo=hub_repo,
        device=device,
        user_json_path=user_json_path,
        tower_path=tower_path,
        top_k=top_k,
        spotify_token=spotify_token,
    )
    app.run()
