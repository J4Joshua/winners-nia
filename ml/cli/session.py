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
  l           ♥ like highlighted song  (one Adam step toward it)
  d           ✕ dislike highlighted song  (one Adam step away)
  c           pick vibe / mood context
  t           pick time-of-day context
  x           clear context
  r           refresh rankings
  s           save tower
  q           quit (auto-saves)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from ml.cli.feedback import online_update
from ml.inference.defaults import DEFAULT_HUB_MODEL_REPO
from ml.inference.demo_users import DEMO_USERS
from ml.inference.device import resolve_device
from ml.inference.loaders import load_artifacts_from_hub, load_user_tower
from ml.inference.search import nearest_neighbors

# ── constants ─────────────────────────────────────────────────────────────────

TIME_HOURS: dict[str, int] = {
    "morning": 8, "afternoon": 14, "evening": 19, "night": 23,
}
VIBES: list[str] = list(DEMO_USERS.keys())
IDX_PEAK_SIN, IDX_PEAK_COS = 14, 15
AUDIO_INDICES = list(range(6))

DEFAULT_JSON = Path("ml/cli/my_user.json")
DEFAULT_TOWER = Path("ml/export/my_user_tower.pt")


# ── pure-Python session state ─────────────────────────────────────────────────

class SessionState:
    def __init__(
        self,
        profile: dict,
        user_tower: nn.Module,
        embeddings: np.ndarray,
        ids: np.ndarray,
        names: dict[str, str],
        device: torch.device,
        save_path: Path,
        top_k: int = 20,
    ) -> None:
        self.user_name: str = profile.get("name", "User")
        self.base_features: list[float] = list(profile["features"])
        self.user_tower = user_tower
        self.embeddings = embeddings
        self.ids = ids
        self.names = names
        self.device = device
        self.save_path = save_path
        self.top_k = top_k
        self.id_to_emb: dict[str, np.ndarray] = {
            str(ids[i]): embeddings[i] for i in range(len(ids))
        }
        self.optimizer = torch.optim.Adam(user_tower.parameters(), lr=1e-4)
        self.context_vibe: str | None = None
        self.context_time: str | None = None
        self.feedback_log: list[str] = []

    def set_vibe(self, v: str | None) -> None: self.context_vibe = v
    def set_time(self, t: str | None) -> None: self.context_time = t

    def context_label(self) -> str:
        parts = [p for p in [self.context_time, self.context_vibe] if p]
        return " + ".join(parts) if parts else "none"

    def _ctx_features(self) -> list[float]:
        x = list(self.base_features)
        if self.context_time:
            h = TIME_HOURS.get(self.context_time, 14)
            x[IDX_PEAK_SIN] = math.sin(2 * math.pi * h / 24)
            x[IDX_PEAK_COS] = math.cos(2 * math.pi * h / 24)
        if self.context_vibe and self.context_vibe in DEMO_USERS:
            vv = list(DEMO_USERS[self.context_vibe]["features"])
            for i in AUDIO_INDICES:
                if i < len(vv) and i < len(x):
                    x[i] = 0.65 * x[i] + 0.35 * vv[i]
        return x

    def _query(self) -> np.ndarray:
        x = torch.tensor([self._ctx_features()], dtype=torch.float32, device=self.device)
        self.user_tower.eval()
        with torch.no_grad():
            return self.user_tower(x).cpu().numpy()[0].astype(np.float32)

    def recommendations(self) -> list[tuple[str, float, str, str]]:
        results = nearest_neighbors(self._query(), self.embeddings, self.ids, top_k=self.top_k)
        out = []
        for tid, score in results:
            d = self.names.get(tid, tid).split(" — ", 1)
            out.append((tid, score, d[0], d[1] if len(d) > 1 else ""))
        return out

    def _update(self, tid: str, like: bool) -> float | None:
        if tid not in self.id_to_emb:
            return None
        x = torch.tensor([self._ctx_features()], dtype=torch.float32, device=self.device)
        te = torch.tensor([self.id_to_emb[tid]], dtype=torch.float32, device=self.device)
        sim = online_update(self.user_tower, self.optimizer, x, te, like=like)
        name = self.names.get(tid, tid).split(" — ")[0]
        self.feedback_log.append(f"{'♥' if like else '✕'}  {name[:40]}   sim → {sim:.3f}")
        return sim

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
    redirect_uri: str,
):
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.screen import Screen, ModalScreen
    from textual.widgets import (
        Button, DataTable, Footer, Header, Input,
        Label, ListItem, ListView, Static,
    )
    from textual.worker import WorkerState

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
            yield ListView(*[ListItem(Label(f"  {o}"), id=o) for o in self._options])

        def on_list_view_selected(self, e: ListView.Selected) -> None:
            self.dismiss(e.item.id)

        def action_dismiss_none(self) -> None:
            self.dismiss(None)

    # ── onboard screen ───────────────────────────────────────────────────────

    class OnboardScreen(Screen):
        CSS = """
        OnboardScreen {
            align: center middle;
            layout: vertical;
        }
        #card {
            width: 66;
            height: auto;
            border: round $accent;
            padding: 1 3;
            background: $surface;
        }
        #card Label { margin: 1 0 0 0; color: $text-muted; }
        #card Input { margin: 0 0 1 0; }
        #submit { margin-top: 1; width: 100%; }
        #note { color: $text-muted; text-align: center; margin-top: 1; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            with Static(id="card"):
                yield Label("🎵  Attune — connect your Spotify account", id="title")
                yield Label("Client ID")
                yield Input(placeholder="756b281cb305…", id="cid")
                yield Label("Client Secret")
                yield Input(placeholder="••••••••••••", id="cs", password=True)
                yield Label("Redirect URI")
                yield Input(value=redirect_uri, id="ruri")
                yield Button("Connect →  (browser will open)", id="submit", variant="primary")
                yield Static(
                    "[dim]Get credentials: developer.spotify.com → Dashboard → your app[/dim]",
                    id="note",
                )
            yield Footer()

        def on_button_pressed(self, e: Button.Pressed) -> None:
            if e.button.id == "submit":
                cid = self.query_one("#cid", Input).value.strip()
                cs = self.query_one("#cs", Input).value.strip()
                ruri = self.query_one("#ruri", Input).value.strip()
                if not cid or not cs:
                    self.query_one("#note", Static).update("[red]Both Client ID and Secret are required[/red]")
                    return
                self.app.push_screen(ProgressScreen(cid, cs, ruri))

    # ── progress screen ──────────────────────────────────────────────────────

    class ProgressScreen(Screen):
        CSS = """
        ProgressScreen { align: center middle; layout: vertical; }
        #log {
            width: 70; height: 20;
            border: round $primary;
            padding: 1 2;
            background: $surface;
        }
        """

        def __init__(self, client_id: str, client_secret: str, ruri: str) -> None:
            super().__init__()
            self._cid, self._cs, self._ruri = client_id, client_secret, ruri
            self._lines: list[str] = []

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("", id="log")
            yield Footer()

        def on_mount(self) -> None:
            self._log("⏳  Connecting to Spotify…  (your browser will open)")
            self._run_onboarding()

        def _log(self, msg: str) -> None:
            self._lines.append(msg)
            self.query_one("#log", Static).update("\n".join(self._lines[-18:]))

        @work(thread=True)
        def _run_onboarding(self) -> None:
            def log(msg: str) -> None:
                self.app.call_from_thread(self._log, msg)

            try:
                # 1. Load catalog
                log("📦  Loading song catalog from Hub…")
                model, embeddings, ids, names = load_artifacts_from_hub(hub_repo, device)
                catalog["model"] = model
                catalog["embeddings"] = embeddings
                catalog["ids"] = ids
                catalog["names"] = names
                log(f"  ✓  {len(ids):,} tracks  ({len(names):,} named)")

                # 2. Spotify OAuth
                log("🔐  Authenticating with Spotify…")
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth
                scope = "user-top-read user-read-recently-played user-library-read"
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id=self._cid,
                    client_secret=self._cs,
                    redirect_uri=self._ruri,
                    scope=scope,
                ))
                me = sp.me()
                display_name = me.get("display_name", "User")
                log(f"  ✓  Logged in as [bold]{display_name}[/bold]")

                # 3. Fetch listening data
                from ml.cli.spotify_profile import (
                    compute_user_features, fetch_audio_features_spotify,
                    fetch_liked_tracks_http, genre_diversity_from_top_tracks,
                    hydrate_top_tracks_http,
                )
                from ml.cli.spotify_audio import spotify_access_token

                log("🎵  Fetching top tracks…")
                top = sp.current_user_top_tracks(limit=50, time_range="medium_term").get("items", [])
                top = hydrate_top_tracks_http(spotify_access_token(sp), top)
                log(f"  ✓  {len(top)} top tracks")

                log("🎵  Fetching audio features…")
                af = fetch_audio_features_spotify(sp, [t["id"] for t in top if t.get("id")])

                log("🎵  Fetching genre diversity…")
                gdiv = genre_diversity_from_top_tracks(spotify_access_token(sp), top)

                log("🎵  Fetching recently played…")
                recent = sp.current_user_recently_played(limit=50).get("items", [])
                log(f"  ✓  {len(recent)} plays")

                log("🎵  Fetching liked songs (up to 2000)…")
                liked = fetch_liked_tracks_http(spotify_access_token(sp), limit=2000)
                log(f"  ✓  {len(liked)} liked tracks")

                # 4. Compute profile + save JSON
                profile = compute_user_features(
                    top_tracks=top, audio_features=af, recent_tracks=recent,
                    display_name=display_name, genre_diversity_score=gdiv,
                    liked_track_ids=liked,
                )
                json_path = DEFAULT_JSON
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(profile, indent=2))
                log(f"  ✓  Profile saved → {json_path}")

                # 5. Train user tower
                log("🧠  Training User Tower…")
                from ml.training.train_user_cold import train_user_cold
                import argparse
                ta = argparse.Namespace(
                    user_json=str(json_path),
                    output=str(DEFAULT_TOWER),
                    hub_repo=None,
                    embeddings=None,
                    ids=None,
                    positives="all",
                    epochs=500,
                    lr=5e-3,
                    seed=42,
                    min_positives=3,
                    cpu=device.type == "cpu",
                )
                train_user_cold(ta)
                log(f"  ✓  User Tower saved → {DEFAULT_TOWER}")

                # 6. Load tower + enter session
                log("✨  All done! Entering session…")
                user_tower = load_user_tower(DEFAULT_TOWER, device)
                state = SessionState(
                    profile=profile,
                    user_tower=user_tower,
                    embeddings=embeddings,
                    ids=ids,
                    names=names,
                    device=device,
                    save_path=DEFAULT_TOWER,
                    top_k=top_k,
                )
                self.app.call_from_thread(
                    lambda: self.app.switch_screen(SessionScreen(state))
                )

            except Exception as exc:
                self.app.call_from_thread(self._log, f"[red]Error: {exc}[/red]")

    # ── session screen ───────────────────────────────────────────────────────

    class SessionScreen(Screen):
        CSS = """
        SessionScreen { layout: vertical; }
        #context-bar {
            height: 3; background: $surface; padding: 0 2;
            border: round $accent; content-align: left middle;
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
            Binding("l", "like", "♥ Like"),
            Binding("d", "dislike", "✕ Dislike"),
            Binding("c", "pick_vibe", "Vibe"),
            Binding("t", "pick_time", "Time"),
            Binding("x", "clear_ctx", "Clear"),
            Binding("r", "refresh_recs", "Refresh"),
            Binding("s", "save", "Save"),
            Binding("q", "quit_session", "Quit"),
        ]

        def __init__(self, state: SessionState) -> None:
            super().__init__()
            self.state = state
            self._row_ids: list[str] = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("", id="context-bar")
            t = DataTable(id="songs", cursor_type="row", zebra_stripes=True)
            t.add_columns("  #", "Score ", "Track", "Artist")
            yield t
            yield Static("", id="feedback-log")
            yield Footer()

        def on_mount(self) -> None:
            self._refresh_bar()
            self._refresh_songs()

        def _refresh_bar(self) -> None:
            self.query_one("#context-bar", Static).update(
                f"[bold]{self.state.user_name}[/bold]   "
                f"Context: [yellow]{self.state.context_label()}[/yellow]   "
                f"[dim]l=like  d=dislike  c=vibe  t=time  x=clear  r=refresh  s=save  q=quit[/dim]"
            )

        def _refresh_songs(self) -> None:
            recs = self.state.recommendations()
            table = self.query_one("#songs", DataTable)
            table.clear()
            self._row_ids = []
            for i, (tid, score, track, artist) in enumerate(recs, 1):
                table.add_row(f"{i:>3}", f"{score:.4f}", track, artist)
                self._row_ids.append(tid)

        def _refresh_log(self) -> None:
            self.query_one("#feedback-log", Static).update(
                "\n".join(self.state.feedback_log[-7:])
            )

        def _cur_tid(self) -> str | None:
            t = self.query_one("#songs", DataTable)
            return self._row_ids[t.cursor_row] if 0 <= t.cursor_row < len(self._row_ids) else None

        def action_like(self) -> None:
            if tid := self._cur_tid():
                self.state.like(tid)
                self._refresh_log()
                self._refresh_songs()

        def action_dislike(self) -> None:
            if tid := self._cur_tid():
                self.state.dislike(tid)
                self._refresh_log()
                self._refresh_songs()

        def action_pick_vibe(self) -> None:
            def _done(v: str | None) -> None:
                if v:
                    self.state.set_vibe(v)
                    self._refresh_bar()
                    self._refresh_songs()
            self.app.push_screen(PickerModal("Choose a vibe:", VIBES), _done)

        def action_pick_time(self) -> None:
            def _done(p: str | None) -> None:
                if p:
                    self.state.set_time(p)
                    self._refresh_bar()
                    self._refresh_songs()
            self.app.push_screen(PickerModal("Choose time of day:", list(TIME_HOURS)), _done)

        def action_clear_ctx(self) -> None:
            self.state.set_vibe(None)
            self.state.set_time(None)
            self._refresh_bar()
            self._refresh_songs()

        def action_refresh_recs(self) -> None:
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

            _, embeddings, ids, names = load_artifacts_from_hub(hub_repo, device)
            profile = json.loads(user_json_path.read_text())
            user_tower = load_user_tower(tower_path, device)
            state = SessionState(
                profile=profile,
                user_tower=user_tower,
                embeddings=embeddings,
                ids=ids,
                names=names,
                device=device,
                save_path=tower_path,
                top_k=top_k,
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
    redirect_uri: str = getattr(args, "redirect_uri", "http://127.0.0.1:8888/callback")

    user_json_path: Path | None = Path(args.user_json) if getattr(args, "user_json", None) else None
    tower_path: Path | None = Path(args.user_tower) if getattr(args, "user_tower", None) else None

    # Auto-detect existing files if not explicitly provided
    if user_json_path is None and DEFAULT_JSON.exists():
        user_json_path = DEFAULT_JSON
    if tower_path is None and DEFAULT_TOWER.exists():
        tower_path = DEFAULT_TOWER

    app = _build_app(
        hub_repo=hub_repo,
        device=device,
        user_json_path=user_json_path,
        tower_path=tower_path,
        top_k=top_k,
        redirect_uri=redirect_uri,
    )
    app.run()
