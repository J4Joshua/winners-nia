"""
Attune interactive TUI session.

  python -m ml session \
      --user-json ml/cli/my_user.json \
      --user-tower ml/export/my_user_tower.pt

Keyboard shortcuts
──────────────────
  ↑ / ↓        navigate song list
  l            like highlighted song  → one Adam step toward it
  d            dislike highlighted song → one Adam step away
  c            pick a vibe/mood context
  t            pick a time-of-day context
  x            clear context
  r            refresh recommendations
  s            save updated User Tower
  q            quit (auto-saves)
"""

from __future__ import annotations

import json
import math
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
    "morning": 8,
    "afternoon": 14,
    "evening": 19,
    "night": 23,
}

VIBES: list[str] = list(DEMO_USERS.keys())
IDX_PEAK_SIN = 14
IDX_PEAK_COS = 15
AUDIO_INDICES = list(range(6))  # danceability, energy, valence, acousticness, instr, tempo


# ── session state (pure Python, framework-agnostic) ──────────────────────────

class SessionState:
    """All model state; shared by the TUI and its widgets."""

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

    def set_vibe(self, vibe: str | None) -> None:
        self.context_vibe = vibe

    def set_time(self, period: str | None) -> None:
        self.context_time = period

    def context_label(self) -> str:
        parts = []
        if self.context_time:
            parts.append(self.context_time)
        if self.context_vibe:
            parts.append(self.context_vibe)
        return " + ".join(parts) if parts else "none"

    def _context_features(self) -> list[float]:
        x = list(self.base_features)
        if self.context_time:
            h = TIME_HOURS.get(self.context_time, 14)
            x[IDX_PEAK_SIN] = math.sin(2 * math.pi * h / 24)
            x[IDX_PEAK_COS] = math.cos(2 * math.pi * h / 24)
        if self.context_vibe and self.context_vibe in DEMO_USERS:
            vibe_vec = list(DEMO_USERS[self.context_vibe]["features"])
            alpha = 0.35
            for i in AUDIO_INDICES:
                if i < len(vibe_vec) and i < len(x):
                    x[i] = (1 - alpha) * x[i] + alpha * vibe_vec[i]
        return x

    def _query_embedding(self) -> np.ndarray:
        feats = self._context_features()
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        self.user_tower.eval()
        with torch.no_grad():
            emb = self.user_tower(x).cpu().numpy()[0]
        return emb.astype(np.float32)

    def recommendations(self) -> list[tuple[str, float, str, str]]:
        """Returns [(track_id, score, track_name, artist)]."""
        query = self._query_embedding()
        results = nearest_neighbors(query, self.embeddings, self.ids, top_k=self.top_k)
        out = []
        for tid, score in results:
            display = self.names.get(tid, tid)
            parts = display.split(" — ", 1)
            out.append((tid, score, parts[0], parts[1] if len(parts) > 1 else ""))
        return out

    def _update(self, track_id: str, like: bool) -> float | None:
        if track_id not in self.id_to_emb:
            return None
        feats = self._context_features()
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        track_emb = torch.tensor(
            [self.id_to_emb[track_id]], dtype=torch.float32, device=self.device
        )
        sim = online_update(self.user_tower, self.optimizer, x, track_emb, like=like)
        name = self.names.get(track_id, track_id).split(" — ")[0]
        icon = "♥" if like else "✕"
        self.feedback_log.append(f"{icon} {name[:40]}  →  sim {sim:.3f}")
        return sim

    def like(self, track_id: str) -> float | None:
        return self._update(track_id, like=True)

    def dislike(self, track_id: str) -> float | None:
        return self._update(track_id, like=False)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.user_tower.cpu().state_dict()}, self.save_path)
        self.user_tower.to(self.device)
        self.feedback_log.append(f"💾  Saved → {self.save_path}")


# ── Textual TUI ───────────────────────────────────────────────────────────────

def _build_textual_app(state: SessionState):  # type: ignore[return]
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.screen import ModalScreen
    from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static

    class PickerModal(ModalScreen):
        DEFAULT_CSS = """
        PickerModal { align: center middle; }
        PickerModal > Label {
            margin: 1 2;
            text-style: bold;
            color: $accent;
        }
        PickerModal > ListView {
            width: 48;
            height: auto;
            max-height: 22;
            border: round $accent;
            background: $surface;
        }
        """
        BINDINGS = [("escape", "dismiss_none", "Cancel")]

        def __init__(self, title: str, options: list[str]) -> None:
            super().__init__()
            self._title = title
            self._options = options

        def compose(self) -> ComposeResult:
            yield Label(self._title)
            yield ListView(*[ListItem(Label(f"  {o}"), id=o) for o in self._options])

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            self.dismiss(event.item.id)

        def action_dismiss_none(self) -> None:
            self.dismiss(None)

    class AttuneApp(App):
        TITLE = f"Attune  ·  {state.user_name}"
        CSS = """
        Screen { layout: vertical; }
        #context-bar {
            height: 3;
            background: $surface;
            padding: 0 2;
            border: round $accent;
            content-align: left middle;
        }
        DataTable {
            height: 1fr;
            border: round $primary;
        }
        DataTable > .datatable--cursor { background: $accent 30%; }
        DataTable > .datatable--header { text-style: bold; color: $accent; }
        #feedback-log {
            height: 8;
            padding: 0 2;
            border: round $warning;
            background: $surface-darken-1;
            color: $text-muted;
        }
        """
        BINDINGS: ClassVar[list[Binding]] = [
            Binding("l", "like", "♥ Like"),
            Binding("d", "dislike", "✕ Dislike"),
            Binding("c", "pick_vibe", "Vibe"),
            Binding("t", "pick_time", "Time"),
            Binding("x", "clear_context", "Clear ctx"),
            Binding("r", "refresh_recs", "Refresh"),
            Binding("s", "save", "Save"),
            Binding("q", "quit_session", "Quit"),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("", id="context-bar")
            table = DataTable(id="songs", cursor_type="row", zebra_stripes=True)
            table.add_columns("  #", "Score ", "Track", "Artist")
            yield table
            yield Static("", id="feedback-log")
            yield Footer()

        def on_mount(self) -> None:
            self._row_ids: list[str] = []
            self._update_context_bar()
            self._refresh_songs()

        # ── ui helpers ───────────────────────────────────────────────────────

        def _update_context_bar(self) -> None:
            ctx = state.context_label()
            self.query_one("#context-bar", Static).update(
                f"[bold]{state.user_name}[/bold]   "
                f"Context: [yellow]{ctx}[/yellow]   "
                f"[dim]l=like  d=dislike  c=vibe  t=time  x=clear  r=refresh  s=save  q=quit[/dim]"
            )

        def _refresh_songs(self) -> None:
            recs = state.recommendations()
            table = self.query_one("#songs", DataTable)
            table.clear()
            self._row_ids = []
            for i, (tid, score, track, artist) in enumerate(recs, 1):
                table.add_row(f"{i:>3}", f"{score:.4f}", track, artist)
                self._row_ids.append(tid)

        def _update_log(self) -> None:
            lines = state.feedback_log[-7:]
            self.query_one("#feedback-log", Static).update("\n".join(lines))

        def _current_tid(self) -> str | None:
            table = self.query_one("#songs", DataTable)
            idx = table.cursor_row
            return self._row_ids[idx] if 0 <= idx < len(self._row_ids) else None

        # ── actions ──────────────────────────────────────────────────────────

        def action_like(self) -> None:
            tid = self._current_tid()
            if tid:
                state.like(tid)
                self._update_log()
                self._refresh_songs()

        def action_dislike(self) -> None:
            tid = self._current_tid()
            if tid:
                state.dislike(tid)
                self._update_log()
                self._refresh_songs()

        def action_pick_vibe(self) -> None:
            def _done(vibe: str | None) -> None:
                if vibe is not None:
                    state.set_vibe(vibe)
                    self._update_context_bar()
                    self._refresh_songs()
            self.push_screen(PickerModal("Choose a vibe:", VIBES), _done)

        def action_pick_time(self) -> None:
            def _done(period: str | None) -> None:
                if period is not None:
                    state.set_time(period)
                    self._update_context_bar()
                    self._refresh_songs()
            self.push_screen(PickerModal("Choose time of day:", list(TIME_HOURS)), _done)

        def action_clear_context(self) -> None:
            state.set_vibe(None)
            state.set_time(None)
            self._update_context_bar()
            self._refresh_songs()

        def action_refresh_recs(self) -> None:
            self._refresh_songs()

        def action_save(self) -> None:
            state.save()
            self._update_log()

        def action_quit_session(self) -> None:
            state.save()
            self.exit()

    return AttuneApp()


# ── entry point ───────────────────────────────────────────────────────────────

def run_session(args) -> None:
    try:
        import textual  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "textual is required for `ml session`.\n"
            "Install it: pip install textual>=0.80.0"
        )

    device = resolve_device(getattr(args, "device", None))

    print("Loading catalog from Hugging Face Hub…")
    repo = getattr(args, "hub_repo", None) or DEFAULT_HUB_MODEL_REPO
    _, embeddings, ids, names = load_artifacts_from_hub(repo, device)
    print(f"  {len(ids):,} tracks  ({len(names):,} names)")

    profile = json.loads(Path(args.user_json).read_text())
    user_tower = load_user_tower(args.user_tower, device)

    state = SessionState(
        profile=profile,
        user_tower=user_tower,
        embeddings=embeddings,
        ids=ids,
        names=names,
        device=device,
        save_path=Path(args.user_tower),
        top_k=getattr(args, "top_k", 20),
    )

    app = _build_textual_app(state)
    app.run()
