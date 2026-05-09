"""CLI arguments for `ml run` (retrieval smoke test)."""

from __future__ import annotations

import argparse

from ml.inference.demo_users import DEMO_USERS

_VIBES = list(DEMO_USERS.keys())


def build_run_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "ml run",
        description="Run nearest-track retrieval (PyTorch Song/User towers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Vibes (--vibe / --demo-user):\n"
            + "  " + ", ".join(_VIBES)
        ),
    )

    p.add_argument(
        "--song-tower",
        type=str,
        default=None,
        help="Local Song Tower: TorchScript or .pt checkpoint (if set, skips Hub)",
    )
    p.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="Hugging Face model id (default when --song-tower omitted: MrlolDev/attune-v0)",
    )

    p.add_argument("--embeddings", type=str, default=None)
    p.add_argument("--ids", type=str, default=None)
    p.add_argument("--faiss-index", type=str, default=None)

    user = p.add_mutually_exclusive_group()
    user.add_argument(
        "--vibe", "--demo-user",
        dest="demo_user",
        choices=_VIBES,
        metavar="VIBE",
        help=f"Preset vibe query. Choices: {', '.join(_VIBES)}",
    )
    user.add_argument("--user-json", type=str, help="JSON from `ml spotify` (your real profile)")

    p.add_argument("--user-tower", type=str, default=None, help="trained UserTower checkpoint (.pt)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--query-track", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--spotify-token",
        type=str,
        default=None,
        metavar="TOKEN",
        help="Spotify Bearer token to resolve track IDs → names (or set SPOTIFY_TOKEN env var)",
    )
    return p


def parse_run_args(argv: list[str] | None = None):
    return build_run_parser().parse_args(argv)
