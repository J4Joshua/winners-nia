"""CLI arguments for `ml run` (retrieval smoke test)."""

from __future__ import annotations

import argparse

from ml.inference.demo_users import DEMO_USERS


def build_run_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "ml run",
        description="Run nearest-track retrieval (PyTorch Song/User towers)",
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
    user.add_argument("--demo-user", choices=list(DEMO_USERS.keys()))
    user.add_argument("--user-json", type=str)

    p.add_argument("--user-tower", type=str, default=None, help="trained UserTower checkpoint (.pt)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--query-track", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p


def parse_run_args(argv: list[str] | None = None):
    return build_run_parser().parse_args(argv)
