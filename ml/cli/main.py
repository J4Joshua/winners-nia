"""Unified CLI: train, run, faiss, hub, spotify."""

from __future__ import annotations

import os
import sys


def _usage() -> str:
    return """\
Attune ML (PyTorch)

  python -m ml train      --gpu-preset h100 ...
  python -m ml run        --vibe chill
  python -m ml session    --user-json ml/cli/my_user.json --user-tower ml/export/my_user_tower.pt
  python -m ml faiss      --embeddings ml/export/song_embeddings_v1.npy ...
  python -m ml hub        --repo-id ORG/name --output-dir ml/export
  python -m ml train-user --user-json ml/cli/my_user.json ...
  python -m ml spotify    --client-id ... --client-secret ...

`ml session` opens an interactive TUI:
  ↑↓ navigate · l like · d dislike · c vibe · t time · s save · q quit
"""


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(_usage())
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd in ("-h", "--help"):
        print(_usage())
        sys.exit(0)

    if cmd == "train":
        from ml.training.args import parse_train_args
        from ml.training.engine import run_training

        args = parse_train_args(rest)
        run_training(args)
        return

    if cmd == "run":
        from ml.cli.run_args import parse_run_args
        from ml.inference.run import run_inference

        args = parse_run_args(rest)
        run_inference(args)
        return

    if cmd == "faiss":
        from ml.export.build_faiss import build_index

        p = argparse.ArgumentParser(prog="ml faiss", description="Build FAISS IndexFlatIP")
        p.add_argument("--embeddings", default="ml/export/song_embeddings_v1.npy")
        p.add_argument("--ids", default="ml/export/song_ids_v1.npy")
        p.add_argument("--output", default="ml/export/faiss_song_v1.index")
        fa = p.parse_args(rest)
        build_index(fa.embeddings, fa.ids, fa.output)
        return

    if cmd == "hub":
        from ml.export.push_to_hub import push_song_tower

        p = argparse.ArgumentParser(prog="ml hub", description="Push artifacts to HuggingFace Hub")
        p.add_argument("--repo-id", required=True)
        p.add_argument("--output-dir", default="ml/export")
        p.add_argument("--token", default=None)
        p.add_argument("--private", action="store_true")
        ha = p.parse_args(rest)
        token = ha.token or os.environ.get("HF_TOKEN")
        push_song_tower(ha.repo_id, ha.output_dir, token=token, private=ha.private)
        return

    if cmd == "train-user":
        from ml.training.train_user_cold import parse_train_user_args, train_user_cold

        args = parse_train_user_args(rest)
        train_user_cold(args)
        return

    if cmd == "spotify":
        from ml.cli.spotify_profile import parse_spotify_args, run_spotify_profile

        args = parse_spotify_args(rest)
        run_spotify_profile(args)
        return

    if cmd == "session":
        import argparse
        from ml.cli.session import run_session
        from ml.inference.demo_users import DEMO_USERS

        p = argparse.ArgumentParser(prog="ml session", description="Interactive Attune TUI")
        p.add_argument("--user-json", required=True, help="Profile JSON from `ml spotify`")
        p.add_argument("--user-tower", required=True, help="User Tower checkpoint (.pt)")
        p.add_argument("--hub-repo", default=None, help="HF repo (default: MrlolDev/attune-v0)")
        p.add_argument("--top-k", type=int, default=20, help="Songs to show")
        p.add_argument("--device", default=None)
        run_session(p.parse_args(rest))
        return

    print(f"Unknown command: {cmd!r}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
