"""Unified CLI: train, run, faiss, hub, spotify."""

from __future__ import annotations

import os
import sys


def _usage() -> str:
    return """\
Attune ML (PyTorch)

  python -m ml train   --gpu-preset h100 ...
  python -m ml run     --song-tower ml/export/song_tower_v1.pt ...
  python -m ml faiss   --embeddings ml/export/song_embeddings_v1.npy ...
  python -m ml hub     --repo-id ORG/name --output-dir ml/export
  python -m ml train-user --user-json ml/cli/my_user.json ...
  python -m ml spotify --token ... --out ml/cli/my_user.json

Subcommands match modules; legacy entrypoints still work:

  python -m ml.training.train_song_tower
  python -m ml.cli.test_user  (same as `ml run`)
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

    print(f"Unknown command: {cmd!r}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
