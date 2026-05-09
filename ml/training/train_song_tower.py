"""
Song Tower training entrypoint.

Prefer the unified CLI:

    python -m ml train ...

Direct module:

    python -m ml.training.train_song_tower ...
"""

from __future__ import annotations

from ml.training.args import parse_train_args
from ml.training.engine import run_training


def main() -> None:
    args = parse_train_args()
    run_training(args)


if __name__ == "__main__":
    main()
