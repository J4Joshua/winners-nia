"""Backward-compatible entrypoint for retrieval (`python -m ml.cli.test_user`)."""

from __future__ import annotations

from ml.cli.run_args import parse_run_args
from ml.inference.run import run_inference


def main() -> None:
    args = parse_run_args()
    run_inference(args)


if __name__ == "__main__":
    main()
