"""Argument parsers for Song Tower training."""

from __future__ import annotations

import argparse

from ml.inference.defaults import DEFAULT_HUB_MODEL_REPO
from ml.training.presets import GPU_PRESETS


def build_train_parser() -> argparse.ArgumentParser:
    epilog_lines = [
        "GPU presets (--gpu-preset):",
        *(f"  {p.key:8}  batch={p.batch_size:<5}  compile={'yes' if p.compile_model else 'no ':5}  {p.description}"
          for p in GPU_PRESETS.values()),
        "",
        "Recommended: H100 or A100 for fastest runs (large batches help InfoNCE).",
        "Consumer 4090/L4 work fine with smaller batches.",
    ]

    p = argparse.ArgumentParser(
        description="Train Song Tower (PyTorch, InfoNCE, HuggingFace dataset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )

    p.add_argument(
        "--gpu-preset",
        choices=list(GPU_PRESETS.keys()),
        default=None,
        help="Apply batch_size / num_workers / compile for your GPU",
    )
    p.add_argument("--epochs", type=int, default=100,
                   help="100 epochs ≈ 15–30 min on H100. More = better genre clustering.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Larger batches = more SupCon positives per step = better training")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.15,
                   help="Initial SupCon temperature — annealed to --tau-end over training")
    p.add_argument("--tau-end", type=float, default=0.05,
                   help="Final SupCon temperature (warm→cold annealing helps learn coarse then fine structure)")
    p.add_argument("--uniformity-weight", type=float, default=0.5,
                   help="Weight for uniformity loss (Wang & Isola 2020); prevents embedding collapse")
    p.add_argument("--prototype-weight", type=float, default=0.3,
                   help="Weight for prototype alignment loss; pulls embeddings toward genre centroids")
    p.add_argument("--noise-std", type=float, default=0.04,
                   help="Gaussian noise std on audio scalar augmentation (0 = disable)")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--embed-batch-size", type=int, default=4096, help="Batch size for embedding export pass")
    p.add_argument("--output-dir", type=str, default="ml/export")
    p.add_argument("--hf-cache", type=str, default=None)
    p.add_argument("--hf-revision", type=str, default=None)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile (recommended on H100/A100)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument(
        "--hub-repo",
        type=str,
        default=DEFAULT_HUB_MODEL_REPO,
        help="Hugging Face model repo for --push-to-hub (org/name)",
    )
    p.add_argument("--hub-token", type=str, default=None)
    p.add_argument("--export-embeddings", action="store_true", default=True)
    p.add_argument("--no-export-embeddings", dest="export_embeddings", action="store_false")
    return p


def parse_train_args(argv: list[str] | None = None):
    from ml.training.presets import apply_gpu_preset

    args = build_train_parser().parse_args(argv)
    if args.gpu_preset:
        preset = GPU_PRESETS[args.gpu_preset]
        apply_gpu_preset(args.gpu_preset, args)
        print(f"GPU preset [{args.gpu_preset}]: {preset.description}")
        print(f"  batch_size={args.batch_size}  num_workers={args.num_workers}  compile={args.compile}")
    return args
