"""Training loop, checkpoints, and export for Song Tower (PyTorch)."""

from __future__ import annotations

import math
import os
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ml.models.song_tower import (
    SongTower, augment_features, supcon_loss,
    uniformity_loss, prototype_alignment_loss,
)
from ml.training.dataset import SpotifyTracksDataset, load_spotify_dataset
from ml.export.push_to_hub import push_song_tower


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _unwrap_for_export(model: nn.Module) -> nn.Module:
    """torch.compile and some wrappers need unwrapping for torch.jit.trace."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod  # type: ignore[return-value, no-any-return]
    return model



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    tau: float,
    use_amp: bool,
    noise_std: float = 0.04,
    uniformity_weight: float = 0.5,
    prototype_weight: float = 0.3,
) -> float:
    """
    Triple-objective training: SupCon + Uniformity + Prototype alignment.

    Loss = SupCon(v1‖v2, labels×2, τ)    — genre cluster separation
         + λ_u · Uniformity(v1‖v2)        — spread over hypersphere
         + λ_p · Prototype(v1, labels)     — pull toward genre centroid

    SupCon:    genres cluster; two-view augmentation adds robustness
    Uniformity: prevents mode collapse (Wang & Isola 2020)
    Prototype:  stabilizes intra-genre structure, reduces cluster variance
    τ-anneal:   τ starts high (easy: coarse genre structure) → low (fine: subtle differences)
    """
    model.train()
    total_loss = 0.0

    for x, genre_labels in loader:
        x = x.to(device, non_blocking=True)
        genre_labels = genre_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            v1 = model(augment_features(x, noise_std))
            v2 = model(augment_features(x, noise_std))
            z          = torch.cat([v1, v2], dim=0)
            labels_2x  = torch.cat([genre_labels, genre_labels], dim=0)
            loss_supcon    = supcon_loss(z, labels_2x, temperature=tau)
            loss_uni       = uniformity_loss(z)
            loss_proto     = prototype_alignment_loss(v1, genre_labels)
            loss = loss_supcon + uniformity_weight * loss_uni + prototype_weight * loss_proto

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    use_amp: bool,
) -> float:
    """Validation: SupCon on clean features (no noise, eval mode dropout off)."""
    model.eval()
    total_loss = 0.0

    for x, genre_labels in loader:
        x = x.to(device, non_blocking=True)
        genre_labels = genre_labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            z = model(x)
            # Duplicate to form two "views" so SupCon can compute cross-view positives
            z2 = torch.cat([z, z], dim=0)
            labels_2x = torch.cat([genre_labels, genre_labels], dim=0)
            loss = supcon_loss(z2, labels_2x, temperature=tau)
        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _unwrap_for_export(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    _unwrap_for_export(model).load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0)


@torch.no_grad()
def export_embeddings(
    model: nn.Module,
    train_ds: SpotifyTracksDataset,
    val_ds: SpotifyTracksDataset,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    track_names: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    import json as _json

    inner = _unwrap_for_export(model)
    inner.eval()

    all_ds = ConcatDataset([train_ds, val_ds])
    all_ids = train_ds.track_ids + val_ds.track_ids
    loader = DataLoader(
        all_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    chunks: list[torch.Tensor] = []
    for x, _genre in loader:
        x = x.to(device)
        chunks.append(inner(x).cpu())

    embeddings = torch.cat(chunks, dim=0).numpy()
    ids_array = np.array(all_ids, dtype=object)

    np.save(output_dir / "song_embeddings_v1.npy", embeddings)
    np.save(output_dir / "song_ids_v1.npy", ids_array)
    print(f"Saved embeddings {embeddings.shape} → {output_dir}/song_embeddings_v1.npy")

    if track_names:
        names_path = output_dir / "song_names_v1.json"
        names_path.write_text(_json.dumps(track_names, ensure_ascii=False))
        print(f"Saved track names ({len(track_names):,}) → {names_path}")

    return embeddings, all_ids


def export_torchscript(model: nn.Module, output_dir: Path, device: torch.device) -> Path:
    inner = _unwrap_for_export(model)
    inner.eval()
    # Warm up torch.compile once here at export time (single forward, no CUDAGraph issue).
    if hasattr(torch, "compile"):
        inner = torch.compile(inner, mode="reduce-overhead")  # type: ignore[assignment]
    from ml.training.dataset import SONG_FEATURE_DIM
    example = torch.zeros(1, SONG_FEATURE_DIM, device=device)
    with torch.no_grad():
        inner(example)  # trigger compile
    scripted = torch.jit.trace(_unwrap_for_export(inner), example)
    out_path = output_dir / "song_tower_v1.pt"
    scripted.save(str(out_path))
    print(f"TorchScript saved → {out_path}")
    return out_path


def run_training(args: Namespace) -> None:
    """Full Song Tower train + export; driven by argparse Namespace from train_args()."""
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}  compile: {args.compile}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _all_ids, track_names = load_spotify_dataset(
        val_fraction=args.val_fraction,
        seed=args.seed,
        cache_dir=args.hf_cache,
        revision=args.hf_revision,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model: nn.Module = SongTower(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # torch.compile is intentionally skipped during training.
    # InfoNCE calls model(x) twice per batch; torch.compile with CUDAGraphs aliases the two
    # output buffers, causing "tensor output of CUDAGraphs overwritten" on any PyTorch build.
    # For this MLP the kernel overhead is negligible — AMP + H100 is already fast.
    # Compile is applied to the unwrapped model at export time (export_torchscript).
    if args.compile:
        print("Note: --compile is ignored during training (applied at export). See engine.py.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, warmup_fraction=args.warmup_frac)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
            print(f"Resumed from {resume_path} at epoch {start_epoch}")

    print(f"\nTraining for {args.epochs} epochs (resuming from {start_epoch})...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'τ':>7}  {'LR':>8}  {'Time':>6}")
    print("─" * 62)

    tau_start = getattr(args, "tau",     0.15)
    tau_end   = getattr(args, "tau_end", 0.05)
    uni_w     = getattr(args, "uniformity_weight", 0.5)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Cosine temperature annealing: warm start (coarse genre clusters) → cool end (fine-grained)
        progress = (epoch - start_epoch) / max(1, args.epochs - 1)
        tau = tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * progress))

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, tau, use_amp,
            noise_std=getattr(args, "noise_std", 0.04),
            uniformity_weight=uni_w,
            prototype_weight=getattr(args, "prototype_weight", 0.3),
        )
        val_loss = evaluate(model, val_loader, device, tau, use_amp)

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(
            f"{epoch + 1:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}"
            f"  τ={tau:.3f}  {current_lr:>8.2e}  {elapsed:>6.1f}s"
        )

        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch + 1, val_loss)
            print(f"  checkpoint → {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "song_tower_best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch + 1, val_loss)
            print(f"  new best val loss {best_val_loss:.4f} → {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    best_path = output_dir / "song_tower_best.pt"
    if best_path.exists():
        load_checkpoint(best_path, model, None, None, None, device)
        print("Loaded best checkpoint for export")

    export_torchscript(model, output_dir, device)

    if args.export_embeddings:
        print("\nExporting all song embeddings...")
        export_embeddings(
            model,
            train_ds,
            val_ds,
            output_dir,
            device,
            batch_size=args.embed_batch_size,
            num_workers=args.num_workers,
            track_names=track_names,
        )

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        token = args.hub_token or os.environ.get("HF_TOKEN")
        push_song_tower(
            repo_id=args.hub_repo,
            output_dir=output_dir,
            token=token,
        )

    print("\nDone.")
