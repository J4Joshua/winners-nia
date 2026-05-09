"""GPU presets for Song Tower training (batch size, workers, torch.compile)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuPreset:
    key: str
    batch_size: int
    num_workers: int
    compile_model: bool
    description: str


GPU_PRESETS: dict[str, GpuPreset] = {
    "h100": GpuPreset(
        key="h100",
        batch_size=2048,
        num_workers=8,
        compile_model=True,
        description="NVIDIA H100 80GB — 100 epochs ≈ 10–15 min; SupCon loves large batches",
    ),
    "a100": GpuPreset(
        key="a100",
        batch_size=1024,
        num_workers=8,
        compile_model=True,
        description="NVIDIA A100 40/80GB — 100 epochs ≈ 20–30 min",
    ),
    "4090": GpuPreset(
        key="4090",
        batch_size=512,
        num_workers=4,
        compile_model=False,
        description="RTX 4090 24GB — 100 epochs ≈ 40–60 min",
    ),
    "l4": GpuPreset(
        key="l4",
        batch_size=256,
        num_workers=4,
        compile_model=False,
        description="NVIDIA L4 24GB — 100 epochs ≈ 60–90 min",
    ),
}


def apply_gpu_preset(name: str | None, args: Any) -> None:
    """Mutate argparse Namespace: batch_size, num_workers, compile."""
    if not name:
        return
    preset = GPU_PRESETS[name]
    args.batch_size = preset.batch_size
    args.num_workers = preset.num_workers
    if preset.compile_model:
        args.compile = True
