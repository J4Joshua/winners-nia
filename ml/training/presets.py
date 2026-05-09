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
        description="NVIDIA H100 80GB — recommended for hackathon speed (~5–10 min)",
    ),
    "a100": GpuPreset(
        key="a100",
        batch_size=1024,
        num_workers=8,
        compile_model=True,
        description="NVIDIA A100 40/80GB — strong balance (~8–15 min)",
    ),
    "4090": GpuPreset(
        key="4090",
        batch_size=512,
        num_workers=4,
        compile_model=False,
        description="RTX 4090 24GB — default consumer GPU (~20–40 min)",
    ),
    "l4": GpuPreset(
        key="l4",
        batch_size=256,
        num_workers=4,
        compile_model=False,
        description="NVIDIA L4 24GB — serverless-friendly (~30–60 min)",
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
