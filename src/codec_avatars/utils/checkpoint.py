"""Checkpoint save/load. The model state_dict carries geometry-stat buffers,
so a checkpoint is self-sufficient for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model, optimizer, step: int, cfg_dict: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": cfg_dict,
        },
        str(path),
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)
