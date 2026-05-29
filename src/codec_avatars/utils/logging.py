"""Lightweight logger: console scalars plus optional TensorBoard."""

from __future__ import annotations

from pathlib import Path


class Logger:
    def __init__(self, out_dir: str, use_tensorboard: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.out_dir / "tb"))
            except Exception:  # tensorboard not installed -> console only
                self.writer = None

    def log(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}{k}", v, step)

    def console(self, step: int, metrics: dict[str, float], extra: str = "") -> None:
        parts = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step:>7}] {parts} {extra}".rstrip(), flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
