"""Dataset factory.

Returns train/val splits plus the geometry standardisation stats and the true
vertex count (the Multiface topology dictates ``n_vertices``, so the model is
sized from the data, not the other way around).
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Subset

from codec_avatars.config import Config
from codec_avatars.data.multiface import MultifaceDataset
from codec_avatars.data.synthetic import SyntheticCodecDataset

__all__ = ["build_datasets", "build_base_dataset", "MultifaceDataset", "SyntheticCodecDataset"]


def _split(ds: Dataset, val_fraction: float, seed: int = 0) -> tuple[Dataset, Dataset]:
    n = len(ds)  # type: ignore[arg-type]
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    if not train_idx:  # tiny datasets: don't starve training
        train_idx, val_idx = perm.tolist(), perm[:1].tolist()
    return Subset(ds, train_idx), Subset(ds, val_idx)


def build_base_dataset(cfg: Config) -> tuple[Dataset, tuple[np.ndarray, float], int]:
    """Build the full (un-split) dataset, its geometry stats, and vertex count."""
    m = cfg.model
    if cfg.data.name == "synthetic":
        base: Dataset = SyntheticCodecDataset(
            n_vertices=m.n_vertices,
            tex_size=m.tex_size,
            tex_channels=m.tex_channels,
            view_dim=m.view_dim,
            size=cfg.data.synthetic_size,
            n_identities=cfg.data.synthetic_identities,
            seed=cfg.train.seed,
        )
        n_vertices = m.n_vertices
    elif cfg.data.name == "multiface":
        base = MultifaceDataset(
            root=cfg.data.root,
            n_vertices=m.n_vertices,
            tex_size=m.tex_size,
            tex_channels=m.tex_channels,
            view_dim=m.view_dim,
            subjects=cfg.data.subjects or None,
            cameras=cfg.data.cameras,
            expressions=cfg.data.expressions,
            max_frames_per_seq=cfg.data.max_frames_per_seq,
        )
        n_vertices = base.n_vertices
    else:
        raise ValueError(f"Unknown dataset '{cfg.data.name}' (expected 'synthetic' or 'multiface')")
    return base, base.geometry_stats(), n_vertices


def build_datasets(cfg: Config) -> tuple[Dataset, Dataset, tuple[np.ndarray, float], int]:
    base, stats, n_vertices = build_base_dataset(cfg)
    train_ds, val_ds = _split(base, cfg.data.val_fraction, seed=cfg.train.seed)
    return train_ds, val_ds, stats, n_vertices
