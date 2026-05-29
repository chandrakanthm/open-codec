"""In-memory synthetic dataset matching the Codec Avatar batch contract.

No download required. Geometry autoencodes; texture is a deterministic
view-dependent shading of a per-identity base texture, so the VAE has a real
mapping to learn. Used by unit tests and the CPU smoke run.

Batch contract (per item, before collation):
    verts    : (V, 3)      standardized geometry
    avg_tex  : (C, H, W)   view-averaged texture in [0,1]   (encoder input)
    tex      : (C, H, W)   this view's texture in [0,1]      (decoder target)
    view     : (3,)        unit view direction
    tex_mask : (1, H, W)   valid-texel mask in {0,1}
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticCodecDataset(Dataset):
    def __init__(
        self,
        n_vertices: int,
        tex_size: int,
        tex_channels: int = 3,
        view_dim: int = 3,
        size: int = 64,
        n_identities: int = 4,
        seed: int = 0,
        expr_scale: float = 2.0,
    ):
        self.n_vertices = n_vertices
        self.tex_size = tex_size
        self.tex_channels = tex_channels
        self.view_dim = view_dim
        self.size = size
        self.n_identities = max(1, n_identities)
        self.seed = seed
        self.expr_scale = expr_scale

        rng = np.random.default_rng(seed)
        self.mean_face = rng.normal(0.0, 50.0, size=(n_vertices, 3)).astype(np.float32)
        self.id_offset = rng.normal(0.0, 5.0, size=(self.n_identities, n_vertices, 3)).astype(np.float32)
        self.base_tex = rng.uniform(0.2, 0.8, size=(self.n_identities, tex_channels, tex_size, tex_size)).astype(np.float32)

        verts_all = self.mean_face[None] + self.id_offset  # (n_id, V, 3)
        self._mean = verts_all.mean(axis=0).astype(np.float32)  # mean face (V, 3)
        self._std = float(verts_all.std() + 1e-6)

    def geometry_stats(self) -> tuple[np.ndarray, float]:
        return self._mean, self._std

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        idn = i % self.n_identities

        expr = rng.normal(0.0, 1.0, size=(self.n_vertices, 3)).astype(np.float32) * self.expr_scale
        verts_raw = self.mean_face + self.id_offset[idn] + expr
        verts_std = (verts_raw - self._mean) / self._std

        v = rng.normal(size=3)
        v[2] = abs(v[2]) + 1e-3  # front hemisphere
        view = (v / np.linalg.norm(v)).astype(np.float32)

        avg_tex = self.base_tex[idn]
        shade = 0.5 + 0.5 * float(view[2])  # brighter when frontal
        tint = (0.85 + 0.15 * view.reshape(-1, 1, 1)[: self.tex_channels]).astype(np.float32)
        noise = rng.normal(0.0, 0.02, size=avg_tex.shape).astype(np.float32)
        tex = np.clip(avg_tex * shade * tint + noise, 0.0, 1.0).astype(np.float32)

        return {
            "verts": torch.from_numpy(verts_std),
            "avg_tex": torch.from_numpy(avg_tex.copy()),
            "tex": torch.from_numpy(tex),
            "view": torch.from_numpy(view[: self.view_dim].copy()),
            "tex_mask": torch.ones(1, self.tex_size, self.tex_size),
        }
