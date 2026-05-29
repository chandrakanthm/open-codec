"""Appearance decoder: latent code (+ view) -> mesh geometry and texture.

Geometry depends only on the code (view-independent). The texture additionally
takes the unit view direction so the model can reproduce view-dependent effects
(specular highlights, subsurface shifts) -- the key idea behind the Deep
Appearance Model.
"""

from __future__ import annotations

import torch
from torch import nn

from codec_avatars.config import ModelConfig
from codec_avatars.models.layers import ConvUp, mlp, n_scales, resize_schedule


class MeshDecoder(nn.Module):
    """z -> vertex offsets (standardized geometry space)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden = list(reversed(cfg.mesh_hidden))
        self.net = mlp([cfg.latent_dim, *hidden, cfg.n_vertices * 3], last_act=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        return self.net(z).reshape(b, self.cfg.n_vertices, 3)


class TextureDecoder(nn.Module):
    """(z, view) -> texture in [0,1]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        n_up = n_scales(cfg.tex_size)  # 4x4 -> tex_size
        ch = resize_schedule(cfg.dec_tex_channels, n_up, tail=True)
        self.seed_ch = ch[0]
        self.seed = nn.Linear(cfg.latent_dim + cfg.view_dim, self.seed_ch * 4 * 4)

        blocks: list[nn.Module] = []
        in_ch = self.seed_ch
        for out_ch in ch:
            blocks.append(ConvUp(in_ch, out_ch))
            in_ch = out_ch
        self.up = nn.Sequential(*blocks)
        self.to_rgb = nn.Conv2d(in_ch, cfg.tex_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        h = torch.cat([z, view], dim=1)
        h = self.seed(h).reshape(b, self.seed_ch, 4, 4)
        h = self.up(h)
        return torch.sigmoid(self.to_rgb(h))


class AppearanceDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mesh = MeshDecoder(cfg)
        self.texture = TextureDecoder(cfg)

    def forward(self, z: torch.Tensor, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mesh(z), self.texture(z, view)
