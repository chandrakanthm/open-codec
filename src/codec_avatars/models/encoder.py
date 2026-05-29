"""Appearance encoder: (view-averaged texture, mesh) -> latent distribution.

Geometry is view-independent, so the encoder consumes the view-*averaged*
texture together with the tracked mesh and produces the parameters (mu, logvar)
of the latent code. This mirrors the Deep Appearance Model encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from codec_avatars.config import ModelConfig
from codec_avatars.models.layers import ConvDown, mlp, n_scales, resize_schedule


class AppearanceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        n_down = n_scales(cfg.tex_size)  # downsample tex_size -> 4x4
        ch = resize_schedule(cfg.enc_tex_channels, n_down, tail=False)

        blocks: list[nn.Module] = []
        in_ch = cfg.tex_channels
        for out_ch in ch:
            blocks.append(ConvDown(in_ch, out_ch))
            in_ch = out_ch
        self.tex_conv = nn.Sequential(*blocks)
        self.tex_feat_dim = in_ch * 4 * 4  # spatial is 4x4 after the stack
        self.tex_proj = nn.Linear(self.tex_feat_dim, cfg.latent_dim)

        self.mesh_mlp = mlp([cfg.n_vertices * 3, *cfg.mesh_hidden, cfg.latent_dim], last_act=True)

        fused = 2 * cfg.latent_dim
        self.to_mu = nn.Linear(fused, cfg.latent_dim)
        self.to_logvar = nn.Linear(fused, cfg.latent_dim)

    def forward(self, avg_tex: torch.Tensor, verts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """avg_tex: (B, C, H, W) in [0,1]; verts: (B, V, 3) standardized."""
        b = avg_tex.shape[0]
        t = self.tex_conv(avg_tex).reshape(b, -1)
        t = self.tex_proj(t)
        m = self.mesh_mlp(verts.reshape(b, -1))
        h = torch.cat([t, m], dim=1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(self.cfg.min_logvar, self.cfg.max_logvar)
        return mu, logvar
