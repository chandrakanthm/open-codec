"""CodecAvatar: the conditional VAE that *is* the codec.

Encoder  : (view-averaged texture, mesh)            -> latent code z   [the "codec"]
Decoder  : (z, view direction)                       -> mesh + texture  [the avatar]

The latent ``z`` is the compact representation transmitted in a telepresence
setting: a few hundred floats per frame instead of multi-view video. Geometry
standardisation statistics are stored as buffers so a checkpoint is fully
self-contained for inference/export.
"""

from __future__ import annotations

import torch
from torch import nn

from codec_avatars.config import ModelConfig
from codec_avatars.models.decoder import AppearanceDecoder
from codec_avatars.models.encoder import AppearanceEncoder


class CodecAvatar(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AppearanceEncoder(cfg)
        self.decoder = AppearanceDecoder(cfg)

        # Geometry standardisation (set from data via set_geometry_stats).
        self.register_buffer("vert_mean", torch.zeros(cfg.n_vertices, 3))
        self.register_buffer("vert_std", torch.ones(()))

    # --- geometry standardisation -------------------------------------------------
    def set_geometry_stats(self, mean: torch.Tensor, std: torch.Tensor | float) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(self.cfg.n_vertices, 3)
        std_t = torch.as_tensor(float(std), dtype=torch.float32)
        self.vert_mean.copy_(mean)
        self.vert_std.copy_(std_t)

    def standardize_geometry(self, verts_mm: torch.Tensor) -> torch.Tensor:
        return (verts_mm - self.vert_mean) / self.vert_std

    def unstandardize_geometry(self, verts_std: torch.Tensor) -> torch.Tensor:
        return verts_std * self.vert_std + self.vert_mean

    # --- VAE core -----------------------------------------------------------------
    def encode(self, avg_tex: torch.Tensor, verts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(avg_tex, verts)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z, view)

    def forward(self, batch: dict[str, torch.Tensor], sample: bool = True) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(batch["avg_tex"], batch["verts"])
        z = self.reparameterize(mu, logvar) if (sample and self.training) else mu
        verts_hat, tex_hat = self.decode(z, batch["view"])
        return {"verts_hat": verts_hat, "tex_hat": tex_hat, "mu": mu, "logvar": logvar, "z": z}

    # --- codec convenience --------------------------------------------------------
    @torch.no_grad()
    def encode_code(self, avg_tex: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
        """Deterministic code (the posterior mean) used for transmission."""
        mu, _ = self.encode(avg_tex, verts)
        return mu

    @property
    def code_dim(self) -> int:
        return self.cfg.latent_dim

    def code_bytes(self, dtype_bytes: int = 4) -> int:
        """Size of a single transmitted code, in bytes (default float32)."""
        return self.code_dim * dtype_bytes

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig) -> CodecAvatar:
    return CodecAvatar(cfg)
