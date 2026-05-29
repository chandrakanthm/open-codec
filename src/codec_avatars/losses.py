"""Training objective for the Deep Appearance Model.

total = w_geom * geometry_MSE + w_tex * texture_L1 + beta(step) * KL

``beta`` is linearly annealed from 0 to ``w_kl`` over ``kl_warmup_steps`` so the
decoder learns to reconstruct before the latent is squeezed toward the prior
(KL-collapse avoidance).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from codec_avatars.config import LossConfig


def kl_beta(cfg: LossConfig, step: int) -> float:
    if cfg.kl_warmup_steps <= 0:
        return cfg.w_kl
    return cfg.w_kl * min(1.0, step / float(cfg.kl_warmup_steps))


def geometry_loss(verts_hat: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    # Mean squared vertex error in standardized geometry space.
    return F.mse_loss(verts_hat, verts)


def texture_loss(tex_hat: torch.Tensor, tex: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # L1 is more robust than L2 for textures; optional mask ignores empty UV.
    if mask is None:
        return F.l1_loss(tex_hat, tex)
    diff = (tex_hat - tex).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(N(mu, sigma^2) || N(0, I)), averaged over the batch, summed over dims.
    per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample.mean()


def compute_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: LossConfig,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    geom = geometry_loss(out["verts_hat"], batch["verts"])
    tex = texture_loss(out["tex_hat"], batch["tex"], batch.get("tex_mask"))
    kl = kl_divergence(out["mu"], out["logvar"])
    beta = kl_beta(cfg, step)

    total = cfg.w_geom * cfg.geom_scale * geom + cfg.w_tex * tex + beta * kl

    metrics = {
        "loss/total": float(total.detach()),
        "loss/geom": float(geom.detach()),
        "loss/tex": float(tex.detach()),
        "loss/kl": float(kl.detach()),
        "loss/beta": beta,
    }
    return total, metrics
