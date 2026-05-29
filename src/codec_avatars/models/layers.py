"""Reusable network blocks for the Deep Appearance Model."""

from __future__ import annotations

import math

import torch
from torch import nn


def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def n_scales(tex_size: int, seed: int = 4) -> int:
    """Number of x2 (down/up) steps between ``tex_size`` and ``seed`` (4x4)."""
    if not is_pow2(tex_size) or not is_pow2(seed) or tex_size < seed:
        raise ValueError(f"tex_size ({tex_size}) and seed ({seed}) must be powers of two with tex_size >= seed")
    return int(round(math.log2(tex_size))) - int(round(math.log2(seed)))


def resize_schedule(channels: tuple[int, ...] | list[int], n: int, *, tail: bool) -> list[int]:
    """Fit a channel schedule to exactly ``n`` stages.

    ``tail=True`` keeps the fine-detail end (used by the decoder, which should
    finish on the smallest channel count); ``tail=False`` keeps the head (used
    by the encoder).
    """
    ch = list(channels)
    if n <= 0:
        return []
    if len(ch) == n:
        return ch
    if len(ch) > n:
        return ch[-n:] if tail else ch[:n]
    pad = n - len(ch)
    return ([ch[0]] * pad + ch) if tail else (ch + [ch[-1]] * pad)


class ConvDown(nn.Module):
    """3x3 conv, then strided 2x downsample. LeakyReLU throughout."""

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvUp(nn.Module):
    """Bilinear 2x upsample, then 3x3 convs. Avoids transpose-conv checkerboarding."""

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.up(x))


def mlp(sizes: list[int], slope: float = 0.2, last_act: bool = False) -> nn.Sequential:
    """Build an MLP over ``sizes`` (in, hidden..., out) with LeakyReLU between."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or last_act:
            layers.append(nn.LeakyReLU(slope, inplace=True))
    return nn.Sequential(*layers)
