"""Shared data transforms: texture image I/O and camera -> view-direction math."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_texture(path: str | Path, size: int, channels: int = 3) -> np.ndarray:
    """Load an image as (C, H, W) float32 in [0,1], resized to ``size`` x ``size``."""
    mode = "RGB" if channels == 3 else "L"
    img = Image.open(path).convert(mode)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[..., None]
    return np.transpose(arr, (2, 0, 1)).copy()


def save_texture(path: str | Path, tex: np.ndarray) -> None:
    """Save a (C,H,W) or (H,W,C) float[0,1] texture as PNG."""
    arr = np.asarray(tex, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(str(path))


def camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World-space camera center from extrinsics (world->camera): c = -R^T t."""
    return -R.T @ T.reshape(3)


def view_direction(R: np.ndarray, T: np.ndarray, head_center: np.ndarray) -> np.ndarray:
    """Unit direction from the head to the camera, in world coordinates.

    The avatar is roughly head-centered, so the world-space direction is a stable
    conditioning signal for view-dependent appearance. Returns (3,) float32.
    """
    c = camera_center(R, T)
    d = c - np.asarray(head_center, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (d / n).astype(np.float32)
