"""Device + mixed-precision resolution with graceful fallback (cuda -> mps -> cpu)."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    req = (requested or "cpu").lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device(req)
    if req == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if req.startswith("cuda") and not torch.cuda.is_available():
        # Fall back rather than crash so the same config runs locally and on RunPod.
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def resolve_amp(enabled: bool, amp_dtype: str, device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve autocast settings. Returns ``(use_amp, dtype)``.

    AMP is CUDA-only here. We prefer **bfloat16** for VAE stability (no loss
    scaling, no NaNs from the exp() in reparameterisation), and fall back to
    float16 only on GPUs without bf16 support (pre-Ampere). When AMP is off or
    the device isn't CUDA, returns ``(False, float32)``.
    """
    if not (enabled and device.type == "cuda"):
        return False, torch.float32
    want = (amp_dtype or "bf16").lower()
    if want in ("bf16", "bfloat16"):
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return True, torch.bfloat16
        return True, torch.float16  # older GPU: bf16 unsupported, use fp16
    return True, torch.float16
