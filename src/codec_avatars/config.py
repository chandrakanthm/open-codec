"""Configuration: nested dataclasses with YAML loading and CLI dotted overrides.

The model owns the canonical tensor dimensions (``n_vertices``, ``tex_size``,
``tex_channels``, ``view_dim``); datasets receive those at build time so the two
can never silently disagree.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelConfig:
    latent_dim: int = 256
    # Canonical tensor dimensions (source of truth for the whole pipeline).
    n_vertices: int = 7306  # Multiface MUGSY topology; override per dataset.
    tex_size: int = 1024
    tex_channels: int = 3
    view_dim: int = 3  # unit view direction in head coordinates
    # Texture encoder downsamples tex_size -> 4x4 by these channel widths.
    enc_tex_channels: tuple[int, ...] = (32, 64, 128, 256, 256, 256, 256, 256)
    # Texture decoder upsamples a 4x4 seed back up to tex_size.
    dec_tex_channels: tuple[int, ...] = (256, 256, 256, 256, 256, 128, 64, 32)
    # Mesh (geometry) MLP widths for encoder and decoder.
    mesh_hidden: tuple[int, ...] = (512, 256)
    # Clamp range for predicted log-variance (stabilises the VAE).
    min_logvar: float = -6.0
    max_logvar: float = 4.0


@dataclass
class DataConfig:
    name: str = "synthetic"  # "synthetic" | "multiface"
    root: str = "data/multiface"
    subjects: list[str] = field(default_factory=list)  # empty = autodiscover
    cameras: list[str] | None = None  # None = all cameras found
    expressions: list[str] | None = None  # None = all expressions/segments
    max_frames_per_seq: int | None = None  # subsample long sequences
    val_fraction: float = 0.02
    num_workers: int = 8
    pin_memory: bool = True
    # Synthetic-only knobs (ignored by the Multiface loader).
    synthetic_size: int = 64
    synthetic_identities: int = 4


@dataclass
class LossConfig:
    w_geom: float = 1.0
    w_tex: float = 1.0
    w_kl: float = 1.0e-3  # target KL weight (beta) after warmup
    kl_warmup_steps: int = 5000  # linear anneal 0 -> w_kl over these steps
    # Geometry is supervised in millimetres; texture in [0, 1]. Per-channel
    # scaling keeps the two terms comparable early in training.
    geom_scale: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    max_steps: int = 200_000
    grad_clip: float = 1.0
    batch_size: int = 4
    lr_decay_steps: int = 0  # 0 = constant LR; else cosine to lr_min
    lr_min: float = 1.0e-5


@dataclass
class TrainConfig:
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    amp: bool = True  # mixed precision (CUDA only; ignored on cpu/mps)
    amp_dtype: str = "bf16"  # "bf16" (stable, default) | "fp16" (loss-scaled)
    seed: int = 0
    out_dir: str = "runs/dam"
    log_every: int = 50
    ckpt_every: int = 5000
    val_every: int = 2000
    # Checkpoint to resume from. "auto"/"latest" -> {out_dir}/latest.pt if present
    # (spot-pod friendly: re-running the same command picks up where it left off).
    resume: str | None = None
    tensorboard: bool = True
    max_val_batches: int = 20


@dataclass
class Config:
    name: str = "dam"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "loss": LossConfig,
    "optim": OptimConfig,
    "train": TrainConfig,
}


def _coerce(value: Any, ref: Any) -> Any:
    """Coerce a YAML/CLI value to roughly match the dataclass default's type."""
    if ref is None:
        return value
    if isinstance(ref, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(ref, tuple):
        return tuple(value)
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    return value


def _build_section(cls: type, overrides: dict[str, Any]) -> Any:
    defaults = cls()  # type: ignore[call-arg]
    valid = {f.name for f in dataclasses.fields(cls)}
    for key, val in overrides.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for section '{cls.__name__}'")
        setattr(defaults, key, _coerce(val, getattr(defaults, key)))
    return defaults


def config_from_dict(d: dict[str, Any]) -> Config:
    """Build a Config from a (possibly partial) nested dict."""
    d = dict(d or {})
    sections = {}
    for name, cls in _SECTIONS.items():
        sections[name] = _build_section(cls, d.get(name, {}) or {})
    return Config(name=d.get("name", "dam"), **sections)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` strings (from the CLI) onto a Config."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) == 1:  # top-level scalar, e.g. name=foo
            setattr(cfg, parts[0], raw)
            continue
        if len(parts) != 2 or parts[0] not in _SECTIONS:
            raise ValueError(f"Override '{item}' must target a known section, e.g. optim.lr=3e-4")
        section, key = parts
        sec_obj = getattr(cfg, section)
        if not hasattr(sec_obj, key):
            raise KeyError(f"Unknown key '{key}' in section '{section}'")
        # YAML-parse the scalar so 3e-4, true, [a,b] all behave.
        val = yaml.safe_load(raw)
        setattr(sec_obj, key, _coerce(val, getattr(sec_obj, key)))
    return cfg


def load_config(path: str | None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file (optional) and apply CLI overrides."""
    data: dict[str, Any] = {}
    if path:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    cfg = config_from_dict(data)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    """Serialise a Config back to a plain (YAML-friendly) dict."""
    out = dataclasses.asdict(cfg)

    def _clean(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    return _clean(out)
