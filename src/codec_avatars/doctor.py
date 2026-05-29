"""Preflight check — run this *before* a long training job.

    python -m codec_avatars.doctor --config configs/multiface_dam.yaml \
        --set data.root=data/multiface

RunPod time is expensive, and the failures that hurt most are the ones that only
surface after an hours-long download: a mis-shaped dataset, an OOM at the chosen
batch size, a CUDA/driver mismatch. ``doctor`` exercises the whole pipeline once
and reports a verdict in seconds:

  1. environment   — torch / CUDA build, GPU name + total VRAM
  2. dataset       — builds the *real* dataset, reports samples / vertices / stats
  3. model         — parameter count + code size per frame
  4. one step      — a timed forward+backward at the configured batch size,
                     reporting step time, est. it/s, and peak VRAM vs. capacity

Exit code is 0 on success, 1 on the first hard failure (with a fix hint).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

from codec_avatars.config import Config, load_config
from codec_avatars.data import build_datasets
from codec_avatars.losses import compute_losses
from codec_avatars.models import build_model
from codec_avatars.utils import resolve_amp, resolve_device, set_seed


def _gb(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


def _fail(section: str, err: Exception, hint: str = "") -> int:
    print(f"  [FAIL] {section}: {err}")
    if hint:
        print(f"         hint: {hint}")
    print("\n" + "-" * 64)
    print("PREFLIGHT FAILED — fix the above before launching training.")
    print("-" * 64)
    traceback.print_exc()
    return 1


def run(cfg: Config) -> int:
    print("=" * 64)
    print("CODEC AVATAR — preflight (doctor)")
    print("=" * 64)

    # 1. Environment ------------------------------------------------------------
    print("[1/4] environment")
    _info(f"torch {torch.__version__}  (CUDA build: {torch.version.cuda or 'cpu-only'})")
    device = resolve_device(cfg.train.device)
    total_vram = 0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        _ok(f"GPU: {props.name}  ({_gb(total_vram):.1f} GB, bf16={'yes' if bf16 else 'no'})")
    elif cfg.train.device.startswith("cuda"):
        _info("requested cuda but it is unavailable — running this check on CPU")
        _info("(that's fine for validating data/model; rerun on the GPU pod for VRAM numbers)")
    else:
        _ok(f"device: {device}")

    use_amp, amp_dtype = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    _info(f"amp: {str(amp_dtype).split('.')[-1] if use_amp else 'off'}")

    set_seed(cfg.train.seed)

    # 2. Dataset ----------------------------------------------------------------
    print("[2/4] dataset")
    try:
        train_ds, val_ds, (mean, std), n_vertices = build_datasets(cfg)
    except Exception as e:  # noqa: BLE001 - report any dataset error clearly
        hint = (
            "check data.root points at the downloaded dataset; expected per-subject "
            "tracked_mesh/, unwrapped_uv_1024/, and a KRT file"
            if cfg.data.name == "multiface"
            else "synthetic dataset failed to build — check model dims"
        )
        return _fail("dataset", e, hint)
    cfg.model.n_vertices = n_vertices
    _ok(f"{cfg.data.name}: train={len(train_ds)} val={len(val_ds)} samples")
    _info(f"vertices={n_vertices}  tex_size={cfg.model.tex_size}  view_dim={cfg.model.view_dim}")
    _info(f"geom stats: mean|mm|~{abs(float(mean.mean())):.2f}  std={float(std):.2f}")
    if len(train_ds) == 0:
        return _fail("dataset", RuntimeError("no training samples"), "loosen camera/expression filters")

    # 3. Model ------------------------------------------------------------------
    print("[3/4] model")
    try:
        model = build_model(cfg.model).to(device)
        model.set_geometry_stats(mean, std)
    except Exception as e:  # noqa: BLE001
        return _fail("model", e, "check model.* dims (tex_size must be a power of two)")
    _ok(f"params={model.num_parameters():,}  latent={model.code_dim}  code={model.code_bytes()} B/frame")

    # 4. One real step ----------------------------------------------------------
    print("[4/4] one training step")
    bs = cfg.optim.batch_size
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    try:
        batch = next(iter(loader))
    except StopIteration:
        return _fail("step", RuntimeError(f"not enough samples for batch_size={bs}"),
                     f"lower optim.batch_size (have {len(train_ds)} samples)")
    batch = {k: v.to(device) for k, v in batch.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    try:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch, sample=True)
            loss, metrics = compute_losses(out, batch, cfg.loss, step=0)
        loss.backward()
        opt.step()
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        return _fail("step (OOM)", e,
                     f"batch_size={bs} at tex_size={cfg.model.tex_size} is too big — "
                     "halve optim.batch_size, then lower model.tex_size (512) if still tight")
    except Exception as e:  # noqa: BLE001
        return _fail("step", e, "see traceback above")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0

    _ok(f"forward+backward OK  loss={float(loss.detach()):.4f}  (geom={metrics['loss/geom']:.4f} tex={metrics['loss/tex']:.4f})")
    _info(f"step time: {dt * 1000:.0f} ms  (~{1.0 / max(dt, 1e-6):.1f} it/s, single process)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        frac = peak / max(total_vram, 1)
        _info(f"peak VRAM: {_gb(peak):.2f} / {_gb(total_vram):.1f} GB  ({frac * 100:.0f}%)")
        if frac < 0.7:
            _ok(f"fits comfortably — you have headroom to raise optim.batch_size")
        elif frac < 0.9:
            _ok("fits — close to capacity; raise batch size only with care")
        else:
            _info("WARNING: >90% VRAM on one step; training may OOM. Lower batch_size or tex_size.")

    print("\n" + "=" * 64)
    print("PREFLIGHT PASSED — safe to launch training.")
    print(f"  python -m codec_avatars.train --config <your.yaml> --set data.root={cfg.data.root}")
    print("=" * 64)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preflight check for Codec Avatar training")
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="section.key=value", help="Override a config value (repeatable)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
