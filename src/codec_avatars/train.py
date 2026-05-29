"""Train the Codec Avatar (Deep Appearance Model).

    python -m codec_avatars.train --config configs/smoke_test.yaml
    python -m codec_avatars.train --config configs/multiface_dam.yaml \
        --set data.root=/workspace/data/multiface optim.batch_size=8

Geometry is supervised in standardized space; texture in [0,1]. The latent code
is annealed toward the prior over ``loss.kl_warmup_steps``.
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from codec_avatars.config import Config, load_config, to_dict
from codec_avatars.data import build_datasets
from codec_avatars.losses import compute_losses
from codec_avatars.models import build_model
from codec_avatars.utils import Logger, resolve_amp, resolve_device, save_checkpoint, set_seed
from codec_avatars.utils.checkpoint import load_checkpoint


def _resolve_resume(resume: str | None, out_dir: str) -> str | None:
    """Resolve the resume target. 'auto'/'latest' -> {out_dir}/latest.pt if present."""
    if resume in (None, "", "none", "None"):
        return None
    if resume in ("auto", "latest"):
        cand = Path(out_dir) / "latest.pt"
        return str(cand) if cand.exists() else None
    return resume


def _lr_at(step: int, cfg: Config) -> float:
    o = cfg.optim
    if o.lr_decay_steps <= 0:
        return o.lr
    t = min(step, o.lr_decay_steps) / float(o.lr_decay_steps)
    return o.lr_min + 0.5 * (o.lr - o.lr_min) * (1.0 + math.cos(math.pi * t))


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device, step: int, max_batches: int) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        out = model(batch, sample=False)
        _, metrics = compute_losses(out, batch, cfg.loss, step)
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"val/{k}": v / max(n, 1) for k, v in agg.items()}


def train(cfg: Config) -> str:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    train_ds, val_ds, (mean, std), n_vertices = build_datasets(cfg)
    cfg.model.n_vertices = n_vertices  # data dictates topology size

    model = build_model(cfg.model).to(device)
    model.set_geometry_stats(mean, std)

    use_amp, amp_dtype = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    workers = cfg.data.num_workers if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
        num_workers=workers, pin_memory=(cfg.data.pin_memory and device.type == "cuda"),
        drop_last=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False,
        num_workers=workers, pin_memory=False, persistent_workers=workers > 0,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2), weight_decay=cfg.optim.weight_decay,
    )
    # Loss scaling is only needed for float16; bf16 has fp32 dynamic range.
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    start_step = 0
    resume_path = _resolve_resume(cfg.train.resume, cfg.train.out_dir)
    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {resume_path} at step {start_step}")

    logger = Logger(cfg.train.out_dir, use_tensorboard=cfg.train.tensorboard)
    amp_mode = f"{str(amp_dtype).split('.')[-1]}" if use_amp else "off"
    print(
        f"device={device} amp={amp_mode} params={model.num_parameters():,} "
        f"code={model.code_dim} dims ({model.code_bytes()} B/frame) "
        f"train={len(train_ds)} val={len(val_ds)} V={n_vertices}"
    )

    model.train()
    data_iter = cycle(train_loader)
    t0 = time.time()
    step = start_step
    for step in range(start_step, cfg.optim.max_steps):
        batch = _to_device(next(data_iter), device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch, sample=True)
            loss, metrics = compute_losses(out, batch, cfg.loss, step)

        scaler.scale(loss).backward()
        if cfg.optim.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.train.log_every == 0:
            metrics["lr"] = opt.param_groups[0]["lr"]
            rate = (step - start_step + 1) / (time.time() - t0 + 1e-9)
            logger.log(metrics, step)
            logger.console(step, metrics, extra=f"({rate:.1f} it/s)")
        if cfg.train.val_every and step > 0 and step % cfg.train.val_every == 0:
            val = evaluate(model, val_loader, cfg, device, step, cfg.train.max_val_batches)
            logger.log(val, step)
            logger.console(step, val, extra="[val]")
        if cfg.train.ckpt_every and step > 0 and step % cfg.train.ckpt_every == 0:
            cfg_dict = to_dict(cfg)
            save_checkpoint(f"{cfg.train.out_dir}/step_{step}.pt", model, opt, step, cfg_dict)
            # latest.pt is the spot-pod resume anchor (resume: auto).
            save_checkpoint(f"{cfg.train.out_dir}/latest.pt", model, opt, step, cfg_dict)

    final = f"{cfg.train.out_dir}/final.pt"
    cfg_dict = to_dict(cfg)
    save_checkpoint(final, model, opt, step + 1, cfg_dict)
    save_checkpoint(f"{cfg.train.out_dir}/latest.pt", model, opt, step + 1, cfg_dict)
    logger.close()
    print(f"Saved final checkpoint to {final}")
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Codec Avatar (Deep Appearance Model)")
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="section.key=value", help="Override a config value (repeatable)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
