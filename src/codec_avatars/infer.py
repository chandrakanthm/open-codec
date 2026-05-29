"""Encode a face to a code, report the codec's footprint, and decode an avatar.

    python -m codec_avatars.infer --ckpt runs/dam/final.pt --index 0 --out outputs/demo

This is the end-to-end "codec" demonstration: a frame is compressed to a few
hundred floats (printed as bytes/frame and an equivalent bitrate), then the
decoder reconstructs the 3D mesh (.obj) and view-dependent texture (.png). With
``--multiview N`` it re-decodes the same code across a yaw sweep to show that the
appearance is genuinely view-dependent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from codec_avatars.config import config_from_dict
from codec_avatars.data import build_base_dataset
from codec_avatars.data.transforms import save_texture
from codec_avatars.models import build_model
from codec_avatars.utils.checkpoint import load_checkpoint
from codec_avatars.utils.device import resolve_device
from codec_avatars.utils.mesh import save_obj


def _add_batch(sample: dict, device) -> dict:
    return {k: v.unsqueeze(0).to(device) for k, v in sample.items()}


def _yaw_views(n: int, elevation: float = 0.0) -> torch.Tensor:
    """n unit view directions sweeping azimuth around the head."""
    out = []
    for i in range(n):
        a = (i / max(n, 1)) * 2 * math.pi
        out.append([math.sin(a) * math.cos(elevation), math.sin(elevation), math.cos(a) * math.cos(elevation)])
    return torch.tensor(out, dtype=torch.float32)


def _montage(tiles: list[np.ndarray], cols: int, pad: int = 4, padval: float = 1.0) -> np.ndarray:
    """Tile a list of (C,H,W) [0,1] textures into one (C,H',W') grid with gaps."""
    n = len(tiles)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    c, h, w = tiles[0].shape
    grid = np.full((c, rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad), padval, np.float32)
    for i, t in enumerate(tiles):
        r, q = divmod(i, cols)
        y, x = r * (h + pad), q * (w + pad)
        grid[:, y : y + h, x : x + w] = t
    return grid


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = config_from_dict(ckpt.get("config", {}))

    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    base, _, n_vertices = build_base_dataset(cfg)
    if args.index >= len(base):
        raise IndexError(f"--index {args.index} out of range (dataset has {len(base)} samples)")
    sample = base[args.index]
    batch = _add_batch(sample, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- ENCODE: face -> code ---------------------------------------------------
    with torch.no_grad():
        code = model.encode_code(batch["avg_tex"], batch["verts"])  # (1, D)
    code_np = code.squeeze(0).cpu().numpy().astype(np.float32)
    np.save(out_dir / "code.npy", code_np)

    d = code_np.shape[0]
    fp32 = d * 4
    fp16 = d * 2
    kbps32 = fp32 * 8 * args.fps / 1000.0
    kbps16 = fp16 * 8 * args.fps / 1000.0

    # --- DECODE: code (+view) -> avatar ----------------------------------------
    with torch.no_grad():
        verts_std, tex = model.decode(code, batch["view"])
        verts_mm = model.unstandardize_geometry(verts_std).squeeze(0).cpu().numpy()
        tgt_mm = model.unstandardize_geometry(batch["verts"]).squeeze(0).cpu().numpy()

    geom_rmse = float(np.sqrt(((verts_mm - tgt_mm) ** 2).mean()))
    tex_l1 = float((tex - batch["tex"]).abs().mean())

    # Topology (faces/UVs) for a textured OBJ, if the dataset provides it.
    faces = uvs = None
    if hasattr(base, "topology"):
        topo = base.topology(0)
        if topo is not None:
            faces, uvs = topo

    recon_np = tex.squeeze(0).cpu().numpy()
    target_np = batch["tex"].squeeze(0).cpu().numpy()
    save_texture(out_dir / "texture.png", recon_np)
    save_texture(out_dir / "texture_target.png", target_np)
    # Side-by-side target vs. reconstruction for a quick eyeball check.
    save_texture(out_dir / "compare.png", _montage([target_np, recon_np], cols=2))
    save_obj(
        out_dir / "avatar.obj", verts_mm,
        faces=faces, uvs=uvs,
        face_uvs=None,
        texture_name="texture.png" if (faces is not None and uvs is not None) else None,
    )

    # --- Optional view sweep to demonstrate view-dependence ---------------------
    if args.multiview > 1:
        views = _yaw_views(args.multiview).to(device)
        rep = code.repeat(args.multiview, 1)
        with torch.no_grad():
            _, tex_mv = model.decode(rep, views)
        tiles = [tex_mv[i].cpu().numpy() for i in range(args.multiview)]
        for i, t in enumerate(tiles):
            save_texture(out_dir / f"texture_view{i:02d}.png", t)
        # One glanceable contact sheet of the whole yaw sweep.
        save_texture(out_dir / "contact_sheet.png", _montage(tiles, cols=min(args.multiview, 8)))

    print("=" * 64)
    print("CODEC AVATAR — encode/decode report")
    print("=" * 64)
    print(f"dataset           : {cfg.data.name}  (sample #{args.index})")
    print(f"mesh vertices     : {n_vertices}")
    print(f"latent code dim   : {d}")
    print(f"code size / frame : {fp32} B (float32)   |   {fp16} B (float16)")
    print(f"stream @ {args.fps:>3} fps  : {kbps32:.1f} kbit/s (fp32) | {kbps16:.1f} kbit/s (fp16)")
    print(f"geometry RMSE     : {geom_rmse:.3f} mm   (decoded vs. tracked)")
    print(f"texture L1        : {tex_l1:.4f}        (decoded vs. captured view)")
    print(f"params            : {model.num_parameters():,}")
    sheet = "  contact_sheet.png" if args.multiview > 1 else ""
    print(f"outputs           : {out_dir}/  (code.npy, avatar.obj, texture*.png, compare.png{sheet})")
    print("=" * 64)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Encode->code->decode a Codec Avatar")
    p.add_argument("--ckpt", required=True, help="Path to a trained checkpoint (.pt)")
    p.add_argument("--index", type=int, default=0, help="Sample index to encode")
    p.add_argument("--out", default="outputs/demo", help="Output directory")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--fps", type=int, default=30, help="Frame rate for the bitrate estimate")
    p.add_argument("--multiview", type=int, default=0, help="Decode N yaw views to show view-dependence")
    return p


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
