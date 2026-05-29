#!/usr/bin/env python3
"""Download a face-capture dataset from the Hugging Face Hub.

    python scripts/download_hf.py --repo-id <org/dataset> --dest data/multiface

Use this for HF-hosted mirrors of multi-view capture data (Multiface, Ava-256),
or any HF dataset that follows the same per-subject layout the Multiface loader
expects. For raw image datasets (FFHQ/CelebA-HQ) you'd need a different decoder
head -- the Deep Appearance Model here is built for multi-view + tracked-mesh
capture, which is what gives a *3D* codec avatar rather than a 2D portrait.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Snapshot-download a dataset from the HF Hub")
    p.add_argument("--repo-id", required=True, help="e.g. some-org/multiface-mini")
    p.add_argument("--dest", default="data/multiface", help="local destination directory")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument("--allow-patterns", nargs="*", default=None, help="glob(s) to include")
    p.add_argument("--token", default=None, help="HF token for gated datasets")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub") from e

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.dest,
        allow_patterns=args.allow_patterns,
        token=args.token,
    )
    print(f"Downloaded {args.repo_id} -> {path}")
    print(f"Train with: python -m codec_avatars.train --config configs/multiface_dam.yaml --set data.root={args.dest}")


if __name__ == "__main__":
    main()
