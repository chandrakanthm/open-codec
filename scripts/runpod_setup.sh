#!/usr/bin/env bash
# One-shot setup on a fresh RunPod pod (use a "PyTorch 2.x / CUDA 12.x" template).
#
#   bash scripts/runpod_setup.sh
#
# Installs the package, verifies the GPU, and prints next steps. It does NOT
# download data (that can be hundreds of GB) -- see scripts/download_multiface.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> Installing lightweight deps (torch is expected from the base image) ..."
pip install -q numpy pyyaml pillow tqdm tensorboard "huggingface_hub>=0.23" requests

echo ">> Installing codec-avatars (no-deps so torch is left untouched) ..."
pip install -q -e . --no-deps

echo ">> Environment:"
python - <<'PY'
import torch
print("  torch       :", torch.__version__)
print("  cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device      :", torch.cuda.get_device_name(0))
    print("  vram (GB)   :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo ">> Sanity check on synthetic data (a few CPU/GPU steps) ..."
python -m codec_avatars.train --config configs/smoke_test.yaml \
  --set train.device=cuda --set optim.max_steps=10 --set train.out_dir=runs/_setup_check || \
python -m codec_avatars.train --config configs/smoke_test.yaml --set optim.max_steps=10 --set train.out_dir=runs/_setup_check

cat <<'EOF'

================================================================
Setup complete. Next steps (or just use the Makefile: make data/doctor/train/demo):

  1) Download data (~16 GB example set by default):
       bash scripts/download_multiface.sh data/multiface

  2) PREFLIGHT before the long run — validates data + does one real GPU step,
     printing peak VRAM and a fit verdict in seconds:
       python -m codec_avatars.doctor --config configs/multiface_dam.yaml \
         --set data.root=data/multiface

  3) Train (auto-resumes from runs/.../latest.pt if a spot pod was preempted):
       python -m codec_avatars.train --config configs/multiface_dam.yaml \
         --set data.root=data/multiface
       # tensorboard --logdir runs/multiface_dam --host 0.0.0.0 --port 6006

  4) Codec demo (writes code.npy, avatar.obj, a yaw contact_sheet.png + compare.png):
       python -m codec_avatars.infer --ckpt runs/multiface_dam/latest.pt \
         --device cuda --index 0 --out outputs/demo --multiview 8
================================================================
EOF
