# Codec Avatars demo/training image for RunPod (or any CUDA host).
# Built on the official PyTorch CUDA image so torch + CUDA are preinstalled.
#
# Build & push for RunPod (GPUs are linux/amd64):
#   IMAGE=docker.io/youruser/codec-avatars:latest bash scripts/runpod_build_push.sh
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEMO_MODE=smoke

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Code lives in /opt so a mounted /workspace volume (data + checkpoints) never
# shadows it. The demo writes all outputs under /workspace.
WORKDIR /opt/codec-avatars

# Install lighter deps first (torch is already in the base image).
COPY requirements.txt ./
RUN pip install numpy pyyaml pillow tqdm tensorboard "huggingface_hub>=0.23" requests

# Install the package itself without re-resolving torch.
COPY . .
RUN pip install -e . --no-deps && chmod +x scripts/*.sh

# On boot: GPU check -> doctor -> demo -> idle. See the entrypoint for DEMO_MODE.
CMD ["bash", "-lc", "/opt/codec-avatars/scripts/runpod_demo_entrypoint.sh"]
