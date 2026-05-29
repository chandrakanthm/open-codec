#!/usr/bin/env bash
# Build the Codec Avatars demo image and push it to your container registry so
# RunPod can pull it. RunPod GPUs are linux/amd64, so this cross-builds amd64
# (via buildx) even on an Apple Silicon Mac.
#
#   docker login                                  # to your registry first
#   IMAGE=docker.io/youruser/codec-avatars:latest bash scripts/runpod_build_push.sh
#
# The repo may be PRIVATE: store the pull creds in RunPod once with
#   runpodctl registry create --name myreg --username <user> --password <token>
# and pass the printed id to pod create via --registry-auth-id (see runpod_template.sh).
#
# Then create the template:  IMAGE=$IMAGE bash scripts/runpod_template.sh
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=<registry>/<repo>:<tag>, e.g. docker.io/youruser/codec-avatars:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! docker buildx version >/dev/null 2>&1; then
  echo "!! docker buildx is required (Docker Desktop ships it). Install/enable it and retry."
  exit 1
fi

# Reuse a dedicated builder so repeated builds are cached.
docker buildx inspect ca-builder >/dev/null 2>&1 || docker buildx create --name ca-builder >/dev/null
docker buildx use ca-builder

echo ">> Building $IMAGE for linux/amd64 and pushing (this image is multi-GB; first push is slow) ..."
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$IMAGE" --push .

echo
echo ">> Pushed $IMAGE"
echo ">> Next:  IMAGE=$IMAGE bash scripts/runpod_template.sh"
