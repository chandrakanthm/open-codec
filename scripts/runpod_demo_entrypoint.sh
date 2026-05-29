#!/usr/bin/env bash
# Container boot command for the RunPod Codec Avatars demo template.
#
# Flow:  (ssh setup) -> GPU info -> doctor -> demo -> idle
# Controlled by env vars (set on the template / pod):
#   DEMO_MODE = smoke (default) | multiface | none
#   IDLE      = 1 (default; `sleep infinity` so the pod stays up) | 0 (exit after)
#   WORK      = /workspace (persistent volume for data + checkpoints + outputs)
#
# NOTE: no `set -e` — if a demo step fails we still want the pod to idle so you
# can SSH in and debug instead of the container dying on boot.
set -uo pipefail

CODE_DIR="${CODE_DIR:-/opt/codec-avatars}"
WORK="${WORK:-/workspace}"
DEMO_MODE="${DEMO_MODE:-smoke}"
IDLE="${IDLE:-1}"
PYBIN="${PYTHON:-python}"
cd "$CODE_DIR" || { echo "!! code dir $CODE_DIR missing"; exit 1; }

run() { echo "+ $*"; "$@" || echo "!! step failed (continuing so the pod stays up): $*"; }

# --- SSH: RunPod injects the user's key as $PUBLIC_KEY ------------------------
if [ -n "${PUBLIC_KEY:-}" ] && command -v sshd >/dev/null 2>&1; then
  mkdir -p ~/.ssh && printf '%s\n' "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
  chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1 || true
  mkdir -p /run/sshd
  /usr/sbin/sshd && echo ">> sshd started on :22" || echo ">> sshd unavailable (web terminal still works)"
fi

DEV="$("$PYBIN" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)"

echo "================================================================"
echo "Codec Avatars demo container"
"$PYBIN" -c 'import torch; print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")' 2>/dev/null || true
echo "DEMO_MODE=$DEMO_MODE  device=$DEV  code=$CODE_DIR  workspace=$WORK"
echo "================================================================"

mkdir -p "$WORK/outputs" "$WORK/runs" "$WORK/data"

case "$DEMO_MODE" in
  smoke)
    echo ">> Smoke demo: validate the pipeline on the GPU with synthetic data (no download)."
    run "$PYBIN" -m codec_avatars.doctor --config configs/smoke_test.yaml --set train.device="$DEV"
    run "$PYBIN" -m codec_avatars.train  --config configs/smoke_test.yaml \
        --set train.device="$DEV" --set train.out_dir="$WORK/runs/smoke"
    run "$PYBIN" -m codec_avatars.infer  --ckpt "$WORK/runs/smoke/final.pt" \
        --device "$DEV" --out "$WORK/outputs/demo" --multiview 8
    echo ">> Demo done. Artifacts in $WORK/outputs/demo/ (contact_sheet.png, compare.png, avatar.obj, code.npy)."
    ;;
  multiface)
    echo ">> Real run: download the Multiface example set, preflight, then train."
    run bash scripts/download_multiface.sh "$WORK/data/multiface"
    run "$PYBIN" -m codec_avatars.doctor --config configs/multiface_dam.yaml \
        --set data.root="$WORK/data/multiface" --set train.device="$DEV"
    run "$PYBIN" -m codec_avatars.train  --config configs/multiface_dam.yaml \
        --set data.root="$WORK/data/multiface" --set train.out_dir="$WORK/runs/multiface_dam" \
        --set train.device="$DEV"
    echo ">> Training exited. Demo a checkpoint:"
    echo "   $PYBIN -m codec_avatars.infer --ckpt $WORK/runs/multiface_dam/latest.pt --device $DEV --out $WORK/outputs/demo --multiview 8"
    ;;
  none|*)
    echo ">> DEMO_MODE=$DEMO_MODE: skipping the auto-demo; pod is ready for manual use."
    ;;
esac

if [ "$IDLE" = "1" ]; then
  echo ">> Container idle. SSH in or open a web terminal. TensorBoard: tensorboard --logdir $WORK/runs --host 0.0.0.0 --port 6006"
  sleep infinity
fi
