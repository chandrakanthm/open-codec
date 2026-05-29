#!/usr/bin/env bash
# Download Meta's public Multiface dataset via the official downloader.
#
#   bash scripts/download_multiface.sh [DEST] [CONFIG_JSON]
#
#   DEST         where to put the data   (default: data/multiface)
#   CONFIG_JSON  a manifest from the multiface repo selecting which
#                subject(s)/assets to fetch.
#                (default: mini_download_config.json — the ~16 GB example set)
#                Pass "list" to print all available manifests and exit.
#
# What this trainer needs from the manifest:
#   texture  (unwrapped_uv_1024) + mesh (tracked_mesh) + metadata (KRT, stats).
# It does NOT use the raw `images/` (huge) or `audio/`, so a manifest with
#   "image": false, "audio": false
# downloads far less. The shipped `vert_mean.bin`/`vert_var.txt` and per-frame
# `*_transform.txt` are used when present; the loader falls back gracefully if
# a given manifest omits them. Reference: https://github.com/facebookresearch/multiface
set -euo pipefail

DEST=${1:-data/multiface}
CONFIG=${2:-mini_download_config.json}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_ABS="$ROOT/$DEST"
mkdir -p "$DEST_ABS"

WORK="$ROOT/.multiface_repo"
if [ ! -d "$WORK" ]; then
  echo ">> Cloning facebookresearch/multiface ..."
  git clone --depth 1 https://github.com/facebookresearch/multiface.git "$WORK"
fi

DL=""
for cand in "$WORK/download_dataset.py" "$WORK/download.py"; do
  [ -f "$cand" ] && DL="$cand" && break
done
if [ -z "$DL" ]; then
  echo "!! Could not find the downloader script in $WORK."
  echo "   Check the repo README for the current download instructions."
  exit 1
fi

if [ "$CONFIG" = "list" ]; then
  echo ">> Available download manifests (pass one as the 2nd argument):"
  find "$WORK" -maxdepth 2 -name '*.json' | sed "s|$WORK/||" | sort || true
  exit 0
fi

if [ ! -f "$WORK/$CONFIG" ]; then
  echo "!! Manifest '$CONFIG' not found in the multiface repo."
  echo "   Run with 'list' to see the available manifests:"
  echo "     bash scripts/download_multiface.sh $DEST list"
  exit 1
fi

echo ">> Downloading to $DEST_ABS using manifest $CONFIG ..."
echo "   (this trainer needs texture + mesh + metadata; image/audio are optional)"
python "$DL" --dest "$DEST_ABS" --download_config "$WORK/$CONFIG"

echo
echo ">> Done. Verify it before the long run, then train:"
echo "   python -m codec_avatars.doctor --config configs/multiface_dam.yaml --set data.root=$DEST"
echo "   python -m codec_avatars.train  --config configs/multiface_dam.yaml --set data.root=$DEST"
