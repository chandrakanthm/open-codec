#!/usr/bin/env bash
# Create a RunPod *template* (a saved pod config) that runs the Codec Avatars
# demo, using the runpodctl CLI. Creating a template is free; launching a pod
# from it costs money.
#
# Prereqs:
#   1) runpodctl authenticated:   runpodctl doctor     (or: export RUNPOD_API_KEY=...)
#   2) IMAGE built & pushed:       IMAGE=... bash scripts/runpod_build_push.sh
#
# Usage:
#   IMAGE=docker.io/youruser/codec-avatars:latest bash scripts/runpod_template.sh
#
# Tunables (env):
#   NAME=codec-avatars-demo  DEMO_MODE=smoke  CONTAINER_DISK_GB=30
#   VOLUME_GB=50  PORTS=22/tcp,6006/http
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=<registry>/<repo>:<tag> (the pushed demo image). Build it with scripts/runpod_build_push.sh}"
NAME="${NAME:-codec-avatars-demo}"
DEMO_MODE="${DEMO_MODE:-smoke}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
VOLUME_GB="${VOLUME_GB:-50}"
PORTS="${PORTS:-22/tcp,6006/http}"
# For a PRIVATE image: store creds once with `runpodctl registry create` and pass
# the resulting id here. It attaches at `pod create` (templates have no auth flag).
REGISTRY_AUTH_ID="${REGISTRY_AUTH_ID:-}"
AUTH_FLAG=""
[ -n "$REGISTRY_AUTH_ID" ] && AUTH_FLAG=" --registry-auth-id $REGISTRY_AUTH_ID"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "!! runpodctl not found. Install it: https://github.com/runpod/runpodctl"
  exit 1
fi
if ! runpodctl user >/dev/null 2>&1; then
  echo "!! runpodctl is not authenticated."
  echo "   Run:  runpodctl doctor    (prompts for + saves your API key)"
  echo "   or:   export RUNPOD_API_KEY=<your-key>"
  exit 1
fi

echo ">> Creating template '$NAME'"
echo "     image      = $IMAGE"
echo "     demo mode  = $DEMO_MODE"
echo "     disk/vol   = ${CONTAINER_DISK_GB}GB container / ${VOLUME_GB}GB volume @ /workspace"
echo "     ports      = $PORTS"

OUT="$(runpodctl template create \
  --name "$NAME" \
  --image "$IMAGE" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --volume-in-gb "$VOLUME_GB" \
  --volume-mount-path /workspace \
  --ports "$PORTS" \
  --env "{\"DEMO_MODE\":\"$DEMO_MODE\"}" \
  --docker-start-cmd "bash,-lc,/opt/codec-avatars/scripts/runpod_demo_entrypoint.sh" \
  --readme "Codec Avatars demo. Boot -> GPU doctor -> smoke train+infer -> writes /workspace/outputs/demo/contact_sheet.png, then idles for SSH. Set DEMO_MODE=multiface for the real Multiface run." \
  -o json)"

echo "$OUT"

# Best-effort: pull the new template id out of the JSON to print a launch command.
TPL_ID="$(printf '%s' "$OUT" | python3 -c '
import sys, json
def find_id(o):
    if isinstance(o, dict):
        for k in ("id", "templateId"):
            v = o.get(k)
            if isinstance(v, str) and v:
                return v
        for v in o.values():
            r = find_id(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_id(v)
            if r:
                return r
    return ""
try:
    print(find_id(json.load(sys.stdin)))
except Exception:
    print("")
' 2>/dev/null || true)"

echo
echo "================================================================"
echo "Template '$NAME' created."
TID="${TPL_ID:-<ID>}"
echo "Launch a pod from it (pick a GPU from 'runpodctl gpu list'):"
echo
echo "  runpodctl pod create --template-id $TID \\"
echo "    --gpu-id \"NVIDIA GeForce RTX 4090\" --name codec-demo --gpu-count 1$AUTH_FLAG"
[ -z "$TPL_ID" ] && echo "  # (find the id with: runpodctl template list)"
if [ -z "$REGISTRY_AUTH_ID" ]; then
  echo
  echo "Private image? Store creds once (RunPod pulls with them), then add --registry-auth-id:"
  echo "  runpodctl registry create --name myreg --username <user> --password <token>   # prints an id"
  echo "  REGISTRY_AUTH_ID=<id> bash scripts/runpod_template.sh   # re-run to bake the launch line"
fi
echo
echo "Then watch it boot:   runpodctl pod list   &&   (web console > Logs)"
echo "Outputs land in:      /workspace/outputs/demo/   (contact_sheet.png, compare.png)"
echo "================================================================"
