#!/usr/bin/env bash
# Create a RunPod *template* (a saved pod config) that runs the Codec Avatars
# demo, using the runpodctl CLI. Creating a template is free; launching a pod
# from it costs money.
#
# Two ways to get the code onto the pod — pick ONE:
#
#   A) Clone-at-boot (no image build): point at a PUBLIC git repo. The template
#      uses a stock PyTorch base image and clones + installs at boot.
#        REPO_URL=https://github.com/you/your-repo.git bash scripts/runpod_template.sh
#
#   B) Baked image: build & push first (private ok — see runpod_build_push.sh).
#        IMAGE=docker.io/you/codec-avatars:latest bash scripts/runpod_template.sh
#
# Prereq: runpodctl authenticated —  runpodctl doctor   (or export RUNPOD_API_KEY=...)
#
# Tunables (env): NAME DEMO_MODE CONTAINER_DISK_GB VOLUME_GB PORTS
#                 BASE_IMAGE (clone mode)  REGISTRY_AUTH_ID (baked private image)
set -euo pipefail

NAME="${NAME:-codec-avatars-demo}"
DEMO_MODE="${DEMO_MODE:-smoke}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
VOLUME_GB="${VOLUME_GB:-50}"
PORTS="${PORTS:-22/tcp,6006/http}"
REPO_URL="${REPO_URL:-}"
IMAGE="${IMAGE:-}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"
# Baked private image: store creds once (runpodctl registry create) + pass the id;
# it attaches at `pod create` (templates have no auth flag).
REGISTRY_AUTH_ID="${REGISTRY_AUTH_ID:-}"
AUTH_FLAG=""
[ -n "$REGISTRY_AUTH_ID" ] && AUTH_FLAG=" --registry-auth-id $REGISTRY_AUTH_ID"

# --- choose how code reaches the pod -----------------------------------------
if [ -n "$REPO_URL" ]; then
  MODE="clone-at-boot"
  IMG="$BASE_IMAGE"
  CLONE_DIR="$(basename "$REPO_URL" .git)"
  # Self-contained boot: ensure git/ssh -> clone/pull -> install -> run the demo.
  # MUST contain NO commas (runpodctl splits --docker-start-cmd on commas).
  BOOT="set -e; export DEBIAN_FRONTEND=noninteractive; command -v git >/dev/null || { apt-get update && apt-get install -y --no-install-recommends git ca-certificates openssh-server; }; mkdir -p /workspace; cd /workspace; if [ -d $CLONE_DIR/.git ]; then cd $CLONE_DIR && (git pull --ff-only || true); else git clone --depth 1 $REPO_URL $CLONE_DIR && cd $CLONE_DIR; fi; pip install -q numpy pyyaml pillow tqdm tensorboard huggingface_hub requests || true; pip install -q -e . --no-deps || true; export CODE_DIR=/workspace/$CLONE_DIR WORK=/workspace; exec bash scripts/runpod_demo_entrypoint.sh"
  START_CMD="bash,-lc,$BOOT"
  README="Codec Avatars demo (clone-at-boot from $REPO_URL). Boot -> clone+install -> GPU doctor -> smoke train+infer -> /workspace/outputs/demo/contact_sheet.png, then idle. Set DEMO_MODE=multiface for the real run."
elif [ -n "$IMAGE" ]; then
  MODE="baked-image"
  IMG="$IMAGE"
  START_CMD="bash,-lc,/opt/codec-avatars/scripts/runpod_demo_entrypoint.sh"
  README="Codec Avatars demo (baked image). Boot -> GPU doctor -> smoke train+infer -> /workspace/outputs/demo/contact_sheet.png, then idle. Set DEMO_MODE=multiface for the real run."
else
  echo "!! Set REPO_URL=<public git repo>  (clone-at-boot)  OR  IMAGE=<registry>/<repo>:<tag>  (baked)."
  exit 1
fi

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

# Idempotent: RunPod requires unique template names. If one with this name
# already exists, delete it so we can recreate with the current config (note:
# `template update` cannot change the start command, so we replace instead).
EXISTING_ID="$(runpodctl template list --type user -o json 2>/dev/null | python3 -c '
import sys, json
name = sys.argv[1]
try: d = json.load(sys.stdin)
except Exception: sys.exit(0)
def walk(o):
    out = []
    if isinstance(o, list):
        for x in o: out += walk(x)
    elif isinstance(o, dict):
        if "name" in o and any(k in o for k in ("id", "templateId", "imageName")): out.append(o)
        for v in o.values(): out += walk(v)
    return out
for t in walk(d):
    if str(t.get("name", "")) == name:
        tid = t.get("id") or t.get("templateId")
        if tid: print(tid); break
' "$NAME" 2>/dev/null || true)"
if [ -n "$EXISTING_ID" ]; then
  echo ">> Existing template '$NAME' ($EXISTING_ID) found — replacing it (delete + recreate)."
  echo "   (pass NAME=<other> to keep a separate template instead.)"
  runpodctl template delete "$EXISTING_ID" >/dev/null 2>&1 || echo "   !! delete failed; create may still hit the name clash."
fi

echo ">> Creating template '$NAME'  [$MODE]"
echo "     image      = $IMG"
[ "$MODE" = clone-at-boot ] && echo "     repo       = $REPO_URL"
echo "     demo mode  = $DEMO_MODE"
echo "     disk/vol   = ${CONTAINER_DISK_GB}GB container / ${VOLUME_GB}GB volume @ /workspace"
echo "     ports      = $PORTS"

OUT="$(runpodctl template create \
  --name "$NAME" \
  --image "$IMG" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --volume-in-gb "$VOLUME_GB" \
  --volume-mount-path /workspace \
  --ports "$PORTS" \
  --env "{\"DEMO_MODE\":\"$DEMO_MODE\"}" \
  --docker-start-cmd "$START_CMD" \
  --readme "$README" \
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
if [ "$MODE" = baked-image ] && [ -z "$REGISTRY_AUTH_ID" ]; then
  echo
  echo "Private image? Store creds once (RunPod pulls with them), then add --registry-auth-id:"
  echo "  runpodctl registry create --name myreg --username <user> --password <token>   # prints an id"
  echo "  REGISTRY_AUTH_ID=<id> bash scripts/runpod_template.sh   # re-run to bake the launch line"
fi
echo
echo "Then watch it boot:   runpodctl pod list   &&   (web console > Logs)"
echo "Outputs land in:      /workspace/outputs/demo/   (contact_sheet.png, compare.png)"
echo "================================================================"
