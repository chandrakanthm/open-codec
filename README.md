# Codec Avatars (open source)

A from-scratch, Apache-2.0 implementation of a **neural Codec Avatar**: a
conditional VAE — a *Deep Appearance Model* — that **encodes** a face into a
compact latent *code* and **decodes** that code back into a **view-dependent 3D
avatar** (mesh geometry + texture).

This is the open, documented lineage behind Meta Reality Labs' Codec Avatars
(Lombardi et al., *Deep Appearance Models for Face Rendering*, SIGGRAPH 2018) and
the architecture you can actually train on **public** multi-view data such as
Meta's **Multiface** dataset. Apple's Vision Pro "Personas" are a closed system;
there is no public model to replicate, so this project follows the open Meta line
and is designed to train on a single GPU (e.g. on [RunPod](https://runpod.io)).

> **Why "codec"?** In telepresence you can't stream multi-view video of a face.
> Instead you *encode* each frame into a few hundred floats (the code), transmit
> that, and *decode* a photoreal-ish 3D avatar on the far end. This repo makes
> that loop concrete and even reports the code's byte size / equivalent bitrate.

---

## How it works

```
                 ┌──────────────────── ENCODER ────────────────────┐
 view-averaged   │  texture conv ↓↓↓ ─┐                             │
 texture  ───────┼────────────────────┤── fuse ─→ μ, logσ² ─→  z    │   "the code"
 tracked mesh ───┼──  mesh MLP ────────┘            (reparam.)      │   (latent_dim
                 └──────────────────────────────────────────────────┘    floats/frame)
                                                       │
                          z  +  view direction  ───────┤
                 ┌──────────────────── DECODER ────────▼────────────┐
                 │  mesh MLP        ─→  vertices  (view-independent) │
                 │  texture conv ↑↑↑ ─→  texture  (view-DEPENDENT)   │
                 └──────────────────────────────────────────────────┘
```

- **Geometry is view-independent**; the **texture is conditioned on the view
  direction**, so the model reproduces specular/Fresnel-like changes as the head
  turns. (This is the core idea of the Deep Appearance Model.)
- The encoder consumes the **view-averaged** texture + mesh; the decoder is asked
  to reproduce a **specific camera's** texture given that camera's view vector.
- Trained as a **β-VAE**: geometry MSE + texture L1 + KL, with the KL weight
  linearly annealed in to avoid posterior collapse.

Key modules:

| File | Role |
|------|------|
| [`models/encoder.py`](src/codec_avatars/models/encoder.py) | `(avg texture, mesh) → μ, logσ²` |
| [`models/decoder.py`](src/codec_avatars/models/decoder.py) | `z → mesh`, `(z, view) → texture` |
| [`models/codec_avatar.py`](src/codec_avatars/models/codec_avatar.py) | the VAE + the `encode_code` / `decode` codec API |
| [`losses.py`](src/codec_avatars/losses.py) | geometry + texture + KL (with anneal) |
| [`data/multiface.py`](src/codec_avatars/data/multiface.py) | Multiface loader (shipped avg/stats/headpose, KRT → view dirs) |
| [`data/synthetic.py`](src/codec_avatars/data/synthetic.py) | download-free dataset for tests/smoke |
| [`train.py`](src/codec_avatars/train.py) · [`infer.py`](src/codec_avatars/infer.py) | training loop · encode→code→decode demo |
| [`doctor.py`](src/codec_avatars/doctor.py) | preflight: validate data + one real step + VRAM verdict |

---

## Quickstart (local, CPU — no download)

```bash
pip install -e .            # or: pip install -r requirements.txt

# Train a tiny model on synthetic data (~seconds on CPU)
python -m codec_avatars.train --config configs/smoke_test.yaml

# Encode a face → code, then decode the avatar + a yaw sweep
python -m codec_avatars.infer --ckpt runs/smoke/final.pt --out outputs/demo --multiview 8
```

`infer` prints a codec report and writes `code.npy`, `avatar.obj` (+ `.mtl`),
`texture*.png`, a target-vs-reconstruction `compare.png`, and (with `--multiview`)
a single `contact_sheet.png` of the yaw sweep:

```
latent code dim   : 16
code size / frame : 64 B (float32)   |   32 B (float16)
stream @  30 fps  : 15.4 kbit/s (fp32) | 7.7 kbit/s (fp16)
geometry RMSE     : 4.675 mm   (decoded vs. tracked)
```

Run the tests:

```bash
pytest            # CPU-only, ~2s (incl. an on-disk Multiface-layout fixture)
```

There's also a `Makefile` with one-word targets — `make smoke` / `make test`
locally, and `make data doctor train demo` for the RunPod flow below.

---

## Train for real on RunPod

The whole flow is five `make` targets. RunPod time is metered, so step 3
(**`doctor`**) is the one that earns its keep: it builds the *real* dataset and
runs one forward+backward at your batch size, printing peak VRAM and a fit
verdict in seconds — catching a mis-shaped download or an OOM *before* you commit
to an hours-long run.

1. **Start a pod** with a "PyTorch 2.x / CUDA 12.x" template (A100 / 4090 / L40S).
   Mount a volume at `/workspace` for data + checkpoints.

2. **Set up** (installs the package, verifies the GPU, runs a 10-step check):

   ```bash
   git clone <your-fork>/codec-avatars && cd codec-avatars
   make setup                  # = bash scripts/runpod_setup.sh
   ```

   Or build the image in [`docker/Dockerfile`](docker/Dockerfile).

3. **Get data + preflight.** The downloader defaults to the ~16 GB example set;
   this trainer needs only `texture` + `mesh` + `metadata` (not the raw images):

   ```bash
   make data                   # downloads the Multiface example set to data/multiface
   make doctor                 # PREFLIGHT: validates data + one real step + VRAM verdict
   ```

4. **Train** (vertex count auto-detected; bf16 AMP; auto-resumes from `latest.pt`
   if a spot pod was preempted):

   ```bash
   make train                  # raise batch size once doctor shows headroom: make train BS=8
   make tb                     # tensorboard on 0.0.0.0:6006
   ```

5. **Demo the codec** (writes `compare.png` + a yaw `contact_sheet.png`):

   ```bash
   make demo                   # uses runs/multiface_dam/latest.pt
   ```

Every `make` target is a thin wrapper — run the underlying module directly for
full control. Any config value is overridable with `--set section.key=value`,
e.g. `--set model.tex_size=1024 --set optim.batch_size=8 --set model.latent_dim=512`.
Defaults are sized to fit a 16 GB card (`tex_size=512`, `batch_size=4`); scale up
per the `doctor` verdict. A fresh run = new `train.out_dir` (or delete `latest.pt`),
since `resume: auto` will otherwise continue the existing run.

### One-command pod via `runpodctl` (from your laptop)

Prefer to spin the demo up from the CLI instead of the web console? The repo ships
a self-contained image + a RunPod **template** that, on boot, runs `doctor` and the
synthetic smoke demo (GPU validation + a `contact_sheet.png`) and then idles for
SSH. Code is baked into the image (no clone on the GPU clock); all outputs go to
the `/workspace` volume.

```bash
# 0) one-time: authenticate the CLI (saves your key)
runpodctl doctor

# 1) build the image for amd64 and push to your registry (docker login first)
IMAGE=docker.io/youruser/codec-avatars:latest bash scripts/runpod_build_push.sh

# 2) create the template (free; prints a ready-to-run `pod create` command)
IMAGE=docker.io/youruser/codec-avatars:latest bash scripts/runpod_template.sh

# 3) launch a pod from it (costs money) — pick a GPU from `runpodctl gpu list`
runpodctl pod create --template-id <ID> --gpu-id "NVIDIA GeForce RTX 4090" --name codec-demo
```

**Private image?** You don't need a public repo. `runpodctl registry` (alias `reg`)
stores pull credentials in RunPod — it's registry *auth*, not a host, so you still
push to Docker Hub/GHCR/ECR, just privately. The auth attaches at `pod create`
(templates have no auth flag):

```bash
runpodctl registry create --name myreg --username <user> --password <token>   # prints an auth id
runpodctl pod create --template-id <ID> --registry-auth-id <AUTH_ID> \
  --gpu-id "NVIDIA GeForce RTX 4090" --name codec-demo
# or let the helper bake it into the printed command:
REGISTRY_AUTH_ID=<AUTH_ID> IMAGE=... bash scripts/runpod_template.sh
```

The boot behavior is set by the `DEMO_MODE` env var on the template/pod:
`smoke` (default — fast, no download), `multiface` (download the example set →
`doctor` → real training), or `none` (just idle). Watch progress in the pod's
**Logs**; find outputs in `/workspace/outputs/demo/`. See
[`scripts/runpod_template.sh`](scripts/runpod_template.sh),
[`runpod_build_push.sh`](scripts/runpod_build_push.sh), and
[`runpod_demo_entrypoint.sh`](scripts/runpod_demo_entrypoint.sh).

> Creating a template is free; only a running **pod** is billed. `runpodctl` 2.3+
> is required (it added the `template` command).

---

## Datasets

The decoder produces a **3D** avatar, so it needs **multi-view capture + tracked
meshes** — not just portrait images.

- **Multiface** (Meta) — the primary target. Per-camera unwrapped textures,
  tracked meshes, KRT calibration. CC-BY-NC 4.0 (non-commercial).
  <https://github.com/facebookresearch/multiface> — use
  [`scripts/download_multiface.sh`](scripts/download_multiface.sh). The loader
  needs the **texture**, **mesh**, and **metadata** assets; the raw `images/`
  and `audio/` are not used, so manifests with `"image": false` download far less.
- **Ava-256 / Goliath** (Meta) — larger, multi-identity capture for *universal*
  codec avatars. Point `data.root` at a Multiface-style layout to reuse the loader.
- **Hugging Face** — for HF-hosted mirrors of the above, use
  [`scripts/download_hf.py`](scripts/download_hf.py)
  (`--repo-id <org/dataset> --dest data/multiface`).

The loader mirrors Meta's own `dataset.py`: it auto-detects the texture dir
(`unwrapped_uv_1024` / `unwrapped_uv` / …) and uses the **shipped** view-averaged
texture (`<EXPR>/average/`), geometry stats (`vert_mean.bin` / `vert_var.txt`), and
per-frame head pose (`*_transform.txt`) when present — each with a graceful
fallback (recompute the average to a `*_avg/` cache, compute stats to
`codec_geom_stats.npz`, or a head-centroid view) so partial manifests still train.
**Respect each dataset's license and the likeness rights of captured subjects.**

---

## Configuration

Defaults live in [`config.py`](src/codec_avatars/config.py); the two shipped
presets are [`configs/smoke_test.yaml`](configs/smoke_test.yaml) (tiny, CPU) and
[`configs/multiface_dam.yaml`](configs/multiface_dam.yaml) (full GPU run).

| Section | Notable keys |
|---------|--------------|
| `model` | `latent_dim` (code size), `tex_size` (power of 2), `enc/dec_tex_channels`, `mesh_hidden` |
| `data`  | `name` (`synthetic`/`multiface`), `root`, `subjects`, `cameras`, `max_frames_per_seq` |
| `loss`  | `w_geom`, `w_tex`, `w_kl`, `kl_warmup_steps`, `geom_scale` |
| `optim` | `lr`, `batch_size`, `max_steps`, `grad_clip`, `lr_decay_steps` |
| `train` | `device`, `amp`, `out_dir`, `*_every`, `resume` |

---

## Scope, limitations & extensions

This implements the **Deep Appearance Model** faithfully and trainably. It is a
foundation, not a finished telepresence stack. Natural next steps:

- **Screen-space rendering loss** via a differentiable rasterizer
  (`nvdiffrast` / PyTorch3D) instead of pure texture/mesh supervision.
- **Volumetric / Gaussian decoder** to follow Pixel Codec Avatars (CVPR 2021) and
  Relightable Gaussian Codec Avatars (CVPR 2024).
- **Identity conditioning** for a *universal* decoder across many subjects.
- **Headset-camera driving** (encode from HMD cameras rather than the full mesh).

The model/decoder are intentionally modular so these can be swapped in.

**Responsible use.** This builds photorealistic avatars of real people. Only train
on data you are licensed to use, with the consent of the captured subjects, and
don't use it to impersonate or deceive.

---

## References

- Lombardi, Saragih, Simon, Sheikh. *Deep Appearance Models for Face Rendering.* SIGGRAPH 2018.
- Wuu et al. *Multiface: A Dataset for Neural Face Rendering.* 2022.
- Ma et al. *Pixel Codec Avatars.* CVPR 2021.
- Saito et al. *Relightable Gaussian Codec Avatars.* CVPR 2024.

## License

Apache-2.0 — see [LICENSE](LICENSE). Dataset and trained-model licenses are
separate and remain with their respective owners.
