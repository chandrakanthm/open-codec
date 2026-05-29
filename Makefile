# Convenience targets for the Codec Avatar project.
# Works whether or not the package is pip-installed (PYTHONPATH=src covers source runs).
#
#   make setup    # install deps + verify GPU (RunPod)
#   make data     # download the ~16 GB Multiface example set
#   make doctor   # PREFLIGHT: validate data + one real step + VRAM verdict
#   make train    # train the Deep Appearance Model
#   make demo     # encode->code->decode a sample, write a visual demo
#   make smoke    # tiny CPU train on synthetic data (no download)
#   make test     # run the test suite
#   make tb       # launch TensorBoard on the training run
#
# Override paths/knobs on the command line, e.g.:
#   make train DATA=/workspace/data/multiface BS=8
#   make demo CKPT=runs/multiface_dam/step_50000.pt

PY      ?= python3
CONFIG  ?= configs/multiface_dam.yaml
DATA    ?= data/multiface
OUT     ?= runs/multiface_dam
CKPT    ?= $(OUT)/latest.pt
DEMO    ?= outputs/demo
BS      ?=
DEVICE  ?= cuda

export PYTHONPATH := src:$(PYTHONPATH)

# Optional batch-size override passed through to train/doctor.
BS_FLAG := $(if $(BS),--set optim.batch_size=$(BS),)

.PHONY: setup data doctor train demo smoke test tb clean help
.DEFAULT_GOAL := help

setup:
	bash scripts/runpod_setup.sh

data:
	bash scripts/download_multiface.sh $(DATA)

doctor:
	$(PY) -m codec_avatars.doctor --config $(CONFIG) --set data.root=$(DATA) $(BS_FLAG)

train:
	$(PY) -m codec_avatars.train --config $(CONFIG) --set data.root=$(DATA) $(BS_FLAG)

demo:
	$(PY) -m codec_avatars.infer --ckpt $(CKPT) --device $(DEVICE) --index 0 --out $(DEMO) --multiview 8

smoke:
	$(PY) -m codec_avatars.train --config configs/smoke_test.yaml

test:
	$(PY) -m pytest -q

tb:
	tensorboard --logdir $(OUT) --host 0.0.0.0 --port 6006

clean:
	rm -rf runs/_setup_check runs/smoke outputs/demo

help:
	@echo "Targets: setup | data | doctor | train | demo | smoke | test | tb | clean"
	@echo "Vars:    DATA=$(DATA)  OUT=$(OUT)  CKPT=$(CKPT)  BS=$(BS)  DEVICE=$(DEVICE)"
	@echo "Typical RunPod flow:  make setup && make data && make doctor && make train && make demo"
