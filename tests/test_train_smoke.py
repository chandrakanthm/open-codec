"""End-to-end integration: train a tiny model, then run the codec demo."""

import argparse

from codec_avatars.config import Config, DataConfig, ModelConfig, OptimConfig, TrainConfig
from codec_avatars.doctor import run as doctor_run
from codec_avatars.infer import run as infer_run
from codec_avatars.train import train


def _tiny_cfg(out_dir: str) -> Config:
    return Config(
        name="test",
        model=ModelConfig(
            latent_dim=8, n_vertices=40, tex_size=32,
            enc_tex_channels=(8, 16, 32), dec_tex_channels=(32, 16, 8), mesh_hidden=(32, 16),
        ),
        data=DataConfig(name="synthetic", synthetic_size=16, synthetic_identities=2, num_workers=0, val_fraction=0.2),
        optim=OptimConfig(lr=1e-3, batch_size=4, max_steps=6, lr_decay_steps=0),
        train=TrainConfig(device="cpu", amp=False, out_dir=out_dir, log_every=2, val_every=4,
                          ckpt_every=0, tensorboard=False, seed=0),
    )


def test_train_then_infer(tmp_path):
    out_dir = str(tmp_path / "run")
    cfg = _tiny_cfg(out_dir)
    final = train(cfg)
    assert final.endswith("final.pt")

    out = str(tmp_path / "demo")
    args = argparse.Namespace(ckpt=final, index=0, out=out, device="cpu", fps=30, multiview=3)
    infer_run(args)

    for name in ("code.npy", "avatar.obj", "texture.png", "texture_target.png", "compare.png"):
        assert (tmp_path / "demo" / name).exists(), f"missing {name}"
    # multiview sweep + contact sheet
    assert (tmp_path / "demo" / "texture_view00.png").exists()
    assert (tmp_path / "demo" / "texture_view02.png").exists()
    assert (tmp_path / "demo" / "contact_sheet.png").exists()


def test_resume_from_latest(tmp_path):
    out_dir = str(tmp_path / "run")
    cfg = _tiny_cfg(out_dir)
    cfg.train.ckpt_every = 4  # write latest.pt mid-run
    train(cfg)
    assert (tmp_path / "run" / "latest.pt").exists()

    # Re-run with resume=auto: should pick up latest.pt and not crash.
    cfg2 = _tiny_cfg(out_dir)
    cfg2.train.resume = "auto"
    cfg2.optim.max_steps = 8
    final = train(cfg2)
    assert final.endswith("final.pt")


def test_doctor_passes_on_synthetic(tmp_path):
    cfg = _tiny_cfg(str(tmp_path / "run"))
    assert doctor_run(cfg) == 0
