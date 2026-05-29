import torch

from codec_avatars.config import LossConfig
from codec_avatars.losses import compute_losses, kl_beta, kl_divergence


def test_kl_nonnegative_and_zero_at_prior():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)  # N(0, I) == prior
    assert abs(float(kl_divergence(mu, logvar))) < 1e-6
    kl = kl_divergence(torch.randn(4, 8), torch.randn(4, 8))
    assert float(kl) >= 0.0


def test_kl_beta_warmup():
    cfg = LossConfig(w_kl=1e-3, kl_warmup_steps=100)
    assert kl_beta(cfg, 0) == 0.0
    assert kl_beta(cfg, 50) == 0.5e-3
    assert kl_beta(cfg, 100) == 1e-3
    assert kl_beta(cfg, 999) == 1e-3
    assert kl_beta(LossConfig(w_kl=1e-3, kl_warmup_steps=0), 0) == 1e-3


def test_compute_losses_finite_and_keys():
    cfg = LossConfig()
    b, v, s = 2, 32, 16
    out = {
        "verts_hat": torch.randn(b, v, 3),
        "tex_hat": torch.rand(b, 3, s, s),
        "mu": torch.randn(b, 8),
        "logvar": torch.randn(b, 8),
    }
    batch = {
        "verts": torch.randn(b, v, 3),
        "tex": torch.rand(b, 3, s, s),
        "tex_mask": torch.ones(b, 1, s, s),
    }
    total, metrics = compute_losses(out, batch, cfg, step=10)
    assert torch.isfinite(total)
    for k in ("loss/total", "loss/geom", "loss/tex", "loss/kl", "loss/beta"):
        assert k in metrics
    # perfect reconstruction -> only KL remains
    out2 = {"verts_hat": batch["verts"].clone(), "tex_hat": batch["tex"].clone(),
            "mu": torch.zeros(b, 8), "logvar": torch.zeros(b, 8)}
    total2, m2 = compute_losses(out2, batch, cfg, step=10)
    assert m2["loss/geom"] < 1e-6 and m2["loss/tex"] < 1e-6
    assert float(total2) < 1e-5
