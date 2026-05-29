import pytest
import torch

from codec_avatars.config import ModelConfig
from codec_avatars.models import build_model
from codec_avatars.models.layers import n_scales, resize_schedule


def tiny_cfg(tex_size=64, n_vertices=64, latent=8):
    return ModelConfig(
        latent_dim=latent, n_vertices=n_vertices, tex_size=tex_size,
        enc_tex_channels=(8, 16, 32, 32, 32), dec_tex_channels=(32, 32, 32, 16, 8),
        mesh_hidden=(32, 16),
    )


def test_n_scales_and_schedule():
    assert n_scales(1024) == 8
    assert n_scales(64) == 4
    with pytest.raises(ValueError):
        n_scales(48)  # not a power of two
    assert resize_schedule((1, 2, 3, 4, 5, 6, 7, 8), 4, tail=False) == [1, 2, 3, 4]
    assert resize_schedule((1, 2, 3, 4, 5, 6, 7, 8), 4, tail=True) == [5, 6, 7, 8]
    assert resize_schedule((9,), 3, tail=True) == [9, 9, 9]


@pytest.mark.parametrize("tex_size", [32, 64, 128])
def test_forward_shapes(tex_size):
    cfg = tiny_cfg(tex_size=tex_size)
    m = build_model(cfg)
    b = 3
    batch = {
        "avg_tex": torch.rand(b, 3, tex_size, tex_size),
        "verts": torch.randn(b, cfg.n_vertices, 3),
        "view": torch.randn(b, 3),
        "tex": torch.rand(b, 3, tex_size, tex_size),
    }
    m.train()
    out = m(batch)
    assert out["verts_hat"].shape == (b, cfg.n_vertices, 3)
    assert out["tex_hat"].shape == (b, 3, tex_size, tex_size)
    assert out["mu"].shape == (b, cfg.latent_dim)
    assert out["logvar"].shape == (b, cfg.latent_dim)
    # texture is bounded to [0,1] by the sigmoid head
    assert out["tex_hat"].min() >= 0.0 and out["tex_hat"].max() <= 1.0


def test_backward_produces_grads():
    cfg = tiny_cfg()
    m = build_model(cfg)
    batch = {
        "avg_tex": torch.rand(2, 3, 64, 64),
        "verts": torch.randn(2, cfg.n_vertices, 3),
        "view": torch.randn(2, 3),
        "tex": torch.rand(2, 3, 64, 64),
    }
    m.train()
    out = m(batch)
    (out["verts_hat"].pow(2).mean() + out["tex_hat"].pow(2).mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_logvar_is_clamped():
    cfg = tiny_cfg()
    m = build_model(cfg)
    batch = {"avg_tex": torch.rand(2, 3, 64, 64) * 50, "verts": torch.randn(2, cfg.n_vertices, 3) * 50,
             "view": torch.randn(2, 3), "tex": torch.rand(2, 3, 64, 64)}
    mu, logvar = m.encode(batch["avg_tex"], batch["verts"])
    assert logvar.min() >= cfg.min_logvar - 1e-5
    assert logvar.max() <= cfg.max_logvar + 1e-5


def test_geometry_standardization_roundtrip():
    cfg = tiny_cfg()
    m = build_model(cfg)
    mean = torch.randn(cfg.n_vertices, 3) * 100
    m.set_geometry_stats(mean, 12.5)
    v = torch.randn(4, cfg.n_vertices, 3) * 100
    back = m.unstandardize_geometry(m.standardize_geometry(v))
    assert torch.allclose(v, back, atol=1e-3)


def test_codec_roundtrip_and_bytes():
    cfg = tiny_cfg(latent=16)
    m = build_model(cfg)
    m.eval()
    avg_tex = torch.rand(1, 3, 64, 64)
    verts = torch.randn(1, cfg.n_vertices, 3)
    code = m.encode_code(avg_tex, verts)
    assert code.shape == (1, 16)
    vh, th = m.decode(code, torch.randn(1, 3))
    assert vh.shape == (1, cfg.n_vertices, 3)
    assert m.code_bytes() == 16 * 4
    assert m.code_bytes(2) == 16 * 2
