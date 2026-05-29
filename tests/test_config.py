from codec_avatars.config import apply_overrides, config_from_dict, load_config, to_dict


def test_defaults_roundtrip():
    cfg = config_from_dict({})
    d = to_dict(cfg)
    assert d["model"]["latent_dim"] == 256
    assert isinstance(d["model"]["enc_tex_channels"], list)  # tuples -> lists
    # rebuild from serialized dict
    cfg2 = config_from_dict(d)
    assert cfg2.model.latent_dim == cfg.model.latent_dim


def test_partial_dict_merges_with_defaults():
    cfg = config_from_dict({"model": {"latent_dim": 32}, "data": {"name": "synthetic"}})
    assert cfg.model.latent_dim == 32
    assert cfg.model.tex_size == 1024  # untouched default
    assert cfg.data.name == "synthetic"


def test_cli_overrides_and_coercion():
    cfg = config_from_dict({})
    cfg = apply_overrides(cfg, ["optim.lr=3e-4", "train.amp=false", "model.tex_size=256"])
    assert abs(cfg.optim.lr - 3e-4) < 1e-12
    assert cfg.train.amp is False
    assert cfg.model.tex_size == 256 and isinstance(cfg.model.tex_size, int)


def test_load_smoke_config():
    cfg = load_config("configs/smoke_test.yaml")
    assert cfg.data.name == "synthetic"
    assert cfg.train.device == "cpu"
    assert cfg.model.tex_size == 64
