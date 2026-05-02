from dataclasses import asdict, replace

from model.config import Config


def test_default_construction():
    cfg = Config()
    assert cfg.d_model == 768
    assert cfg.n_heads == 12
    assert cfg.n_layers == 12


def test_override_by_kwargs():
    cfg = Config(d_model=128, n_heads=4, d_vocab=1000)
    assert cfg.d_model == 128
    assert cfg.n_heads == 4
    assert cfg.d_vocab == 1000
    # untouched fields keep defaults
    assert cfg.n_layers == 12


def test_equality():
    assert Config(d_model=10) == Config(d_model=10)
    assert Config(d_model=10) != Config(d_model=20)


def test_asdict_roundtrip():
    cfg = Config(d_model=64, n_heads=2)
    d = asdict(cfg)
    assert d["d_model"] == 64
    assert d["n_heads"] == 2
    # round-trips through dict
    assert Config(**d) == cfg


def test_replace_returns_copy():
    cfg = Config(d_model=64)
    cfg2 = replace(cfg, d_model=128)
    assert cfg2.d_model == 128
    assert cfg.d_model == 64  # original unchanged
