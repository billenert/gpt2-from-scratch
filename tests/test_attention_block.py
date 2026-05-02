import torch
import torch.nn as nn

from model.config import Config
from model.attention_block import AttentionBlock
from model.attention.rope_attention import RoPEAttention
from model.activation_functions import GeLU


def make_cfg(**overrides) -> Config:
    base = dict(d_model=32, n_heads=4, d_head=8, d_mlp=64, n_ctx=16, n_layers=2, d_vocab=50)
    base.update(overrides)
    return Config(**base)


def make_block(cfg: Config) -> AttentionBlock:
    return AttentionBlock(
        cfg,
        attention_cls=RoPEAttention,
        norm_cls=nn.LayerNorm,
        activation_cls=GeLU,
    )


def test_constructs():
    make_block(make_cfg())


def test_submodules_registered():
    block = make_block(make_cfg())
    children = {name for name, _ in block.named_children()}
    assert {"norm1", "attention", "norm2", "mlp"} <= children


def test_output_shape_matches_input():
    cfg = make_cfg()
    block = make_block(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    assert block(x).shape == x.shape


def test_handles_short_sequences():
    cfg = make_cfg(n_ctx=16)
    block = make_block(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    assert block(x).shape == (2, 5, cfg.d_model)


def test_gradients_flow_to_all_parameters():
    cfg = make_cfg()
    block = make_block(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    block(x).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.abs().sum() > 0, f"{name} has zero gradient"


def test_residual_connection_preserves_input():
    """If attention and MLP both output zeros, the block should be the identity (after layernorms).
    We zero the W_O of attention and the second linear of the MLP to kill both branches' contributions."""
    cfg = make_cfg()
    block = make_block(cfg)
    with torch.no_grad():
        block.attention.W_O.zero_()
        block.attention.b_O.zero_()
        block.mlp[2].weight.zero_()
        block.mlp[2].bias.zero_()
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    assert torch.allclose(block(x), x, atol=1e-5)


def test_forward_is_deterministic():
    cfg = make_cfg()
    torch.manual_seed(0); b1 = make_block(cfg)
    torch.manual_seed(0); b2 = make_block(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    assert torch.equal(b1(x), b2(x))
