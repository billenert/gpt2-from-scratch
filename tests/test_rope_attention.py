import pytest
import torch

from model.config import Config
from model.attention.rope_attention import RoPEAttention


def make_cfg(**overrides) -> Config:
    base = dict(d_model=32, n_heads=4, d_head=8, d_mlp=64, n_ctx=16, n_layers=2, d_vocab=50)
    base.update(overrides)
    return Config(**base)


# --- construction ---

def test_constructs():
    RoPEAttention(make_cfg())


def test_parameters_registered():
    m = RoPEAttention(make_cfg())
    names = {n for n, _ in m.named_parameters()}
    assert {"W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"} <= names


def test_buffers_registered():
    m = RoPEAttention(make_cfg())
    names = {n for n, _ in m.named_buffers()}
    assert {"rope_cos", "rope_sin", "causal_attention_mask"} <= names


def test_biases_initialized_to_zero():
    m = RoPEAttention(make_cfg())
    for name in ("b_Q", "b_K", "b_V", "b_O"):
        b = getattr(m, name)
        assert torch.all(b == 0), f"{name} should init to zero"


# --- shapes ---

def test_forward_output_shape():
    cfg = make_cfg()
    m = RoPEAttention(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    assert m(x).shape == (2, cfg.n_ctx, cfg.d_model)


def test_forward_handles_short_sequences():
    cfg = make_cfg(n_ctx=16)
    m = RoPEAttention(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    assert m(x).shape == (2, 5, cfg.d_model)


# --- RoPE properties ---

def test_rope_preserves_norm():
    """RoPE is a rotation, so it should preserve vector magnitudes (per pair)."""
    cfg = make_cfg()
    m = RoPEAttention(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.n_heads, cfg.d_head)
    y = m.RoPE(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    """At position 0, the RoPE rotation angle is 0, so input should pass through unchanged."""
    cfg = make_cfg()
    m = RoPEAttention(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.n_heads, cfg.d_head)
    y = m.RoPE(x)
    assert torch.allclose(x[:, 0], y[:, 0], atol=1e-5)


# --- causality ---

def test_causal_mask_blocks_future_positions():
    """Output at position t must not depend on inputs at positions > t."""
    torch.manual_seed(0)
    cfg = make_cfg(n_ctx=8)
    m = RoPEAttention(cfg).eval()

    x1 = torch.randn(1, cfg.n_ctx, cfg.d_model)
    x2 = x1.clone()
    x2[:, -1] = torch.randn(cfg.d_model)

    with torch.no_grad():
        y1 = m(x1)
        y2 = m(x2)

    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)
    assert not torch.allclose(y1[:, -1], y2[:, -1])


# --- gradient flow ---

def test_gradients_flow_to_all_parameters():
    cfg = make_cfg()
    m = RoPEAttention(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    m(x).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.abs().sum() > 0, f"{name} has zero gradient"


# --- device portability (skip if no accelerator) ---

@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(),
                    reason="no accelerator available")
def test_moves_to_device():
    device = "cuda" if torch.cuda.is_available() else "mps"
    cfg = make_cfg()
    m = RoPEAttention(cfg).to(device)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model, device=device)
    out = m(x)
    assert out.device.type == device
    # buffers should have moved too
    assert m.rope_cos.device.type == device
    assert m.causal_attention_mask.device.type == device


# --- determinism ---

def test_forward_is_deterministic():
    cfg = make_cfg()
    torch.manual_seed(0); m1 = RoPEAttention(cfg)
    torch.manual_seed(0); m2 = RoPEAttention(cfg)
    x = torch.randn(2, cfg.n_ctx, cfg.d_model)
    assert torch.equal(m1(x), m2(x))
