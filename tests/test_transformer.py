import pytest
import torch
import torch.nn as nn

from model.config import Config
from model.transformer import Transformer


def make_cfg(**overrides) -> Config:
    base = dict(d_model=32, n_heads=4, d_head=8, d_mlp=64, n_ctx=16, n_layers=2, d_vocab=50)
    base.update(overrides)
    return Config(**base)


def random_tokens(batch: int, seq_len: int, d_vocab: int) -> torch.Tensor:
    return torch.randint(0, d_vocab, (batch, seq_len))


# --- construction ---

def test_constructs_with_defaults():
    Transformer(make_cfg())


def test_top_level_modules_registered():
    m = Transformer(make_cfg())
    children = {name for name, _ in m.named_children()}
    assert {"embed", "attention_blocks", "final_norm", "unembed"} <= children


def test_correct_number_of_blocks():
    cfg = make_cfg(n_layers=5)
    m = Transformer(cfg)
    assert len(m.attention_blocks) == 5


# --- shapes ---

def test_forward_output_shape():
    cfg = make_cfg()
    m = Transformer(cfg)
    tokens = random_tokens(2, cfg.n_ctx, cfg.d_vocab)
    out = m(tokens)
    assert out.shape == (2, cfg.n_ctx, cfg.d_vocab)


def test_handles_short_sequences():
    cfg = make_cfg(n_ctx=16)
    m = Transformer(cfg)
    tokens = random_tokens(2, 5, cfg.d_vocab)
    out = m(tokens)
    assert out.shape == (2, 5, cfg.d_vocab)


# --- causality (the integration test) ---

def test_causality_end_to_end():
    """Logits at position t should be invariant to tokens at positions > t."""
    torch.manual_seed(0)
    cfg = make_cfg(n_ctx=8)
    m = Transformer(cfg).eval()

    tokens1 = random_tokens(1, cfg.n_ctx, cfg.d_vocab)
    tokens2 = tokens1.clone()
    tokens2[:, -1] = (tokens2[:, -1] + 1) % cfg.d_vocab  # change only the last token

    with torch.no_grad():
        out1 = m(tokens1)
        out2 = m(tokens2)

    # all positions except the last should produce identical logits
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    # the last position is allowed to differ
    assert not torch.allclose(out1[:, -1], out2[:, -1])


# --- gradient flow ---

def test_gradients_flow_to_all_parameters():
    cfg = make_cfg()
    m = Transformer(cfg)
    tokens = random_tokens(2, cfg.n_ctx, cfg.d_vocab)
    m(tokens).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"parameters with no gradient: {missing}"


# --- determinism ---

def test_forward_is_deterministic():
    cfg = make_cfg()
    torch.manual_seed(0); m1 = Transformer(cfg)
    torch.manual_seed(0); m2 = Transformer(cfg)
    tokens = random_tokens(2, cfg.n_ctx, cfg.d_vocab)
    assert torch.equal(m1(tokens), m2(tokens))


# --- dependency injection ---

def test_custom_activation_class_is_used():
    """The activation_cls kwarg should actually take effect."""
    cfg = make_cfg()
    m = Transformer(cfg, activation_cls=nn.ReLU)
    # peek into the first block's mlp — index 1 is the activation in the Sequential
    assert isinstance(m.attention_blocks[0].mlp[1], nn.ReLU)


# --- device portability ---

@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(),
                    reason="no accelerator available")
def test_moves_to_device():
    device = "cuda" if torch.cuda.is_available() else "mps"
    cfg = make_cfg()
    m = Transformer(cfg).to(device)
    tokens = random_tokens(2, cfg.n_ctx, cfg.d_vocab).to(device)
    out = m(tokens)
    assert out.device.type == device
