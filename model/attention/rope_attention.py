import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import Tensor
from jaxtyping import Float
from ..config import Config


class RoPEAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # WILL EVENTUALLY CONCAT EVERYTHING TOGETHER IN FUTURE WHEN NOT USING EINOPS CODE
        self.W_Q = nn.Parameter(torch.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))

        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_Q, std=cfg.init_range)
        nn.init.normal_(self.W_K, std=cfg.init_range)
        nn.init.normal_(self.W_V, std=cfg.init_range)
        nn.init.normal_(self.W_O, std=cfg.init_range / (2 * cfg.n_layers) ** 0.5)

        rope_cos, rope_sin = self.build_rope_cache(cfg.n_ctx, cfg.d_head)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def build_rope_cache(
        self,
        seq_len: int,
        d_head: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Float[Tensor, "1 seq_len 1 d_head//2"], Float[Tensor, "1 seq_len 1 d_head//2"]]:
        # Builds cos/sin RoPE tables, shaped to broadcast over (batch, seq, n_heads, d_head//2).
        assert d_head % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, :, None, :].to(dtype)
        sin = freqs.sin()[None, :, None, :].to(dtype)
        return cos, sin

    def RoPE(self, vec: Float[Tensor, "batch seq_len n_heads d_head"]) -> Float[Tensor, "batch seq_len n_heads d_head"]:
        seq_len = vec.shape[1]
        cos = self.rope_cos[:, :seq_len]
        sin = self.rope_sin[:, :seq_len]

        vec_even = vec[..., 0::2]
        vec_odd = vec[..., 1::2]

        out = torch.empty_like(vec)
        out[..., 0::2] = vec_even * cos - vec_odd * sin
        out[..., 1::2] = vec_even * sin + vec_odd * cos
        return out

    def forward(self, resid_stream: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        q = einops.einsum(
            resid_stream, self.W_Q,
            "batch seq_len d_model, d_model n_heads d_head -> batch seq_len n_heads d_head",
        ) + self.b_Q
        k = einops.einsum(
            resid_stream, self.W_K,
            "batch seq_len d_model, d_model n_heads d_head -> batch seq_len n_heads d_head",
        ) + self.b_K
        v = einops.einsum(
            resid_stream, self.W_V,
            "batch seq_len d_model, d_model n_heads d_head -> batch seq_len n_heads d_head",
        ) + self.b_V

        q = self.RoPE(q)
        k = self.RoPE(k)

        # SDPA expects (batch, n_heads, seq, d_head). Transpose in, run, transpose out.
        # is_causal=True replaces the explicit upper-triangular mask and lets PyTorch
        # pick the FlashAttention kernel where available.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended_queries = attended.transpose(1, 2)

        out = einops.einsum(
            attended_queries, self.W_O,
            "batch seq_len n_heads d_head, n_heads d_head d_model -> batch seq_len d_model",
        ) + self.b_O
        return out
