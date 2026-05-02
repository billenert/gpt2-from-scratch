import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Int, Float

from .config import Config
from .attention_block import AttentionBlock
from .attention.rope_attention import RoPEAttention
from .embedding.non_positional_embedding import NonPositionalEmbedding
from .activation_functions import GeLU


class Transformer(nn.Module):

    def __init__(
        self,
        cfg: Config,
        *,
        embedding_cls: type[nn.Module] = NonPositionalEmbedding,
        attention_cls: type[nn.Module] = RoPEAttention,
        norm_cls: type[nn.Module] = nn.LayerNorm,
        activation_cls: type[nn.Module] = GeLU,
    ):
        super().__init__()
        self.embed = embedding_cls(cfg)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                cfg,
                attention_cls=attention_cls,
                norm_cls=norm_cls,
                activation_cls=activation_cls,
            )
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = norm_cls(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab)

    def forward(self, tokens: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len d_vocab"]:
        resid_stream = self.embed(tokens)
        for attention_block in self.attention_blocks:
            resid_stream = attention_block(resid_stream)
        resid_stream = self.final_norm(resid_stream)
        return self.unembed(resid_stream)
