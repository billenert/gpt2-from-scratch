import torch.nn as nn
import torch
from torch import Tensor
from jaxtyping import Float, Int
from ..config import Config

class NonPositionalEmbedding(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.embedding, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        return self.embedding[tokens]