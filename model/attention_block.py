import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from .config import Config


class AttentionBlock(nn.Module):
    def __init__(
        self,
        cfg: Config,
        *,
        attention_cls: type[nn.Module],
        norm_cls: type[nn.Module],
        activation_cls: type[nn.Module],
    ):
        super().__init__()
        self.norm1 = norm_cls(cfg.d_model)
        self.attention = attention_cls(cfg)
        self.norm2 = norm_cls(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            activation_cls(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(self, resid_stream: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        resid_mid = resid_stream + self.attention(self.norm1(resid_stream))
        resid_post = resid_mid + self.mlp(self.norm2(resid_mid))
        return resid_post
