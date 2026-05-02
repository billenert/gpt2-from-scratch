import torch
from model.config import Config
from model.embedding.non_positional_embedding import NonPositionalEmbedding

def test_output_shape():
    cfg = Config(d_model=10, d_vocab=5)
    embed = NonPositionalEmbedding(cfg)
    out = embed(torch.tensor([[0, 1, 2]]))
    assert out.shape == (1, 3, 10)

def test_embedding_table_shape():
    cfg = Config(d_model=64, d_vocab=100)
    embed = NonPositionalEmbedding(cfg)
    assert embed.embedding.shape == (100, 64)

def test_embedding_initialized_near_zero():
    torch.manual_seed(0)
    cfg = Config(d_model=64, d_vocab=1000, init_range=0.02)
    embed = NonPositionalEmbedding(cfg)
    assert embed.embedding.std().item() < 0.05
    assert abs(embed.embedding.mean().item()) < 0.01

def test_lookup_matches_table():
    cfg = Config(d_model=8, d_vocab=4)
    embed = NonPositionalEmbedding(cfg)
    out = embed(torch.tensor([[2, 0]]))
    assert torch.equal(out[0, 0], embed.embedding[2])
    assert torch.equal(out[0, 1], embed.embedding[0])
