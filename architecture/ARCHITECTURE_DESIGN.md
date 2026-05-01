# Architecture (for now, just for architecture while training):

## High Level Overview:
Our transformer should take an initial batch of sequences of dimension: `(batch_size seq_len num_tokens)` where we 1-hot encode the num_tokens, and ultimately output a sequence of dimension `(batch_size seq_len num_tokens)`, where the output at each position in the `seq_len` axis in output corresponds to a probability distribution of the most likely next tokens.

We should specify everything through a config file. 

Config file should contain the following parameters:
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    norm_mechanism: LayerNorm
    attention_mechanism: RoPE Attention
    embedding_mechanism: Non-positional Embedding

    

For now, we will not implement KV-caching. This is something we will do later.

## Components:
Our transformer looks, roughly like the following:

tokens `(batch_size seq_len num_tokens)` -> (into embedding) -> residual stream `(batch_size seq_len d_model)` -> (attention + mlp block) x n -> residual stream `(batch_size seq_len d_model)` -> (unembedding) -> output distribution `(batch_size seq_len num_tokens)`

### Embedding
Embedding will be learned via an embedding matrix of dimension `(num_tokens d_model)`

Positional embedding will be learned via RoPE Embedding, which occurs with every attention block, so it is excluded from the purposes of this section.

Learnable parameters will be: W_embed

### Attention Block:

The attention block consists of an attention mechanism and an MLP. It will take the residual stream `(batch_size seq_len d_model)` and add a returned result of `(batch_size seq_len d_model)` to the residual stream (so it'll return: computed result + residual stream). Takes in config parameters, and rope_base

Learnable parameters will be: W_K, W_Q, W_V, W_O, W_mlp_expand, W_mlp_contract.  

#### Attention mechanism:

The attention mechanism will take in `resid_stream: (batch_size seq_len d_model)`, and transform into `q: (batch_size seq_len n_heads d_head), k: (batch_size seq_len n_heads d_head), v: (batch_size seq_len n_heads d_head)`. Then, we rotate the `q, k` by applying the RoPE rotation. Then we compute scaled dot product attention using `q, k,` to get a multi-head score function of `(batch_size nheads seq_len_Q seq_len_K)`. We then apply a causal attention mask, and output with our values to get `(batch_size seq_len_Q nheads d_head)`. Then we use our W_O matrix to transform `(batch_size, seq_len_Q, nheads, d_head)` to `(batch_size, seq_len_Q, d_model).`

#### MLP
The MLP will take in the residual stream `resid_stream: (batch_size seq_len d_model)`, expand to `(batch_size seq_len d_mlp)` via a linear layer, then compress back down. 

#### Norm Mechanism:
We use nn.LayerNorm. 

Attention will look like: `x1 = x + attention(norm(x)), x2 = x1 + mlp(norm(x1))` and returns `x2`.

### Unembedding Matrix:

We will take the residual stream then expand it. Not much else to say.








