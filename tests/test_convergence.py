import pytest
import torch
from model.config import Config
from model.transformer import Transformer
# from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_cfg(**overrides) -> Config:
    base = dict(d_model=32, n_heads=4, d_head=8, d_mlp=64, n_ctx=16, n_layers=2, d_vocab=50)
    base.update(overrides)
    return Config(**base)

def random_tokens(batch: int, seq_len: int, d_vocab: int) -> torch.Tensor:
    return torch.randint(0, d_vocab, (batch, seq_len))

@pytest.mark.slow
def test_overfits_a_tiny_batch():
    torch.manual_seed(0) 
    cfg = make_cfg()
    model = Transformer(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    tokens = random_tokens(2, seq_len=cfg.n_ctx, d_vocab=cfg.d_vocab).to(device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    epochs = 100
    initial_loss = -torch.inf

    # pbar = tqdm(range(epochs))
    for epoch in range(epochs):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.d_vocab),
            targets.reshape(-1)
        )
        if epoch == 0:
            initial_loss = loss.item()
        # pbar.set_postfix(training_loss=loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * initial_loss

