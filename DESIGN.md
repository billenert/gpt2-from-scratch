## Design:

# Model:
Model consists of a GPT2-style transformer using RoPE positional embedding. Should consist of a modular transformer class, as well as attention mechanisms, norm functions, MLP, and positional embedding implementations.

# Data:
Data will be taken from https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle. We will use GPT2's tokenizer from transformer's AutoTokenizer.

# Training Loop:
We will use AdamW optimizer and (... to be determined)