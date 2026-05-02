import torch
import tiktoken
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset


class FineWebEduStream(IterableDataset):

    def __init__(self, seq_len: int, compact_every: int = 1_000_000):
        super().__init__()
        self.seq_len = seq_len
        self.compact_every = compact_every

    def __iter__(self):
        # per worker setup
        tokenizer = tiktoken.get_encoding("gpt2")
        eot = tokenizer.eot_token

        ds = load_dataset(
            "karpathy/fineweb-edu-100b-shuffle",
            split="train",
            streaming=True,
        )

        # shard if num workers > 0
        info = get_worker_info()
        if info is not None:
            ds = ds.shard(num_shards=info.num_workers, index=info.id)

        chunk_size = self.seq_len + 1
        buffer: list[int] = []
        start = 0 

        for example in ds:
            buffer.extend(tokenizer.encode_ordinary(example["text"]))
            buffer.append(eot)

            while len(buffer) - start >= chunk_size:
                yield torch.tensor(buffer[start:start + chunk_size], dtype=torch.long)
                start += chunk_size

            if start >= self.compact_every:
                buffer = buffer[start:]
                start = 0
