"""
IterableDataset that mmap's a pretokenized .bin file (uint16 token IDs) and
yields random fixed-length slices. Much faster than streaming+tokenizing —
no Python tokenization in the hot path, the OS handles caching, and random
access means each batch element comes from anywhere in the corpus.

Produced by scripts/pretokenize.py.
"""
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class PretokenizedDataset(IterableDataset):
    def __init__(self, path: str, seq_len: int):
        super().__init__()
        self.path = path
        self.seq_len = seq_len

    def __iter__(self):
        # mmap inside __iter__ so each DataLoader worker opens its own handle
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        n_max = len(data) - self.seq_len - 1

        # Per-worker RNG seeded by worker id so workers don't draw identical
        # samples. (They'd still overlap in expectation, but won't lockstep.)
        info = get_worker_info()
        seed = info.id if info is not None else 0
        rng = np.random.default_rng(seed)

        while True:
            offset = int(rng.integers(0, n_max))
            chunk = np.asarray(data[offset:offset + self.seq_len + 1], dtype=np.int64)
            yield torch.from_numpy(chunk)
