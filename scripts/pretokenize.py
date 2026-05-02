"""
One-time pre-tokenization of FineWeb-EDU into a flat uint16 binary file.

Run once:
    python scripts/pretokenize.py --out data/fineweb_edu_2B.bin --tokens 2_000_000_000

Then training can mmap the result instead of streaming + tokenizing on the fly,
which removes the streaming/tokenization bottleneck and lets the GPU run flat-out.

GPT-2's vocab is 50257 < 65536, so token IDs fit in uint16 (2 bytes per token).
A 2B-token corpus is ~4 GB on disk.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output .bin path")
    p.add_argument("--tokens", type=int, default=2_000_000_000, help="how many tokens to write")
    p.add_argument(
        "--dataset",
        type=str,
        default="karpathy/fineweb-edu-100b-shuffle",
        help="HF dataset to stream from",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — refusing to overwrite")

    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    ds = load_dataset(args.dataset, split="train", streaming=True)

    written = 0
    pbar = tqdm(total=args.tokens, unit="tok", unit_scale=True, smoothing=0.05)

    # write incrementally so memory stays bounded regardless of total size
    with open(out_path, "wb") as f:
        for example in ds:
            ids = tokenizer.encode_ordinary(example["text"])
            ids.append(eot)
            arr = np.asarray(ids, dtype=np.uint16)

            # if this example would overshoot the budget, truncate
            remaining = args.tokens - written
            if len(arr) > remaining:
                arr = arr[:remaining]

            f.write(arr.tobytes())
            written += len(arr)
            pbar.update(len(arr))

            if written >= args.tokens:
                break

    pbar.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {written / 1e6:.1f}M tokens to {out_path}  ({size_mb:.1f} MB on disk)")


if __name__ == "__main__":
    main()
