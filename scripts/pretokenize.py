"""
One-time pre-tokenization of FineWeb-EDU into a flat uint16 binary file.

Run once:
    python scripts/pretokenize.py --out data/fineweb_edu_2B.bin --tokens 2_000_000_000

Resume an interrupted run:
    python scripts/pretokenize.py --out data/fineweb_edu_2B.bin --tokens 2_000_000_000 --resume

Then training can mmap the result instead of streaming + tokenizing on the fly,
which removes the streaming/tokenization bottleneck and lets the GPU run flat-out.

GPT-2's vocab is 50257 < 65536, so token IDs fit in uint16 (2 bytes per token).
A 2B-token corpus is ~4 GB on disk.

A sidecar JSON next to the .bin tracks (examples_seen, tokens_written) so
--resume can fast-forward the dataset stream without re-tokenizing.
"""
import argparse
import json
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
    p.add_argument(
        "--resume",
        action="store_true",
        help="if --out exists and has a .meta.json sidecar, append to it instead of erroring",
    )
    p.add_argument(
        "--meta-every",
        type=int,
        default=1000,
        help="flush the metadata sidecar every N examples (controls resume granularity)",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name(out_path.name + ".meta.json")

    examples_seen = 0
    written = 0
    file_mode = "wb"

    if out_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{out_path} already exists — pass --resume to continue, or delete it"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"--resume requested but no metadata sidecar at {meta_path}; "
                f"delete {out_path} and re-run from scratch"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        examples_seen = meta["examples_seen"]
        written = meta["tokens_written"]
        file_mode = "ab"
        print(
            f"resuming: {written / 1e6:.1f}M tokens already in {out_path} "
            f"(after {examples_seen} examples)"
        )
        if written >= args.tokens:
            print("already at or past target; nothing to do")
            return

    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    ds = load_dataset(args.dataset, split="train", streaming=True)
    if examples_seen > 0:
        # ds.skip iterates through the parquet files but doesn't tokenize/yield —
        # much cheaper than re-running BPE on those documents.
        print(f"fast-forwarding stream past {examples_seen} examples...")
        ds = ds.skip(examples_seen)

    pbar = tqdm(
        total=args.tokens,
        initial=written,
        unit="tok",
        unit_scale=True,
        smoothing=0.05,
    )

    with open(out_path, file_mode) as f:
        for example in ds:
            ids = tokenizer.encode_ordinary(example["text"])
            ids.append(eot)
            arr = np.asarray(ids, dtype=np.uint16)

            remaining = args.tokens - written
            if len(arr) > remaining:
                arr = arr[:remaining]

            f.write(arr.tobytes())
            written += len(arr)
            examples_seen += 1
            pbar.update(len(arr))

            if examples_seen % args.meta_every == 0:
                _write_meta(meta_path, examples_seen, written)

            if written >= args.tokens:
                break

    _write_meta(meta_path, examples_seen, written)
    pbar.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {written / 1e6:.1f}M tokens to {out_path}  ({size_mb:.1f} MB on disk)")


def _write_meta(path: Path, examples_seen: int, tokens_written: int):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"examples_seen": examples_seen, "tokens_written": tokens_written}, f)
    tmp.replace(path)  # atomic rename so we never see a half-written meta file


if __name__ == "__main__":
    main()
