"""
Entry point for training a from-scratch GPT.

Usage:
    python scripts/train.py                                   # full run
    python scripts/train.py --quick                           # 1M-token sanity run
    python scripts/train.py --no-wandb                        # disable wandb
    python scripts/train.py --run-name baseline-256d-6L       # name the wandb run
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import FineWebEduStream
from model.config import Config
from model.transformer import Transformer
from training.trainer import Trainer


SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default=None, help="wandb run name")
    p.add_argument("--n-tokens", type=int, default=300_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/run")
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    p.add_argument(
        "--quick",
        action="store_true",
        help="sanity run: 1M tokens, no wandb, frequent logging",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from a checkpoint path, or pass --resume with no value to load the latest in --ckpt-dir",
    )
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir: str | Path) -> Path:
    paths = list(Path(ckpt_dir).glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no step_*.pt checkpoints in {ckpt_dir}")
    return max(paths, key=lambda p: int(p.stem.split("_")[1]))


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16 autocast is iffy on MPS — keep fp32
    return "cpu", torch.float32


def main():
    args = parse_args()
    if args.quick:
        args.n_tokens = 1_000_000
        args.no_wandb = True
        print("[quick] 1M-token sanity run, wandb disabled")

    device, dtype = pick_device_and_dtype()
    print(f"device={device}  dtype={dtype}")

    cfg = Config(
        d_model=256,
        n_heads=8,
        d_head=32,
        d_mlp=1024,
        n_layers=6,
        n_ctx=512,
        d_vocab=50257,
    )
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params")

    dataset = FineWebEduStream(seq_len=cfg.n_ctx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    if not args.no_wandb:
        import wandb
        wandb.init(
            project="gpt2-from-scratch",
            name=args.run_name,
            config={
                **asdict(cfg),
                "n_tokens": args.n_tokens,
                "batch_size": args.batch_size,
                "peak_lr": args.peak_lr,
                "n_params": n_params,
                "device": device,
                "dtype": str(dtype),
            },
        )

    trainer = Trainer(
        model,
        loader,
        n_tokens=args.n_tokens,
        peak_lr=args.peak_lr,
        log_every=10 if args.quick else 50,
        ckpt_every=500 if args.quick else 2000,
        eval_every=0,  # no val loader yet
        ckpt_dir=args.ckpt_dir,
        device=device,
        dtype=dtype,
        sample_prompts=SAMPLE_PROMPTS,
        sample_every=200 if args.quick else 2000,
        sample_max_new_tokens=40,
        n_ctx=cfg.n_ctx,
    )

    if args.resume:
        ckpt_path = (
            find_latest_checkpoint(args.ckpt_dir) if args.resume == "auto" else Path(args.resume)
        )
        print(f"resuming from {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        print(f"  step={trainer.step}  tokens_seen={trainer.tokens_seen / 1e6:.1f}M")

    try:
        trainer.fit()
    finally:
        if not args.no_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
