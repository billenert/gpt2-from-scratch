"""
End-to-end pretokenize → train pipeline. Run it, go to sleep.

Both steps auto-resume:
- If the pretokenized .bin already exists with a metadata sidecar, pretokenize
  fast-forwards past what's already written.
- If checkpoints exist in --ckpt-dir, training resumes from the latest.

So if anything crashes (OOM, disk-full, network blip), just re-run the same
command and it'll pick up where it left off.

Usage:
    python scripts/full_train.py --model medium --n-tokens 5_000_000_000

Anything else passes through to train.py:
    python scripts/full_train.py --model small --n-tokens 2_500_000_000 --batch-size 32 --no-wandb
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tiny", "small", "medium"], default="medium")
    p.add_argument("--n-tokens", type=int, default=5_000_000_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--peak-lr", type=float, default=2.5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/run")
    p.add_argument("--data-path", type=str, default=None,
                   help="explicit pretokenized .bin path; default is data/fineweb_edu_<N>B.bin")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--skip-pretok", action="store_true",
                   help="assume --data-path already exists and is complete; skip pretokenize step")
    return p.parse_args()


def banner(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70, flush=True)


def run(cmd: list[str]) -> int:
    """Run a subprocess, streaming output. Returns exit code."""
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def default_data_path(n_tokens: int) -> str:
    if n_tokens % 1_000_000_000 == 0:
        return f"data/fineweb_edu_{n_tokens // 1_000_000_000}B.bin"
    return f"data/fineweb_edu_{n_tokens // 1_000_000}M.bin"


def pretokenize(data_path: Path, n_tokens: int):
    """Run pretokenize. Resumes from sidecar if present; warns and re-tokenizes otherwise."""
    meta_path = data_path.with_name(data_path.name + ".meta.json")
    cmd = ["python", "scripts/pretokenize.py", "--out", str(data_path), "--tokens", str(n_tokens)]

    if data_path.exists():
        if meta_path.exists():
            print(f"existing partial file at {data_path} with metadata — will resume")
            cmd.append("--resume")
        else:
            print(f"existing file at {data_path} has no metadata sidecar; deleting and restarting")
            data_path.unlink()
    else:
        print(f"no existing pretokenized file at {data_path} — starting fresh")

    rc = run(cmd)
    if rc != 0:
        sys.exit(f"pretokenize failed with exit code {rc}")


def train(args, data_path: Path):
    """Run training. Resumes from checkpoint dir if any step_*.pt exists."""
    cmd = [
        "python", "scripts/train.py",
        "--model", args.model,
        "--n-tokens", str(args.n_tokens),
        "--batch-size", str(args.batch_size),
        "--peak-lr", str(args.peak_lr),
        "--num-workers", str(args.num_workers),
        "--data-path", str(data_path),
        "--ckpt-dir", args.ckpt_dir,
    ]
    if args.run_name:
        cmd += ["--run-name", args.run_name]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.no_compile:
        cmd.append("--no-compile")

    ckpt_dir = Path(args.ckpt_dir)
    has_checkpoints = ckpt_dir.exists() and any(ckpt_dir.glob("step_*.pt"))
    if has_checkpoints:
        print(f"existing checkpoints in {ckpt_dir} — will resume from latest")
        cmd.append("--resume")
    else:
        print(f"no existing checkpoints in {ckpt_dir} — starting fresh")

    rc = run(cmd)
    if rc != 0:
        sys.exit(f"train failed with exit code {rc}")


def main():
    args = parse_args()
    data_path = Path(args.data_path or default_data_path(args.n_tokens))

    t_start = time.time()

    if not args.skip_pretok:
        banner(f"STEP 1/2 — pretokenize → {data_path}  ({args.n_tokens / 1e9:.2f}B tokens)")
        pretokenize(data_path, args.n_tokens)
        t_pretok = time.time()
        print(f"\npretokenize done in {(t_pretok - t_start) / 60:.1f} min")
    else:
        if not data_path.exists():
            sys.exit(f"--skip-pretok set but {data_path} does not exist")
        print(f"skipping pretokenize; using existing {data_path}")
        t_pretok = t_start

    banner(f"STEP 2/2 — train  (model={args.model}, batch={args.batch_size}, lr={args.peak_lr})")
    train(args, data_path)
    t_end = time.time()

    print()
    print("=" * 70)
    print(f"  TOTAL: pretokenize {(t_pretok - t_start) / 60:.1f} min  +  "
          f"train {(t_end - t_pretok) / 60:.1f} min  =  {(t_end - t_start) / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
