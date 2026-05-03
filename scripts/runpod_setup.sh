#!/usr/bin/env bash
#
# One-time setup for a fresh RunPod instance.
#
# Run this once after cloning the repo:
#     git clone https://github.com/billenert/gpt2-from-scratch.git
#     cd gpt2-from-scratch
#     bash scripts/runpod_setup.sh
#
# Then set your auth tokens and launch training (see the printed instructions
# at the end of this script).

set -e

echo "==> installing system packages"
apt-get update -qq
apt-get install -y -qq tmux htop

echo "==> installing python package (editable, with dev extras)"
pip install -e ".[dev]" -q

echo "==> ensuring /workspace dirs exist"
mkdir -p /workspace/data /workspace/checkpoints

echo "==> verifying CUDA"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"cuda available: {ok}")
if ok:
    print(f"device:         {torch.cuda.get_device_name(0)}")
    print(f"capability:     {torch.cuda.get_device_capability(0)}")
    print(f"memory:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise SystemExit("CUDA not available — pod is misconfigured")
PY

echo "==> checking disk"
echo "container / disk:"
df -h / | tail -1
echo "/workspace network volume:"
df -h /workspace 2>/dev/null | tail -1 || echo "(no /workspace mounted — checkpoints will go on container disk)"

echo
echo "================================================================"
echo "  setup complete"
echo "================================================================"
echo
echo "next steps:"
echo
echo "1) Set auth tokens (add to ~/.bashrc to persist across sessions):"
echo "     export HF_TOKEN=hf_..."
echo "     export WANDB_API_KEY=..."
echo "     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo
echo "2) Start a tmux session so the run survives SSH disconnect:"
echo "     tmux new -s train"
echo
echo "3) Kick off the full pipeline (pretokenize + train):"
echo "     bash scripts/runpod_train.sh"
echo
echo "4) Detach with Ctrl-B then d. Reattach later with:"
echo "     tmux a -t train"
