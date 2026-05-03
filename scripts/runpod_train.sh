#!/usr/bin/env bash
#
# Wrapper for a full pretokenize + train run on RunPod.
#
# Defaults to GPT-2 medium at Chinchilla-optimal (7B tokens). Override via env:
#     MODEL=small N_TOKENS=2_500_000_000 BATCH_SIZE=32 PEAK_LR=3e-4 bash scripts/runpod_train.sh
#
# Data + checkpoints land on /workspace (network volume) so they survive pod
# destruction. If /workspace is missing (no volume mounted) it falls back to
# the container disk — but you'll likely run out of room quickly.

set -e

MODEL="${MODEL:-medium}"
N_TOKENS="${N_TOKENS:-7000000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PEAK_LR="${PEAK_LR:-2.5e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_NAME="${RUN_NAME:-gpt2-${MODEL}-chinchilla}"

# /workspace if mounted, container disk otherwise
if [ -d /workspace ]; then
    DATA_DIR="/workspace/data"
    CKPT_DIR="/workspace/checkpoints/${MODEL}"
else
    echo "warning: /workspace not mounted — using container disk for data + ckpts"
    DATA_DIR="data"
    CKPT_DIR="checkpoints/${MODEL}"
fi
mkdir -p "$DATA_DIR" "$CKPT_DIR"

# pretokenized .bin path: data/fineweb_edu_<N>B.bin
TOKENS_BILLIONS=$(( N_TOKENS / 1000000000 ))
DATA_PATH="${DATA_DIR}/fineweb_edu_${TOKENS_BILLIONS}B.bin"

# warn if auth missing (training won't crash, but you'll lose visibility / hit rate limits)
[ -z "${HF_TOKEN:-}" ] && echo "warning: HF_TOKEN not set — streaming may rate-limit"
[ -z "${WANDB_API_KEY:-}" ] && echo "warning: WANDB_API_KEY not set — wandb will prompt or run offline"

# expandable_segments helps fit larger batches by reducing fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================"
echo "  configuration"
echo "================================================================"
echo "  MODEL=$MODEL"
echo "  N_TOKENS=$N_TOKENS  ($(echo "scale=2; $N_TOKENS / 1000000000" | bc)B)"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  PEAK_LR=$PEAK_LR"
echo "  RUN_NAME=$RUN_NAME"
echo "  DATA_PATH=$DATA_PATH"
echo "  CKPT_DIR=$CKPT_DIR"
echo "================================================================"
echo

python scripts/full_train.py \
    --model "$MODEL" \
    --n-tokens "$N_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --peak-lr "$PEAK_LR" \
    --num-workers "$NUM_WORKERS" \
    --data-path "$DATA_PATH" \
    --ckpt-dir "$CKPT_DIR" \
    --run-name "$RUN_NAME"
