#!/bin/bash
# Profile latency on plan2 (direct GPU access, no SLURM).
# Usage: GPU=0 bash run_profile_plan2.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d "$ROOT_DIR/.venv" ]; then
    echo "Error: $ROOT_DIR/.venv not found. Run venv setup first."
    exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"
mkdir -p logs

MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT=${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}
GPU=${GPU:-0}

echo "Using GPU: $GPU"
echo "Model: $MODEL"
echo "Draft: $DRAFT"

CUDA_VISIBLE_DEVICES=$GPU python profile_latency.py \
    --model-name-or-path "$MODEL" \
    --draft-name-or-path "$DRAFT" \
    --context-lengths 128 256 512 1024 2048 4096 \
    --block-sizes 1 2 4 8 16 32 64 128 256 512 \
    --num-warmup 10 \
    --num-runs 50 \
    --output "logs/profile_h200nvl_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee "logs/profile_h200nvl_$(date +%Y%m%d_%H%M%S).log"

echo "Done! Results saved to logs/"
