#!/bin/bash
# Train the original (pre-fork) R2-Dreamer world model on Crafter.
# Crafter renders to numpy, so no xvfb / MUJOCO_GL is needed.
#
# Usage (task spooler, 1 GPU):
#   ts -G 1 bash run_crafter.sh logdir=./logdir/original_wm_crafter/01 env=crafter seed=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running on user: $(whoami)"
echo "Python: $($HOME/.local/bin/uv run python --version 2>&1)"
echo
echo "GPUs (nvidia-smi):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

$HOME/.local/bin/uv run "$SCRIPT_DIR/train.py" "$@"
