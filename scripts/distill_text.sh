#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running on user: $(whoami)"
echo "Python: $($HOME/.local/bin/uv run python --version 2>&1)"
echo
echo "GPUs (nvidia-smi):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  $HOME/.local/bin/uv run "$PROJECT_ROOT/distill_text.py" model.compile=True "$@"
