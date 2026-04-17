#!/bin/bash
set -euo pipefail

echo "Running on user: $(whoami)"
echo "Python: $($HOME/.local/bin/uv run python --version 2>&1)"
echo
echo "GPUs (nvidia-smi):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ts -G 1 bash random_goal.sh logdir=./goal_dreamer_with_text/01 seed=0 +mission_text=False
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  $HOME/.local/bin/uv run train.py "$@"
