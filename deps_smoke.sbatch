#!/bin/bash
# Fast smoke test: do the pip install + minimum imports inside a fresh
# python:3.10-slim apptainer sandbox. Skips pre_compile.py. 5-10 min vs
# the 30-50 min full build, so we can iterate on requirements.txt
# without burning a GPU node each time.
#
#   sbatch deps_smoke.sbatch
#
#SBATCH --job-name=deps_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SANDBOX=/scratch/$USER/deps-smoke-sandbox
rm -rf "$SANDBOX"

echo "[$(date)] building sandbox from python:3.10-slim ..."
apptainer build --fakeroot --sandbox "$SANDBOX" docker://python:3.10-slim-bookworm 2>&1 | tail -5

echo "[$(date)] apt install ..."
apptainer exec --writable --fakeroot "$SANDBOX" bash -c '
set -eux
apt-get update
apt-get install -y --no-install-recommends \
    gcc g++ patchelf \
    libosmesa6-dev libgl1-mesa-dev \
    libglew-dev libglfw3-dev \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    ca-certificates curl
rm -rf /var/lib/apt/lists/*
' 2>&1 | tail -5

echo "[$(date)] pip install torch ..."
apptainer exec --writable --fakeroot "$SANDBOX" \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo "[$(date)] pip install requirements ..."
apptainer exec --writable --fakeroot "$SANDBOX" bash -c "
cat > /tmp/req.txt <<'EOF'
$(cat requirements.txt)
EOF
pip install --no-cache-dir -r /tmp/req.txt
" 2>&1 | tail -8

echo "[$(date)] smoke test imports ..."
apptainer exec --writable --fakeroot "$SANDBOX" python -c "
import gymnasium as gym
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
print('gymnasium', gym.__version__)
print('torch', torch.__version__)
print('ray', ray.__version__)
import mujoco
print('mujoco', mujoco.__version__)
env = gym.make('Hopper-v4')
print('Hopper-v4 obs:', env.observation_space)
print('Hopper-v4 act:', env.action_space)
print('SMOKE PASS')
" 2>&1 | tail -15

# Don't keep the sandbox; clean up scratch.
rm -rf "$SANDBOX"
echo "[$(date)] DONE"
