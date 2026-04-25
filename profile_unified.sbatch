#!/bin/bash
#SBATCH --job-name=unified_prof
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MUJOCO_GL=egl

eval "$(conda shell.bash hook)"
conda activate /home/pan.zhiyu/nitro_env

echo "============================================"
echo "Job:    $SLURM_JOB_ID on $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "GPU:    ${SLURM_GPUS_ON_NODE:-none}"
echo "Date:   $(date)"
echo "Python: $(which python)"
echo "============================================"

if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_FLAG="--gpu"
else
    GPU_FLAG=""
fi

cd ~/Rllib/unified_container

STATE=/tmp/unified_state_${SLURM_JOB_ID}.pkl
WEIGHTS=/tmp/unified_weights_${SLURM_JOB_ID}.pkl
BATCH=/tmp/unified_batch_${SLURM_JOB_ID}.pkl

# 1) Producer (CPU only) — single source of truth for all 4 tiers / both roles.
python -u profile_unified.py --producer \
    --state-path "$STATE" --weights-path "$WEIGHTS" --batch-path "$BATCH"

# 2) Actor: 4 tiers × 15 runs.
python -u profile_unified.py --role actor --n-runs 15 \
    --state-path "$STATE" --weights-path "$WEIGHTS" --batch-path "$BATCH" \
    --summary-out /tmp/unified_actor_${SLURM_JOB_ID}.json $GPU_FLAG

# 3) Learner: 4 tiers × 15 runs (uses the GPU).
python -u profile_unified.py --role learner --n-runs 15 \
    --state-path "$STATE" --weights-path "$WEIGHTS" --batch-path "$BATCH" \
    --summary-out /tmp/unified_learner_${SLURM_JOB_ID}.json $GPU_FLAG

# 4) Copy JSON summaries back to the project dir for parsing.
cp /tmp/unified_actor_${SLURM_JOB_ID}.json   ~/Rllib/unified_container/ 2>/dev/null || true
cp /tmp/unified_learner_${SLURM_JOB_ID}.json ~/Rllib/unified_container/ 2>/dev/null || true

echo "Done: $(date)"
