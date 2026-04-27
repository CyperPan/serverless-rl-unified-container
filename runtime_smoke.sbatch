#!/bin/bash
# Smoke-test the unified.sif at runtime: bring up redis, seed, then
# apptainer run --nv as actor and as learner. Validates that the
# Lambda RIE-equivalent %runscript works end-to-end with GPU.
#
#SBATCH --job-name=rt_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Skip SIF (slow squashfs compression) — apptainer can run the sandbox
# directory directly. SIF is for distribution; for testing the sandbox is faster.
TARGET=/scratch/$USER/dev-sandbox
if [[ ! -d "$TARGET" ]]; then
    echo "ERROR: sandbox missing at $TARGET. Run dev_iter.sbatch first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate /home/pan.zhiyu/nitro_env

REDIS_HOST=$(hostname)
REDIS_PORT=6380

echo "[$(date)] starting redis on ${REDIS_HOST}:${REDIS_PORT}"
redis-server --daemonize yes --port "$REDIS_PORT" --protected-mode no \
    --save '' --appendonly no \
    --logfile /tmp/redis_${SLURM_JOB_ID}.log
sleep 2
redis-cli -p "$REDIS_PORT" ping

echo "[$(date)] seeding redis (host CPU) ..."
REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" \
    python local_4060ti/seed_redis.py 2>&1 | tail -10

echo
echo "===== T0 (cold) actor ====="
t0=$(date +%s.%N)
apptainer exec --nv --cleanenv \
    --env "REDIS_HOST=$REDIS_HOST" \
    --env "REDIS_PORT=$REDIS_PORT" \
    --env "ROLE=actor" \
    --env "ALGO_NAME=ppo" \
    --env "ENV_NAME=Hopper-v4" \
    --env "ROLLOUT_FRAGMENT_LENGTH=512" \
    --env "TRAIN_BATCH_SIZE=4096" \
    --env "NUM_GPUS_LEARNER=1" \
    --env "UNIFIED_PREWARM=1" \
    --env "UNIFIED_WARMUP_LEARN=0" \
    "$TARGET" python /opt/unified/handler_runner.py 2>&1 | tail -30
t1=$(date +%s.%N)
echo "[t0_actor wall: $(echo "$t1 - $t0" | bc)s]"

echo
echo "===== T0 (cold) learner ====="
t0=$(date +%s.%N)
apptainer exec --nv --cleanenv \
    --env "REDIS_HOST=$REDIS_HOST" \
    --env "REDIS_PORT=$REDIS_PORT" \
    --env "ROLE=learner" \
    --env "BATCH_ID=seed" \
    --env "ALGO_NAME=ppo" \
    --env "ENV_NAME=Hopper-v4" \
    --env "ROLLOUT_FRAGMENT_LENGTH=512" \
    --env "TRAIN_BATCH_SIZE=4096" \
    --env "NUM_GPUS_LEARNER=1" \
    --env "UNIFIED_PREWARM=1" \
    --env "UNIFIED_WARMUP_LEARN=0" \
    "$TARGET" python /opt/unified/handler_runner.py 2>&1 | tail -30
t1=$(date +%s.%N)
echo "[t0_learner wall: $(echo "$t1 - $t0" | bc)s]"

redis-cli -p "$REDIS_PORT" shutdown nosave || true
echo "[$(date)] DONE"
