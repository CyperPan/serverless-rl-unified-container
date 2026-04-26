#!/bin/bash
# Cluster equivalent of local_4060ti/bench_cold_start.sh:
#   measures `apptainer run` cold start of the unified container,
#   N times, with a fresh Redis seeded once at the top.
#
#   sbatch profile_apptainer.sbatch
#
# Requires unified.sif to already exist (run build_apptainer.sbatch first).
#
#SBATCH --job-name=apptainer_prof
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -f unified.sif ]]; then
    echo "ERROR: unified.sif missing. Run build_apptainer.sbatch first." >&2
    exit 1
fi

# Activate conda for redis + the seed script (uses host Python).
eval "$(conda shell.bash hook)"
conda activate /home/pan.zhiyu/nitro_env

# ── 1. Bring up Redis on this compute node ────────────────────────────
REDIS_HOST=$(hostname)
REDIS_PORT=6380
echo "[$(date)] starting redis on ${REDIS_HOST}:${REDIS_PORT}"
redis-server --daemonize yes --port "$REDIS_PORT" --protected-mode no \
    --save '' --appendonly no \
    --logfile /tmp/redis_${SLURM_JOB_ID}.log
sleep 2
redis-cli -p "$REDIS_PORT" ping || { echo "redis failed"; exit 1; }

# ── 2. Seed initial weights / learner_state / sample_batch ────────────
echo "[$(date)] seeding redis"
REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" \
    python local_4060ti/seed_redis.py

# ── 3. N apptainer cold starts ────────────────────────────────────────
N="${N_RUNS:-10}"
RESULTS="results_apptainer_${SLURM_JOB_ID}.csv"
echo "iter,role,client_wall_s,server_duration_s,t0_total_s" > "$RESULTS"

for i in $(seq 1 "$N"); do
    role="actor"
    if (( i % 2 == 0 )); then role="learner"; fi

    t_a=$(date +%s.%N)
    # `apptainer run --nv` = full container start. Output is the JSON the
    # %runscript prints; parse `lambda_duration` and `t0_phase.total`.
    out=$(apptainer run --nv --cleanenv \
            --env "REDIS_HOST=$REDIS_HOST" \
            --env "REDIS_PORT=$REDIS_PORT" \
            --env "ROLE=$role" \
            --env "BATCH_ID=seed" \
            --env "ALGO_NAME=ppo" \
            --env "ENV_NAME=Hopper-v4" \
            --env "ROLLOUT_FRAGMENT_LENGTH=512" \
            --env "TRAIN_BATCH_SIZE=4096" \
            --env "NUM_GPUS_LEARNER=1" \
            --env "UNIFIED_PREWARM=1" \
            --env "UNIFIED_WARMUP_LEARN=1" \
            unified.sif 2>&1 || true)
    t_b=$(date +%s.%N)

    wall=$(echo "$t_b - $t_a" | bc -l)
    server=$(echo "$out" | grep -oE '"lambda_duration": [0-9.]+' | head -1 \
                       | awk '{print $2}' || echo 0)
    t0=$(echo "$out" | grep -oE '"total": [0-9.]+' | head -1 \
                     | awk '{print $2}' || echo 0)

    printf "%d,%s,%.4f,%.4f,%.4f\n" "$i" "$role" "$wall" "${server:-0}" "${t0:-0}" \
        | tee -a "$RESULTS"
done

# ── 4. Tear down redis ────────────────────────────────────────────────
redis-cli -p "$REDIS_PORT" shutdown nosave || true

echo "[$(date)] DONE — CSV at $(pwd)/$RESULTS"
echo "summary:"
python - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$RESULTS")))
for col in ("client_wall_s","server_duration_s","t0_total_s"):
    vals = [float(r[col]) for r in rows]
    print(f"  {col}: mean {statistics.mean(vals):.3f}s, "
          f"std {statistics.stdev(vals) if len(vals)>1 else 0:.3f}s, "
          f"min {min(vals):.3f}s, max {max(vals):.3f}s")
EOF
