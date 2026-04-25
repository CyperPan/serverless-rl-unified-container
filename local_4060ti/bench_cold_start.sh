#!/usr/bin/env bash
# bench_cold_start.sh — measure end-to-end cold start of the unified container.
#
# Each iteration:
#   1. docker rm any lingering container
#   2. docker run -d unified         (image already pulled / built)
#   3. wait until /2015-03-31/functions/function/invocations responds
#   4. invoke once (this is the cold call — pays imports + prewarm)
#   5. invoke a second time (warm baseline for comparison)
#   6. docker stop + rm
#
# Reports per-iteration: container_start_s, first_invoke_s, second_invoke_s.
# This is what the cluster's subprocess-isolated profiling COULD NOT measure.
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-10}"
RESULTS="${RESULT_DIR:-./results}"
mkdir -p "$RESULTS"
OUT="$RESULTS/cold_start_$(date +%Y%m%d_%H%M%S).csv"
echo "iter,container_start_s,first_invoke_s,second_invoke_s,first_role" > "$OUT"

# Make sure redis is up; build image if missing.
docker compose up -d redis >/dev/null
docker compose build unified

# Confirm seeds exist; if not, run seed_redis.py once.
if ! docker compose exec -T redis redis-cli exists model_weights | grep -q 1 ; then
    echo "[bench] seeding redis (one-time) ..."
    REDIS_HOST=localhost REDIS_PORT=6379 python3 seed_redis.py
fi

for i in $(seq 1 "$N"); do
    role="actor"
    if (( i % 2 == 0 )); then
        role="learner"
    fi

    echo "[iter $i/$N] role=$role"
    docker rm -f unified_handler 2>/dev/null || true

    t0=$(date +%s.%N)
    docker compose up -d unified >/dev/null

    # Block until the RIE accepts a connection.
    while ! curl -s -o /dev/null -w "%{http_code}" \
            -X POST -d '{}' \
            http://localhost:9000/2015-03-31/functions/function/invocations \
            --max-time 1 | grep -qE '^[245]' ; do
        sleep 0.2
    done
    t_ready=$(date +%s.%N)

    # First invocation — pays the cold module-load + prewarm.
    t_first_a=$(date +%s.%N)
    if [[ "$role" == "actor" ]]; then
        python3 invoke.py actor >/dev/null
    else
        python3 invoke.py learner --batch-id seed >/dev/null
    fi
    t_first_b=$(date +%s.%N)

    # Second invocation — warm baseline.
    t_second_a=$(date +%s.%N)
    if [[ "$role" == "actor" ]]; then
        python3 invoke.py actor >/dev/null
    else
        python3 invoke.py learner --batch-id seed >/dev/null
    fi
    t_second_b=$(date +%s.%N)

    cstart=$(echo "$t_ready - $t0" | bc -l)
    first=$(echo "$t_first_b - $t_first_a" | bc -l)
    second=$(echo "$t_second_b - $t_second_a" | bc -l)
    printf "%d,%.4f,%.4f,%.4f,%s\n" "$i" "$cstart" "$first" "$second" "$role" | tee -a "$OUT"

    docker compose stop unified >/dev/null
done

echo
echo "[done] CSV at $OUT"
echo "summary:"
python3 - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$OUT")))
for col in ("container_start_s","first_invoke_s","second_invoke_s"):
    vals = [float(r[col]) for r in rows]
    print(f"  {col}: mean {statistics.mean(vals):.3f}s, std {statistics.stdev(vals) if len(vals)>1 else 0:.3f}s, min {min(vals):.3f}s, max {max(vals):.3f}s")
EOF
