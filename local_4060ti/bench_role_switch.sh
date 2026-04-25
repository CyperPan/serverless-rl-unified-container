#!/usr/bin/env bash
# bench_role_switch.sh — keep one warm container, repeatedly switch roles.
#
# Demonstrates the T0.5 hypothesis: after the unified container's first
# cold start, role switches cost only the per-invocation Redis I/O +
# sample/learn execution. There is no rebuild.
#
# Sequence run inside the same container instance:
#   actor → learner → actor → learner → ... (N total)
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-30}"
RESULTS="${RESULT_DIR:-./results}"
mkdir -p "$RESULTS"
OUT="$RESULTS/role_switch_$(date +%Y%m%d_%H%M%S).csv"

docker compose up -d redis >/dev/null
docker compose build unified

if ! docker compose exec -T redis redis-cli exists model_weights | grep -q 1 ; then
    REDIS_HOST=localhost REDIS_PORT=6379 python3 seed_redis.py
fi

# Start one container (cold start).
docker rm -f unified_handler 2>/dev/null || true
t0=$(date +%s.%N)
docker compose up -d unified >/dev/null

while ! curl -s -o /dev/null -X POST -d '{}' \
        http://localhost:9000/2015-03-31/functions/function/invocations \
        --max-time 1 ; do
    sleep 0.2
done
t_ready=$(date +%s.%N)
echo "[bench] container ready in $(echo "$t_ready - $t0" | bc -l)s"

# Pay the first (cold) invocation outside the loop so the loop measures pure warm.
echo "[bench] cold-pay invocation (actor) ..."
python3 invoke.py actor >/dev/null

echo "iter,role,wall_s" > "$OUT"
for i in $(seq 1 "$N"); do
    role="actor"
    if (( i % 2 == 0 )); then
        role="learner"
    fi
    t_a=$(date +%s.%N)
    python3 invoke.py "$role" --batch-id seed >/dev/null
    t_b=$(date +%s.%N)
    wall=$(echo "$t_b - $t_a" | bc -l)
    printf "%d,%s,%.4f\n" "$i" "$role" "$wall" | tee -a "$OUT"
done

docker compose stop unified >/dev/null

echo
python3 - <<EOF
import csv, statistics
rows = list(csv.DictReader(open("$OUT")))
for role in ("actor","learner"):
    vals = [float(r["wall_s"]) for r in rows if r["role"]==role]
    if vals:
        print(f"  {role:7}: N={len(vals)}  mean {statistics.mean(vals):.3f}s  "
              f"std {statistics.stdev(vals) if len(vals)>1 else 0:.3f}s  "
              f"min {min(vals):.3f}s  max {max(vals):.3f}s")
EOF
echo "[done] $OUT"
