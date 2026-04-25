#!/usr/bin/env bash
# Verify the local toolchain and build the image. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

echo "[setup] checking docker ..."
docker --version
docker compose version

echo "[setup] checking nvidia GPU passthrough ..."
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>/dev/null | grep -q "4060 Ti" ; then
    echo "[setup] GPU passthrough OK (4060 Ti detected)"
elif docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>/dev/null | grep -q "GeForce" ; then
    echo "[setup] GPU passthrough OK (consumer NVIDIA card detected)"
elif docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 ; then
    echo "[setup] GPU passthrough OK (NVIDIA card detected)"
else
    echo "[setup] WARNING: no GPU passthrough — falling back to CPU"
    echo "[setup]   set NUM_GPUS_LEARNER=0 in .env"
fi

# Ensure mujoco210 is reachable for the Dockerfile.
if [[ ! -e ../mujoco210 ]]; then
    echo "[setup] symlinking ../mujoco210 from sim-nitro/aws_lambda/"
    if [[ -d ../../../fault_Nitro/sim-nitro/aws_lambda/mujoco210 ]]; then
        ln -sf ../../../fault_Nitro/sim-nitro/aws_lambda/mujoco210 ../mujoco210
    else
        echo "[setup] ERROR: mujoco210 not found." >&2
        exit 1
    fi
fi

echo "[setup] building image ..."
docker compose build unified

echo "[setup] starting redis ..."
docker compose up -d redis

echo "[setup] seeding redis ..."
REDIS_HOST=localhost REDIS_PORT=6379 python3 seed_redis.py

echo "[setup] DONE"
echo "  Run cold-start benchmark:    ./bench_cold_start.sh"
echo "  Run warm role-switch:        ./bench_role_switch.sh"
echo "  One-off invocation:          python invoke.py sequence"
