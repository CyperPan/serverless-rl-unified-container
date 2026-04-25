#!/usr/bin/env bash
# Build the unified actor+learner image locally.
#
# Requires: docker, mujoco210/ next to the Dockerfile (binary release from
# https://mujoco.org/download — the same one sim-nitro/aws_lambda uses).
set -euo pipefail

IMAGE="${IMAGE:-serverless-rl-unified}"
TAG="${TAG:-latest}"

cd "$(dirname "$0")"

if [[ ! -d mujoco210 ]]; then
    echo "[build.sh] symlinking mujoco210 from sim-nitro/aws_lambda/" >&2
    if [[ -d ../../fault_Nitro/sim-nitro/aws_lambda/mujoco210 ]]; then
        ln -sf ../../fault_Nitro/sim-nitro/aws_lambda/mujoco210 mujoco210
    else
        echo "ERROR: mujoco210/ not found. Place it next to Dockerfile." >&2
        exit 1
    fi
fi

echo "[build.sh] docker build -t ${IMAGE}:${TAG} ."
docker build --platform linux/amd64 -t "${IMAGE}:${TAG}" .
echo "[build.sh] built ${IMAGE}:${TAG}"
