#!/bin/bash
# Build the unified container as an apptainer SIF on the cluster.
#
#   sbatch build_apptainer.sbatch
#
# Result: ~/Rllib/unified_container/unified.sif (~ 5-7 GB after deps).
#
# pre_compile.py does a CPU-only construction of both roles, so this
# can run on any node with apptainer + enough scratch space.
#
#SBATCH --job-name=apptainer_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "$(dirname "$0")"

# Apptainer needs scratch for build cache and intermediate layers.
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "[$(date)] starting apptainer build on $(hostname)"
echo "  cachedir = $APPTAINER_CACHEDIR"
echo "  tmpdir   = $APPTAINER_TMPDIR"
echo "  cwd      = $(pwd)"

# --fakeroot uses /etc/subuid mapping. Most HPC sites enable it for
# /home; if it fails, fall back to a sandbox in /scratch.
if apptainer build --fakeroot --force unified.sif unified.def 2>&1 | tee build.log ; then
    echo "[$(date)] fakeroot build succeeded"
else
    echo "[$(date)] fakeroot failed, trying sandbox in /scratch"
    SANDBOX=/scratch/$USER/unified-sandbox
    rm -rf "$SANDBOX"
    apptainer build --sandbox "$SANDBOX" unified.def 2>&1 | tee build.log
    apptainer build --force unified.sif "$SANDBOX"
fi

ls -lah unified.sif
echo "[$(date)] DONE"
