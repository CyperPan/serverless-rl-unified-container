#!/bin/bash
# Convert /scratch/$USER/dev-sandbox into unified.sif. Fast (~1-2 min)
# vs the 50-min full apptainer build because deps are already baked.
#
#SBATCH --job-name=sb2sif
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp

SANDBOX=/scratch/$USER/dev-sandbox
SIF=unified.sif

if [[ ! -d "$SANDBOX" ]]; then
    echo "ERROR: sandbox missing at $SANDBOX. Run dev_iter.sbatch first." >&2
    exit 1
fi

echo "[$(date)] converting sandbox → $SIF ..."
apptainer build --force "$SIF" "$SANDBOX"
ls -lh "$SIF"
echo "[$(date)] DONE"
