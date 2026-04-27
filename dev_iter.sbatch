#!/bin/bash
# Fast iteration loop for the apptainer build:
#   1. If /scratch/$USER/dev-sandbox doesn't exist, do the full apt+pip install
#      (~25 min, one-time).
#   2. Copy current source files from this dir into the sandbox.
#   3. Run pre_compile.py inside.
#
# After step 1 finishes, every subsequent run is only ~30 s. So we can
# iterate on serverless_learner.py / handler.py without paying for
# torch+ray reinstall each time.
#
# To force a clean rebuild:  rm -rf /scratch/$USER/dev-sandbox
#
#SBATCH --job-name=dev_iter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SANDBOX=/scratch/$USER/dev-sandbox

if [[ ! -d "$SANDBOX" ]]; then
    echo "[$(date)] sandbox missing — full build (one-time, ~25 min)"
    apptainer build --fakeroot --sandbox "$SANDBOX" docker://python:3.10-slim-bookworm 2>&1 | tail -5

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

    apptainer exec --writable --fakeroot "$SANDBOX" \
        pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
            --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

    apptainer exec --writable --fakeroot "$SANDBOX" bash -c "
cat > /tmp/req.txt <<'REQEOF'
$(cat requirements.txt)
REQEOF
pip install --no-cache-dir -r /tmp/req.txt
" 2>&1 | tail -3

    apptainer exec --writable --fakeroot "$SANDBOX" mkdir -p /opt/unified
    echo "[$(date)] sandbox build done"
else
    echo "[$(date)] reusing existing sandbox at $SANDBOX"
fi

# Copy fresh source on every iteration.
echo "[$(date)] syncing source files into /opt/unified ..."
for f in config.py serverless_actor.py serverless_learner.py handler.py pre_compile.py handler_runner.py seed_inside.py; do
    apptainer exec --writable --fakeroot "$SANDBOX" \
        cp "/opt/unified/$f" "/opt/unified/$f.prev" 2>/dev/null || true
    cat "$f" | apptainer exec --writable --fakeroot "$SANDBOX" \
        bash -c "cat > /opt/unified/$f"
done

# Run pre_compile.
echo "[$(date)] running pre_compile.py ..."
apptainer exec --writable --fakeroot "$SANDBOX" bash -c "
cd /opt/unified
python3 pre_compile.py
"

echo "[$(date)] DONE"
