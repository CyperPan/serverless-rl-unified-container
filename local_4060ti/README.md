# Local 4060 Ti test bench

Local test rig for the unified actor+learner container, targeted at an
RTX 4060 Ti workstation. Measures things the cluster runs CAN'T:

1. **True Docker cold start** — `docker run --rm` to first handler
   response, including image read + RIE startup + Python module load +
   prewarm.
2. **Warm role-switch latency** — keep one container alive, alternate
   `actor → learner → actor` invocations, time each.

Both numbers are central to the unified-container thesis (T0.5 design):
"the second invocation in any role costs T3, not T1".

## Prerequisites

| Host | What you need |
|------|----------------|
| **Linux + 4060 Ti** | Docker Engine ≥ 24, `nvidia-container-toolkit` (`apt install nvidia-container-toolkit`), `bc`, Python 3.10+ |
| **Windows + 4060 Ti** | WSL2 (Ubuntu), Docker Desktop with WSL2 backend, NVIDIA driver ≥ 525 with WSL2 GPU support enabled in Windows |
| **macOS** | No GPU passthrough — set `NUM_GPUS_LEARNER=0` in `.env`. Only CPU benchmark; useful for validating the pipeline but learner numbers won't reflect GPU |

The host also needs the same Python deps as `serverless_actor.py` to
seed Redis from outside Docker (one-time):
```bash
pip install gymnasium torch redis 'ray[rllib]==2.8.0' mujoco
```
(or use a conda env that has them — anything compatible with the cluster's `nitro_env` works)

## Setup

```bash
cp .env.example .env
./setup.sh        # checks toolchain, links mujoco210, builds image, seeds redis
```

If anything fails, the script prints which step. Re-run is idempotent.

## Benchmarks

### 1. Cold start (Docker provisioning + prewarm)

```bash
N=10 ./bench_cold_start.sh
```

Iterates N times: `docker run` → wait for handler → first invoke (cold) →
second invoke (warm) → `docker stop`. Outputs CSV with three columns:

| Column | Meaning | Expected on 4060 Ti |
|--------|---------|---------------------|
| `container_start_s` | `docker run` to handler ready | 30–60 s (most is module import + prewarm) |
| `first_invoke_s` | first POST to RIE → response | 0.05–0.5 s (handler is already warm) |
| `second_invoke_s` | second POST | same as first (T3 latency) |

Compare `container_start_s` here against the cluster's "T0 wall ~40 s"
to see how much of that comes from Docker's container provisioning vs
imports.

### 2. Warm pool role switch

```bash
N=30 ./bench_role_switch.sh
```

Keeps one container alive, sends 30 alternating `actor / learner`
invocations. Outputs CSV with `iter,role,wall_s` per line.

| Column | Expected | Why |
|--------|----------|-----|
| `actor` rows | ~6.8 s each | sample-bound (4096 MuJoCo steps); not affected by warm cache |
| `learner` rows | ~0.17 s each | GPU MLP step, dominated by Adam |

If the unified-container hypothesis holds, `actor`'s std deviation is
small (the role switch itself adds <10 ms) — the only variance is
MuJoCo timing and Redis fetch.

### 3. One-off interactive invoke

```bash
python invoke.py actor                  # one actor call
python invoke.py learner --batch-id seed
python invoke.py sequence               # actor → learner → actor → learner
```

Prints per-phase timings the handler returns.

## Tuning for 4060 Ti

The defaults in `docker-compose.yml` use `Hopper-v4` PPO with
`train_batch_size=4096`. RSS peak ≈ 1.5 GB. Both 8 GB and 16 GB cards
fit. To experiment with bigger workloads, edit `docker-compose.yml`
environment vars:

| Var | Comment |
|-----|---------|
| `ENV_NAME` | `Humanoid-v4` doubles RSS (~3 GB GPU). Still fits 16 GB. |
| `TRAIN_BATCH_SIZE` | Halve for tighter VRAM headroom on 8 GB. |
| `NUM_GPUS_LEARNER` | `0` for CPU-only host. Learner becomes ~50× slower. |

For ViT-class encoders (303 M+), 4060 Ti can't run them — use a cluster
A100. The relevant cluster JSON is in
`unified_container/unified_actor_6335834.json` /
`unified_learner_6335834.json` for cross-reference.

## What's in this folder

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Two services: `redis` (state store) + `unified` (handler image with RIE). |
| `.env.example` | Copy to `.env` to override `NUM_GPUS_LEARNER`, prewarm flags, etc. |
| `setup.sh` | One-shot bringup — verify toolchain, build image, seed redis. |
| `seed_redis.py` | CPU-only producer that pushes initial `model_weights`, `learner_state`, and a seed `sample_batch` into Redis. |
| `invoke.py` | HTTP client for the Lambda RIE (`POST /2015-03-31/functions/function/invocations`). |
| `bench_cold_start.sh` | Measures `docker run` → first invoke latency. |
| `bench_role_switch.sh` | Measures warm-pool actor↔learner switching. |
| `results/` | CSVs land here. |

## Cross-reference to cluster results

The cluster's `unified_prof_6335834.out` measured:

| Phase | Cluster (V100) | Expected on 4060 Ti |
|-------|----------------|---------------------|
| Total module load + prewarm | ~50 s | **30–40 s** (faster CUDA init, faster NVMe) |
| Actor T3 (sample) | 6.77 s | similar (MuJoCo CPU-bound) |
| Learner T3 (learn) | 0.17 s | **0.06–0.12 s** (Ada faster than Volta on small MLP) |

The cluster could not measure `container_start_s` because it ran via
sbatch + subprocess, not Docker. That's the gap this folder fills.
