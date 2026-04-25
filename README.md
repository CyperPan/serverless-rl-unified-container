# Unified actor + learner container

One Docker image that handles **both** PPO actor and learner roles for
serverless RL training. The same warm container can be invoked as either
role via `event['role']`; switching between roles costs ~10 ms instead
of paying another container cold start.

This is the "T0.5" design referenced in
[sim-nitro](https://github.com/CyperPan/Nitro): if a learner Lambda dies,
any warm actor in the pool can take over immediately, and vice versa.

## Why one image instead of two

Cluster measurements (V100, Hopper-v4 PPO, N=15 runs/tier) showed that
the cold-start cost of either role is dominated by the **shared** import
graph (torch + ray + rllib + gymnasium + mujoco ≈ 25 s) plus CUDA init
(≈ 11 s). Once those are paid, the role-specific build is small:

| Phase | Cost (V100) | Whose? |
|-------|-------------|--------|
| Imports + CUDA init | ≈ 25 s | shared |
| Actor `RolloutWorker` build | 8.6 s | actor |
| Learner `Policy` + Adam build | 11.0 s | learner |
| Warmup learn (CUDA JIT) | 3.6 s | learner-only, but exercised once at boot |

If the actor and learner each ship as their own image, every cold start
re-pays the 25 s of shared imports. A single image with module-level
prewarm of both roles pays ≈ 50 s once and lets all subsequent
invocations land at warm-pool latency (actor 6.8 s sample / learner
0.17 s learn).

Full per-tier numbers from the cluster benchmark: see
`unified_prof_6335834.out` and the two JSON files.

## Quick start

```bash
# Build the image. Drops a symlink to mujoco210 if missing.
./build.sh

# On a 4060 Ti workstation (or any nvidia-docker host):
cd local_4060ti
cp .env.example .env
./setup.sh                 # build, start redis, seed initial weights
N=10 ./bench_cold_start.sh   # measure docker run → first invoke
N=30 ./bench_role_switch.sh  # measure warm-pool actor↔learner switching
```

For Lambda deployment, the image is RIE-compatible — push it to ECR and
point a Lambda function at it. Set `event['role']` per invocation.

## Files

| | |
|---|---|
| `Dockerfile` | `public.ecr.aws/lambda/python:3.10` base + union deps + COPY src + `RUN pre_compile.py` |
| `requirements.txt` | torch + gymnasium[atari,mujoco] + ray[rllib] + mujoco + redis |
| `config.py` | env / algo metadata (verbatim port from sim-nitro) |
| `serverless_actor.py` | RolloutWorker + Redis I/O |
| `serverless_learner.py` | Policy + Adam + Redis I/O, mirrors actor shape |
| `handler.py` | Lambda entrypoint, dispatches by `event['role']`. Module-level prewarm of both roles. |
| `pre_compile.py` | Build-time validation of both construction paths. |
| `local_test.py` | Host-Python exerciser: actor → learner → actor → learner. |
| `build.sh` | Local docker build helper. |
| `local_4060ti/` | Workstation test bench (docker compose + cold-start + role-switch benchmarks). |
| `profile_unified.py` | Cluster profiler — 4 tiers × N rounds, both roles. |
| `profile_unified.sbatch` | Slurm submitter for the profiler. |
| `unified_prof_6335834.out` | Reference cluster run (V100, Hopper-v4 PPO, 15 runs/tier). |
| `unified_actor_6335834.json` | Per-run + per-phase JSON for actor side. |
| `unified_learner_6335834.json` | Same for learner. |

## Lambda invocation contract

```json
{
  "role": "actor" | "learner",
  "redis_host": "...",
  "redis_port": 6379,
  "redis_password": "...",
  "algo_name": "ppo",
  "env_name": "Hopper-v4",
  "rollout_fragment_length": 512,
  "batch_id": "<uuid for learner>"
}
```

Returns:
```json
{
  "aws_request_id": "...",
  "role": "actor",
  "lambda_duration": 6.78,
  "phase_timings": { "ensure_role": 0.0, "redis_get_weights": 0.001, "set_weights": 0.011, "sample": 6.77, "redis_push_batch": 0.012 },
  "t0_phase": { "imports_done": 25.4, "actor_build": 8.6, "learner_build": 11.0, "warmup_exec": 3.6, "total": 48.6 },
  "prewarm": true
}
```

`t0_phase` is the same on every response (it's a module-level constant
captured at container boot). `phase_timings` is per-invocation.

## Env vars (override at deploy)

| Var | Default | Effect |
|---|---|---|
| `UNIFIED_PREWARM` | `1` | Build BOTH actor + learner at module load. `0` for lazy. |
| `UNIFIED_WARMUP_LEARN` | `1` | Run one learn step at module load to amortize CUDA JIT. |
| `ALGO_NAME` | `ppo` | One of ppo / appo / impala. |
| `ENV_NAME` | `Hopper-v4` | gym env id. |
| `ROLLOUT_FRAGMENT_LENGTH` | from `config.py` | Per-rollout sample count. |
| `TRAIN_BATCH_SIZE` | `8 × ROLLOUT` | Learner training batch. |
| `NUM_GPUS_LEARNER` | `1` | GPUs visible to learner. Actor sees 0. |
| `REDIS_HOST/PORT/PASSWORD` | localhost / 6379 / "" | State store. |

## Acknowledgements

Actor pattern is verbatim from
[sim-nitro/aws_lambda](https://github.com/CyperPan/Nitro). Learner
pattern mirrors the actor shape so the unified handler can dispatch
between them. MuJoCo binaries are not redistributed — `build.sh`
symlinks `mujoco210/` from your local copy if missing.

## License

[ASL 2.0]
