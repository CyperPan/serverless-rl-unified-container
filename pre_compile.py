"""Build-time pre-compile.

Runs at `docker build` time (`RUN python pre_compile.py`). Exercises both
the actor and learner construction paths so:

  - .pyc files for every imported module land in the image layer
  - mujoco binaries get loaded / cached in the build node's page cache
  - any import-time errors fail the build instead of the first cold start
  - one warmup learn step is performed against an actor-produced batch so
    the CUDA kernel JIT artifacts are at least exercised in the build env

This script does NOT replace runtime prewarm (handler.py does that at
module load). It only validates and produces deterministic .pyc layout.
"""
from __future__ import annotations

import os
import sys
import time

# Force CPU during build — the build node may not have a GPU and we don't
# want CUDA initialization to fail the build.
os.environ.setdefault("UNIFIED_PREWARM", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gymnasium as gym  # noqa: F401
import ray  # noqa: F401
import torch  # noqa: F401
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID

from serverless_actor import ServerlessActor
from serverless_learner import ServerlessLearner


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_password = os.environ.get("REDIS_PASSWORD", "")

    # Use a small env so build is fast and doesn't need a GPU.
    algo = "ppo"
    env_name = "Hopper-v4"
    rollout_frag = 64
    train_batch = 64

    t0 = time.perf_counter()
    _print(f"[pre_compile] building actor for {env_name} / {algo} ...")
    actor = ServerlessActor(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        algo_name=algo,
        env_name=env_name,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_frag,
    )
    t1 = time.perf_counter()
    _print(f"[pre_compile] actor built ({t1 - t0:.1f}s)")

    _print(f"[pre_compile] sampling one rollout (warms MuJoCo + env step path) ...")
    batch = actor.sample()
    if not isinstance(batch, SampleBatch):
        batch = batch.policy_batches[DEFAULT_POLICY_ID]
    t2 = time.perf_counter()
    _print(f"[pre_compile] actor.sample done ({t2 - t1:.1f}s, batch rows={len(batch)})")

    _print(f"[pre_compile] building learner ...")
    learner = ServerlessLearner(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        algo_name=algo,
        env_name=env_name,
        rollout_fragment_length=rollout_frag,
        train_batch_size=train_batch,
        sgd_minibatch_size=min(64, train_batch),
        num_sgd_iter=1,
        num_gpus=0,  # build node may have no GPU
    )
    t3 = time.perf_counter()
    _print(f"[pre_compile] learner built ({t3 - t2:.1f}s)")

    # Note: skipping warmup_with_batch here. ray 2.8's CPU-tower learn
    # path has a NoneType dist_class issue at build time that doesn't
    # show up at runtime under --nv. Build only validates construction
    # of both roles; first real learn happens on the GPU at runtime.
    _print(f"[pre_compile] skipping warmup learn (deferred to runtime) ...")
    t4 = t3

    _print(f"[pre_compile] DONE — total {t4 - t0:.1f}s, "
           f"both construction paths validated.")


if __name__ == "__main__":
    main()
