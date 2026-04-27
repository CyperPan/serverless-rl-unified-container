"""Seed Redis from INSIDE the unified container.

Mirrors local_4060ti/seed_redis.py but assumes:
  - Source code lives at /opt/unified (apptainer convention)
  - Runs via `apptainer exec ... seed_inside.py`
  - Therefore uses the container's API stack — important because ray 2.8
    + gymnasium 0.28.1 in the container produces RLModule-shaped state_dict,
    while host nitro_env (gymnasium 1.2.3) produces legacy shape. Mismatched
    keys cause `set_weights` to fail at runtime.

Reads config from env vars (REDIS_HOST/PORT, ALGO_NAME, ENV_NAME,
ROLLOUT_FRAGMENT_LENGTH, TRAIN_BATCH_SIZE).
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import redis as _redis  # noqa: E402

from serverless_actor import ServerlessActor  # noqa: E402
from serverless_learner import ServerlessLearner  # noqa: E402
from ray.rllib.policy.sample_batch import (  # noqa: E402
    SampleBatch, DEFAULT_POLICY_ID, concat_samples,
)


def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_pass = os.environ.get("REDIS_PASSWORD", "")
    algo_name = os.environ.get("ALGO_NAME", "ppo")
    env_name = os.environ.get("ENV_NAME", "Hopper-v4")
    rollout_frag = int(os.environ.get("ROLLOUT_FRAGMENT_LENGTH", "512"))
    train_batch = int(os.environ.get("TRAIN_BATCH_SIZE", "4096"))

    print(f"[seed-in] redis://{redis_host}:{redis_port}  env={env_name}")
    r = _redis.Redis(host=redis_host, port=redis_port,
                     password=redis_pass or None)
    r.ping()

    print("[seed-in] building actor (container's API stack) ...")
    actor = ServerlessActor(
        redis_host=redis_host, redis_port=redis_port,
        redis_password=redis_pass,
        algo_name=algo_name, env_name=env_name,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_frag,
    )

    print("[seed-in] sampling batch ...")
    batch = actor.worker.sample()
    if not isinstance(batch, SampleBatch):
        batch = batch.policy_batches[DEFAULT_POLICY_ID]
    while len(batch) < train_batch:
        nb = actor.worker.sample()
        if not isinstance(nb, SampleBatch):
            nb = nb.policy_batches[DEFAULT_POLICY_ID]
        batch = concat_samples([batch, nb])
    batch = batch[:train_batch]

    print("[seed-in] building learner (CPU, num_gpus=0) ...")
    learner = ServerlessLearner(
        redis_host=redis_host, redis_port=redis_port,
        redis_password=redis_pass,
        algo_name=algo_name, env_name=env_name,
        rollout_fragment_length=rollout_frag,
        train_batch_size=train_batch,
        sgd_minibatch_size=int(os.environ.get("SGD_MINIBATCH_SIZE", "128")),
        num_sgd_iter=1,
        num_gpus=0,
    )

    weights_b = pickle.dumps(actor.worker.get_policy().get_weights())
    state_b = pickle.dumps(learner.get_state())
    batch_b = pickle.dumps(batch)
    r.set("model_weights", weights_b)
    r.set("learner_state", state_b)
    r.hset("sample_batch", "seed", batch_b)

    print(f"[seed-in] model_weights {len(weights_b)/1024:.1f} KB")
    print(f"[seed-in] learner_state {len(state_b)/1024/1024:.1f} MB")
    print(f"[seed-in] batch[seed]   {len(batch_b)/1024/1024:.1f} MB")
    print("[seed-in] done")


if __name__ == "__main__":
    main()
