#!/usr/bin/env python3
"""Seed Redis with the artifacts the unified handler expects.

Builds an actor + learner ON CPU (host Python — no Docker), produces:
  - model_weights      from a freshly-built actor's policy
  - learner_state      from a freshly-built learner after one warmup step
  - sample_batch:seed  one rollout the first learner invocation can consume

Usage:
    python seed_redis.py
    REDIS_HOST=localhost REDIS_PORT=6379 python seed_redis.py
"""
from __future__ import annotations

import os
import pickle
import sys

# Reuse the unified container's modules from the parent dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import redis as _redis  # noqa: E402

from serverless_actor import ServerlessActor  # noqa: E402
from serverless_learner import ServerlessLearner  # noqa: E402
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID  # noqa: E402


def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_pass = os.environ.get("REDIS_PASSWORD", "")
    algo_name = os.environ.get("ALGO_NAME", "ppo")
    env_name = os.environ.get("ENV_NAME", "Hopper-v4")
    rollout_frag = int(os.environ.get("ROLLOUT_FRAGMENT_LENGTH", "512"))
    train_batch = int(os.environ.get("TRAIN_BATCH_SIZE", "4096"))

    print(f"[seed] redis://{redis_host}:{redis_port}  env={env_name}  algo={algo_name}")

    r = _redis.Redis(host=redis_host, port=redis_port, password=redis_pass or None)
    r.ping()

    print("[seed] building actor (CPU) ...")
    actor = ServerlessActor(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_pass,
        algo_name=algo_name,
        env_name=env_name,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_frag,
    )

    print("[seed] sampling one rollout (this is the actor's first batch) ...")
    batch = actor.worker.sample()
    if not isinstance(batch, SampleBatch):
        batch = batch.policy_batches[DEFAULT_POLICY_ID]

    # Pad to train_batch_size so the learner has a full batch.
    while len(batch) < train_batch:
        nb = actor.worker.sample()
        if not isinstance(nb, SampleBatch):
            nb = nb.policy_batches[DEFAULT_POLICY_ID]
        from ray.rllib.policy.sample_batch import concat_samples
        batch = concat_samples([batch, nb])
    batch = batch[:train_batch]

    print("[seed] building learner (CPU) ...")
    learner = ServerlessLearner(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_pass,
        algo_name=algo_name,
        env_name=env_name,
        rollout_fragment_length=rollout_frag,
        train_batch_size=train_batch,
        sgd_minibatch_size=int(os.environ.get("SGD_MINIBATCH_SIZE", "128")),
        num_sgd_iter=1,
        num_gpus=0,  # CPU during seed
    )
    learner.warmup_with_batch(batch)

    print("[seed] writing keys ...")
    weights_bytes = pickle.dumps(actor.worker.get_policy().get_weights())
    state_bytes = pickle.dumps(learner.get_state())
    batch_bytes = pickle.dumps(batch)

    r.set("model_weights", weights_bytes)
    r.set("learner_state", state_bytes)
    r.hset("sample_batch", "seed", batch_bytes)

    print(f"[seed] model_weights:   {len(weights_bytes)/1024:.1f} KB")
    print(f"[seed] learner_state:   {len(state_bytes)/1024/1024:.1f} MB")
    print(f"[seed] sample_batch[seed]: {len(batch_bytes)/1024/1024:.1f} MB")
    print("[seed] done")


if __name__ == "__main__":
    main()
