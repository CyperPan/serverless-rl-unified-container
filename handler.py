"""Unified Lambda handler — one container, two roles.

Cold-start (first invocation in this container) pays the union of:
  - imports: torch, ray, rllib, gymnasium, mujoco, redis
  - Ray init (only used by actor's RolloutWorker internals; learner doesn't need ray.init())
  - env construction (gym.make, MuJoCo dlopen)
  - actor RolloutWorker build (worker + connector pipeline)
  - learner Policy build (Adam optimizer + GPU placement)
  - one warmup execution path that exercises CUDA JIT (so the learner's first
    real `learn` doesn't pay the kernel-compilation tax)

After this, every invocation — actor or learner — runs at "T3 resident"
latency: only set_weights/set_state + sample/learn.

Invocation contract (event):
  {
    "role": "actor" | "learner",
    "redis_host": "...", "redis_port": 6379, "redis_password": "...",
    "algo_name": "ppo" | "appo" | "impala",
    "env_name": "Hopper-v3" | "Hopper-v4" | ...,
    "num_envs_per_worker": 1,                 # actor only
    "rollout_fragment_length": 512,
    "train_batch_size": 4096,                 # learner only
    "batch_id": "<uuid>",                     # learner only — which batch to consume
  }

Returns:
  {
    "aws_request_id": str,
    "role": "actor" | "learner",
    "lambda_duration": float,
    "phase_timings": { ... }                  # cold/warm breakdown
  }
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")

# ── Imports happen exactly once per container lifetime ────────────────
import gymnasium as gym  # noqa: F401
import ray  # noqa: F401
import torch  # noqa: F401
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID

from serverless_actor import ServerlessActor
from serverless_learner import ServerlessLearner
import config


# ── Container-lifetime configuration (Lambda env vars) ────────────────
#
# Anything that must be identical across actor and learner invocations
# is set via env vars at deploy time. Per-invocation knobs (role,
# batch_id) come through `event`.
ALGO_NAME = os.environ.get("ALGO_NAME", "ppo")
ENV_NAME = os.environ.get("ENV_NAME", "Hopper-v3")
ROLLOUT_FRAGMENT_LENGTH = int(
    os.environ.get("ROLLOUT_FRAGMENT_LENGTH",
                   config.envs.get(ENV_NAME, {}).get("rollout_fragment_length", 512)))
NUM_ENVS_PER_WORKER = int(os.environ.get("NUM_ENVS_PER_WORKER",
                                         config.num_envs_per_worker))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE",
                                      ROLLOUT_FRAGMENT_LENGTH * 8))
SGD_MINIBATCH_SIZE = int(os.environ.get("SGD_MINIBATCH_SIZE", 128))
NUM_SGD_ITER = int(os.environ.get("NUM_SGD_ITER", 1))
NUM_GPUS_LEARNER = float(os.environ.get("NUM_GPUS_LEARNER", "1"))

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

# Toggle eager prewarm. Default ON: first cold start pays both build paths
# so the second invocation (any role) is T3-fast. Set UNIFIED_PREWARM=0
# for lazy mode (cheaper cold start, T1 cost when role first used).
PREWARM = os.environ.get("UNIFIED_PREWARM", "1") == "1"
WARMUP_LEARN = os.environ.get("UNIFIED_WARMUP_LEARN", "1") == "1"


# ── Module-level caches (persist across warm invocations) ─────────────
_actor: Optional[ServerlessActor] = None
_learner: Optional[ServerlessLearner] = None
_warmup_done: bool = False


# ── Builders ──────────────────────────────────────────────────────────

def _build_actor() -> ServerlessActor:
    a = ServerlessActor(
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_password=REDIS_PASSWORD,
        algo_name=ALGO_NAME,
        env_name=ENV_NAME,
        num_envs_per_worker=NUM_ENVS_PER_WORKER,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
    )
    a.init_redis_client()
    return a


def _build_learner() -> ServerlessLearner:
    l = ServerlessLearner(
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        redis_password=REDIS_PASSWORD,
        algo_name=ALGO_NAME,
        env_name=ENV_NAME,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        train_batch_size=TRAIN_BATCH_SIZE,
        sgd_minibatch_size=SGD_MINIBATCH_SIZE,
        num_sgd_iter=NUM_SGD_ITER,
        num_gpus=NUM_GPUS_LEARNER,
    )
    l.init_redis_client()
    return l


def _ensure_actor() -> ServerlessActor:
    global _actor
    if _actor is None:
        _actor = _build_actor()
    return _actor


def _ensure_learner() -> ServerlessLearner:
    global _learner
    if _learner is None:
        _learner = _build_learner()
    return _learner


def _warmup_learn_with_actor_batch() -> None:
    """Bridge the two roles: produce one batch via actor, run one learn step
    on the learner. Triggers CUDA kernel JIT inside the learner so that the
    first real invocation doesn't pay for it."""
    global _warmup_done
    if _warmup_done:
        return
    actor = _ensure_actor()
    learner = _ensure_learner()
    batch = actor.worker.sample()
    if not isinstance(batch, SampleBatch):
        batch = batch.policy_batches[DEFAULT_POLICY_ID]
    learner.warmup_with_batch(batch)
    _warmup_done = True


# ── Eager prewarm at module load (T0 covers both roles) ───────────────

_T0_START = time.perf_counter()
_T0_PHASE: Dict[str, float] = {"imports_done": time.perf_counter() - _T0_START}

if PREWARM:
    _t = time.perf_counter()
    _ensure_actor()
    _T0_PHASE["actor_build"] = time.perf_counter() - _t

    _t = time.perf_counter()
    _ensure_learner()
    _T0_PHASE["learner_build"] = time.perf_counter() - _t

    if WARMUP_LEARN:
        _t = time.perf_counter()
        _warmup_learn_with_actor_batch()
        _T0_PHASE["warmup_exec"] = time.perf_counter() - _t

_T0_PHASE["total"] = time.perf_counter() - _T0_START


# ── Handler ───────────────────────────────────────────────────────────

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entrypoint. Dispatches by event['role']."""
    invocation_start = time.perf_counter()
    role = event.get("role", "actor")
    aws_request_id = getattr(context, "aws_request_id", "local")

    timings: Dict[str, float] = {}

    if role == "actor":
        # ── Actor path: pull weights, sample, push batch ───────────────
        t = time.perf_counter()
        actor = _ensure_actor()  # cached if PREWARM=1
        timings["ensure_role"] = time.perf_counter() - t

        t = time.perf_counter()
        weights = actor.redis_get_model_weights()
        timings["redis_get_weights"] = time.perf_counter() - t

        t = time.perf_counter()
        actor.set_model_weights(weights)
        timings["set_weights"] = time.perf_counter() - t

        t = time.perf_counter()
        sample_batch = actor.sample()
        timings["sample"] = time.perf_counter() - t

        t = time.perf_counter()
        actor.redis_hset_sample_batch("sample_batch", aws_request_id, sample_batch)
        timings["redis_push_batch"] = time.perf_counter() - t

        duration = time.perf_counter() - invocation_start
        actor.redis_hset_lambda_duration("lambda_duration", aws_request_id, duration)

    elif role == "learner":
        # ── Learner path: pull state + batch, learn, push new state ───
        t = time.perf_counter()
        learner = _ensure_learner()
        timings["ensure_role"] = time.perf_counter() - t

        t = time.perf_counter()
        state = learner.redis_get_state()
        timings["redis_get_state"] = time.perf_counter() - t

        t = time.perf_counter()
        learner.set_state(state)
        timings["set_state"] = time.perf_counter() - t

        t = time.perf_counter()
        batch = learner.redis_get_sample_batch(event["batch_id"])
        timings["redis_get_batch"] = time.perf_counter() - t

        t = time.perf_counter()
        grad_info = learner.learn(batch)  # noqa: F841
        if torch.cuda.is_available() and NUM_GPUS_LEARNER > 0:
            torch.cuda.synchronize()
        timings["learn"] = time.perf_counter() - t

        t = time.perf_counter()
        learner.redis_set_state(learner.get_state())
        learner.redis_set_model_weights(learner.get_weights())
        timings["redis_push_state"] = time.perf_counter() - t

        duration = time.perf_counter() - invocation_start

    else:
        raise ValueError(f"Unsupported role: {role!r}")

    return {
        "aws_request_id": aws_request_id,
        "role": role,
        "lambda_duration": duration,
        "phase_timings": timings,
        "t0_phase": _T0_PHASE,
        "prewarm": PREWARM,
    }
