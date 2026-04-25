"""Local exerciser — invoke handler.handler in actor → learner → actor order
to demonstrate that the warm container switches roles at T3 latency.

Requires a running Redis on REDIS_HOST:REDIS_PORT (defaults to localhost:6379).
Bootstraps Redis with an initial weights blob so the first actor call has
something to load.

Usage:
    UNIFIED_PREWARM=1 python local_test.py
    UNIFIED_PREWARM=0 python local_test.py     # lazy mode for comparison
"""
from __future__ import annotations

import os
import pickle
import time
import uuid
from types import SimpleNamespace

import handler  # noqa: F401  triggers eager prewarm (if enabled)


def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    redis_password = os.environ.get("REDIS_PASSWORD", "")

    print(f"\n[local_test] PREWARM={handler.PREWARM} t0_phase={handler._T0_PHASE}\n")

    # Seed Redis with initial weights so the first actor call has something to load.
    import redis as _redis
    r = _redis.Redis(host=redis_host, port=redis_port, password=redis_password)
    if r.get("model_weights") is None:
        actor_for_seed = handler._ensure_actor()
        r.set("model_weights", pickle.dumps(actor_for_seed.worker.get_policy().get_weights()))
        print("[local_test] seeded redis: model_weights")

    # Sequence: actor → learner → actor. The interesting number is the
    # second actor call (after a learner intervened) — it should be T3 fast.
    sequence = [
        ("actor", "first actor call (cold or prewarmed)"),
        ("learner", "first learner call (role switch)"),
        ("actor", "second actor call (back to actor)"),
        ("learner", "second learner call (warm)"),
    ]

    initial_state_pushed = False
    for role, label in sequence:
        if role == "learner" and not initial_state_pushed:
            # Seed Redis with initial learner state so set_state has data.
            l = handler._ensure_learner()
            r.set("learner_state", pickle.dumps(l.get_state()))
            initial_state_pushed = True

        event = {
            "role": role,
            "redis_host": redis_host,
            "redis_port": redis_port,
            "redis_password": redis_password,
            "algo_name": handler.ALGO_NAME,
            "env_name": handler.ENV_NAME,
            "num_envs_per_worker": handler.NUM_ENVS_PER_WORKER,
            "rollout_fragment_length": handler.ROLLOUT_FRAGMENT_LENGTH,
        }
        if role == "learner":
            # Learner consumes a batch the actor just pushed.
            batch_id = next(iter(r.hkeys("sample_batch") or [b"none"])).decode()
            event["batch_id"] = batch_id

        ctx = SimpleNamespace(aws_request_id=str(uuid.uuid4()))

        t = time.perf_counter()
        result = handler.handler(event, ctx)
        wall = time.perf_counter() - t

        print(f"[{role:7}] {label}")
        print(f"  wall={wall*1000:.1f}ms  duration={result['lambda_duration']*1000:.1f}ms")
        for k, v in result["phase_timings"].items():
            print(f"    {k:<22} {v*1000:7.2f} ms")
        print()


if __name__ == "__main__":
    main()
