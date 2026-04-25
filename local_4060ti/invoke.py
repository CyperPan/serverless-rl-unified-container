#!/usr/bin/env python3
"""POST a JSON event to the Lambda Runtime Interface Emulator on :9000.

Usage:
    python invoke.py actor                   # one actor call
    python invoke.py learner --batch-id seed # one learner call on the seeded batch
    python invoke.py sequence                # actor → learner → actor → learner

Each invocation prints the per-phase timings the handler returns, so you
can directly diff cold vs warm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

URL = "http://localhost:9000/2015-03-31/functions/function/invocations"


def invoke(event: dict, timeout: float = 900.0) -> dict:
    data = json.dumps(event).encode()
    req = urllib.request.Request(
        URL,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    wall = time.perf_counter() - t
    try:
        result = json.loads(body)
    except Exception:
        result = {"raw": body.decode(errors="replace")}
    return {"client_wall_s": wall, "response": result}


def fmt_phase(timings: dict) -> str:
    parts = []
    for k, v in timings.items():
        ms = v * 1000 if isinstance(v, (int, float)) else v
        parts.append(f"{k}={ms:.1f}ms" if isinstance(ms, (int, float)) else f"{k}={ms}")
    return " ".join(parts)


def one(role: str, batch_id: str | None) -> dict:
    event = {"role": role}
    if role == "learner":
        event["batch_id"] = batch_id or "seed"
    out = invoke(event)
    resp = out["response"]
    print(f"[{role:7}]  client {out['client_wall_s']*1000:7.1f}ms  "
          f"server {resp.get('lambda_duration', 0)*1000:7.1f}ms  "
          f"prewarm={resp.get('prewarm', '?')}")
    if "phase_timings" in resp:
        print(f"           {fmt_phase(resp['phase_timings'])}")
    if "t0_phase" in resp and out["client_wall_s"] > 5:
        # Only print T0 breakdown on the first/cold call.
        print(f"           t0_phase: {fmt_phase(resp['t0_phase'])}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["actor", "learner", "sequence"])
    p.add_argument("--batch-id", default="seed")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    if args.role == "sequence":
        # Demonstrate role switch on a warm container.
        steps = ["actor", "learner", "actor", "learner"]
        for s in steps:
            one(s, args.batch_id)
    else:
        for _ in range(args.repeats):
            one(args.role, args.batch_id)


if __name__ == "__main__":
    main()
