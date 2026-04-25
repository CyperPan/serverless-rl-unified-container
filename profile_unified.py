#!/usr/bin/env python3
"""Unified actor+learner container — 4-tier resource & latency profile.

A single Python module shaped like the union of ServerlessActor and
StatelessLearner. The same import surface, env setup, and Policy build
serves both roles. The `--role` flag picks which execution path runs.

Roles:
  actor   — RolloutWorker.sample() produces a SampleBatch
  learner — PPOTorchPolicy.learn_on_batch() runs one PPO update

Tiers (15 runs each by default):
  T0  fresh Python subprocess; cold imports + CUDA init + build + load + exec
  T1  reuse interpreter; rebuild Policy/Worker each iter; reload state + exec
  T2  reuse Policy/Worker; reload state + batch; exec
  T3  reuse Policy/Worker + state resident; exec only (load batch only)

Producer (one-time, not measured):
  Build an actor on CPU, sample a batch (the learner consumes it). Pickle:
    - state_learner.pkl   weights + Adam moments + optimizer config
    - weights_actor.pkl   weights only (what actors pull each iteration)
    - batch_learner.pkl   one SampleBatch the learner trains on
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


# ── Resource snapshot (mirrors profile_visual_learner.py) ──────────────

def _snap(pid: Optional[int] = None, gpu: bool = False) -> Dict[str, Any]:
    import resource as res
    if pid is None:
        pid = os.getpid()
    s: Dict[str, Any] = {"ts": time.perf_counter(), "wall_ts": time.time()}
    try:
        with open(f"/proc/{pid}/statm") as f:
            pages = int(f.read().split()[1])
        s["rss_bytes"] = pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        s["rss_bytes"] = 0
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
        tps = os.sysconf("SC_CLK_TCK")
        s["utime_s"] = int(fields[11]) / tps
        s["stime_s"] = int(fields[12]) / tps
    except Exception:
        s["utime_s"] = s["stime_s"] = 0.0
    s["vol_ctx"] = s["nonvol_ctx"] = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches:"):
                    s["vol_ctx"] = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    s["nonvol_ctx"] = int(line.split()[1])
    except Exception:
        pass
    s["read_bytes"] = s["write_bytes"] = 0
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    s["read_bytes"] = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    s["write_bytes"] = int(line.split()[1])
    except Exception:
        pass
    ru = res.getrusage(res.RUSAGE_SELF)
    s["majflt"] = ru.ru_majflt
    s["minflt"] = ru.ru_minflt
    if gpu:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            parts = r.stdout.strip().split(",")
            s["gpu_util"] = float(parts[0])
            s["gpu_mem_used_mb"] = float(parts[1])
            s["gpu_mem_total_mb"] = float(parts[2])
        except Exception:
            s["gpu_util"] = -1
    return s


def _delta(b: Dict[str, Any], a: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    d["delta_rss_mb"] = (a["rss_bytes"] - b["rss_bytes"]) / (1024 * 1024)
    d["peak_rss_mb"] = a["rss_bytes"] / (1024 * 1024)
    cpu_b = b["utime_s"] + b["stime_s"]
    cpu_a = a["utime_s"] + a["stime_s"]
    d["cpu_time_s"] = cpu_a - cpu_b
    d["mean_cpu_percent"] = ((cpu_a - cpu_b) / elapsed_s * 100) if elapsed_s > 0 else 0.0
    d["vol_ctx"] = a["vol_ctx"] - b["vol_ctx"]
    d["nonvol_ctx"] = a["nonvol_ctx"] - b["nonvol_ctx"]
    d["total_ctx"] = d["vol_ctx"] + d["nonvol_ctx"]
    d["majflt"] = a["majflt"] - b["majflt"]
    d["minflt"] = a["minflt"] - b["minflt"]
    d["read_bytes"] = a["read_bytes"] - b["read_bytes"]
    d["write_bytes"] = a["write_bytes"] - b["write_bytes"]
    if "gpu_util" in a and a["gpu_util"] >= 0:
        d["gpu_util"] = a["gpu_util"]
        d["gpu_mem_used_mb"] = a["gpu_mem_used_mb"]
    d["wall_s"] = elapsed_s
    return d


def _fmt_bytes(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1024 * 1024: return f"{b/1024:.0f} KB"
    if b < 1024 * 1024 * 1024: return f"{b/(1024*1024):.1f} MB"
    return f"{b/(1024**3):.2f} GB"


def _fmt_phase(name: str, d: Dict[str, Any]) -> str:
    rss = f"+{d['delta_rss_mb']:.0f} MB" if d["delta_rss_mb"] > 0.5 else "0 MB"
    parts = [rss,
             f"wall {d['wall_s']*1000:.0f}ms",
             f"cpu {d['mean_cpu_percent']:.0f}%"]
    if d["majflt"] > 0:
        parts.append(f"flt {d['majflt']}")
    parts.append(f"ctx {d['total_ctx']}")
    if d["read_bytes"] > 0:
        parts.append(f"rd {_fmt_bytes(d['read_bytes'])}")
    if d["write_bytes"] > 0:
        parts.append(f"wr {_fmt_bytes(d['write_bytes'])}")
    if "gpu_util" in d:
        parts.append(f"gpu {d['gpu_util']:.0f}%")
    return f"  {name:<28} " + " · ".join(parts)


def print_tier_table(tier_name: str, results: Dict[str, Any]) -> None:
    print(f"\n{'='*84}")
    print(f"{tier_name}  total={results.get('_total_wall', 0):.4f}s  "
          f"peakRSS={results.get('_peak_rss_mb', 0):.0f} MB")
    print(f"{'='*84}")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(_fmt_phase(k, v))


# ── Workload config ────────────────────────────────────────────────────

DEFAULTS = {
    "algo": "ppo",
    "env": "Hopper-v4",            # cluster gymnasium ships v4; sim-nitro uses v3
    "rollout_frag": 512,
    "train_batch": 4096,
    "sgd_mini": 128,
    "sgd_iter": 1,
    "num_envs_per_worker": 1,
    "num_gpus_learner": 1,         # learner uses 1 GPU
}


# ── Build helpers (union of actor + learner construction paths) ────────

def _build_config(wl: Dict[str, Any], role: str):
    """Return a fully-configured PPOConfig identical between roles
    except for `num_gpus` (learner only) and `train_batch_size` (learner)."""
    import gymnasium as gym
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    env = gym.make(wl["env"])
    cfg = (
        PPOConfig()
        .framework("torch")
        .environment(env=wl["env"],
                     observation_space=env.observation_space,
                     action_space=env.action_space)
        .rollouts(rollout_fragment_length=wl["rollout_frag"],
                  num_rollout_workers=0,
                  num_envs_per_worker=wl["num_envs_per_worker"],
                  batch_mode="truncate_episodes")
        .training(train_batch_size=wl["train_batch"],
                  sgd_minibatch_size=wl["sgd_mini"],
                  num_sgd_iter=wl["sgd_iter"])
        .debugging(log_level="ERROR",
                   logger_config={"type": ray.tune.logger.NoopLogger},
                   log_sys_usage=False)
    )
    num_gpus = wl["num_gpus_learner"] if role == "learner" else 0
    cfg = cfg.resources(num_gpus=num_gpus)
    if hasattr(cfg, "_enable_new_api_stack"):
        cfg._enable_new_api_stack = False
    return env, cfg


def _build_actor_worker(wl: Dict[str, Any]):
    """Mirrors ServerlessActor.__init__: env, PPOConfig, RolloutWorker."""
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    env, cfg = _build_config(wl, role="actor")
    worker = RolloutWorker(
        env_creator=lambda _: env,
        config=cfg,
        default_policy_class=PPOTorchPolicy,
    )
    return worker


def _build_learner_policy(wl: Dict[str, Any]):
    """Mirrors StatelessLearner.__init__: env, PPOConfig, Policy directly."""
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
    env, cfg = _build_config(wl, role="learner")
    policy = PPOTorchPolicy(env.observation_space, env.action_space, cfg.to_dict())
    return policy


# ── Tier runners ───────────────────────────────────────────────────────
#
# Each tier returns dict { phase_name -> _delta result, _total_wall, _peak_rss_mb }.
# Phases differ per role; common keys keep the same name where semantics match.

def run_t0_actor(weights_path: str, wl: Dict[str, Any], gpu: bool = False) -> Dict[str, Any]:
    r: Dict[str, Any] = {}

    s0 = _snap(gpu=gpu); t0 = time.perf_counter()

    sb = s0; tb = t0
    import gymnasium  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_gymnasium"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    import torch  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_torch_cuda"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    import ray  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_ray"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy  # noqa: F401
    from ray.rllib.evaluation.rollout_worker import RolloutWorker  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_rllib"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    worker = _build_actor_worker(wl)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["build_worker"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(weights_path, "rb") as f:
        weights_bytes = f.read()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["weights_disk_read"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    weights = pickle.loads(weights_bytes)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["weights_deserialize"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    worker.get_policy().set_weights(weights)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_weights"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    batch = worker.sample()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_sample"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


def run_t0_learner(state_path: str, batch_path: str, wl: Dict[str, Any], gpu: bool = False) -> Dict[str, Any]:
    r: Dict[str, Any] = {}
    s0 = _snap(gpu=gpu); t0 = time.perf_counter()

    sb = s0; tb = t0
    import gymnasium  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_gymnasium"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    import torch  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_torch_cuda"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    import ray  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_ray"] = _delta(sb, sa, ta - tb)

    sb, tb = sa, ta
    from ray.rllib.algorithms.ppo import PPOConfig  # noqa: F401
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy  # noqa: F401
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["import_rllib"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy = _build_learner_policy(wl)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["build_policy"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(state_path, "rb") as f:
        state_bytes = f.read()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["state_disk_read"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    state = pickle.loads(state_bytes)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["state_deserialize"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.set_state(state)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_state"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(batch_path, "rb") as f:
        batch = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["batch_load"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.learn_on_batch(batch)
    if wl["num_gpus_learner"] > 0:
        import torch as _torch
        _torch.cuda.synchronize()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_learn"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


def run_t1_actor(weights_path: str, wl: Dict[str, Any], gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    worker = _build_actor_worker(wl)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["build_worker"] = _delta(sb, sa, ta - t0)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(weights_path, "rb") as f:
        weights = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["weights_load"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    worker.get_policy().set_weights(weights)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_weights"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    batch = worker.sample()  # noqa: F841
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_sample"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r, worker


def run_t1_learner(state_path: str, batch_path: str, wl: Dict[str, Any], gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    policy = _build_learner_policy(wl)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["build_policy"] = _delta(sb, sa, ta - t0)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(state_path, "rb") as f:
        state = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["state_load"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.set_state(state)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_state"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(batch_path, "rb") as f:
        batch = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["batch_load"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.learn_on_batch(batch)
    if wl["num_gpus_learner"] > 0:
        import torch as _torch
        _torch.cuda.synchronize()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_learn"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r, policy


def run_t2_actor(weights_path: str, worker, gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    with open(weights_path, "rb") as f:
        weights = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["weights_load"] = _delta(sb, sa, ta - t0)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    worker.get_policy().set_weights(weights)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_weights"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    worker.sample()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_sample"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


def run_t2_learner(state_path: str, batch_path: str, policy, wl: Dict[str, Any], gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    with open(state_path, "rb") as f:
        state = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["state_load"] = _delta(sb, sa, ta - t0)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.set_state(state)
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["set_state"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    with open(batch_path, "rb") as f:
        batch = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["batch_load"] = _delta(sb, sa, ta - tb)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.learn_on_batch(batch)
    if wl["num_gpus_learner"] > 0:
        import torch as _torch
        _torch.cuda.synchronize()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_learn"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


def run_t3_actor(worker, gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    worker.sample()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_sample"] = _delta(sb, sa, ta - t0)
    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


def run_t3_learner(batch_path: str, policy, wl: Dict[str, Any], gpu: bool = False):
    r: Dict[str, Any] = {}
    sb = _snap(gpu=gpu); t0 = time.perf_counter()
    with open(batch_path, "rb") as f:
        batch = pickle.loads(f.read())
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["batch_load"] = _delta(sb, sa, ta - t0)

    sb = _snap(gpu=gpu); tb = time.perf_counter()
    policy.learn_on_batch(batch)
    if wl["num_gpus_learner"] > 0:
        import torch as _torch
        _torch.cuda.synchronize()
    ta = time.perf_counter(); sa = _snap(gpu=gpu)
    r["execution_learn"] = _delta(sb, sa, ta - tb)

    r["_total_wall"] = ta - t0
    r["_peak_rss_mb"] = sa["rss_bytes"] / (1024 * 1024)
    return r


# ── Producer ───────────────────────────────────────────────────────────

def produce(state_path: str, weights_path: str, batch_path: str, wl: Dict[str, Any]) -> None:
    """Build an actor on CPU, sample a batch, and pickle three artifacts."""
    this = os.path.abspath(__file__)
    script = f"""
import os, sys, pickle
sys.path.insert(0, {os.path.dirname(this)!r})
from profile_unified import _build_actor_worker, _build_learner_policy

wl = {wl!r}
# Force CPU producer for reproducible state files (no GPU dep).
wl["num_gpus_learner"] = 0

# Actor produces one rollout (= sample batch the learner will train on).
worker = _build_actor_worker(wl)
batch = worker.sample()
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
if not isinstance(batch, SampleBatch):
    batch = batch.policy_batches[DEFAULT_POLICY_ID]

# Pad to train_batch_size by collecting more rollouts if needed.
TARGET = {wl['train_batch']}
while len(batch) < TARGET:
    nb = worker.sample()
    if not isinstance(nb, SampleBatch):
        nb = nb.policy_batches[DEFAULT_POLICY_ID]
    from ray.rllib.policy.sample_batch import concat_samples
    batch = concat_samples([batch, nb])
batch = batch[:TARGET]

with open({batch_path!r}, "wb") as f:
    pickle.dump(batch, f)

policy = worker.get_policy()
with open({weights_path!r}, "wb") as f:
    pickle.dump(policy.get_weights(), f)

# Build a learner separately to get a fresh state including Adam moments.
learner = _build_learner_policy(wl)
# Run one learn step so Adam m / v are populated to realistic shapes.
learner.learn_on_batch(batch)
with open({state_path!r}, "wb") as f:
    pickle.dump(learner.get_state(), f)

print(
    f"batch_rows={{len(batch)}} "
    f"batch_MB={{os.path.getsize({batch_path!r})/1024/1024:.1f}} "
    f"weights_MB={{os.path.getsize({weights_path!r})/1024/1024:.1f}} "
    f"state_MB={{os.path.getsize({state_path!r})/1024/1024:.1f}}"
)
"""
    print(f"[producer] env={wl['env']} algo={wl['algo']} rollout_frag={wl['rollout_frag']} "
          f"train_batch={wl['train_batch']}")
    t = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[producer] FAILED:\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[producer] {proc.stdout.strip()} ({time.perf_counter() - t:.1f}s)")


# ── T0 subprocess plumbing ─────────────────────────────────────────────

def run_t0_in_subprocess(role: str, paths: Dict[str, str], wl: Dict[str, Any], gpu: bool) -> Optional[Dict[str, Any]]:
    cmd = [sys.executable, os.path.abspath(__file__), "--t0-subprocess",
           "--role", role,
           "--state-path", paths["state"],
           "--weights-path", paths["weights"],
           "--batch-path", paths["batch"],
           "--env", wl["env"],
           "--rollout-frag", str(wl["rollout_frag"]),
           "--train-batch", str(wl["train_batch"]),
           "--sgd-mini", str(wl["sgd_mini"]),
           "--sgd-iter", str(wl["sgd_iter"]),
           "--num-gpus-learner", str(wl["num_gpus_learner"])]
    if gpu:
        cmd.append("--gpu")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload: Optional[Dict[str, Any]] = None
    for line in proc.stdout.split("\n"):
        if line.startswith("__JSON__"):
            try:
                payload = json.loads(line[len("__JSON__"):])
            except Exception:
                payload = None
        else:
            print(line)
    for line in proc.stderr.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(skip in s for skip in ("DeprecationWarning", "FutureWarning", "UserWarning")):
            continue
        print(f"  [stderr] {line}", file=sys.stderr)
    return payload


def _t0_subprocess_entry(args: argparse.Namespace, wl: Dict[str, Any]) -> None:
    if args.role == "actor":
        results = run_t0_actor(args.weights_path, wl, gpu=args.gpu)
        print_tier_table("T0 actor (cold)", results)
    else:
        results = run_t0_learner(args.state_path, args.batch_path, wl, gpu=args.gpu)
        print_tier_table("T0 learner (cold)", results)
    print("__JSON__" + json.dumps(results, default=str))


# ── Aggregation ────────────────────────────────────────────────────────

def summarize_runs(label: str, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect run-level totals and per-phase wall_s arrays."""
    if not all_results:
        return {}
    totals = [r["_total_wall"] for r in all_results]
    peaks = [r["_peak_rss_mb"] for r in all_results]
    phase_walls: Dict[str, List[float]] = {}
    phase_rss: Dict[str, List[float]] = {}
    phase_cpu: Dict[str, List[float]] = {}
    for r in all_results:
        for k, v in r.items():
            if k.startswith("_"):
                continue
            phase_walls.setdefault(k, []).append(v["wall_s"])
            phase_rss.setdefault(k, []).append(v["delta_rss_mb"])
            phase_cpu.setdefault(k, []).append(v["mean_cpu_percent"])
    return {
        "label": label,
        "n": len(all_results),
        "total_wall": totals,
        "peak_rss_mb": peaks,
        "phase_walls": phase_walls,
        "phase_rss": phase_rss,
        "phase_cpu": phase_cpu,
    }


def _stats(xs: List[float]) -> str:
    import statistics
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4g}"
    return f"{statistics.mean(xs):.4g} ± {statistics.stdev(xs):.4g} (min {min(xs):.4g}, max {max(xs):.4g})"


def print_tier_summary(s: Dict[str, Any]) -> None:
    if not s:
        return
    print(f"\n--- {s['label']} (N={s['n']}) ---")
    print(f"  total_wall_s : {_stats(s['total_wall'])}")
    print(f"  peak_rss_mb  : {_stats(s['peak_rss_mb'])}")
    print("  per-phase wall_ms:")
    for k, xs in s["phase_walls"].items():
        ms = [x * 1000 for x in xs]
        print(f"    {k:<26} {_stats(ms)}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["actor", "learner"])
    p.add_argument("--n-runs", type=int, default=15)
    p.add_argument("--state-path", default="/tmp/unified_state.pkl")
    p.add_argument("--weights-path", default="/tmp/unified_weights.pkl")
    p.add_argument("--batch-path", default="/tmp/unified_batch.pkl")
    p.add_argument("--env", default=DEFAULTS["env"])
    p.add_argument("--rollout-frag", type=int, default=DEFAULTS["rollout_frag"])
    p.add_argument("--train-batch", type=int, default=DEFAULTS["train_batch"])
    p.add_argument("--sgd-mini", type=int, default=DEFAULTS["sgd_mini"])
    p.add_argument("--sgd-iter", type=int, default=DEFAULTS["sgd_iter"])
    p.add_argument("--num-envs-per-worker", type=int, default=DEFAULTS["num_envs_per_worker"])
    p.add_argument("--num-gpus-learner", type=int, default=DEFAULTS["num_gpus_learner"])
    p.add_argument("--gpu", action="store_true", help="Sample nvidia-smi in snapshots")
    p.add_argument("--producer", action="store_true",
                   help="Generate state/weights/batch artifacts and exit")
    p.add_argument("--t0-subprocess", action="store_true",
                   help="Internal: run a single T0 measurement in this process")
    p.add_argument("--summary-out", default=None,
                   help="Path to write JSON aggregate (one role's runs)")
    args = p.parse_args()

    wl = {
        "algo": "ppo",
        "env": args.env,
        "rollout_frag": args.rollout_frag,
        "train_batch": args.train_batch,
        "sgd_mini": args.sgd_mini,
        "sgd_iter": args.sgd_iter,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus_learner": args.num_gpus_learner,
    }

    if args.t0_subprocess:
        _t0_subprocess_entry(args, wl)
        return

    if args.producer:
        produce(args.state_path, args.weights_path, args.batch_path, wl)
        return

    if args.role is None:
        p.error("must specify --role actor|learner (unless --producer or --t0-subprocess)")

    if not (os.path.exists(args.state_path) and os.path.exists(args.weights_path)
            and os.path.exists(args.batch_path)):
        produce(args.state_path, args.weights_path, args.batch_path, wl)

    paths = {"state": args.state_path,
             "weights": args.weights_path,
             "batch": args.batch_path}

    print(f"\n{'#'*84}")
    print(f"# Unified container · role={args.role} · env={args.env}  algo=ppo")
    print(f"# rollout_frag={args.rollout_frag}  train_batch={args.train_batch}")
    print(f"# Runs/tier: {args.n_runs}  GPU sample: {args.gpu}  "
          f"num_gpus_learner: {args.num_gpus_learner}")
    print(f"{'#'*84}")
    print(f"[pre] torch in modules: {'torch' in sys.modules}, "
          f"ray in modules: {'ray' in sys.modules}")

    summary: Dict[str, List[Dict[str, Any]]] = {"T0": [], "T1": [], "T2": [], "T3": []}

    # ── T0 — fresh subprocess each run ──
    print(f"\n{'='*84}\nPHASE T0  ({args.n_runs}× fresh subprocess)\n{'='*84}")
    for i in range(args.n_runs):
        print(f"\n--- T0 {args.role} run {i+1}/{args.n_runs} ---")
        res = run_t0_in_subprocess(args.role, paths, wl, gpu=args.gpu)
        if res is not None:
            summary["T0"].append(res)

    # ── T1 — same process, rebuild Policy/Worker ──
    print(f"\n{'='*84}\nPHASE T1  ({args.n_runs}× rebuild)\n{'='*84}")
    persistent = None
    for i in range(args.n_runs):
        print(f"\n--- T1 {args.role} run {i+1}/{args.n_runs} ---")
        if args.role == "actor":
            res, obj = run_t1_actor(args.weights_path, wl, gpu=args.gpu)
        else:
            res, obj = run_t1_learner(args.state_path, args.batch_path, wl, gpu=args.gpu)
        print_tier_table(f"T1 {args.role} run {i+1}", res)
        summary["T1"].append(res)
        if persistent is None:
            persistent = obj
        else:
            del obj
            import gc
            gc.collect()
            if args.num_gpus_learner > 0:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

    # ── T2 — reuse Policy/Worker ──
    print(f"\n{'='*84}\nPHASE T2  ({args.n_runs}× reuse, reload state)\n{'='*84}")
    for i in range(args.n_runs):
        print(f"\n--- T2 {args.role} run {i+1}/{args.n_runs} ---")
        if args.role == "actor":
            res = run_t2_actor(args.weights_path, persistent, gpu=args.gpu)
        else:
            res = run_t2_learner(args.state_path, args.batch_path, persistent, wl, gpu=args.gpu)
        print_tier_table(f"T2 {args.role} run {i+1}", res)
        summary["T2"].append(res)

    # ── T3 — state resident ──
    print(f"\n{'='*84}\nPHASE T3  ({args.n_runs}× exec only)\n{'='*84}")
    for i in range(args.n_runs):
        print(f"\n--- T3 {args.role} run {i+1}/{args.n_runs} ---")
        if args.role == "actor":
            res = run_t3_actor(persistent, gpu=args.gpu)
        else:
            res = run_t3_learner(args.batch_path, persistent, wl, gpu=args.gpu)
        print_tier_table(f"T3 {args.role} run {i+1}", res)
        summary["T3"].append(res)

    # ── Summary ──
    print(f"\n\n{'#'*84}\n# SUMMARY · role={args.role} · {args.n_runs} runs/tier\n{'#'*84}")
    aggregates = {tier: summarize_runs(f"{tier} {args.role}", runs)
                  for tier, runs in summary.items()}
    for tier in ["T0", "T1", "T2", "T3"]:
        print_tier_summary(aggregates[tier])

    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump({"role": args.role, "wl": wl, "tiers": summary}, f, default=str)
        print(f"\n[wrote] {args.summary_out}")

    print(f"\n{'='*84}\nDONE\n{'='*84}")


if __name__ == "__main__":
    main()
