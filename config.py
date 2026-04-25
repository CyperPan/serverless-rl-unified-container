"""Verbatim port of sim-nitro/aws_lambda/config.py — single source of truth
for env / algo metadata so the unified container stays drop-in compatible
with the existing actor deployment.
"""
framework = "torch"

algos = [
    "ppo",
    "appo",
    "impala",
]

envs = {
    "Hopper-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 600,
        "budget": float("inf"),
        "rollout_fragment_length": 512,
    },
    "Hopper-v4": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 600,
        "budget": float("inf"),
        "rollout_fragment_length": 512,
    },
    "Humanoid-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 1000,
        "budget": 0.2,
        "rollout_fragment_length": 512,
    },
    "Walker2d-v3": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 1000,
        "budget": float("inf"),
        "rollout_fragment_length": 512,
    },
    "SpaceInvadersNoFrameskip-v4": {
        "is_env_discrete": True,
        "min_reward": 0,
        "max_reward": 1200,
        "budget": 2,
        "rollout_fragment_length": 16,
    },
    "QbertNoFrameskip-v4": {
        "is_env_discrete": True,
        "min_reward": 0,
        "max_reward": 300,
        "budget": float("inf"),
        "rollout_fragment_length": 16,
    },
    "GravitarNoFrameskip-v4": {
        "is_env_discrete": True,
        "min_reward": 0,
        "max_reward": 300,
        "budget": float("inf"),
        "rollout_fragment_length": 16,
    },
}

num_envs_per_worker = 4
