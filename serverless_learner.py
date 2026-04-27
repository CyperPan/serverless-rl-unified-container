"""Learner — mirrors ServerlessActor's shape so the unified container can
treat both roles symmetrically.

State I/O via Redis (same pattern as the actor): pull state/batch by key,
push new state/weights back.

The class itself is a thin wrapper around `policy.learn_on_batch`. All
config is plain kwargs; nothing crosses a wire as a config dict.
"""
import pickle
from typing import Any, Dict, Optional

import gymnasium as gym
import ray
import redis as _redis

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo.appo_torch_policy import APPOTorchPolicy
from ray.rllib.algorithms.impala.impala import ImpalaConfig
from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch

import config

import warnings
warnings.filterwarnings("ignore")


class ServerlessLearner:
    """Learner packaged as a serverless function. Mirrors ServerlessActor."""

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: str,
        algo_name: str,
        env_name: str,
        rollout_fragment_length: int,
        train_batch_size: int,
        sgd_minibatch_size: int = 128,
        num_sgd_iter: int = 1,
        num_gpus: float = 1.0,
        lr: Optional[float] = None,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.algo_name = algo_name
        self.env_name = env_name

        env = gym.make(env_name)

        if algo_name == "ppo":
            learner_config = PPOConfig()
            policy_cls = PPOTorchPolicy
        elif algo_name == "appo":
            learner_config = APPOConfig()
            policy_cls = APPOTorchPolicy
        elif algo_name == "impala":
            learner_config = ImpalaConfig()
            policy_cls = ImpalaTorchPolicy
        else:
            raise ValueError(f"Unsupported algo: {algo_name}")

        training_kwargs: Dict[str, Any] = {
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": num_sgd_iter,
        }
        if lr is not None:
            training_kwargs["lr"] = lr

        learner_config = (
            learner_config
            .framework(framework=config.framework)
            .environment(
                env=env_name,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            .resources(num_gpus=num_gpus)
            .rollouts(
                rollout_fragment_length=rollout_fragment_length,
                num_rollout_workers=0,
                batch_mode="truncate_episodes",
            )
            .training(**training_kwargs)
            .debugging(
                log_level="ERROR",
                logger_config={"type": ray.tune.logger.NoopLogger},
                log_sys_usage=False,
            )
        )

        # Force LEGACY model stack to match the actor's state_dict shape
        # (see serverless_actor.py for context).
        if hasattr(learner_config, "experimental"):
            try:
                learner_config = learner_config.experimental(
                    _enable_new_api_stack=False)
            except TypeError:
                pass
        if hasattr(learner_config, "_enable_new_api_stack"):
            learner_config._enable_new_api_stack = False

        # Construct via RolloutWorker (same path as ServerlessActor); see
        # serverless_actor.py for why direct PPOTorchPolicy construction
        # fails on ray 2.8.
        self.worker = RolloutWorker(
            env_creator=lambda _: env,
            config=learner_config,
            default_policy_class=policy_cls,
        )
        self.policy = self.worker.get_policy()

    # ---- Redis I/O (mirrors ServerlessActor) -------------------------- #

    def init_redis_client(self) -> None:
        self.pool = _redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
        )
        self.redis_client = _redis.Redis(connection_pool=self.pool)

    def redis_get_state(self) -> Dict[str, Any]:
        """Full Policy state: weights + Adam moments + optimizer counters."""
        return pickle.loads(self.redis_client.get("learner_state"))

    def redis_set_state(self, state: Dict[str, Any]) -> None:
        self.redis_client.set("learner_state", pickle.dumps(state))

    def redis_set_model_weights(self, weights: Dict[str, Any]) -> None:
        """Push the *weights only* — what actors pull each iteration."""
        self.redis_client.set("model_weights", pickle.dumps(weights))

    def redis_get_sample_batch(self, batch_id: str) -> SampleBatch:
        return pickle.loads(self.redis_client.hget("sample_batch", batch_id))

    # ---- Update ------------------------------------------------------- #

    def set_state(self, state: Dict[str, Any]) -> None:
        self.policy.set_state(state)

    def get_state(self) -> Dict[str, Any]:
        return self.policy.get_state()

    def get_weights(self) -> Dict[str, Any]:
        return self.policy.get_weights()

    def learn(self, batch: SampleBatch) -> Dict[str, Any]:
        return self.policy.learn_on_batch(batch)

    # ---- Warmup ------------------------------------------------------- #

    def warmup_with_batch(self, batch: SampleBatch) -> None:
        """Run one learn step to amortize CUDA kernel JIT cost.

        Caller passes a real SampleBatch (the actor's output works). After
        this returns, subsequent learn() calls land on the warm path.
        """
        self.policy.learn_on_batch(batch)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
