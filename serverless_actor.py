"""Actor — verbatim port of sim-nitro/aws_lambda/serverless_actor.py.

Kept identical to the upstream so the unified container stays a drop-in
replacement for the actor-only image.
"""
import pickle
import gymnasium as gym
import ray
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo.appo_torch_policy import APPOTorchPolicy
from ray.rllib.algorithms.impala.impala import ImpalaConfig
from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy

import redis as _redis
import config

import warnings
warnings.filterwarnings("ignore")


class ServerlessActor:
    """Actor packaged as a serverless function. State I/O via Redis."""

    def __init__(
        self,
        redis_host,
        redis_port,
        redis_password,
        algo_name,
        env_name,
        num_envs_per_worker,
        rollout_fragment_length,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.algo_name = algo_name
        self.env_name = env_name
        self.rollout_fragment_length = rollout_fragment_length

        env = gym.make(env_name)

        if algo_name == "ppo":
            sampler_config = PPOConfig()
        elif algo_name == "appo":
            sampler_config = APPOConfig()
        elif algo_name == "impala":
            sampler_config = ImpalaConfig()

        sampler_config = (
            sampler_config
            .framework(framework=config.framework)
            .environment(
                env=env_name,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            .rollouts(
                rollout_fragment_length=rollout_fragment_length,
                num_rollout_workers=0,
                num_envs_per_worker=num_envs_per_worker,
                batch_mode="truncate_episodes",
            )
            .training(train_batch_size=rollout_fragment_length)
            .debugging(
                log_level="ERROR",
                logger_config={"type": ray.tune.logger.NoopLogger},
                log_sys_usage=False,
            )
        )

        # Force LEGACY model stack so weights are interchangeable with the
        # host's seeded model_weights (host nitro_env uses gymnasium 1.2.3
        # which produces legacy state_dict; container has gymnasium 0.28.1
        # which would default to RLModule API and produce a different shape).
        if hasattr(sampler_config, "experimental"):
            try:
                sampler_config = sampler_config.experimental(
                    _enable_new_api_stack=False)
            except TypeError:
                pass
        if hasattr(sampler_config, "_enable_new_api_stack"):
            sampler_config._enable_new_api_stack = False

        if algo_name == "ppo":
            policy_cls = PPOTorchPolicy
        elif algo_name == "appo":
            policy_cls = APPOTorchPolicy
        else:
            policy_cls = ImpalaTorchPolicy

        self.worker = RolloutWorker(
            env_creator=lambda _: env,
            config=sampler_config,
            default_policy_class=policy_cls,
        )

    def init_redis_client(self):
        self.pool = _redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
        )
        self.redis_client = _redis.Redis(connection_pool=self.pool)

    def redis_hset_sample_batch(self, name, batch_id, batch):
        self.redis_client.hset(name, batch_id, pickle.dumps(batch))

    def redis_hset_lambda_duration(self, name, batch_id, lambda_duration):
        self.redis_client.hset(name, batch_id, lambda_duration)

    def sample(self):
        return self.worker.sample()

    def redis_get_model_weights(self):
        return pickle.loads(self.redis_client.get("model_weights"))

    def set_model_weights(self, model_weights):
        self.worker.get_policy().set_weights(model_weights)
