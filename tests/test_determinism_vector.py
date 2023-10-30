import gym
import procgen  # noqa: F401
import random
import numpy as np
from procgen import ProcgenEnv
from syllabus.tests import evaluate_random_policy
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
import multiprocessing as mp
from syllabus.core import MultiProcessingSyncWrapper


N_EPISODES = 10
import os
seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(10)]
print(seeds)


def make_env(seed=None):
    def thunk():
        env_id = "bigfish"
        env = gym.make(f"procgen-{env_id}-v0", start_level=seed, num_levels=1, distribution_mode="easy")
        env = ProcgenTaskWrapper(env, env_id, seed)
        # Seed environment
        gym.utils.seeding.np_random(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# env = make_env(seed=0)
# env.seed(0)
# obs = env.reset()
# print(np.mean(obs))
# env.close()


envs = ProcgenEnv(num_envs=4, env_name="bigfish", start_level=0, num_levels=1, distribution_mode="easy")
envs.seed(0)
obs = envs.reset()
print(obs["rgb"].shape)
print(obs["rgb"][:, :5, :5, 0] / 255.0)
envs.close()


envs = gym.vector.SyncVectorEnv(
    [
        make_env(0)
        for i in range(4)
    ]
)
envs.is_vector_env = True
envs = gym.wrappers.NormalizeReward(envs, gamma=0.999)
envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
print("preseeded")
envs.seed(0)
print("seeded")
obs = envs.reset()
print(obs.shape)
print(obs[:, 0, :5, :5])
envs.close()
