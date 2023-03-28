import gym
import ray
import numpy as np
from ray.tune.registry import register_env
from ray import tune
from gym.spaces import Box
from syllabus.core import TaskWrapper, RaySyncWrapper, make_ray_curriculum
from syllabus.curricula import SimpleBoxCurriculum
from syllabus.examples import CartPoleTaskWrapper


def env_creator(config):
    env = gym.make("CartPole-v1")
    env = CartPoleTaskWrapper(env)
    return RaySyncWrapper(env, default_task=(-0.02, 0.02), task_space=Box(-0.3, 0.3, shape=(2,)), update_on_step=False)


ray.init()
register_env("task_cartpole", env_creator)

curriculum = make_ray_curriculum(SimpleBoxCurriculum, task_space=Box(-0.3, 0.3, shape=(2,)))

config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 8,
        "framework": "torch",
}

tuner = tune.Tuner("APEX", param_space=config)
results = tuner.fit()
