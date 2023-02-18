import gym
import ray
import numpy as np
from ray.tune.registry import register_env
from ray import tune
from gym.spaces import Box
from syllabus import TaskWrapper, RaySyncWrapper, RayCurriculumWrapper
from curricula import SimpleBoxCurriculum


class CartPoleTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.task = (-0.02, 0.02)
        self.total_reward = 0

    def reset(self, *args, **kwargs):
        self.env.reset()
        self.total_reward = 0
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.change_task(new_task)
        return np.array(self.env.state, dtype=np.float32)

    def change_task(self, new_task):
        low, high = new_task
        self.env.state = self.env.np_random.uniform(low=low, high=high, size=(4,))
        self.task = new_task

    def _task_completion(self, obs, rew, done, info) -> float:
        # Return percent of optimal reward
        self.total_reward += rew
        return self.total_reward / 500.0


def env_creator(config):
    env = gym.make("CartPole-v1")
    env = CartPoleTaskWrapper(env)
    return RaySyncWrapper(env, default_task=(-0.02, 0.02), task_space=Box(-0.3, 0.3, shape=(2,)), update_on_step=False)


ray.init()
register_env("task_cartpole", env_creator)

curriculum = RayCurriculumWrapper(SimpleBoxCurriculum, task_space=Box(-0.3, 0.3, shape=(2,)))

config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 8,
        "framework": "torch",
}

tuner = tune.Tuner("APEX", param_space=config)
results = tuner.fit()
