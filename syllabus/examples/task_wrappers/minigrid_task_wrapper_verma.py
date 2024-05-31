import gymnasium as gym
import numpy as np
from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from gymnasium.spaces import Box

class MinigridTaskWrapperVerma(TaskWrapper):
    def __init__(self, env: gym.Env, env_id, seed=0):
        super().__init__(env)
        self.env.unwrapped.seed(seed)
        self.task_space = TaskSpace(gym.spaces.Discrete(200), list(np.arange(0, 200)))
        self.env_id = env_id
        self.task = seed
        self.episode_return = 0
        m, n, c = self.env.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], 
            [c, m, n],
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        obs = obs.transpose(2, 0, 1)
        return obs

    def reset(self, new_task=None, **kwargs):
        self.episode_return = 0.0
        if new_task is not None:
            self.change_task(new_task)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        seed = int(new_task)
        self.task = seed
        self.seed(seed)

    def seed(self, seed):
        self.env.unwrapped.seed(int(seed))

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.episode_return += rew
        return self.observation(obs), rew, term, trunc, info
