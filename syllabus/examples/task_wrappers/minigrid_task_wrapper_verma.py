import gymnasium as gym
import numpy as np
from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class MinigridTaskWrapperVerma(TaskWrapper):
    def __init__(self, env: gym.Env, env_id, seed=0):
        super().__init__(env)
        self.task_space = TaskSpace(gym.spaces.Discrete(200), list(np.arange(0, 200)))
        self.env_id = env_id
        self.task = seed
        self.episode_return = 0
        
        env_fn = [partial(self._make_minigrid_env, env_name, seeds[i]) for i in range(num_envs)]

        self.observation_space = self.env.observation_space

    @staticmethod
    def _make_minigrid_env(env_name, seed):
        self.seed(seed)
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    def seed(self, seed):
        self.env.gym_env.unwrapped._venv.seed(int(seed), 0)
