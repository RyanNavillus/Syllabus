""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import copy
import time
from typing import List
import numpy as np
import gym
from gym import spaces

from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace

class MinihackTaskWrapper(TaskWrapper):
    """
    This wrapper simply changes the seed of a Minigrid environment.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

        self.task: str = 1

        # This is set to False during reset
        self.done = True

        # Task completion metrics
        self.episode_return = 0
        self.task_space = TaskSpace(spaces.Discrete(1000), list(range(1000)))

    def reset(self, new_task: int = None, **kwargs):
        # Change task if new one is provided
        # if new_task is not None:
        #     self.change_task(new_task)

        self.done = False
        self.episode_return = 0
        self.current_task = new_task
        self.env.seed(new_task)
        return self.observation(self.env.reset(**kwargs))
