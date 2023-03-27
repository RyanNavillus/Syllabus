""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import copy
import time
from typing import List
import numpy as np
import gymnasium as gym
from gym import spaces
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v0

from syllabus.core import PettingZooTaskWrapper


class PistonballTaskWrapper(PettingZooTaskWrapper):
    """
    This wrapper simply changes the seed of a Minigrid environment.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.task: str = 1

        # This is set to False during reset
        self.done = True

        # Task completion metrics
        self.episode_return = 0
        self.task_space = spaces.Discrete(11)   # 0.1 - 1.0 friction

    def reset(self, new_task: int = None, **kwargs):
        # Change task if new one is provided
        # if new_task is not None:
        #     self.change_task(new_task)

        self.done = False
        self.episode_return = 0
        if new_task is not None:
            new_task /= 10
            # Inject current_task into the environment
            # frame_size = (64, 64)
            # env = pistonball_v6.parallel_env(
            #     ball_friction=new_task, continuous=False, max_cycles=125
            # )
            # env = color_reduction_v0(env)
            # env = resize_v1(env, frame_size[0], frame_size[1])
            # env = frame_stack_v1(env, stack_size=4)
            # self.env = env
        return self.observation(self.env.reset(**kwargs))
