""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import copy
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from syllabus.task_space import DiscreteTaskSpace

from .task_wrapper import TaskWrapper


class SubclassTaskWrapper(TaskWrapper):
    # TODO: Automated tests
    """
    This is a general wrapper for tasks defined as subclasses of a base environment.

    This wrapper reinitializes the environment with the provided env function at the start of each episode.
    This is a simple, general solution to using Syllabus with tasks that need to be reinitialized, but it is inefficient.
    It's likely that you can achieve better performance by using a more specialized wrapper.
    """

    def __init__(self, env: gym.Env, task_subclasses: List[gym.Env] = None, **env_init_kwargs):
        super().__init__(env)

        self.task_list = task_subclasses
        self.task_space = DiscreteTaskSpace(len(self.task_list), self.task_list)
        self._env_init_kwargs = env_init_kwargs  # kwargs for reinitializing the base environment

        # Add goal space to observation
        self.observation_space = copy.deepcopy(self.env.observation_space)
        self.observation_space["goal"] = spaces.MultiBinary(len(self.task_list))

        # Tracking episode end
        self.done = True

        # Initialize all tasks
        original_class = self.env.__class__
        for task in self.task_list:
            self.env.__class__ = task
            self.env.__init__(**self._env_init_kwargs)

        self.env.__class__ = original_class
        self.env.__init__(**self._env_init_kwargs)

    @property
    def current_task(self):
        return self.env.__class__

    def _task_name(self, task):
        return self.task.__name__

    def reset(self, new_task: int = None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.
        """
        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        self.done = False
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        # Ignore new task if mid episode
        if self.current_task.__init__ != self._task_class(new_task).__init__ and not self.done:
            raise RuntimeError("Cannot change task mid-episode.")

        # Ignore if task is unknown
        if new_task >= len(self.task_list):
            raise RuntimeError(f"Unknown task {new_task}.")

        # Update current task
        prev_task = self.task
        self.task = new_task
        self.env.__class__ = self._task_class(new_task)

        # If task requires reinitialization
        if type(self.env).__init__ != prev_task.__init__:
            self.env.__init__(**self._env_init_kwargs)

    def _encode_goal(self):
        goal_encoding = np.zeros(len(self.task_list))
        goal_encoding[self.task] = 1
        return goal_encoding

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, term, trunc, info = self.env.step(action)
        self.done = term or trunc
        info["task_completion"] = self._task_completion(obs, rew, term, trunc, info)
        return self.observation(obs), rew, term, trunc, info
