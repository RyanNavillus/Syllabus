""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import copy
import time
from typing import List
import numpy as np
import gymnasium as gym
from gym import spaces

from syllabus.core import TaskWrapper
from nle.env import base
from nle.env.tasks import (NetHackScore,
                           NetHackStaircase,
                           NetHackStaircasePet,
                           NetHackOracle,
                           NetHackGold,
                           NetHackEat,
                           NetHackScout)

class SubclassTaskWrapper(TaskWrapper):
    """
    This is a general wrapper for tasks defined as subclasses of a base environment.

    This wrapper reinitializes the environment with the task-specific subclass at the start of each episode.
    This is a simple, general solution to using Syllabus with subclass tasks, but it is likely inefficient.
    It's likely that you can achieve better performance by using a more specialized wrapper.
    """
    def __init__(self, env: gym.Env, task_subclasses: List[gym.Env] = None, **env_init_kwargs):
        super().__init__(env)

        self.task_list = task_subclasses
        self.task_space = spaces.Discrete(len(self.task_list))
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

    def _task_class(self, task):
        return self.task_list[task]

    def _task_name(self, task):
        return self._task_class(task).__name__

    def reset(self, new_task: int = None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.
        """
        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        self.done = False

        return self.observation(self.env.reset(**kwargs))

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
        obs, rew, done, info = self.env.step(action)
        self.done = done
        info["task_completion"] = self._task_completion(obs, rew, done, info)
        return self.observation(obs), rew, done, info


if __name__ == "__main__":
    def run_episode(env, task: str = None, verbose=1):
        env.reset(new_task=task)
        task_name = type(env.unwrapped).__name__
        done = False
        ep_rew = 0
        while not done:
            action = env.action_space.sample()
            _, rew, done, _ = env.step(action)
            ep_rew += rew
        if verbose:
            print(f"Episodic reward for {task_name}: {ep_rew}")

    print("Testing NethackTaskWrapper")
    N_EPISODES = 100

    # Initialize NLE
    nethack_env = NetHackScore()
    nethack_task_env = SubclassTaskWrapper(nethack_env, task_subclasses=[NetHackScore, NetHackStaircase, NetHackStaircasePet, NetHackOracle, NetHackGold, NetHackEat, NetHackScout])

    start_time = time.time()

    for _ in range(N_EPISODES):
        run_episode(nethack_task_env, verbose=0)

    end_time = time.time()
    print(f"Run time same task: {end_time - start_time}")
    start_time = time.time()

    for _ in range(N_EPISODES):
        nethack_task = nethack_task_env.task_space.sample()
        run_episode(nethack_task_env, task=nethack_task, verbose=0)

    end_time = time.time()
    print(f"Run time swapping tasks: {end_time - start_time}")
