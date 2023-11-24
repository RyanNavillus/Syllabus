import gym
import numpy as np

from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace


class ProcgenTaskWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.
    """
    def __init__(self, env: gym.Env, env_id: str, seed):
        super().__init__(env)
        self.env_id = env_id
        self.task_space = TaskSpace(gym.spaces.Discrete(200), list(np.arange(0, 200)))
        self.task = seed
        self.seed(seed)

        self.observation_space = self.env.observation_space

    def seed(self, seed):
        self.env.unwrapped._venv.seed(seed, 0)

    def reset(self, new_task=None, **kwargs):
        """
        Resets the environment along with all available tasks, and change the current task.

        This ensures that all instance variables are reset, not just the ones for the current task.
        We do this efficiently by keeping track of which reset functions have already been called,
        since very few tasks override reset. If new_task is provided, we change the task before
        calling the final reset.
        """
        # Change task if new one is provided
        if new_task is not None:
            self.change_task(new_task)

        self.done = False
        self.episode_return = 0

        return self.observation(self.env.reset(**kwargs))

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        seed = int(new_task)
        self.task = seed
        self.seed(seed)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        obs, rew, done, info = self.env.step(action)
        return self.observation(obs), rew, done, info

    def observation(self, obs):
        return obs
