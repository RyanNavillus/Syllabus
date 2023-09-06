""" Task wrapper that can select a new MiniGrid task on reset. """
import gym
import numpy as np
from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX


class MinigridTaskWrapper(TaskWrapper):
    """
    This wrapper allows you to change the task of an NLE environment.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )
        m, n, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], 
            [c, m, n],
            dtype=self.observation_space.dtype)

        # Set up task space
        self.task_space = TaskSpace(gym.spaces.Discrete(4000), list(np.arange(4000)))
        self.task = None

        # Time limit
        # self._max_episode_steps = env.max_steps
        # self._elapsed_steps = None

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
        # self._elapsed_steps = 0

        return self.observation(self.env.reset(**kwargs)["image"])

    def change_task(self, new_task: int):
        """
        Change task by directly editing environment class.

        Ignores requests for unknown tasks or task changes outside of a reset.
        """
        seed = int(new_task)
        self.task = seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

    def step(self, action):
        """
        Step through environment and update task completion.
        """
        # assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        obs, rew, done, info = self.env.step(action)
        obs = self.observation(obs["image"])

        # # Check time limit
        # self._elapsed_steps += 1
        # if self._elapsed_steps >= self._max_episode_steps:
        #     info['truncated'] = not done
        #     info['truncated_obs'] = obs
        #     done = True

        self.episode_return += rew
        self.done = done
        info["task_completion"] = self._task_completion(obs, rew, done, info)

        return obs, rew, done, info

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        obs = full_grid
        return obs.transpose(2, 0, 1)