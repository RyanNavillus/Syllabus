import gym
import numpy as np
from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace

CRAFTER_RETURN_BOUNDS = {
    "CrafterReward-v1": (0, 100),
    "CrafterNoReward-v1": (0, 0),
}

class CrafterTaskWrapper(TaskWrapper):
    def __init__(self, env, seed=None):
        super().__init__(env, TaskSpace(200))
        self.seed(seed)
        self.episode_return = 0

    def seed(self, seed=None):
        self.env.seed(seed)
        return [seed]

    def reset(self, task=None):
        self.episode_return = 0
        if task is not None:
            self.change_task(task)
        return self.observation(self.env.reset())

    def change_task(self, task):
        if task not in CRAFTER_RETURN_BOUNDS:
            raise ValueError(f"Unknown task {task}")
        self.env.task = task


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_return += reward
        min_return, max_return = CRAFTER_RETURN_BOUNDS[self.env.task]
        info["task_completion"] = (self.episode_return - min_return) / (max_return - min_return) if max_return > min_return else 0.0
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return observation