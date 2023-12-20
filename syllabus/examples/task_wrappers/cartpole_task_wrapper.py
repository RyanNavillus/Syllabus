import numpy as np
from syllabus.core import TaskWrapper


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
        return np.array(self.env.state, dtype=np.float32), {}

    def change_task(self, new_task):
        low, high = new_task
        self.env.state = self.env.np_random.uniform(low=low, high=high, size=(4,))
        self.task = new_task

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        # Return percent of optimal reward
        self.total_reward += rew
        return self.total_reward / 500.0
