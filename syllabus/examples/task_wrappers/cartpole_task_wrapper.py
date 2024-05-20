from gymnasium.spaces import Box

from syllabus.core import TaskWrapper
from syllabus.task_space import TaskSpace


class CartPoleTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.task_space = TaskSpace(Box(-0.3, 0.3, shape=(2,)))
        self.task = (-0.02, 0.02)
        self.total_reward = 0

    def reset(self, *args, **kwargs):
        self.total_reward = 0
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.task = new_task

        return self.env.reset(options={"low": self.task[0], "high": self.task[1]})

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        # Return percent of optimal reward
        self.total_reward += rew
        return self.total_reward / 500.0
