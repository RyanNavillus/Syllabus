from gymnasium.spaces import Box

from syllabus.core import TaskWrapper
from syllabus.task_space import BoxTaskSpace, DiscreteTaskSpace


class CartPoleTaskWrapper(TaskWrapper):
    def __init__(self, env, discretize=False):
        super().__init__(env)
        self.discretize = discretize
        if self.discretize:
            self.task_space = DiscreteTaskSpace(10)
        else:
            self.task_space = BoxTaskSpace(Box(-0.3, 0.3, shape=(2,)))

        self.task = (-0.02, 0.02)
        self.total_reward = 0

    def reset(self, **kwargs):
        self.total_reward = 0
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            if self.discretize:
                new_task = (3 * new_task / 50.0) - 0.3  # [-0.3, 0.3]
                new_task = (new_task, new_task)
            self.task = new_task
            # task = (3 * new_task / 50.0) - 0.3

        return self.env.reset(options={"low": -abs(self.task[0]), "high": abs(self.task[1])})

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        # Return percent of optimal reward
        self.total_reward += rew
        return self.total_reward / 500.0
