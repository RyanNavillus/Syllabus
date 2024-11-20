""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import gymnasium as gym
from syllabus.core import TaskWrapper
from syllabus.task_space import DiscreteTaskSpace


class MinihackTaskWrapper(TaskWrapper):
    """
    This wrapper simply changes the seed of a Minigrid environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

        self.task: str = 1

        # Task completion metrics
        self.episode_return = 0
        self.task_space = DiscreteTaskSpace(1000)

    def reset(self, new_task: int = None, **kwargs):
        # Change task if new one is provided
        # if new_task is not None:
        #     self.change_task(new_task)

        self.episode_return = 0
        self.current_task = new_task
        self.env.seed(new_task)
        return self.observation(self.env.reset(**kwargs))
