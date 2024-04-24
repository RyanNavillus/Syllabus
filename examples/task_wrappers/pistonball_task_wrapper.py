""" Task wrapper for NLE that can change tasks at reset using the NLE's task definition format. """
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.butterfly import pistonball_v6
from syllabus.core import PettingZooTaskWrapper
from syllabus.task_space import TaskSpace


class PistonballTaskWrapper(PettingZooTaskWrapper):
    """
    This wrapper simply changes the seed of a Minigrid environment.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.task: str = 1

        # Task completion metrics
        self.episode_return = 0
        self.task_space = TaskSpace(spaces.Discrete(11), list(range(11)))   # 0.1 - 1.0 friction

    def reset(self, new_task: int = None, **kwargs):
        # Change task if new one is provided
        # if new_task is not None:
        #     self.change_task(new_task)

        self.episode_return = 0
        if new_task is not None:
            task = new_task / 10
            # Inject current_task into the environment
            self.env = pistonball_v6.parallel_env(
                ball_friction=task, continuous=False, max_cycles=125
            )
            self.env.unwrapped.task = new_task
        return self.observation(self.env.reset(**kwargs))
